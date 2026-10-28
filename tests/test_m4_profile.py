"""M4 current route contracts, lineage approvals and source integrity, without live ML."""

import copy
import json
import sys
from pathlib import Path

import pytest
from cpr_agent.api import create_app
from cpr_agent.branch_vision import validate_pipeline
from cpr_agent.config import ROOT, Settings
from cpr_agent.evidence import file_hash
from cpr_agent.models import ClipPrediction, ReviewGoal
from cpr_agent.vision import VisionError, build_response
from fastapi.testclient import TestClient
from pydantic import ValidationError

pytestmark = pytest.mark.skipif(ROOT.name != "m4-routing-v1", reason="Isolated M4 workspace required")
PROFILE = ROOT / "data/m4-scheduling-v1/pipeline-manifest.json"


def settings(tmp_path, profile=PROFILE):
    return Settings(
        data_dir=tmp_path,
        vision_mode="real",
        ml_python=Path(sys.executable),
        vision_pipeline="branches",
        branch_manifest=profile,
        context_mode="bound-v1",
        graph_version="cpr-review-12-v0.7-routing",
    )


def prediction():
    return ClipPrediction(
        execution_id="one",
        clip_id="clip",
        asset_id="asset",
        video_sha256="a" * 64,
        bundle_sha256="b" * 64,
        profile_id="single-view-v15-branches-v1:" + "c" * 64,
        artifact_rel="analysis/one/result.json",
        artifact_sha256="d" * 64,
        decoded_frames=100,
        reported_fps=25,
        predicted_label_ids=[4],
        supported_label_ids=[3, 4, 8, 11],
        scores=[
            {"original_label_id": k, "uncalibrated_score": 0.9 if k == 4 else 0.1, "threshold": 0.5}
            for k in [3, 4, 8, 11]
        ],
        analysis_seconds=1,
        analysis_route="body-qualified-v1",
    )


def test_current_sources_validate_and_upstream_numerics_are_preserved(tmp_path):
    assert validate_pipeline(settings(tmp_path)) == file_hash(PROFILE)
    for name, expected in json.loads((ROOT / "data/m3-calibration-v1/upstream.json").read_text())[
        "sources"
    ].items():
        assert file_hash(ROOT / name) == expected


def test_m4_is_improved_and_m5_remains_a_baseline(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as c:
        caps = c.get("/api/capabilities").json()
        assert caps["module_slots"]["M4"]["kind"] == "improved"
        assert caps["module_slots"]["M5"]["kind"] == "baseline"
        assert len(caps["nodes"]) == 12
        assert caps["llm"]["mode"] == "disabled"


@pytest.mark.parametrize(
    "name",
    [
        "experiments/m4_policy_v1/algorithm.py",
        "experiments/m4_policy_v1/stages.py",
        "backend/cpr_agent/m4_lineage.py",
        "data/m4-scheduling-v1/parameters.json",
        "frontend/src/App.tsx",
    ],
)
def test_changed_route_code_or_parameters_block_startup(tmp_path, name):
    value = json.loads(PROFILE.read_text())
    value["sources"][name] = "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(VisionError, match="PIPELINE_SOURCE_CHANGED"):
        validate_pipeline(settings(tmp_path, path))


def test_partial_scope_is_explicit_and_cannot_masquerade_as_full():
    p = prediction()
    assert p.supported_label_ids == [3, 4, 8, 11]
    with pytest.raises(ValidationError):
        ClipPrediction(**{**p.model_dump(), "analysis_route": "full-v15"})
    with pytest.raises(ValidationError):
        ClipPrediction(**{**p.model_dump(), "supported_label_ids": [3, 4, 6, 8, 11]})


def test_unanalysed_categories_never_receive_negative_claims():
    r = build_response(ReviewGoal(intent="analyse", target_label_ids=[1, 4]), prediction())
    assert [c.label_id for c in r.classification_claims] == [4]
    assert r.target_states["1"] == "not_evaluated"
    r = build_response(ReviewGoal(intent="explain", target_label_ids=[1]), prediction())
    assert r.classification_claims == [] and r.target_states["1"] == "not_evaluated"


def records():
    return {
        "f": {
            "id": "f",
            "run_id": "old",
            "status": "succeeded",
            "request": {
                "tool_id": "fusion.predict",
                "clip_id": "clip",
                "input_bindings": {"skeleton_receipt_id": "s", "route_profile": "body-qualified-v1"},
            },
        },
        "s": {
            "id": "s",
            "run_id": "old",
            "status": "succeeded",
            "request": {
                "tool_id": "skeleton.predict",
                "clip_id": "clip",
                "input_bindings": {"pose_receipt_id": "p"},
            },
        },
        "p": {
            "id": "p",
            "run_id": "old",
            "status": "succeeded",
            "request": {"tool_id": "pose.extract", "clip_id": "clip", "input_bindings": {}},
        },
    }


def test_cache_closure_contains_only_the_actual_approved_dag():
    from types import SimpleNamespace

    from cpr_agent.m4_lineage import closure

    a = records()
    assert closure(SimpleNamespace(get_tool=lambda key: a[key]), "f") == {
        "fusion.predict": "f",
        "skeleton.predict": "s",
        "pose.extract": "p",
    }


@pytest.mark.parametrize("mode", ["wrong_kind", "undeclared", "not_successful"])
def test_invalid_dependency_graph_is_rejected(mode):
    from types import SimpleNamespace

    from cpr_agent.m4_lineage import closure

    a = records()
    if mode == "wrong_kind":
        a["p"]["request"]["tool_id"] = "rgb.predict"
    elif mode == "undeclared":
        a["p"]["request"]["input_bindings"]["pose_receipt_id"] = "p"
    else:
        a["s"]["status"] = "failed"
    with pytest.raises(ValueError):
        closure(SimpleNamespace(get_tool=lambda key: a[key]), "f")


def test_reuse_needs_the_server_bound_source_and_its_dependency_membership(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cpr_agent import m4_lineage, vision

    a = records()
    a["f"]["output"] = {}
    snapshot = {"source": {"execution_id": "f", "video_sha256": "v"}}
    store = SimpleNamespace(
        get_tool=lambda key: a[key],
        get_run=lambda rid: {"turn_id": "turn", "clip_id": "clip"},
        get=lambda *args: {"context_snapshot": snapshot},
    )
    request = {"run_id": "new", "input_bindings": {"reuse_source_execution_id": "f", "video_sha256": "v"}}
    monkeypatch.setattr(vision, "validate_receipt", lambda *args: SimpleNamespace(clip_id="clip"))
    assert m4_lineage.reusable(store, settings(tmp_path), request, a["s"])
    other = copy.deepcopy(a["s"])
    other["id"] = "unapproved"
    assert not m4_lineage.reusable(store, settings(tmp_path), request, other)
    snapshot["source"]["execution_id"] = "different"
    assert not m4_lineage.reusable(store, settings(tmp_path), request, a["s"])


def test_scope_upgrade_missing_work_does_not_repeat_shared_stages():
    from cpr_agent.m4_controller import missing_tools

    assert missing_tools(
        "full-v15",
        {"pose.extract": "p", "skeleton.predict": "s", "fusion.predict": "f"},
        [3, 4, 8, 11],
        [1, 4],
    ) == ["rgb.predict", "fusion.predict"]
    with pytest.raises(ValueError):
        missing_tools("body-qualified-v1", {}, [], [6])


def test_expansion_is_not_misreported_as_zero_new_analysis():
    from types import SimpleNamespace

    from cpr_agent.graph import ReviewGraph
    from cpr_agent.modules.contracts import Decision

    graph = ReviewGraph.__new__(ReviewGraph)
    graph.settings = SimpleNamespace(vision_pipeline="branches", vision_mode="real")
    graph.store = SimpleNamespace(event=lambda *a, **kw: None, get_run=lambda rid: {"id": rid})
    graph.policy = SimpleNamespace(
        implementation="goal-cost-dag-v1",
        decide=lambda *args: Decision(
            action="run_full_pipeline", profile_id="full-v15", reason="Expand requested scope."
        ),
    )
    state = {
        "run_id": "new",
        "goal": ReviewGoal(intent="analyse", target_label_ids=[1, 4]).model_dump(mode="json"),
        "receipts": {"pose.extract": "p", "skeleton.predict": "s", "fusion.predict": "f"},
        "failed_tools": [],
        "module_assessment": {"states": {"4": "model_flagged", "1": "not_evaluated"}},
        "context": {"reused_result": True},
    }
    command = graph.n06(state)
    assert [step["tool_id"] for step in command.update["plan"]] == ["rgb.predict", "fusion.predict"]
    assert command.update["context"]["reused_result"] is False
