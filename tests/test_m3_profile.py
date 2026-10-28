"""Current M3 profile and quality-provenance checks; explicit controller doubles only."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cpr_agent.api import create_app
from cpr_agent.branch_vision import validate_pipeline
from cpr_agent.config import ROOT, Settings
from cpr_agent.evidence import file_hash
from cpr_agent.vision import VisionError
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(ROOT.name != "m3-calibrated-v1", reason="Isolated M3 backend required")
PROFILE = ROOT / "data/m3-calibration-v1/pipeline-manifest.json"


def settings(tmp_path, profile=PROFILE):
    return Settings(
        data_dir=tmp_path,
        vision_mode="real",
        ml_python=Path(sys.executable),
        vision_pipeline="branches",
        branch_manifest=profile,
        context_mode="bound-v1",
    )


def test_current_profile_pins_the_actual_m3_and_preserves_upstream(tmp_path):
    assert validate_pipeline(settings(tmp_path)) == file_hash(PROFILE)
    for name, expected in json.loads((ROOT / "data/m3-calibration-v1/upstream.json").read_text())[
        "sources"
    ].items():
        assert file_hash(ROOT / name) == expected


def test_capabilities_advertise_m3_without_claiming_m4_or_m5(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as c:
        caps = c.get("/api/capabilities").json()
        assert caps["module_slots"]["M3"]["kind"] == "improved"
        assert caps["module_slots"]["M3"]["implementation_id"] == "source-quality-selective-v1"
        assert all(caps["module_slots"][k]["kind"] == "baseline" for k in ["M4", "M5"])
        assert caps["llm"]["mode"] == "disabled" and len(caps["nodes"]) == 12


@pytest.mark.parametrize(
    "name",
    [
        "experiments/m3_calibration_v1/algorithm.py",
        "experiments/m3_calibration_v1/module.py",
        "backend/cpr_agent/m3_quality.py",
        "data/m3-calibration-v1/selection.json",
        "frontend/src/EvidencePanel.tsx",
    ],
)
def test_tampered_algorithm_parameters_or_ui_blocks_profile(tmp_path, name):
    value = json.loads(PROFILE.read_text())
    value["sources"][name] = "0" * 64
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))
    with pytest.raises(VisionError, match="PIPELINE_SOURCE_CHANGED"):
        validate_pipeline(settings(tmp_path, path))


def test_previous_profile_cannot_be_relabelled_as_m3(tmp_path):
    previous = ROOT / "data/m1-m2-refined-v1/pipeline-manifest.json"
    with pytest.raises(VisionError, match="PIPELINE_SOURCE_CHANGED"):
        validate_pipeline(settings(tmp_path, previous))


def fixture_quality(tmp_path, monkeypatch):
    from cpr_agent import m3_quality

    output = SimpleNamespace(clip_id="one", video_sha256="v", bundle_sha256="b", profile_id="p")
    receipts = {
        name: SimpleNamespace(**vars(output), artifact_sha256=name + "hash", tool_id=tool)
        for name, tool in [("sk", "skeleton.predict"), ("rgb", "rgb.predict")]
    }
    paths = {}
    for name in receipts:
        value = {
            "shared_pose_sha256": "same-actor",
            "decoded_frames": 20,
            "predicted_original_label_ids": [1] if name == "sk" else [],
            "supported_label_ids": [1, 2, 5, 7, 9],
        }
        p = tmp_path / (name + ".json")
        p.write_text(json.dumps(value))
        paths[name + "hash"] = p
    records = {
        name: {"status": "succeeded", "request": {"tool_id": r.tool_id}, "output": name}
        for name, r in receipts.items()
    }
    monkeypatch.setattr(m3_quality, "validate_branch", lambda key, *args: receipts[key])
    monkeypatch.setattr(m3_quality, "artifact_path", lambda receipt, *args: paths[receipt.artifact_sha256])
    raw = {
        "dependencies": {"skeleton.predict": "skhash", "rgb.predict": "rgbhash"},
        "decoded_frames": 20,
        "fullpose_audit": {"m1_stability": {"original_observed_points": 300, "frames": 20}},
        "rgb_audit": {"frames": 12, "requested_frames": 16},
    }
    execution = {"request": {"input_bindings": {"skeleton_receipt_id": "sk", "rgb_receipt_id": "rgb"}}}
    return (
        m3_quality,
        SimpleNamespace(get_tool=lambda key: records[key]),
        SimpleNamespace(data_dir=tmp_path),
        execution,
        output,
        raw,
        receipts,
    )


def test_quality_comes_from_same_video_actor_and_executed_dependencies(tmp_path, monkeypatch):
    module, store, cfg, execution, output, raw, _ = fixture_quality(tmp_path, monkeypatch)
    q = module.bound_quality(store, cfg, execution, output, raw)
    assert q.pose_visible_fraction == 300 / 340 and q.rgb_usable_samples == 12
    assert q.extras["disagreement_1"] == 1 and q.extras["disagreement_2"] == 0


@pytest.mark.parametrize("failure", ["wrong_video", "wrong_dependency", "missing_binding"])
def test_cross_source_or_unbound_quality_is_rejected(tmp_path, monkeypatch, failure):
    module, store, cfg, execution, output, raw, receipts = fixture_quality(tmp_path, monkeypatch)
    if failure == "wrong_video":
        receipts["rgb"].clip_id = "other"
    elif failure == "wrong_dependency":
        raw["dependencies"]["rgb.predict"] = "other-hash"
    else:
        execution["request"]["input_bindings"].pop("rgb_receipt_id")
    with pytest.raises(ValueError):
        module.bound_quality(store, cfg, execution, output, raw)


def test_missing_pose_quality_stays_unknown(tmp_path, monkeypatch):
    module, store, cfg, execution, output, raw, _ = fixture_quality(tmp_path, monkeypatch)
    raw["fullpose_audit"] = {}
    assert module.bound_quality(store, cfg, execution, output, raw).pose_visible_fraction is None
