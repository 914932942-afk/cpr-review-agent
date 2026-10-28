"""Module contracts and actual controller injection. Vision uses explicit test doubles here."""

import json
import runpy
import sys

import pytest
from cpr_agent.api import create_app
from cpr_agent.branch_vision import BranchVisionTool, validate_pipeline
from cpr_agent.config import ROOT
from cpr_agent.evidence import file_hash
from cpr_agent.modules.baselines import BaselineCalibration, BaselineScheduling
from cpr_agent.modules.contracts import (
    LABEL_IDS,
    Assessment,
    Binding,
    CalibrationInput,
    Decision,
    FusionInput,
    Prediction,
    QualityIndicators,
    Readout,
)
from cpr_agent.modules.registry import DEFAULT_SELECTIONS, ModuleSpec, build_registry
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_branches import configured, fake_stage
from test_real_video import submit, uploaded_session, wait

PROFILE = ROOT / "data/module-slots-v1/pipeline-manifest.json"


def bound():
    return Binding(
        clip_id="clip",
        video_sha256="a" * 64,
        bundle_sha256="b" * 64,
        pipeline_sha256="c" * 64,
        actor_source_sha256="d" * 64,
    )


def prediction(**extra):
    value = dict(
        prediction_id="prediction",
        binding=bound(),
        supported_label_ids=[2],
        flagged_label_ids=[2],
        score_type="uncalibrated_score",
        scores={2: 0.9},
        thresholds={2: 0.5},
    )
    return Prediction(**{**value, **extra})


@pytest.mark.parametrize(
    "extra",
    [
        {"scores": {}},
        {"scores": {2: float("nan")}},
        {"flagged_label_ids": [4]},
        {"supported_label_ids": [12], "scores": {12: 0.9}},
        {"scores": {2: 1.1}},
    ],
)
def test_invalid_prediction_never_becomes_negative_or_probability(extra):
    with pytest.raises(ValidationError):
        prediction(**extra)


def test_fusion_cannot_mix_actor_or_clip():
    first = prediction()
    second = first.model_copy(update={"binding": bound().model_copy(update={"clip_id": "other"})})
    with pytest.raises(ValueError, match="Cross-video"):
        FusionInput(bound(), "execution", Readout(first, {}), Readout(second, {}), QualityIndicators())


def test_uncalibrated_baseline_and_missing_class_semantics():
    request = CalibrationInput(
        target_label_ids=[2, 4], predictions=[prediction()], quality=QualityIndicators()
    )
    result = BaselineCalibration().assess(request)
    assert result.states == {2: "model_flagged", 4: "not_evaluated"}
    assert result.calibrated_probabilities == {} and result.calibration_status == "uncalibrated"
    with pytest.raises(ValidationError):
        Assessment(**{**result.model_dump(), "calibrated_probabilities": {2: 0.9}})


def test_registry_unknown_duplicate_and_contract_version_fail():
    registry = build_registry()
    with pytest.raises(ValueError, match="NOT_REGISTERED"):
        registry.validate_selection({**DEFAULT_SELECTIONS, "M5": "member-not-delivered"})
    with pytest.raises(ValueError, match="REGISTRATION_INVALID"):
        registry.register(ModuleSpec("M2", "baseline-v15", object))
    with pytest.raises(ValueError, match="CONTRACT_VERSION"):
        registry.register(ModuleSpec("M3", "bad", object, contract_version="unknown"))
    registry.register(ModuleSpec("M3", "missing-method", object))
    with pytest.raises(ValueError, match="METHOD_MISSING"):
        registry.create("M3", "missing-method")


def slot_fake(self, request):
    output = fake_stage(self, request)
    if request.tool_id not in {"rgb.predict", "fusion.predict"}:
        return output
    path = self.settings.data_dir / output.artifact_rel
    raw = json.loads(path.read_text())
    if request.tool_id == "rgb.predict":
        pose_sha = raw["shared_pose_sha256"]
        executions = {}
        module_id = "M5"
    else:
        source = self.store.get_tool(request.input_bindings["rgb_receipt_id"])["output"]
        rgb = json.loads((self.settings.data_dir / source["artifact_rel"]).read_text())
        pose_sha = rgb["shared_pose_sha256"]
        executions = rgb["module_executions"]
        module_id = "M2"
    executions[module_id] = dict(
        module_id=module_id,
        implementation_id=DEFAULT_SELECTIONS[module_id],
        kind="baseline",
        contract_version="cpr-modules-v1",
        prediction_id=request.execution_id,
        binding=dict(
            clip_id=request.clip_id,
            video_sha256=raw["video_sha256"],
            bundle_sha256=raw["bundle_sha256"],
            pipeline_sha256=raw["pipeline_sha256"],
            actor_source_sha256=pose_sha,
        ),
    )
    raw.update(module_contract_version="cpr-modules-v1", module_executions=executions)
    path.write_text(json.dumps(raw))
    return output.model_copy(update={"artifact_sha256": file_hash(path)})


def test_baseline_slots_execute_and_persist_without_claiming_new_algorithms(tmp_path, monkeypatch):
    monkeypatch.setattr(BranchVisionTool, "execute", slot_fake)
    settings = configured(tmp_path, branch_manifest=PROFILE)
    with TestClient(create_app(settings)) as client:
        caps = client.get("/api/capabilities").json()
        assert len(caps["module_slots"]) == 4
        assert all(s["algorithm_status"] == "baseline_only" for s in caps["module_slots"].values())
        sid, _, _ = uploaded_session(client)
        run = wait(client, submit(client, sid))
        assert run["status"] == "completed", run
        assert run["response"]["module_assessment"]["calibration_status"] == "uncalibrated"
        assert run["response"]["target_states"] == {"2": "model_flagged", "4": "not_flagged"}
    with TestClient(create_app(settings)) as client:
        following = wait(client, submit(client, sid, intent="summarise", analysis_mode="existing_only"))
        assert following["status"] == "completed" and following["response"]["reused_result"]
        assert client.get(f"/api/runs/{following['id']}/tools").json() == []


@pytest.mark.parametrize("bad", [False, True])
def test_registered_m4_is_called_and_unavailable_actions_are_blocked(tmp_path, monkeypatch, bad):
    calls = []

    class TestPolicy:
        def choose(self, observation):
            calls.append(observation)
            return (
                Decision(action="run_rgb", profile_id="full-v15", reason="test-only illegal action")
                if bad
                else BaselineScheduling().choose(observation)
            )

    registry = build_registry()
    registry.register(ModuleSpec("M4", "test-policy", TestPolicy, kind="improved"))
    monkeypatch.setattr("cpr_agent.modules.registry.build_registry", lambda: registry)
    monkeypatch.setattr("cpr_agent.module_runtime.build_registry", lambda: registry)
    config = json.loads(PROFILE.read_text())
    config["module_slots"]["M4"] = "test-policy"
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(BranchVisionTool, "execute", slot_fake)
    with TestClient(create_app(configured(tmp_path, branch_manifest=path))) as client:
        sid, _, _ = uploaded_session(client)
        run = wait(client, submit(client, sid))
        assert calls and calls[0].available_actions == ["run_full_pipeline", "STOP"]
        assert run["status"] == ("failed" if bad else "completed"), run
        if bad:
            assert run["response"]["classification_claims"] == []
            assert client.get(f"/api/runs/{run['id']}/tools").json() == []


def test_selected_module_source_requires_pin(tmp_path):
    config = json.loads(PROFILE.read_text())
    config["sources"].pop("backend/cpr_agent/modules/members.py")
    path = tmp_path / "unpinned.json"
    path.write_text(json.dumps(config))
    with pytest.raises(Exception, match="MODULE_SOURCE_NOT_PINNED"):
        validate_pipeline(configured(tmp_path, branch_manifest=path))


def test_m3_replacement_preserves_uncertainty_separately_from_raw_cv(tmp_path, monkeypatch):
    class TestCalibration:
        def assess(self, request):
            original = BaselineCalibration().assess(request)
            return Assessment(
                states={
                    label: ("uncertain" if request.predictions else "not_evaluated")
                    for label in request.target_label_ids
                },
                source_prediction_ids=original.source_prediction_ids,
                calibration_profile="test-calibration",
                calibration_status="calibrated",
                calibrated_probabilities={label: 0.5 for label in request.target_label_ids}
                if request.predictions
                else {},
            )

    registry = build_registry()
    registry.register(ModuleSpec("M3", "test-calibration", TestCalibration, kind="improved"))
    monkeypatch.setattr("cpr_agent.modules.registry.build_registry", lambda: registry)
    monkeypatch.setattr("cpr_agent.module_runtime.build_registry", lambda: registry)
    config = json.loads(PROFILE.read_text())
    config["module_slots"]["M3"] = "test-calibration"
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(BranchVisionTool, "execute", slot_fake)
    with TestClient(create_app(configured(tmp_path, branch_manifest=path))) as client:
        sid, _, _ = uploaded_session(client)
        run = wait(client, submit(client, sid))
        assert run["status"] == "completed", run
        assert run["response"]["target_states"] == {"2": "uncertain", "4": "uncertain"}
        assert run["response"]["module_assessment"]["calibrated_probabilities"] == {"2": 0.5, "4": 0.5}
        assert run["response"]["classification_claims"][0]["state"] == "model_flagged"


def test_unassessed_probability_is_rejected():
    with pytest.raises(ValidationError, match="Unassessed"):
        Assessment(
            states={4: "not_evaluated"},
            source_prediction_ids=[],
            calibration_profile="invalid",
            calibration_status="calibrated",
            calibrated_probabilities={4: 0.9},
        )


def test_profile_generation_is_deterministic(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    monkeypatch.setattr(
        sys, "argv", ["build_module_profile.py", "--base-profile", str(PROFILE), "--output", str(output)]
    )
    runpy.run_path(str(ROOT / "scripts/build_module_profile.py"), run_name="__main__")
    assert output.read_bytes() == PROFILE.read_bytes()


def test_switching_one_module_keeps_other_member_selection(tmp_path, monkeypatch):
    from cpr_agent.modules.baselines import BaselineFusion

    registry = build_registry()
    registry.register(ModuleSpec("M2", "test-existing-member", BaselineFusion, kind="improved"))
    monkeypatch.setattr("cpr_agent.modules.registry.build_registry", lambda: registry)
    inherited = json.loads(PROFILE.read_text())
    inherited["module_slots"]["M2"] = "test-existing-member"
    base = tmp_path / "base.json"
    base.write_text(json.dumps(inherited))
    output = tmp_path / "output.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_module_profile.py",
            "--base-profile",
            str(base),
            "--m5",
            "baseline-rgb-v15",
            "--output",
            str(output),
        ],
    )
    runpy.run_path(str(ROOT / "scripts/build_module_profile.py"), run_name="__main__")
    assert json.loads(output.read_text())["module_slots"]["M2"] == "test-existing-member"


def test_explicit_missing_base_is_not_silently_replaced(tmp_path, monkeypatch):
    output = tmp_path / "output.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_module_profile.py",
            "--base-profile",
            str(tmp_path / "missing.json"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(FileNotFoundError):
        runpy.run_path(str(ROOT / "scripts/build_module_profile.py"), run_name="__main__")
    assert not output.exists()


def test_contract_labels_match_existing_model_scope():
    from cpr_agent.models import CANDIDATE_LABELS

    assert LABEL_IDS == CANDIDATE_LABELS
