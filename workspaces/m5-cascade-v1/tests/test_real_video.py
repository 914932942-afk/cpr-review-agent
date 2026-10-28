"""Fast integration checks of orchestration and result boundaries, not model accuracy tests."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from cpr_agent.api import create_app
from cpr_agent.config import ROOT, Settings
from cpr_agent.models import CANDIDATE_LABELS, NODE_NAMES, TERMINAL, ClipPrediction
from cpr_agent.vision import RealVisionTool, VisionError, file_hash, media_context
from fastapi.testclient import TestClient
from pydantic import ValidationError


def wait(client, rid, predicate=None):
    until = time.monotonic() + 8
    while time.monotonic() < until:
        run = client.get(f"/api/runs/{rid}").json()
        if predicate(run) if predicate else run["status"] in TERMINAL:
            return run
        time.sleep(0.02)
    pytest.fail(str(run))


def submit(client, sid, intent="analyse", targets=None, **extra):
    response = client.post(
        f"/api/sessions/{sid}/turns",
        headers={"Idempotency-Key": str(uuid4())},
        json={"goal": {"intent": intent, "target_label_ids": targets or [2, 4], **extra}},
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def uploaded_session(client):
    video = (ROOT / "fixtures/workflow-fixture.mp4").read_bytes()
    asset = client.post("/api/assets", files={"file": ("input.mp4", video, "video/mp4")}).json()
    clip = client.post("/api/clips", json={"asset_id": asset["id"]}).json()
    session = client.post("/api/sessions", json={"clip_id": clip["id"]}).json()
    return session["id"], asset, clip


def make_prediction(self, request):
    _, asset, _ = media_context(self.store, self.settings, request.clip_id)
    folder = self.settings.data_dir / "analysis" / request.execution_id
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "fake-contract-output.json"
    scores = [
        {"original_label_id": i, "uncalibrated_score": 0.9 if i == 2 else 0.1, "threshold": 0.5}
        for i in CANDIDATE_LABELS
    ]
    artifact.write_text(
        json.dumps(
            {
                "schema": "cpr-offline-analysis-v2",
                "status": "completed_experimental",
                "video_sha256": asset["sha256"],
                "bundle_sha256": file_hash(self.settings.vision_manifest),
                "decoded_frames": 50,
                "reported_fps": 25,
                "predicted_original_label_ids": [2],
                "supported_label_ids": list(CANDIDATE_LABELS),
                "scores": scores,
                "public_label_verification": None,
                "model_or_threshold_modified": False,
                "rgb_audit": {"frames": 12, "requested_frames": 16},
                "hand_shape_audit": {"usable_frames": 8, "requested_frames": 16, "available": True},
                "finished_utc": "2026-10-21T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return ClipPrediction(
        execution_id=request.execution_id,
        clip_id=request.clip_id,
        asset_id=asset["id"],
        video_sha256=asset["sha256"],
        bundle_sha256=file_hash(self.settings.vision_manifest),
        artifact_rel=artifact.relative_to(self.settings.data_dir).as_posix(),
        artifact_sha256=file_hash(artifact),
        decoded_frames=50,
        reported_fps=25,
        predicted_label_ids=[2],
        supported_label_ids=list(CANDIDATE_LABELS),
        scores=scores,
        analysis_seconds=0.01,
    )


@pytest.fixture
def real_client(tmp_path, monkeypatch):
    # Explicit test-only double. A separate actual HTTP acceptance uses the real ML subprocess.
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)
    settings = Settings(data_dir=tmp_path, vision_mode="real", ml_python=Path(__import__("sys").executable))
    with TestClient(create_app(settings)) as client:
        yield client


def test_full_twelve_nodes_and_followup_reuse(real_client):
    client = real_client
    sid, asset, clip = uploaded_session(client)
    start = client.post(
        f"/api/sessions/{sid}/turns", json={"message": "Review"}, headers={"Idempotency-Key": str(uuid4())}
    ).json()
    rid = start["run_id"]
    paused = wait(client, rid, lambda r: r["status"] == "waiting_user" and r["wait"].get("interrupt_id"))
    assert (
        client.post(
            f"/api/runs/{rid}/resume",
            headers={"Idempotency-Key": str(uuid4())},
            json={
                "kind": "user_answer",
                "revision": paused["revision"],
                "wait_id": paused["wait"]["wait_id"],
                "interrupt_id": paused["wait"]["interrupt_id"],
                "goal": {"intent": "analyse", "target_label_ids": [2, 4]},
            },
        ).status_code
        == 202
    )
    run = wait(client, rid)
    assert run["status"] == "completed" and run["error_code"] is None
    assert run["mode"] == "experimental" and run["response"]["analysis_status"] == "completed_experimental"
    assert run["response"]["target_states"] == {"2": "model_flagged", "4": "not_flagged"}
    events = client.get(f"/api/runs/{rid}/events").json()["events"]
    assert {e["node_id"] for e in events if e["kind"] == "node.enter"} == set(NODE_NAMES)
    tools = client.get(f"/api/runs/{rid}/tools").json()
    assert len(tools) == 1 and tools[0]["request"]["tool_id"] == "video.analyze"
    result = run["response"]["prediction"]
    assert result["video_sha256"] == asset["sha256"] and result["clip_id"] == clip["id"]
    assert result["evidence_scope"] == "whole_clip" and not tools[0]["output"]["is_stub"]
    report = client.get(f"/api/reports/{run['report_id']}").json()
    assert report["is_scaffold"] is False
    assert "Scaffold only" not in client.get(f"/api/reports/{run['report_id']}/export").text
    for intent in ("explain", "summarise", "verify", "analyse"):
        next_id = submit(client, sid, intent=intent, analysis_mode="existing_only")
        followup = wait(client, next_id)
        assert followup["status"] == "completed" and followup["response"]["reused_result"]
        assert followup["response"]["prediction"]["execution_id"] == result["execution_id"]
        assert client.get(f"/api/runs/{next_id}/tools").json() == []


def test_reject_fixture_and_partial_clips(real_client):
    client = real_client
    demo = client.post("/api/demo/session").json()
    run = wait(client, submit(client, demo["session"]["id"]))
    assert run["status"] == "rejected" and run["error_code"] == "SYNTHETIC_FIXTURE_NOT_CPR"
    assert client.get(f"/api/runs/{run['id']}/tools").json() == []
    _, asset, _ = uploaded_session(client)
    response = client.post("/api/clips", json={"asset_id": asset["id"], "start_seconds": 1})
    assert response.status_code == 422 and response.json()["error"]["error_code"] == "WHOLE_CLIP_ONLY"


def test_existing_only_without_analysis_and_new_clip_isolation(real_client):
    client = real_client
    sid, _, _ = uploaded_session(client)
    first = wait(client, submit(client, sid, analysis_mode="existing_only"))
    assert first["error_code"] == "NO_EXISTING_ANALYSIS"
    assert first["response"]["classification_claims"] == []
    assert wait(client, submit(client, sid))["status"] == "completed"
    second_sid, _, _ = uploaded_session(client)
    second = wait(client, submit(client, second_sid, analysis_mode="existing_only"))
    assert second["error_code"] == "NO_EXISTING_ANALYSIS"


def test_changed_media_and_missing_artifact_never_reused(real_client):
    client = real_client
    sid, asset, _ = uploaded_session(client)
    done = wait(client, submit(client, sid))
    runtime = client.app.state.runtime
    artifact = runtime.settings.data_dir / done["response"]["prediction"]["artifact_rel"]
    artifact.write_text("changed", encoding="utf-8")
    unavailable = wait(client, submit(client, sid, analysis_mode="existing_only"))
    assert unavailable["error_code"] == "NO_EXISTING_ANALYSIS"
    (runtime.settings.data_dir / asset["storage_rel"]).write_bytes(b"changed")
    rejected = wait(client, submit(client, sid))
    assert rejected["error_code"] == "MEDIA_CHANGED"


def test_tool_error_is_not_success(real_client, monkeypatch):
    def fail(self, request):
        raise VisionError("VISION_TIMEOUT")

    monkeypatch.setattr(RealVisionTool, "execute", fail)
    sid, _, _ = uploaded_session(real_client)
    run = wait(real_client, submit(real_client, sid))
    assert run["status"] == "failed" and run["error_code"] == "VISION_TIMEOUT"
    assert run["response"]["classification_claims"] == []


def test_wrong_result_clip_fails_closed(real_client, monkeypatch):
    def wrong(self, request):
        result = make_prediction(self, request)
        return result.model_copy(update={"execution_id": "forged"})

    monkeypatch.setattr(RealVisionTool, "execute", wrong)
    sid, _, _ = uploaded_session(real_client)
    run = wait(real_client, submit(real_client, sid))
    assert run["status"] == "failed" and run["response"]["classification_claims"] == []


@pytest.mark.parametrize(
    "change",
    [
        {"predicted_label_ids": [2, 2]},
        {"predicted_label_ids": [10]},
        {"supported_label_ids": [2]},
        {"scores": []},
        {"reported_fps": 0},
        {"decoded_frames": 901},
        {"video_sha256": "bad"},
    ],
)
def test_output_contract_rejects_invalid_fields(real_client, change):
    sid, _, _ = uploaded_session(real_client)
    done = wait(real_client, submit(real_client, sid))
    output = done["response"]["prediction"]
    with pytest.raises(ValidationError):
        ClipPrediction.model_validate({**output, **change})


def test_real_settings_missing_runtime_fails_early():
    with pytest.raises(ValueError, match="CPR_ML_PYTHON"):
        Settings(vision_mode="real")
    with pytest.raises(ValueError, match="must be"):
        replace(Settings(), vision_mode="typo")


@pytest.mark.parametrize("operation", ["cancel", "timeout"])
def test_actual_subprocess_is_killed_on_cancel_or_timeout(tmp_path, monkeypatch, operation):
    processes = []
    original_popen = subprocess.Popen

    def slow_process(command, **kwargs):
        child = original_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr("cpr_agent.vision.subprocess.Popen", slow_process)
    settings = Settings(
        data_dir=tmp_path,
        vision_mode="real",
        ml_python=Path(sys.executable),
        vision_timeout_seconds=0.2 if operation == "timeout" else 30,
    )
    with TestClient(create_app(settings)) as client:
        sid, _, _ = uploaded_session(client)
        rid = submit(client, sid)
        wait(client, rid, lambda r: bool(processes))
        if operation == "cancel":
            client.post(f"/api/runs/{rid}/cancel").raise_for_status()
        run = wait(client, rid)
        assert run["status"] == ("cancelled" if operation == "cancel" else "failed")
        assert run["error_code"] == ("USER_CANCELLED" if operation == "cancel" else "VISION_TIMEOUT")
        assert not (run["response"] or {}).get("classification_claims")
    assert processes and all(p.poll() is not None for p in processes)


def test_graceful_shutdown_keeps_retryable_checkpoint(tmp_path, monkeypatch):
    processes = []
    original_popen = subprocess.Popen

    def slow_process(command, **kwargs):
        child = original_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr("cpr_agent.vision.subprocess.Popen", slow_process)
    settings = Settings(data_dir=tmp_path, vision_mode="real", ml_python=Path(sys.executable))
    with TestClient(create_app(settings)) as client:
        sid, _, _ = uploaded_session(client)
        rid = submit(client, sid)
        wait(client, rid, lambda r: bool(processes) and bool((r["wait"] or {}).get("interrupt_id")))
        first_tool = client.get(f"/api/runs/{rid}/tools").json()[0]["id"]
    assert all(p.poll() is not None for p in processes)
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)
    with TestClient(create_app(settings)) as client:
        run = wait(client, rid)
        assert run["status"] == "completed"
        tools = client.get(f"/api/runs/{rid}/tools").json()
        assert len(tools) == 1 and tools[0]["id"] == first_tool


def test_completed_result_reused_after_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)
    settings = Settings(data_dir=tmp_path, vision_mode="real", ml_python=Path(sys.executable))
    with TestClient(create_app(settings)) as client:
        sid, _, _ = uploaded_session(client)
        first = wait(client, submit(client, sid))["response"]["prediction"]["execution_id"]
    with TestClient(create_app(settings)) as client:
        run = wait(client, submit(client, sid, analysis_mode="existing_only"))
        assert run["status"] == "completed" and run["response"]["reused_result"]
        assert run["response"]["prediction"]["execution_id"] == first
        assert client.get(f"/api/runs/{run['id']}/tools").json() == []


def test_failed_response_never_displays_old_findings(real_client):
    from cpr_agent.models import ReviewGoal
    from cpr_agent.vision import build_response

    sid, _, _ = uploaded_session(real_client)
    done = wait(real_client, submit(real_client, sid))
    prediction = ClipPrediction.model_validate(done["response"]["prediction"])
    response = build_response(
        ReviewGoal(intent="analyse", target_label_ids=[2]), prediction, "SERVICE_TIMEOUT"
    )
    assert response.classification_claims == [] and response.prediction is None


def test_evidence_contains_bound_readouts_and_scoped_followups(real_client):
    client = real_client
    sid, asset, _ = uploaded_session(client)
    first = wait(client, submit(client, sid, targets=[2, 4]))
    evidence = client.get(f"/api/reports/{first['report_id']}/evidence").json()
    assert evidence["status"] == "available"
    review = evidence["review"]
    assert review["source_run_id"] == first["id"]
    assert review["video_sha256"] == asset["sha256"]
    assert review["displayed_label_ids"] == [2, 4]
    assert review["flagged_label_ids"] == [2]
    assert review["assessed_label_ids"] == list(CANDIDATE_LABELS)
    assert review["nominal_duration_seconds"] == 2
    assert review["rgb_usable_samples"] == 12 and review["hand_usable_samples"] == 8
    assert review["findings"][0]["threshold"] == 0.5
    assert review["findings"][0]["score_meets_threshold"] is True
    for intent, title in [("explain", "Finding evidence"), ("summarise", "Review summary")]:
        followup = wait(
            client, submit(client, sid, intent=intent, targets=[4], analysis_mode="existing_only")
        )
        response = followup["response"]
        assert response["title"] == title
        assert response["review"]["displayed_label_ids"] == [4]
        assert response["review"]["source_run_id"] == first["id"]
        assert response["reused_result"] and response["review"]["flagged_label_ids"] == []
        assert client.get(f"/api/runs/{followup['id']}/tools").json() == []


def test_reopened_evidence_fails_closed_when_source_changes_but_snapshot_is_retained(real_client):
    client = real_client
    sid, _, _ = uploaded_session(client)
    run = wait(client, submit(client, sid))
    snapshot = client.get(f"/api/reports/{run['report_id']}").json()
    artifact = client.app.state.runtime.settings.data_dir / run["response"]["prediction"]["artifact_rel"]
    artifact.write_text("{}", encoding="utf-8")
    result = client.get(f"/api/reports/{run['report_id']}/evidence").json()
    assert result == {"status": "unavailable", "review": None, "error_code": "RESULT_ARTIFACT_CHANGED"}
    assert client.get(f"/api/reports/{run['report_id']}").json() == snapshot
    assert client.get(f"/api/sessions/{sid}/history").json()["items"][0]["report_id"] == run["report_id"]
    assert len(client.get(f"/api/runs/{run['id']}/tools").json()) == 1


@pytest.mark.parametrize("replacement", [[], {"schema": "wrong"}])
def test_raw_artifact_content_is_validated_beyond_its_hash(real_client, monkeypatch, replacement):
    def invalid(self, request):
        result = make_prediction(self, request)
        path = self.settings.data_dir / result.artifact_rel
        path.write_text(json.dumps(replacement), encoding="utf-8")
        return result.model_copy(update={"artifact_sha256": file_hash(path)})

    monkeypatch.setattr(RealVisionTool, "execute", invalid)
    sid, _, _ = uploaded_session(real_client)
    run = wait(real_client, submit(real_client, sid))
    assert run["status"] == "failed"
    assert not (run.get("response") or {}).get("classification_claims")


def test_branch_threshold_does_not_override_recorded_fusion_decision(real_client, monkeypatch):
    def branch_disagrees(self, request):
        result = make_prediction(self, request)
        path = self.settings.data_dir / result.artifact_rel
        raw = json.loads(path.read_text())
        raw["scores"][3]["uncalibrated_score"] = 0.8  # Label 4 is not in final predictions.
        path.write_text(json.dumps(raw), encoding="utf-8")
        return result.model_copy(
            update={
                "scores": [type(result.scores[0]).model_validate(s) for s in raw["scores"]],
                "artifact_sha256": file_hash(path),
            }
        )

    monkeypatch.setattr(RealVisionTool, "execute", branch_disagrees)
    sid, _, _ = uploaded_session(real_client)
    run = wait(real_client, submit(real_client, sid, targets=[4]))
    finding = run["response"]["review"]["findings"][0]
    assert finding["state"] == "not_flagged" and finding["score_meets_threshold"]
    assert "does not determine the fused result" in finding["decision_note"]
