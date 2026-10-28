"""Branch orchestration boundaries with explicit test doubles; not vision accuracy tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from cpr_agent.api import create_app
from cpr_agent.branch_vision import BranchVisionTool
from cpr_agent.config import Settings
from cpr_agent.models import BranchReceipt
from cpr_agent.vision import RealVisionTool, VisionError, file_hash, media_context, vision_profile
from fastapi.testclient import TestClient
from test_real_video import make_prediction, submit, uploaded_session, wait


def fake_stage(self, request):
    _, asset, _ = media_context(self.store, self.settings, request.clip_id)
    pipeline_sha = file_hash(self.settings.branch_manifest)
    deps = {}
    for binding, tool in {
        "pose_receipt_id": "pose.extract",
        "skeleton_receipt_id": "skeleton.predict",
        "rgb_receipt_id": "rgb.predict",
    }.items():
        if binding in request.input_bindings:
            deps[tool] = self.store.get_tool(request.input_bindings[binding])["output"]["artifact_sha256"]
    if request.tool_id == "fusion.predict":
        result = make_prediction(self, request)
        path = self.settings.data_dir / result.artifact_rel
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.update(pipeline_sha256=pipeline_sha, dependencies=deps)
        path.write_text(json.dumps(raw), encoding="utf-8")
        return result.model_copy(
            update={"profile_id": vision_profile(self.settings), "artifact_sha256": file_hash(path)}
        )
    folder = self.settings.data_dir / "analysis" / request.execution_id
    folder.mkdir(parents=True, exist_ok=True)
    feature = folder / "explicit-test-double.bin"
    feature.write_bytes(b"No real vision features; unit-test double only")
    raw = {
        "schema": "cpr-decomposed-stage-v1",
        "status": "completed_experimental",
        "tool_id": request.tool_id,
        "video_sha256": asset["sha256"],
        "bundle_sha256": file_hash(self.settings.vision_manifest),
        "pipeline_sha256": pipeline_sha,
        "labels_used": False,
        "model_or_threshold_modified": False,
        "decoded_frames": 50,
        "reported_fps": 25,
        "supported_label_ids": []
        if request.tool_id == "pose.extract"
        else ([1, 2, 5, 7, 9] if request.tool_id == "rgb.predict" else [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]),
        "files": [{"name": feature.name, "sha256": file_hash(feature)}],
        "dependencies": deps,
    }
    if deps:
        raw["shared_pose_sha256"] = deps["pose.extract"]
    path = folder / "stage.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return BranchReceipt(
        execution_id=request.execution_id,
        tool_id=request.tool_id,
        clip_id=request.clip_id,
        asset_id=asset["id"],
        profile_id=request.profile_id,
        video_sha256=asset["sha256"],
        bundle_sha256=file_hash(self.settings.vision_manifest),
        artifact_rel=path.relative_to(self.settings.data_dir).as_posix(),
        artifact_sha256=file_hash(path),
        analysis_seconds=0.01,
    )


def configured(tmp_path, **extra):
    return Settings(
        data_dir=tmp_path,
        vision_mode="real",
        vision_pipeline="branches",
        ml_python=Path(sys.executable),
        **extra,
    )


@pytest.fixture
def branch_client(tmp_path, monkeypatch):
    monkeypatch.setattr(BranchVisionTool, "execute", fake_stage)
    with TestClient(create_app(configured(tmp_path))) as client:
        yield client


def test_four_real_contracts_and_followups_only_reuse_final_evidence(branch_client):
    client = branch_client
    assert client.get("/api/capabilities").json()["tools"] == [
        "pose.extract",
        "skeleton.predict",
        "rgb.predict",
        "fusion.predict",
    ]
    sid, _, _ = uploaded_session(client)
    run = wait(client, submit(client, sid))
    assert run["status"] == "completed", run
    tools = client.get(f"/api/runs/{run['id']}/tools").json()
    assert [t["request"]["tool_id"] for t in tools] == [
        "pose.extract",
        "skeleton.predict",
        "rgb.predict",
        "fusion.predict",
    ]
    assert all(t["status"] == "succeeded" and not t["output"]["is_stub"] for t in tools)
    assert all(t["output"]["output_kind"] == "branch_artifact" for t in tools[:3])
    assert tools[3]["output"]["output_kind"] == "clip_prediction"
    assert (
        tools[1]["request"]["input_bindings"]["pose_receipt_id"]
        == tools[2]["request"]["input_bindings"]["pose_receipt_id"]
        == tools[0]["id"]
    )
    assert run["response"]["prediction"]["profile_id"].startswith("single-view-v15-branches-v1:")
    assert client.get(f"/api/reports/{run['report_id']}/evidence").json()["status"] == "available"
    for intent in ["explain", "summarise", "verify"]:
        following = wait(client, submit(client, sid, intent=intent, analysis_mode="existing_only"))
        assert following["status"] == "completed" and following["response"]["reused_result"]
        assert following["response"]["prediction"]["execution_id"] == tools[3]["id"]
        assert client.get(f"/api/runs/{following['id']}/tools").json() == []


@pytest.mark.parametrize(
    "failure,index", [("pose.extract", 0), ("skeleton.predict", 1), ("rgb.predict", 2), ("fusion.predict", 3)]
)
def test_failed_stage_stops_dependents_without_publishing_partial_findings(
    branch_client, monkeypatch, failure, index
):
    def broken(self, request):
        if request.tool_id == failure:
            raise VisionError("DELIBERATE_STAGE_FAILURE")
        return fake_stage(self, request)

    monkeypatch.setattr(BranchVisionTool, "execute", broken)
    sid, _, _ = uploaded_session(branch_client)
    run = wait(branch_client, submit(branch_client, sid))
    assert run["status"] == "failed" and run["error_code"] == "DELIBERATE_STAGE_FAILURE"
    assert run["response"]["classification_claims"] == []
    assert len(branch_client.get(f"/api/runs/{run['id']}/tools").json()) == index + 1


def test_mismatched_actor_track_is_not_fused(branch_client, monkeypatch):
    def mismatched(self, request):
        result = fake_stage(self, request)
        if request.tool_id == "rgb.predict":
            path = self.settings.data_dir / result.artifact_rel
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["shared_pose_sha256"] = "0" * 64
            path.write_text(json.dumps(raw), encoding="utf-8")
            return result.model_copy(update={"artifact_sha256": file_hash(path)})
        return result

    monkeypatch.setattr(BranchVisionTool, "execute", mismatched)
    sid, _, _ = uploaded_session(branch_client)
    run = wait(branch_client, submit(branch_client, sid))
    assert run["status"] == "failed" and not (run.get("response") or {}).get("classification_claims")
    assert len(branch_client.get(f"/api/runs/{run['id']}/tools").json()) == 3


def test_changed_intermediate_artifact_invalidates_reuse(branch_client):
    client = branch_client
    sid, _, _ = uploaded_session(client)
    run = wait(client, submit(client, sid))
    pose = client.get(f"/api/runs/{run['id']}/tools").json()[0]
    path = client.app.state.runtime.settings.data_dir / pose["output"]["artifact_rel"]
    path.with_name("explicit-test-double.bin").write_bytes(b"changed")
    assert client.get(f"/api/reports/{run['report_id']}/evidence").json()["status"] == "unavailable"
    followup = wait(client, submit(client, sid, analysis_mode="existing_only"))
    assert followup["status"] == "failed" and followup["error_code"] == "NO_EXISTING_ANALYSIS"
    assert client.get(f"/api/runs/{followup['id']}/tools").json() == []


@pytest.mark.parametrize("operation", ["cancel", "timeout"])
def test_branch_subprocess_stops_on_cancel_or_timeout(tmp_path, monkeypatch, operation):
    processes = []
    original = subprocess.Popen

    def slow(command, **kwargs):
        child = original([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr("cpr_agent.branch_vision.subprocess.Popen", slow)
    with TestClient(
        create_app(configured(tmp_path, vision_timeout_seconds=0.2 if operation == "timeout" else 30))
    ) as client:
        sid, _, _ = uploaded_session(client)
        rid = submit(client, sid)
        wait(client, rid, lambda r: bool(processes))
        if operation == "cancel":
            client.post(f"/api/runs/{rid}/cancel").raise_for_status()
        run = wait(client, rid)
        assert run["status"] == ("cancelled" if operation == "cancel" else "failed")
        assert run["error_code"] == ("USER_CANCELLED" if operation == "cancel" else "VISION_TIMEOUT")
        assert not (run.get("response") or {}).get("classification_claims")
    assert all(p.poll() is not None for p in processes)


def test_branch_result_and_trace_recover_after_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(BranchVisionTool, "execute", fake_stage)
    settings = configured(tmp_path)
    with TestClient(create_app(settings)) as client:
        sid, _, _ = uploaded_session(client)
        original = wait(client, submit(client, sid))
    with TestClient(create_app(settings)) as client:
        history = client.get(f"/api/sessions/{sid}/history").json()
        assert history["items"][0]["run_id"] == original["id"]
        following = wait(client, submit(client, sid, intent="summarise", analysis_mode="existing_only"))
        assert following["status"] == "completed" and following["response"]["reused_result"]
        assert (
            following["response"]["prediction"]["execution_id"]
            == original["response"]["prediction"]["execution_id"]
        )


def test_monolithic_receipt_cannot_be_reused_as_a_branch_result(tmp_path, monkeypatch):
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)
    with TestClient(
        create_app(Settings(data_dir=tmp_path, vision_mode="real", ml_python=Path(sys.executable)))
    ) as client:
        old_sid, _, clip = uploaded_session(client)
        assert wait(client, submit(client, old_sid))["status"] == "completed"
    monkeypatch.setattr(BranchVisionTool, "execute", fake_stage)
    with TestClient(create_app(configured(tmp_path))) as client:
        old_workspace = next(w for w in client.get("/api/workspaces").json() if w["session_id"] == old_sid)
        assert old_workspace["bundle_compatible"] is False
        sid = client.post("/api/sessions", json={"clip_id": clip["id"]}).json()["id"]
        run = wait(client, submit(client, sid, analysis_mode="existing_only"))
        assert run["status"] == "failed" and run["error_code"] == "NO_EXISTING_ANALYSIS"
        assert client.get(f"/api/runs/{run['id']}/tools").json() == []


def test_repeated_submission_does_not_duplicate_the_four_stages(branch_client):
    client = branch_client
    sid, _, _ = uploaded_session(client)
    headers = {"Idempotency-Key": str(uuid4())}
    body = {"goal": {"intent": "analyse", "target_label_ids": [2]}}
    first = client.post(f"/api/sessions/{sid}/turns", json=body, headers=headers)
    second = client.post(f"/api/sessions/{sid}/turns", json=body, headers=headers)
    assert first.status_code == second.status_code == 202 and first.json() == second.json()
    run = wait(client, first.json()["run_id"])
    assert run["status"] == "completed"
    assert len(client.get(f"/api/runs/{run['id']}/tools").json()) == 4
