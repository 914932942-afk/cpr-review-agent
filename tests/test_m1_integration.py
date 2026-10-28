"""M1 profile binding and evidence boundaries; numerical gates run in the ML environment."""

from __future__ import annotations

import json

import pytest
from cpr_agent.api import create_app
from cpr_agent.branch_vision import BranchVisionTool, pipeline_worker, validate_pipeline
from cpr_agent.config import ROOT
from cpr_agent.evidence import EvidenceError, file_hash, skeleton_processing
from cpr_agent.vision import VisionError
from fastapi.testclient import TestClient
from test_branches import configured, fake_stage
from test_real_video import submit, uploaded_session, wait

MANIFEST = ROOT / "data/m1-stability-v1/pipeline-manifest.json"


def audit():
    return {
        "schema": "cpr-m1-stability-v3",
        "frames": 50,
        "geometry_available": True,
        "original_observed_points": 800,
        "isolated_points_corrected": 3,
        "short_gap_points_reconstructed": 10,
        "unavailable_points_preserved": 40,
        "reconstructed_points_are_observations": False,
    }


def test_m1_worker_and_frozen_sources_are_bound(tmp_path):
    settings = configured(tmp_path, branch_manifest=MANIFEST)
    assert validate_pipeline(settings) == file_hash(MANIFEST)
    value = json.loads(MANIFEST.read_text())
    assert pipeline_worker(value) == ROOT / "scripts/m1_branch_worker.py"
    value["sources"]["experiments/m1_stability_v3/algorithm.py"] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value))
    with pytest.raises(VisionError, match="PIPELINE_SOURCE_CHANGED"):
        validate_pipeline(configured(tmp_path, branch_manifest=changed))


@pytest.mark.parametrize(
    "worker", ["scripts/other.py", "../m1_branch_worker.py", "scripts/m1_branch_worker.py"]
)
def test_unknown_or_unpinned_workers_rejected(worker):
    with pytest.raises(VisionError, match="PIPELINE_WORKER_NOT_PINNED"):
        pipeline_worker({"worker_entrypoint": worker, "sources": {}})


def test_old_reports_have_no_synthetic_m1_audit():
    assert skeleton_processing({}) is None
    assert skeleton_processing({"fullpose_audit": {}}) is None


def test_m1_audit_frame_count_matches_its_bound_video():
    raw = {"decoded_frames": 50, "fullpose_audit": {"m1_stability": audit()}}
    assert skeleton_processing(raw).frames == 50
    raw["decoded_frames"] = 51
    with pytest.raises(EvidenceError, match="SKELETON_AUDIT_INVALID"):
        skeleton_processing(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("isolated_points_corrected", 801),
        ("unavailable_points_preserved", 41),
        ("short_gap_points_reconstructed", -1),
        ("schema", "unverified"),
        ("reconstructed_points_are_observations", True),
        ("frames", 7),
    ],
)
def test_invalid_m1_audit_is_not_published(field, value):
    source = audit()
    source[field] = value
    with pytest.raises(EvidenceError, match="SKELETON_AUDIT_INVALID"):
        skeleton_processing({"decoded_frames": 50, "fullpose_audit": {"m1_stability": source}})


def test_m1_status_evidence_and_memory_survive_restart(tmp_path, monkeypatch):
    def with_audit(self, request):
        result = fake_stage(self, request)
        if request.tool_id == "fusion.predict":
            path = self.settings.data_dir / result.artifact_rel
            raw = json.loads(path.read_text())
            raw["fullpose_audit"] = {"m1_stability": audit()}
            path.write_text(json.dumps(raw))
            result = result.model_copy(update={"artifact_sha256": file_hash(path)})
        return result

    monkeypatch.setattr(BranchVisionTool, "execute", with_audit)
    settings = configured(tmp_path, branch_manifest=MANIFEST)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/capabilities").json()["module_status"] == {
            "M1": "Validated stability improvement · active"
        }
        sid, _, _ = uploaded_session(client)
        original = wait(client, submit(client, sid))
        assert original["status"] == "completed", original
        proof = client.get(f"/api/reports/{original['report_id']}/evidence").json()
        assert proof["review"]["skeleton_processing"]["isolated_points_corrected"] == 3
    with TestClient(create_app(settings)) as client:
        following = wait(client, submit(client, sid, intent="summarise", analysis_mode="existing_only"))
        assert following["status"] == "completed" and following["response"]["reused_result"]
        assert (
            following["response"]["review"]["skeleton_processing"] == proof["review"]["skeleton_processing"]
        )
        assert client.get(f"/api/runs/{following['id']}/tools").json() == []


def test_original_branch_profile_does_not_claim_m1(tmp_path):
    with TestClient(create_app(configured(tmp_path))) as client:
        assert client.get("/api/capabilities").json()["module_status"] == {}
