"""Optimization profile isolation; numerical and real-video evidence is recorded separately."""

from __future__ import annotations

import json

import pytest
from cpr_agent.api import create_app
from cpr_agent.branch_vision import BranchVisionTool, validate_pipeline
from cpr_agent.config import ROOT
from cpr_agent.evidence import file_hash
from cpr_agent.vision import VisionError
from fastapi.testclient import TestClient
from test_branches import configured, fake_stage
from test_real_video import submit, uploaded_session, wait

FAST = ROOT / "data/m1-optimization-v2/pipeline-manifest.json"
ORIGINAL = ROOT / "data/m1-stability-v1/pipeline-manifest.json"


def test_accepted_optimization_status_and_profile_are_distinct(tmp_path):
    settings = configured(tmp_path, branch_manifest=FAST)
    assert validate_pipeline(settings) == file_hash(FAST) != file_hash(ORIGINAL)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["llm"] == "disabled"
        assert client.get("/api/capabilities").json()["module_status"] == {
            "M1": "Validated M1 · optimized implementation"
        }


@pytest.mark.parametrize("change", ["implementation", "unbound_worker"])
def test_changed_implementation_or_unbound_worker_is_not_executed(tmp_path, change):
    value = json.loads(FAST.read_text())
    if change == "implementation":
        value["sources"]["experiments/m1_fast_v1/algorithm.py"] = "0" * 64
        expected = "PIPELINE_SOURCE_CHANGED"
    else:
        value["sources"].pop("scripts/m1_fast_branch_worker.py")
        expected = "PIPELINE_WORKER_NOT_PINNED"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))
    with pytest.raises(VisionError, match=expected):
        validate_pipeline(configured(tmp_path, branch_manifest=path))


def test_original_m1_result_cannot_be_silently_relabelled_as_optimized(tmp_path, monkeypatch):
    monkeypatch.setattr(BranchVisionTool, "execute", fake_stage)
    with TestClient(create_app(configured(tmp_path, branch_manifest=ORIGINAL))) as client:
        old_sid, _, clip = uploaded_session(client)
        original = wait(client, submit(client, old_sid))
        assert original["status"] == "completed"
    with TestClient(create_app(configured(tmp_path, branch_manifest=FAST))) as client:
        old = next(w for w in client.get("/api/workspaces").json() if w["session_id"] == old_sid)
        assert old["bundle_compatible"] is False
        sid = client.post("/api/sessions", json={"clip_id": clip["id"]}).json()["id"]
        run = wait(client, submit(client, sid, analysis_mode="existing_only"))
        assert run["status"] == "failed" and run["error_code"] == "NO_EXISTING_ANALYSIS"
        assert client.get(f"/api/runs/{run['id']}/tools").json() == []
        assert client.get(f"/api/reports/{original['report_id']}").status_code == 200
