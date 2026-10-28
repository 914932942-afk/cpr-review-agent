"""Operational navigation and secret-free startup; no external requests or real CV."""

import importlib.util
import json
from pathlib import Path

import pytest
from cpr_agent import mainline_demo
from cpr_agent.api import create_app
from cpr_agent.vision import RealVisionTool, file_hash
from fastapi.testclient import TestClient
from test_context_binding import first_review, settings
from test_real_video import make_prediction, uploaded_session

PROJECT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("mainline_launcher", PROJECT / "scripts/run_main.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_literal_config_and_explicit_environment_priority(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        '\ufeff# private\nDEEPSEEK_API_KEY="fixture"\nDEEPSEEK_MODEL=example\nOTHER=ignored\n',
        encoding="utf-8",
    )
    value = launcher.provider_values(path, {"DEEPSEEK_MODEL": "override"})
    assert value == {"DEEPSEEK_API_KEY": "fixture", "DEEPSEEK_MODEL": "override"}
    assert launcher.provider_values(tmp_path / "absent", {}) == {}
    path.write_text("DEEPSEEK_API_KEY=$(not-executed)\n", encoding="utf-8")
    assert launcher.provider_values(path, {})["DEEPSEEK_API_KEY"] == "$(not-executed)"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)
    manifest = tmp_path / "demo.json"
    monkeypatch.setattr(mainline_demo, "MANIFEST", manifest)
    with TestClient(create_app(settings(tmp_path))) as c:
        yield c, manifest


def catalogue(c, path, case):
    path.write_text(
        json.dumps(
            {"pipeline_sha256": file_hash(c.app.state.runtime.settings.branch_manifest), "cases": [case]}
        ),
        encoding="utf-8",
    )


def test_verified_report_navigation_is_read_only_and_escaped(client):
    c, path = client
    sid, asset, clip, run = first_review(c)
    case = {
        "id": "D01",
        "title": "A < B",
        "workspace": {
            "session_id": sid,
            "clip_id": clip["id"],
            "asset_id": asset["id"],
            "name": "<script>fixture</script>",
        },
        "run_id": run["id"],
    }
    catalogue(c, path, case)
    before = c.get(f"/api/runs/{run['id']}/tools").json()
    response = c.get("/demo/open/D01")
    assert response.status_code == 200 and "A &lt; B" in response.text
    assert "\\u003cscript>" in response.text
    assert "no-store" in response.headers["cache-control"]
    assert c.get(f"/api/runs/{run['id']}/tools").json() == before
    assert c.get("/demo/open/unknown").status_code == 404


def test_catalogue_version_and_asset_mismatch_fail_closed(client):
    c, path = client
    sid, asset, clip = uploaded_session(c)
    case = {
        "id": "D03",
        "title": "New",
        "workspace": {"session_id": sid, "clip_id": clip["id"], "asset_id": "wrong", "name": "New"},
    }
    catalogue(c, path, case)
    assert c.get("/demo/open/D03").status_code == 409
    data = json.loads(path.read_text())
    data["pipeline_sha256"] = "changed"
    path.write_text(json.dumps(data))
    assert c.get("/demo/open/D03").json()["error"]["error_code"] == "DEMO_VERSION_CHANGED"


def test_unanalysed_case_never_inherits_another_prediction(client):
    c, path = client
    first_review(c)
    sid, asset, clip = uploaded_session(c)
    case = {
        "id": "D03",
        "title": "New",
        "workspace": {"session_id": sid, "clip_id": clip["id"], "asset_id": asset["id"], "name": "New"},
    }
    catalogue(c, path, case)
    response = c.get("/demo/open/D03")
    assert response.status_code == 200 and 'removeItem("cpr.runId")' in response.text
    assert c.get(f"/api/sessions/{sid}/context").json()["source"] is None


def test_navigation_refuses_active_tasks(client, monkeypatch):
    c, path = client
    sid, asset, clip = uploaded_session(c)
    case = {
        "id": "D03",
        "title": "New",
        "workspace": {"session_id": sid, "clip_id": clip["id"], "asset_id": asset["id"], "name": "New"},
    }
    catalogue(c, path, case)
    monkeypatch.setattr(c.app.state.runtime.store, "active_runs", lambda: [{"id": "active"}])
    assert c.get("/demo/open/D03").json()["error"]["error_code"] == "DEMO_REVIEW_BUSY"
