"""Persistence and scope checks for browsing saved reviews (no vision or LLM calls)."""

from __future__ import annotations

from uuid import uuid4

from cpr_agent.api import create_app
from cpr_agent.config import Settings
from fastapi.testclient import TestClient


def add_turn(client, session_id, message="Saved request"):
    response = client.post(
        f"/api/sessions/{session_id}/turns",
        json={"message": message, "goal": {"intent": "summarise"}},
        headers={"Idempotency-Key": str(uuid4())},
    )
    assert response.status_code == 202
    return response.json()


def test_history_cursor_is_stable_for_equal_timestamps_and_across_restart(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        sid = client.post("/api/sessions", json={}).json()["id"]
        store = client.app.state.runtime.store
        # Seed committed records directly; timestamps intentionally tie to test the cursor.
        ids = []
        for index in range(5):
            result = store.create_turn(
                sid,
                {"message": f"Request {index}", "goal": None},
                str(uuid4()),
                settings.bundle_id,
                settings.graph_version,
            )
            store.commit_run(result["run_id"], "failed", None, "TEST_ERROR")
            ids.append(result["turn_id"])
        with store.tx() as db:
            db.execute("UPDATE objects SET created=100 WHERE kind='turn'")
        first = client.get(f"/api/sessions/{sid}/history?limit=2").json()
        assert [i["turn_id"] for i in first["items"]] == sorted(ids, reverse=True)[:2]
        assert first["has_more"]
        cursor = first["next_before_turn_id"]
    with TestClient(create_app(settings)) as client:
        second = client.get(
            f"/api/sessions/{sid}/history", params={"limit": 2, "before_turn_id": cursor}
        ).json()
        third = client.get(
            f"/api/sessions/{sid}/history",
            params={"limit": 2, "before_turn_id": second["next_before_turn_id"]},
        ).json()
        combined = first["items"] + second["items"] + third["items"]
        assert [i["turn_id"] for i in combined] == sorted(ids, reverse=True)
        assert third["has_more"] is False and third["next_before_turn_id"] is None
        assert all(i["status"] == "failed" and i["error_code"] == "TEST_ERROR" for i in combined)


def test_history_cursor_scope_and_owner_are_enforced(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        sid = client.post("/api/sessions", json={}).json()["id"]
        other = client.post("/api/sessions", json={}).json()["id"]
        result = add_turn(client, other)
        assert (
            client.get(
                f"/api/sessions/{sid}/history", params={"before_turn_id": result["turn_id"]}
            ).status_code
            == 404
        )
        assert client.get(f"/api/sessions/{sid}/history?limit=0").status_code == 422
        assert client.get(f"/api/sessions/{sid}/history?limit=51").status_code == 422
    with TestClient(create_app(Settings(data_dir=tmp_path, owner_id="another-owner"))) as client:
        assert client.get(f"/api/sessions/{sid}/history").status_code == 404
        assert client.get("/api/workspaces").json() == []


def test_workspaces_recover_without_browser_state_and_do_not_expose_paths(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        demo = client.post("/api/demo/session").json()
        sid = demo["session"]["id"]
        result = add_turn(client, sid, "Remember this request")
    with TestClient(create_app(settings)) as client:
        rows = client.get("/api/workspaces").json()
        row = next(r for r in rows if r["session_id"] == sid)
        assert row["name"] == demo["asset"]["name"]
        assert row["asset_id"] == demo["asset"]["id"] and row["bundle_compatible"]
        assert row["latest_run_id"] == result["run_id"]
        history = client.get(f"/api/sessions/{sid}/history").json()
        entry = history["items"][0]
        assert entry["message"] == "Remember this request"
        assert entry["run_id"] == result["run_id"]
        assert not {"storage_rel", "prediction", "llm_resolution", "api_key"} & set(entry)
        assert "storage_rel" not in str(rows)
        assert client.get(f"/api/runs/{result['run_id']}").status_code == 200


def test_old_bundle_sessions_remain_browsable_but_cannot_create_new_turns(tmp_path):
    first = Settings(data_dir=tmp_path, bundle_id="old-bundle")
    with TestClient(create_app(first)) as client:
        sid = client.post("/api/sessions", json={}).json()["id"]
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        row = client.get("/api/workspaces").json()[0]
        assert row["bundle_compatible"] is False
        assert client.get(f"/api/sessions/{sid}/history").json()["items"] == []
        response = client.post(
            f"/api/sessions/{sid}/turns",
            json={"goal": {"intent": "summarise"}},
            headers={"Idempotency-Key": str(uuid4())},
        )
        assert response.status_code == 409


def test_empty_and_unfinished_sessions_are_visible(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        sid = client.post("/api/sessions", json={}).json()["id"]
        assert client.get(f"/api/sessions/{sid}/history").json()["items"] == []
        assert client.get("/api/workspaces").json()[0]["latest_run_id"] is None
        result = client.post(
            f"/api/sessions/{sid}/turns",
            json={"message": "Needs clarification"},
            headers={"Idempotency-Key": str(uuid4())},
        ).json()
        entry = client.get(f"/api/sessions/{sid}/history").json()["items"][0]
        assert entry["run_id"] == result["run_id"]
        assert entry["status"] in {"queued", "running", "waiting_user"}
        assert entry["report_id"] is None
