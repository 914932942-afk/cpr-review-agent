from __future__ import annotations

import time
from uuid import uuid4

import pytest
from cpr_agent.api import create_app
from cpr_agent.config import ROOT, Settings
from cpr_agent.models import NODE_NAMES, TERMINAL
from fastapi.testclient import TestClient


def key():
    return {"Idempotency-Key": str(uuid4())}


def wait_run(client, run_id, predicate=None):
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if predicate(run) if predicate else run["status"] in TERMINAL:
            return run
        time.sleep(0.03)
    pytest.fail(f"Run did not reach the requested state: {run}")


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, stub_delay_seconds=0.01))) as api:
        yield api


def session(client):
    response = client.post("/api/demo/session")
    assert response.status_code == 201, response.text
    return response.json()["session"]["id"]


def turn(client, session_id, body, headers=None):
    response = client.post(f"/api/sessions/{session_id}/turns", json=body, headers=headers or key())
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def resume_body(run, goal):
    return {
        "kind": "user_answer",
        "wait_id": run["wait"]["wait_id"],
        "interrupt_id": run["wait"]["interrupt_id"],
        "revision": run["revision"],
        "goal": goal,
    }


def test_all_twelve_nodes_real_interrupts_and_empty_claims(client):
    run_id = turn(client, session(client), {"message": "Help me review this video"})
    waiting = wait_run(
        client, run_id, lambda r: r["status"] == "waiting_user" and (r["wait"] or {}).get("interrupt_id")
    )
    response = client.post(
        f"/api/runs/{run_id}/resume",
        headers=key(),
        json=resume_body(waiting, {"intent": "analyse", "target_label_ids": [4, 5]}),
    )
    assert response.status_code == 202
    run = wait_run(client, run_id)
    assert run["status"] == "failed" and run["error_code"] == "MODULES_NOT_IMPLEMENTED"
    assert run["response"]["workflow_reached_end"] is True
    assert run["response"]["classification_claims"] == []
    assert set(run["response"]["target_states"].values()) == {"not_evaluated"}
    events = client.get(f"/api/runs/{run_id}/events").json()["events"]
    assert {e["node_id"] for e in events if e["kind"] == "node.enter"} == set(NODE_NAMES)
    assert len({e["seq"] for e in events}) == len(events)
    executions = client.get(f"/api/runs/{run_id}/tools").json()
    assert {e["request"]["tool_id"] for e in executions} == {
        "pose.extract",
        "skeleton.predict",
        "rgb.predict",
        "fusion.predict",
    }
    assert all(e["output"]["is_stub"] and "scores" not in e["output"] for e in executions)
    assert len([e for e in events if e["kind"] == "run.finished"]) == 1
    assert client.get(f"/api/reports/{run['report_id']}/export").status_code == 200


def test_idempotency_conflict_and_single_active_run(client):
    sid, headers, body = session(client), key(), {"message": "Use the form fallback"}
    first = turn(client, sid, body, headers)
    assert turn(client, sid, body, headers) == first
    path = f"/api/sessions/{sid}/turns"
    assert client.post(path, json={"message": "different"}, headers=headers).status_code == 409
    assert client.post(path, json=body, headers=key()).status_code == 409


def test_stale_and_forged_resume_are_rejected(client):
    rid = turn(client, session(client), {"message": "Please review"})
    waiting = wait_run(
        client, rid, lambda r: r["status"] == "waiting_user" and (r["wait"] or {}).get("interrupt_id")
    )
    body = resume_body(waiting, {"intent": "analyse", "target_label_ids": [4]})
    assert (
        client.post(
            f"/api/runs/{rid}/resume", json={**body, "kind": "tool_complete"}, headers=key()
        ).status_code
        == 422
    )
    assert (
        client.post(f"/api/runs/{rid}/resume", json={**body, "wait_id": "stale"}, headers=key()).status_code
        == 409
    )
    headers = key()
    assert client.post(f"/api/runs/{rid}/resume", json=body, headers=headers).status_code == 202
    assert client.post(f"/api/runs/{rid}/resume", json=body, headers=headers).status_code == 202
    assert client.post(f"/api/runs/{rid}/resume", json=body, headers=key()).status_code == 409
    wait_run(client, rid)


def test_existing_only_never_dispatches_tools(client):
    rid = turn(
        client,
        session(client),
        {"goal": {"intent": "analyse", "target_label_ids": [4], "analysis_mode": "existing_only"}},
    )
    run = wait_run(client, rid)
    assert run["response"]["classification_claims"] == []
    assert client.get(f"/api/runs/{rid}/tools").json() == []


def test_examples_without_video_and_explicit_memory(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    rid = turn(client, sid, {"goal": {"intent": "show_example"}})
    assert wait_run(client, rid)["status"] == "completed"
    assert client.get(f"/api/runs/{rid}/tools").json() == []
    rid = turn(
        client,
        sid,
        {
            "goal": {
                "intent": "manage_memory",
                "memory_commands": [{"operation": "remember_focus", "label_ids": [4, 5]}],
            }
        },
    )
    saved = wait_run(client, rid)
    assert saved["response"]["memory_receipts"][0]["saved"] is True
    assert client.get("/api/preferences").json()[0]["label_ids"] == [4, 5]
    rid = turn(
        client,
        sid,
        {"goal": {"intent": "manage_memory", "memory_commands": [{"operation": "forget_preference"}]}},
    )
    wait_run(client, rid)
    assert client.get("/api/preferences").json() == []


def test_no_video_analysis_is_rejected(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    rid = turn(client, sid, {"goal": {"intent": "analyse", "target_label_ids": [4]}})
    run = wait_run(client, rid)
    assert run["status"] == "rejected" and run["error_code"] == "CLIP_REQUIRED"
    assert client.get(f"/api/runs/{rid}/tools").json() == []


def test_cancel_during_tool_wait_never_commits_pending_memory(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, stub_delay_seconds=0.4))) as client:
        rid = turn(
            client,
            session(client),
            {
                "goal": {
                    "intent": "analyse",
                    "target_label_ids": [4],
                    "memory_commands": [{"operation": "remember_focus", "label_ids": [4]}],
                }
            },
        )
        wait_run(client, rid, lambda r: r["status"] == "waiting_tool")
        assert client.post(f"/api/runs/{rid}/cancel").status_code == 202
        run = wait_run(client, rid)
        assert run["status"] == "cancelled" and run["report_id"] is None
        assert client.get("/api/preferences").json() == []


@pytest.mark.parametrize("waiting_kind", ["waiting_user", "waiting_tool"])
def test_persistent_checkpoint_resumes_after_runtime_restart(tmp_path, waiting_kind):
    settings = Settings(data_dir=tmp_path, stub_delay_seconds=0.05)
    body = (
        {"message": "Review"}
        if waiting_kind == "waiting_user"
        else {"goal": {"intent": "analyse", "target_label_ids": [4]}}
    )
    with TestClient(create_app(settings)) as client:
        rid = turn(client, session(client), body)
        waiting = wait_run(
            client, rid, lambda r: r["status"] == waiting_kind and (r["wait"] or {}).get("interrupt_id")
        )
    with TestClient(create_app(settings)) as client:
        if waiting_kind == "waiting_user":
            assert (
                client.post(
                    f"/api/runs/{rid}/resume",
                    headers=key(),
                    json=resume_body(waiting, {"intent": "analyse", "target_label_ids": [4]}),
                ).status_code
                == 202
            )
        run = wait_run(client, rid)
        assert run["error_code"] == "MODULES_NOT_IMPLEMENTED"
        tools = client.get(f"/api/runs/{rid}/tools").json()
        assert len(tools) == 4 and len({t["id"] for t in tools}) == 4


def test_tool_failure_stops_invalid_dependencies(tmp_path):
    with TestClient(
        create_app(Settings(data_dir=tmp_path, stub_delay_seconds=0.01, fail_tool="skeleton.predict"))
    ) as client:
        rid = turn(client, session(client), {"goal": {"intent": "analyse", "target_label_ids": [4]}})
        run = wait_run(client, rid)
        assert run["status"] == "failed" and run["response"]["classification_claims"] == []
        tools = client.get(f"/api/runs/{rid}/tools").json()
        assert len(tools) <= 3
        assert "fusion.predict" not in {t["request"]["tool_id"] for t in tools}


def test_timeout_reaches_a_terminal_state(tmp_path):
    with TestClient(
        create_app(Settings(data_dir=tmp_path, service_budget_seconds=0.03, stub_delay_seconds=0.2))
    ) as client:
        rid = turn(client, session(client), {"goal": {"intent": "analyse", "target_label_ids": [4]}})
        run = wait_run(client, rid)
        assert run["status"] == "failed" and run["error_code"] == "SERVICE_TIMEOUT"
        assert run["response"]["classification_claims"] == []


def test_upload_and_clip_binding_boundaries(client):
    assert (
        client.post("/api/assets", files={"file": ("x.exe", b"bad", "application/octet-stream")}).status_code
        == 415
    )
    assert client.post("/api/assets", files={"file": ("x.mp4", b"not video", "video/mp4")}).status_code == 415
    with (ROOT / "fixtures/workflow-fixture.mp4").open("rb") as video:
        result = client.post("/api/assets", files={"file": ("sample.mp4", video, "video/mp4")})
    assert result.status_code == 201 and result.json()["validation_level"] == "storage_and_magic_only"
    clip = client.post("/api/clips", json={"asset_id": result.json()["id"]}).json()
    sid = client.post("/api/sessions", json={}).json()["id"]
    path = f"/api/sessions/{sid}/clip"
    assert client.post(path, json={"clip_id": clip["id"]}).status_code == 200
    assert client.post(path, json={"clip_id": clip["id"]}).status_code == 409


def test_user_form_expiry_closes_without_a_report(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, user_wait_seconds=0.1))) as client:
        rid = turn(client, session(client), {"message": "Review using the form"})
        run = wait_run(client, rid)
        assert run["status"] == "failed" and run["error_code"] == "USER_WAIT_EXPIRED"
        assert run["report_id"] is None


def test_user_wait_does_not_consume_service_budget(tmp_path):
    settings = Settings(data_dir=tmp_path, service_budget_seconds=0.25, user_wait_seconds=3)
    with TestClient(create_app(settings)) as client:
        rid = turn(client, session(client), {"message": "Choose a task"})
        waiting = wait_run(
            client, rid, lambda r: r["status"] == "waiting_user" and (r["wait"] or {}).get("interrupt_id")
        )
        time.sleep(0.35)  # Longer than the complete service budget, shorter than user expiry.
        response = client.post(
            f"/api/runs/{rid}/resume", headers=key(), json=resume_body(waiting, {"intent": "show_example"})
        )
        assert response.status_code == 202
        assert wait_run(client, rid)["status"] == "completed"
