"""Conversation binding, not CV accuracy. All optional LLM calls use MockTransport."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from cpr_agent.api import create_app
from cpr_agent.config import ROOT, Settings
from cpr_agent.llm import create_model_gateway, request_context
from cpr_agent.review_context import ContextBinder
from cpr_agent.vision import RealVisionTool, file_hash
from fastapi.testclient import TestClient
from test_real_video import make_prediction, uploaded_session, wait


def settings(tmp_path, **kwargs):
    return Settings(
        data_dir=tmp_path,
        vision_mode="real",
        ml_python=Path(sys.executable),
        context_mode="bound-v1",
        **kwargs,
    )


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setattr(
        httpx.HTTPTransport, "handle_request", lambda *a, **k: pytest.fail("Real network forbidden")
    )
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as c:
        yield c


def post(
    client,
    sid,
    targets=None,
    *,
    intent="explain",
    report=None,
    scope="explicit",
    source="current_clip",
    message=None,
    headers=None,
):
    body = (
        {"message": message}
        if message is not None
        else {"goal": {"intent": intent, "target_label_ids": targets or [], "analysis_mode": "adaptive"}}
    )
    body["context"] = {"source_report_id": report, "reference": {"scope": scope, "source": source}}
    result = client.post(
        f"/api/sessions/{sid}/turns", json=body, headers=headers or {"Idempotency-Key": str(uuid4())}
    )
    assert result.status_code == 202, result.text
    return result.json()["run_id"]


def first_review(client, targets=None):
    sid, asset, clip = uploaded_session(client)
    run = wait(client, post(client, sid, targets or [2, 4], intent="analyse"))
    assert run["status"] == "completed", run
    return sid, asset, clip, run


def waiting(client, rid):
    return wait(client, rid, lambda r: r["status"] == "waiting_user" and r["wait"].get("interrupt_id"))


def resume_body(run, targets=None, intent="explain"):
    return {
        "wait_id": run["wait"]["wait_id"],
        "interrupt_id": run["wait"]["interrupt_id"],
        "revision": run["revision"],
        "goal": {"intent": intent, "target_label_ids": targets or []},
        "context_confirmation": {"snapshot_id": run["wait"]["context_snapshot_id"], "use_bound_source": True},
    }


def resume(client, run, targets=None, intent="explain", headers=None):
    return client.post(
        f"/api/runs/{run['id']}/resume",
        json=resume_body(run, targets, intent),
        headers=headers or {"Idempotency-Key": str(uuid4())},
    )


def test_context_persisted_atomically_and_older_view_wins_over_latest(client):
    sid, asset, clip, first = first_review(client)
    original_prediction = first["response"]["prediction"]
    second = wait(client, post(client, sid, [4], report=first["report_id"]))
    previous = wait(
        client, post(client, sid, report=first["report_id"], scope="previous_scope", source="viewed_report")
    )
    assert previous["response"]["review"]["displayed_label_ids"] == [2, 4]
    assert previous["response"]["prediction"] == original_prediction
    assert client.get(f"/api/runs/{previous['id']}/tools").json() == []
    receipt = previous["review_context"]
    assert receipt["source_report_id"] == first["report_id"] != second["report_id"]
    assert receipt["clip_id"] == clip["id"] and receipt["video_sha256"] == asset["sha256"]
    assert receipt["target_label_ids"] == [2, 4] and receipt["scope_reference"] == "previous_scope"
    assert receipt["result_report_id"] == previous["report_id"]
    assert client.get(f"/api/reports/{previous['report_id']}").json()["review_context"] == receipt
    turn = client.get(f"/api/turns/{previous['turn_id']}").json()
    assert turn["context_snapshot"]["source"]["report_id"] == first["report_id"]
    assert second["turn_id"] not in turn["context_snapshot"]["recent_turn_ids"]
    events = client.get(f"/api/runs/{previous['id']}/events").json()["events"]
    assert len([e for e in events if e["kind"] == "context.bound"]) == 1


def test_single_finding_is_inherited_only_when_unique(client):
    sid, _, _, first = first_review(client, [4])
    followup = wait(client, post(client, sid, [2], scope="single_finding", report=first["report_id"]))
    assert followup["response"]["review"]["displayed_label_ids"] == [4]
    assert client.get(f"/api/runs/{followup['id']}/tools").json() == []


def test_ambiguous_scope_requires_explicit_labels_and_fresh_confirmation(client):
    sid, _, _, first = first_review(client)
    rid = post(client, sid, scope="single_finding", report=first["report_id"])
    paused = waiting(client, rid)
    assert paused["wait"]["context_code"] == "CONTEXT_SCOPE_AMBIGUOUS"
    assert client.get(f"/api/turns/{paused['turn_id']}").json().get("resolved_goal") is None
    assert resume(client, paused).status_code == 422
    body = resume_body(paused, [4])
    body["context_confirmation"]["snapshot_id"] = "0" * 64
    assert (
        client.post(
            f"/api/runs/{rid}/resume", json=body, headers={"Idempotency-Key": str(uuid4())}
        ).status_code
        == 409
    )
    headers = {"Idempotency-Key": str(uuid4())}
    assert resume(client, paused, [4], headers=headers).status_code == 202
    done = wait(client, rid)
    assert resume(client, paused, [4], headers=headers).status_code == 202
    assert done["response"]["review"]["displayed_label_ids"] == [4]
    assert done["review_context"]["source_report_id"] == first["report_id"]
    assert done["review_context"]["scope_reference"] == "explicit"
    assert client.get(f"/api/runs/{rid}/tools").json() == []


def test_scope_ambiguity_can_be_resolved_as_explicit_full_summary(client):
    sid, _, _, first = first_review(client)
    rid = post(client, sid, scope="single_finding", report=first["report_id"])
    paused = waiting(client, rid)
    assert resume(client, paused, intent="summarise").status_code == 202
    done = wait(client, rid)
    assert len(done["response"]["classification_claims"]) == 11
    assert done["response"]["reused_result"]


def test_source_ambiguity_stays_on_bound_video_until_user_confirms(client):
    sid, asset, clip, first = first_review(client)
    rid = post(client, sid, [4], source="ambiguous", report=first["report_id"])
    paused = waiting(client, rid)
    assert paused["wait"]["context_code"] == "CONTEXT_SOURCE_AMBIGUOUS"
    assert paused["wait"]["bound_source_report_id"] == first["report_id"]
    body = resume_body(paused, [4])
    body["context_confirmation"]["clip_id"] = "another-video"
    assert (
        client.post(
            f"/api/runs/{rid}/resume", json=body, headers={"Idempotency-Key": str(uuid4())}
        ).status_code
        == 422
    )
    assert resume(client, paused, [4]).status_code == 202
    done = wait(client, rid)
    assert done["response"]["prediction"]["clip_id"] == clip["id"]
    assert done["response"]["prediction"]["video_sha256"] == asset["sha256"]


def test_cross_session_report_is_rejected_even_for_identical_video_bytes(client):
    sid, _, _, first = first_review(client)
    other_sid, _, _ = uploaded_session(client)
    body = {
        "goal": {"intent": "explain", "target_label_ids": [4]},
        "context": {"source_report_id": first["report_id"]},
    }
    response = client.post(
        f"/api/sessions/{other_sid}/turns", json=body, headers={"Idempotency-Key": str(uuid4())}
    )
    assert response.status_code == 409 and response.json()["error"]["error_code"] == "CONTEXT_SOURCE_MISMATCH"
    assert client.get(f"/api/sessions/{other_sid}/history").json()["items"] == []
    assert (
        client.get(
            f"/api/sessions/{other_sid}/context", params={"source_report_id": first["report_id"]}
        ).status_code
        == 409
    )
    assert client.get(f"/api/sessions/{sid}/context").json()["source"]["report_id"] == first["report_id"]


def test_new_video_does_not_inherit_history_or_global_preference(client):
    sid, _, _, first = first_review(client, [4])
    store = client.app.state.runtime.store
    store.put("preference", {"id": "focus", "label_ids": [4], "source_turn_id": first["turn_id"]})
    new_sid, _, _ = uploaded_session(client)
    preview = client.get(f"/api/sessions/{new_sid}/context").json()
    assert preview["active_label_ids"] == [] and preview["recent_goals"] == [] and preview["source"] is None
    rid = post(client, new_sid, scope="previous_scope")
    paused = waiting(client, rid)
    assert paused["wait"]["context_code"] == "CONTEXT_SCOPE_UNAVAILABLE"
    client.post(f"/api/runs/{rid}/cancel").raise_for_status()
    done = wait(client, rid)
    assert done["status"] == "cancelled" and not done["review_context"]["goal_confirmed"]
    assert done["review_context"]["result_execution_id"] is None and done["report_id"] is None


def test_failed_and_cancelled_turns_do_not_replace_confirmed_scope(client):
    sid, _, _, first = first_review(client, [2])
    store = client.app.state.runtime.store
    artifact = client.app.state.runtime.settings.data_dir / first["response"]["prediction"]["artifact_rel"]
    original = artifact.read_bytes()
    artifact.write_text("{}", encoding="utf-8")
    failed = wait(client, post(client, sid, [5], intent="analyse"))
    assert failed["status"] == "rejected" and failed["error_code"] == "CONTEXT_SOURCE_UNAVAILABLE"
    artifact.write_bytes(original)
    paused = waiting(client, post(client, sid, message="Not yet confirmed"))
    client.post(f"/api/runs/{paused['id']}/cancel")
    cancelled = wait(client, paused["id"])
    assert cancelled["status"] == "cancelled"
    preview = client.get(f"/api/sessions/{sid}/context").json()
    assert preview["active_label_ids"] == [2]
    assert failed["turn_id"] not in preview["recent_turn_ids"]
    assert cancelled["turn_id"] not in preview["recent_turn_ids"]
    assert len(store.list_objects("analysis")) == 1


def test_source_changed_while_waiting_never_falls_back_or_launches_inference(client):
    sid, _, _, first = first_review(client)
    paused = waiting(client, post(client, sid, message="Why this issue?", report=first["report_id"]))
    artifact = client.app.state.runtime.settings.data_dir / first["response"]["prediction"]["artifact_rel"]
    artifact.write_text("{}", encoding="utf-8")
    assert resume(client, paused, [4]).status_code == 202
    done = wait(client, paused["id"])
    assert done["status"] == "rejected" and done["error_code"] == "CONTEXT_SOURCE_UNAVAILABLE"
    assert done["response"]["classification_claims"] == []
    assert done["review_context"]["source_report_id"] == first["report_id"]
    assert done["review_context"]["result_execution_id"] is None
    assert client.get(f"/api/runs/{done['id']}/tools").json() == []


def test_submission_idempotency_freezes_one_snapshot(client):
    sid, _, _, first = first_review(client)
    headers = {"Idempotency-Key": str(uuid4())}
    one = post(client, sid, [4], report=first["report_id"], headers=headers)
    done = wait(client, one)
    saved = client.get(f"/api/turns/{done['turn_id']}").json()["context_snapshot"]
    assert post(client, sid, [4], report=first["report_id"], headers=headers) == one
    assert client.get(f"/api/turns/{done['turn_id']}").json()["context_snapshot"] == saved


def test_checkpoint_restart_keeps_binding_and_confirmation(tmp_path):
    config = settings(tmp_path)
    with TestClient(create_app(config)) as c:
        sid, _, _, first = first_review(c)
        paused = waiting(c, post(c, sid, message="That issue", report=first["report_id"]))
        frozen = c.get(f"/api/turns/{paused['turn_id']}").json()["context_snapshot"]
    with TestClient(create_app(config)) as c:
        again = waiting(c, paused["id"])
        assert again["wait"]["context_snapshot_id"] == paused["wait"]["context_snapshot_id"]
        assert c.get(f"/api/turns/{paused['turn_id']}").json()["context_snapshot"] == frozen
        assert resume(c, again, [4]).status_code == 202
        done = wait(c, again["id"])
        assert done["status"] == "completed" and done["response"]["reused_result"]
        assert done["review_context"]["source_report_id"] == first["report_id"]
        assert c.get(f"/api/runs/{done['id']}/tools").json() == []


def test_report_binding_owner_boundary(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as c:
        sid, _, _, first = first_review(c)
    with TestClient(create_app(settings(tmp_path, owner_id="other-user"))) as c:
        assert c.get(f"/api/sessions/{sid}/context").status_code == 404
        new = c.post("/api/sessions", json={}).json()["id"]
        assert (
            c.get(f"/api/sessions/{new}/context", params={"source_report_id": first["report_id"]}).status_code
            == 404
        )


def test_snapshot_tampering_is_rejected_before_any_tool(tmp_path, monkeypatch):
    original = ContextBinder.snapshot

    def changed(self, run, selection):
        result = original(self, run, selection)
        result.snapshot_id = "tampered"
        return result

    monkeypatch.setattr(ContextBinder, "snapshot", changed)
    with TestClient(create_app(settings(tmp_path))) as c:
        sid, _, _ = uploaded_session(c)
        done = wait(c, post(c, sid, [4], intent="analyse"))
        assert done["status"] == "failed"
        assert c.get(f"/api/runs/{done['id']}/tools").json() == []


def test_model_reference_uses_bound_scope_and_receipt_once(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if "findings" in json.loads(calls[-1]["messages"][1]["content"]):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps({"ordered_label_ids": [4], "lead": "evidence"})
                            },
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "goal": {"intent": "explain", "target_label_ids": [5]},
                                    "reference": {"scope": "single_finding", "source": "viewed_report"},
                                }
                            )
                        },
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "cpr_agent.runtime.create_model_gateway", lambda s: create_model_gateway(s, transport=transport)
    )
    with TestClient(
        create_app(settings(tmp_path, llm_mode="deepseek", deepseek_api_key="mock-key-only"))
    ) as c:
        sid, _, _, first = first_review(c, [4])
        done = wait(c, post(c, sid, message="Explain this issue", report=first["report_id"]))
        assert done["response"]["review"]["displayed_label_ids"] == [4]
        assert len(calls) == 2  # One goal parse and one bounded explanation plan, no duplicate parsing.
        assert done["grounded_review"]["mode"] == "model_organised_evidence"
        payload = json.loads(calls[0]["messages"][1]["content"])
        assert payload["context"]["active_review"]["label_ids"] == [4]
        encoded = json.dumps(payload)
        for secret in [
            sid,
            first["report_id"],
            first["response"]["prediction"]["video_sha256"],
            "mock-key-only",
        ]:
            assert secret not in encoded
        assert c.get(f"/api/runs/{done['id']}/tools").json() == []


@pytest.mark.parametrize(
    "reference",
    [
        {"scope": "single_finding", "source": "viewed_report"},
        {"scope": "ambiguous", "source": "current_clip"},
        {"scope": "explicit", "source": "ambiguous"},
    ],
)
def test_mock_model_ambiguity_enters_form_without_claiming_a_result(tmp_path, monkeypatch, reference):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps({"goal": {"intent": "explain"}, "reference": reference})
                        },
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "cpr_agent.runtime.create_model_gateway",
        lambda s: create_model_gateway(s, transport=httpx.MockTransport(handler)),
    )
    with TestClient(
        create_app(settings(tmp_path, llm_mode="deepseek", deepseek_api_key="mock-key-only"))
    ) as c:
        sid, _, _, first = first_review(c)
        paused = waiting(c, post(c, sid, message="What about that?", report=first["report_id"]))
        assert paused["response"] is None
        assert paused["wait"]["context_code"] in {"CONTEXT_SOURCE_AMBIGUOUS", "CONTEXT_SCOPE_AMBIGUOUS"}
        assert c.get(f"/api/runs/{paused['id']}/tools").json() == []
        c.post(f"/api/runs/{paused['id']}/cancel")
        wait(c, paused["id"])


def test_bound_gateway_rejects_missing_reference_and_filters_private_context():
    context = {
        "binding_version": "bound-v1",
        "clip_id": "private",
        "binding": {"private": "full-snapshot"},
        "active_review": {"label_ids": [4, 12], "source_available": True, "report_id": "private-report"},
    }
    assert request_context(context)["active_review"] == {"label_ids": [4], "source_available": True}
    gateway = create_model_gateway(
        Settings(llm_mode="deepseek", deepseek_api_key="mock-only"),
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": '{"goal":{"intent":"explain"}}'}}
                    ]
                },
            )
        ),
    )
    assert gateway.parse_goal("this", context).code == "LLM_INVALID_RESPONSE"
    assert "private" not in json.dumps(request_context(context))


def test_legacy_mode_rejects_new_context_fields_instead_of_ignoring_them(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as c:
        sid = c.post("/api/sessions", json={}).json()["id"]
        r = c.post(
            f"/api/sessions/{sid}/turns",
            json={"goal": {"intent": "summarise"}, "context": {}},
            headers={"Idempotency-Key": str(uuid4())},
        )
        assert r.status_code == 409
        assert c.get(f"/api/sessions/{sid}/context").status_code == 409


def test_preview_does_not_write_or_invoke_models(client):
    sid, _, _, _ = first_review(client)
    history = client.get(f"/api/sessions/{sid}/history").json()
    store = client.app.state.runtime.store
    count = len(store.list_objects("turn"))
    for _ in range(3):
        preview = client.get(f"/api/sessions/{sid}/context").json()
        assert preview["source_status"] == "available" and preview["run_id"] == "preview"
    assert len(store.list_objects("turn")) == count
    assert client.get(f"/api/sessions/{sid}/history").json() == history


def test_new_analysis_refuses_video_replaced_after_submission(client):
    sid, asset, _ = uploaded_session(client)
    paused = waiting(client, post(client, sid, message="Review this video"))
    store = client.app.state.runtime.store
    stored = store.get("asset", asset["id"])
    path = client.app.state.runtime.settings.data_dir / stored["storage_rel"]
    path.write_bytes(path.read_bytes() + b"changed-after-submission")
    store.put("asset", {**stored, "sha256": file_hash(path)})
    assert resume(client, paused, [4], intent="analyse").status_code == 202
    done = wait(client, paused["id"])
    assert done["status"] == "rejected" and done["error_code"] == "CONTEXT_VIDEO_CHANGED"
    assert client.get(f"/api/runs/{done['id']}/tools").json() == []


def test_bound_report_content_change_is_rejected_even_with_same_prediction(client):
    sid, _, _, first = first_review(client)
    paused = waiting(client, post(client, sid, message="Explain this", report=first["report_id"]))
    store = client.app.state.runtime.store
    report = store.get("report", first["report_id"])
    report["response"]["message"] = "Changed after the context was bound"
    store.put("report", report)
    assert resume(client, paused, [4]).status_code == 202
    done = wait(client, paused["id"])
    assert done["status"] == "rejected" and done["error_code"] == "CONTEXT_SOURCE_UNAVAILABLE"
    assert done["response"]["classification_claims"] == []
    assert client.get(f"/api/runs/{done['id']}/tools").json() == []


def test_invalid_latest_report_is_not_silently_replaced_by_an_older_one(client):
    sid, _, _, first = first_review(client)
    second = wait(client, post(client, sid, [4], report=first["report_id"]))
    store = client.app.state.runtime.store
    altered = store.get("report", second["report_id"])
    altered["response"]["message"] = "corrupted current report"
    store.put("report", altered)
    assert client.get(f"/api/sessions/{sid}/context").status_code == 409
    response = client.post(
        f"/api/sessions/{sid}/turns",
        json={"goal": {"intent": "explain", "target_label_ids": [4]}},
        headers={"Idempotency-Key": str(uuid4())},
    )
    assert response.status_code == 409
    explicit_old = wait(client, post(client, sid, [2], report=first["report_id"]))
    assert explicit_old["status"] == "completed"
    assert explicit_old["review_context"]["source_report_id"] == first["report_id"]


def test_start_check_refuses_incompatible_wait_without_mutating_it(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as c:
        sid = c.post("/api/sessions", json={}).json()["id"]
        rid = c.post(
            f"/api/sessions/{sid}/turns",
            json={"message": "wait for input"},
            headers={"Idempotency-Key": str(uuid4())},
        ).json()["run_id"]
        paused = waiting(c, rid)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/context_start_check.py"),
                "--data-dir",
                str(tmp_path),
                "--graph-version",
                "cpr-review-12-v0.5-context",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2 and "ACTIVE_GRAPH_VERSION_CONFLICT" in result.stdout
        assert c.get(f"/api/runs/{rid}").json()["status"] == paused["status"]
        c.post(f"/api/runs/{rid}/cancel")
        wait(c, rid)
