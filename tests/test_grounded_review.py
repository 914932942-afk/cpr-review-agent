"""Grounding, fallback and review continuity. CV is a labelled test double, LLM is MockTransport."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest
from cpr_agent.api import create_app
from cpr_agent.context_graph import ContextReviewGraph
from cpr_agent.explanation_model import validate_plan
from cpr_agent.grounded_review import GLOSSARY, build_grounded_review
from cpr_agent.llm import create_model_gateway
from cpr_agent.models import CANDIDATE_LABELS
from cpr_agent.vision import RealVisionTool
from fastapi.testclient import TestClient
from test_context_binding import first_review, post, settings
from test_real_video import make_prediction, wait


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    monkeypatch.setattr(
        httpx.HTTPTransport, "handle_request", lambda *a, **k: pytest.fail("Real HTTP forbidden")
    )
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as c:
        yield c


def explanation(c, run):
    response = c.get(f"/api/reports/{run['report_id']}/explanation")
    assert response.status_code == 200
    return response.json()


def mock_gateway(monkeypatch, value=None, *, handler=None):
    calls = []

    def answer(request):
        calls.append(json.loads(request.content))
        if handler:
            return handler(request)
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}
        )

    monkeypatch.setattr(
        "cpr_agent.runtime.create_model_gateway",
        lambda s: create_model_gateway(s, transport=httpx.MockTransport(answer)),
    )
    return calls


def test_offline_explanation_has_exact_evidence_scope_and_does_not_reinterpret_score(client):
    sid, _, _, first = first_review(client)
    run = wait(client, post(client, sid, [2, 4], report=first["report_id"]))
    result = explanation(client, run)
    assert result["status"] == "available"
    review = result["review"]
    assert review["scope_label_ids"] == [2, 4] and review["flagged_label_ids"] == [2]
    assert review["mode"] == "evidence_template"
    assert review["organisation_receipt"]["code"] == "LLM_DISABLED"
    assert not review["organisation_receipt"]["called"]
    assert review["entries"][0]["name"] == "Clenched hands"
    assert review["entries"][0]["uncalibrated_score"] == 0.9
    assert "does not confirm correct" in review["entries"][1]["recorded_result"]
    assert review["source_execution_id"] == first["response"]["prediction"]["execution_id"]
    assert run["response"]["prediction"] == first["response"]["prediction"]
    assert client.get(f"/api/runs/{run['id']}/tools").json() == []
    report = client.get(f"/api/reports/{run['report_id']}").json()
    assert report["grounded_review"] == run["grounded_review"] == review
    exported = client.get(f"/api/reports/{run['report_id']}/export").text
    assert "Evidence-constrained review" in exported and "0.9000" in exported
    assert "Error onset times" in exported


def test_scope_summary_and_full_summary_are_distinct_and_saved(client):
    sid, _, _, first = first_review(client)
    narrow = wait(client, post(client, sid, [4], report=first["report_id"], intent="summarise"))
    assert explanation(client, narrow)["review"]["scope_label_ids"] == [4]
    assert len(explanation(client, narrow)["review"]["outside_scope_label_ids"]) == 10
    full = wait(client, post(client, sid, [], report=narrow["report_id"], intent="summarise"))
    assert explanation(client, full)["review"]["scope_label_ids"] == list(CANDIDATE_LABELS)
    history = client.get(f"/api/sessions/{sid}/history").json()
    assert history["latest_recap"]["run_id"] == full["id"]
    assert history["latest_recap"]["scope_label_ids"] == list(CANDIDATE_LABELS)
    reopened = wait(client, post(client, sid, [], report=narrow["report_id"], scope="previous_scope"))
    assert reopened["grounded_review"]["scope_label_ids"] == [4]
    assert reopened["review_context"]["source_report_id"] == narrow["report_id"]
    assert client.get(f"/api/sessions/{sid}/history").json()["latest_recap"]["run_id"] == full["id"]


@pytest.mark.parametrize(
    "value",
    [
        {"ordered_label_ids": [2], "lead": "evidence"},
        {"ordered_label_ids": [2, 4, 4], "lead": "evidence"},
        {"ordered_label_ids": [2, 99], "lead": "findings"},
        {"ordered_label_ids": [2, 4], "lead": "diagnosis"},
        {"ordered_label_ids": [2, 4], "lead": "findings", "explanation": "You have a medical condition"},
        {"ordered_label_ids": ["2", 4], "lead": "findings"},
    ],
)
def test_invalid_organisation_cannot_change_scope_or_add_prose(value):
    with pytest.raises(ValueError):
        validate_plan(value, [2, 4])


def test_mock_model_only_organises_server_facts_and_is_idempotent(tmp_path, monkeypatch):
    calls = mock_gateway(monkeypatch, {"ordered_label_ids": [4, 2], "lead": "evidence"})
    with TestClient(create_app(settings(tmp_path, llm_mode="deepseek", deepseek_api_key="mock-secret"))) as c:
        sid, _, _, first = first_review(c)
        headers = {"Idempotency-Key": str(uuid4())}
        rid = post(c, sid, [2, 4], report=first["report_id"], headers=headers)
        run = wait(c, rid)
        assert post(c, sid, [2, 4], report=first["report_id"], headers=headers) == rid
        assert len(calls) == 1
        result = explanation(c, run)["review"]
        assert result["mode"] == "model_organised_evidence"
        assert [e["label_id"] for e in result["entries"]] == [4, 2]
        assert result["flagged_label_ids"] == [2]
        payload = json.loads(calls[0]["messages"][1]["content"])
        assert set(payload) == {"intent", "findings", "schema"}
        assert set(payload["findings"][0]) == {"label_id", "state"}
        for secret in [
            sid,
            first["report_id"],
            "mock-secret",
            "video_sha256",
            "uncalibrated_score",
            "storage_rel",
        ]:
            assert secret not in json.dumps(payload)
        assert c.get(f"/api/runs/{rid}/tools").json() == []
        assert c.get(f"/api/reports/{run['report_id']}/export").status_code == 200
        assert len(calls) == 1  # Reopen/export never call the provider.


@pytest.mark.parametrize("case", ["invalid", "timeout", "rate_limit"])
def test_provider_failure_falls_back_without_losing_the_cv_result(tmp_path, monkeypatch, case):
    def handler(request):
        if case == "timeout":
            raise httpx.ReadTimeout("do not log provider text")
        if case == "rate_limit":
            return httpx.Response(429)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"ordered_label_ids":[2,4],"lead":"findings","claim":"at 3 seconds"}'
                        },
                    }
                ]
            },
        )

    calls = mock_gateway(monkeypatch, handler=handler)
    with TestClient(create_app(settings(tmp_path, llm_mode="deepseek", deepseek_api_key="mock-secret"))) as c:
        sid, _, _, first = first_review(c)
        run = wait(c, post(c, sid, [2, 4], report=first["report_id"]))
        assert run["status"] == "completed" and run["error_code"] is None
        assert run["response"]["prediction"] == first["response"]["prediction"]
        review = explanation(c, run)["review"]
        assert review["mode"] == "evidence_template" and review["organisation_receipt"]["called"]
        assert len(calls) == 1
        assert "at 3 seconds" not in json.dumps(review)


def test_source_change_hides_explanation_and_export_does_not_claim_revalidation(client):
    _, _, _, run = first_review(client)
    artifact = client.app.state.runtime.settings.data_dir / run["response"]["prediction"]["artifact_rel"]
    artifact.write_text("{}")
    value = explanation(client, run)
    assert value["status"] == "unavailable" and value["review"] is None
    exported = client.get(f"/api/reports/{run['report_id']}/export").text
    assert "saved explanation could not be revalidated" in exported
    assert "Source files and bindings verified" not in exported


def test_changed_saved_explanation_is_not_rendered(client):
    _, _, _, run = first_review(client)
    store = client.app.state.runtime.store
    report = store.get("report", run["report_id"])
    report["grounded_review"]["entries"][0]["recorded_result"] = "invented medical certainty"
    store.put("report", report)
    assert explanation(client, run)["error_code"] == "EXPLANATION_SOURCE_MISMATCH"
    # Even if both metadata copies are changed, deterministic reconstruction rejects it.
    # Test-only corruption bypasses the normal API, which correctly freezes terminal runs.
    with store.tx() as db:
        saved_run = store._run(db, run["id"])
        saved_run["grounded_review"] = report["grounded_review"]
        store._write_run(db, saved_run)
    assert explanation(client, run)["error_code"] == "EXPLANATION_RECORD_INVALID"


def test_grounding_rejects_claim_wording_and_score_drift(client):
    _, _, _, run = first_review(client)
    response = run["response"]
    response["classification_claims"][0]["uncalibrated_score"] = 0.123
    with pytest.raises(ValueError):
        build_grounded_review(response, "explain", {"plan": None})
    assert set(GLOSSARY) == set(CANDIDATE_LABELS)


def test_interrupted_dispatch_falls_back_instead_of_requesting_again(tmp_path, monkeypatch):
    calls = mock_gateway(monkeypatch, {"ordered_label_ids": [2, 4], "lead": "evidence"})
    original = ContextReviewGraph.n11

    def interrupted(self, state):
        if (state.get("goal") or {}).get("intent") == "explain":
            turn = self.store.get("turn", self.store.get_run(state["run_id"])["turn_id"])
            self.store.put("turn", {**turn, "explanation_attempted": True})
        return original(self, state)

    monkeypatch.setattr(ContextReviewGraph, "n11", interrupted)
    with TestClient(create_app(settings(tmp_path, llm_mode="deepseek", deepseek_api_key="mock-secret"))) as c:
        sid, _, _, first = first_review(c)
        run = wait(c, post(c, sid, [2, 4], report=first["report_id"]))
        assert run["status"] == "completed" and calls == []
        assert run["grounded_review"]["organisation_receipt"]["code"] == "LLM_PLAN_INTERRUPTED"


def test_summary_and_explanation_survive_restart_without_memory_leak(tmp_path):
    configured = settings(tmp_path)
    with TestClient(create_app(configured)) as c:
        sid, _, _, first = first_review(c)
        run = wait(c, post(c, sid, [4], intent="summarise", report=first["report_id"]))
    with TestClient(create_app(configured)) as c:
        assert explanation(c, run)["review"] == run["grounded_review"]
        wait(c, post(c, sid, [2], report=run["report_id"]))
        # The summary pointer is independent of the requested history page size.
        assert c.get(f"/api/sessions/{sid}/history?limit=1").json()["latest_recap"]["run_id"] == run["id"]
        assert c.get("/api/preferences").json() == []  # Summaries never silently become preferences.
        latest = wait(c, post(c, sid, [], report=run["report_id"], scope="previous_scope"))
        assert latest["grounded_review"]["scope_label_ids"] == [4]
    with TestClient(create_app(replace(configured, owner_id="another-owner"))) as c:
        assert c.get(f"/api/sessions/{sid}/history").status_code == 404
        assert c.get(f"/api/reports/{run['report_id']}/explanation").status_code == 404


def test_no_saved_prediction_never_creates_explanation(client):
    from test_real_video import uploaded_session

    sid, _, _ = uploaded_session(client)
    run = wait(client, post(client, sid, [4]))
    assert not run.get("grounded_review")
    assert explanation(client, run)["status"] == "not_available"


def test_explanation_circuit_stops_further_requests():
    calls = []

    def bad(request):
        calls.append(1)
        return httpx.Response(429)

    gateway = create_model_gateway(
        settings(__import__("pathlib").Path.cwd(), llm_mode="deepseek", deepseek_api_key="mock-secret"),
        transport=httpx.MockTransport(bad),
    )
    facts = {"intent": "explain", "findings": [{"label_id": 4, "state": "model_flagged"}]}
    for _ in range(3):
        assert gateway.plan_explanation(facts).called
    blocked = gateway.plan_explanation(facts)
    assert blocked.code == "LLM_CIRCUIT_OPEN" and not blocked.called
    assert len(calls) == 3


def test_cancel_during_explanation_does_not_publish_report_or_recap(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def handler(request):
        entered.set()
        assert release.wait(5)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"ordered_label_ids":[4],"lead":"findings"}'},
                    }
                ]
            },
        )

    calls = mock_gateway(monkeypatch, handler=handler)
    with TestClient(create_app(settings(tmp_path, llm_mode="deepseek", deepseek_api_key="mock-secret"))) as c:
        sid, _, _, first = first_review(c)
        rid = post(c, sid, [4], intent="summarise", report=first["report_id"])
        try:
            assert entered.wait(5)
            assert c.post(f"/api/runs/{rid}/cancel").status_code == 202
        finally:
            release.set()
        run = wait(c, rid)
        assert run["status"] == "cancelled" and run["response"] is None and run["report_id"] is None
        assert not run.get("grounded_review")
        assert c.get(f"/api/sessions/{sid}/history").json()["latest_recap"] is None
        assert len(calls) == 1
