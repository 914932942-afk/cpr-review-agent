"""All DeepSeek requests here use MockTransport; no real API keys or network calls."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from cpr_agent.api import create_app
from cpr_agent.config import Settings
from cpr_agent.llm import create_model_gateway
from cpr_agent.models import TERMINAL
from cpr_agent.ports import DisabledModelGateway
from fastapi.testclient import TestClient

FAKE_KEY = "unit-test-key-never-sent-to-network"


@pytest.fixture(autouse=True)
def no_real_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real HTTP transport is prohibited in LLM tests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)


def configured(**kwargs):
    return Settings(llm_mode="deepseek", deepseek_api_key=FAKE_KEY, **kwargs)


def answer(goal=None, *, content=None, finish="stop", extra=None):
    message = {"content": content if content is not None else json.dumps({"goal": goal})}
    message.update(extra or {})
    return {"choices": [{"finish_reason": finish, "message": message}]}


def gateway_for(handler, **settings):
    return create_model_gateway(configured(**settings), transport=httpx.MockTransport(handler))


def test_disabled_with_key_never_constructs_http_client(monkeypatch):
    gateway = create_model_gateway(Settings(deepseek_api_key=FAKE_KEY))
    assert isinstance(gateway, DisabledModelGateway)
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: pytest.fail("Client constructed in disabled mode"))
    result = gateway.parse_goal("Analyse this clip", {"clip_id": "x"})
    assert result.code == "LLM_DISABLED" and not result.called and result.goal is None
    assert gateway.status()["network_enabled"] is False
    assert gateway.status()["key_configured"] is True
    assert FAKE_KEY not in repr(Settings(deepseek_api_key=FAKE_KEY))


def test_mocked_live_request_and_minimal_context():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=answer({"intent": "analyse", "target_label_ids": [4, 5]}))

    gateway = gateway_for(handler)
    assert not requests and gateway.status()["connection_verified"] is False
    result = gateway.parse_goal(
        "只看手臂问题",
        {
            "clip_id": "private-clip-id",
            "storage_rel": "private/path.mp4",
            "api_key": "private-other-key",
            "recent_turn_ids": ["private-turn"],
            "recent_goals": [{"intent": "analyse", "target_label_ids": [4]}],
            "preferences": [{"id": "focus", "label_ids": [4, 5], "source_turn_id": "private-source"}],
        },
    )
    assert result.code == "LLM_GOAL_PARSED" and result.goal.target_label_ids == [4, 5]
    assert len(requests) == 1 and requests[0].headers["Authorization"] == "Bearer " + FAKE_KEY
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://api.deepseek.com/chat/completions"
    assert body["thinking"] == {"type": "disabled"} and body["response_format"] == {"type": "json_object"}
    payload = json.loads(body["messages"][1]["content"])
    assert payload["context"]["explicit_focus"] == [4, 5]
    assert payload["context"]["recent_goals"][0]["target_label_ids"] == [4]
    assert "private-" not in requests[0].content.decode()
    assert FAKE_KEY not in json.dumps(result.receipt())
    assert gateway.status()["connection_verified"] is True


@pytest.mark.parametrize(
    "response,code",
    [
        (answer(), "LLM_NEEDS_CLARIFICATION"),
        (answer(content=" "), "LLM_EMPTY_RESPONSE"),
        (answer(content="not json"), "LLM_INVALID_RESPONSE"),
        (answer(content='{"goal":null,"goal":null}'), "LLM_INVALID_RESPONSE"),
        (answer(content='{"goal":null,"findings":["correct CPR"]}'), "LLM_INVALID_RESPONSE"),
        (answer({"intent": "analyse", "target_label_ids": [10]}), "LLM_INVALID_RESPONSE"),
        (answer({"intent": "analyse", "target_label_ids": ["4"]}), "LLM_INVALID_RESPONSE"),
        (answer({"intent": "analyse", "target_label_ids": []}), "LLM_INVALID_RESPONSE"),
        (answer({"intent": "delete_video"}), "LLM_INVALID_RESPONSE"),
        (answer({"intent": "analyse", "target_label_ids": [4], "score": 0.99}), "LLM_INVALID_RESPONSE"),
        (answer({"intent": "analyse", "target_label_ids": [4]}, finish="length"), "LLM_INCOMPLETE_RESPONSE"),
        (answer(extra={"tool_calls": [{"function": {"name": "unregistered"}}]}), "LLM_INCOMPLETE_RESPONSE"),
        (
            answer({"intent": "manage_memory", "memory_commands": [{"operation": "forget_preference"}]}),
            "LLM_MEMORY_CONFIRMATION_REQUIRED",
        ),
        (
            answer(
                {
                    "intent": "analyse",
                    "target_label_ids": [4],
                    "memory_commands": [{"operation": "remember_focus", "label_ids": [4]}],
                }
            ),
            "LLM_MEMORY_CONFIRMATION_REQUIRED",
        ),
        ({"choices": []}, "LLM_INVALID_RESPONSE"),
    ],
)
def test_invalid_or_uncertain_outputs_return_form(response, code):
    gateway = gateway_for(lambda request: httpx.Response(200, json=response))
    result = gateway.parse_goal("Please review", {})
    assert result.goal is None and result.code == code and result.called


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "LLM_AUTH_FAILED"),
        (402, "LLM_BALANCE_UNAVAILABLE"),
        (403, "LLM_ACCESS_DENIED"),
        (404, "LLM_MODEL_OR_ENDPOINT_UNAVAILABLE"),
        (429, "LLM_RATE_LIMITED"),
        (500, "LLM_HTTP_ERROR"),
        (302, "LLM_HTTP_ERROR"),
    ],
)
def test_provider_errors_do_not_expose_body_or_retry(status, code):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status, text=FAKE_KEY, headers={"Location": "https://untrusted.invalid"})

    result = gateway_for(handler).parse_goal("review", {})
    assert result.code == code and result.goal is None and len(calls) == 1
    assert FAKE_KEY not in json.dumps(result.receipt())


def test_timeout_is_sanitized_and_large_response_rejected():
    def timeout(request):
        raise httpx.ReadTimeout("Sensitive provider error " + FAKE_KEY, request=request)

    assert gateway_for(timeout).parse_goal("review", {}).code == "LLM_TIMEOUT"
    assert (
        gateway_for(lambda r: httpx.Response(200, content=b"x" * 65537)).parse_goal("review", {}).code
        == "LLM_RESPONSE_TOO_LARGE"
    )


def test_missing_key_never_sends_request():
    gateway = create_model_gateway(Settings(llm_mode="deepseek"))
    assert gateway.parse_goal("review", {}).code == "LLM_KEY_MISSING"
    assert not gateway.status()["key_configured"]


def test_circuit_breaker_stops_calls_and_recovers(monkeypatch):
    clock, calls = [0.0], []
    monkeypatch.setattr("cpr_agent.llm.time", SimpleNamespace(monotonic=lambda: clock[0]))

    def handler(request):
        calls.append(1)
        return (
            httpx.Response(500)
            if len(calls) <= 3
            else httpx.Response(200, json=answer({"intent": "summarise"}))
        )

    gateway = gateway_for(handler)
    for _ in range(3):
        assert gateway.parse_goal("summary", {}).code == "LLM_HTTP_ERROR"
    assert gateway.status()["circuit_open"]
    assert gateway.parse_goal("summary", {}).code == "LLM_CIRCUIT_OPEN" and len(calls) == 3
    clock[0] = 61
    assert gateway.parse_goal("summary", {}).code == "LLM_GOAL_PARSED" and len(calls) == 4
    assert not gateway.status()["circuit_open"]


@pytest.mark.parametrize(
    "url",
    [
        "http://api.deepseek.com",
        "https://evil.example",
        "https://api.deepseek.com@evil.example",
        "https://api.deepseek.com/?key=secret",
        "https://api.deepseek.com/other",
    ],
)
def test_disallow_credential_forwarding_to_other_endpoints(url):
    with pytest.raises(ValueError):
        configured(deepseek_base_url=url)


def test_dotenv_loading_and_explicit_switch(tmp_path, monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith(("CPR_", "DEEPSEEK_")):
            monkeypatch.delenv(name)
    path = tmp_path / ".env"
    path.write_text('DEEPSEEK_API_KEY="' + FAKE_KEY + '"\nDEEPSEEK_MODEL=deepseek-flash\n', encoding="utf-8")
    settings = Settings.from_env(path)
    assert settings.llm_mode == "disabled" and settings.deepseek_api_key == FAKE_KEY
    path.write_text(path.read_text() + "CPR_LLM_MODE=deepseek\n", encoding="utf-8")
    assert Settings.from_env(path).llm_mode == "deepseek"
    monkeypatch.setenv("CPR_LLM_MODE", "disabled")
    assert Settings.from_env(path).llm_mode == "disabled"


def wait(client, rid, predicate=None):
    until = time.monotonic() + 8
    while time.monotonic() < until:
        run = client.get(f"/api/runs/{rid}").json()
        if predicate(run) if predicate else run["status"] in TERMINAL:
            return run
        time.sleep(0.02)
    pytest.fail(str(run))


def post_turn(client, sid, body, headers=None):
    response = client.post(
        f"/api/sessions/{sid}/turns", json=body, headers=headers or {"Idempotency-Key": str(uuid4())}
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def wire_mock(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "cpr_agent.runtime.create_model_gateway",
        lambda settings: create_model_gateway(settings, transport=transport),
    )


def test_mock_live_graph_parses_and_persists_goal_once(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=answer({"intent": "summarise", "target_label_ids": [4]}))

    wire_mock(monkeypatch, handler)
    with TestClient(create_app(configured(data_dir=tmp_path))) as client:
        assert client.get("/api/health").json()["llm_details"]["connection_verified"] is False
        assert not calls  # Neither startup nor health probes the real provider.
        sid = client.post("/api/sessions", json={}).json()["id"]
        headers, body = {"Idempotency-Key": str(uuid4())}, {"message": "总结一下"}
        rid = post_turn(client, sid, body, headers)
        assert post_turn(client, sid, body, headers) == rid
        run = wait(client, rid)
        assert run["status"] == "completed" and len(calls) == 1
        turn = client.get(f"/api/turns/{run['turn_id']}").json()
        assert turn["resolved_goal"]["intent"] == "summarise"
        assert turn["llm_resolution"]["code"] == "LLM_GOAL_PARSED"
        events = client.get(f"/api/runs/{rid}/events").json()["events"]
        assert "N03" not in {e["node_id"] for e in events}
        assert FAKE_KEY not in json.dumps([turn, run, events, client.get("/api/health").json()])
        second = wait(client, post_turn(client, sid, {"message": "仍然是刚才的类别"}))
        assert second["status"] == "completed" and len(calls) == 2
        context = json.loads(calls[1]["messages"][1]["content"])["context"]
        assert context["recent_goals"][0]["target_label_ids"] == [4]
    assert FAKE_KEY not in (tmp_path / "logs/runtime.jsonl").read_text(encoding="utf-8")


def test_mock_live_failure_resumes_form_without_another_call(tmp_path, monkeypatch):
    calls = []
    wire_mock(monkeypatch, lambda request: calls.append(1) or httpx.Response(401, text=FAKE_KEY))
    with TestClient(create_app(configured(data_dir=tmp_path))) as client:
        sid = client.post("/api/sessions", json={}).json()["id"]
        rid = post_turn(client, sid, {"message": "总结一下"})
        run = wait(
            client, rid, lambda r: r["status"] == "waiting_user" and (r["wait"] or {}).get("interrupt_id")
        )
        assert (
            client.post(
                f"/api/runs/{rid}/resume",
                headers={"Idempotency-Key": str(uuid4())},
                json={
                    "kind": "user_answer",
                    "revision": run["revision"],
                    "wait_id": run["wait"]["wait_id"],
                    "interrupt_id": run["wait"]["interrupt_id"],
                    "goal": {"intent": "summarise"},
                },
            ).status_code
            == 202
        )
        done = wait(client, rid)
        assert done["status"] == "completed" and len(calls) == 1
        assert client.get("/api/preferences").json() == []
        assert FAKE_KEY not in json.dumps(client.get(f"/api/runs/{rid}/events").json())


def test_structured_goal_bypasses_enabled_model(tmp_path, monkeypatch):
    wire_mock(monkeypatch, lambda request: pytest.fail("Structured goal must not call model"))
    with TestClient(create_app(configured(data_dir=tmp_path))) as client:
        sid = client.post("/api/sessions", json={}).json()["id"]
        run = wait(client, post_turn(client, sid, {"goal": {"intent": "summarise"}}))
        assert run["status"] == "completed"


def test_disabled_mode_with_key_runs_form_without_network(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, deepseek_api_key=FAKE_KEY))) as client:
        health = client.get("/api/health").json()
        assert health["llm"] == "disabled" and not health["llm_details"]["network_enabled"]
        assert health["llm_details"]["key_configured"]
        sid = client.post("/api/sessions", json={}).json()["id"]
        rid = post_turn(client, sid, {"message": "总结一下"})
        run = wait(client, rid, lambda r: r["status"] == "waiting_user")
        turn = client.get(f"/api/turns/{run['turn_id']}").json()
        assert turn["llm_resolution"]["code"] == "LLM_DISABLED" and not turn["llm_resolution"]["called"]
        client.post(f"/api/runs/{rid}/cancel")
        assert wait(client, rid)["status"] == "cancelled"


def test_cancellation_during_parsing_discards_result(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def handler(request):
        entered.set()
        assert release.wait(3)
        return httpx.Response(200, json=answer({"intent": "analyse", "target_label_ids": [4]}))

    wire_mock(monkeypatch, handler)
    with TestClient(create_app(configured(data_dir=tmp_path))) as client:
        sid = client.post("/api/demo/session").json()["session"]["id"]
        rid = post_turn(client, sid, {"message": "analyse"})
        assert entered.wait(3)
        client.post(f"/api/runs/{rid}/cancel").raise_for_status()
        release.set()
        run = wait(client, rid)
        assert run["status"] == "cancelled" and client.get(f"/api/runs/{rid}/tools").json() == []
