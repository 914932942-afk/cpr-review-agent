"""Cross-practice scope isolation; CV uses isolated test fixtures, not accuracy claims."""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from cpr_agent.api import create_app
from cpr_agent.context_contracts import ContextSnapshot, GoalReference
from cpr_agent.models import ReviewGoal
from cpr_agent.review_context import ContextBinder
from cpr_agent.vision import RealVisionTool
from fastapi.testclient import TestClient
from test_context_binding import first_review, post, settings
from test_real_video import make_prediction, uploaded_session, wait


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)
    monkeypatch.setattr(
        httpx.HTTPTransport, "handle_request", lambda *a, **k: pytest.fail("Network forbidden")
    )
    with TestClient(create_app(settings(tmp_path))) as c:
        yield c


def prepared(c):
    sid, asset, clip, first = first_review(c, [4])
    recap = wait(c, post(c, sid, [4], intent="summarise", report=first["report_id"]))
    new_sid, _, _ = uploaded_session(c)
    return sid, asset, clip, first, recap, new_sid


def view(c, sid):
    response = c.get(f"/api/sessions/{sid}/continuity")
    assert response.status_code == 200, response.text
    return response.json()


def choose(c, sid, report, revision=0, key=None):
    return c.post(
        f"/api/sessions/{sid}/continuity",
        json={"source_report_id": report, "revision": revision},
        headers={"Idempotency-Key": key or str(uuid4())},
    )


def test_candidate_and_preview_never_change_current_context(client):
    sid, _, _, first, recap, new_sid = prepared(client)
    data = view(client, new_sid)
    assert data["selection"]["status"] == "not_selected"
    assert data["candidates"][0]["report_id"] == recap["report_id"]
    assert data["candidates"][0]["same_video_file"] is True
    preview = client.get(f"/api/sessions/{new_sid}/continuity/recaps/{recap['report_id']}").json()
    assert preview["belongs_to_previous_video"] is True and preview["session_id"] == sid
    assert preview["label_ids"] == [4] and "uncalibrated_score" not in preview["recap"]["entries"][0]
    context = client.get(f"/api/sessions/{new_sid}/context").json()
    assert context["source"] is None and context["active_label_ids"] == []
    assert client.get(f"/api/runs/{first['id']}").json()["response"] == first["response"]


def test_explicit_scope_is_not_a_prediction_and_new_video_is_analysed_independently(client):
    _, _, _, first, recap, new_sid = prepared(client)
    assert choose(client, new_sid, recap["report_id"]).status_code == 200
    context = client.get(f"/api/sessions/{new_sid}/context").json()
    assert context["source"] is None and context["source_status"] == "none"
    assert context["active_label_ids"] == [4] and context["recent_goals"] == []
    goal, reason = ContextBinder.resolve(
        ContextSnapshot.model_validate(context),
        ReviewGoal(intent="analyse", target_label_ids=[5]),
        GoalReference(scope="previous_scope"),
    )
    assert reason is None and goal.target_label_ids == [4]
    analysed = wait(client, post(client, new_sid, [4], intent="analyse"))
    assert analysed["status"] == "completed"
    prediction = analysed["response"]["prediction"]
    assert prediction["clip_id"] == context["clip_id"] != first["clip_id"]
    assert prediction["execution_id"] != first["response"]["prediction"]["execution_id"]
    assert len(client.get(f"/api/runs/{analysed['id']}/tools").json()) == 1
    assert view(client, new_sid)["locked"] is True


def test_idempotent_apply_stale_revision_and_clear_do_not_write_preferences(client):
    *_, recap, new_sid = prepared(client)
    key = str(uuid4())
    applied = choose(client, new_sid, recap["report_id"], key=key)
    assert applied.status_code == 200
    assert choose(client, new_sid, recap["report_id"], key=key).json() == applied.json()
    assert view(client, new_sid)["revision"] == 1
    assert choose(client, new_sid, None, key=key).status_code == 409
    assert choose(client, new_sid, None).status_code == 409
    assert choose(client, new_sid, None, revision=1).status_code == 200
    assert view(client, new_sid)["selection"]["status"] == "not_selected"
    assert client.get(f"/api/sessions/{new_sid}/context").json()["active_label_ids"] == []
    assert client.get("/api/preferences").json() == []


@pytest.mark.parametrize("kind", ["not_recap", "same_session", "future", "bundle", "owner"])
def test_ineligible_sources_cannot_be_carried_or_previewed(client, kind):
    sid, _, _, first, recap, new_sid = prepared(client)
    report_id = first["report_id"] if kind == "not_recap" else recap["report_id"]
    store = client.app.state.runtime.store
    if kind == "same_session":
        new_sid = sid
    if kind == "future":
        target = store.get("session", new_sid)
        store.put("session", {**target, "created_at": 0})
    if kind == "bundle":
        previous = store.get("session", sid)
        store.put("session", {**previous, "bundle_id": "other"})
    if kind == "owner":
        with store.tx() as db:
            db.execute("UPDATE objects SET owner='someone-else' WHERE id=?", (report_id,))
    assert client.get(f"/api/sessions/{new_sid}/continuity/recaps/{report_id}").status_code == 410
    assert choose(client, new_sid, report_id).status_code in {409, 410}
    assert report_id not in [p["report_id"] for p in view(client, new_sid)["candidates"]]


@pytest.mark.parametrize("kind", ["media", "recap", "focus_record"])
def test_invalidated_sources_or_bindings_fail_closed(client, kind):
    _, asset, _, _, recap, new_sid = prepared(client)
    assert choose(client, new_sid, recap["report_id"]).status_code == 200
    store, cfg = client.app.state.runtime.store, client.app.state.runtime.settings
    if kind == "media":
        (cfg.data_dir / asset["storage_rel"]).unlink()
    elif kind == "recap":
        record = store.get("report", recap["report_id"])
        store.put(
            "report",
            {**record, "grounded_review": {**record["grounded_review"], "summary": "invented improvement"}},
        )
    else:
        record = store.get("practice_focus", "practice-focus:" + new_sid)
        store.put("practice_focus", {**record, "label_ids": [5]})
    assert view(client, new_sid)["selection"]["status"] == "unavailable"
    context = client.get(f"/api/sessions/{new_sid}/context").json()
    assert context["source"] is None and context["active_label_ids"] == []


def test_current_report_scope_wins_and_starting_goal_cannot_rewrite_analysis(client):
    *_, recap, new_sid = prepared(client)
    assert choose(client, new_sid, recap["report_id"]).status_code == 200
    current = wait(client, post(client, new_sid, [5], intent="analyse"))
    context = client.get(f"/api/sessions/{new_sid}/context").json()
    assert context["source"]["report_id"] == current["report_id"]
    assert context["active_label_ids"] == [5]
    assert choose(client, new_sid, None, revision=1).status_code == 409
    assert view(client, new_sid)["selection"]["label_ids"] == [4]


def test_active_turn_blocks_scope_mutation(client):
    *_, recap, new_sid = prepared(client)
    rid = post(client, new_sid, message="Please confirm a goal")
    run = wait(client, rid, lambda r: r["status"] == "waiting_user")
    assert choose(client, new_sid, recap["report_id"]).status_code == 409
    client.post(f"/api/runs/{run['id']}/cancel")
    assert wait(client, rid)["status"] == "cancelled"


def test_selection_restores_after_app_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)
    cfg = settings(tmp_path)
    with TestClient(create_app(cfg)) as c:
        *_, recap, sid = prepared(c)
        assert choose(c, sid, recap["report_id"]).status_code == 200
    with TestClient(create_app(cfg)) as c:
        assert view(c, sid)["selection"]["label_ids"] == [4]
        context = c.get(f"/api/sessions/{sid}/context").json()
        assert context["source"] is None and context["active_label_ids"] == [4]


def test_new_recap_of_an_earlier_practice_is_still_eligible(client):
    sid, _, _, first, _, new_sid = prepared(client)
    later_recap = wait(client, post(client, sid, [5], intent="summarise", report=first["report_id"]))
    assert view(client, new_sid)["candidates"][0]["report_id"] == later_recap["report_id"]
    assert choose(client, new_sid, later_recap["report_id"]).status_code == 200
    assert client.get(f"/api/sessions/{new_sid}/context").json()["active_label_ids"] == [5]


@pytest.mark.parametrize(
    "payload", [{"revision": -1}, {"revision": 0, "label_ids": [5]}, {"source_report_id": "x"}]
)
def test_focus_endpoint_requires_a_valid_explicit_choice(client, payload):
    sid, _, _ = uploaded_session(client)
    result = client.post(
        f"/api/sessions/{sid}/continuity", json=payload, headers={"Idempotency-Key": str(uuid4())}
    )
    assert result.status_code == 422
