"""User notes remain annotations; only confirmed categories become future goals."""

from uuid import uuid4

import pytest
from cpr_agent.api import create_app
from cpr_agent.vision import RealVisionTool
from fastapi.testclient import TestClient
from test_context_binding import settings
from test_real_video import make_prediction, uploaded_session


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)
    with TestClient(create_app(settings(tmp_path))) as c:
        yield c


def save(c, sid, revision=0, key=None, **extra):
    return c.post(
        f"/api/sessions/{sid}/notebook",
        headers={"Idempotency-Key": key or str(uuid4())},
        json={
            "revision": revision,
            "note": "Private reflection <script>plain text</script>",
            "items": [{"label_id": 4, "status": "next_focus"}, {"label_id": 5, "status": "reviewed"}],
            **extra,
        },
    )


def adopt(c, sid, old, revision=0, source_revision=1, key=None):
    return c.post(
        f"/api/sessions/{sid}/continuity",
        headers={"Idempotency-Key": key or str(uuid4())},
        json={
            "revision": revision,
            "source_notebook_session_id": old,
            "source_notebook_revision": source_revision,
        },
    )


def test_notes_revision_idempotency_and_session_isolation(client):
    sid, _, _ = uploaded_session(client)
    other, _, _ = uploaded_session(client)
    assert client.get(f"/api/sessions/{sid}/notebook").json()["revision"] == 0
    key = str(uuid4())
    first = save(client, sid, key=key)
    assert first.status_code == 200
    assert save(client, sid, key=key).json() == first.json()
    assert save(client, sid, key=key, note="Different").status_code == 409
    assert save(client, sid).status_code == 409
    assert client.get(f"/api/sessions/{other}/notebook").json()["note"] == ""
    assert client.get(f"/api/sessions/{sid}/context").json()["active_label_ids"] == []
    assert client.get("/api/preferences").json() == []
    assert client.app.state.runtime.store.active_runs() == []
    assert save(client, sid, revision=1, note="", items=[]).json()["revision"] == 2


def test_only_explicitly_adopted_focus_reaches_new_context_and_edits_invalidate_it(client):
    sid, _, _ = uploaded_session(client)
    assert save(client, sid).status_code == 200
    target, _, _ = uploaded_session(client)
    view = client.get(f"/api/sessions/{target}/continuity").json()
    assert view["selection"]["status"] == "not_selected"
    assert view["notebook_candidates"][0]["label_ids"] == [4]
    assert "note" not in view["notebook_candidates"][0]
    assert client.get(f"/api/sessions/{target}/context").json()["active_label_ids"] == []
    key = str(uuid4())
    result = adopt(client, target, sid, key=key)
    assert result.status_code == 200
    assert adopt(client, target, sid, key=key).json() == result.json()
    context = client.get(f"/api/sessions/{target}/context").json()
    assert context["source"] is None and context["active_label_ids"] == [4]
    assert context["recent_goals"] == [] and "Private reflection" not in str(context)
    assert save(client, sid, revision=1, note="Edited").status_code == 200
    assert client.get(f"/api/sessions/{target}/continuity").json()["selection"]["status"] == "unavailable"
    assert client.get(f"/api/sessions/{target}/context").json()["active_label_ids"] == []
    assert adopt(client, target, sid, revision=1).status_code == 409
    assert adopt(client, target, sid, revision=1, source_revision=2).status_code == 200


@pytest.mark.parametrize("kind", ["same", "future", "owner", "bundle", "tamper", "empty"])
def test_ineligible_or_changed_goal_sources_are_rejected(client, kind):
    sid, _, _ = uploaded_session(client)
    save(client, sid, items=[] if kind == "empty" else [{"label_id": 4, "status": "next_focus"}])
    target, _, _ = uploaded_session(client)
    store = client.app.state.runtime.store
    if kind == "same":
        target = sid
    elif kind == "future":
        session = store.get("session", target)
        store.put("session", {**session, "created_at": 0})
    elif kind == "bundle":
        session = store.get("session", sid)
        store.put("session", {**session, "bundle_id": "different"})
    elif kind == "owner":
        with store.tx() as db:
            db.execute("UPDATE objects SET owner='other' WHERE id=?", ("notebook:" + sid,))
    elif kind == "tamper":
        record = store.get("review_notebook", "notebook:" + sid)
        store.put("review_notebook", {**record, "note": "Changed behind the API"})
    assert adopt(client, target, sid).status_code in {409, 410}
    assert client.get(f"/api/sessions/{target}/context").json()["active_label_ids"] == []


@pytest.mark.parametrize(
    "extra",
    [
        {"items": [{"label_id": 0, "status": "next_focus"}]},
        {"items": [{"label_id": 4, "status": "correct"}]},
        {"items": [{"label_id": 4, "status": "next_focus"}] * 2},
        {"note": "x" * 2001},
        {"revision": -1},
    ],
)
def test_notebook_validates_supported_unique_categories_and_limits(client, extra):
    sid, _, _ = uploaded_session(client)
    assert save(client, sid, **extra).status_code == 422
    assert client.get(f"/api/sessions/{sid}/notebook").json()["revision"] == 0


def test_notebook_and_adopted_goals_survive_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)
    cfg = settings(tmp_path)
    with TestClient(create_app(cfg)) as c:
        sid, _, _ = uploaded_session(c)
        save(c, sid)
        target, _, _ = uploaded_session(c)
        assert adopt(c, target, sid).status_code == 200
    with TestClient(create_app(cfg)) as c:
        assert c.get(f"/api/sessions/{sid}/notebook").json()["items"][0]["status"] == "next_focus"
        assert c.get(f"/api/sessions/{target}/context").json()["active_label_ids"] == [4]
