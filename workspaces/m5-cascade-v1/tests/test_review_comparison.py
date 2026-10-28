"""Shared-scope comparison of independently verified reports; no inferred progress."""

from __future__ import annotations

import pytest
from cpr_agent.api import create_app
from cpr_agent.vision import RealVisionTool
from fastapi.testclient import TestClient
from test_context_binding import first_review, settings
from test_real_video import make_prediction


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)
    with TestClient(create_app(settings(tmp_path))) as c:
        first = first_review(c, [2, 4])
        second = first_review(c, [4, 5])
        yield c, first, second


def compare(c, a, b):
    return c.get("/api/review-comparison", params={"left_report_id": a, "right_report_id": b})


def test_comparison_only_includes_shared_report_scope_and_no_score_or_improvement(prepared):
    c, first, second = prepared
    r = compare(c, first[3]["report_id"], second[3]["report_id"])
    assert r.status_code == 200
    v = r.json()
    assert v["status"] == "available" and v["common_label_ids"] == [4]
    assert v["left_only_label_ids"] == [2] and v["right_only_label_ids"] == [5]
    assert v["same_video_file"] is True
    assert v["rows"] == [
        {"label_id": 4, "name": "Bent arms", "left_state": "not_flagged", "right_state": "not_flagged"}
    ]
    assert "score" not in v["rows"][0] and "improvement" not in v
    options = c.get(f"/api/reports/{second[3]['report_id']}/comparison-options").json()
    assert options["items"][0]["report_id"] == first[3]["report_id"]
    assert len(c.get(f"/api/runs/{first[3]['id']}/tools").json()) == 1


@pytest.mark.parametrize(
    "kind", ["same_practice", "different_version", "missing_source", "no_overlap", "other_owner"]
)
def test_incomparable_or_invalid_reports_do_not_return_rows(prepared, kind):
    c, first, second = prepared
    a, b = first[3]["report_id"], second[3]["report_id"]
    store = c.app.state.runtime.store
    if kind == "same_practice":
        b = a
    elif kind == "missing_source":
        (c.app.state.runtime.settings.data_dir / first[1]["storage_rel"]).unlink()
    elif kind == "no_overlap":
        third = first_review(c, [5])[3]
        a, b = first[3]["report_id"], third["report_id"]
    elif kind == "other_owner":
        with store.tx() as db:
            db.execute("UPDATE objects SET owner='other' WHERE id=?", (a,))
    else:
        # Keep the saved report and run structurally consistent; version mismatch must still be rejected.
        report = store.get("report", a)
        run = store.get_run(report["run_id"])
        report["response"]["prediction"]["profile_id"] = "single-view-v15-branches-v1:" + "a" * 64
        run["response"] = report["response"]
        store.put("report", report)
        with store.tx() as db:
            store._write_run(db, run)
    response = compare(c, a, b)
    if kind == "other_owner":
        assert response.status_code == 404
    else:
        assert response.status_code == 200 and response.json()["rows"] == []
        assert response.json()["status"] in ["incomparable", "unavailable"]
