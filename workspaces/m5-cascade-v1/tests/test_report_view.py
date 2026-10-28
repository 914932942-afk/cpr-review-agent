"""Report presentation and source validity; model outputs are explicit test doubles."""

from __future__ import annotations

import json

from cpr_agent.report_view import render_saved_review
from test_real_video import real_client as real_client
from test_real_video import submit, uploaded_session, wait


def test_report_escapes_content_and_keeps_raw_cv_separate_from_assessment():
    snapshot = {
        "id": "report-id",
        "run_id": "run-id",
        "is_scaffold": False,
        "response": {
            "title": "<script>alert(1)</script>",
            "message": "<img src=x onerror=alert(1)>",
            "classification_claims": [
                {
                    "label_id": 4,
                    "name": "Recorded category",
                    "state": "model_flagged",
                    "uncalibrated_score": 0.7,
                }
            ],
            "module_assessment": {"states": {"4": "uncertain"}},
            "limitations": ["<unsafe>"],
        },
    }
    result = render_saved_review(snapshot, {"status": "unavailable", "error_code": "MISSING"}, "<video>")
    assert "<script>" not in result and "<img src=x" not in result
    assert "&lt;script&gt;" in result and "&lt;video&gt;" in result
    assert "<td>Model flagged</td><td>Uncertain</td>" in result
    assert "Historical saved findings only" in result
    assert "<td>0.7000</td><td>Unavailable</td>" in result
    assert "Source files and bindings verified" not in result


def test_empty_or_scaffold_record_does_not_imply_successful_analysis():
    result = render_saved_review(
        {"id": "empty", "is_scaffold": True, "response": {}}, {"status": "not_applicable"}
    )
    assert "Scaffold only. No CPR analysis was performed." in result
    assert "No findings were produced" in result
    assert "model flags in displayed scope" not in result


def test_export_has_readable_scoped_findings_without_new_tool_calls(real_client):
    client = real_client
    sid, _, _ = uploaded_session(client)
    original = wait(client, submit(client, sid))
    followup = wait(client, submit(client, sid, intent="explain", targets=[4], analysis_mode="existing_only"))
    report_id = followup["report_id"]
    before = client.get(f"/api/reports/{report_id}").json()
    html = client.get(f"/api/reports/{report_id}/export").text
    assert "Source files and bindings verified" in html
    assert "<td>#4</td>" in html and "<td>#2</td>" not in html
    assert "Raw CV finding" in html and "Print / Save as PDF" in html
    assert "&quot;source_execution_id&quot;" in html
    assert client.get(f"/api/runs/{followup['id']}/tools").json() == []
    assert len(client.get(f"/api/runs/{original['id']}/tools").json()) == 1
    assert client.get(f"/api/reports/{report_id}").json() == before


def test_export_marks_historical_findings_when_artifact_changes(real_client):
    client = real_client
    sid, _, _ = uploaded_session(client)
    run = wait(client, submit(client, sid))
    report_id = run["report_id"]
    before = client.get(f"/api/reports/{report_id}").json()
    artifact = client.app.state.runtime.settings.data_dir / run["response"]["prediction"]["artifact_rel"]
    artifact.write_text(json.dumps({"changed": True}), encoding="utf-8")
    result = client.get(f"/api/reports/{report_id}/export")
    assert result.status_code == 200
    assert "Historical saved findings only" in result.text
    assert "RESULT_ARTIFACT_CHANGED" in result.text
    assert "Source files and bindings verified" not in result.text
    assert "<td>#2</td>" in result.text
    assert client.get(f"/api/reports/{report_id}").json() == before
    assert len(client.get(f"/api/runs/{run['id']}/tools").json()) == 1
