"""Readable, printable rendering of a saved review. Never performs inference."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone


def render_saved_review(
    snapshot: dict, evidence: dict, video_name: str | None = None, explanation: dict | None = None
) -> str:
    def esc(value):
        return html.escape(str(value), quote=True)

    response = snapshot.get("response") or {}
    claims = response.get("classification_claims") or []
    assessment = response.get("module_assessment") or {}
    assessed = assessment.get("states") or {}
    source = evidence.get("review") if evidence.get("status") == "available" else None
    grounded = (explanation or {}).get("review") if (explanation or {}).get("status") == "available" else None
    explanation_html = ""
    if grounded:
        explanation_html = (
            "<h2>Evidence-constrained review</h2>"
            f"<p>{esc(grounded['summary'])}</p>"
            + "".join(
                f"<h3>#{esc(entry['label_id'])} · {esc(entry['name'])}</h3>"
                f"<p>{esc(entry['meaning'])}</p><p>{esc(entry['recorded_result'])}</p>"
                f"<p class='muted'>{esc(entry['decision_note'])}</p>"
                for entry in grounded["entries"]
            )
            + "<ul>"
            + "".join(f"<li>{esc(item)}</li>" for item in grounded["limitations"])
            + "</ul>"
        )
    elif snapshot.get("grounded_review"):
        explanation_html = (
            "<p class='notice'>The saved explanation could not be revalidated and is hidden.</p>"
        )
    readouts = {f["label_id"]: f for f in source["findings"]} if source else {}
    state_names = {
        "model_flagged": "Model flagged",
        "not_flagged": "Not flagged",
        "uncertain": "Uncertain",
        "not_evaluated": "Not evaluated",
    }
    if snapshot.get("is_scaffold"):
        notice = "Scaffold only. No CPR analysis was performed."
    else:
        notice = (
            "Experimental whole-clip training review. No supported error flagged does not establish "
            "correct CPR. Scores are uncalibrated model outputs, not medical probabilities."
        )
    if source:
        source_note = "Source files and bindings verified when this report was opened."
    elif claims:
        source_note = (
            "Historical saved findings only. Source evidence is currently unavailable; "
            "these findings have not been revalidated. " + str(evidence.get("error_code") or "")
        )
    else:
        source_note = "No source evidence is attached to this request."
    rows = []
    for claim in sorted(claims, key=lambda c: c["state"] != "model_flagged"):
        label = claim["label_id"]
        finding = readouts.get(label)
        module_state = assessed.get(str(label), assessed.get(label))
        state = state_names.get(claim["state"], claim["state"])
        score = f"{claim['uncalibrated_score']:.4f}"
        threshold = f"{finding['threshold']:.4f}" if finding else "Unavailable"
        rows.append(
            f"<tr><td>#{esc(label)}</td><th scope='row'>{esc(claim['name'])}</th>"
            f"<td>{esc(state)}</td><td>{esc(state_names.get(module_state, module_state) or 'Not attached')}</td>"
            f"<td>{esc(score)}</td><td>{esc(threshold)}</td></tr>"
        )
    if rows:
        findings = (
            "<div class='table-scroll'><table><thead><tr><th>ID</th><th>Category</th>"
            "<th>Raw CV finding</th><th>Separate assessment</th><th>Branch score</th>"
            "<th>Branch threshold</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            "<p class='muted'>Branch scores and thresholds do not fully explain the final fusion decision. "
            "The separate assessment does not overwrite the recorded CV finding.</p>"
        )
    else:
        findings = "<p>No findings were produced for this request.</p>"
    flagged = sum(c["state"] == "model_flagged" for c in claims)
    summary = (
        f"<div class='metrics'><div><b>{flagged}</b><span>model flags in displayed scope</span></div>"
        f"<div><b>{len(claims)}</b><span>categories displayed</span></div>"
        f"<div><b>{'Saved result' if response.get('reused_result') else 'Request record'}</b>"
        "<span>No inference is triggered by exporting</span></div></div>"
        if claims
        else ""
    )
    limitations = "".join(f"<li>{esc(line)}</li>" for line in response.get("limitations", []))
    prediction = response.get("prediction") or {}
    provenance = {
        "Report ID": snapshot.get("id"),
        "Request run": snapshot.get("run_id"),
        "Source execution": prediction.get("execution_id"),
        "Profile": prediction.get("profile_id"),
        "Video SHA256": prediction.get("video_sha256"),
        "Bundle SHA256": prediction.get("bundle_sha256"),
        "Result SHA256": prediction.get("artifact_sha256"),
    }
    source_rows = "".join(
        f"<dt>{esc(key)}</dt><dd>{esc(value)}</dd>" for key, value in provenance.items() if value
    )
    opened = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    raw = esc(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>CPR Review · Saved record</title><style>"
        "*{box-sizing:border-box}body{font:15px/1.7 system-ui,sans-serif;color:#193e46;"
        "background:#f3f6f6;margin:0}main{max-width:1080px;margin:36px auto;padding:32px;"
        "background:white;border:1px solid #dce7e6;border-radius:16px}"
        "h1{font-size:30px;line-height:1.2}h2{font-size:18px;margin-top:30px}"
        ".eyebrow{font-size:11px;letter-spacing:2px;color:#14786d}.muted{color:#5d747c;font-size:13px}"
        ".notice{background:#fff5df;padding:16px;border-radius:10px}.source-note{padding:12px;"
        "border:1px solid #dce7e6;border-radius:8px}.metrics{display:flex;gap:24px;flex-wrap:wrap;"
        "margin:24px 0}.metrics div{flex:1;min-width:150px;background:#eef6f3;padding:16px;"
        "border-radius:10px}.metrics b,.metrics span{display:block}.metrics b{font-size:23px}"
        ".metrics span{font-size:12px;color:#5d747c}.table-scroll{overflow:auto}"
        "table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;"
        "padding:12px;border-bottom:1px solid #dce7e6}thead{background:#edf4f3}"
        "dl{display:grid;grid-template-columns:150px minmax(0,1fr);gap:10px;font-size:12px}"
        "dd{margin:0;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "font-size:11px;background:#f3f6f6;padding:16px}summary{cursor:pointer}"
        "@media(max-width:600px){main{margin:0;padding:20px;border:0}dl{grid-template-columns:1fr}"
        ".metrics{gap:10px}}@media print{body,main{background:white;margin:0;border:0;padding:0}"
        ".raw-record{display:none}thead{display:table-header-group}tr{break-inside:avoid}"
        ".metrics{break-inside:avoid}h2{break-after:avoid}}"
        "</style><main><p class='eyebrow'>CPR REVIEW / SAVED TRAINING RECORD</p>"
        f"<h1>{esc(response.get('title') or 'Saved review record')}</h1>"
        f"<p class='muted'>Video: {esc(video_name or 'Not available')} · Opened {opened}</p>"
        f"<p class='notice'>{esc(notice)}</p><p>{esc(response.get('message') or 'Request saved.')}</p>"
        f"<p class='source-note'>{esc(source_note)}</p>{summary}{explanation_html}<h2>Recorded findings</h2>{findings}"
        "<h2>How to use this record</h2><ol><li>Review the flagged categories in the source video.</li>"
        "<li>Open recorded evidence in the workspace to inspect the model output and its scope.</li>"
        "<li>Keep this record for the next training review. It is not proof of clinical competence.</li></ol>"
        f"<h2>Scope and limitations</h2><ul>{limitations}</ul>"
        f"<h2>Record source</h2><dl>{source_rows}</dl>"
        "<p class='muted'>Findings cover the whole clip. Error onset timestamps, compression depth "
        "and measured BPM are not supplied. Save this page or use the browser Print / Save as PDF command.</p>"
        f"<details class='raw-record'><summary>Original saved record (JSON)</summary><pre>{raw}</pre></details>"
        "</main></html>"
    )
