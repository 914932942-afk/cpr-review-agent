"""Reproducible explanations of verified records, separate from the pinned CV response."""

from __future__ import annotations

from .explanation_model import validate_plan
from .models import ReviewEvidence, ReviewResponse
from .store import digest
from .vision import LABEL_NAMES

VERSION = "grounded-review-v1"
GLOSSARY = {
    1: "A dataset category for the overlapping-hands configuration.",
    2: "A dataset category for the clenched-hands configuration.",
    3: "A dataset category for a single-hand compression configuration.",
    4: "A dataset category referring to arm bending during compression.",
    5: "A dataset category referring to arm tilt during compression.",
    6: "A dataset category named jump pressing, related to the compression movement pattern.",
    7: "A dataset category referring to a squatting body position.",
    8: "A dataset category referring to a standing body position.",
    9: "A dataset category for hand placement.",
    11: "A dataset category named slow frequency; this classifier does not measure BPM.",
    13: "A dataset category for changes in pressing position across the clip.",
}
LIMITATIONS = [
    "These are experimental whole-clip model findings, not confirmed mistakes or a CPR competence assessment.",
    "Not flagged does not mean correct CPR; categories outside this review scope are not summarised here.",
    "Branch scores are uncalibrated model outputs, not medical probabilities or a full explanation of fusion.",
    "Error onset times, physical compression depth, measured BPM and causes are not available.",
    "Category meanings describe dataset labels, not clinical instructions. Source playback covers the whole clip.",
]


def explanation_input(response):
    parsed = ReviewResponse.model_validate(response)
    if not parsed.prediction or not parsed.review or not parsed.classification_claims:
        raise ValueError("No explainable evidence")
    # Check saved evidence against the response rather than trusting any generated text.
    review = parsed.review
    prediction = parsed.prediction
    claims = parsed.classification_claims
    if (
        review.source_execution_id != prediction.execution_id
        or review.video_sha256 != prediction.video_sha256
        or review.artifact_sha256 != prediction.artifact_sha256
        or review.displayed_label_ids != [c.label_id for c in claims]
        or len(review.findings) != len(claims)
    ):
        raise ValueError("Explanation evidence mismatch")
    for claim, finding in zip(claims, review.findings, strict=True):
        if claim.model_dump() != finding.model_dump(include=set(type(claim).model_fields)):
            raise ValueError("Explanation claim mismatch")
        if claim.name != LABEL_NAMES[claim.label_id]:
            raise ValueError("Unknown category wording")
    return parsed


def build_grounded_review(response, intent, resolution):
    parsed = explanation_input(response)
    review = parsed.review
    labels = review.displayed_label_ids
    order = sorted(labels, key=lambda i: (i not in review.flagged_label_ids, i))
    lead = "evidence" if intent == "explain" else "findings"
    mode = "evidence_template"
    if resolution.get("plan"):
        plan = validate_plan(resolution["plan"], labels)
        order, lead, mode = plan.ordered_label_ids, plan.lead, "model_organised_evidence"
    findings = {f.label_id: f for f in review.findings}
    entries = []
    for label in order:
        f = findings[label]
        entries.append(
            {
                "label_id": label,
                "name": LABEL_NAMES[label],
                "meaning": GLOSSARY[label],
                "state": f.state,
                "recorded_result": (
                    "The saved final output flags this category for this clip. It is a model finding to review."
                    if f.state == "model_flagged"
                    else "The saved final output does not flag this category. This does not confirm correct technique."
                ),
                "uncalibrated_score": f.uncalibrated_score,
                "threshold": f.threshold,
                "decision_note": f.decision_note,
                "source_execution_id": f.source_execution_id,
            }
        )
    flagged = review.flagged_label_ids
    value = {
        "version": VERSION,
        "intent": intent,
        "mode": mode,
        "lead": lead,
        "opening": {
            "findings": "Start with the recorded model flags, then check their evidence and limits.",
            "evidence": "Each explanation below is bound to the saved model output and its verified source.",
            "limits": "These findings cannot locate mistakes in time or establish correct CPR or physical measurements.",
        }[lead],
        "scope_label_ids": labels,
        "assessed_label_ids": review.assessed_label_ids,
        "outside_scope_label_ids": [i for i in review.assessed_label_ids if i not in labels],
        "flagged_label_ids": flagged,
        "summary": f"{len(flagged)} model-flagged categories in this {len(labels)}-category review scope.",
        "entries": entries,
        "limitations": LIMITATIONS,
        "next_steps": [
            "Open the source video and compare the recorded category with the whole clip.",
            "Choose a category to review its saved evidence, or change the scope without rerunning vision.",
            "Save a scope summary and reopen it from this session's review trail next time.",
        ],
        "source_execution_id": review.source_execution_id,
        "video_sha256": review.video_sha256,
        "artifact_sha256": review.artifact_sha256,
        "evidence_digest": digest(review.model_dump()),
        "reused_result": parsed.reused_result,
        "organisation_receipt": resolution,
    }
    value["content_sha256"] = digest(value)
    return value


def verified_grounded_review(snapshot, evidence):
    """Reopening/exporting verifies both current source files and the exact saved explanation."""
    saved = snapshot.get("grounded_review")
    if not saved:
        return {"status": "not_available", "review": None, "error_code": "LEGACY_REVIEW_NO_EXPLANATION"}
    if evidence.get("status") != "available":
        return {"status": "unavailable", "review": None, "error_code": evidence.get("error_code")}
    try:
        current = ReviewEvidence.model_validate(evidence["review"]).model_dump()
        if digest(current) != saved["evidence_digest"]:
            raise ValueError("Evidence changed")
        rebuilt = build_grounded_review(snapshot["response"], saved["intent"], saved["organisation_receipt"])
        if rebuilt != saved:
            raise ValueError("Explanation changed")
        return {"status": "available", "review": saved, "error_code": None}
    except (KeyError, TypeError, ValueError):
        return {"status": "unavailable", "review": None, "error_code": "EXPLANATION_RECORD_INVALID"}
