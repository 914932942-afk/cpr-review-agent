"""Bounded explanations of recorded model output, not clinical or causal explanations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import ClipPrediction, FindingEvidence, ModelClaim, ReviewEvidence, SkeletonProcessingEvidence


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class EvidenceError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def read_bound_artifact(prediction: ClipPrediction, data_dir: Path) -> dict:
    try:
        path = (data_dir / prediction.artifact_rel).resolve()
        if not path.is_relative_to((data_dir / "analysis").resolve()) or not path.is_file():
            raise EvidenceError("RESULT_ARTIFACT_MISSING")
        with path.open("rb") as stream:
            payload = stream.read(1_048_577)
        if len(payload) > 1_048_576:
            raise EvidenceError("RESULT_ARTIFACT_TOO_LARGE")
        if hashlib.sha256(payload).hexdigest() != prediction.artifact_sha256:
            raise EvidenceError("RESULT_ARTIFACT_CHANGED")
        raw = json.loads(payload.decode("utf-8"))
        if not isinstance(raw, dict):
            raise EvidenceError("RESULT_ARTIFACT_INVALID")
        expected = {
            "schema": "cpr-offline-analysis-v2",
            "status": "completed_experimental",
            "video_sha256": prediction.video_sha256,
            "bundle_sha256": prediction.bundle_sha256,
            "decoded_frames": prediction.decoded_frames,
            "reported_fps": prediction.reported_fps,
            "predicted_original_label_ids": prediction.predicted_label_ids,
            "supported_label_ids": prediction.supported_label_ids,
            "scores": [s.model_dump() for s in prediction.scores],
            "public_label_verification": None,
            "model_or_threshold_modified": any(
                item.get("kind") == "improved" for item in raw.get("module_executions", {}).values()
            ),
        }
        if prediction.analysis_route == "body-qualified-v1":
            expected["analysis_route"] = "body-qualified-v1"
        if prediction.profile_id.startswith("single-view-v15-branches-v1:"):
            expected["pipeline_sha256"] = prediction.profile_id.split(":", 1)[1]
        if any(k not in raw or raw[k] != v for k, v in expected.items()):
            raise EvidenceError("RESULT_ARTIFACT_MISMATCH")
    except (TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvidenceError("RESULT_ARTIFACT_INVALID") from exc
    except OSError as exc:
        raise EvidenceError("RESULT_ARTIFACT_UNAVAILABLE") from exc
    return raw


def _sample_count(value):
    return value if type(value) is int and value >= 0 else None


def skeleton_processing(raw):
    fullpose = raw.get("fullpose_audit")
    audit = fullpose.get("m1_stability") if isinstance(fullpose, dict) else None
    if audit is None:
        return None
    if not isinstance(audit, dict):
        raise EvidenceError("SKELETON_AUDIT_INVALID")
    try:
        counts = [
            "original_observed_points",
            "isolated_points_corrected",
            "short_gap_points_reconstructed",
            "unavailable_points_preserved",
        ]
        result = SkeletonProcessingEvidence(
            method=audit["schema"],
            frames=audit["frames"],
            geometry_available=audit["geometry_available"],
            reconstructed_points_are_observations=audit["reconstructed_points_are_observations"],
            **{key: audit[key] for key in counts},
        )
        total = result.frames * 17
        if (
            result.frames != raw.get("decoded_frames")
            or result.isolated_points_corrected > result.original_observed_points
            or result.original_observed_points
            + result.short_gap_points_reconstructed
            + result.unavailable_points_preserved
            != total
        ):
            raise ValueError("Inconsistent keypoint counts")
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError("SKELETON_AUDIT_INVALID") from exc


def build_review_evidence(
    prediction: ClipPrediction, claims: list[ModelClaim], data_dir: Path, source_run_id: str
) -> ReviewEvidence:
    raw = read_bound_artifact(prediction, data_dir)
    score_map = {s.original_label_id: s for s in prediction.scores}
    findings = []
    for claim in claims:
        if claim.label_id not in score_map:
            raise EvidenceError("RESULT_CLAIM_MISMATCH")
        score = score_map[claim.label_id]
        expected_state = (
            "model_flagged" if claim.label_id in prediction.predicted_label_ids else "not_flagged"
        )
        if (
            claim.source_execution_id != prediction.execution_id
            or claim.state != expected_state
            or claim.uncalibrated_score != score.uncalibrated_score
        ):
            raise EvidenceError("RESULT_CLAIM_MISMATCH")
        meets = score.uncalibrated_score >= score.threshold
        if claim.state == "model_flagged":
            note = "This category is retained in the stored final model output."
        else:
            note = "This category is not retained in the stored final output; this does not establish correct CPR."
        if meets and claim.state == "not_flagged":
            note += " Its branch score meets the listed threshold, but this alone does not determine the fused result."
        else:
            note += (
                " The score comparison is one branch readout, not a reconstruction of all fusion decisions."
            )
        findings.append(
            FindingEvidence(
                **claim.model_dump(),
                threshold=score.threshold,
                score_meets_threshold=meets,
                decision_note=note,
            )
        )
    rgb, hands = raw.get("rgb_audit", {}), raw.get("hand_shape_audit", {})
    rgb = rgb if isinstance(rgb, dict) else {}
    hands = hands if isinstance(hands, dict) else {}
    return ReviewEvidence(
        displayed_label_ids=[c.label_id for c in claims],
        assessed_label_ids=prediction.supported_label_ids,
        flagged_label_ids=[c.label_id for c in claims if c.state == "model_flagged"],
        source_run_id=source_run_id,
        source_execution_id=prediction.execution_id,
        clip_id=prediction.clip_id,
        asset_id=prediction.asset_id,
        video_sha256=prediction.video_sha256,
        bundle_sha256=prediction.bundle_sha256,
        artifact_sha256=prediction.artifact_sha256,
        nominal_duration_seconds=prediction.decoded_frames / prediction.reported_fps,
        source_finished_utc=raw.get("finished_utc") if isinstance(raw.get("finished_utc"), str) else None,
        rgb_usable_samples=_sample_count(rgb.get("frames")),
        rgb_requested_samples=_sample_count(rgb.get("requested_frames")),
        hand_usable_samples=_sample_count(hands.get("usable_frames")),
        hand_requested_samples=_sample_count(hands.get("requested_frames")),
        hand_shape_available=hands.get("available") if type(hands.get("available")) is bool else None,
        skeleton_processing=skeleton_processing(raw),
        findings=findings,
    )
