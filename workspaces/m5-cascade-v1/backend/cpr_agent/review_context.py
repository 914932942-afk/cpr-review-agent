"""Bind one turn to one immutable video/report/scope snapshot. No language heuristics."""

from __future__ import annotations

import time

from .context_contracts import ContextSelection, ContextSnapshot, SourceAnchor
from .evidence import build_review_evidence
from .models import Intent, ReviewGoal, ReviewResponse
from .store import DomainError, digest
from .vision import VisionError, validate_receipt

REVIEW_INTENTS = {Intent.ANALYSE, Intent.VERIFY, Intent.EXPLAIN, Intent.SUMMARY}


class ContextBinder:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings

    def report(self, session, report_id, before=float("inf")):
        report = self.store.get("report", report_id)
        run = self.store.get_run(report["run_id"])
        if (
            report["session_id"] != session["id"]
            or run["session_id"] != session["id"]
            or run["clip_id"] != session["clip_id"]
            or run["bundle_id"] != session["bundle_id"]
            or run["status"] not in {"completed", "partially_completed"}
            or run.get("report_id") != report["id"]
            or run.get("response") != report.get("response")
            or report["created_at"] > before
        ):
            raise DomainError(
                409, "CONTEXT_SOURCE_MISMATCH", "Choose a saved analysis report from this video session"
            )
        try:
            response = ReviewResponse.model_validate(report["response"])
        except ValueError as exc:
            raise DomainError(
                422, "CONTEXT_REPORT_INVALID", "The saved report cannot be used as analysis context"
            ) from exc
        if not response.prediction or response.prediction.clip_id != session["clip_id"]:
            raise DomainError(
                422, "CONTEXT_REPORT_HAS_NO_ANALYSIS", "Choose a report containing video analysis"
            )
        return report, run, response

    def validate_selection(self, session_id, selection):
        session = self.store.get("session", session_id)
        if selection.source_report_id:
            self.report(session, selection.source_report_id)

    def snapshot(self, run, selection):
        session = self.store.get("session", run["session_id"])
        if session["clip_id"] != run["clip_id"] or session["bundle_id"] != run["bundle_id"]:
            raise DomainError(409, "CONTEXT_SESSION_CHANGED", "Video binding changed; start a new request")
        reports, candidates = [], []
        # Only completed/partial, owner-scoped analysis reports predating this request are eligible.
        for report in self.store.list_objects("report"):
            if report["session_id"] != session["id"] or report["created_at"] > run["created_at"]:
                continue
            if not (report.get("response") or {}).get("prediction"):
                continue
            try:
                report_run = self.store.get_run(report["run_id"])
                if (
                    report_run["status"] not in {"completed", "partially_completed"}
                    or report_run["clip_id"] != run["clip_id"]
                    or report_run["bundle_id"] != run["bundle_id"]
                ):
                    continue
                candidates.append(report)
                reports.append(self.report(session, report["id"], run["created_at"]))
            except (DomainError, ValueError):
                continue
        reports.sort(key=lambda r: (r[0]["created_at"], r[0]["id"]), reverse=True)
        candidates.sort(key=lambda r: (r["created_at"], r["id"]), reverse=True)
        chosen = (
            self.report(session, selection.source_report_id, run["created_at"])
            if selection.source_report_id
            else (self.report(session, candidates[0]["id"], run["created_at"]) if candidates else None)
        )
        values = dict(
            session_id=session["id"], run_id=run["id"], clip_id=run["clip_id"], bundle_id=run["bundle_id"]
        )
        if run["clip_id"]:
            clip = self.store.get("clip", run["clip_id"])
            asset = self.store.get("asset", clip["asset_id"])
            values.update(asset_id=asset["id"], video_name=asset["name"], video_sha256=asset["sha256"])
        if chosen:
            report, report_run, response = chosen
            prediction = response.prediction
            anchor = SourceAnchor(
                session_id=session["id"],
                report_id=report["id"],
                report_run_id=report_run["id"],
                execution_id=prediction.execution_id,
                profile_id=prediction.profile_id,
                video_sha256=prediction.video_sha256,
                bundle_sha256=prediction.bundle_sha256,
                artifact_sha256=prediction.artifact_sha256,
                label_ids=[c.label_id for c in response.classification_claims],
            )
            values.update(
                source=anchor,
                source_status="available",
                active_label_ids=anchor.label_ids,
                source_selection="viewed_report" if selection.source_report_id else "latest_confirmed",
            )
            try:
                self.prediction(anchor, run["clip_id"])
            except (DomainError, ValueError, VisionError, OSError) as exc:
                values.update(
                    source_status="unavailable",
                    source_error_code=getattr(exc, "code", "CONTEXT_SOURCE_INVALID"),
                )
            recent_goals, recent_ids = [], []
            for item, item_run, _ in reports:
                if item["created_at"] > report["created_at"]:
                    continue  # Viewing an older report must not inherit a newer scope.
                turn = self.store.get("turn", item_run["turn_id"])
                draft = turn.get("resolved_goal")
                if not draft or turn["id"] in recent_ids:
                    continue
                goal = ReviewGoal.model_validate(draft)
                if goal.intent in REVIEW_INTENTS:
                    recent_goals.append(goal)
                    recent_ids.append(turn["id"])
                if len(recent_goals) == 4:
                    break
            values.update(recent_goals=recent_goals, recent_turn_ids=recent_ids)
        if not chosen:
            from .practice_continuity import PracticeContinuity

            starting_focus = PracticeContinuity(self.store, self.settings).selected(
                session, run["created_at"]
            )
            if starting_focus["status"] == "available":
                # An explicitly chosen goal is not a source report or a saved prediction for this clip.
                values["active_label_ids"] = starting_focus["label_ids"]
        snapshot = ContextSnapshot(**values)
        snapshot.snapshot_id = digest(snapshot.model_dump(exclude={"snapshot_id"}, mode="json"))
        return snapshot

    def preview(self, session_id, source_report_id=None):
        session = self.store.get("session", session_id)
        return self.snapshot(
            {
                "id": "preview",
                "session_id": session_id,
                "clip_id": session["clip_id"],
                "bundle_id": session["bundle_id"],
                "created_at": time.time(),
            },
            ContextSelection(source_report_id=source_report_id),
        )

    def prediction(self, anchor, clip_id):
        report = self.store.get("report", anchor.report_id)
        report_run = self.store.get_run(report["run_id"])
        prediction = ReviewResponse.model_validate(report["response"]).prediction
        if (
            not prediction
            or report_run["status"] not in {"completed", "partially_completed"}
            or report_run.get("report_id") != report["id"]
            or report_run.get("response") != report.get("response")
            or report["session_id"] != anchor.session_id
            or report_run["session_id"] != anchor.session_id
            or report_run["id"] != anchor.report_run_id
            or report_run["clip_id"] != clip_id
            or prediction.clip_id != clip_id
            or prediction.execution_id != anchor.execution_id
            or prediction.profile_id != anchor.profile_id
            or prediction.video_sha256 != anchor.video_sha256
            or prediction.bundle_sha256 != anchor.bundle_sha256
            or prediction.artifact_sha256 != anchor.artifact_sha256
        ):
            raise DomainError(409, "CONTEXT_SOURCE_CHANGED", "The bound report changed; open it again")
        result = validate_receipt(prediction.model_dump(), self.store, self.settings)
        execution = self.store.get_tool(result.execution_id)
        source_run = self.store.get_run(execution["run_id"])
        if (
            execution["status"] != "succeeded"
            or execution["output"] != result.model_dump()
            or source_run["clip_id"] != clip_id
        ):
            raise DomainError(409, "CONTEXT_SOURCE_CHANGED", "The bound analysis is no longer valid")
        build_review_evidence(
            result,
            ReviewResponse.model_validate(report["response"]).classification_claims,
            self.settings.data_dir,
            execution["run_id"],
        )
        return result

    @staticmethod
    def resolve(snapshot, goal, reference, confirmed=False):
        if not confirmed and reference.source == "ambiguous":
            return None, "CONTEXT_SOURCE_AMBIGUOUS"
        if not confirmed and reference.source == "viewed_report" and snapshot.source is None:
            return None, "CONTEXT_SOURCE_UNAVAILABLE"
        if not confirmed and reference.scope == "ambiguous":
            return None, "CONTEXT_SCOPE_AMBIGUOUS"
        targets = goal.target_label_ids
        if not confirmed and reference.scope in {"previous_scope", "single_finding"}:
            targets = snapshot.active_label_ids
            if not targets:
                return None, "CONTEXT_SCOPE_UNAVAILABLE"
            if reference.scope == "single_finding" and len(targets) != 1:
                return None, "CONTEXT_SCOPE_AMBIGUOUS"
        draft = goal.model_dump(mode="json")
        draft["target_label_ids"] = list(targets)
        if goal.intent in {Intent.EXPLAIN, Intent.SUMMARY}:
            draft["analysis_mode"] = "existing_only"
        return ReviewGoal.model_validate(draft), None


def snapshot_context(snapshot, preferences):
    """Context passed to the gateway is filtered again before any optional network call."""
    return {
        "clip_id": snapshot.clip_id,
        "binding_version": snapshot.version,
        "recent_turn_ids": snapshot.recent_turn_ids,
        "recent_goals": [g.model_dump(mode="json") for g in snapshot.recent_goals],
        "preferences": preferences,
        "active_review": {
            "label_ids": snapshot.active_label_ids,
            "source_available": snapshot.source_status == "available",
        },
        "binding": snapshot.model_dump(mode="json"),
    }
