"""Verified, read-only report comparison. No scoring changes or progress inference."""

from __future__ import annotations

from .context_contracts import SourceAnchor
from .review_context import ContextBinder
from .store import DomainError
from .vision import LABEL_NAMES, VisionError


class ReviewRecords:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings

    def structural(self, report_id):
        report = self.store.get("report", report_id)
        session = self.store.get("session", report["session_id"])
        report, run, response = ContextBinder(self.store, self.settings).report(session, report_id)
        asset = self.store.get("asset", response.prediction.asset_id)
        return report, run, response, session, asset

    @staticmethod
    def metadata(parts):
        report, _, response, session, asset = parts
        return {
            "report_id": report["id"],
            "session_id": session["id"],
            "clip_id": session["clip_id"],
            "asset_id": asset["id"],
            "video_name": asset["name"],
            "created_at": report["created_at"],
            "scope_label_ids": [c.label_id for c in response.classification_claims],
        }

    def verify(self, parts):
        report, run, response, session, _ = parts
        prediction = response.prediction
        anchor = SourceAnchor(
            session_id=session["id"],
            report_id=report["id"],
            report_run_id=run["id"],
            execution_id=prediction.execution_id,
            profile_id=prediction.profile_id,
            video_sha256=prediction.video_sha256,
            bundle_sha256=prediction.bundle_sha256,
            artifact_sha256=prediction.artifact_sha256,
            label_ids=[c.label_id for c in response.classification_claims],
        )
        ContextBinder(self.store, self.settings).prediction(anchor, session["clip_id"])

    def options(self, report_id):
        current = self.structural(report_id)
        try:
            self.verify(current)
        except (DomainError, ValueError, VisionError, OSError, KeyError, TypeError):
            return {"status": "unavailable", "current": self.metadata(current), "items": []}
        reports = sorted(
            self.store.list_objects("report"), key=lambda r: (r["created_at"], r["id"]), reverse=True
        )
        seen, items = set(), []
        for report in reports:
            if report["session_id"] == current[3]["id"] or not (report.get("response") or {}).get(
                "prediction"
            ):
                continue
            try:
                parts = self.structural(report["id"])
                identity = (
                    parts[3]["id"],
                    tuple(c.label_id for c in parts[2].classification_claims),
                    parts[2].prediction.profile_id,
                )
                if identity in seen:
                    continue
                seen.add(identity)
                available = True
                try:
                    self.verify(parts)
                except (DomainError, ValueError, VisionError, OSError, KeyError, TypeError):
                    available = False
                items.append({**self.metadata(parts), "source_available": available})
            except (DomainError, ValueError, KeyError, TypeError):
                continue
            if len(items) == 30:
                break
        return {"status": "available", "current": self.metadata(current), "items": items}

    def compare(self, left_id, right_id):
        left, right = self.structural(left_id), self.structural(right_id)
        result = {
            "status": "incomparable",
            "rows": [],
            "left": self.metadata(left),
            "right": self.metadata(right),
        }
        if left[3]["id"] == right[3]["id"]:
            return {**result, "reason": "SAME_PRACTICE"}
        p, q = left[2].prediction, right[2].prediction
        if (
            p.profile_id != q.profile_id
            or p.bundle_sha256 != q.bundle_sha256
            or left[1]["bundle_id"] != right[1]["bundle_id"]
        ):
            return {**result, "reason": "DIFFERENT_MODEL_VERSION"}
        try:
            self.verify(left)
            self.verify(right)
        except (DomainError, ValueError, VisionError, OSError, KeyError, TypeError):
            return {**result, "status": "unavailable", "reason": "SOURCE_UNAVAILABLE"}
        a, b = (
            {c.label_id: c for c in left[2].classification_claims},
            {c.label_id: c for c in right[2].classification_claims},
        )
        common = sorted(a.keys() & b.keys())
        if not common:
            return {**result, "reason": "NO_SHARED_SCOPE"}
        return {
            **result,
            "status": "available",
            "common_label_ids": common,
            "left_only_label_ids": sorted(a.keys() - b.keys()),
            "right_only_label_ids": sorted(b.keys() - a.keys()),
            "same_video_file": p.video_sha256 == q.video_sha256,
            "rows": [
                {"label_id": i, "name": LABEL_NAMES[i], "left_state": a[i].state, "right_state": b[i].state}
                for i in common
            ],
            "notice": "Separate model records, not a progress score or a competence assessment. Not flagged does not mean correct technique.",
        }
