"""Explicit, session-scoped starting goals. Old findings never become new video evidence."""

from __future__ import annotations

import time

from pydantic import Field, model_validator

from .context_contracts import SourceAnchor
from .evidence import build_review_evidence
from .grounded_review import verified_grounded_review
from .models import CANDIDATE_LABELS, StrictModel
from .review_notebook import ReviewNotebook
from .store import DomainError, digest
from .vision import VisionError, media_context


class ContinuityUpdate(StrictModel):
    source_report_id: str | None = Field(default=None, min_length=1, max_length=120)
    source_notebook_session_id: str | None = Field(default=None, min_length=1, max_length=120)
    source_notebook_revision: int | None = Field(default=None, ge=1)
    revision: int = Field(ge=0)

    @model_validator(mode="after")
    def one_source(self):
        if self.source_notebook_session_id:
            if self.source_report_id or self.source_notebook_revision is None:
                raise ValueError("Choose one source with its revision")
        elif self.source_notebook_revision is not None:
            raise ValueError("Notebook source is required")
        return self


class PracticeContinuity:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings

    def record(self, session_id):
        try:
            return self.store.get("practice_focus", "practice-focus:" + session_id)
        except DomainError as exc:
            if exc.status != 404:
                raise
            return None

    def source(self, target, report_id):
        from .review_context import ContextBinder

        try:
            report = self.store.get("report", report_id)
            previous = self.store.get("session", report["session_id"])
            if (
                previous["id"] == target["id"]
                or previous["clip_id"] == target["clip_id"]
                or previous["bundle_id"] != target["bundle_id"]
                or previous["created_at"] >= target["created_at"]
            ):
                raise ValueError("Not an earlier compatible practice")
            binder = ContextBinder(self.store, self.settings)
            report, run, response = binder.report(previous, report_id)
            if (report.get("grounded_review") or {}).get("intent") != "summarise" or run.get(
                "grounded_review"
            ) != report.get("grounded_review"):
                raise ValueError("A saved recap is required")
            prediction = response.prediction
            anchor = SourceAnchor(
                session_id=previous["id"],
                report_id=report_id,
                report_run_id=run["id"],
                execution_id=prediction.execution_id,
                profile_id=prediction.profile_id,
                video_sha256=prediction.video_sha256,
                bundle_sha256=prediction.bundle_sha256,
                artifact_sha256=prediction.artifact_sha256,
                label_ids=[c.label_id for c in response.classification_claims],
            )
            binder.prediction(anchor, previous["clip_id"])
            execution = self.store.get_tool(prediction.execution_id)
            evidence = build_review_evidence(
                prediction, response.classification_claims, self.settings.data_dir, execution["run_id"]
            )
            checked = verified_grounded_review(
                report, {"status": "available", "review": evidence.model_dump()}
            )
            if checked["status"] != "available" or not checked["review"]["scope_label_ids"]:
                raise ValueError("Recap cannot be verified")
            asset = self.store.get("asset", prediction.asset_id)
            current_clip = self.store.get("clip", target["clip_id"])
            current_asset = self.store.get("asset", current_clip["asset_id"])
            review = checked["review"]
            return {
                "report_id": report_id,
                "session_id": previous["id"],
                "clip_id": previous["clip_id"],
                "video_name": asset["name"],
                "created_at": report["created_at"],
                "label_ids": review["scope_label_ids"],
                "same_video_file": asset["sha256"] == current_asset["sha256"],
                "report_digest": digest(report),
                "recap": {
                    "summary": review["summary"],
                    "entries": [
                        {k: e[k] for k in ["label_id", "name", "meaning", "state", "recorded_result"]}
                        for e in review["entries"]
                    ],
                },
            }
        except (DomainError, ValueError, KeyError, TypeError, OSError, VisionError) as exc:
            raise DomainError(
                410, "PREVIOUS_RECAP_UNAVAILABLE", "The earlier recap could not be verified"
            ) from exc

    def selected(self, target, before=float("inf")):
        record = self.record(target["id"])
        if not record or not (record.get("source_report_id") or record.get("source_notebook_session_id")):
            return {"status": "not_selected", "label_ids": [], "source": None}
        try:
            if (
                record["digest"] != digest({k: v for k, v in record.items() if k != "digest"})
                or record["session_id"] != target["id"]
                or record["clip_id"] != target["clip_id"]
                or record["bundle_id"] != target["bundle_id"]
                or record["created_at"] > before
                or not record["label_ids"]
                or any(type(i) is not int or i not in CANDIDATE_LABELS for i in record["label_ids"])
            ):
                raise ValueError("Starting focus binding changed")
            if record.get("source_kind") == "notebook":
                source = ReviewNotebook(self.store).goal_source(
                    target, record["source_notebook_session_id"], record["source_notebook_revision"]
                )
                matches = record["source_digest"] == source["source_digest"]
            else:
                source = self.source(target, record["source_report_id"])
                matches = record["report_digest"] == source["report_digest"]
            if not matches or record["label_ids"] != source["label_ids"]:
                raise ValueError("Earlier scope changed")
            return {"status": "available", "label_ids": record["label_ids"], "source": self.metadata(source)}
        except (DomainError, ValueError, KeyError, TypeError):
            return {"status": "unavailable", "label_ids": [], "source": None}

    @staticmethod
    def metadata(source):
        return {k: v for k, v in source.items() if k not in {"report_digest", "recap", "source_digest"}}

    def locked(self, session_id):
        if any(r["session_id"] == session_id for r in self.store.active_runs()):
            return True
        for report in self.store.list_objects("report"):
            if report["session_id"] == session_id and (report.get("response") or {}).get("prediction"):
                if self.store.get_run(report["run_id"])["status"] in {"completed", "partially_completed"}:
                    return True
        return False

    def view(self, session_id):
        target = self.store.get("session", session_id)
        record = self.record(session_id)
        selected = self.selected(target)
        candidates, seen = [], set()
        reports = sorted(
            self.store.list_objects("report"), key=lambda r: (r["created_at"], r["id"]), reverse=True
        )
        for report in reports:
            if (report.get("grounded_review") or {}).get("intent") != "summarise" or report[
                "session_id"
            ] in seen:
                continue
            try:
                source = self.source(target, report["id"])
            except DomainError:
                continue
            candidates.append(self.metadata(source))
            seen.add(report["session_id"])
            if len(candidates) == 5:
                break
        return {
            "session_id": session_id,
            "revision": record["revision"] if record else 0,
            "locked": self.locked(session_id),
            "selection": selected,
            "candidates": candidates,
            "notebook_candidates": ReviewNotebook(self.store).candidates(target),
        }

    def update(self, session_id, body, key, current_bundle):
        # Keep the existing recap request representation for persisted idempotency keys.
        payload = {"source_report_id": body.source_report_id, "revision": body.revision}
        if body.source_notebook_session_id:
            payload.update(
                source_notebook_session_id=body.source_notebook_session_id,
                source_notebook_revision=body.source_notebook_revision,
            )
        with self.store.tx() as db:
            scope = f"{self.store.owner}:practice-focus:{session_id}"
            cached = self.store._cached(db, scope, key, payload)
            if cached:
                return cached
            target = self.store._object(db, "session", session_id)
            if target["bundle_id"] != current_bundle or not target["clip_id"]:
                raise DomainError(409, "PRACTICE_INCOMPATIBLE", "Open a compatible video practice")
            if self.locked(session_id):
                raise DomainError(
                    409, "PRACTICE_ALREADY_STARTED", "Starting focus is fixed once review has started"
                )
            previous = self.record(session_id)
            if body.revision != (previous["revision"] if previous else 0):
                raise DomainError(
                    409, "PRACTICE_FOCUS_CHANGED", "Refresh this practice before changing its starting focus"
                )
            if self.settings.vision_mode == "real":
                media_context(self.store, self.settings, target["clip_id"])
            source = self.source(target, body.source_report_id) if body.source_report_id else None
            if body.source_notebook_session_id:
                source = ReviewNotebook(self.store).goal_source(
                    target, body.source_notebook_session_id, body.source_notebook_revision
                )
            record = {
                "id": "practice-focus:" + session_id,
                "session_id": session_id,
                "clip_id": target["clip_id"],
                "bundle_id": target["bundle_id"],
                "revision": body.revision + 1,
                "created_at": time.time(),
                "source_report_id": body.source_report_id,
                "report_digest": source.get("report_digest") if source else None,
                "label_ids": source["label_ids"] if source else [],
                "scope_only": True,
            }
            if body.source_notebook_session_id:
                record.update(
                    source_kind="notebook",
                    source_notebook_session_id=body.source_notebook_session_id,
                    source_notebook_revision=body.source_notebook_revision,
                    source_digest=source["source_digest"],
                )
            record["digest"] = digest(record)
            self.store._put(db, "practice_focus", record)
            result = {"revision": record["revision"], "label_ids": record["label_ids"], "scope_only": True}
            self.store._remember(db, scope, key, payload, result)
            return result
