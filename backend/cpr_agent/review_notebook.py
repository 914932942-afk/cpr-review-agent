"""Personal annotations and explicitly chosen goals; never model or training labels."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import Field, model_validator

from .models import CANDIDATE_LABELS, StrictModel
from .store import DomainError, digest


class NotebookItem(StrictModel):
    label_id: int = Field(strict=True)
    status: Literal["to_review", "reviewed", "next_focus"]


class NotebookUpdate(StrictModel):
    revision: int = Field(ge=0, strict=True)
    note: str = Field(default="", max_length=2000)
    items: list[NotebookItem] = Field(default_factory=list, max_length=11)

    @model_validator(mode="after")
    def valid_categories(self):
        ids = [item.label_id for item in self.items]
        if len(ids) != len(set(ids)) or any(i not in CANDIDATE_LABELS for i in ids):
            raise ValueError("Choose each supported category at most once")
        return self


class ReviewNotebook:
    def __init__(self, store):
        self.store = store

    def view(self, session_id):
        session = self.store.get("session", session_id)
        try:
            record = self.store.get("review_notebook", "notebook:" + session_id)
        except DomainError as exc:
            if exc.status != 404:
                raise
            return {"session_id": session_id, "revision": 0, "note": "", "items": [], "updated_at": None}
        if (
            record.get("digest") != digest({k: v for k, v in record.items() if k != "digest"})
            or record.get("session_id") != session_id
            or record.get("clip_id") != session["clip_id"]
            or record.get("bundle_id") != session["bundle_id"]
        ):
            raise DomainError(410, "NOTEBOOK_UNAVAILABLE", "Saved notes could not be verified")
        NotebookUpdate.model_validate({k: record[k] for k in ("revision", "note", "items")})
        return record

    def update(self, session_id, body, key):
        payload = body.model_dump(mode="json")
        with self.store.tx() as db:
            scope = f"{self.store.owner}:notebook:{session_id}"
            cached = self.store._cached(db, scope, key, payload)
            if cached:
                return cached
            session = self.store._object(db, "session", session_id)
            previous = self.view(session_id)
            if body.revision != previous["revision"]:
                raise DomainError(
                    409,
                    "NOTEBOOK_CHANGED",
                    "Notes changed elsewhere. Reload the saved version before editing",
                )
            record = {
                "id": "notebook:" + session_id,
                "session_id": session_id,
                "clip_id": session["clip_id"],
                "bundle_id": session["bundle_id"],
                "revision": body.revision + 1,
                "updated_at": time.time(),
                "note": body.note,
                "items": payload["items"],
                "user_annotations_only": True,
            }
            record["digest"] = digest(record)
            self.store._put(db, "review_notebook", record)
            self.store._remember(db, scope, key, payload, record)
            return record

    def goal_source(self, target, source_session_id, revision):
        previous = self.store.get("session", source_session_id)
        notebook = self.view(source_session_id)
        labels = sorted(i["label_id"] for i in notebook["items"] if i["status"] == "next_focus")
        if (
            previous["id"] == target["id"]
            or previous["clip_id"] == target["clip_id"]
            or previous["created_at"] >= target["created_at"]
            or previous["bundle_id"] != target["bundle_id"]
            or revision != notebook["revision"]
            or not labels
        ):
            raise DomainError(
                409, "NOTEBOOK_GOALS_CHANGED", "Choose current goals from an earlier compatible practice"
            )
        clip = self.store.get("clip", previous["clip_id"])
        asset = self.store.get("asset", clip["asset_id"])
        current_clip = self.store.get("clip", target["clip_id"])
        current_asset = self.store.get("asset", current_clip["asset_id"])
        return {
            "kind": "notebook",
            "report_id": None,
            "session_id": source_session_id,
            "video_name": asset["name"],
            "created_at": notebook["updated_at"],
            "label_ids": labels,
            "notebook_revision": notebook["revision"],
            "source_digest": notebook["digest"],
            "same_video_file": asset["sha256"] == current_asset["sha256"],
        }

    def candidates(self, target):
        result = []
        records = sorted(
            self.store.list_objects("review_notebook"), key=lambda n: n["updated_at"], reverse=True
        )
        for record in records:
            try:
                source = self.goal_source(target, record["session_id"], record["revision"])
                result.append({k: v for k, v in source.items() if k != "source_digest"})
            except (DomainError, ValueError, KeyError, TypeError):
                continue
            if len(result) == 3:
                break
        return result
