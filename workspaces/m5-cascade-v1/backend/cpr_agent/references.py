"""Read-only, source-bound reference retrieval; never a CV result or medical recommendation."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path

from .config import ROOT
from .context_contracts import ContextSnapshot
from .models import CANDIDATE_LABELS
from .store import DomainError, digest
from .vision import VisionError

SCHEMA = "cpr-reference-catalog-v1"
TOOL = "reference.search"
REVISION = "b8667b8fd6a70ac63cc09fc6b763a9459b7c160a"
NOTICE = "Dataset labels describe research samples, not professional certification or proof of correct CPR. References cover whole clips and do not diagnose your practice."


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ReferenceError(ValueError):
    pass


class ReferenceCatalogue:
    def __init__(self, path: Path | None, root: Path = ROOT):
        self.path, self.root = path, root.resolve()

    def _path(self, relative, *, media=False):
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ReferenceError("REFERENCE_PATH")
        path = (self.root / relative).resolve()
        boundary = self.root / "data/public_cpr/rgb-acceptance-v1/records"
        if not path.is_relative_to(boundary) or (media and path.suffix != ".mp4"):
            raise ReferenceError("REFERENCE_PATH")
        return path

    def load(self):
        try:
            if self.path is None or not self.path.resolve().is_relative_to(self.root):
                raise ReferenceError("REFERENCE_DISABLED")
            payload = self.path.read_bytes()
            if len(payload) > 262144:
                raise ReferenceError("REFERENCE_CATALOG_SIZE")
            value = json.loads(payload)
            if value.get("dataset_revision") != REVISION:
                raise ReferenceError("REFERENCE_REVISION_CHANGED")
            if (
                value.get("dataset_url") != "https://huggingface.co/datasets/ShunliWang/CPR-Coach"
                or value.get("repository_url") != "https://github.com/Shunli-Wang/CPR-Coach"
            ):
                raise ReferenceError("REFERENCE_SOURCE_INVALID")
            entries = value["entries"]
            if value["schema"] != SCHEMA or not 1 <= len(entries) <= 64:
                raise ReferenceError("REFERENCE_CATALOG_INVALID")
            if len({e["id"] for e in entries}) != len(entries):
                raise ReferenceError("REFERENCE_CATALOG_INVALID")
            annotation = (self.root / value["annotation_rel"]).resolve()
            if (
                annotation != self.root / "data/public_cpr/assets/ann.tar.gz"
                or sha(annotation) != value["annotation_sha256"]
            ):
                raise ReferenceError("REFERENCE_ANNOTATION_CHANGED")
            with tarfile.open(annotation) as tar:
                member = tar.getmember("./ann/trainlist.txt")
                if member.size > 2000000:
                    raise ReferenceError("REFERENCE_ANNOTATION_SIZE")
                lines = set(tar.extractfile(member).read().decode("utf-8").splitlines())
            for entry in entries:
                if (
                    not re.fullmatch(r"cpr-c[0-9]{2}-r[0-9]{2}-ch0", entry["id"])
                    or entry["view"] != "ch0"
                    or type(entry["label_id"]) is not int
                    or entry["label_id"] not in {0, *CANDIDATE_LABELS}
                ):
                    raise ReferenceError("REFERENCE_ENTRY_INVALID")
                if (
                    entry["annotation_member"] != "./ann/trainlist.txt"
                    or entry["annotation_line"] not in lines
                ):
                    raise ReferenceError("REFERENCE_ANNOTATION_MISMATCH")
                parts = entry["annotation_line"].split()
                if (
                    int(parts[1]) != entry["frames"]
                    or [int(n) for n in parts[2:]] != [entry["label_id"]]
                    or not 0 < entry["fps"] <= 120
                ):
                    raise ReferenceError("REFERENCE_LABEL_MISMATCH")
            return value, hashlib.sha256(payload).hexdigest()
        except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
            raise ReferenceError("REFERENCE_CATALOG_UNAVAILABLE") from exc

    def validate_entry(self, entry):
        try:
            manifest = self._path(entry["manifest_rel"])
            if sha(manifest) != entry["manifest_sha256"]:
                raise ReferenceError("REFERENCE_MANIFEST_CHANGED")
            value = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                value["original_labels"] != [entry["label_id"]]
                or value.get("dataset_revision") != REVISION
                or value["record_id"] != entry["record_id"]
                or value["video_sha256"] != entry["media_sha256"]
                or value["annotation_frame_dir"] != entry["annotation_line"].split()[0]
                or value["label_names"] != [entry["dataset_label"]]
                or value["view"] != entry["view"]
                or value["video_validation"]["decoded_frames"] != entry["frames"]
                or value["video_validation"]["fps"] != entry["fps"]
            ):
                raise ReferenceError("REFERENCE_MANIFEST_MISMATCH")
            media = self._path(entry["media_rel"], media=True)
            if sha(media) != entry["media_sha256"]:
                raise ReferenceError("REFERENCE_MEDIA_CHANGED")
            return media
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ReferenceError("REFERENCE_UNAVAILABLE") from exc

    @staticmethod
    def public(entry):
        return {
            k: entry[k] for k in ["id", "record_id", "label_id", "dataset_label", "view", "frames", "fps"]
        } | {
            "duration_seconds": entry["frames"] / entry["fps"],
            "kind": "normal_label" if entry["label_id"] == 0 else "error_label",
        }

    def search(self, labels, exclude_sha=None):
        requested = sorted(set(labels))
        result = {
            "schema": "cpr-reference-lookup-v1",
            "tool_id": TOOL,
            "requested_label_ids": requested,
            "status": "unavailable",
            "catalog_sha256": None,
            "entry_ids": [],
            "entry_bindings": {},
            "unavailable_label_ids": requested,
            "notice": NOTICE,
        }
        try:
            catalog, checksum = self.load()
            chosen = []
            missing = []
            for label in [0, *requested]:
                candidates = [
                    e
                    for e in catalog["entries"]
                    if e["label_id"] == label and e["media_sha256"] != exclude_sha
                ]
                found = None
                for entry in candidates:
                    try:
                        self.validate_entry(entry)
                        found = entry
                        break
                    except ReferenceError:
                        continue
                if found:
                    chosen.append(found)
                elif label != 0:
                    missing.append(label)
            result.update(
                status="available" if chosen else "empty",
                catalog_sha256=checksum,
                entry_ids=[e["id"] for e in chosen],
                entry_bindings={e["id"]: e["media_sha256"] for e in chosen},
                unavailable_label_ids=missing,
            )
        except ReferenceError:
            pass
        return result

    def verify(self, receipt, only=None):
        try:
            catalog, checksum = self.load()
            if checksum != receipt["catalog_sha256"]:
                raise ReferenceError("REFERENCE_CATALOG_CHANGED")
            by_id = {e["id"]: e for e in catalog["entries"]}
            items, invalid = [], []
            identities = (
                receipt["entry_ids"] if only is None else ([only] if only in receipt["entry_ids"] else [])
            )
            for identity in identities:
                entry = by_id[identity]
                try:
                    if entry["label_id"] not in {0, *receipt["requested_label_ids"]}:
                        raise ReferenceError("REFERENCE_SCOPE_CHANGED")
                    self.validate_entry(entry)
                    if receipt["entry_bindings"].get(identity) != entry["media_sha256"]:
                        raise ReferenceError("REFERENCE_BINDING_CHANGED")
                    items.append(self.public(entry))
                except ReferenceError:
                    invalid.append(identity)
            return {
                "status": "available" if items else "unavailable",
                "items": items,
                "default_entry_id": items[0]["id"] if items else None,
                "unavailable_entry_ids": invalid,
                "requested_label_ids": receipt["requested_label_ids"],
                "unavailable_label_ids": receipt["unavailable_label_ids"],
                "notice": NOTICE,
                "dataset": catalog["dataset"],
                "dataset_url": catalog["dataset_url"],
                "repository_url": catalog["repository_url"],
                "dataset_revision": catalog["dataset_revision"],
                "license_note": catalog["license_note"],
            }
        except (ReferenceError, KeyError, TypeError):
            return {
                "status": "unavailable",
                "items": [],
                "default_entry_id": None,
                "notice": NOTICE,
                "requested_label_ids": receipt.get("requested_label_ids", []),
                "unavailable_label_ids": receipt.get("requested_label_ids", []),
            }

    def media(self, receipt, identity):
        verified = self.verify(receipt, only=identity)
        if identity not in {e["id"] for e in verified["items"]}:
            raise DomainError(410, "REFERENCE_UNAVAILABLE", "The requested reference is unavailable")
        catalog, _ = self.load()
        return self.validate_entry(next(e for e in catalog["entries"] if e["id"] == identity))


def lookup_id(run_id):
    return "reference:" + run_id


def persisted_lookup(store, run_id):
    try:
        return store.get("reference_lookup", lookup_id(run_id))
    except DomainError as exc:
        if exc.status != 404:
            raise
        return None


def prepare_lookup(store, settings, run, snapshot, labels):
    existing = persisted_lookup(store, run["id"])
    binding = {
        "run_id": run["id"],
        "session_id": run["session_id"],
        "clip_id": run["clip_id"],
        "snapshot_id": snapshot.snapshot_id,
        "source_report_id": snapshot.source.report_id if snapshot.source else None,
    }
    if existing is not None:
        if any(existing.get(k) != v for k, v in binding.items()) or existing["requested_label_ids"] != sorted(
            set(labels)
        ):
            raise ReferenceError("REFERENCE_REQUEST_CHANGED")
        return existing
    result = ReferenceCatalogue(settings.reference_catalog).search(labels, snapshot.video_sha256)
    return store.put("reference_lookup", {"id": lookup_id(run["id"]), **binding, **result})


def verify_lookup(store, settings, run_id, entry_id=None):
    run = store.get_run(run_id)
    receipt = persisted_lookup(store, run_id)
    if not receipt:
        return {"status": "not_requested", "items": []}
    if run["status"] not in {"completed", "partially_completed"}:
        return {"status": "unavailable", "items": [], "notice": NOTICE}
    try:
        turn = store.get("turn", run["turn_id"])
        snapshot = ContextSnapshot.model_validate(turn["context_snapshot"])
        goal = turn.get("resolved_goal") or {}
        if (
            goal.get("intent") != "show_example"
            or sorted(set(goal.get("target_label_ids", []))) != receipt["requested_label_ids"]
        ):
            raise ReferenceError("REFERENCE_SCOPE_CHANGED")
        if snapshot.video_sha256 in receipt["entry_bindings"].values():
            raise ReferenceError("REFERENCE_SELF_COMPARISON")
        if snapshot.snapshot_id != digest(snapshot.model_dump(exclude={"snapshot_id"}, mode="json")) or any(
            receipt.get(k) != v
            for k, v in {
                "run_id": run_id,
                "session_id": run["session_id"],
                "clip_id": run["clip_id"],
                "snapshot_id": snapshot.snapshot_id,
                "source_report_id": snapshot.source.report_id if snapshot.source else None,
            }.items()
        ):
            raise ReferenceError("REFERENCE_CONTEXT_CHANGED")
        if snapshot.source:
            from .review_context import ContextBinder

            ContextBinder(store, settings).prediction(snapshot.source, run["clip_id"])
        elif run["clip_id"]:
            from .vision import media_context

            media_context(store, settings, run["clip_id"])
        return ReferenceCatalogue(settings.reference_catalog).verify(receipt, only=entry_id)
    except (ValueError, OSError, KeyError, TypeError, AttributeError, DomainError, VisionError):
        return {
            "status": "unavailable",
            "items": [],
            "notice": "The practice source or reference could not be verified. Reopen a valid saved review before comparing.",
        }
