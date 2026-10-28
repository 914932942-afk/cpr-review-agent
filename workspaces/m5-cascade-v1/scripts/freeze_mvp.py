"""Snapshot the accepted local MVP and verify an extracted restore, excluding credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Never overwrite a frozen release")
    source_record = ROOT / "data/review-integration-v1/acceptance.json"
    accepted = json.loads(source_record.read_text(encoding="utf-8"))
    assert accepted["passed"]
    for name, expected in accepted["sources"].items():
        assert sha(ROOT / name) == expected, name
    manifest = ROOT / "data/public_cpr/web-integration-v1/bundle-manifest.json"
    model = json.loads(manifest.read_text(encoding="utf-8"))
    for name, expected in model["pinned_sources_and_assets"].items():
        assert sha(ROOT / name) == expected, name
    with closing(sqlite3.connect(ROOT / "data/business.sqlite3")) as db:
        active = db.execute(
            "SELECT count(*) FROM runs WHERE status NOT IN "
            "('completed','partially_completed','failed','cancelled','rejected')"
        ).fetchone()[0]
        assert active == 0, "Wait for active requests before freezing"
    output.mkdir(parents=True)
    stage = output / "snapshot"
    stage.mkdir()
    files = []
    for folder in ["backend", "frontend/src", "frontend/dist", "scripts", "tests", "docs"]:
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for folder in [
        "data/media",
        "data/analysis",
        "data/review-integration-v1",
        "data/llm-integration-v1",
        "test-results/review-integration-v1",
        "test-results/web-real-v1",
    ]:
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "before" not in p.parts)
    for name in [
        "README.md",
        "pyproject.toml",
        "requirements.lock",
        ".gitignore",
        ".env.example",
        "frontend/package.json",
        "frontend/package-lock.json",
        "frontend/tsconfig.json",
        "frontend/vite.config.ts",
        "frontend/index.html",
    ]:
        files.append(ROOT / name)
    files.append(manifest)
    for path in sorted(set(files)):
        relative = path.relative_to(ROOT)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in ["business.sqlite3", "checkpoints.sqlite3"]:
        target = stage / "data" / name
        with (
            closing(sqlite3.connect(ROOT / "data" / name)) as source,
            closing(sqlite3.connect(target)) as dest,
        ):
            source.backup(dest)
            assert dest.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    archive = output / "application-and-demo.zip"
    hashes = {p.relative_to(stage).as_posix(): sha(p) for p in stage.rglob("*") if p.is_file()}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name in hashes:
            zipped.write(stage / name, name)
    with tempfile.TemporaryDirectory(prefix="cpr-restore-", dir=output) as restored:
        with zipfile.ZipFile(archive) as zipped:
            assert zipped.testzip() is None
            zipped.extractall(restored)
        assert all(sha(Path(restored) / name) == expected for name, expected in hashes.items())
        with closing(sqlite3.connect(Path(restored) / "data/business.sqlite3")) as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            for row in db.execute("SELECT body FROM objects WHERE kind='asset'"):
                asset = json.loads(row[0])
                if asset.get("storage_rel"):
                    assert sha(Path(restored) / "data" / asset["storage_rel"]) == asset["sha256"]
    record = {
        "schema": "cpr-frozen-mvp-release-v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "accepted_tests": accepted["tests_passed"],
        "archive_sha256": sha(archive),
        "archive_bytes": archive.stat().st_size,
        "extracted_restore_hashes_verified": True,
        "sqlite_integrity_verified": True,
        "llm_mode": "disabled",
        "credentials_included": False,
        "model_assets": "Referenced by immutable 145-pin manifest in the existing CODE tree; not duplicated",
        "source_root": str(ROOT),
        "files": hashes,
        "pinned_sources_and_assets": model["pinned_sources_and_assets"],
    }
    (output / "release.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "RESTORE.txt").write_text(
        "Frozen accepted MVP. Keep this directory unchanged.\n"
        "application-and-demo.zip contains application source, built UI, database backups and demo media/results.\n"
        "It excludes .env/API keys, virtual environments, node_modules and large model assets.\n"
        "Restore application files into the original CODE tree only while its service is stopped.\n"
        "Keep the existing ML environment and verify all model references against release.json.\n"
        "The snapshot/ directory is the exact uncompressed copy. Do not start two services against the same data directory.\n"
        "Use your local .env with CPR_LLM_MODE=disabled and scripts/start.ps1.\n",
        encoding="utf-8",
    )
    print(json.dumps({k: record[k] for k in ["passed", "archive_bytes", "archive_sha256", "accepted_tests"]}))


if __name__ == "__main__":
    main()
