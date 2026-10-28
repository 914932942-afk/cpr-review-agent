"""Freeze the accepted M1 workspace, back up SQLite, and smoke-test an extracted restore."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from cpr_agent.config import Settings

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "releases/stable-m1-20261022-v1"
RUNTIME = Path("data/m1-optimization-v2/runtime")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if RELEASE.exists():
        raise FileExistsError("Never overwrite a release")
    accepted = read(ROOT / "data/m1-optimization-v2/acceptance.json")
    assert accepted["passed"]
    for name, expected in accepted["source_hashes"].items():
        assert sha(ROOT / name) == expected, name
    model_path = ROOT / "data/public_cpr/web-integration-v1/bundle-manifest.json"
    model = read(model_path)
    for name, expected in model["pinned_sources_and_assets"].items():
        assert sha(ROOT / name) == expected, name
    with closing(sqlite3.connect(ROOT / RUNTIME / "business.sqlite3")) as db:
        assert (
            db.execute(
                "SELECT count(*) FROM runs WHERE status NOT IN ('completed','partially_completed','failed','cancelled','rejected')"
            ).fetchone()[0]
            == 0
        )
    RELEASE.mkdir(parents=True)
    stage = RELEASE / "snapshot"
    stage.mkdir()
    files = set()
    for folder in [
        "backend",
        "frontend/src",
        "frontend/dist",
        "scripts",
        "tests",
        "docs",
        "fixtures",
        "experiments/branch_pipeline_v1",
        "experiments/m1_fast_v1",
        "experiments/m1_stability_v3",
        "test-results/m1-optimization-v2",
    ]:
        files.update(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for folder in ["data/branch-integration-v1", "data/m1-stability-v1", "data/m1-optimization-v2"]:
        files.update(
            p
            for p in (ROOT / folder).rglob("*.json")
            if "runtime" not in p.parts and "integration-before" not in p.parts
        )
    for folder in [RUNTIME / "media", RUNTIME / "analysis"]:
        files.update(p for p in (ROOT / folder).rglob("*") if p.is_file())
    for name in model["pinned_sources_and_assets"]:
        p = ROOT / name
        if p.suffix in {".py", ".json"}:
            files.add(p)
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
        files.add(ROOT / name)
    files.add(model_path)
    for source in sorted(files):
        target = stage / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ["business.sqlite3", "checkpoints.sqlite3"]:
        target = stage / RUNTIME / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with (
            closing(sqlite3.connect(ROOT / RUNTIME / name)) as source,
            closing(sqlite3.connect(target)) as dest,
        ):
            source.backup(dest)
            assert dest.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    key = Settings.from_env().deepseek_api_key.encode()
    hashes = {}
    for p in stage.rglob("*"):
        if p.is_file():
            assert p.name != ".env"
            assert not key or key not in p.read_bytes(), "Configured key found in snapshot"
            hashes[p.relative_to(stage).as_posix()] = sha(p)
    archive = RELEASE / "application-and-demo.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name in hashes:
            zipped.write(stage / name, name)
    restored = RELEASE / "restore-check"
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        zipped.extractall(restored)
    assert all(sha(restored / name) == expected for name, expected in hashes.items())
    source_acceptance = read(ROOT / "test-results/m1-optimization-v2/http-acceptance.json")["cases"][0]
    smoke = r"""
import json,sys
from pathlib import Path
from fastapi.testclient import TestClient
from cpr_agent.api import create_app
from cpr_agent.config import Settings,ROOT
original=Path(sys.argv[1]); case=json.loads(sys.argv[2])
assert Path(sys.modules['cpr_agent.api'].__file__).is_relative_to(ROOT)
settings=Settings(data_dir=ROOT/'data/m1-optimization-v2/runtime',vision_mode='real',vision_pipeline='branches',
    ml_python=Path(sys.argv[3]),vision_manifest=original/'data/public_cpr/web-integration-v1/bundle-manifest.json',
    branch_manifest=ROOT/'data/m1-optimization-v2/pipeline-manifest.json',graph_version='cpr-review-12-v0.3-branches',llm_mode='disabled')
with TestClient(create_app(settings)) as c:
    assert c.get('/api/health').json()['llm']=='disabled'
    r=c.get('/api/runs/'+case['run_id']).json();assert r['status']=='completed'
    assert r['response']['prediction']['predicted_label_ids']==case['predicted_labels']
    assert c.get('/api/reports/'+case['report_id']+'/evidence').json()['status']=='available'
    assert len(c.get('/api/runs/'+case['run_id']+'/tools').json())==4
    assert c.get('/api/sessions/'+case['session_id']+'/history').status_code==200
    assert c.get('/api/reports/'+case['report_id']+'/export').status_code==200
    assert c.get('/api/assets/'+case['asset_id']+'/media',headers={'Range':'bytes=0-1023'}).status_code==206
print(json.dumps({'passed':True,'restored_application_imported':True,'source_evidence_revalidated':True,'new_vision_calls':0,'real_llm_calls':0}))
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CPR_", "DEEPSEEK_"))}
    env.update(PYTHONPATH=str(restored / "backend"), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            smoke,
            str(ROOT),
            json.dumps(source_acceptance),
            str(Settings.from_env().ml_python),
        ],
        cwd=restored,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    if completed.returncode:
        (RELEASE / "restore-error.txt").write_text(completed.stderr, encoding="utf-8")
        raise RuntimeError("Extracted restore smoke failed; see release restore-error.txt")
    verification = json.loads(completed.stdout.strip().splitlines()[-1])
    result = {
        "schema": "cpr-stable-m1-release-v1",
        "passed": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(ROOT),
        "runtime_relative": RUNTIME.as_posix(),
        "accepted_tests": accepted["tests_passed"],
        "archive_sha256": sha(archive),
        "archive_bytes": archive.stat().st_size,
        "files": hashes,
        "restore_smoke": verification,
        "sqlite_backup_verified": True,
        "extracted_hashes_verified": True,
        "credentials_included": False,
        "model_pins": model["pinned_sources_and_assets"],
        "model_and_environments": "Referenced in existing CODE tree; not duplicated. Restore into original root with services stopped.",
    }
    (RELEASE / "release.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (RELEASE / "RESTORE.txt").write_text(
        "Stable M1 baseline before module-slot integration. Keep snapshot/ and archive unchanged.\n"
        "To roll back: stop the service; first back up current files/data; restore archive into the original CODE root.\n"
        "Keep existing model assets and Python environments; validate model_pins and file hashes in release.json.\n"
        "No .env or API key is included. Use local configuration and keep DeepSeek disabled.\n"
        "Run scripts/start_m1_fast.ps1 (port 8003). Its restored runtime is data/m1-optimization-v2/runtime.\n"
        "restore-check/ is a separate writable extraction used for API/report/evidence/playback verification.\n"
        "This is a local rollback package with model references, not a self-contained cross-machine installer.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "files": len(hashes),
                "archive_mib": round(result["archive_bytes"] / 2**20, 2),
                "restore_smoke": verification,
            }
        )
    )


if __name__ == "__main__":
    main()
