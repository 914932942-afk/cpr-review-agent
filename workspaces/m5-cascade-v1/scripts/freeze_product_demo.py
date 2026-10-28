"""Freeze the accepted product demo and verify the application from an extracted archive."""

from __future__ import annotations

import argparse
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
from demo_preflight import MODEL, PIPELINE, ROOT, require, sha
from filelock import FileLock
from freeze_demo import release_path

RUNTIME = Path("data/product-demo-v2/runtime")
ACCEPTANCE = Path("data/product-demo-v2/acceptance.json")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(name):
    release = release_path(ROOT, name)
    accepted = read(ROOT / ACCEPTANCE)
    require(accepted["passed"], "Product acceptance has not passed")
    for rel, expected in accepted["source_hashes"].items():
        require(sha(ROOT / rel) == expected, "Accepted source changed: " + rel)
    runtime = ROOT / RUNTIME
    with FileLock(runtime / "runtime.lock", timeout=0):
        with closing(sqlite3.connect((runtime / "business.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
            require(
                db.execute(
                    "SELECT count(*) FROM runs WHERE status NOT IN ('completed','partially_completed','failed','rejected','cancelled')"
                ).fetchone()[0]
                == 0,
                "Finish active requests before freezing",
            )
        stage = release / "snapshot"
        stage.mkdir(parents=True)
        files = set()
        for area in [
            "backend",
            "frontend/src",
            "frontend/dist",
            "frontend/public",
            "scripts",
            "tests",
            "docs",
            "fixtures",
            "test-results/product-demo-v2",
            str(RUNTIME / "media"),
            str(RUNTIME / "analysis"),
        ]:
            files.update(p for p in (ROOT / area).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        for manifest, field in [(MODEL, "pinned_sources_and_assets"), (PIPELINE, "sources")]:
            files.add(ROOT / manifest)
            for rel, expected in read(ROOT / manifest)[field].items():
                source = (ROOT / rel).resolve()
                require(source.is_relative_to(ROOT) and sha(source) == expected, "Pinned dependency changed")
                if source.suffix in {".py", ".json"}:
                    files.add(source)
        for rel in [
            "README.md",
            "pyproject.toml",
            "requirements.lock",
            ".gitignore",
            ".env.example",
            "frontend/package.json",
            "frontend/package-lock.json",
            "frontend/index.html",
            "frontend/tsconfig.json",
            "frontend/vite.config.ts",
            str(ACCEPTANCE),
            "data/product-ui-v1/acceptance.json",
            "test-results/product-ui-v1/ui-final/acceptance.json",
        ]:
            files.add(ROOT / rel)
        for source in sorted(files):
            destination = stage / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for name in ["business.sqlite3", "checkpoints.sqlite3"]:
            target = stage / RUNTIME / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with (
                closing(sqlite3.connect((runtime / name).as_uri() + "?mode=ro", uri=True)) as source,
                closing(sqlite3.connect(target)) as dest,
            ):
                source.backup(dest)
                require(dest.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite backup failed")
    key = Settings.from_env().deepseek_api_key.encode()
    hashes = {}
    for path in stage.rglob("*"):
        if not path.is_file():
            continue
        require(
            path.name != ".env" and "__pycache__" not in path.parts, "Private environment or cache included"
        )
        require(not key or key not in path.read_bytes(), "Configured credential detected")
        hashes[path.relative_to(stage).as_posix()] = sha(path)
    archive = release / "application-and-demo.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zipped:
        for name in hashes:
            zipped.write(stage / name, name)
    restored = release / "restore-check"
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.testzip() is None, "Archive integrity failed")
        zipped.extractall(restored)
    require(
        all(sha(restored / name) == expected for name, expected in hashes.items()), "Extracted file changed"
    )
    case = read(ROOT / "test-results/product-demo-v2/browser/acceptance.json")
    smoke = r"""
import json,sys,sqlite3
from pathlib import Path
from fastapi.testclient import TestClient
import cpr_agent.api
from cpr_agent.config import ROOT,Settings
original=Path(sys.argv[1]);case=json.loads(sys.argv[2])
assert ROOT==Path.cwd() and Path(cpr_agent.api.__file__).is_relative_to(ROOT)
data=ROOT/'data/product-demo-v2/runtime'
settings=Settings(data_dir=data,vision_mode='real',vision_pipeline='branches',ml_python=Path(sys.argv[3]),vision_manifest=original/'data/public_cpr/web-integration-v1/bundle-manifest.json',branch_manifest=ROOT/'data/module-slots-v1/pipeline-manifest.json',context_mode='bound-v1',llm_mode='disabled')
with TestClient(cpr_agent.api.create_app(settings)) as c:
    assert c.get('/api/health').json()['llm']=='disabled'
    run=c.get('/api/runs/'+case['summary_run_id']).json()
    assert run['status']=='completed' and run['response']['prediction']==case['prediction']
    assert c.get('/api/reports/'+run['report_id']+'/explanation').json()['status']=='available'
    assert c.get('/api/reports/'+run['report_id']+'/evidence').json()['status']=='available'
    assert c.get('/api/reports/'+run['report_id']+'/export').status_code==200
    assert len(c.get('/api/runs/'+case['analysis_run_id']+'/tools').json())==4
    assert c.get('/api/assets/'+case['workspace']['asset_id']+'/media',headers={'Range':'bytes=0-1023'}).status_code==206
    assert c.get('/api/sessions/'+case['workspace']['session_id']+'/history').json()['latest_recap']['run_id']==case['summary_run_id']
    assert 'Your practice, in focus' in c.get('/').text
with sqlite3.connect(data/'business.sqlite3') as db:
    assert db.execute('select count(*) from tools').fetchone()[0]==4
print(json.dumps({'passed':True,'actual_extracted_application_imported':True,'copied_media_and_evidence_revalidated':True,'saved_summary_and_export_passed':True,'new_vision_calls':0,'paid_llm_calls':0}))
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CPR_", "DEEPSEEK_"))}
    env.update(PYTHONPATH=str(restored / "backend"), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    process = subprocess.run(
        [sys.executable, "-c", smoke, str(ROOT), json.dumps(case), str(Settings.from_env().ml_python)],
        cwd=restored,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    if process.returncode:
        (release / "restore-error.txt").write_text(process.stderr, encoding="utf-8")
    require(process.returncode == 0, "Extracted application restore failed; not an accepted release")
    verification = json.loads(process.stdout.strip().splitlines()[-1])
    record = {
        "schema": "cpr-product-demo-release-v2",
        "passed": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "files": hashes,
        "archive_sha256": sha(archive),
        "archive_bytes": archive.stat().st_size,
        "restore_check": verification,
        "model_pins_referenced": read(ROOT / MODEL)["pinned_sources_and_assets"],
        "sqlite_backup_verified": True,
        "credentials_included": False,
        "environments_included": False,
        "runtime": RUNTIME.as_posix(),
        "restore_scope": "Local overlay restore into original CODE root with existing model assets and environments retained; not a standalone installer.",
    }
    (release / "release.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (release / "RESTORE.txt").write_text(
        """Restore product demo v2

1. Stop the product demo service before restoring; back up current code and runtime first.
2. Retain CODE/data/public_cpr model assets, the Web .venv and the separate ML environment.
3. Extract application-and-demo.zip into the original CODE root as an overlay.
4. Keep your local .env separately. No real key, virtual environment or model weight is bundled.
5. Run .\\scripts\\start_product_demo.ps1 and open http://127.0.0.1:8007 . Default LLM mode is disabled.
6. Choose video.mp4 in Your practices. Open Latest recap or Saved recap in Review history.
7. To enable live questions, run the same script with -EnableDeepSeek using your local key.
8. Verify model/module pins with scripts/demo_preflight.py. Model/environment paths must remain valid.

The extracted application was actually tested with its copied media, saved evidence, recap and export.
The original model assets are referenced. This is a local rollback package, not a cross-machine installer.
This public sample is a known engineering regression, not a new accuracy or clinical assessment.
M2-M5 remain baseline adapters reserved for teammate improvements.
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "release": str(release),
                "files": len(hashes),
                "archive_MiB": round(archive.stat().st_size / 2**20, 2),
                "restore_check": verification,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="stable-product-20261023-v2")
    freeze(parser.parse_args().name)


if __name__ == "__main__":
    main()
