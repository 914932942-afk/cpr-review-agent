"""Freeze the accepted demo, including consistent SQLite backups and an actual restore check."""

from __future__ import annotations

import argparse
import json
import os
import re
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

RUNTIME = Path("data/demo-acceptance-v1/runtime")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def release_path(root, name):
    require(re.fullmatch(r"[a-z][a-z0-9-]{2,79}", name) is not None, "Invalid release name")
    path = root / "releases" / name
    require(not path.exists(), "Never overwrite a release; choose a new name")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="stable-mainline-20261022-v1")
    args = parser.parse_args()
    release = release_path(ROOT, args.name)
    accepted = read(ROOT / "data/demo-acceptance-v1/acceptance.json")
    require(accepted["passed"], "Demo acceptance has not passed")
    for rel, expected in accepted["source_hashes"].items():
        require(sha(ROOT / rel) == expected, "Accepted source changed: " + rel)
    with closing(
        sqlite3.connect((ROOT / RUNTIME / "business.sqlite3").as_uri() + "?mode=ro", uri=True)
    ) as db:
        require(
            db.execute(
                "SELECT count(*) FROM runs WHERE status NOT IN "
                "('completed','partially_completed','failed','rejected','cancelled')"
            ).fetchone()[0]
            == 0,
            "Finish or cancel active demo requests before freezing",
        )
    stage = release / "snapshot"
    stage.mkdir(parents=True)
    files = set()
    for area in [
        "backend",
        "frontend/src",
        "frontend/dist",
        "scripts",
        "tests",
        "docs",
        "fixtures",
        "test-results/demo-acceptance-v1",
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
    for name in [
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
        "data/demo-acceptance-v1/acceptance.json",
        "test-results/m1-stability-v1/oracle/raw26-with-scores.json",
    ]:
        files.add(ROOT / name)
    for source in files:
        destination = stage / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name in ["business.sqlite3", "checkpoints.sqlite3"]:
        destination = stage / RUNTIME / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect((ROOT / RUNTIME / name).as_uri() + "?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(destination)) as target:
                source.backup(target)
                require(
                    target.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite backup failed"
                )
    key = Settings.from_env().deepseek_api_key.encode()
    hashes = {}
    for path in stage.rglob("*"):
        if path.is_file():
            require(path.name != ".env", "Real environment file must not be archived")
            require(
                not key or key not in path.read_bytes(), "Configured credential detected; archive not created"
            )
            hashes[path.relative_to(stage).as_posix()] = sha(path)
    archive = release / "application-and-demo.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zipped:
        for name in hashes:
            zipped.write(stage / name, name)
    restored = release / "restore-check"
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.testzip() is None, "Archive integrity check failed")
        zipped.extractall(restored)
    require(all(sha(restored / p) == expected for p, expected in hashes.items()), "Restored files changed")
    case = read(ROOT / "test-results/demo-acceptance-v1/http/acceptance.json")
    # Verify the extracted application, not the editable installation in the current project.
    smoke = r"""
import json,sys
from pathlib import Path
from fastapi.testclient import TestClient
import cpr_agent.api
from cpr_agent.config import ROOT,Settings
assert ROOT==Path.cwd() and Path(cpr_agent.api.__file__).is_relative_to(ROOT)
original=Path(sys.argv[1]); case=json.loads(sys.argv[2])
settings=Settings(data_dir=ROOT/'data/demo-acceptance-v1/runtime',vision_mode='real',vision_pipeline='branches',
    ml_python=Path(sys.argv[3]),vision_manifest=original/'data/public_cpr/web-integration-v1/bundle-manifest.json',
    branch_manifest=ROOT/'data/module-slots-v1/pipeline-manifest.json',context_mode='bound-v1',llm_mode='disabled')
with TestClient(cpr_agent.api.create_app(settings)) as c:
    assert c.get('/api/health').json()['llm']=='disabled'
    r=c.get('/api/runs/'+case['final_summary_run_id']).json(); assert r['status']=='completed'
    assert r['response']['prediction']==case['prediction']
    assert c.get('/api/reports/'+r['report_id']+'/explanation').json()['status']=='available'
    assert c.get('/api/reports/'+r['report_id']+'/export').status_code==200
    assert c.get('/api/runs/'+case['analysis_run_id']+'/tools').json().__len__()==4
    assert c.get('/api/runs/'+case['cancelled_run_id']).json()['status']=='cancelled'
    assert c.get('/api/runs/'+case['failed_decode_run_id']).json()['status']=='failed'
    assert c.get('/api/assets/'+case['workspace']['asset_id']+'/media',headers={'Range':'bytes=0-1023'}).status_code==206
    assert c.get('/api/sessions/'+case['workspace']['session_id']+'/history').status_code==200
print(json.dumps({'passed':True,'actual_extracted_application_imported':True,'saved_evidence_revalidated':True,
                  'new_vision_calls':0,'real_llm_calls':0}))
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
    require(process.returncode == 0, "Extracted restore check failed; release is not accepted")
    restore_result = json.loads(process.stdout.strip())
    record = {
        "schema": "cpr-stable-mainline-demo-v1",
        "passed": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "files": hashes,
        "archive_sha256": sha(archive),
        "archive_bytes": archive.stat().st_size,
        "restore_check": restore_result,
        "model_pins_referenced": read(ROOT / MODEL)["pinned_sources_and_assets"],
        "credentials_included": False,
        "environments_included": False,
        "restore_scope": "Extracted code replays saved results with original pinned model assets. Full inference restore overlays the original CODE root with its models and environments retained.",
    }
    (release / "release.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (release / "RESTORE.txt").write_text(
        "Restore the accepted CPR demo\n\n"
        "1. Stop the affected CPR services before restoring files.\n"
        "2. Keep the existing CODE/data/public_cpr assets, .venv and ML environment. They are referenced, not bundled.\n"
        "3. Back up current files. Extract application-and-demo.zip into the original CODE root. This is an overlay restore.\n"
        "4. Keep your .env separately; it is deliberately excluded. Default demo startup disables DeepSeek.\n"
        "5. Run .\\scripts\\start_demo.ps1 and open http://127.0.0.1:8005 .\n"
        "6. Choose demo-training.mp4 in Saved session, reopen the saved summary and check its evidence.\n"
        "7. The restore-check folder proves extracted-code playback with original model assets. It is not a standalone model bundle.\n"
        "The public clip is a known engineering regression, not an independent accuracy/clinical test. M2-M5 remain baselines.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "release": str(release),
                "files": len(hashes),
                "archive_MiB": round(archive.stat().st_size / 1024**2, 2),
                "restore_check": restore_result,
            }
        )
    )


if __name__ == "__main__":
    main()
