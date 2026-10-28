"""Download/install the frozen team resource pack; credentials never reach stdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspaces/m5-cascade-v1"
META = ROOT / "team-release.json"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def github_token() -> str | None:
    # Uses the same GitHub credential helper as an authenticated HTTPS clone.
    try:
        result = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n", text=True,
            capture_output=True, timeout=30,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"},
        )
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        return fields.get("password") if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def download(meta: dict) -> Path:
    import httpx

    target = ROOT / "downloads" / meta["asset_name"]
    target.parent.mkdir(exist_ok=True)
    if target.is_file() and sha(target) == meta["sha256"]:
        return target
    token = github_token()
    if not token:
        raise RuntimeError(
            "Private Release download requires your own GitHub Git credentials. "
            "Alternatively download the ZIP in your browser from " + meta["release_url"]
            + " and rerun setup_team.ps1 -AssetsZip <ZIP path>."
        )
    headers = {"Accept": "application/octet-stream", "Authorization": "Bearer " + token,
               "User-Agent": "CPR-Team-Setup"}
    api = meta["asset_api_url"]
    if not api.startswith("https://api.github.com/repos/914932942-afk/cpr-review-agent/releases/assets/"):
        raise RuntimeError("Unexpected asset API destination")
    partial = target.with_suffix(".zip.part")
    with httpx.Client(timeout=120, trust_env=False, follow_redirects=False) as client:
        with client.stream("GET", api, headers=headers) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location", "")
            else:
                location = None
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        stream.write(chunk)
        if location is not None:
            from urllib.parse import urlsplit
            parsed = urlsplit(location)
            if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".githubusercontent.com"):
                raise RuntimeError("Unexpected GitHub asset redirect")
            # Never forward the GitHub token to the signed storage URL.
            with client.stream("GET", location) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        stream.write(chunk)
    if sha(partial) != meta["sha256"]:
        raise RuntimeError("Downloaded resource pack checksum mismatch; pack was not installed")
    partial.replace(target)
    return target


def install(path: Path, meta: dict) -> None:
    from zipfile import ZipFile

    if sha(path) != meta["sha256"]:
        raise RuntimeError("ZIP checksum mismatch; use the matching Release asset")
    with ZipFile(path) as archive:
        entries = json.loads(archive.read("TEAM-ASSETS.json"))["files"]
        if len(entries) != len(set(entries)):
            raise RuntimeError("Duplicate archive paths")
        allowed = ("workspaces/m5-cascade-v1/data/", "data/mainline-delivery-v1/", "data/demo-cases-v1/controls/")
        # Validate every destination before changing any existing file.
        for name, digest in entries.items():
            destination = (ROOT / name).resolve()
            if not destination.is_relative_to(ROOT) or not name.startswith(allowed) or "\\" in name:
                raise RuntimeError("Invalid resource destination")
            if destination.exists() and (not destination.is_file() or sha(destination) != digest):
                raise RuntimeError("Existing local data differs; install into a fresh clone: " + name)
        for index, (name, digest) in enumerate(entries.items(), 1):
            destination = ROOT / name
            if not destination.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            if sha(destination) != digest:
                raise RuntimeError("Extracted file checksum mismatch: " + name)
            if index % 100 == 0:
                print(f"Installed {index}/{len(entries)} resource files", flush=True)
    (ROOT / ".team-assets-installed.json").write_text(json.dumps({"sha256": meta["sha256"]}) + "\n")


def verify() -> dict:
    pipeline_path = WORKSPACE / "data/m5-cascade-v1/pipeline-manifest.json"
    bundle_path = WORKSPACE / "data/public_cpr/web-integration-v1/bundle-manifest.json"
    checked = 0
    for path, key in [(pipeline_path, "sources"), (bundle_path, "pinned_sources_and_assets")]:
        value = json.loads(path.read_text(encoding="utf-8"))
        for name, expected in value[key].items():
            candidate = WORKSPACE / name
            if not candidate.is_file() or sha(candidate) != expected:
                raise RuntimeError("Missing or changed frozen resource: " + name)
            checked += 1
    cases = json.loads((ROOT / "data/mainline-delivery-v1/demo-cases.json").read_text(encoding="utf-8"))
    if cases["pipeline_sha256"] != sha(pipeline_path) or cases["bundle_sha256"] != sha(bundle_path):
        raise RuntimeError("Demo version mismatch")
    for filename in ("business.sqlite3", "checkpoints.sqlite3"):
        db = WORKSPACE / "data/m5-cascade-v1/runtime" / filename
        with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Database check failed: " + filename)
    return {"passed": True, "checked_frozen_files": checked, "demo_cases": len(cases["cases"]),
            "vision_or_llm_calls": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        meta = json.loads(META.read_text(encoding="utf-8"))
        marker = ROOT / ".team-assets-installed.json"
        installed = marker.is_file() and json.loads(marker.read_text()).get("sha256") == meta["sha256"]
        if not installed:
            path = args.archive.resolve() if args.archive else download(meta)
            install(path, meta)
    print(json.dumps(verify(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Do not print HTTP exception URLs or credentials.
        if type(error).__module__.startswith("httpx"):
            print("GitHub download failed. Check repository access or use -AssetsZip.", file=sys.stderr)
        else:
            print(str(error), file=sys.stderr)
        raise SystemExit(1)
