"""Build an isolated M2 application; keep every existing profile and application source intact."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "workspaces/m2-quality-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if TARGET.exists():
        raise FileExistsError("Preserve the existing M2 workspace; do not overwrite accepted code")
    decision = ROOT / "data/m2-quality-v1/experiment-01/decision.json"
    if not read(decision)["accepted"]:
        raise ValueError("M2 offline algorithm acceptance is required")
    files = set()
    for area in [
        "backend",
        "scripts",
        "tests",
        "frontend/src",
        "frontend/dist",
        "frontend/public",
        "fixtures",
        "docs",
    ]:
        files.update(p for p in (ROOT / area).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.update(
        p
        for p in (ROOT / "experiments").rglob("*")
        if p.is_file() and p.suffix in {".py", ".json", ".txt"} and "__pycache__" not in p.parts
    )
    for name in [
        "pyproject.toml",
        "requirements.lock",
        "README.md",
        "frontend/package.json",
        "frontend/package-lock.json",
        "frontend/tsconfig.json",
        "frontend/vite.config.ts",
        "frontend/index.html",
    ]:
        files.add(ROOT / name)
    manifests = [
        (ROOT / "data/public_cpr/web-integration-v1/bundle-manifest.json", "pinned_sources_and_assets"),
        (ROOT / "data/module-slots-v1/pipeline-manifest.json", "sources"),
    ]
    for manifest, field in manifests:
        files.add(manifest)
        for name, expected in read(manifest)[field].items():
            source = ROOT / name
            if sha(source) != expected:
                raise ValueError("Existing pinned source changed: " + name)
            files.add(source)
    files.add(ROOT / "data/m1-optimization-v2/pipeline-manifest.json")
    for name in ["protocol.json", "selection.json", "decision.json"]:
        files.add(decision.parent / name)
    catalog = ROOT / "data/reference-v1/catalog.json"
    files.add(catalog)
    value = read(catalog)
    files.add(ROOT / value["annotation_rel"])
    for entry in value["entries"]:
        files.update([ROOT / entry["media_rel"], ROOT / entry["manifest_rel"]])
    size = 0
    for source in sorted(files):
        if not source.resolve().is_relative_to(ROOT) or not source.is_file() or source.name == ".env":
            raise ValueError("Invalid workspace source")
        target = TARGET / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        size += source.stat().st_size
    # Windows FileLock removes its file on release: hard links cannot safely share it over time.
    # Keep the actual canonical parent lock path for every new worker invocation instead.
    adapter = TARGET / "backend/cpr_agent/branch_vision.py"
    source = adapter.read_text(encoding="utf-8")
    old_lock = 'str(ROOT / "data/vision-gpu.lock")'
    if source.count(old_lock) != 1:
        raise ValueError("Unexpected lock adapter; do not patch blindly")
    adapter.write_text(
        source.replace(old_lock, 'str(ROOT.parents[1] / "data/vision-gpu.lock")'), encoding="utf-8"
    )
    # Explicit registration only inside the isolated copy; the stable application remains byte-identical.
    members = TARGET / "backend/cpr_agent/modules/members.py"
    members.write_text(
        '''"""Accepted M2 registration in the isolated M2 workspace."""
from pathlib import Path
import importlib.util
import sys

from .registry import ModuleSpec

ROOT = Path(__file__).resolve().parents[3]

def create_m2():
    name = "cpr_m2_quality_member"
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/m2_quality_v1/module.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.QualityAwareFusion()

def register_members(registry):
    registry.register(ModuleSpec("M2", "quality-gated-disagreement-v1", create_m2, kind="improved", required_files=(
        "experiments/m2_quality_v1/module.py", "experiments/m2_quality_v1/fusion_algorithm.py",
        "data/m2-quality-v1/experiment-01/protocol.json", "data/m2-quality-v1/experiment-01/selection.json",
        "data/m2-quality-v1/experiment-01/decision.json")))
''',
        encoding="utf-8",
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CPR_", "DEEPSEEK_"))}
    env["PYTHONPATH"] = str(TARGET / "backend")
    process = subprocess.run(
        [
            sys.executable,
            str(TARGET / "scripts/build_module_profile.py"),
            "--base-profile",
            str(TARGET / "data/module-slots-v1/pipeline-manifest.json"),
            "--m2",
            "quality-gated-disagreement-v1",
            "--output",
            str(TARGET / "data/m2-quality-v1/pipeline-manifest.json"),
        ],
        cwd=TARGET,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode:
        raise RuntimeError("Isolated profile build failed: " + process.stderr[-1500:])
    manifest = read(TARGET / "data/m2-quality-v1/pipeline-manifest.json")
    record = {
        "workspace": str(TARGET),
        "source_project": str(ROOT),
        "copied_files": len(files),
        "copied_bytes": size,
        "env_copied": False,
        "shared_gpu_lock": True,
        "runtime_data": str(TARGET / "data/m2-quality-v1/runtime"),
        "module_slots": manifest["module_slots"],
        "pipeline_sha256": sha(TARGET / "data/m2-quality-v1/pipeline-manifest.json"),
        "registration_source_sha256": sha(members),
    }
    (TARGET / "workspace.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
