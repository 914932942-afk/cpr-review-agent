"""Freeze accepted M3 source, parameters, offline evidence and end-to-end engineering proof."""

import hashlib
import json
import sqlite3
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/m3-calibration-v1"
WS = ROOT / "workspaces/m3-calibrated-v1"
RESULTS = ROOT / "test-results/m3-calibration-v1"
RELEASE = ROOT / "releases/m3-calibrated-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if RELEASE.exists() or (DATA / "acceptance.json").exists():
        raise FileExistsError("Preserve accepted M3 release")
    original = read(DATA / "before.json")
    assert all(sha(ROOT / p) == value for p, value in original["sources"].items())
    assert sha(ROOT / ".env") == original["env_sha256"]
    assert read(DATA / "decision.json")["accepted"]
    for name in ["algorithm/acceptance.json", "http/acceptance.json", "ui/acceptance.json", "restart.json"]:
        assert read(RESULTS / name)["passed"]
    suites = list(ET.parse(RESULTS / "current-core.xml").getroot().iter("testsuite"))
    assert sum(int(s.get("tests", 0)) for s in suites) == 240
    assert all(
        int(s.get("failures", 0)) + int(s.get("errors", 0)) + int(s.get("skipped", 0)) == 0 for s in suites
    )
    profile_path = WS / "data/m3-calibration-v1/pipeline-manifest.json"
    profile = read(profile_path)
    assert all(sha(WS / p) == value for p, value in profile["sources"].items())
    assert all(sha(WS / p) == value for p, value in read(DATA / "upstream.json")["sources"].items())
    bundle = WS / "data/public_cpr/web-integration-v1/bundle-manifest.json"
    assert sha(bundle) == profile["model_bundle_sha256"]
    assert all(sha(WS / p) == value for p, value in read(bundle)["pinned_sources_and_assets"].items())
    runtime = WS / "data/m3-calibration-v1/runtime"
    connection = sqlite3.connect(runtime / "business.sqlite3")
    statuses = connection.execute("select status,count(*) from runs group by status").fetchall()
    connection.close()
    assert all(
        status in ["completed", "failed", "partially_completed", "rejected", "cancelled"]
        for status, _ in statuses
    )
    metrics = read(DATA / "raw26.json")["overall"]
    validation = read(DATA / "validation.json")["overall"]
    report = {
        "accepted": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(WS),
        "entrypoint": "scripts/start_m3.ps1",
        "url": "http://127.0.0.1:8010",
        "profile_sha256": sha(profile_path),
        "parent_profile_sha256": read(WS / "workspace.json")["parent_profile_sha256"],
        "model_bundle_sha256": sha(bundle),
        "algorithm": "source-quality-selective-v1",
        "m1_m2_algorithms_unchanged": True,
        "original_sources_and_env_unchanged": True,
        "m4_m5": "unchanged baselines",
        "cv_predictions_unchanged": True,
        "new_vision_training": False,
        "new_manual_labels": False,
        "new_collection": False,
        "provider_calls": 0,
        "training_scope": read(DATA / "prepared-audit.json")["summary"],
        "validation_positions": validation["selected"]["positions"],
        "raw26_positions": metrics["selected"]["positions"],
        "validation": validation,
        "raw26": metrics,
        "software_tests": 240,
        "algorithm_checks": 16,
        "real_readout_replays": 26,
        "actual_cold_videos": 2,
        "live_vision_tool_executions": 8,
        "uncertainty_persists_in_explanation_summary_export_and_restart": True,
        "actual_browser_widths": [1440, 390],
        "scope": "Previously inspected public records and declared programmed stress. Post-hoc OOF is for the calibrator; it does not undo upstream training reuse. No independent-blind, cross-person or clinical probability claim.",
        "quality_adapter_boundary": "Only validated same-video, same-profile, same-actor executed branch dependencies. Missing quality => uncertain without invented probability.",
        "known_tradeoff": "Raw26 defers12 positions:2 wrong,10 correct. At the same coverage ordinary temperature also defers2 wrong; incremental quality value is chiefly Brier improvement in this sample.",
    }
    (DATA / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    RELEASE.mkdir()
    for name in ["business", "checkpoints"]:
        source = sqlite3.connect(runtime / (name + ".sqlite3"))
        target = sqlite3.connect(RELEASE / (name + ".sqlite3"))
        source.backup(target)
        target.close()
        source.close()
    files = set(
        [
            ROOT / "README.md",
            ROOT / "docs/M3置信度校准与复核-v1.html",
            profile_path,
            WS / "workspace.json",
            ROOT / "tests/test_m3_profile.py",
        ]
    )
    files.update(p for p in DATA.rglob("*") if p.is_file() and p.suffix in [".json", ".npz"])
    files.update(p for p in RESULTS.rglob("*") if p.is_file() and p.suffix in [".json", ".xml", ".log"])
    files.update((ROOT / "experiments/m3_calibration_v1").glob("*.py"))
    for pattern in ["*m3*.py", "start_m3.ps1", "m3_ui_smoke.cjs"]:
        files.update((ROOT / "scripts").glob(pattern))
    for name in profile["sources"]:
        if name.startswith("frontend/") or name in [
            "backend/cpr_agent/modules/members.py",
            "backend/cpr_agent/module_runtime.py",
            "backend/cpr_agent/graph.py",
            "backend/cpr_agent/grounded_review.py",
            "backend/cpr_agent/m3_quality.py",
            "experiments/m3_calibration_v1/module.py",
        ]:
            files.add(WS / name)
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    archive = RELEASE / "source-parameters-and-evidence.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            assert p.is_relative_to(ROOT) and p.name != ".env"
            z.write(p, p.relative_to(ROOT).as_posix())
    manifest = {
        "files": hashes,
        "archive_sha256": sha(archive),
        "contains_calibration_parameters_and_public_readouts": True,
        "contains_vision_weights_or_video_or_keys": False,
        "snapshots": {p.name: sha(p) for p in RELEASE.glob("*.sqlite3")},
        "preserved_parent": "workspaces/m1-m2-refined-v1",
        "runtime_entrypoint": "scripts/start_m3.ps1",
    }
    (RELEASE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": True,
                "profile_sha256": sha(profile_path),
                "source_files": len(files),
                "archive_bytes": archive.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
