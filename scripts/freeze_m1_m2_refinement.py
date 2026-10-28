"""Fail-closed acceptance and a new lightweight source/evidence archive; prior releases are preserved."""

import hashlib
import json
import sqlite3
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data/m1-m2-exploration-v1"
WORKSPACE = ROOT / "workspaces/m1-m2-refined-v1"
RESULTS = ROOT / "test-results/m1-m2-refined-v1"
RELEASE = ROOT / "releases/m1-m2-refined-v1"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def suite(p, expected):
    rows = ET.parse(p).getroot().iter("testsuite")
    totals = {k: 0 for k in ["tests", "failures", "errors", "skipped"]}
    for r in rows:
        for k in totals:
            totals[k] += int(r.get(k, 0))
    assert totals == {"tests": expected, "failures": 0, "errors": 0, "skipped": 0}, totals
    return totals


def main():
    destination = EVIDENCE / "acceptance.json"
    if destination.exists() or RELEASE.exists():
        raise FileExistsError("Preserve existing acceptance/release")
    before = read(EVIDENCE / "before.json")
    assert all(sha(ROOT / p) == value for p, value in before["sources"].items())
    assert sha(ROOT / ".env") == before["env_sha256"]
    for p in [EVIDENCE / "m1-coherent/decision.json", EVIDENCE / "m2-competence/decision.json"]:
        assert read(p)["accepted"]
    for p in [
        EVIDENCE / "combined/raw26.json",
        RESULTS / "http/acceptance.json",
        RESULTS / "restart.json",
        RESULTS / "adapter.json",
        RESULTS / "algorithm-tests/acceptance.json",
    ]:
        assert read(p)["passed"]
    old_tests = suite(ROOT / "test-results/m1-m2-exploration-v1/base-software.xml", 267)
    new_tests = suite(RESULTS / "current-core.xml", 234)
    profile_path = WORKSPACE / "data/m1-m2-refined-v1/pipeline-manifest.json"
    profile = read(profile_path)
    assert all(sha(WORKSPACE / p) == value for p, value in profile["sources"].items())
    bundle = WORKSPACE / "data/public_cpr/web-integration-v1/bundle-manifest.json"
    assert sha(bundle) == profile["model_bundle_sha256"]
    assert all(sha(WORKSPACE / p) == value for p, value in read(bundle)["pinned_sources_and_assets"].items())
    runtime = WORKSPACE / "data/m1-m2-refined-v1/runtime"
    m1_proofs = []
    with httpx.Client(base_url="http://127.0.0.1:8009", timeout=15) as client:
        caps = client.get("/api/capabilities").raise_for_status().json()
        assert caps["llm"]["mode"] == "disabled"
        assert all(caps["module_slots"][k]["kind"] == "baseline" for k in ["M3", "M4", "M5"])
        for row in read(RESULTS / "http/state.json").values():
            tools = client.get("/api/runs/" + row["run_id"] + "/tools").raise_for_status().json()
            raw = read(runtime / tools[1]["output"]["artifact_rel"])
            audit = raw["fullpose_audit"]["m1_stability"]
            assert audit["implementation_revision"] == "m1-coherent-offset-v3"
            assert audit["config"] == read(EVIDENCE / "m1-coherent/selection.json")["config"]
            m1_proofs.append({"run_id": row["run_id"], "audit": audit})
    connection = sqlite3.connect(runtime / "business.sqlite3")
    statuses = connection.execute("select status,count(*) from runs group by status").fetchall()
    connection.close()
    assert all(
        status in ["completed", "failed", "partially_completed", "rejected", "cancelled"]
        for status, _ in statuses
    )
    selected = [
        ROOT / "experiments/m1_coherent_v3/algorithm.py",
        ROOT / "experiments/m2_competence_v3/fusion_algorithm.py",
        ROOT / "experiments/m2_competence_v3/module.py",
    ]
    report = {
        "accepted": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(WORKSPACE),
        "entrypoint": "scripts/start_refined.ps1",
        "url": "http://127.0.0.1:8009",
        "profile_sha256": sha(profile_path),
        "model_bundle_sha256": sha(bundle),
        "stable_sources_and_env_unchanged": True,
        "m3_m4_m5": "unchanged baselines",
        "provider_calls": 0,
        "software": {
            "original_root_historical_profiles": old_tests,
            "current_workspace_with_new_profile_tests": new_tests,
            "scope": "40 historical-version-specific tests run in the original root; initial copied historical suite failed on missing/old profile preconditions. Not counted as current-workspace passing tests.",
        },
        "m1": read(EVIDENCE / "m1-coherent/validation.json")["assessment"],
        "m2": read(EVIDENCE / "m2-competence/validation.json")["checks"],
        "combined": read(EVIDENCE / "combined/raw26.json")["metrics"],
        "actual_m1_execution_audits": m1_proofs,
        "selected_algorithm_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in selected},
        "new_manual_labels": False,
        "new_data_collection": False,
        "new_backbone_or_head_training": False,
        "evidence_scope": "Iterative engineering on reused public validation and raw26, not blind/cross-person/clinical accuracy. Controlled coordinate noise is not actual video occlusion.",
        "rejected_candidates": [
            "m1 kinematic guard: no measurable value",
            "m1 burst v2: validation regression",
            "m2 boundary v2: raw26 regression",
        ],
    }
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    RELEASE.mkdir()
    for name in ["business", "checkpoints"]:
        source = sqlite3.connect(runtime / (name + ".sqlite3"))
        target = sqlite3.connect(RELEASE / (name + ".sqlite3"))
        source.backup(target)
        target.close()
        source.close()
    files = set(
        [
            destination,
            ROOT / "docs/M1-M2算法优化与验收-v1.html",
            ROOT / "README.md",
            profile_path,
            WORKSPACE / "workspace.json",
        ]
    )
    for area in [EVIDENCE, RESULTS]:
        files.update(p for p in area.rglob("*") if p.is_file() and p.suffix in [".json", ".xml", ".log"])
    for area in ["m1_kinematic_v1", "m1_burst_v2", "m1_coherent_v3", "m2_boundary_v2", "m2_competence_v3"]:
        files.update((ROOT / "experiments" / area).glob("*.py"))
    for pattern in [
        "explore_m1*.py",
        "explore_m2*.py",
        "*refined*.py",
        "*refinement*.py",
        "start_refined.ps1",
        "m2_competence_ablation.py",
    ]:
        files.update((ROOT / "scripts").glob(pattern))
    files.add(ROOT / "tests/test_refined_profile.py")
    for p in [
        "experiments/m1_fast_v1/algorithm.py",
        "experiments/m1_fast_v1/m1_fast_pipeline.py",
        "backend/cpr_agent/modules/members.py",
        "backend/cpr_agent/branch_vision.py",
    ]:
        files.add(WORKSPACE / p)
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    archive = RELEASE / "source-and-evidence.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zip:
        for p in sorted(files):
            assert p.is_relative_to(ROOT) and p.name != ".env"
            zip.write(p, p.relative_to(ROOT).as_posix())
    manifest = {
        "files": hashes,
        "zip_sha256": sha(archive),
        "contains_weights_or_video_or_keys": False,
        "database_snapshots": {p.name: sha(p) for p in RELEASE.glob("*.sqlite3")},
        "restore": "Keep parent public assets and prior stable versions. start_refined.ps1 uses the independent workspace/profile and shared parent GPU lock.",
    }
    (RELEASE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": True,
                "profile_sha256": sha(profile_path),
                "archive_files": len(files),
                "archive_bytes": archive.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
