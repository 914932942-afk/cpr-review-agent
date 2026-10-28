"""Freeze M4 only after policy, source equivalence, real routed execution and recovery pass."""

import hashlib
import json
import sqlite3
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/m4-scheduling-v1"
WS = ROOT / "workspaces/m4-routing-v1"
RESULTS = ROOT / "test-results/m4-scheduling-v1"
RELEASE = ROOT / "releases/m4-routing-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if RELEASE.exists() or (DATA / "acceptance.json").exists():
        raise FileExistsError("Preserve accepted M4 release")
    before = read(DATA / "before.json")
    assert all(sha(ROOT / p) == h for p, h in before["sources"].items())
    assert sha(ROOT / ".env") == before["env_sha256"]
    assert read(DATA / "decision.json")["accepted"] and read(DATA / "offline-validation.json")["passed"]
    for name in [
        "algorithm/acceptance.json",
        "http/acceptance.json",
        "ui/acceptance.json",
        "ui-expansion/acceptance.json",
        "restart.json",
    ]:
        assert read(RESULTS / name)["passed"]
    suites = list(ET.parse(RESULTS / "current-core-final.xml").getroot().iter("testsuite"))
    assert sum(int(s.get("tests", 0)) for s in suites) == 243 and all(
        sum(int(s.get(k, 0)) for k in ["errors", "failures", "skipped"]) == 0 for s in suites
    )
    latest = list(ET.parse(RESULTS / "profile-final.xml").getroot().iter("testsuite"))
    assert all(int(s.get("errors", 0)) + int(s.get("failures", 0)) == 0 for s in latest)
    profile_path = WS / "data/m4-scheduling-v1/pipeline-manifest.json"
    profile = read(profile_path)
    assert all(sha(WS / p) == h for p, h in profile["sources"].items())
    parent = ROOT / "workspaces/m3-calibrated-v1/data/m3-calibration-v1/pipeline-manifest.json"
    parent_profile = read(parent)
    numerical = [
        p
        for p in parent_profile["sources"]
        if p.startswith(
            (
                "experiments/m1_fast_v1/",
                "experiments/m2_competence_v3/",
                "experiments/m3_calibration_v1/",
                "experiments/branch_pipeline_v1/",
                "data/m1-m2-exploration-v1/",
                "data/m3-calibration-v1/",
            )
        )
    ]
    for p in numerical:
        assert sha(WS / p) == parent_profile["sources"][p]
    bundle = WS / "data/public_cpr/web-integration-v1/bundle-manifest.json"
    assert sha(bundle) == profile["model_bundle_sha256"]
    assert all(sha(WS / p) == h for p, h in read(bundle)["pinned_sources_and_assets"].items())
    runtime = WS / "data/m4-scheduling-v1/runtime"
    c = sqlite3.connect(runtime / "business.sqlite3")
    statuses = c.execute("select status,count(*) from runs group by status").fetchall()
    tool_counts = dict(c.execute("select status,count(*) from tools group by status").fetchall())
    c.close()
    assert all(
        status in ["completed", "partially_completed", "failed", "rejected", "cancelled"]
        for status, _ in statuses
    )
    assert tool_counts.get("succeeded") == 11
    simulation = read(DATA / "offline-validation.json")
    report = {
        "accepted": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(WS),
        "entrypoint": "scripts/start_m4.ps1",
        "url": "http://127.0.0.1:8011",
        "graph_version": "cpr-review-12-v0.7-routing",
        "algorithm": "goal-cost-dag-v1",
        "decision_mode": "Bounded Plan & Execute with state reassessment; deterministic dependency/cost optimisation",
        "profile_sha256": sha(profile_path),
        "parent_profile_sha256": sha(parent),
        "model_bundle_sha256": sha(bundle),
        "m1_m2_m3_numerical_algorithms_unchanged": True,
        "m5": "baseline",
        "env_unchanged": True,
        "routes": {
            "full-v15": {"labels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13], "tools": 4},
            "body-qualified-v1": {"labels": [3, 4, 8, 11], "tools": 3, "rgb_executed": False},
        },
        "validation": {
            "paired38_requested_scores_and_flags_exact": True,
            "raw26_requested_scores_and_flags_exact": True,
            "branch_property_cases": 200,
            "scope_property_cases": 2047,
            "current_software_tests": 243,
            "policy_and_ml_stage_checks": 22,
            "api_successful_tool_executions": 9,
            "actual_expand_button_successful_executions": 2,
            "all_tool_attempt_counts": tool_counts,
            "recovery_and_gpu_lock": True,
        },
        "counterfactual_task_mix": {
            k: simulation[k]
            for k in [
                "tasks",
                "fixed_full_tool_calls",
                "policy_tool_calls",
                "estimated_seconds_fixed",
                "estimated_seconds_policy",
                "task_mix",
                "timing_claim",
            ]
        },
        "gain_semantics": "New requested coverage per marginal cost, not accuracy improvement or entropy gain. Hard coverage and source-equivalence constraints.",
        "cost_scope": "Six historical traces including repeated videos. Median ranking; observed-max x1.25 upper guard plus8s reserve. Bootstrap diagnostic is not a population guarantee.",
        "important_tradeoff": "Focused3 + later expansion2 =5 tool calls versus upfront full4. The planner optimises the current explicit request, not an assumed future goal. Reuse follows the selected report; no silent source replacement.",
        "initial_failure": "One worker import-path failure retained, fixed before successful inference, with5 ML-stage tests added. Its old session was not relabelled under the new profile.",
        "new_vision_training": False,
        "new_manual_labels": False,
        "new_data_collection": False,
        "provider_calls": 0,
        "scope": "Previously inspected public-data engineering regression, not blind/cross-person/clinical evidence.",
    }
    (DATA / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    RELEASE.mkdir()
    for name in ["business", "checkpoints"]:
        source = sqlite3.connect(runtime / (name + ".sqlite3"))
        target = sqlite3.connect(RELEASE / (name + ".sqlite3"))
        source.backup(target)
        target.close()
        source.close()
    files = {
        ROOT / "README.md",
        ROOT / "docs/M4目标与成本感知调度-v1.html",
        profile_path,
        WS / "workspace.json",
        ROOT / "tests/test_m4_profile.py",
    }
    files.update(p for p in DATA.rglob("*") if p.is_file() and p.suffix == ".json")
    files.update(p for p in RESULTS.rglob("*") if p.is_file() and p.suffix in [".json", ".xml", ".log"])
    files.update((ROOT / "experiments/m4_policy_v1").glob("*.py"))
    for pattern in ["*m4*.py", "m4*.cjs", "start_m4.ps1"]:
        files.update((ROOT / "scripts").glob(pattern))
    changed = [p for p, h in profile["sources"].items() if parent_profile["sources"].get(p) != h]
    files.update(WS / p for p in changed)
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    archive = RELEASE / "source-parameters-and-evidence.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            assert p.is_relative_to(ROOT) and p.name != ".env"
            z.write(p, p.relative_to(ROOT).as_posix())
    (RELEASE / "manifest.json").write_text(
        json.dumps(
            {
                "files": hashes,
                "archive_sha256": sha(archive),
                "database_snapshots": {p.name: sha(p) for p in RELEASE.glob("*.sqlite3")},
                "contains_vision_weights_or_video_or_keys": False,
                "contains_cost_parameters": True,
                "parent_workspace_preserved": "workspaces/m3-calibrated-v1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "accepted": True,
                "profile_sha256": sha(profile_path),
                "archive_files": len(files),
                "archive_bytes": archive.stat().st_size,
                "tool_counts": tool_counts,
            }
        )
    )


if __name__ == "__main__":
    main()
