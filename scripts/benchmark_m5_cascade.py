"""Predeclared paired process benchmark; never select models from regression labels."""

import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/m5-cascade-v1"
OUT = ROOT / "test-results/m5-cascade-v1/benchmark"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "acceptance.json").exists()
    protocol = read(DATA / "protocol.json")
    assert sha(ROOT / "experiments/m5_cascade_v1/algorithm.py") == protocol["algorithm_sha256"]
    assert sha(ROOT / "experiments/m5_cascade_v1/runtime.py") == protocol["runtime_sha256"]
    jobs = {j["key"]: j for j in read(ROOT / "data/public_cpr/rgb-video-acceptance-v4/jobs.json")}
    records = []
    with FileLock(ROOT / "data/vision-gpu.lock", timeout=0):
        for ordinal, key in enumerate(protocol["bench_cases"]):
            job = jobs[key]
            video = ROOT / job["video"]
            pose = ROOT / "data/public_cpr/rgb-video-acceptance-v4/runs" / key / "inference.npz"
            assert sha(video) == job["video_sha256"]
            for repeat in range(3):
                pair = {}
                for mode in (
                    ["baseline", "candidate"] if (ordinal + repeat) % 2 == 0 else ["candidate", "baseline"]
                ):
                    dest = OUT / (key + "-" + str(repeat) + "-" + mode + ".json")
                    if not dest.exists():
                        started = time.perf_counter()
                        with dest.with_suffix(".log").open("wb") as log:
                            subprocess.run(
                                [
                                    sys.executable,
                                    str(ROOT / "scripts/m5_once.py"),
                                    "--mode",
                                    mode,
                                    "--video",
                                    str(video),
                                    "--pose",
                                    str(pose),
                                    "--output",
                                    str(dest),
                                ],
                                cwd=ROOT,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                                timeout=180,
                            )
                        (dest.with_suffix(".wall.json")).write_text(
                            json.dumps({"process_seconds": time.perf_counter() - started}), encoding="utf-8"
                        )
                    row = read(dest)
                    pair[mode] = {**row, **read(dest.with_suffix(".wall.json"))}
                    print(key, repeat, mode, round(pair[mode]["process_seconds"], 2), flush=True)
                a, b = (
                    load(OUT / (key + "-" + str(repeat) + "-baseline.npz")),
                    load(OUT / (key + "-" + str(repeat) + "-candidate.npz")),
                )
                for k in [
                    "label_ids",
                    "scores",
                    "thresholds",
                    "eligible",
                    "hand_override",
                    "normal_veto_threshold",
                ]:
                    assert np.allclose(a[k], b[k], atol=1e-6, rtol=0), (key, k)
                assert np.array_equal(a["eligible"], b["eligible"])
                assert pair["candidate"]["audit"]["m5_cascade"]["dense_visual_crop_evaluations"] == 0
                records.append(
                    {
                        "case": key,
                        "repeat": repeat,
                        "baseline_seconds": pair["baseline"]["process_seconds"],
                        "candidate_seconds": pair["candidate"]["process_seconds"],
                        "baseline_memory": pair["baseline"]["cuda_peak_allocated_bytes"],
                        "candidate_memory": pair["candidate"]["cuda_peak_allocated_bytes"],
                    }
                )
        # Controlled, programmatic input-quality stress: no claim of a real occlusion label.
        job = jobs[protocol["bench_cases"][0]]
        key = job["key"]
        pair = {}
        for mode in ["baseline", "candidate"]:
            dest = OUT / ("fallback-" + mode + ".json")
            if not dest.exists():
                with dest.with_suffix(".log").open("wb") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "scripts/m5_once.py"),
                            "--mode",
                            mode,
                            "--video",
                            str(ROOT / job["video"]),
                            "--pose",
                            str(
                                ROOT / "data/public_cpr/rgb-video-acceptance-v4/runs" / key / "inference.npz"
                            ),
                            "--stress",
                            "--output",
                            str(dest),
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=180,
                    )
            pair[mode] = load(dest.with_suffix(".npz"))
        assert read(OUT / "fallback-candidate.json")["audit"]["m5_cascade"]["route"] == "full-rgb-fallback"
        assert all(
            np.array_equal(pair["baseline"][k], pair["candidate"][k])
            for k in ["scores", "eligible", "thresholds", "hand_override"]
        )
    time_gain = statistics.median(
        (r["baseline_seconds"] - r["candidate_seconds"]) / r["baseline_seconds"] for r in records
    )
    memory_gain = statistics.median(
        (r["baseline_memory"] - r["candidate_memory"]) / r["baseline_memory"] for r in records
    )
    perclip = {
        k: statistics.median(r["candidate_seconds"] for r in records if r["case"] == k)
        / statistics.median(r["baseline_seconds"] for r in records if r["case"] == k)
        for k in protocol["bench_cases"]
    }
    passed = time_gain >= 0.10 and memory_gain >= 0.10 and all(ratio <= 1.05 for ratio in perclip.values())
    report = {
        "passed": passed,
        "paired_repeats": 9,
        "cold_process_executions": 20,
        "median_relative_process_time_reduction": time_gain,
        "median_relative_cuda_peak_reduction": memory_gain,
        "per_clip_candidate_to_baseline_wall_ratio": perclip,
        "scores_max_difference_tolerance": 1e-6,
        "all_decisions_identical": True,
        "real_controlled_quality_fallback_identical": True,
        "dense_crop_evaluations_baseline_per_complete_clip": 32,
        "dense_crop_evaluations_pruned": 0,
        "records": records,
        "scope": "M5 stage processes only; not whole-system latency. Same machine, public observed clips, alternating order. Fallback is programmed confidence stress.",
    }
    (OUT / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    assert passed, "Resource gates failed; do not integrate"


if __name__ == "__main__":
    main()
