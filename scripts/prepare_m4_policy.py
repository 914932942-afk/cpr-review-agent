"""Freeze route certificates and stage cost distributions from accepted real tool traces."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/m4-scheduling-v1"
TOOLS = ["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = OUT / "parameters.json"
    if destination.exists():
        raise FileExistsError("Preserve frozen M4 cost parameters")
    metadata = ROOT / "data/public_cpr/rgb-evaluation-v15/protocol.json"
    reliability = read(metadata)["reliability"]
    m2 = read(ROOT / "data/m1-m2-exploration-v1/m2-competence/selection.json")["config"]
    assert m2["lock_unshared"] and all(reliability[str(k)]["qualified"] for k in [3, 4, 8, 11])
    jobs = {row["key"]: row for row in read(ROOT / "data/public_cpr/rgb-video-acceptance-v4/jobs.json")}
    records = []
    sources = {}
    samples = {tool: [] for tool in TOOLS}
    frames = []
    for name in ["m2-quality-v1", "m1-m2-refined-v1", "m3-calibration-v1"]:
        path = ROOT / "test-results" / name / "http/acceptance.json"
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        for row in read(path)["records"]:
            key = row["name"].removeprefix("m2-").removeprefix("m3-").removesuffix(".mp4")
            if key not in jobs:
                raise ValueError("Timing trace missing public video identity")
            frames.append(jobs[key]["expected_frames"])
            timing = {t["id"]: t["ms"] / 1000 for t in row["tools"]}
            assert set(timing) == set(TOOLS) and all(v > 0 for v in timing.values())
            for tool in TOOLS:
                samples[tool].append(timing[tool])
            records.append(
                {"source": name, "public_record": key, "decoded_frames": frames[-1], "seconds": timing}
            )
    rng = np.random.default_rng(20261027)
    costs = {}
    for tool, values in samples.items():
        a = np.array(values)
        bootstrap = np.median(rng.choice(a, (2000, len(a)), replace=True), axis=1)
        costs[tool] = {
            "median_seconds": float(np.median(a)),
            "bootstrap_median_upper95": float(np.quantile(bootstrap, 0.95)),
            "upper_seconds": float(max(np.quantile(bootstrap, 0.95), max(a) * 1.25)),
            "reference_frames": float(np.median(frames)),
            "observations": len(a),
        }
    parameters = {
        "algorithm": "goal-cost-dag-v1",
        "cost_model": costs,
        "closing_reserve_seconds": 8.0,
        "route_certificates": {
            "full-v15": "Accepted M1/M2/M3 full11 pipeline",
            "body-qualified-v1": "Frozen qualified non-RGB decisions are protected by guarded selection; M2 scope lock preserves these four. Exact requested-score/flag equivalence, not an accuracy guarantee.",
        },
        "certified_body_label_ids": [3, 4, 8, 11],
        "source_hashes": {**sources, metadata.relative_to(ROOT).as_posix(): sha(metadata)},
        "timing_scope": "Six real stage traces on this machine, including repeated recordings. Bootstrap describes observed costs only; max-observed x1.25 plus closing reserve is a conservative guard, not a population runtime guarantee. Pose cost scales upwards for longer clips; no favourable extrapolation to short clips.",
    }
    destination.write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    (OUT / "timing-records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    protocol = {
        "algorithm_sha256": sha(ROOT / "experiments/m4_policy_v1/algorithm.py"),
        "parameters_sha256": sha(destination),
        "selection": "Certified hard coverage, legal dependency closure, approved same-source reuse, conservative budget feasibility; then maximise new requested coverage / marginal expected seconds. Never use labels at inference.",
        "acceptance": "Requested predictions/scores and M3 states identical to the full route for certified scope across public paired38 and raw26; no unsupported claims. Savings in predeclared 11-single-focus + all11 task mix; partial/full cold-video, expansion reuse, existing-only, uncertainty STOP, tampering, restart and GPU-lock checks.",
        "scope": "Engineering public-data regression; no blind or clinical claim; no extra vision training or manual labels.",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"traces": len(records), "costs": costs}))


if __name__ == "__main__":
    main()
