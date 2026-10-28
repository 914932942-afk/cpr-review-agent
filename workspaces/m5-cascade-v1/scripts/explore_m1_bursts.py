"""Trusted-anchor burst repair: train-only selection and reused public validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


common = mod("shared_m1_explorer", ROOT / "scripts/explore_m1_kinematic.py")
burst = mod("candidate_m1_bursts", ROOT / "experiments/m1_burst_v2/algorithm.py")
original_corrupt = common.helper.corrupt
OUT = ROOT / "data/m1-m2-exploration-v1/m1-burst"
CONFIGS = {
    f"burst{r}_vel{v}": burst.Config(burst_residual=r, anchor_velocity_difference=v)
    for r in [0.06, 0.08, 0.12]
    for v in [0.03, 0.06]
}
TRAIN = ["clean", "jitter", "mixed", "confidence_only", "burst3", "burst4", "burst_confidence_only"]
VALID = [*TRAIN, "jitter_seed2", "mixed_seed2", "burst3_seed2", "burst4_seed2"]


def write(name, v):
    text = json.dumps(v, indent=2, allow_nan=False)
    with (OUT / name).open("x", encoding="utf-8") as f:
        f.write(text)


def corrupt(points, confidence, condition, seed):
    if not condition.startswith("burst"):
        return original_corrupt(points, confidence, condition, seed)
    p, q = points.copy(), confidence.copy()
    rng = np.random.default_rng(seed + (10003 if condition.endswith("seed2") else 10001))
    length = 4 if condition.startswith("burst4") else 3
    size = np.linalg.norm(p[:, [11, 12]].mean(1) - p[:, [5, 6]].mean(1), axis=1)
    valid = size[(q[:, [5, 6, 11, 12]] > 0.45).all(1)]
    scale = float(np.median(valid)) if len(valid) else 100.0
    for joint in rng.choice([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 5, replace=False):
        choices = [
            i for i in range(4, len(p) - length - 4) if (q[i - 2 : i + length + 2, joint] >= 0.45).all()
        ]
        if not choices:
            continue
        start = int(rng.choice(choices))
        offset = rng.normal(size=2)
        offset = offset / max(np.linalg.norm(offset), 1e-6) * 0.2 * scale
        if condition != "burst_confidence_only":
            p[start : start + length, joint] += offset
        q[start : start + length, joint] = 0.25
    return p, q


def evaluate(indices, methods, conditions, phase):
    # Evaluation-only dependency injection; the accepted application and algorithm remain untouched.
    common.new = burst
    common.helper.corrupt = corrupt
    return common.evaluate(indices, methods, conditions, phase)


def assess(report, name):
    base = report["baseline"]["conditions"]
    cand = report[name]["conditions"]
    noisy = [k for k in base if k != "clean"]
    bursts = [k for k in base if k.startswith("burst") and k != "burst_confidence_only"]
    gain = float(np.mean([cand[k]["macro_f1"] - base[k]["macro_f1"] for k in bursts]))
    ratio = float(
        np.mean([cand[k]["descriptor_drift"] / max(base[k]["descriptor_drift"], 1e-12) for k in bursts])
    )
    checks = {
        "clean_noninferior": cand["clean"]["macro_f1"] >= base["clean"]["macro_f1"] - 1e-12,
        "old_stresses_loss_le_002": all(
            cand[k]["macro_f1"] >= base[k]["macro_f1"] - 0.002 for k in noisy if not k.startswith("burst")
        ),
        "bursts_noninferior": all(cand[k]["macro_f1"] >= base[k]["macro_f1"] - 1e-12 for k in bursts),
        "confidence_only_noninferior": cand["burst_confidence_only"]["macro_f1"]
        >= base["burst_confidence_only"]["macro_f1"] - 1e-12,
        "normal_false_alarms_no_increase": all(
            cand[k]["normal_false_alarms"] <= base[k]["normal_false_alarms"] for k in base
        ),
        "cpu_le_250ms": report[name]["seconds"] <= 0.25,
        "value": gain >= 0.002 or ratio <= 0.9,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mean_burst_f1_gain": gain,
        "burst_descriptor_drift_ratio": ratio,
        "clean_gain": cand["clean"]["macro_f1"] - base["clean"]["macro_f1"],
    }


def select():
    OUT.mkdir(exist_ok=False)
    _, _, ids, _, train, val = common.data()
    write(
        "protocol.json",
        {
            "algorithm_sha256": common.sha(ROOT / "experiments/m1_burst_v2/algorithm.py"),
            "evaluator_sha256": common.sha(Path(__file__)),
            "shared_evaluator_sha256": common.sha(ROOT / "scripts/explore_m1_kinematic.py"),
            "configs": {k: asdict(v) for k, v in CONFIGS.items()},
            "train_ids": [str(ids[i]) for i in train],
            "validation_ids": [str(ids[i]) for i in val],
            "train_conditions": TRAIN,
            "validation_conditions": VALID,
            "selection": "128 original train records only. Require no clean/normal-FA regression, old stress loss <=.002, short-burst and confidence-only noninferior, CPU<=250ms, and >=.002 burst F1 gain or >=10% burst drift reduction. Then rank clean F1, burst F1, drift and conservative thresholds.",
            "scope": "Controlled coordinate corruption of original public records, not newly collected actions or blind/cross-person validation.",
        },
    )
    report, _ = evaluate(train, CONFIGS, TRAIN, "burst-training")
    checks = {k: assess(report, k) for k in CONFIGS}
    write("training.json", {"methods": report, "assessment": checks})
    allowed = [k for k in checks if checks[k]["passed"]]
    if not allowed:
        write("decision.json", {"accepted": False, "reason": "No train-qualified short-burst candidate"})
        print("No short-burst candidate qualified", flush=True)
        return
    name = max(
        allowed,
        key=lambda k: (
            round(report[k]["conditions"]["clean"]["macro_f1"], 8),
            round(checks[k]["mean_burst_f1_gain"], 8),
            -checks[k]["burst_descriptor_drift_ratio"],
            CONFIGS[k].burst_residual,
            -CONFIGS[k].anchor_velocity_difference,
        ),
    )
    write(
        "selection.json",
        {
            "name": name,
            "config": asdict(CONFIGS[name]),
            "selected_before_validation": True,
            "protocol_sha256": common.sha(OUT / "protocol.json"),
        },
    )
    print(json.dumps({"selected": name, "training": checks[name]}), flush=True)


def validate():
    choice = json.loads((OUT / "selection.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert common.sha(ROOT / "experiments/m1_burst_v2/algorithm.py") == protocol["algorithm_sha256"]
    cfg = burst.Config(**choice["config"])
    _, _, _, _, _, val = common.data()
    name = choice["name"]
    report, outputs = evaluate(val, {name: cfg}, VALID, "burst-validation")
    checked = assess(report, name)
    write(
        "validation.json",
        {"passed": checked["passed"], "assessment": checked, "methods": report, "records": len(val)},
    )
    write(
        "decision.json",
        {
            "accepted": checked["passed"],
            "selected": name,
            "selection_sha256": common.sha(OUT / "selection.json"),
            "assessment": checked,
        },
    )
    np.savez_compressed(
        OUT / "validation-readouts.npz",
        **{
            n + "__" + c + "__" + k: a
            for n, conditions in outputs.items()
            for c, arrays in conditions.items()
            for k, a in arrays.items()
        },
    )
    print(json.dumps({"validation": checked}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["select", "validate"])
    globals()[p.parse_args().stage]()
