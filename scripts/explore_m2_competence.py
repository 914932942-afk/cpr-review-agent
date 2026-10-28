"""Select a finite boundary-aware fusion candidate on original train32, then freeze and validate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from evaluate_m2_quality import condition, load, metric, sliced

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/m1-m2-exploration-v1/m2-competence"


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


old = mod("accepted_m2_quality", ROOT / "experiments/m2_quality_v1/fusion_algorithm.py")
new = mod("candidate_m2_competence", ROOT / "experiments/m2_competence_v3/fusion_algorithm.py")
FIXED = {"full_rgb_ratio": 0.75, "minimum_rgb_weight": 0.5}
CONFIGS = {
    f"band{b}_adv{a}": new.Config(**FIXED, boundary_band=b, skeleton_advantage=a)
    for b in [0.35, 0.7, 1.4]
    for a in [0.5, 1.0, 2.0]
}
CONDITIONS = ["clean", "rgb_noise_half_support", "rgb_noise_reduced_support"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name, value):
    payload = json.dumps(value, indent=2, allow_nan=False)
    with (OUT / name).open("x", encoding="utf-8") as f:
        f.write(payload)


def prediction(args, config):
    sk, rg, used, requested, hands = args
    n = len(used)
    result = []
    audits = []
    for i in range(n):
        method = old.fuse if config is None else new.fuse
        cfg = old.Config(**FIXED) if config is None else config
        a, audit = method(
            sliced(sk, i, n), sliced(rg, i, n), int(used[i]), int(requested[i]), bool(hands[i]), cfg
        )
        result.append(a["predictions"][0])
        audits.append(audit)
    return np.stack(result), audits


def evaluate(bundle, indices, config):
    return {
        c: metric(prediction(condition(bundle, indices, c), config)[0], bundle["targets"][indices])
        for c in CONDITIONS
    }


def select():
    OUT.mkdir(exist_ok=False)
    bundle = load(ROOT / "data/m2-quality-v1/prepared/development.npz")
    train = np.flatnonzero(bundle["splits"] == "train")
    protocol = {
        "algorithm_sha256": sha(ROOT / "experiments/m2_competence_v3/fusion_algorithm.py"),
        "structural_rule": "High-quality boundary arbitration cannot overturn frozen qualified RGB specialists. Final predictions outside RGB scope equal the same-input frozen merge. Low-quality v1 gate remains.",
        "reuse_disclosure": "Iterative engineering after v2 failed on previously inspected raw26; these are regression checks, not blind generalisation estimates.",
        "dataset_sha256": sha(ROOT / "data/m2-quality-v1/prepared/development.npz"),
        "configs": {k: asdict(v) for k, v in CONFIGS.items()},
        "conditions": CONDITIONS,
        "selection": "Train32 only; first require clean/shared Hamming noninferior, each stress noninferior; rank clean error, mean stress error, then narrower uncertainty band and stronger required skeleton advantage. Validation6 and previously inspected raw26 are not used for parameter choice.",
        "gates": "No clean/shared regression; no stress/shared regression; raw26 Macro-F1 noninferior and no additional normal FA; must improve raw F1 >=.005 OR validation stress error by >=1 position with a genuine nonzero natural boundary activation.",
        "prior": "Class reliability is the existing frozen v15 development metadata; score margins are not calibrated probabilities.",
    }
    write("protocol.json", protocol)
    base = evaluate(bundle, train, None)
    candidates = {}
    for name, cfg in CONFIGS.items():
        report = evaluate(bundle, train, cfg)
        passed = all(
            report[c]["shared_hamming_error"] <= base[c]["shared_hamming_error"] + 1e-12 for c in CONDITIONS
        )
        candidates[name] = {"eligible": passed, "metrics": report}
    write("training.json", {"baseline": base, "candidates": candidates})
    allowed = [k for k in CONFIGS if candidates[k]["eligible"]]
    if not allowed:
        write("decision.json", {"accepted": False, "reason": "No train-qualified boundary candidate"})
        return
    name = min(
        allowed,
        key=lambda k: (
            candidates[k]["metrics"]["clean"]["shared_hamming_error"],
            np.mean([candidates[k]["metrics"][c]["shared_hamming_error"] for c in CONDITIONS[1:]]),
            CONFIGS[k].boundary_band,
            -CONFIGS[k].skeleton_advantage,
        ),
    )
    write(
        "selection.json",
        {
            "name": name,
            "config": asdict(CONFIGS[name]),
            "protocol_sha256": sha(OUT / "protocol.json"),
            "selected_before_validation": True,
        },
    )
    print("M2 boundary candidate frozen", name, flush=True)


def validate():
    selection = json.loads((OUT / "selection.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert sha(ROOT / "experiments/m2_competence_v3/fusion_algorithm.py") == protocol["algorithm_sha256"]
    cfg = new.Config(**selection["config"])
    bundle = load(ROOT / "data/m2-quality-v1/prepared/development.npz")
    val = np.flatnonzero(bundle["splits"] == "validation")
    base = evaluate(bundle, val, None)
    candidate = evaluate(bundle, val, cfg)
    raw = load(ROOT / "data/m2-quality-v1/experiment-01/raw-readouts.npz")
    args = (
        {k[3:]: v for k, v in raw.items() if k.startswith("sk_")},
        {k[4:]: v for k, v in raw.items() if k.startswith("rgb_")},
        raw["usable"],
        raw["requested"],
        raw["hand_available"],
    )
    p, audits = prediction(args, cfg)
    bp, _ = prediction(args, None)
    b = metric(bp, raw["targets"])
    c = metric(p, raw["targets"])
    activated = sum(bool(a["boundary_adapted_label_ids"]) for a in audits)
    stress_before = sum(round(base[k]["shared_hamming_error"] * len(val) * 5) for k in CONDITIONS[1:])
    stress_after = sum(round(candidate[k]["shared_hamming_error"] * len(val) * 5) for k in CONDITIONS[1:])
    checks = {
        "validation_clean_noninferior": candidate["clean"]["shared_hamming_error"]
        <= base["clean"]["shared_hamming_error"],
        "each_validation_stress_noninferior": all(
            candidate[k]["shared_hamming_error"] <= base[k]["shared_hamming_error"] for k in CONDITIONS[1:]
        ),
        "raw26_f1_noninferior": c["macro_f1"] >= b["macro_f1"] - 1e-12,
        "raw_normal_false_alarms_no_increase": c["normal_false_alarms"] <= b["normal_false_alarms"],
        "value": c["macro_f1"] >= b["macro_f1"] + 0.005 or (stress_after < stress_before and activated > 0),
    }
    write(
        "validation.json",
        {
            "checks": checks,
            "validation6": {
                "baseline": base,
                "candidate": candidate,
                "stress_error_positions_before": stress_before,
                "stress_error_positions_after": stress_after,
            },
            "raw26": {
                "baseline": b,
                "candidate": c,
                "activated_clips": activated,
                "changed_decisions": int((p != bp).sum()),
                "audits": audits,
            },
            "previously_inspected_regression": True,
        },
    )
    write(
        "decision.json",
        {
            "accepted": all(checks.values()),
            "checks": checks,
            "selected": selection["name"],
            "selection_sha256": sha(OUT / "selection.json"),
        },
    )
    print(
        json.dumps(
            {
                "checks": checks,
                "raw_f1_before": b["macro_f1"],
                "raw_f1_after": c["macro_f1"],
                "natural_activations": activated,
                "stress_before": stress_before,
                "stress_after": stress_after,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["select", "validate"])
    globals()[parser.parse_args().stage]()
