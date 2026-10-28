"""Finite train-only exploration of local kinematic protection; stable sources are never edited."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/branch_pipeline_v1"))
from pipeline import sha, skeleton_readout  # noqa: E402


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


old = module("accepted_fast_m1", ROOT / "experiments/m1_fast_v1/algorithm.py")
new = module("candidate_kinematic_m1", ROOT / "experiments/m1_kinematic_v1/algorithm.py")
helper = module("accepted_m1_eval", ROOT / "scripts/optimize_m1_fast.py")
DATA = ROOT / "data/public_cpr"
OUT = ROOT / "data/m1-m2-exploration-v1/m1"
CONFIGS = {
    f"bone_{t}_{v}": new.Config(bone_tolerance=t, bone_variation_factor=v)
    for t in [0.01, 0.02, 0.04]
    for v in [1.0, 2.0]
}
TRAIN_CONDITIONS = ["clean", "jitter", "mixed", "confidence_only"]
VALID_CONDITIONS = [*TRAIN_CONDITIONS, "jitter_seed2", "mixed_seed2"]


def write(name, value):
    payload = json.dumps(value, indent=2, allow_nan=False)
    with (OUT / name).open("x", encoding="utf-8") as f:
        f.write(payload)


def data():
    with np.load(DATA / "fullpose-v2/development.npz", allow_pickle=False) as f:
        x, y, splits, ids = [f[k].copy() for k in ["features", "targets", "splits", "record_ids"]]
    meta = {r["record_id"]: r for r in json.loads((DATA / "run/records.json").read_text())}
    train = np.asarray(
        sorted(
            np.flatnonzero(splits == "train"), key=lambda i: hashlib.sha256(str(ids[i]).encode()).hexdigest()
        )[:128]
    )
    val = np.flatnonzero(splits == "validation")
    return x, y.astype(bool), ids, meta, train, val


def trajectory(config):
    t = np.arange(121)
    p = np.repeat(np.array([[j * 3.0, j * 6.0] for j in range(17)])[None], len(t), axis=0)
    p[:, 11:13, 1] += 80
    q = np.full((len(t), 17), 0.95)
    theta = 0.7 * np.sin((t - 60) * 0.4)
    p[:, 9] = p[:, 7] + 60 * np.stack([np.cos(theta), np.sin(theta)], 1)
    q[60, 9] = 0.25
    a, _, _, _ = old.stabilize(p, q)
    b, _, _, audit = new.stabilize(p, q, config)
    old_error = float(np.linalg.norm(a[60, 9] - p[60, 9]))
    new_error = float(np.linalg.norm(b[60, 9] - p[60, 9]))
    return {
        "baseline_edit_error": old_error,
        "candidate_edit_error": new_error,
        "topology_rejected": audit["topology_rejected_corrections"],
        "scope": "Known generated constant-radius 2D joint trajectory; not CPR action ground truth.",
    }


def evaluate(indices, methods, conditions, phase):
    x, y, ids, meta, train, _ = data()
    with np.load(DATA / "body-mechanics-v3/features.npz", allow_pickle=False) as f:
        mm = {str(k): v for k, v in zip(f["record_ids"], f["features"])}
    scale = np.r_[x[train, 512:783].std(0), np.stack([mm[str(ids[i])] for i in train]).std(0)]
    scale[scale < 1e-3] = 1
    names = ["baseline", *methods]
    values = {name: {c: [] for c in conditions} for name in names}
    timings = {name: [] for name in names}
    audits = {name: {c: {"corrected": 0, "rejected": 0} for c in conditions} for name in names}
    for ordinal, i in enumerate(indices):
        row = meta[str(ids[i])]
        path = DATA / "run" / row["record_file"]
        assert sha(path) == row["record_file_sha256"]
        with np.load(path, allow_pickle=False) as f:
            p, q = f["keypoint"][0], f["keypoint_score"][0]
        for condition in conditions:
            pp, qq = helper.corrupt(p, q, condition, 20261022 + int(i))
            for name in names if ordinal % 2 == 0 else names[::-1]:
                tick = time.perf_counter()
                g, m, a = old.features(pp, qq) if name == "baseline" else new.features(pp, qq, methods[name])
                timings[name].append(time.perf_counter() - tick)
                values[name][condition].append(np.r_[g, m])
                audits[name][condition]["corrected"] += a["isolated_points_corrected"]
                audits[name][condition]["rejected"] += a.get("topology_rejected_corrections", 0)
        if (ordinal + 1) % 16 == 0:
            print(json.dumps({"phase": phase, "records": ordinal + 1, "total": len(indices)}), flush=True)
    normal = np.asarray([meta[str(ids[i])]["original_labels"] == [0] for i in indices])
    report = {}
    outputs = {}
    for name in names:
        stacked = {k: np.stack(v) for k, v in values[name].items()}
        report[name] = {"conditions": {}, "seconds": float(np.mean(timings[name])), "audit": audits[name]}
        outputs[name] = {}
        for condition, features in stacked.items():
            full = x[indices].copy()
            full[:, 512:783] = features[:, :271]
            prediction = skeleton_readout(full, features[:, 271:])
            outputs[name][condition] = prediction
            met = helper.metric(prediction["eligible"], y[indices], normal)
            if condition != "clean":
                met["descriptor_drift"] = float(
                    np.abs(np.clip((features - stacked["clean"]) / scale, -5, 5)).mean()
                )
            report[name]["conditions"][condition] = met
    return report, outputs


def assess(report, name, config):
    base = report["baseline"]["conditions"]
    candidate = report[name]["conditions"]
    noisy = [c for c in base if c != "clean"]
    gain = float(np.mean([candidate[c]["macro_f1"] - base[c]["macro_f1"] for c in noisy]))
    drift = float(
        np.mean(
            [
                candidate[c]["descriptor_drift"] / max(base[c]["descriptor_drift"], 1e-12)
                for c in noisy
                if c != "confidence_only"
            ]
        )
    )
    motion = trajectory(config)
    checks = {
        "clean_noninferior": candidate["clean"]["macro_f1"] >= base["clean"]["macro_f1"] - 1e-12,
        "each_stress_loss_le_002": all(
            candidate[c]["macro_f1"] >= base[c]["macro_f1"] - 0.002 for c in noisy
        ),
        "normal_false_alarms_no_increase": all(
            candidate[c]["normal_false_alarms"] <= base[c]["normal_false_alarms"] for c in base
        ),
        "drift_le_1_02": drift <= 1.02,
        "cpu_le_250ms": report[name]["seconds"] <= 0.25,
        "value": candidate["clean"]["macro_f1"] >= base["clean"]["macro_f1"] + 0.001
        or gain >= 0.001
        or drift <= 0.95
        or (
            motion["baseline_edit_error"] > 1
            and motion["candidate_edit_error"] <= motion["baseline_edit_error"] * 0.1
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "clean_gain": candidate["clean"]["macro_f1"] - base["clean"]["macro_f1"],
        "noisy_mean_gain": gain,
        "drift_ratio": drift,
        "trajectory": motion,
    }


def select():
    OUT.mkdir(exist_ok=False)
    _, _, ids, _, train, val = data()
    write(
        "protocol.json",
        {
            "algorithm_sha256": sha(ROOT / "experiments/m1_kinematic_v1/algorithm.py"),
            "evaluator_sha256": sha(Path(__file__)),
            "configs": {k: asdict(v) for k, v in CONFIGS.items()},
            "train_ids": [str(ids[i]) for i in train],
            "validation_ids": [str(ids[i]) for i in val],
            "train_conditions": TRAIN_CONDITIONS,
            "validation_conditions": VALID_CONDITIONS,
            "gate": "Clean F1 noninferior; per-stress loss<=.002; no extra normal FA; descriptor drift<=1.02x; feature CPU<=250ms; quality gain or >=90% less false edit on declared kinematic invariant. Rank eligible by clean F1, mean noisy F1, lower drift, conservative tolerance.",
            "selection": "128 original train records only; existing validation is reused engineering evidence, not a blind benchmark.",
        },
    )
    report, _ = evaluate(train, CONFIGS, TRAIN_CONDITIONS, "training")
    checks = {name: assess(report, name, cfg) for name, cfg in CONFIGS.items()}
    write("training.json", {"methods": report, "assessment": checks})
    allowed = [name for name in CONFIGS if checks[name]["passed"]]
    if not allowed:
        write(
            "decision.json",
            {"accepted": False, "reason": "No train-qualified kinematic candidate; stable M1 retained."},
        )
        print("No M1 candidate passed training gates", flush=True)
        return
    name = max(
        allowed,
        key=lambda n: (
            round(report[n]["conditions"]["clean"]["macro_f1"], 8),
            round(checks[n]["noisy_mean_gain"], 8),
            -checks[n]["drift_ratio"],
            CONFIGS[n].bone_tolerance,
            CONFIGS[n].bone_variation_factor,
        ),
    )
    write(
        "selection.json",
        {
            "name": name,
            "config": asdict(CONFIGS[name]),
            "selected_before_validation": True,
            "protocol_sha256": sha(OUT / "protocol.json"),
        },
    )
    print(json.dumps({"selected": name, "train_gain": checks[name]}), flush=True)


def validate():
    choice = json.loads((OUT / "selection.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert sha(ROOT / "experiments/m1_kinematic_v1/algorithm.py") == protocol["algorithm_sha256"]
    _, _, _, _, _, val = data()
    cfg = new.Config(**choice["config"])
    name = choice["name"]
    report, outputs = evaluate(val, {name: cfg}, VALID_CONDITIONS, "validation")
    checked = assess(report, name, cfg)
    write(
        "validation.json",
        {
            "passed": checked["passed"],
            "assessment": checked,
            "methods": report,
            "n": len(val),
            "scope": "Original validation196 reused; no new labels or independent blind claims.",
        },
    )
    write(
        "decision.json",
        {
            "accepted": checked["passed"],
            "selected": name,
            "selection_sha256": sha(OUT / "selection.json"),
            "checks": checked["checks"],
            "gains": {k: checked[k] for k in ["clean_gain", "noisy_mean_gain", "drift_ratio", "trajectory"]},
        },
    )
    np.savez_compressed(
        OUT / "validation-readouts.npz",
        **{
            name + "__" + c + "__" + k: a
            for name, conditions in outputs.items()
            for c, arrays in conditions.items()
            for k, a in arrays.items()
        },
    )
    print(json.dumps({"validation": checked}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["select", "validate"])
    globals()[parser.parse_args().stage]()
