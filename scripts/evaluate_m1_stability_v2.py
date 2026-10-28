"""Train-only candidate selection, then one validation of skeleton-derived feature robustness."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/m1_stability_v2"))
sys.path.insert(0, str(ROOT / "experiments/branch_pipeline_v1"))
from algorithm import Config, features, stabilize  # noqa: E402
from pipeline import LABELS, frozen, sha, skeleton_readout  # noqa: E402

geometry = importlib.import_module("geometry")

DATA = ROOT / "data/public_cpr"
CONFIGS = {
    "conservative_half": Config(correction_strength=0.5),
    "conservative_full": Config(correction_strength=1.0),
}
CONDITIONS = ["clean", "low_confidence_jitter", "jitter_and_short_gaps"]


def metric(pred, y, normal):
    tp, fp, fn = (pred & y).sum(0), (pred & ~y).sum(0), (~pred & y).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return {
        "macro_f1": float(f1.mean()),
        "normal_false_alarms": int(pred[normal].any(1).sum()),
        "normal_count": int(normal.sum()),
        "per_label": [
            {"id": label, "f1": float(f), "recall": float(t / max(t + n, 1))}
            for label, f, t, n in zip(LABELS, f1, tp, fn)
        ],
    }


def corrupt(points, confidence, condition, seed):
    p, q = points.copy(), confidence.copy()
    if condition == "clean":
        return p, q
    rng = np.random.default_rng(seed)
    hip, shoulder = p[:, [11, 12]].mean(1), p[:, [5, 6]].mean(1)
    scales = np.linalg.norm(hip - shoulder, axis=1)
    good = (q[:, [5, 6, 11, 12]] >= 0.4).all(1) & (scales > 1)
    scale = float(np.median(scales[good])) if good.any() else 100.0
    mask = (rng.random(q.shape) < 0.03) & (q >= 0.45)
    mask[:2] = False
    mask[-2:] = False
    p[mask] += rng.normal(0, 0.15 * scale, size=(int(mask.sum()), 2))
    q[mask] = 0.25
    if condition == "jitter_and_short_gaps":
        for joint in rng.choice(17, 5, replace=False):
            start = int(rng.integers(3, len(p) - 6))
            p[start : start + 3, joint] = 0
            q[start : start + 3, joint] = 0
    return p, q


def invariants():
    t = np.arange(120)
    base = np.array([[j * 3.0, j * 6.0] for j in range(17)])
    p = base[None] + np.stack([2 * np.sin(t / 6), 3 * np.cos(t / 6)], 1)[:, None]
    p[:, 11:13, 1] += 80
    q = np.full((120, 17), 0.95)
    a, _, _, _ = stabilize(p, q)
    assert np.array_equal(a, p), "High-confidence movement changed"
    noisy = p.copy()
    nq = q.copy()
    noisy[40, 9] += [70, 60]
    nq[40, 9] = 0.25
    filtered, _, _, audit = stabilize(noisy, nq)
    assert audit["isolated_points_corrected"] >= 1
    assert np.linalg.norm(filtered[40, 9] - p[40, 9]) < np.linalg.norm(noisy[40, 9] - p[40, 9]) * 0.1
    missing = p.copy()
    mq = q.copy()
    missing[30:33, 7] = 0
    mq[30:33, 7] = 0
    repaired, _, supported, audit = stabilize(missing, mq)
    assert supported[30:33, 7].all() and audit["short_gap_points_reconstructed"] == 3
    missing[70:82, 8] = 0
    mq[70:82, 8] = 0
    repaired, _, supported, _ = stabilize(missing, mq)
    assert not supported[70:82, 8].any() and np.isnan(repaired[70:82, 8]).all()
    geometry_new, motion_new, _ = features(p, q)
    np.testing.assert_allclose(geometry_new, geometry.describe(p, q)[0], atol=1e-5)
    np.testing.assert_allclose(motion_new, frozen.mechanics_features(p, q), atol=1e-5)
    zeros = np.zeros_like(q)
    geometry_new, motion_new, audit = features(p, zeros)
    assert not audit["geometry_available"] and not geometry_new.any() and not motion_new.any()
    for bad_p, bad_q in [(p[:5], q[:5]), (p, q * 2), (p[:, :16], q[:, :16])]:
        try:
            stabilize(bad_p, bad_q)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid skeleton input accepted")
    return {
        "passed": True,
        "checks": [
            "high-confidence motion unchanged",
            "isolated weaker-confidence spike corrected",
            "short gap repaired",
            "long gap retained as unavailable",
            "compatible clean geometry and mechanics",
            "missing reference produces unavailable zero descriptors",
            "invalid shapes/confidence rejected",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the first M1 experiment")
    args.output.mkdir(parents=True)
    invariants_result = invariants()
    source = DATA / "fullpose-v2/development.npz"
    with np.load(source, allow_pickle=False) as bundle:
        x, y, splits, ids = [bundle[k].copy() for k in ["features", "targets", "splits", "record_ids"]]
    metadata = {
        r["record_id"]: r for r in json.loads((DATA / "run/records.json").read_text(encoding="utf-8"))
    }
    with np.load(DATA / "body-mechanics-v3/features.npz", allow_pickle=False) as bundle:
        motion_map = {str(key): value.copy() for key, value in zip(bundle["record_ids"], bundle["features"])}
    train = np.flatnonzero(splits == "train")
    train = np.asarray(sorted(train, key=lambda i: hashlib.sha256(str(ids[i]).encode()).hexdigest())[:128])
    validation = np.flatnonzero(splits == "validation")
    paths = [
        ROOT / "experiments/m1_stability_v2/algorithm.py",
        Path(__file__).resolve(),
        source,
        DATA / "body-mechanics-v3/features.npz",
        DATA / "rgb-evaluation-v4/record-splits.json",
    ]
    protocol = {
        "schema": "cpr-m1-stability-protocol-v2",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        "configurations": {name: asdict(c) for name, c in CONFIGS.items()},
        "conditions": CONDITIONS,
        "selection": "128 deterministic original training clips only; no validation/test selection",
        "train_ids": [str(ids[i]) for i in train],
        "validation_ids": [str(ids[i]) for i in validation],
        "gates": {
            "clean_macro_f1_loss_max": 0.005,
            "corrupted_macro_f1_loss_max": 0.005,
            "mean_descriptor_drift_reduction_min": 0.15,
            "feature_cpu_seconds_per_clip_max": 0.25,
            "normal_false_alarms_must_not_increase": True,
        },
        "labels": "Unmodified public labels; deterministic artificial sensor errors are not new action annotations",
        "unchanged": "PoseC3D512, hand632, spectrum94, learned weights, thresholds and RGB/fusion",
        "modified": "Skeleton geometry271 and mechanics176 only",
        "scope": "Feature-level perturbation robustness; no claim of pose ground truth or camera generalization",
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "invariants.json").write_text(json.dumps(invariants_result, indent=2), encoding="utf-8")

    def evaluate(indices, phase):
        normal = np.asarray([metadata[str(ids[i])]["original_labels"] == [0] for i in indices])
        truth = y[indices].astype(bool)
        motion_clean = np.stack([motion_map[str(ids[i])] for i in indices])
        scale = np.r_[x[train, 512:783].std(0), np.stack([motion_map[str(ids[i])] for i in train]).std(0)]
        scale[scale < 1e-3] = 1
        arrays = {name: {condition: [] for condition in CONDITIONS} for name in ["baseline", *CONFIGS]}
        durations = {name: [] for name in CONFIGS}
        audits = {
            name: {"isolated_points_corrected": 0, "short_gap_points_reconstructed": 0} for name in CONFIGS
        }
        for ordinal, i in enumerate(indices):
            row = metadata[str(ids[i])]
            path = DATA / "run" / row["record_file"]
            if sha(path) != row["record_file_sha256"]:
                raise ValueError("Public skeleton changed")
            with np.load(path, allow_pickle=False) as bundle:
                points, q = bundle["keypoint"][0].copy(), bundle["keypoint_score"][0].copy()
            for condition in CONDITIONS:
                p, c = corrupt(points, q, condition, 20261022 + int(i))
                g = geometry.describe(p, c)[0]
                m = frozen.mechanics_features(p, c)
                if condition == "clean":
                    np.testing.assert_allclose(g, x[i, 512:783], atol=1e-4)
                    np.testing.assert_allclose(m, motion_clean[ordinal], atol=1e-4)
                arrays["baseline"][condition].append(np.r_[g, m])
                for name, config in CONFIGS.items():
                    start = time.perf_counter()
                    g, m, audit = features(p, c, config)
                    durations[name].append(time.perf_counter() - start)
                    arrays[name][condition].append(np.r_[g, m])
                    for key in audits[name]:
                        audits[name][key] += audit[key]
            if (ordinal + 1) % 40 == 0:
                print(json.dumps({"phase": phase, "processed": ordinal + 1, "of": len(indices)}), flush=True)
        report = {}
        for name, conditions in arrays.items():
            values = {condition: np.stack(rows) for condition, rows in conditions.items()}
            report[name] = {"conditions": {}}
            for condition, data in values.items():
                full = x[indices].copy()
                full[:, 512:783] = data[:, :271]
                result = skeleton_readout(full, data[:, 271:])
                measured = metric(result["eligible"], truth, normal)
                if condition != "clean":
                    measured["descriptor_drift"] = float(
                        np.abs(np.clip((data - values["clean"]) / scale, -5, 5)).mean()
                    )
                report[name]["conditions"][condition] = measured
            if name in durations:
                report[name]["mean_feature_seconds"] = float(np.mean(durations[name]))
                report[name]["max_feature_seconds"] = float(np.max(durations[name]))
                report[name]["processing_audit"] = audits[name]
        return report

    training = evaluate(train, "training_selection")
    (args.output / "training.json").write_text(json.dumps(training, indent=2), encoding="utf-8")

    def passed(report, name):
        base, method = report["baseline"]["conditions"], report[name]["conditions"]
        reductions = [
            1 - method[c]["descriptor_drift"] / max(base[c]["descriptor_drift"], 1e-12)
            for c in CONDITIONS[1:]
        ]
        checks = {
            "clean_classification": method["clean"]["macro_f1"] >= base["clean"]["macro_f1"] - 0.005,
            "corrupted_classification": all(
                method[c]["macro_f1"] >= base[c]["macro_f1"] - 0.005 for c in CONDITIONS[1:]
            ),
            "normal_false_alarms": all(
                method[c]["normal_false_alarms"] <= base[c]["normal_false_alarms"] for c in CONDITIONS
            ),
            "descriptor_stability": float(np.mean(reductions)) >= 0.15,
            "cpu_cost": report[name]["mean_feature_seconds"] <= 0.25,
        }
        return all(checks.values()), checks, float(np.mean(reductions))

    allowed = [name for name in CONFIGS if passed(training, name)[0]]
    if not allowed:
        result = {
            "accepted": False,
            "stage": "training_selection",
            "reason": "No candidate met the preregistered gates",
            "candidate_checks": {name: passed(training, name)[1] for name in CONFIGS},
        }
        (args.output / "decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        return
    selected = max(
        allowed, key=lambda name: (passed(training, name)[2], -training[name]["mean_feature_seconds"])
    )
    selection = {
        "name": selected,
        "config": asdict(CONFIGS[selected]),
        "selection_source": "training128 only",
        "validation_used_for_selection": False,
        "selected_before_validation": True,
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    validation_report = evaluate(validation, "validation")
    (args.output / "validation.json").write_text(json.dumps(validation_report, indent=2), encoding="utf-8")
    accepted, checks, reduction = passed(validation_report, selected)
    result = {
        "accepted": accepted,
        "selected": selected,
        "checks": checks,
        "validation_descriptor_drift_reduction": reduction,
        "validation_clean_baseline_f1": validation_report["baseline"]["conditions"]["clean"]["macro_f1"],
        "validation_clean_candidate_f1": validation_report[selected]["conditions"]["clean"]["macro_f1"],
        "validation_clips": len(validation),
        "train_selection_clips": len(train),
        "main_backbone_training": False,
        "new_manual_labels": False,
        "full_pipeline_or_camera_generalization_claim": False,
    }
    (args.output / "decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
