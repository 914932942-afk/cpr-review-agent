"""Finite train-only M1 candidate selection; one selected candidate is checked on existing validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/m1_fast_v1"))
sys.path.insert(0, str(ROOT / "experiments/branch_pipeline_v1"))
from algorithm import Config, features, stabilize  # noqa: E402
from pipeline import LABELS, sha, skeleton_readout  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "accepted_m1_v3", ROOT / "experiments/m1_stability_v3/algorithm.py"
)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)

DATA = ROOT / "data/public_cpr"
OUTPUT = ROOT / "data/m1-optimization-v2/experiment-03"
CONFIGS = {"vectorized": Config()}

TRAIN_CONDITIONS = ["clean", "jitter", "mixed", "confidence_only"]
VALIDATION_CONDITIONS = [*TRAIN_CONDITIONS, "jitter_seed2", "mixed_seed2"]


def save(name, value):
    with (OUTPUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def corrupt(points, confidence, condition, seed):
    p, q = points.copy(), confidence.copy()
    if condition == "clean":
        return p, q
    rng = np.random.default_rng(seed + (1 if condition.endswith("seed2") else 0))
    size = np.linalg.norm(p[:, [11, 12]].mean(1) - p[:, [5, 6]].mean(1), axis=1)
    good = (q[:, [5, 6, 11, 12]] >= 0.4).all(1) & (size > 1)
    scale = float(np.median(size[good])) if good.any() else 100.0
    mask = (rng.random(q.shape) < 0.03) & (q >= 0.45)
    mask[:2] = mask[-2:] = False
    if condition != "confidence_only":
        p[mask] += rng.normal(0, 0.15 * scale, size=(int(mask.sum()), 2))
    q[mask] = 0.25
    if condition.startswith("mixed"):
        for joint in rng.choice(17, 5, replace=False):
            start = int(rng.integers(3, len(p) - 6))
            p[start : start + 3, joint], q[start : start + 3, joint] = 0, 0
    return p, q


def metric(pred, truth, normal):
    tp, fp, fn = (pred & truth).sum(0), (pred & ~truth).sum(0), (~pred & truth).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return {
        "macro_f1": float(f1.mean()),
        "normal_false_alarms": int(pred[normal].any(1).sum()),
        "normal_count": int(normal.sum()),
        "per_label": [{"id": k, "f1": float(v)} for k, v in zip(LABELS, f1)],
    }


def gates(report, name, timing=True):
    base, candidate = report["current_m1_v3"], report[name]
    bc, cc = base["conditions"], candidate["conditions"]
    conditions = [key for key in bc if key != "clean"]
    drift_ratio = float(
        np.mean(
            [
                cc[k]["descriptor_drift"] / max(bc[k]["descriptor_drift"], 1e-12)
                for k in conditions
                if k != "confidence_only"
            ]
        )
    )
    clean_gain = cc["clean"]["macro_f1"] - bc["clean"]["macro_f1"]
    noisy_gain = float(np.mean([cc[k]["macro_f1"] - bc[k]["macro_f1"] for k in conditions]))
    speedup = 1 - candidate["mean_feature_seconds"] / base["mean_feature_seconds"]
    checks = {
        "clean_noninferior": clean_gain >= -1e-12,
        "each_perturbation_loss_le_002": all(
            cc[k]["macro_f1"] >= bc[k]["macro_f1"] - 0.002 for k in conditions
        ),
        "normal_false_alarms_no_increase": all(
            cc[k]["normal_false_alarms"] <= bc[k]["normal_false_alarms"] for k in bc
        ),
        "descriptor_drift_ratio_le_1_02": drift_ratio <= 1.02,
        "prediction_and_score_equivalence": candidate["prediction_equivalence"]
        and candidate["maximum_score_difference"] <= 1e-6,
        "feature_cpu_le_250ms": candidate["mean_feature_seconds"] <= 0.25,
        "measurable_improvement": clean_gain >= 0.001
        or noisy_gain >= 0.001
        or drift_ratio <= 0.90
        or (timing and speedup >= 0.30),
    }
    return (
        all(checks.values()),
        checks,
        {
            "clean_f1_gain": clean_gain,
            "mean_perturbed_f1_gain": noisy_gain,
            "descriptor_drift_ratio": drift_ratio,
            "feature_cpu_reduction": speedup,
        },
    )


def invariants():
    t = np.arange(120)
    p = (
        np.array([[j * 3.0, j * 6.0] for j in range(17)])[None]
        + np.stack([2 * np.sin(t / 6), 3 * np.cos(t / 6)], 1)[:, None]
    )
    p[:, 11:13, 1] += 80
    q = np.full((120, 17), 0.95)
    checks = []
    for name, config in CONFIGS.items():
        result, _, _, _ = stabilize(p, q, config)
        assert np.array_equal(result, p), name
        missing, confidence = p.copy(), q.copy()
        missing[30:33, 7], confidence[30:33, 7] = 0, 0
        missing[70:82, 8], confidence[70:82, 8] = 0, 0
        filtered, _, support, audit = stabilize(missing, confidence, config)
        assert support[30:33, 7].all() and not support[70:82, 8].any()
        assert np.isnan(filtered[70:82, 8]).all() and not audit["reconstructed_points_are_observations"]
        g, m, _ = features(missing, confidence, config)
        assert g.shape == (271,) and m.shape == (176,) and np.isfinite(g).all() and np.isfinite(m).all()
        for source_p, source_q in [(p, q), (missing, confidence), (p, np.zeros_like(q))]:
            if name == "vectorized":
                a, b, _ = features(source_p, source_q, config)
                ag, bg, _ = old.features(source_p, source_q)
                np.testing.assert_allclose(a, ag, atol=1e-6)
                np.testing.assert_allclose(b, bg, atol=1e-6)
        checks.append(name + ": high confidence, gaps, finite layout")
    # Numerical fidelity with partial visibility, isolated jumps and changing motion.
    rng = np.random.default_rng(20261023)
    for _ in range(12):
        varied = p + rng.normal(0, 4, p.shape)
        confidence = rng.uniform(0.1, 1, q.shape)
        a, b, _ = features(varied, confidence)
        ag, bg, _ = old.features(varied, confidence)
        np.testing.assert_allclose(a, ag, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(b, bg, atol=1e-6, rtol=1e-6)
    checks.append("12 randomized partial-visibility numerical equivalence cases")
    return {"passed": True, "checks": checks}


def main():
    OUTPUT.mkdir(exist_ok=False)
    with np.load(DATA / "fullpose-v2/development.npz", allow_pickle=False) as bundle:
        x, y, splits, ids = [bundle[k].copy() for k in ["features", "targets", "splits", "record_ids"]]
    train = np.asarray(
        sorted(
            np.flatnonzero(splits == "train"), key=lambda i: hashlib.sha256(str(ids[i]).encode()).hexdigest()
        )[:128]
    )
    validation = np.flatnonzero(splits == "validation")
    metadata = {
        r["record_id"]: r for r in json.loads((DATA / "run/records.json").read_text(encoding="utf-8"))
    }
    with np.load(DATA / "body-mechanics-v3/features.npz", allow_pickle=False) as bundle:
        motion = {str(k): v for k, v in zip(bundle["record_ids"], bundle["features"])}
    scale = np.r_[x[train, 512:783].std(0), np.stack([motion[str(ids[i])] for i in train]).std(0)]
    scale[scale < 1e-3] = 1
    paths = [
        Path(__file__).resolve(),
        ROOT / "experiments/m1_fast_v1/algorithm.py",
        ROOT / "experiments/m1_stability_v3/algorithm.py",
        DATA / "fullpose-v2/development.npz",
        DATA / "body-mechanics-v3/features.npz",
        ROOT / "data/m1-stability-v1/pipeline-manifest.json",
    ]
    save(
        "protocol.json",
        {
            "schema": "cpr-m1-optimization-fast-v1",
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
            "configurations": {name: asdict(c) for name, c in CONFIGS.items()},
            "train_ids": [str(ids[i]) for i in train],
            "validation_ids": [str(ids[i]) for i in validation],
            "train_conditions": TRAIN_CONDITIONS,
            "validation_conditions": VALIDATION_CONDITIONS,
            "selection": "Finite candidates on original training128 only; freeze one before validation",
            "ranking": "clean F1, mean perturbed F1, negative drift, CPU reduction; rounded F1 at 1e-8",
            "gates": "Predictions identical, score error<=1e-6; clean F1 noninferior; each perturbation loss<=.002; normal FA cannot increase; drift<=1.02x; CPU<=.25s; positive quality/stability or >=30% CPU gain",
            "scope": "Previously used validation split; repeated engineering validation, not an independent blind accuracy benchmark",
            "no_new_training_or_manual_labels": True,
        },
    )
    save("invariants.json", invariants())

    def evaluate(indices, methods, conditions, phase):
        arrays = {name: {c: [] for c in conditions} for name in ["current_m1_v3", *methods]}
        durations = {name: [] for name in arrays}
        counts = {name: {c: 0 for c in conditions} for name in arrays}
        for ordinal, i in enumerate(indices):
            row = metadata[str(ids[i])]
            path = DATA / "run" / row["record_file"]
            if sha(path) != row["record_file_sha256"]:
                raise ValueError("Public skeleton changed")
            with np.load(path, allow_pickle=False) as bundle:
                points, q = bundle["keypoint"][0], bundle["keypoint_score"][0]
            for condition in conditions:
                p, confidence = corrupt(points, q, condition, 20261022 + int(i))
                names = list(arrays)
                if ordinal % 2:
                    names.reverse()
                for name in names:
                    start = time.perf_counter()
                    if name == "current_m1_v3":
                        g, m, audit = old.features(p, confidence)
                    else:
                        g, m, audit = features(p, confidence, methods[name])
                    durations[name].append(time.perf_counter() - start)
                    arrays[name][condition].append(np.r_[g, m])
                    counts[name][condition] += audit["isolated_points_corrected"]
            if (ordinal + 1) % 40 == 0:
                print(json.dumps({"phase": phase, "processed": ordinal + 1, "of": len(indices)}), flush=True)
        normal = np.asarray([metadata[str(ids[i])]["original_labels"] == [0] for i in indices])
        report = {}
        predictions, scores = {}, {}
        for name, conditions in arrays.items():
            values = {c: np.stack(rows) for c, rows in conditions.items()}
            report[name] = {
                "conditions": {},
                "mean_feature_seconds": float(np.mean(durations[name])),
                "corrections": counts[name],
            }
            predictions[name], scores[name] = {}, {}
            for condition, value in values.items():
                full = x[indices].copy()
                full[:, 512:783] = value[:, :271]
                result = skeleton_readout(full, value[:, 271:])
                predictions[name][condition], scores[name][condition] = result["eligible"], result["scores"]
                measured = metric(result["eligible"], y[indices].astype(bool), normal)
                if condition != "clean":
                    measured["descriptor_drift"] = float(
                        np.abs(np.clip((value - values["clean"]) / scale, -5, 5)).mean()
                    )
                report[name]["conditions"][condition] = measured
        for name in report:
            report[name]["prediction_equivalence"] = all(
                np.array_equal(p, predictions["current_m1_v3"][key]) for key, p in predictions[name].items()
            )
            report[name]["maximum_score_difference"] = max(
                float(np.abs(p - scores["current_m1_v3"][key]).max()) for key, p in scores[name].items()
            )
        return report

    training = evaluate(train, CONFIGS, TRAIN_CONDITIONS, "training_only")
    save("training.json", training)
    passing = [name for name in CONFIGS if gates(training, name)[0]]
    if not passing:
        save(
            "decision.json",
            {
                "accepted": False,
                "stage": "training",
                "checks": {name: gates(training, name)[1] for name in CONFIGS},
            },
        )
        print("No candidate passed training gates; active v3 unchanged", flush=True)
        return

    def rank(name):
        _, _, gains = gates(training, name)
        return (
            round(gains["clean_f1_gain"], 8),
            round(gains["mean_perturbed_f1_gain"], 8),
            -gains["descriptor_drift_ratio"],
            gains["feature_cpu_reduction"],
        )

    selected = max(passing, key=rank)
    save(
        "selection.json",
        {
            "name": selected,
            "config": asdict(CONFIGS[selected]),
            "selected_before_validation": True,
            "selection_source": "training128 only",
        },
    )
    validation_report = evaluate(
        validation, {selected: CONFIGS[selected]}, VALIDATION_CONDITIONS, "validation_selected_only"
    )
    save("validation.json", validation_report)
    accepted, checks, gains = gates(validation_report, selected)
    result = {
        "accepted": accepted,
        "selected": selected,
        "checks": checks,
        "gains": gains,
        "validation_clips": len(validation),
        "new_training": False,
        "new_manual_labels": False,
    }
    save("decision.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
