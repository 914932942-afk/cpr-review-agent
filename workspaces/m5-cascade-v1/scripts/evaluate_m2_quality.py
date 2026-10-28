"""Finite train-only selection; untouched validation and previously inspected raw-video regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "experiments/m2_quality_v1"),
    str(ROOT / "experiments/m1_fast_v1"),
    str(ROOT / "experiments/branch_pipeline_v1"),
]
from fusion_algorithm import Config, fuse  # noqa: E402
from m1_fast_pipeline import selected_config, transform  # noqa: E402
from pipeline import LABELS, merge_readouts, rgb_readout, sha, skeleton_readout  # noqa: E402

DATA = ROOT / "data/public_cpr"
OUT = ROOT / "data/m2-quality-v1"
SHARED = [1, 2, 5, 7, 9]


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(p, v):
    payload = json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False)
    with p.open("x", encoding="utf-8") as f:
        f.write(payload)


def load(p):
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def sliced(source, i, n):
    return {
        k: v[i : i + 1].copy()
        if k not in ["label_ids", "normal_veto_threshold"] and v.ndim and len(v) == n
        else v.copy()
        for k, v in source.items()
    }


def metric(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    tp, fp, fn = (pred & y).sum(0), (pred & ~y).sum(0), (~pred & y).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    cols = [LABELS.index(k) for k in SHARED]
    normal = ~y.any(1)
    return {
        "macro_f1": float(f1.mean()),
        "shared_macro_f1": float(f1[cols].mean()),
        "shared_hamming_error": float((pred[:, cols] != y[:, cols]).mean()),
        "exact_matches": int((pred == y).all(1).sum()),
        "normal_records": int(normal.sum()),
        "normal_false_alarms": int(pred[normal].any(1).sum()),
        "per_label": [
            {
                "id": k,
                "support": int((tp + fn)[i]),
                "f1": float(f1[i]),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
            }
            for i, k in enumerate(LABELS)
        ],
    }


def prepare():
    output = OUT / "prepared"
    output.mkdir(exist_ok=False)
    full = load(DATA / "fullpose-v2/development.npz")
    visual = load(DATA / "rgb-domain-features-v5/features.npz")
    hand = load(DATA / "hand-shape-v1/development.npz")
    manifest = read(DATA / "rgb-domain-features-v5/manifest.json")["records"]
    source = {r["record_id"]: r for r in read(DATA / "run/records.json")}
    fm = {str(k): i for i, k in enumerate(full["record_ids"])}
    hm = {str(k): i for i, k in enumerate(hand["record_ids"])}
    vm = {r["record_id"]: r for r in manifest}
    kept = [
        i
        for i, k in enumerate(visual["record_ids"])
        if str(k) in fm and not visual["partial_label_control"][i]
    ]
    ids = visual["record_ids"][kept]
    x = full["features"][[fm[str(k)] for k in ids]].copy()
    motions = []
    hashes = {}
    for j, key in enumerate(ids):
        row = source[str(key)]
        path = DATA / "run" / row["record_file"]
        assert sha(path) == row["record_file_sha256"]
        p = load(path)
        x[j : j + 1], motion, _ = transform(
            x[j : j + 1], p["keypoint"][0], p["keypoint_score"][0], selected_config()
        )
        motions.append(motion[0])
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    y = full["targets"][[fm[str(k)] for k in ids]].astype(bool)
    assert np.array_equal(y, visual["targets"][kept])
    splits = full["splits"][[fm[str(k)] for k in ids]]
    assert np.array_equal(splits, visual["splits"][kept])
    skeleton = skeleton_readout(x, np.stack(motions))
    rgbx = visual["features"][kept]
    handx = hand["features"][[hm[str(k)] for k in ids]]
    handok = hand["available"][[hm[str(k)] for k in ids]]
    used = np.asarray([vm[str(k)]["audit"]["frames"] for k in ids], int)
    requested = np.asarray([vm[str(k)]["audit"]["requested_frames"] for k in ids], int)
    train = splits == "train"
    scale = rgbx[train].std(0)
    scale[scale < 1e-6] = 1e-6
    arrays = dict(
        record_ids=ids,
        targets=y,
        splits=splits,
        rgb_features=rgbx,
        hand_features=handx,
        hand_available=handok,
        usable=used,
        requested=requested,
        noise_scale=scale,
    )
    arrays.update({"sk_" + k: v for k, v in skeleton.items()})
    np.savez_compressed(output / "development.npz", **arrays)
    train_ids = set(ids[train])
    val_ids = set(ids[~train])
    assert not train_ids & val_ids and set(splits) == {"train", "validation"}
    protected = [
        "fullpose-v2/development.npz",
        "rgb-domain-features-v5/features.npz",
        "rgb-domain-features-v5/manifest.json",
        "hand-shape-v1/development.npz",
        "rgb-evaluation-v4/record-splits.json",
    ]
    hashes.update({(DATA / k).relative_to(ROOT).as_posix(): sha(DATA / k) for k in protected})
    write(
        output / "audit.json",
        {
            "source_sha256": hashes,
            "development_sha256": sha(output / "development.npz"),
            "training_records": int(train.sum()),
            "validation_records": int((~train).sum()),
            "train_ids": sorted(map(str, train_ids)),
            "validation_ids": sorted(map(str, val_ids)),
            "excluded_unpaired_or_partial_controls": [
                str(k) for i, k in enumerate(visual["record_ids"]) if i not in kept
            ],
            "locked_test_opened": False,
            "new_labels": False,
            "source_scope": "Public original labels and cached features; official keypoints in development vs predicted keypoints in raw video. No cross-person claim.",
        },
    )
    print(
        "Prepared paired development",
        len(ids),
        "train",
        int(train.sum()),
        "validation",
        int((~train).sum()),
        flush=True,
    )


def condition(bundle, indices, name):
    skeleton = {k[3:]: v[indices] for k, v in bundle.items() if k.startswith("sk_")}
    x = bundle["rgb_features"][indices].copy()
    h = bundle["hand_features"][indices].copy()
    ok = bundle["hand_available"][indices].copy()
    used = bundle["usable"][indices].copy()
    requested = bundle["requested"][indices].copy()
    if name != "clean":
        for j, i in enumerate(indices):
            seed = int(hashlib.sha256((str(bundle["record_ids"][i]) + name).encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            # Controlled feature-level stress, not a claim of actually measured missing video frames.
            noise = 0.75 if name == "rgb_noise_half_support" else 1.5
            x[j] += rng.normal(0, noise, size=x.shape[1]) * bundle["noise_scale"]
            ok[j] = False  # the local hand expert is unavailable under this declared degradation condition
            used[j] = int(np.ceil(requested[j] * (0.5 if name == "rgb_noise_half_support" else 0.625)))
    rgb = rgb_readout(x, h, ok)
    return skeleton, rgb, used, requested, ok


def predict(skeleton, rgb, used, requested, hand, config):
    n = len(used)
    arrays = []
    audits = []
    for i in range(n):
        if config is None:
            a, _ = merge_readouts(sliced(skeleton, i, n), sliced(rgb, i, n))
            audit = {}
        else:
            a, audit = fuse(
                sliced(skeleton, i, n),
                sliced(rgb, i, n),
                int(used[i]),
                int(requested[i]),
                bool(hand[i]),
                config,
            )
        arrays.append(a["predictions"][0])
        audits.append(audit)
    return np.stack(arrays), audits


def evaluate_phase(bundle, indices, config, conditions):
    return {
        name: metric(predict(*condition(bundle, indices, name), config)[0], bundle["targets"][indices])
        for name in conditions
    }


def select():
    output = OUT / "experiment-01"
    output.mkdir(exist_ok=False)
    bundle = load(OUT / "prepared/development.npz")
    train = np.flatnonzero(bundle["splits"] == "train")
    configs = [Config(q, w) for q in [0.75, 0.875, 1.0] for w in [0.0, 0.25, 0.5]]
    conditions = ["clean", "rgb_noise_half_support", "rgb_noise_reduced_support"]
    protocol = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm_sha256": sha(ROOT / "experiments/m2_quality_v1/fusion_algorithm.py"),
        "evaluator_sha256": sha(Path(__file__)),
        "development_sha256": sha(OUT / "prepared/development.npz"),
        "candidate_configs": [asdict(c) for c in configs],
        "conditions": conditions,
        "selection": "Original training records only. Clean shared-F1 loss <=.005; then minimize mean stressed shared-label Hamming error, maximize stressed F1; prefer smaller full_rgb_ratio and higher RGB retention in ties. Validation is consulted once after freezing selection. Frozen branch heads previously fitted train data; this is not out-of-fold full-pipeline performance.",
        "gates": {
            "clean_shared_macro_f1_loss_max": 0.005,
            "stress_shared_hamming_relative_reduction_min": 0.1,
            "no_validation_stress_condition_worsens": True,
            "raw26_macro_f1_loss_max": 0.005,
            "raw26_normal_false_alarms_must_not_increase": True,
        },
        "scope": "Feature/readout robustness study, not real image occlusion, new data, cross-person or blind accuracy evidence. No backbone/classifier/threshold fitting. Only two gate hyperparameters selected.",
    }
    write(output / "protocol.json", protocol)
    baseline = evaluate_phase(bundle, train, None, conditions)
    candidates = []
    for config in configs:
        report = evaluate_phase(bundle, train, config, conditions)
        candidates.append(
            {
                "config": asdict(config),
                "metrics": report,
                "eligible": report["clean"]["shared_macro_f1"]
                >= baseline["clean"]["shared_macro_f1"] - 0.005,
            }
        )
    allowed = [r for r in candidates if r["eligible"]]
    if not allowed:
        raise ValueError("No train-qualified M2 candidate")
    selected = min(
        allowed,
        key=lambda r: (
            np.mean([r["metrics"][k]["shared_hamming_error"] for k in conditions[1:]]),
            -np.mean([r["metrics"][k]["shared_macro_f1"] for k in conditions[1:]]),
            r["config"]["full_rgb_ratio"],
            -r["config"]["minimum_rgb_weight"],
        ),
    )
    write(
        output / "selection.json",
        {
            "selected": selected["config"],
            "train_baseline": baseline,
            "candidates": candidates,
            "validation_used_for_selection": False,
            "protocol_sha256": sha(output / "protocol.json"),
        },
    )
    validate()


def validate():
    output = OUT / "experiment-01"
    protocol = read(output / "protocol.json")
    selection = read(output / "selection.json")
    assert sha(ROOT / "experiments/m2_quality_v1/fusion_algorithm.py") == protocol["algorithm_sha256"]
    assert sha(OUT / "prepared/development.npz") == protocol["development_sha256"]
    assert sha(output / "protocol.json") == selection["protocol_sha256"]
    bundle = load(OUT / "prepared/development.npz")
    validation = np.flatnonzero(bundle["splits"] == "validation")
    conditions = protocol["conditions"]
    config = Config(**selection["selected"])
    base = evaluate_phase(bundle, validation, None, conditions)
    result = evaluate_phase(bundle, validation, config, conditions)
    b = float(np.mean([base[k]["shared_hamming_error"] for k in conditions[1:]]))
    c = float(np.mean([result[k]["shared_hamming_error"] for k in conditions[1:]]))
    checks = {
        "clean_noninferior": result["clean"]["shared_macro_f1"] >= base["clean"]["shared_macro_f1"] - 0.005,
        "stress_improved": b > 0 and (b - c) / b >= 0.1,
        "no_stress_regression": all(
            result[k]["shared_hamming_error"] <= base[k]["shared_hamming_error"] for k in conditions[1:]
        ),
    }
    write(
        output / "validation.json",
        {
            "passed": all(checks.values()),
            "checks": checks,
            "baseline": base,
            "candidate": result,
            "stress_hamming_relative_reduction": float((b - c) / b) if b else None,
            "n": len(validation),
            "selected": selection["selected"],
            "scope": protocol["scope"],
        },
    )
    print(
        json.dumps(
            {
                "selected": selection["selected"],
                "checks": checks,
                "validation_clean_base": base["clean"]["shared_macro_f1"],
                "validation_clean_candidate": result["clean"]["shared_macro_f1"],
                "stress_relative_reduction": float((b - c) / b) if b else None,
            }
        ),
        flush=True,
    )


def raw():
    output = OUT / "experiment-01"
    selection = read(output / "selection.json")
    config = Config(**selection["selected"])
    reference = load(DATA / "rgb-evaluation-v15/raw-predictions.npz")
    body = load(DATA / "rgb-evaluation-v6/raw-predictions.npz")
    visual = load(DATA / "rgb-evaluation-v12/raw-predictions.npz")
    hand = load(DATA / "hand-shape-v1/raw.npz")
    ids = reference["record_ids"]
    n = len(ids)

    def align(source, key):
        m = {str(k): i for i, k in enumerate(source["record_ids"])}
        return np.stack([source[key][m[str(k)]] for k in ids])

    jobs = {r["record_id"]: r for r in read(DATA / "rgb-video-acceptance-v4/jobs.json")}
    source = {r["record_id"]: r for r in read(DATA / "fullpose-raw-v1/feature-manifest.json")["rows"]}
    audits = {
        r["record_id"]: r["audit"]
        for r in read(DATA / "rgb-domain-features-v5/raw26/manifest.json")["records"]
    }
    x = align(body, "features")
    motion = []
    for i, key in enumerate(ids):
        key = str(key)
        p = DATA / "rgb-video-acceptance-v4/runs" / jobs[key]["key"] / "inference.npz"
        assert sha(p) == source[key]["body_npz_sha256"]
        a = load(p)
        x[i : i + 1], m, _ = transform(
            x[i : i + 1], a["keypoint"][0], a["keypoint_score"][0], selected_config()
        )
        motion.append(m[0])
    sk = skeleton_readout(x, np.stack(motion))
    rg = rgb_readout(align(visual, "rgb_features"), align(hand, "features"), align(hand, "available"))
    used = np.asarray([audits[str(k)]["frames"] for k in ids])
    requested = np.asarray([audits[str(k)]["requested_frames"] for k in ids])
    ha = align(hand, "available")
    baseline, _ = predict(sk, rg, used, requested, ha, None)
    pred, audit = predict(sk, rg, used, requested, ha, config)
    assert np.array_equal(baseline, reference["predictions"])
    y = reference["targets"]
    b = metric(baseline, y)
    c = metric(pred, y)
    checks = {
        "raw26_coverage": n == 26,
        "macro_f1_noninferior": c["macro_f1"] >= max(0.85, b["macro_f1"] - 0.005),
        "normal_false_alarms_no_increase": c["normal_false_alarms"] <= b["normal_false_alarms"],
    }
    write(
        output / "raw-regression.json",
        {
            "passed": all(checks.values()),
            "checks": checks,
            "baseline": b,
            "candidate": c,
            "changed_decisions": int((pred != baseline).sum()),
            "audits": audit,
            "scope": "Previously inspected actual-video feature regression; no selection performed on these 26 records. Public labels unchanged. Not a blind test.",
        },
    )
    np.savez_compressed(
        output / "raw-readouts.npz",
        record_ids=ids,
        targets=y,
        **{"sk_" + k: v for k, v in sk.items()},
        **{"rgb_" + k: v for k, v in rg.items()},
        usable=used,
        requested=requested,
        hand_available=ha,
    )
    print(
        json.dumps(
            {
                "raw_checks": checks,
                "baseline_f1": b["macro_f1"],
                "candidate_f1": c["macro_f1"],
                "changed_decisions": int((pred != baseline).sum()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "select", "validate", "raw"])
    globals()[parser.parse_args().stage]()
