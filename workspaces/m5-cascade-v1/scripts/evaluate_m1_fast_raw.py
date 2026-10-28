"""Frozen M1 selection evaluated once on the previously inspected 26 raw-video caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/m1_fast_v1"))
from m1_fast_pipeline import selected_config, transform  # noqa: E402
from pipeline import LABELS, merge_readouts, rgb_readout, sha, skeleton_readout  # noqa: E402


def load(path):
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key].copy() for key in bundle.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve M1 raw regression")
    data = ROOT / "data/public_cpr"
    selection = ROOT / "data/m1-optimization-v2/experiment-03/selection.json"
    config = selected_config()
    reference = load(data / "rgb-evaluation-v15/raw-predictions.npz")
    raw = load(data / "rgb-evaluation-v6/raw-predictions.npz")
    visual = load(data / "rgb-evaluation-v12/raw-predictions.npz")
    hands = load(data / "hand-shape-v1/raw.npz")
    jobs = {
        r["record_id"]: r
        for r in json.loads((data / "rgb-video-acceptance-v4/jobs.json").read_text(encoding="utf-8"))
    }
    manifest = {
        r["record_id"]: r
        for r in json.loads((data / "fullpose-raw-v1/feature-manifest.json").read_text(encoding="utf-8"))[
            "rows"
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan = {
        "selection_sha256": sha(selection),
        "gate": "Original all26 quality gates plus clean fused macro-F1 loss <=.005 against frozen v15",
        "inputs": {
            name: sha(data / name)
            for name in [
                "rgb-evaluation-v15/raw-predictions.npz",
                "rgb-evaluation-v6/raw-predictions.npz",
                "rgb-evaluation-v12/raw-predictions.npz",
                "hand-shape-v1/raw.npz",
                "fullpose-raw-v1/feature-manifest.json",
            ]
        },
        "no_new_training_or_label_changes": True,
        "previously_inspected_engineering_regression": True,
    }
    args.output.with_name("raw-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    ids = reference["record_ids"]

    def aligned(source, key):
        lookup = {str(k): i for i, k in enumerate(source["record_ids"])}
        return np.stack([source[key][lookup[str(k)]] for k in ids])

    full = aligned(raw, "features")
    motions = []
    audits = []
    for i, key in enumerate(ids):
        key = str(key)
        job = jobs[key]
        path = data / "rgb-video-acceptance-v4/runs" / job["key"] / "inference.npz"
        assert sha(path) == manifest[key]["body_npz_sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            transformed, motion, audit = transform(
                full[i : i + 1], arrays["keypoint"][0], arrays["keypoint_score"][0], config
            )
        full[i] = transformed[0]
        motions.append(motion[0])
        audits.append(audit)
    skeleton = skeleton_readout(full, np.stack(motions))
    rgb = rgb_readout(
        aligned(visual, "rgb_features"), aligned(hands, "features"), aligned(hands, "available")
    )
    result, _ = merge_readouts(skeleton, rgb)
    # Targets are consulted after the complete prediction is fixed.
    y = reference["targets"].astype(bool)
    pred = result["predictions"]

    def metric(p):
        tp, fp, fn = (p & y).sum(0), (p & ~y).sum(0), (~p & y).sum(0)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        recall = tp / np.maximum(tp + fn, 1)
        normal = ~y.any(1)
        return {
            "macro_f1": float(f1.mean()),
            "minimum_label_recall": float(recall.min()),
            "normal_false_alarm_rate": float(p[normal].any(1).mean()),
            "exact_matches": int((p == y).all(1).sum()),
            "per_label": [
                {"id": label, "f1": float(f), "recall": float(r)} for label, f, r in zip(LABELS, f1, recall)
            ],
        }

    baseline = metric(reference["predictions"])
    candidate = metric(pred)
    gates = {
        "all26_coverage": len(ids) == 26,
        "macro_f1": candidate["macro_f1"] >= max(0.85, baseline["macro_f1"] - 0.005),
        "minimum_recall": candidate["minimum_label_recall"] >= 0.6,
        "normal_false_alarm": candidate["normal_false_alarm_rate"] <= 0.1,
    }
    report = {
        "accepted": all(gates.values()),
        "checks": gates,
        "baseline": baseline,
        "candidate": candidate,
        "selection_sha256": sha(selection),
        "new_training": False,
        "new_manual_labels": False,
        "scope": "Reused real-video features; no independent blind or new end-to-end accuracy benchmark",
        "changed_label_decisions": int((pred != reference["predictions"]).sum()),
        "processing_audits": audits,
        "rows": [
            {
                "record_id": str(k),
                "predicted": [label for label, yes in zip(LABELS, pred[i]) if yes],
                "scores": result["scores"][i, :11].tolist(),
                "rgb_scores": rgb["scores"][i].tolist(),
                "skeleton_scores": skeleton["scores"][i, :11].tolist(),
            }
            for i, k in enumerate(ids)
        ],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {k: report[k] for k in ["accepted", "checks", "baseline", "candidate", "changed_label_decisions"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
