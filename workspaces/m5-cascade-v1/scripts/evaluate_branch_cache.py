"""Compare decomposed branch readouts on the existing 26-video regression cache, without fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/branch_pipeline_v1"))
from pipeline import LABELS, merge_readouts, rgb_readout, sha, skeleton_readout  # noqa: E402


def load(path):
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key].copy() for key in bundle.files}


def metric(pred, target, columns):
    pred, target = np.asarray(pred, bool)[:, columns], target[:, columns]
    rows = []
    for i, label in enumerate(np.asarray(LABELS)[columns]):
        tp = int((pred[:, i] & target[:, i]).sum())
        fp = int((pred[:, i] & ~target[:, i]).sum())
        fn = int((~pred[:, i] & target[:, i]).sum())
        rows.append(
            {
                "label_id": int(label),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "recall": tp / max(tp + fn, 1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return {"macro_f1": float(np.mean([r["f1"] for r in rows])), "per_label": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the first evaluation")
    data = ROOT / "data/public_cpr"
    paths = [
        data / name
        for name in [
            "rgb-evaluation-v6/raw-predictions.npz",
            "body-mechanics-v3/raw_video-predictions.npz",
            "rgb-evaluation-v12/raw-predictions.npz",
            "hand-shape-v1/raw.npz",
            "rgb-evaluation-v15/raw-predictions.npz",
        ]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan = args.output.with_name("cache-evaluation-plan.json")
    with plan.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in paths},
                "gate": "26/26 final labels and score difference <=1e-6 against frozen v15; no new training",
                "scope": "Previously inspected engineering regression; feature replay is not real execution timing",
            },
            stream,
            indent=2,
        )
    full, motion, rgb, hands, reference = [load(path) for path in paths]
    ids = reference["record_ids"]

    def aligned(source, key):
        rows = {str(k): i for i, k in enumerate(source["record_ids"])}
        assert set(rows) == set(map(str, ids))
        return np.stack([source[key][rows[str(k)]] for k in ids])

    skeleton = skeleton_readout(aligned(full, "features"), aligned(motion, "features"))
    visual = rgb_readout(
        aligned(rgb, "rgb_features"), aligned(hands, "features"), aligned(hands, "available")
    )
    merged, _ = merge_readouts(skeleton, visual)
    # Reference targets are used only after every branch and fused prediction is fixed.
    y = reference["targets"]
    difference = float(np.max(np.abs(merged["scores"] - reference["scores"])))
    assert np.array_equal(merged["predictions"], reference["predictions"])
    assert difference <= 1e-6
    columns = [LABELS.index(int(i)) for i in visual["label_ids"]]
    rgb_full = np.zeros((len(ids), len(LABELS)), bool)
    rgb_full[:, columns] = visual["eligible"]
    disagreements = np.any(skeleton["eligible"][:, columns] != visual["eligible"], axis=1)
    skeleton_correct = skeleton["eligible"][:, columns] == y[:, columns]
    rgb_correct = visual["eligible"] == y[:, columns]
    result = {
        "passed": True,
        "records": len(ids),
        "model_or_threshold_modified": False,
        "new_training": False,
        "new_manual_labels": False,
        "independent_blind_test": False,
        "source_plan_sha256": sha(plan),
        "maximum_score_difference": difference,
        "final_predictions_identical": True,
        "rgb_supported_labels": visual["label_ids"].tolist(),
        "skeleton_all11": metric(skeleton["eligible"], y, list(range(11))),
        "fusion_all11": metric(merged["predictions"], y, list(range(11))),
        "common_scope": {
            "labels": visual["label_ids"].tolist(),
            "skeleton": metric(skeleton["eligible"], y, columns),
            "rgb_context": metric(rgb_full, y, columns),
            "fusion": metric(merged["predictions"], y, columns),
        },
        "disagreement_clips_in_common_scope": int(disagreements.sum()),
        "rgb_correct_skeleton_wrong_label_cells": int((rgb_correct & ~skeleton_correct).sum()),
        "skeleton_correct_rgb_wrong_label_cells": int((skeleton_correct & ~rgb_correct).sum()),
        "scope": "RGB is actor/geometry-conditioned and supports five categories; other six are not evaluated by this branch",
        "timing_claims": "None: these are cached features; real tool timings are recorded separately",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "passed",
                    "records",
                    "maximum_score_difference",
                    "final_predictions_identical",
                    "disagreement_clips_in_common_scope",
                    "rgb_correct_skeleton_wrong_label_cells",
                    "skeleton_correct_rgb_wrong_label_cells",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
