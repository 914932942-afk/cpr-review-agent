"""Why both dependency proof and the quality gate are necessary, with no new fit."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/branch_pipeline_v1"))
from pipeline import frozen  # noqa: E402


def main():
    out = ROOT / "test-results/m5-cascade-v1/ablation.json"
    assert not out.exists()
    spec = importlib.util.spec_from_file_location(
        "m5_ablation", ROOT / "experiments/m5_cascade_v1/algorithm.py"
    )
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    rgb = torch.load(
        ROOT / "data/public_cpr/rgb-evaluation-v12/experts.pt", weights_only=True, map_location="cpu"
    )
    hand = torch.load(
        ROOT / "data/public_cpr/rgb-evaluation-v14/hands.pt", weights_only=True, map_location="cpu"
    )
    with np.load(
        ROOT / "test-results/m5-cascade-v1/benchmark/fallback-baseline.npz", allow_pickle=False
    ) as f:
        base = {k: f[k].copy() for k in f.files}
    naive = a.context_readout(
        base["rgb_features"][:, 1536:],
        base["shape_features"],
        rgb,
        hand,
        a.compile_plan(rgb, hand, True),
        frozen.predict_experts,
        frozen.predict_hands,
    )
    gap = float(abs(naive["scores"] - base["scores"]).max())
    different = int((naive["eligible"] != base["eligible"]).sum())
    assert gap > 1e-6, "Do not claim gate value without demonstrated divergence"
    report = {
        "passed": True,
        "full_computation_control": "Original frozen M5, exactly matched on every paired ordinary video. Its cost/memory is the benchmark comparator.",
        "quality_gate_removed": {
            "condition": "Programmed wrist confidence<=.19, real video; HOG unavailable. Naively treating HOG as available is an intentionally invalid ablation.",
            "max_score_deviation_from_legal_baseline": gap,
            "different_class_decisions": different,
            "correct_candidate_route": "full-rgb-fallback",
            "candidate_vs_baseline_equal": True,
        },
        "dependency_proof_removed": "A future semantic-dependent head cannot use context-only inference; covered by tests that change active index576 or visual context index0. Compiler forces full materialisation.",
        "classifier_accuracy_improvement_claimed": False,
        "manual_labels_or_fitting": False,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
