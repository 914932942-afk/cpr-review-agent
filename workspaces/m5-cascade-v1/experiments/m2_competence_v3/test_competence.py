"""Contract and algorithm invariants; generated inputs are not accuracy measurements."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments/m2_competence_v3"), str(ROOT / "experiments/branch_pipeline_v1")]
from fusion_algorithm import Config, fuse  # noqa: E402
from pipeline import LABELS, merge_readouts  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "old_m2", ROOT / "experiments/m2_quality_v1/fusion_algorithm.py"
)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
CFG = Config(full_rgb_ratio=0.75, minimum_rgb_weight=0.5, boundary_band=0.35, skeleton_advantage=2)


def inputs():
    return (
        {
            "scores": np.full((1, 12), 0.1),
            "thresholds": np.full((1, 11), 0.5),
            "eligible": np.zeros((1, 11), bool),
            "predicted_count": np.array([2]),
        },
        {
            "label_ids": np.array([1, 2, 5, 7, 9]),
            "scores": np.full((1, 5), 0.1),
            "thresholds": np.full((1, 5), 0.5),
            "eligible": np.zeros((1, 5), bool),
            "hand_override": np.zeros((1, 5), bool),
            "normal_veto_threshold": np.array([0.9]),
        },
    )


class CompetenceTests(unittest.TestCase):
    def test_weak_expert_boundary_is_resolved_by_compatible_strong_evidence(self):
        a, b = inputs()
        a["scores"][0, LABELS.index(5)] = 0.05
        b["scores"][0, 2], b["eligible"][0, 2] = 0.55, True
        result, audit = fuse(a, b, 16, 16, False, CFG)
        self.assertEqual(audit["boundary_adapted_label_ids"], [5])
        self.assertFalse(result["eligible"][0, LABELS.index(5)])

    def test_qualified_rgb_specialist_is_not_overturned_at_high_quality(self):
        a, b = inputs()
        b["scores"][0, 4], b["eligible"][0, 4] = 0.52, True
        result, audit = fuse(a, b, 16, 16, False, CFG)
        self.assertTrue(result["predictions"][0, LABELS.index(9)])
        self.assertNotIn(9, audit["boundary_adapted_label_ids"])
        self.assertIn(9, audit["protected_specialist_label_ids"])

    def test_shared_fusion_cannot_change_any_out_of_scope_decision(self):
        rng = np.random.default_rng(20261027)
        for _ in range(100):
            a, b = inputs()
            a["scores"] = rng.uniform(0.001, 0.999, (1, 12))
            b["scores"] = rng.uniform(0.001, 0.999, (1, 5))
            a["eligible"] = a["scores"][:, :11] >= 0.5
            b["eligible"] = b["scores"] >= 0.5
            a["predicted_count"][:] = rng.integers(0, 5)
            baseline, _ = merge_readouts(a, b)
            result, _ = fuse(a, b, 8, 16, False, CFG)
            cols = [i for i, k in enumerate(LABELS) if k not in b["label_ids"]]
            for key in ["scores", "thresholds", "eligible", "predictions"]:
                np.testing.assert_array_equal(result[key][:, cols], baseline[key][:, cols])
            self.assertFalse((result["predictions"] & ~result["eligible"]).any())

    def test_disabling_additions_reproduces_accepted_v1(self):
        a, b = inputs()
        a["scores"][0, 6], a["eligible"][0, 6] = 0.95, True
        cfg = Config(0.75, 0.5, boundary_band=0, lock_unshared=False)
        result, _ = fuse(a, b, 8, 16, False, cfg)
        before, _ = old.fuse(a, b, 8, 16, False, old.Config(0.75, 0.5))
        for key in before:
            np.testing.assert_array_equal(result[key], before[key])

    def test_no_agreement_conflict_preserves_baseline_exactly(self):
        a, b = inputs()
        result, _ = fuse(a, b, 16, 16, False, CFG)
        baseline, _ = merge_readouts(a, b)
        for key in baseline:
            np.testing.assert_array_equal(result[key], baseline[key])

    def test_missing_quality_keeps_baseline(self):
        a, b = inputs()
        result, audit = fuse(a, b, None, 16, None, CFG)
        baseline, _ = merge_readouts(a, b)
        np.testing.assert_array_equal(result["predictions"], baseline["predictions"])
        self.assertFalse(audit["quality_available"])

    def test_input_immutable(self):
        a, b = inputs()
        originals = [{k: v.copy() for k, v in x.items()} for x in [a, b]]
        fuse(a, b, 8, 16, False, CFG)
        for x, before in zip([a, b], originals):
            for k in x:
                np.testing.assert_array_equal(x[k], before[k])

    def test_normal_veto_is_preserved(self):
        a, b = inputs()
        a["scores"][0, 11] = 0.99
        b["scores"][0, 2], b["eligible"][0, 2] = 0.55, True
        result, _ = fuse(a, b, 16, 16, False, CFG)
        self.assertFalse(result["predictions"].any())

    def test_invalid_and_insufficient_quality_rejected(self):
        for used, total in [(7, 16), (17, 16), (True, 16), (-1, 16), (0, 0)]:
            with self.assertRaises(ValueError):
                fuse(*inputs(), used, total, False, CFG)

    def test_nan_rejected(self):
        a, b = inputs()
        b["scores"][0, 2] = np.nan
        with self.assertRaises(ValueError):
            fuse(a, b, 16, 16, False, CFG)


if __name__ == "__main__":
    unittest.main(verbosity=2)
