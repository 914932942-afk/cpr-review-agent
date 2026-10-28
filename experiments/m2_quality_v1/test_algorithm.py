"""ML-environment unit checks. Synthetic contract inputs are not accuracy evidence."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments/m2_quality_v1"), str(ROOT / "experiments/branch_pipeline_v1")]
from fusion_algorithm import Config, fuse  # noqa: E402
from pipeline import LABELS, merge_readouts  # noqa: E402


def inputs():
    sk = {
        "scores": np.full((1, 12), 0.1),
        "thresholds": np.full((1, 11), 0.5),
        "eligible": np.zeros((1, 11), bool),
        "predicted_count": np.array([2]),
    }
    rg = {
        "label_ids": np.array([1, 2, 5, 7, 9]),
        "scores": np.full((1, 5), 0.1),
        "thresholds": np.full((1, 5), 0.5),
        "eligible": np.zeros((1, 5), bool),
        "hand_override": np.zeros((1, 5), bool),
        "normal_veto_threshold": np.array([0.9]),
    }
    return sk, rg


class FusionTests(unittest.TestCase):
    def test_full_quality_is_exact_baseline(self):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            a, b = inputs()
            a["scores"] = rng.uniform(0.01, 0.99, (1, 12))
            a["eligible"] = a["scores"][:, :11] >= 0.5
            b["scores"] = rng.uniform(0.01, 0.99, (1, 5))
            b["eligible"] = b["scores"] >= 0.5
            original, _ = merge_readouts(a, b)
            result, audit = fuse(a, b, 16, 16, False)
            for key in original:
                np.testing.assert_array_equal(result[key], original[key])
            self.assertEqual(audit["adapted_label_ids"], [])

    def test_missing_quality_does_not_mean_zero_quality(self):
        a, b = inputs()
        original, _ = merge_readouts(a, b)
        result, audit = fuse(a, b, None, 16, None)
        np.testing.assert_array_equal(result["scores"], original["scores"])
        self.assertFalse(audit["quality_available"])

    def test_low_quality_conflict_combines_threshold_relative_evidence(self):
        a, b = inputs()
        col = LABELS.index(7)
        a["scores"][0, col] = 0.95
        a["eligible"][0, col] = True
        result, audit = fuse(a, b, 8, 16, False, Config(0.75, 0.5))
        self.assertIn(7, audit["adapted_label_ids"])
        self.assertTrue(result["eligible"][0, col])
        self.assertEqual(result["thresholds"][0, col], 0.5)

    def test_agreement_keeps_raw_scores(self):
        a, b = inputs()
        original, _ = merge_readouts(a, b)
        result, audit = fuse(a, b, 8, 16, False)
        np.testing.assert_array_equal(result["scores"], original["scores"])
        self.assertEqual(audit["adapted_label_ids"], [])

    def test_non_rgb_classes_are_never_modified(self):
        a, b = inputs()
        a["scores"][0, 3] = 0.9
        a["eligible"][0, 3] = True
        original, _ = merge_readouts(a, b)
        result, _ = fuse(a, b, 8, 16, False)
        for label in set(LABELS) - set(b["label_ids"]):
            i = LABELS.index(label)
            self.assertEqual(result["scores"][0, i], original["scores"][0, i])
            self.assertEqual(result["eligible"][0, i], original["eligible"][0, i])

    def test_qualified_hand_specialist_is_preserved(self):
        a, b = inputs()
        b["scores"][0, 0] = 0.9
        b["eligible"][0, 0] = True
        b["hand_override"][0, 0] = True
        result, audit = fuse(a, b, 8, 16, True)
        self.assertEqual(audit["rgb_weights"]["1"], 1)
        self.assertEqual(result["scores"][0, 0], 0.9)

    def test_true_normal_veto_cannot_become_positive_rgb_evidence(self):
        a, b = inputs()
        a["scores"][0, 11] = 0.95
        b["scores"][0, 3] = 0.9
        b["eligible"][0, 3] = True
        result, _ = fuse(a, b, 8, 16, False, Config(0.75, 0.5))
        self.assertFalse(result["predictions"].any())

    def test_input_arrays_are_immutable(self):
        a, b = inputs()
        before_a = {k: v.copy() for k, v in a.items()}
        before_b = {k: v.copy() for k, v in b.items()}
        fuse(a, b, 8, 16, False)
        for source, before in [(a, before_a), (b, before_b)]:
            for k in source:
                np.testing.assert_array_equal(source[k], before[k])

    def test_invalid_quality_is_rejected(self):
        for used, total in [(7, 16), (17, 16), (-1, 16), (1, 0), (True, 16)]:
            a, b = inputs()
            with self.assertRaises(ValueError):
                fuse(a, b, used, total, False)

    def test_nan_is_rejected(self):
        a, b = inputs()
        b["scores"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            fuse(a, b, 8, 16, False)

    def test_all_negative_cannot_be_forced_positive_by_cardinality(self):
        a, b = inputs()
        a["predicted_count"][:] = 4
        result, _ = fuse(a, b, 8, 16, False)
        self.assertFalse(result["predictions"].any())

    def test_extreme_scores_are_finite_and_repeatable(self):
        a, b = inputs()
        a["scores"][0, 6] = 1.0
        a["eligible"][0, 6] = True
        b["scores"][0, 3] = 0.0
        first, _ = fuse(a, b, 10, 16, False)
        second, _ = fuse(a, b, 10, 16, False)
        self.assertTrue(np.isfinite(first["scores"]).all())
        np.testing.assert_array_equal(first["scores"], second["scores"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
