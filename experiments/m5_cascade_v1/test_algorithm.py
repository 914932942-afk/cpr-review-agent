"""Real frozen-head dependency invariants, plus explicit malformed-plan fixtures."""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/branch_pipeline_v1"))
from pipeline import frozen, rgb_readout  # noqa: E402

spec = importlib.util.spec_from_file_location("m5_alg_test", Path(__file__).with_name("algorithm.py"))
alg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alg)
RGB = torch.load(
    ROOT / "data/public_cpr/rgb-evaluation-v12/experts.pt", weights_only=True, map_location="cpu"
)
HAND = torch.load(ROOT / "data/public_cpr/rgb-evaluation-v14/hands.pt", weights_only=True, map_location="cpu")


class DependencyTests(unittest.TestCase):
    def test_quality_available_prunes_visual_only_after_dependency_proof(self):
        p = alg.compile_plan(RGB, HAND, True)
        self.assertFalse(p["dense_visual_required"])
        self.assertTrue(all(i >= 1536 for i in p["required_rgb_indices"]))
        self.assertEqual(
            p["active_experts"],
            {
                "1": "hand-shape",
                "2": "hand-shape",
                "5": "rgb-context",
                "7": "rgb-context",
                "9": "rgb-context",
            },
        )

    def test_quality_unavailable_preserves_visual_fallback(self):
        p = alg.compile_plan(RGB, HAND, False)
        self.assertTrue(p["dense_visual_required"])
        self.assertEqual(p["route"], "full-rgb-fallback")

    def test_future_semantic_hand_head_cannot_be_pruned(self):
        h = copy.deepcopy(HAND)
        h["experts"][0]["indices"][0] = 576
        self.assertTrue(alg.compile_plan(RGB, h, True)["dense_visual_required"])

    def test_future_visual_context_head_cannot_be_pruned(self):
        r = copy.deepcopy(RGB)
        r["experts"][2]["indices"][0] = 0
        self.assertTrue(alg.compile_plan(r, HAND, True)["dense_visual_required"])

    def test_duplicate_indices_rejected(self):
        r = copy.deepcopy(RGB)
        r["experts"][0]["indices"][0] = r["experts"][0]["indices"][1]
        with self.assertRaises(ValueError):
            alg.compile_plan(r, HAND, True)

    def test_out_of_range_indices_rejected(self):
        for v in [-1, 1893]:
            r = copy.deepcopy(RGB)
            r["experts"][0]["indices"][0] = v
            with self.assertRaises(ValueError):
                alg.compile_plan(r, HAND, True)

    def test_non_boolean_quality_rejected(self):
        for q in ["true", 1, None]:
            with self.assertRaises(ValueError):
                alg.compile_plan(RGB, HAND, q)

    def test_wrong_scope_rejected(self):
        r = copy.deepcopy(RGB)
        r["experts"][0]["label_id"] = 3
        with self.assertRaises(ValueError):
            alg.compile_plan(r, HAND, True)

    def test_wrong_layout_rejected(self):
        r = copy.deepcopy(RGB)
        r["feature_dimension"] = 100
        with self.assertRaises(ValueError):
            alg.compile_plan(r, HAND, True)

    def test_input_plan_is_not_mutated(self):
        before = copy.deepcopy(RGB)
        alg.compile_plan(RGB, HAND, True)
        self.assertTrue(
            all(torch.equal(a["indices"], b["indices"]) for a, b in zip(before["experts"], RGB["experts"]))
        )

    def test_context_readout_equals_original_and_uses_only_active_heads(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(1, 1893)).astype(np.float32)
        s = rng.normal(size=(1, 576)).astype(np.float32)
        base = rgb_readout(x, s, np.array([True]))
        seen = []

        def rgb_fn(values, state):
            seen.extend(e["label_id"] for e in state["experts"])
            return frozen.predict_experts(values, state)

        fast = alg.context_readout(
            x[:, 1536:], s, RGB, HAND, alg.compile_plan(RGB, HAND, True), rgb_fn, frozen.predict_hands
        )
        self.assertEqual(seen, [5, 7, 9])
        self.assertTrue(all(np.array_equal(base[k], fast[k]) for k in base))

    def test_tampered_plan_is_rejected(self):
        p = alg.compile_plan(RGB, HAND, True)
        p["active_experts"]["1"] = "rgb-context"
        with self.assertRaises(ValueError):
            alg.context_readout(
                np.zeros((1, 357)),
                np.zeros((1, 576)),
                RGB,
                HAND,
                p,
                frozen.predict_experts,
                frozen.predict_hands,
            )

    def test_fallback_cannot_be_forced_through_context_only(self):
        with self.assertRaises(ValueError):
            alg.context_readout(
                np.zeros((1, 357)),
                np.zeros((1, 576)),
                RGB,
                HAND,
                alg.compile_plan(RGB, HAND, False),
                frozen.predict_experts,
                frozen.predict_hands,
            )

    def test_nonfinite_context_rejected(self):
        x = np.zeros((1, 357))
        x[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            alg.context_readout(
                x,
                np.zeros((1, 576)),
                RGB,
                HAND,
                alg.compile_plan(RGB, HAND, True),
                frozen.predict_experts,
                frozen.predict_hands,
            )

    def test_context_layout_rejected(self):
        with self.assertRaises(ValueError):
            alg.context_readout(
                np.zeros((1, 356)),
                np.zeros((1, 576)),
                RGB,
                HAND,
                alg.compile_plan(RGB, HAND, True),
                frozen.predict_experts,
                frozen.predict_hands,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
