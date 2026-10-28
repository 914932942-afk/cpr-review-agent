"""Pure policy properties. These generated requests are not additional video accuracy evidence."""

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tested_m4", Path(__file__).with_name("algorithm.py"))
algo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = algo
spec.loader.exec_module(algo)
PARAMS = json.loads((ROOT / "data/m4-scheduling-v1/parameters.json").read_text())


def observation(targets=(4,), **extra):
    return {
        "target_label_ids": list(targets),
        "analysis_mode": "adaptive",
        "available_actions": ["run_full_pipeline", "run_skeleton", "STOP"],
        "available_profile_ids": ["full-v15", "body-qualified-v1"],
        "executed_actions": [],
        "failed_actions": [],
        "remaining_budget_seconds": 900.0,
        "decoded_frames": 375,
        "covered_label_ids": [],
        **extra,
    }


class PolicyTests(unittest.TestCase):
    def test_body_goal_chooses_certified_cheaper_closure(self):
        decision = algo.choose(observation(), PARAMS)
        self.assertEqual(decision["profile_id"], "body-qualified-v1")
        self.assertEqual(
            decision["audit"]["selected"]["missing_tools"],
            ["pose.extract", "skeleton.predict", "fusion.predict"],
        )

    def test_rgb_and_weak_count_coupled_classes_require_full(self):
        for k in [1, 2, 5, 6, 7, 9, 13]:
            self.assertEqual(algo.choose(observation((k,)), PARAMS)["profile_id"], "full-v15")

    def test_all_scope_requires_full(self):
        self.assertEqual(algo.choose(observation(algo.LABELS), PARAMS)["profile_id"], "full-v15")

    def test_existing_only_never_starts_vision(self):
        self.assertEqual(
            algo.choose(observation(analysis_mode="existing_only"), PARAMS)["stop_code"], "EXISTING_ONLY"
        )

    def test_covered_uncertain_stops_without_claiming_resolution(self):
        o = observation(covered_label_ids=[4], assessment={"states": {"4": "uncertain"}})
        d = algo.choose(o, PARAMS)
        self.assertEqual(d["stop_code"], "NO_INDEPENDENT_INFORMATION")
        self.assertEqual(o["assessment"]["states"]["4"], "uncertain")

    def test_covered_confident_result_is_reused(self):
        self.assertEqual(algo.choose(observation(covered_label_ids=[4]), PARAMS)["stop_code"], "GOAL_COVERED")

    def test_failure_does_not_trigger_unbounded_retry(self):
        self.assertEqual(
            algo.choose(observation(failed_actions=["rgb.predict"]), PARAMS)["stop_code"], "TOOL_FAILED"
        )

    def test_insufficient_budget_stops_before_expensive_work(self):
        self.assertEqual(
            algo.choose(observation(remaining_budget_seconds=1), PARAMS)["stop_code"],
            "M4_BUDGET_INSUFFICIENT",
        )

    def test_upper_cost_bound_controls_budget_not_only_median(self):
        d = algo.choose(observation(), PARAMS)
        p = d["audit"]["selected"]
        budget = (p["expected_seconds"] + p["upper_seconds"]) / 2 + 8
        self.assertEqual(algo.choose(observation(remaining_budget_seconds=budget), PARAMS)["action"], "STOP")

    def test_cached_expansion_runs_only_new_rgb_and_finalisation(self):
        d = algo.choose(
            observation(
                algo.LABELS,
                covered_label_ids=list(algo.BODY_LABELS),
                executed_actions=["pose.extract", "skeleton.predict", "fusion.predict"],
            ),
            PARAMS,
        )
        self.assertEqual(d["audit"]["selected"]["missing_tools"], ["rgb.predict", "fusion.predict"])

    def test_cheap_route_cannot_be_faked_for_mixed_scope(self):
        self.assertEqual(algo.choose(observation((4, 5)), PARAMS)["profile_id"], "full-v15")

    def test_unregistered_profile_and_action_cannot_be_selected(self):
        o = observation(available_actions=["STOP"], available_profile_ids=["invented"])
        self.assertEqual(algo.choose(o, PARAMS)["stop_code"], "NO_SUPPORTED_ROUTE")

    def test_no_evidence_bootstrap_is_not_suppressed_by_uncertainty(self):
        self.assertNotEqual(
            algo.choose(observation(assessment={"states": {"4": "not_evaluated"}}), PARAMS)["action"], "STOP"
        )

    def test_longer_pose_workload_never_gets_a_cheaper_estimate(self):
        costs = PARAMS["cost_model"]
        a = algo.estimate("pose.extract", 100, costs)
        b = algo.estimate("pose.extract", 900, costs)
        self.assertGreater(b[0], a[0])
        self.assertGreater(b[1], a[1])

    def test_all_2047_nonempty_target_scopes_are_covered_by_selected_route(self):
        for bits in range(1, 1 << 11):
            targets = [k for i, k in enumerate(algo.LABELS) if bits >> i & 1]
            d = algo.choose(observation(targets), PARAMS)
            self.assertNotEqual(d["action"], "STOP")
            self.assertTrue(set(targets) <= set(algo.COVERAGE[d["profile_id"]]))

    def test_repeated_calls_are_deterministic_and_immutable(self):
        o = observation()
        p = copy.deepcopy(o)
        cfg = copy.deepcopy(PARAMS)
        self.assertEqual(algo.choose(o, PARAMS), algo.choose(o, PARAMS))
        self.assertEqual(o, p)
        self.assertEqual(PARAMS, cfg)

    def test_invalid_labels_and_nonfinite_budget_are_rejected(self):
        for o in [observation(()), observation((99,)), observation(remaining_budget_seconds=float("nan"))]:
            with self.assertRaises(ValueError):
                algo.choose(o, PARAMS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
