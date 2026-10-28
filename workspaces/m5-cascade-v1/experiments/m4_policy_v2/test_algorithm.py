"""Boundary/replanning tests not dependent on accuracy labels or a GPU."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("policy_v2", Path(__file__).with_name("algorithm.py"))
alg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alg)
P = json.loads((ROOT / "data/m4-scheduling-v1/parameters.json").read_text())


def obs(**kw):
    return dict(
        target_label_ids=list(alg.LABELS),
        analysis_mode="adaptive",
        remaining_budget_seconds=900,
        decoded_frames=362,
        executed_actions=[],
        failed_actions=[],
        available_actions=list(alg.ACTIONS.values()) + ["STOP"],
        available_profile_ids=list(alg.PATHS),
        **kw,
    )


class PolicyTests(unittest.TestCase):
    def test_consent_is_not_implied(self):
        o = obs()
        o["remaining_budget_seconds"] = 180
        self.assertEqual(alg.choose(o, P)["stop_code"], "M4_BUDGET_INSUFFICIENT")

    def test_partial_when_explicitly_permitted(self):
        o = obs(allow_partial=True)
        o["remaining_budget_seconds"] = 180
        d = alg.choose(o, P)
        self.assertEqual(d["profile_id"], "body-qualified-v1")
        self.assertEqual(d["audit"]["selected"]["uncovered_requested_labels"], [1, 2, 5, 6, 7, 9, 13])

    def test_complete_first_even_when_partial_has_better_gain_cost_ratio(self):
        d = alg.choose(obs(allow_partial=True), P)
        self.assertEqual(d["profile_id"], "full-v15")

    def test_known_cheap_scope_stays_cheap(self):
        o = obs()
        o["target_label_ids"] = [4]
        self.assertEqual(alg.choose(o, P)["profile_id"], "body-qualified-v1")

    def test_no_partial_route_for_visual_only_goal(self):
        o = obs(allow_partial=True)
        o.update(target_label_ids=[1, 2, 5], remaining_budget_seconds=180)
        self.assertEqual(alg.choose(o, P)["action"], "STOP")

    def test_existing_only_never_executes(self):
        o = obs(allow_partial=True)
        o["analysis_mode"] = "existing_only"
        self.assertEqual(alg.choose(o, P)["stop_code"], "EXISTING_ONLY")

    def test_partial_commitment_does_not_silently_widen(self):
        o = obs(allow_partial=True, committed_partial_route="body-qualified-v1")
        o["executed_actions"] = ["pose.extract"]
        d = alg.choose(o, P)
        self.assertEqual(d["profile_id"], "body-qualified-v1")
        self.assertEqual(d["audit"]["selected"]["missing_tools"], ["skeleton.predict", "fusion.predict"])

    def test_delivered_partial_stops(self):
        self.assertEqual(
            alg.choose(obs(allow_partial=True, partial_result_delivered=True), P)["stop_code"],
            "PARTIAL_SCOPE_DELIVERED",
        )

    def test_bad_commitment_rejected(self):
        for fields in [
            dict(committed_partial_route="body-qualified-v1"),
            dict(allow_partial=True, committed_partial_route="fake"),
        ]:
            with self.assertRaises(ValueError):
                alg.choose(obs(**fields), P)

    def test_rgb_failure_can_salvage_real_skeleton_only_with_consent(self):
        o = obs(allow_partial=True)
        o.update(executed_actions=["pose.extract", "skeleton.predict"], failed_actions=["rgb.predict"])
        self.assertEqual(alg.choose(o, P)["audit"]["selected"]["missing_tools"], ["fusion.predict"])
        o["allow_partial"] = False
        self.assertEqual(alg.choose(o, P)["action"], "STOP")

    def test_pose_failure_has_no_fallback(self):
        o = obs(allow_partial=True)
        o["failed_actions"] = ["pose.extract"]
        self.assertEqual(alg.choose(o, P)["action"], "STOP")

    def test_fusion_failure_not_retried_even_if_old_final_cached(self):
        o = obs(allow_partial=True)
        o.update(
            executed_actions=["pose.extract", "skeleton.predict", "fusion.predict"],
            failed_actions=["fusion.predict"],
            covered_label_ids=list(alg.BODY_LABELS),
        )
        self.assertEqual(alg.choose(o, P)["stop_code"], "TOOL_FAILED")

    def test_slow_prefix_changes_remaining_plan(self):
        o = obs(allow_partial=True)
        o.update(
            executed_actions=["pose.extract"],
            remaining_budget_seconds=90,
            observed_tool_seconds={"pose.extract": 120},
        )
        d = alg.choose(o, P)
        self.assertEqual(d["profile_id"], "body-qualified-v1")

    def test_too_little_budget_stops(self):
        o = obs(allow_partial=True)
        o["remaining_budget_seconds"] = 5
        self.assertEqual(alg.choose(o, P)["action"], "STOP")

    def test_cached_expansion_is_only_two_steps(self):
        o = obs()
        o.update(
            executed_actions=["pose.extract", "skeleton.predict", "fusion.predict"],
            covered_label_ids=list(alg.BODY_LABELS),
        )
        self.assertEqual(
            alg.choose(o, P)["audit"]["selected"]["missing_tools"], ["rgb.predict", "fusion.predict"]
        )

    def test_uncertainty_is_not_erased(self):
        o = obs()
        o.update(covered_label_ids=list(alg.LABELS), assessment={"states": {"4": "uncertain"}})
        self.assertEqual(alg.choose(o, P)["stop_code"], "NO_INDEPENDENT_INFORMATION")

    def test_unapproved_actions_not_run(self):
        o = obs(allow_partial=True)
        o["available_actions"] = ["STOP"]
        self.assertEqual(alg.choose(o, P)["action"], "STOP")

    def test_cache_requires_dependencies(self):
        o = obs()
        o["executed_actions"] = ["skeleton.predict"]
        with self.assertRaises(ValueError):
            alg.choose(o, P)

    def test_inputs_immutable(self):
        o = obs(allow_partial=True)
        before = copy.deepcopy(o)
        p = copy.deepcopy(P)
        self.assertEqual(alg.choose(o, P), alg.choose(o, P))
        self.assertEqual(o, before)
        self.assertEqual(P, p)

    def test_invalid_inputs_rejected(self):
        for fields in [
            dict(target_label_ids=[]),
            dict(remaining_budget_seconds=float("nan")),
            dict(allow_partial="true"),
        ]:
            o = obs()
            o.update(fields)
            with self.assertRaises(ValueError):
                alg.choose(o, P)


if __name__ == "__main__":
    unittest.main(verbosity=2)
