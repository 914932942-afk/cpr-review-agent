"""M3 pure inference and contract invariants. Generated inputs do not measure accuracy."""

import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT / "workspaces/m3-calibrated-v1"
sys.path.insert(0, str(WS / "backend"))
from cpr_agent.modules.contracts import Binding, CalibrationInput, Prediction, QualityIndicators  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


member = load_module("test_m3_member", WS / "experiments/m3_calibration_v1/module.py")
algorithm = member.algorithm
PARAMS = json.loads((ROOT / "data/m3-calibration-v1/selection.json").read_text())["parameters"]


def request(labels=(1, 3), flags=(1,), quality=True):
    binding = Binding(
        clip_id="clip-a", video_sha256="a" * 64, bundle_sha256="b" * 64, pipeline_sha256="c" * 64
    )
    p = Prediction(
        prediction_id="fused-one",
        binding=binding,
        supported_label_ids=list(labels),
        flagged_label_ids=[k for k in flags if k in labels],
        score_type="uncalibrated_score",
        scores={k: 0.7 if k in flags else 0.05 for k in labels},
        thresholds={k: 0.5 for k in labels},
    )
    q = (
        QualityIndicators(
            pose_visible_fraction=0.95,
            rgb_usable_samples=16,
            rgb_requested_samples=16,
            extras={"disagreement_1": 0.0},
        )
        if quality
        else QualityIndicators()
    )
    return CalibrationInput(target_label_ids=list(labels), predictions=[p], quality=q)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = member.SourceQualityCalibration()

    def test_probability_is_monotone_for_fixed_source_and_quality(self):
        values = [algorithm.probability(v, 0.5, 1, 0.9, 0, PARAMS) for v in [0.01, 0.1, 0.4, 0.6, 0.9, 0.99]]
        self.assertEqual(values, sorted(values))

    def test_quality_and_disagreement_shrink_certainty(self):
        p = algorithm.probability(0.95, 0.5, 3, 1.0, 0, PARAMS)
        q = algorithm.probability(0.95, 0.5, 3, 0.7, 0, PARAMS)
        self.assertLess(abs(q - 0.5), abs(p - 0.5))
        p = algorithm.probability(0.95, 0.5, 1, 1.0, 0, PARAMS)
        q = algorithm.probability(0.95, 0.5, 1, 1.0, 1, PARAMS)
        self.assertLess(abs(q - 0.5), abs(p - 0.5))

    def test_finite_probability_at_extreme_scores(self):
        for score in [0.0, 1.0]:
            p = algorithm.probability(score, 0.5, 1, 0.8, 1, PARAMS)
            self.assertTrue(math.isfinite(p) and 0 <= p <= 1)

    def test_invalid_score_quality_or_category_is_rejected(self):
        for score, quality, label, d in [
            (float("nan"), 1, 1, 0),
            (0.9, -1, 1, 0),
            (0.9, 1.1, 1, 0),
            (0.9, 1, 99, 0),
            (0.9, 1, 1, 0.5),
        ]:
            with self.assertRaises(ValueError):
                algorithm.probability(score, 0.5, label, quality, d, PARAMS)

    def test_uncertain_is_not_a_reversed_raw_flag(self):
        self.assertEqual(algorithm.review_state(True, 0.1, 1, PARAMS["rgb"]), "uncertain")
        self.assertEqual(algorithm.review_state(False, 0.9, 1, PARAMS["rgb"]), "uncertain")

    def test_no_source_is_not_evaluated_and_has_no_probability(self):
        result = self.module.assess(
            CalibrationInput(target_label_ids=[1, 3], predictions=[], quality=QualityIndicators())
        )
        self.assertEqual(result.states, {1: "not_evaluated", 3: "not_evaluated"})
        self.assertEqual(result.calibrated_probabilities, {})

    def test_missing_quality_is_uncertain_not_confident_negative(self):
        result = self.module.assess(request(quality=False))
        self.assertEqual(set(result.states.values()), {"uncertain"})
        self.assertEqual(result.calibrated_probabilities, {})

    def test_missing_branch_agreement_prevents_rgb_calibrated_probability(self):
        r = request()
        r.quality.extras = {}
        result = self.module.assess(r)
        self.assertEqual(result.states[1], "uncertain")
        self.assertNotIn(1, result.calibrated_probabilities)

    def test_insufficient_pose_quality_is_deferred_without_extrapolated_probability(self):
        r = request(labels=(3,), flags=())
        r.quality.pose_visible_fraction = 0.1
        result = self.module.assess(r)
        self.assertEqual(result.states, {3: "uncertain"})
        self.assertEqual(result.calibrated_probabilities, {})

    def test_unsupported_category_remains_not_evaluated(self):
        r = request(labels=(1,), flags=(1,))
        r.target_label_ids = [1, 3]
        result = self.module.assess(r)
        self.assertEqual(result.states[3], "not_evaluated")
        self.assertNotIn(3, result.calibrated_probabilities)

    def test_missing_threshold_is_rejected(self):
        r = request()
        r.predictions[0].thresholds = {}
        with self.assertRaises(ValueError):
            self.module.assess(r)

    def test_calibrated_source_cannot_be_calibrated_twice(self):
        r = request()
        r.predictions[0].score_type = "calibrated_probability"
        with self.assertRaises(ValueError):
            self.module.assess(r)

    def test_multiple_sources_require_explicit_fusion(self):
        r = request()
        r.predictions.append(r.predictions[0])
        with self.assertRaises(ValueError):
            self.module.assess(r)

    def test_source_scope_and_original_output_are_immutable(self):
        r = request()
        before = r.model_dump()
        result = self.module.assess(r)
        self.assertEqual(r.model_dump(), before)
        self.assertEqual(result.source_prediction_ids, ["fused-one"])
        self.assertEqual(set(result.states), {1, 3})

    def test_inference_is_repeatable(self):
        r = request()
        self.assertEqual(self.module.assess(r), self.module.assess(r))

    def test_no_probability_for_not_evaluated(self):
        r = request(labels=(), flags=())
        r.target_label_ids = [3]
        result = self.module.assess(r)
        self.assertEqual(result.calibration_status, "uncalibrated")
        self.assertEqual(result.calibrated_probabilities, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
