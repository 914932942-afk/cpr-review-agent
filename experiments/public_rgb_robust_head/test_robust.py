import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import robust_head as r
import optimize_robust as exp


def synthetic(bound=None):
    return {'schema': r.SCHEMA, 'label_ids': list(r.LABEL_IDS), 'source_revision': r.REVISION,
        'feature_profile_hash': 'feature', 'protocol_sha256': 'protocol', 'input_dimension': 783,
        'feature_indices': torch.arange(783), 'components': [
            {'kind': 'tree', 'mixture_weight': .75, 'classifiers': [{'bias': 0., 'trees': []} for _ in range(12)]},
            {'kind': 'linear', 'mixture_weight': .25, 'mean': torch.zeros(783, dtype=torch.float64),
             'scale': torch.ones(783, dtype=torch.float64), 'weight1': torch.ones((12, 783), dtype=torch.float64)*.001,
             'bias1': torch.zeros(12, dtype=torch.float64), 'zscore_clip': bound}]}


class RobustContracts(unittest.TestCase):
    def test_unbounded_control_is_identical_to_previous_implementation(self):
        state = synthetic()
        x = np.random.default_rng(1).normal(size=(20, 783))
        np.testing.assert_array_equal(r.RobustHead(state).probability(x), r.CompactHead(state).probability(x))

    def test_extreme_values_are_clipped_before_linear_scoring(self):
        head = r.RobustHead(synthetic(3.))
        np.testing.assert_array_equal(head.probability(np.full((2, 783), 1e100)), head.probability(np.full((2, 783), 3.)))
        np.testing.assert_allclose(head.probability(np.full((1, 783), -1e100)), 1/(1+np.exp(.25*.783*3)))

    def test_clipping_respects_mean_and_scale_in_raw_feature_units(self):
        state = synthetic(3.)
        state['components'][1]['mean'].fill_(10.)
        state['components'][1]['scale'].fill_(2.)
        result = r.RobustHead(state).probability(np.full((1, 783), 100.))
        np.testing.assert_allclose(result, 1/(1+np.exp(-.25*.783*3)))

    def test_invalid_features_fail_closed_instead_of_silent_imputation(self):
        head = r.RobustHead(synthetic(3.))
        for value in [np.zeros((1, 512)), np.zeros(783), np.full((1, 783), np.nan), np.full((1, 783), np.inf)]:
            with self.assertRaises(ValueError):
                head.probability(value)

    def test_invalid_components_and_bounds_are_rejected(self):
        variants = []
        for bound in [0., -3., float('inf'), float('nan')]:
            variants.append(synthetic(bound))
        bad = synthetic(3.); bad['components'][1]['scale'][0] = 0.; variants.append(bad)
        bad = synthetic(3.); bad['components'][1]['weight1'][0, 0] = float('nan'); variants.append(bad)
        bad = synthetic(3.); bad['components'][0]['mixture_weight'] = .9; variants.append(bad)
        for state in variants:
            with self.assertRaises(ValueError):
                r.RobustHead(state)

    def test_profile_and_protocol_binding_and_tensor_only_round_trip(self):
        exp.REPORT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=exp.REPORT) as folder:
            path = Path(folder)/'head.pt'
            torch.save(synthetic(3.), path)
            model, _ = r.load(path, 'feature', 'protocol')
            np.testing.assert_array_equal(model.probability(np.zeros((1, 783))), np.full((1, 12), .5))
            for profile, protocol in [('wrong', 'protocol'), ('feature', 'wrong')]:
                with self.assertRaises(ValueError):
                    r.load(path, profile, protocol)

    def test_validation_selection_requires_brier_gain_and_comparable_f1(self):
        control = {'name': 'v4_control', 'eligible': True, 'validation_rank': [.87], 'validation_brier': .025, 'scalar_fields': 41018}
        challenger = {**control, 'name': 'trial', 'validation_brier': .024}
        self.assertEqual(exp.choose_challenger([control, challenger], control), challenger)
        for change in [{'validation_brier': .02499}, {'validation_rank': [.86]}, {'eligible': False}]:
            self.assertIsNone(exp.choose_challenger([control, {**challenger, **change}], control))

    def test_promotion_rejects_tradeoff_even_when_overall_f1_improves(self):
        metric = {'macro_f1': .85, 'correct_false_alarm_rate': .05, 'original_correct_records': 20, 'per_label': [{'recall': .7}]}
        baseline = {'evaluations': {k: copy.deepcopy(metric) for k in ['all', 'single', 'composite', 'rgb_reserved_public_keypoints']},
                    'reserved_per_record': [{'exact_match': True}], 'quality_checks': {'gate': True}}
        candidate = copy.deepcopy(baseline)
        candidate['evaluations']['all']['macro_f1'] += .01
        self.assertTrue(all(exp.compare_metrics(baseline, candidate).values()))
        candidate['evaluations']['rgb_reserved_public_keypoints']['macro_f1'] -= .01
        self.assertFalse(all(exp.compare_metrics(baseline, candidate).values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
