from pathlib import Path
import tempfile
import unittest

import numpy as np

import optimize_head as o


class OptimizerContracts(unittest.TestCase):
    def test_every_declared_candidate_fits_worst_case_scalar_budget(self):
        self.assertEqual(len(o.CONFIGS), 8)
        for config in o.CONFIGS:
            worst = 12 + 12 * len(config['branches']) * config['trees'] * (2 * config['leaves'] - 1) * 7
            self.assertLessEqual(worst, 70000, config['name'])

    def test_feature_branch_mapping_is_unambiguous(self):
        np.testing.assert_array_equal(o.feature_mask('pose'), np.arange(512))
        np.testing.assert_array_equal(o.feature_mask('geometry'), np.arange(512, 783))
        np.testing.assert_array_equal(o.feature_mask('combined'), np.arange(783))
        with self.assertRaises(ValueError):
            o.feature_mask('unknown')

    def test_two_branch_export_matches_the_training_predictor(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(48, 783)).astype(np.float32)
        signal = (np.arange(48) % 2).astype(float)
        x[:, 0] = signal * 10
        x[:, 512] = signal * 10
        target = np.repeat(signal[:, None], 12, axis=1)
        config = {**o.CONFIGS[-2], 'trees': 3, 'leaves': 3, 'minimum_leaf': 2}
        outputs, reference = o.fit_candidate(config, x[:40], target[:40], np.array(['single'] * 40), x[40:])
        portable = o.PortableHead({'input_dimension': 783, 'classifiers': outputs})
        np.testing.assert_allclose(portable.probability(x[40:]), reference, atol=1e-12, rtol=1e-12)
        active_indices = [tree['feature'][~tree['leaf']].numpy() for tree in outputs[0]['trees']]
        self.assertTrue(any((indices >= 512).any() for indices in active_indices))

    def test_bootstrap_is_deterministic_and_thresholds_remain_finite(self):
        rng = np.random.default_rng(4)
        y = np.zeros((40, 11), dtype=np.float32)
        y[4:20, 0] = 1
        y[20:, 1:3] = 1
        p = rng.uniform(.01, .99, (40, 12))
        cohorts = np.array(['single'] * 20 + ['composite'] * 20)
        a, std = o.stable_thresholds(p, y, cohorts)
        b, _ = o.stable_thresholds(p, y, cohorts)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (11,))
        self.assertTrue(np.isfinite(a).all() and np.isfinite(std).all())

    def test_stable_thresholds_do_not_create_validation_correct_false_alarms(self):
        y = np.zeros((40, 11), dtype=np.float32)
        y[4:20, :] = 1
        y[20:, :] = 1
        p = np.full((40, 12), .8)
        p[:4, :11] = .6
        cohorts = np.array(['single'] * 20 + ['composite'] * 20)
        for option in o.options(p, y, cohorts):
            pred = o.parent.predict(p, option['thresholds'], option['normal_gate'])
            self.assertFalse(pred[:4].any())
            self.assertEqual(pred.shape, y.shape)

    def test_parent_split_and_all_reserved_recordings_stay_unchanged(self):
        protocol = o.parent.checked_protocol(o.BASE)
        self.assertEqual(protocol['split_counts'], {'train': 738, 'validation': 196, 'test': 386})
        self.assertEqual(len(protocol['raw_test_reservations']), 26)
        self.assertEqual(protocol['gates']['composite_macro_f1'], .8)
        self.assertEqual(protocol['gates']['maximum_head_parameters'], 70000)

    def test_new_attempt_cannot_overwrite_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                o.prepare(Path(directory))


if __name__ == '__main__':
    unittest.main(verbosity=2)
