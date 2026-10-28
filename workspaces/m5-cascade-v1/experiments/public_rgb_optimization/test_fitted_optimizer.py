import os
from pathlib import Path
import unittest

import numpy as np

import optimize_head as o


class FittedOptimizationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = Path(os.environ.get('CPR_OPTIMIZATION_RUN', str(o.OUTPUT)))
        cls.protocol = o.checked(cls.output)
        cls.selection = o.parent.read(cls.output / 'selection.json')
        cls.chosen = cls.selection['selected']
        cls.head, cls.state = o.load(cls.output / cls.chosen['checkpoint'],
            cls.protocol['feature_profile_hash'], o.parent.digest(cls.output / 'protocol.json'))

    def test_same_split_and_feature_files_are_used_byte_for_byte(self):
        for name in ['development.npz', 'locked-test.npz', 'record-splits.json']:
            self.assertEqual(o.parent.digest(self.output / name), o.parent.digest(o.BASE / name))
        self.assertEqual(self.protocol['gates'], o.parent.read(o.BASE / 'protocol.json')['gates'])

    def test_winner_is_exactly_the_declared_validation_choice(self):
        eligible = [row for row in self.selection['all_candidates'] if row['eligible']]
        if 'one_standard_error' in self.selection:
            uncertainty = self.selection['one_standard_error']
            eligible = [row for row in eligible if row['validation_rank'][0] >= uncertainty['one_standard_error_floor']]
            expected = min(eligible, key=lambda row: (row['scalar_fields'], -row['validation_rank'][0]))
            self.assertGreater(uncertainty['bootstrap_standard_error'], 0)
            self.assertEqual(expected['checkpoint_sha256'], self.chosen['source_checkpoint_sha256'])
        else:
            expected = max(eligible, key=lambda row: (*row['validation_rank'], -row['scalar_fields']))
            self.assertEqual(expected['checkpoint_sha256'], self.chosen['checkpoint_sha256'])
        self.assertEqual(len(self.selection['all_candidates']), 8)
        self.assertFalse(self.selection['test_read_during_fit_or_selection'])
        self.assertEqual(self.selection['fitted_records'], 738)
        self.assertEqual(self.selection['tuning_records'], 196)
        self.assertLessEqual(self.chosen['scalar_fields'], 70000)

    def test_validation_metrics_replay_with_exported_head(self):
        with np.load(self.output / 'development.npz', allow_pickle=False) as bundle:
            mask = bundle['splits'] == 'validation'
            x, y, cohorts = [bundle[key][mask] for key in ['features', 'targets', 'cohorts']]
        op = self.state['operating_point']
        measured = o.summarize(self.head.probability(x), y, cohorts, op['thresholds'], op['normal_gate'])
        self.assertEqual(measured, self.chosen['validation_cohorts'])
        self.assertLessEqual(self.chosen['portable_vs_training_max_difference'], 1e-10)

    def test_all_fixed_test_predictions_replay_without_any_rejection(self):
        with np.load(self.output / 'locked-test.npz', allow_pickle=False) as bundle:
            x, y, ids = [bundle[key].copy() for key in ['features', 'targets', 'record_ids']]
        with np.load(self.output / 'test-predictions.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['record_ids'], ids)
            np.testing.assert_array_equal(saved['targets'], y)
            np.testing.assert_array_equal(saved['scores'], self.head.probability(x))
            self.assertEqual(len(ids), 386)
            self.assertEqual(saved['predictions'].shape, (386, 11))

    def test_manifest_is_new_and_release_still_requires_video_acceptance(self):
        manifest = o.parent.load_manifest(self.output / 'inference-manifest.json')
        self.assertEqual(manifest['head'], (self.output / self.chosen['checkpoint']).relative_to(o.CODE).as_posix())
        self.assertEqual(len(manifest['public_samples']), 26)
        self.assertFalse(manifest['runtime_release_approved'])
        self.assertTrue(manifest['raw_video_acceptance_pending'])
        self.assertEqual(self.state['protocol_sha256'], o.parent.digest(self.output / 'protocol.json'))

    def test_previous_exposed_test_and_failed_attempt_are_preserved(self):
        o.parent.check_hashes(self.protocol['protected_sources'])
        self.assertTrue(self.protocol['test_observed_before_optimization'])
        self.assertFalse(self.protocol['independent_unseen_test_claim'])
        self.assertFalse(o.parent.read(o.BASE / 'evaluation.json')['public_keypoint_quality_passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
