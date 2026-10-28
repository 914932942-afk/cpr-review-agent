import unittest

import numpy as np

import optimize_robust as e
from robust_head import load


class AcceptanceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = e.o.checked(e.OUTPUT)
        cls.selection = e.o.parent.read(e.OUTPUT/'selection.json')
        cls.old, cls.new = [e.o.parent.read(folder/'evaluation.json') for folder in [e.BASE, e.OUTPUT]]

    def test_every_record_target_and_prediction_is_retained_and_replayable(self):
        selected = self.selection['selected']
        model, state = load(e.OUTPUT/selected['checkpoint'], self.p['feature_profile_hash'], e.o.parent.digest(e.OUTPUT/'protocol.json'))
        with np.load(e.OUTPUT/'locked-test.npz', allow_pickle=False) as original, np.load(e.OUTPUT/'test-predictions.npz', allow_pickle=False) as saved:
            self.assertEqual(len(saved['record_ids']), 386)
            np.testing.assert_array_equal(original['record_ids'], saved['record_ids'])
            np.testing.assert_array_equal(original['targets'], saved['targets'])
            np.testing.assert_array_equal(model.probability(original['features']), saved['scores'])
            op = state['operating_point']
            np.testing.assert_array_equal(e.o.parent.predict(saved['scores'], op['thresholds'], op['normal_gate']), saved['predictions'])
            self.assertTrue(set(self.p['raw_test_reservations']).issubset(set(map(str, original['record_ids']))))

    def test_predeclared_promotion_veto_is_applied_despite_all_gates_passing(self):
        result = e.o.parent.read(e.OUTPUT/'promotion-comparison.json')
        expected = e.compare_metrics(self.old, self.new)
        self.assertEqual(result['metric_checks'], expected)
        self.assertTrue(all(self.new['quality_checks'].values()))
        self.assertFalse(expected['reserved_macro_f1_no_regression'])
        self.assertFalse(expected['minimum_label_recall_no_regression'])
        self.assertFalse(result['metric_promotion_eligible'])

    def test_current_pointer_and_all_old_artifacts_remain_intact(self):
        pointer = e.o.DATA/'current-offline-candidate.json'
        self.assertEqual(e.o.parent.digest(pointer), self.p['candidate_pointer_before_sha256'])
        self.assertEqual(e.o.parent.read(pointer), self.p['candidate_pointer_before'])
        e.o.parent.check_hashes(self.p['protected_sources'])

    def test_original_application_sources_are_unchanged(self):
        hashes = e.o.parent.read(e.CODE/'test-results/public-head-validation/application-source-before.json')
        self.assertEqual(len(hashes), 19)
        for path, expected in hashes.items():
            self.assertEqual(e.o.parent.digest(e.CODE/path), expected, path)

    def test_failed_challenger_cannot_be_exported_for_adoption(self):
        self.assertFalse((e.OUTPUT/'inference-manifest.json').exists())
        with self.assertRaisesRegex(ValueError, 'not eligible'):
            e.export()
        self.assertFalse((e.OUTPUT/'inference-manifest.json').exists())
        self.assertFalse(self.new['raw_video_acceptance_performed'])
        self.assertFalse(self.new['runtime_release_approved'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
