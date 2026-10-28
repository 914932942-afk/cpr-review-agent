import unittest

import numpy as np

import optimize_robust as e
from robust_head import load, count_fields


class DevelopmentContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = e.o.checked(e.OUTPUT)
        cls.s = e.o.parent.read(e.OUTPUT/'selection.json')
        cls.c = cls.s['selected']
        cls.model, cls.state = load(e.OUTPUT/cls.c['checkpoint'], cls.p['feature_profile_hash'], e.o.parent.digest(e.OUTPUT/'protocol.json'))
        with np.load(e.OUTPUT/'development.npz', allow_pickle=False) as data:
            cls.x, cls.y, cls.split, cls.cohorts, cls.ids = [data[k].copy() for k in ['features', 'targets', 'splits', 'cohorts', 'record_ids']]

    def test_original_data_gates_and_reservations_are_unchanged(self):
        original = e.o.parent.read(e.BASE/'protocol.json')
        self.assertEqual(self.p['gates'], original['gates'])
        for name in ['development.npz', 'locked-test.npz', 'record-splits.json']:
            self.assertEqual(e.o.parent.digest(e.OUTPUT/name), e.o.parent.digest(e.BASE/name))
        reserved = {e.o.parent.group_id(k) for k in self.p['raw_test_reservations']}
        self.assertFalse(reserved & {e.o.parent.group_id(str(k)) for k in self.ids})
        self.assertEqual(int((self.split == 'train').sum()), 738)
        self.assertEqual(int((self.split == 'validation').sum()), 196)

    def test_normalization_is_fitted_only_from_training(self):
        train = self.x[self.split == 'train'].astype(np.float64)
        expected_scale = train.std(0)
        expected_scale[expected_scale < 1e-6] = 1.
        np.testing.assert_array_equal(self.state['components'][1]['mean'].numpy(), train.mean(0))
        np.testing.assert_array_equal(self.state['components'][1]['scale'].numpy(), expected_scale)

    def test_selection_is_replayable_from_declared_validation_rule(self):
        expected = e.choose_challenger(self.s['all_candidates'], self.s['control'])
        self.assertEqual(expected, self.c)
        self.assertEqual(len(self.s['all_candidates']), 19)
        self.assertFalse(self.s['test_read_during_fit_or_selection'])
        self.assertTrue(e.o.parent.read(e.OUTPUT/'fit-access-audit.json')['passed'])
        self.assertEqual(e.o.parent.digest(e.OUTPUT/'protocol.json'), self.s['protocol_sha256'])

    def test_selected_probabilities_thresholds_and_budget_replay(self):
        val = self.split == 'validation'
        probability = self.model.probability(self.x[val])
        op = self.state['operating_point']
        self.assertEqual(e.o.summarize(probability, self.y[val], self.cohorts[val], op['thresholds'], op['normal_gate']), self.c['validation_cohorts'])
        self.assertAlmostEqual(e.d.brier(probability, self.y[val], self.cohorts[val]), self.c['validation_brier'], places=14)
        self.assertEqual(e.o.parent.digest(e.OUTPUT/self.c['checkpoint']), self.c['checkpoint_sha256'])
        self.assertEqual(count_fields(self.state['components']), self.c['scalar_fields'])
        self.assertLessEqual(self.c['scalar_fields'], 70000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
