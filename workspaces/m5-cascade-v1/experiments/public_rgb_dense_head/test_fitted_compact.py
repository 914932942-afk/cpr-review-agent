import unittest
import numpy as np
import optimize_dense as d
from compact_head import load,count_fields


class FittedCompactContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=d.o.checked(d.OUTPUT);cls.s=d.o.parent.read(d.OUTPUT/'selection.json');cls.chosen=cls.s['selected']
        cls.head,cls.state=load(d.OUTPUT/cls.chosen['checkpoint'],cls.p['feature_profile_hash'],d.o.parent.digest(d.OUTPUT/'protocol.json'))

    def test_fixed_split_and_gates_are_unchanged(self):
        first=d.o.DATA/'rgb-evaluation-v1'
        for name in ['development.npz','locked-test.npz','record-splits.json']:
            self.assertEqual(d.o.parent.digest(d.OUTPUT/name),d.o.parent.digest(first/name))
        self.assertEqual(self.p['gates'],d.o.parent.read(first/'protocol.json')['gates'])

    def test_all_16_choices_were_ranked_only_by_declared_validation_rule(self):
        eligible=[r for r in self.s['all_candidates'] if r['eligible'] and r['validation_rank'][0]>=self.s['comparable_floor']]
        expected=min(eligible,key=lambda r:(r['validation_brier'],r['scalar_fields']))
        self.assertEqual(expected['name'],self.chosen['name'])
        self.assertEqual(len(self.s['all_candidates']),16)
        self.assertFalse(self.s['test_read_during_fit_or_selection'])
        self.assertTrue(self.s['normalization_fitted_on_train_only'])
        self.assertEqual(self.s['fitted_records'],738)
        self.assertEqual(self.s['tuning_records'],196)

    def test_validation_results_replay_exactly_after_reload(self):
        with np.load(d.OUTPUT/'development.npz',allow_pickle=False) as f:
            m=f['splits']=='validation';x,y,cohorts=[f[k][m].copy() for k in ['features','targets','cohorts']]
        op=self.state['operating_point'];p=self.head.probability(x)
        self.assertEqual(d.o.summarize(p,y,cohorts,op['thresholds'],op['normal_gate']),self.chosen['validation_cohorts'])
        self.assertAlmostEqual(d.brier(p,y,cohorts),self.chosen['validation_brier'],places=14)

    def test_all_test_predictions_and_source_targets_are_retained(self):
        with np.load(d.OUTPUT/'locked-test.npz',allow_pickle=False) as f,np.load(d.OUTPUT/'test-predictions.npz',allow_pickle=False) as saved:
            self.assertEqual(len(f['record_ids']),386)
            np.testing.assert_array_equal(f['record_ids'],saved['record_ids']);np.testing.assert_array_equal(f['targets'],saved['targets'])
            np.testing.assert_array_equal(self.head.probability(f['features']),saved['scores'])
            self.assertTrue(set(self.p['raw_test_reservations']).issubset(set(map(str,f['record_ids']))))

    def test_budget_checkpoint_and_full_quality_gates(self):
        self.assertEqual(count_fields(self.state['components']),self.chosen['scalar_fields'])
        self.assertLessEqual(self.chosen['scalar_fields'],70000)
        self.assertEqual(d.o.parent.digest(d.OUTPUT/self.chosen['checkpoint']),self.chosen['checkpoint_sha256'])
        evaluation=d.o.parent.read(d.OUTPUT/'evaluation.json')
        self.assertTrue(all(evaluation['quality_checks'].values()))
        self.assertTrue(evaluation['public_keypoint_quality_passed'])
        self.assertFalse(evaluation['raw_video_acceptance_performed'])

    def test_compact_entrypoint_is_pinned_without_enabling_web_runtime(self):
        manifest=d.o.parent.load_manifest(d.OUTPUT/'inference-manifest.json')
        self.assertEqual(manifest['head_kind'],'compact');self.assertEqual(len(manifest['public_samples']),26)
        self.assertEqual(manifest['entrypoint'],'experiments/public_rgb_dense_head/infer_compact_video.py')
        paths={r['path'] for r in manifest['pinned_files']}
        self.assertIn('experiments/public_rgb_dense_head/compact_head.py',paths)
        self.assertIn(manifest['entrypoint'],paths)
        self.assertFalse(manifest['runtime_release_approved'])

    def test_old_models_and_failed_attempts_remain_unchanged(self):
        d.o.parent.check_hashes(self.p['protected_sources'])
        self.assertFalse(d.o.parent.read(d.o.DATA/'rgb-evaluation-v1/evaluation.json')['public_keypoint_quality_passed'])
        self.assertFalse(d.o.parent.read(d.o.DATA/'rgb-evaluation-v3/evaluation.json')['public_keypoint_quality_passed'])


if __name__=='__main__':unittest.main(verbosity=2)
