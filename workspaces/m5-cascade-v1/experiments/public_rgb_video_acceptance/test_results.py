import unittest

import numpy as np

from common import CODE, MODEL, OUTPUT, read, digest, verify_lock
from summarize import scored, quality_checks


class CompletedRunContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p, cls.lock = verify_lock(include_videos=True)
        cls.result = read(OUTPUT/'evaluation.json')
        cls.jobs = read(OUTPUT/'jobs.json')

    def test_every_planned_record_has_an_auditable_attempt(self):
        self.assertEqual(len(self.result['rows']), 26)
        self.assertEqual([r['record_id'] for r in self.result['rows']], [j['record_id'] for j in self.jobs])
        self.assertEqual(self.result['completed_records']+self.result['failed_records'], 26)
        for job in self.jobs:
            folder = OUTPUT/'runs'/job['key']
            attempt = read(folder/'attempt.json')
            self.assertFalse(attempt['ground_truth_supplied_to_inference'])
            self.assertEqual(attempt['attempt_number'], 1)
            for artifact in attempt['artifacts']:
                self.assertEqual(digest(folder/artifact['name']), artifact['sha256'])

    def test_full_scope_and_failure_denominators_are_honest(self):
        expected_frames = sum(j['expected_frames'] for j in self.jobs)
        self.assertEqual(expected_frames, 11215)
        observed = sum(r['analysis']['decoded_frames'] for r in self.result['rows'] if r['analysis'])
        self.assertEqual(self.result['execution_checks']['all_expected_frames_decoded'], observed == expected_frames)
        if self.result['failed_records']:
            self.assertIsNone(self.result['raw_metrics'])
            self.assertFalse(self.result['raw_quality_passed'])
        else:
            self.assertEqual(self.result['raw_metrics']['records'], 26)

    def test_raw_aggregate_metrics_replay_without_changing_thresholds(self):
        if self.result['completed_records'] != 26:
            self.assertFalse((OUTPUT/'raw-video-predictions.npz').exists())
            return
        with np.load(OUTPUT/'raw-video-predictions.npz', allow_pickle=False) as saved:
            self.assertEqual(scored(saved['scores'], saved['targets'], saved['predictions']), self.result['raw_metrics'])
            self.assertEqual(saved['features'].shape, (26, 783))
            self.assertEqual(saved['targets'].shape, (26, 11))
            self.assertEqual(self.result['raw_exact_matches'], int((saved['predictions'] == saved['targets']).all(1).sum()))

    def test_quality_decision_uses_all_predeclared_gates(self):
        measured = self.result['raw_metrics']
        checks = quality_checks(self.p, measured, self.result['completed_records'], 26)
        self.assertEqual(checks, self.result['quality_checks'])
        self.assertEqual(self.result['raw_execution_passed'], all(self.result['execution_checks'].values()))
        self.assertEqual(self.result['raw_quality_passed'], self.result['raw_execution_passed'] and all(checks.values()))

    def test_public_keypoint_comparison_is_same_26_not_the_whole_386(self):
        public = self.result['same_26_public_keypoint_metrics']
        self.assertEqual(public['records'], 26)
        self.assertEqual(public['original_correct_records'], 2)
        self.assertAlmostEqual(public['macro_f1'], read(MODEL/'evaluation.json')['evaluations']['rgb_reserved_public_keypoints']['macro_f1'], places=14)
        self.assertEqual(self.result['public_keypoint_exact_matches'], 9)

    def test_no_unsupported_release_or_generalization_claim(self):
        self.assertFalse(self.result['runtime_release_approved'])
        self.assertFalse(self.result['independent_blind_test'])
        self.assertTrue(self.result['no_fitting_or_threshold_changes'])
        self.assertIn('not evaluated', self.result['composite_raw_video_quality'])
        self.assertIn('not evaluated', self.result['cross_person_generalization'])
        self.assertEqual(self.lock['head_sha256'], digest(CODE/self.lock['head']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
