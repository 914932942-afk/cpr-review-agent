import unittest

import numpy as np

from common import CODE, MODEL, OUTPUT, read, digest, verify_lock, validate_result
from summarize import quality_checks, scored, LABEL_IDS


def good_result():
    return {'status': 'completed_experimental', 'video_sha256': 'video', 'manifest_sha256': 'manifest',
        'decoded_frames': 10, 'model_or_threshold_modified': False, 'runtime_release_approved': False,
        'public_label_verification': None, 'repeat_feature_max_difference': 0., 'predicted_original_label_ids': [13],
        'scores': [{'original_label_id': i} for i in LABEL_IDS]}


class PreflightContracts(unittest.TestCase):
    def test_scoring_sources_are_frozen_before_batch_completion(self):
        scoring_lock = read(OUTPUT/'scoring-lock.json')
        for source, expected in scoring_lock['implementation_sources'].items():
            self.assertEqual(digest(CODE/source), expected, source)
        if (OUTPUT/'batch-complete.json').exists():
            self.assertLess(scoring_lock['frozen_utc'], read(OUTPUT/'batch-complete.json')['finished_utc'])

    def test_exact_pipeline_and_unchanged_model_are_hash_locked(self):
        p, lock = verify_lock()
        self.assertEqual(p['record_count'], 26)
        self.assertEqual(p['frame_count'], 11215)
        self.assertEqual(lock['manifest_sha256'], digest(MODEL/'inference-manifest.json'))
        self.assertEqual(lock['head_sha256'], digest(CODE/lock['head']))
        self.assertEqual(lock['head'], read(MODEL/'inference-manifest.json')['head'])

    def test_all_reserved_groups_are_test_only_and_no_truth_is_in_jobs(self):
        jobs = read(OUTPUT/'jobs.json')
        protocol = read(MODEL/'protocol.json')
        split = {r['record_id']: r['application_split'] for r in read(MODEL/'record-splits.json')}
        self.assertEqual({j['record_id'] for j in jobs}, set(protocol['raw_test_reservations']))
        self.assertEqual(len(jobs), len({j['key'] for j in jobs}))
        self.assertTrue(all(split[j['record_id']] == 'test' for j in jobs))
        self.assertTrue(all('original_labels' not in j and 'target' not in j for j in jobs))
        self.assertEqual(sum(j['expected_frames'] for j in jobs), 11215)

    def test_output_contract_rejects_wrong_video_model_frame_or_reference(self):
        job = {'video_sha256': 'video', 'expected_frames': 10}
        validate_result(job, good_result(), 'manifest')
        bad_values = [{'video_sha256': 'other'}, {'manifest_sha256': 'other'}, {'decoded_frames': 9},
            {'public_label_verification': {'label': 13}}, {'model_or_threshold_modified': True},
            {'runtime_release_approved': True}, {'repeat_feature_max_difference': .01},
            {'predicted_original_label_ids': [10]}, {'predicted_original_label_ids': [13, 1]}]
        for values in bad_values:
            with self.assertRaises(ValueError):
                validate_result(job, {**good_result(), **values}, 'manifest')

    def test_failed_video_cannot_be_removed_to_pass_quality(self):
        p = read(OUTPUT/'acceptance-protocol.json')
        result = quality_checks(p, None, 25, 26)
        self.assertFalse(any(result.values()))

    def test_quality_gate_values_match_preexisting_v4_gates(self):
        expected = read(MODEL/'protocol.json')['gates']
        actual = read(OUTPUT/'acceptance-protocol.json')['quality_gates']
        for key in ['single_macro_f1', 'correct_false_alarm_rate', 'minimum_label_recall', 'classification_coverage']:
            self.assertEqual(actual[key], expected[key])

    def test_zero_error_outputs_do_not_hide_missed_errors_or_normal_false_alarms(self):
        truth = np.vstack([np.eye(11, dtype=bool), np.zeros((2, 11), dtype=bool)])
        probability = np.zeros((13, 12))
        none = scored(probability, truth, np.zeros((13, 11), dtype=bool))
        self.assertEqual(none['macro_f1'], 0.)
        self.assertEqual(none['exact_match'], 2/13)
        self.assertEqual(none['correct_false_alarm_rate'], 0.)
        with_fp = truth.copy(); with_fp[-1, 0] = True
        result = scored(probability, truth, with_fp)
        self.assertEqual(result['correct_false_alarm_rate'], .5)
        self.assertLess(result['macro_f1'], 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
