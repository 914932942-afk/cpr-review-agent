import unittest
import numpy as np
import torch
from head import CODE,count_fields
from train_head import o

DATA=CODE/'data/public_cpr';FINAL=DATA/'rgb-evaluation-v15'


class FinalCandidateContracts(unittest.TestCase):
    def test_actual_budget_counts_original_rgb_hand_fallbacks(self):
        bundle=o.parent.read(FINAL/'bundle-manifest-v3.json')
        rgb=torch.load(DATA/'rgb-evaluation-v12/experts.pt',map_location='cpu',weights_only=True)
        hand=torch.load(DATA/'rgb-evaluation-v14/hands.pt',map_location='cpu',weights_only=True)
        card=torch.load(DATA/'rgb-evaluation-v13/cardinality.pt',map_location='cpu',weights_only=True)
        actual=rgb['combined_scalar_fields']+count_fields(hand['experts'])+count_fields(card)
        self.assertEqual(actual,bundle['active_scalar_fields_including_fallback'])
        self.assertEqual(actual,69755);self.assertLessEqual(actual,70000)

    def test_all_original_video_targets_are_retained_and_metrics_recompute(self):
        original={r['record_id']:r['original_labels'] for r in o.parent.read(DATA/'rgb-video-acceptance-v4/targets.json')}
        with np.load(FINAL/'raw-predictions.npz',allow_pickle=False) as f:
            ids,pred,truth=[f[k].copy() for k in ['record_ids','predictions','targets']]
        self.assertEqual(len(ids),26);self.assertEqual(set(map(str,ids)),set(original))
        public=np.asarray([[label in original[str(key)] for label in o.parent.LABEL_IDS] for key in ids],bool)
        np.testing.assert_array_equal(truth,public)
        tp=(pred & truth).sum(0);fp=(pred & ~truth).sum(0);fn=(~pred & truth).sum(0)
        f1=2*tp/np.maximum(2*tp+fp+fn,1);recall=tp/np.maximum(tp+fn,1)
        self.assertGreaterEqual(f1.mean(),.85);self.assertGreaterEqual(recall.min(),.6)
        self.assertEqual((pred[~truth.any(1)].any(1)).sum(),0)
        self.assertEqual(int((~truth.any(1)).sum()),2)
        result=o.parent.read(FINAL/'raw-evaluation.json')
        self.assertAlmostEqual(f1.mean(),result['raw_metrics']['macro_f1'])
        self.assertEqual(int((pred==truth).all(1).sum()),result['exact_matches'])

    def test_frozen_runtime_sources_and_model_assets_match(self):
        bundle=o.parent.read(FINAL/'bundle-manifest-v3.json')
        o.parent.check_hashes(bundle['pinned_sources_and_assets'])
        self.assertFalse(bundle['runtime_release_approved'])
        self.assertEqual(bundle['entrypoint'],'experiments/public_cpr_fullpose/infer_candidate_v2.py')

    def test_three_label_free_cold_videos_replay_the_frozen_result(self):
        root=FINAL/'cold-runs-v2';summary=o.parent.read(root/'summary.json')
        self.assertTrue(summary['passed']);self.assertEqual(summary['completed'],3)
        for row in summary['rows']:
            self.assertTrue(row['passed']);self.assertFalse(row['labels_provided_to_inference'])
            self.assertLessEqual(row['score_max_difference'],1e-5)
            self.assertTrue(row['predictions_identical']);self.assertTrue(row['no_reference_labels'])
            value=o.parent.read(root/row['record_id'].replace('/','--')/'inference.json')
            self.assertIsNone(value['public_label_verification'])
            self.assertFalse(value['runtime_release_approved'])
            self.assertFalse(value['model_or_threshold_modified'])


if __name__=='__main__':unittest.main()
