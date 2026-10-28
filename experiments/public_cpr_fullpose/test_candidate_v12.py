import json
import unittest
import numpy as np
import torch
import train_rgb_experts_v5 as d
from head import count_fields
from baseline import digest


class CandidateContracts(unittest.TestCase):
    def test_primary_public_controls_preserve_labels_and_original_split(self):
        rows=d.o.parent.read(d.DATA/'rgb-primary-controls-v1/manifest.json')['records']
        original={r['record_id']:r for r in d.o.parent.read(d.DATA/'rgb-evaluation-v4/record-splits.json')}
        self.assertEqual(len(rows),8)
        for row in rows:
            source=original[row['record_id']]
            self.assertEqual(row['original_labels'],source['original_labels'])
            self.assertEqual(row['original_labels'],[13])
            self.assertEqual(row['split'],source['application_split'])
            self.assertIn(row['split'],['train','validation'])
            self.assertFalse(source['reserved_for_rgb_acceptance'])
            expected=set(d.o.parent.read(d.DATA/'rgb-primary-controls-v1/plan.json')['records'][row['record_id']]['frame_ordinals'])
            self.assertEqual({r['frame_ordinal'] for r in row['images']},expected)

    def test_no_locked_test_record_in_expanded_rgb_training(self):
        original=d.o.parent.read(d.DATA/'rgb-evaluation-v4/record-splits.json')
        locked={r['record_id'] for r in original if r['application_split']=='test'}
        rows=d.o.parent.read(d.FEATURES/'manifest.json')['records']
        self.assertEqual(len(locked),386)
        self.assertFalse(locked & {r['record_id'] for r in rows})
        with np.load(d.FEATURES/'features.npz',allow_pickle=False) as f:
            self.assertTrue(np.isfinite(f['features']).all())
            self.assertEqual(f['features'].shape[1],1893)
            controls=f['partial_label_control']
            self.assertFalse(f['original_correct'][controls].any())
            self.assertFalse(f['targets'][controls].any())
            self.assertEqual(int(controls.sum()),2)

    def test_reused_raw_features_are_byte_identical_and_label_free(self):
        source=d.DATA/'rgb-domain-features-v3/raw26'
        target=d.FEATURES/'raw26'
        old=d.o.parent.read(source/'manifest.json'); new=d.o.parent.read(target/'manifest.json')
        self.assertFalse(new['labels_read']); self.assertFalse(new['failures'])
        self.assertEqual(new['records'],old['records'])
        for row in new['records']:
            self.assertEqual(digest(target/row['feature_file']),row['feature_sha256'])

    def test_selected_checkpoint_integrity_budget_and_active_mechanics(self):
        p=d.checked(); selection=d.o.parent.read(d.OUTPUT/'selection.json')
        self.assertEqual(digest(d.OUTPUT/'experts.pt'),selection['checkpoint_sha256'])
        state=torch.load(d.OUTPUT/'experts.pt',map_location='cpu',weights_only=True)
        mechanics=torch.load(d.CODE/p['mechanics_head'],map_location='cpu',weights_only=True)
        self.assertEqual([e['label_id'] for e in mechanics['experts']],[6])
        original=torch.load(d.DATA/'body-mechanics-v3/experts.pt',map_location='cpu',weights_only=True)
        from train_mechanics_v3 import predict
        with np.load(d.DATA/'body-mechanics-v3/features.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(predict(f['features'],original)[6][0],predict(f['features'],mechanics)[6][0])
        base=torch.load(d.CODE/p['base_head'],map_location='cpu',weights_only=True)
        fields=count_fields(base['components'])+count_fields(state['experts'])+count_fields(mechanics['experts'])+1
        self.assertEqual(fields,state['combined_scalar_fields'])
        self.assertEqual(fields,selection['combined_scalar_fields'])
        self.assertLessEqual(fields,70000)
        self.assertEqual({e['label_id'] for e in state['experts']},set(d.LABELS))
        self.assertEqual(selection['test_rows_used_for_fit_or_thresholds'],0)

    def test_invalid_rgb_input_fails_before_classification(self):
        state={'feature_dimension':1893,'experts':[]}
        for values in [np.zeros((1,1892)),np.full((1,1893),np.nan),np.zeros(1893)]:
            with self.assertRaises(ValueError): d.predict_experts(values,state)


if __name__=='__main__': unittest.main()
