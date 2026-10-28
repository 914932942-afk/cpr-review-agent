import json
import unittest
from unittest.mock import patch
import numpy as np
import torch

import rgb_features_v3 as r
import train_rgb_experts_v3 as d


class RGBContracts(unittest.TestCase):
    def test_distal_crop_covers_beyond_wrist_and_retains_image_boundary(self):
        actor = np.ones((17, 3))*100
        actor[:, 2] = 1
        actor[[5, 6], :2] = [[160, 80], [200, 80]]
        actor[[7, 8], :2] = [[160, 120], [200, 120]]
        actor[[9, 10], :2] = [[160, 220], [200, 220]]
        actor[[11, 12], :2] = [[160, 250], [200, 250]]
        frame = np.zeros((400, 400, 3), np.uint8)
        boxes = []
        def capture(image, box):
            boxes.append(box)
            return 'image'
        with patch.object(r, 'crop_image', capture):
            r.roi_and_context(frame, actor, None, None)
        self.assertEqual((boxes[0][1]+boxes[0][3])/2, 250)
        self.assertGreater(boxes[0][3], 270)
        self.assertEqual(r.crop_image(frame, [-100, -100, 200, 300]).size, (224, 224))
        with self.assertRaises(ValueError):
            r.crop_image(frame, [500, 500, 600, 600])

    def test_partial_controls_preserve_unsupported_error_not_correct(self):
        controls = json.loads((d.DATA/'rgb-partial-controls-v1/manifest.json').read_text(encoding='utf-8'))['records']
        self.assertEqual(len(controls), 2)
        for row in controls:
            self.assertEqual(row['original_labels'], [12])
            self.assertFalse(row['can_train_correct_auxiliary'])
            self.assertEqual(row['split'], 'train')
        with np.load(d.FEATURES/'features.npz', allow_pickle=False) as f:
            mask = f['partial_label_control']
            self.assertEqual(mask.sum(), 2)
            self.assertFalse(f['original_correct'][mask].any())
            self.assertFalse(f['targets'][mask].any())
            self.assertEqual(f['features'].shape[1], r.DIMENSION)
            self.assertTrue(np.isfinite(f['features']).all())

    def test_no_locked_test_records_in_development(self):
        rows = json.loads((d.FEATURES/'manifest.json').read_text(encoding='utf-8'))['records']
        split = json.loads((d.DATA/'rgb-evaluation-v4/record-splits.json').read_text(encoding='utf-8'))
        # Source schema intentionally checked instead of relying on filename conventions.
        if isinstance(split, dict) and 'records' in split:
            split = split['records']
        tests = {v['record_id'] for v in split if v['application_split'] == 'test' or v['reserved_for_rgb_acceptance']}
        self.assertEqual(len(tests), 386)
        self.assertFalse(tests & {v['record_id'] for v in rows})
        self.assertEqual(len({v['record_id'] for v in rows}), len(rows))

    def test_binary_probability_uses_explicit_threshold_without_top_one(self):
        state = {'experts': [{'label_id': 1, 'indices': torch.tensor([0]), 'mean': torch.tensor([0.]),
            'scale': torch.tensor([1.]), 'weight': torch.tensor([1.]), 'bias': 0., 'threshold': .7}]}
        probability, prediction = d.predict_experts(np.array([[-10.], [0.], [10.]]), state)[1]
        self.assertTrue(np.isfinite(probability).all())
        np.testing.assert_array_equal(prediction, [False, False, True])


if __name__ == '__main__':
    unittest.main()
