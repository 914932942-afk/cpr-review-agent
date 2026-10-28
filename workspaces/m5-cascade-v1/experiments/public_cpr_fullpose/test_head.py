from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from head import FullPoseHead, INPUT_DIMENSION, SCHEMA, LABEL_IDS, REVISION, load


def state(mode='independent', count=1):
    dimension = INPUT_DIMENSION
    model = {'schema': SCHEMA, 'input_dimension': dimension, 'label_ids': list(LABEL_IDS), 'source_revision': REVISION,
        'feature_profile_hash': 'profile', 'protocol_sha256': 'protocol', 'prediction_rule': mode,
        'components': [{'kind': 'linear', 'mixture_weight': 1., 'indices': torch.arange(dimension),
            'mean': torch.zeros(dimension), 'scale': torch.ones(dimension), 'clip': 6.,
            'weight1': torch.zeros((12, dimension)), 'bias1': torch.arange(12, dtype=torch.float32)/10}],
        'operating_point': {'thresholds': [.5]*11, 'normal_gate': None}}
    bias = torch.full((5,), -10.); bias[count] = 10.
    model['cardinality'] = {'indices': torch.arange(dimension), 'mean': torch.zeros(dimension), 'scale': torch.ones(dimension),
                           'weight': torch.zeros((5, dimension)), 'bias': bias, 'classes': torch.arange(5)}
    return model


class HeadContracts(unittest.TestCase):
    def test_learned_cardinality_handles_zero_single_and_composite_without_metadata(self):
        for count in [0, 1, 2, 3, 4]:
            model = FullPoseHead(state('learned_cardinality', count))
            predicted = model.predict(np.zeros((3, INPUT_DIMENSION)))
            np.testing.assert_array_equal(predicted.sum(1), np.full(3, count))
            if count:
                self.assertTrue(predicted[:, -count:].all())

    def test_normal_gate_is_applied_to_both_decision_modes(self):
        for mode in ['independent', 'learned_cardinality']:
            s = state(mode); s['operating_point']['normal_gate'] = .5
            self.assertFalse(FullPoseHead(s).predict(np.zeros((1, INPUT_DIMENSION))).any())

    def test_wrong_dimension_and_nonfinite_features_fail_closed(self):
        model = FullPoseHead(state())
        for values in [np.zeros((1, 783)), np.full((1, INPUT_DIMENSION), np.nan)]:
            with self.assertRaises(ValueError):
                model.probability(values)

    def test_tensor_only_roundtrip_and_protocol_binding(self):
        folder = Path(__file__).resolve().parents[2]/'test-results/public-cpr-fullpose-v1'
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            path = Path(temporary)/'head.pt'; torch.save(state(), path)
            model, _ = load(path, 'profile', 'protocol')
            np.testing.assert_array_equal(model.predict(np.zeros((1, INPUT_DIMENSION))), np.ones((1, 11), dtype=bool))
            with self.assertRaises(ValueError):
                load(path, 'wrong', 'protocol')


if __name__ == '__main__':
    unittest.main(verbosity=2)
