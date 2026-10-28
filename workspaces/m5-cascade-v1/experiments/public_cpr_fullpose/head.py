"""Portable small heads, with an optional learned error-cardinality decision."""
from pathlib import Path
import sys
import numpy as np
import torch

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE/'experiments/public_rgb_dense_head'))
from compact_head import count_fields, LABEL_IDS, REVISION
from quality_head import PortableHead
from descriptor import INPUT_DIMENSION

SCHEMA = 'cpr-fullpose-small-head-v1'


class FullPoseHead:
    def __init__(self, state):
        self.state = state
        self.parts = []
        for component in state['components']:
            if component['kind'] == 'tree':
                self.parts.append((component, PortableHead({'input_dimension': INPUT_DIMENSION, 'classifiers': component['classifiers']})))
            else:
                c = {k: v.numpy().astype(np.float64) if isinstance(v, torch.Tensor) and k != 'indices' else v for k, v in component.items()}
                c['indices'] = c['indices'].numpy()
                self.parts.append((c, None))

    @staticmethod
    def normal(x, c):
        z = (x[:, c['indices']]-c['mean'])/c['scale']
        return np.clip(z, -c['clip'], c['clip']) if c.get('clip') is not None else z

    def probability(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != INPUT_DIMENSION or not np.isfinite(x).all():
            raise ValueError('Invalid fullpose feature contract')
        logits = np.zeros((len(x), 12))
        for c, tree in self.parts:
            if tree is not None:
                p = np.clip(tree.probability(x), 1e-12, 1-1e-12)
                raw = np.log(p/(1-p))
            else:
                raw = self.normal(x, c) @ c['weight1'].T + c['bias1']
                if c['kind'] == 'mlp':
                    raw = np.maximum(raw, 0) @ c['weight2'].T + c['bias2']
            logits += c['mixture_weight']*raw
        return 1/(1+np.exp(-np.clip(logits, -60, 60)))

    def predict(self, x, probability=None):
        probability = self.probability(x) if probability is None else probability
        op = self.state['operating_point']
        thresholds = np.asarray(op['thresholds'])
        if self.state['prediction_rule'] == 'independent':
            errors = probability[:, :11] >= thresholds
        else:
            c = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in self.state['cardinality'].items()}
            z = np.clip((np.asarray(x)[:, c['indices']]-c['mean'])/c['scale'], -6, 6)
            logits = z @ c['weight'].T+c['bias']
            count = c['classes'][logits.argmax(1)].astype(int)
            margin = np.log(np.clip(probability[:, :11], 1e-12, 1-1e-12)/(1-np.clip(probability[:, :11], 1e-12, 1-1e-12)))
            margin -= np.log(np.clip(thresholds, 1e-9, 1-1e-9)/(1-np.clip(thresholds, 1e-9, 1-1e-9)))
            order = np.argsort(-margin, axis=1, kind='stable')
            errors = np.zeros((len(x), 11), dtype=bool)
            for i, k in enumerate(count):
                errors[i, order[i, :k]] = True
        if op['normal_gate'] is not None:
            errors &= (probability[:, 11] < op['normal_gate'])[:, None]
        return errors


def load(path, profile_hash, protocol_hash):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state.get('schema') != SCHEMA or state.get('label_ids') != list(LABEL_IDS)
        or state.get('source_revision') != REVISION or state.get('input_dimension') != INPUT_DIMENSION
        or state.get('feature_profile_hash') != profile_hash or state.get('protocol_sha256') != protocol_hash
        or state.get('prediction_rule') not in ['independent', 'learned_cardinality']
        or count_fields({'components': state['components'], 'cardinality': state.get('cardinality')}) > 70000):
        raise ValueError('Incompatible fullpose model/labels/profile/protocol/budget')
    return FullPoseHead(state), state
