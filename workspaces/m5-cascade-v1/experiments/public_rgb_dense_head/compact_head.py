"""Tensor-only compact linear/MLP/tree mixtures over unchanged frozen features."""
import sys
from pathlib import Path
import numpy as np
import torch

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / 'experiments/public_cpr_improved'))
from quality_head import PortableHead
from improve import LABEL_IDS, REVISION


def count_fields(value):
    if isinstance(value, torch.Tensor): return value.numel()
    if isinstance(value, (float, int, bool)): return 1
    if isinstance(value, dict): return sum(count_fields(v) for v in value.values())
    if isinstance(value, list): return sum(count_fields(v) for v in value)
    return 0


class CompactHead:
    def __init__(self, state):
        self.components = []
        for component in state['components']:
            if component['kind'] == 'tree':
                self.components.append((component, PortableHead({'input_dimension': 783, 'classifiers': component['classifiers']})))
            else:
                self.components.append(({k: v.numpy().astype(np.float64) if isinstance(v, torch.Tensor) else v for k,v in component.items()}, None))

    def probability(self, feature):
        x = np.asarray(feature, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 783 or not np.isfinite(x).all(): raise ValueError('Expected finite [N,783] frozen features')
        logits = np.zeros((len(x), 12))
        for component, tree in self.components:
            if tree is not None:
                probability = np.clip(tree.probability(x), 1e-12, 1-1e-12)
                raw = np.log(probability / (1-probability))
            else:
                normal = (x-component['mean']) / component['scale']
                raw = normal @ component['weight1'].T + component['bias1']
                if component['kind'] == 'mlp': raw = np.maximum(raw, 0) @ component['weight2'].T + component['bias2']
            logits += component['mixture_weight'] * raw
        return 1 / (1 + np.exp(-np.clip(logits, -60, 60)))


def load(path, profile_hash, protocol_hash):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state.get('schema') != 'cpr-compact-head-v1' or state.get('label_ids') != list(LABEL_IDS)
        or state.get('source_revision') != REVISION or state.get('feature_profile_hash') != profile_hash
        or state.get('protocol_sha256') != protocol_hash or count_fields(state.get('components',[])) > 70000):
        raise ValueError('Incompatible compact head/labels/profile/protocol/budget')
    if not state['components'] or any(c['kind'] not in ['linear','mlp','tree'] for c in state['components']):
        raise ValueError('Unsupported compact component')
    return CompactHead(state), state
