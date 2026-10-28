"""Bounded standardization for a small frozen-feature tree/linear mixture.

An isolated experimental format. Historical compact loaders are not changed.
"""
from pathlib import Path
import math
import sys

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / 'experiments/public_rgb_dense_head'))
from compact_head import CompactHead, count_fields, LABEL_IDS, REVISION

SCHEMA = 'cpr-robust-compact-head-v1'


class RobustHead(CompactHead):
    def __init__(self, state):
        parts = state.get('components', [])
        if len(parts) != 2 or [c.get('kind') for c in parts] != ['tree', 'linear']:
            raise ValueError('Expected one geometry tree and one linear head')
        weights = [c['mixture_weight'] for c in parts]
        if not all(math.isfinite(w) and 0 <= w <= 1 for w in weights) or abs(sum(weights)-1) > 1e-12:
            raise ValueError('Mixture weights must be finite, nonnegative and sum to one')
        linear = parts[1]
        for key, shape in [('mean', (783,)), ('scale', (783,)), ('weight1', (12, 783)), ('bias1', (12,))]:
            value = linear.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError('Invalid linear tensor: '+key)
        if (linear['scale'] <= 0).any():
            raise ValueError('Normalization scales must be positive')
        clip = linear.get('zscore_clip')
        if clip is not None and (not math.isfinite(clip) or clip <= 0):
            raise ValueError('Clipping bound must be positive or None')
        if len(parts[0].get('classifiers', [])) != 12 or count_fields(parts) > 70000:
            raise ValueError('Invalid output count or oversized head')
        super().__init__(state)

    def probability(self, feature):
        x = np.asarray(feature, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 783 or not np.isfinite(x).all():
            raise ValueError('Expected finite [N,783] frozen features')
        logits = np.zeros((len(x), 12))
        for component, tree in self.components:
            if tree is not None:
                p = np.clip(tree.probability(x), 1e-12, 1-1e-12)
                raw = np.log(p/(1-p))
            else:
                normal = (x-component['mean'])/component['scale']
                bound = component.get('zscore_clip')
                if bound is not None:
                    normal = np.clip(normal, -bound, bound)
                raw = normal @ component['weight1'].T + component['bias1']
            logits += component['mixture_weight']*raw
        return 1/(1+np.exp(-np.clip(logits, -60, 60)))


def load(path, profile_hash, protocol_hash):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state.get('schema') != SCHEMA or state.get('label_ids') != list(LABEL_IDS)
        or state.get('source_revision') != REVISION or state.get('feature_profile_hash') != profile_hash
        or state.get('protocol_sha256') != protocol_hash or state.get('input_dimension') != 783
        or not torch.equal(state.get('feature_indices', torch.empty(0)), torch.arange(783))):
        raise ValueError('Incompatible robust head/labels/profile/protocol')
    return RobustHead(state), state
