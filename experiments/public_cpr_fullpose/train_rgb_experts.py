"""Small error-specific RGB/context heads with leave-combination-out selection."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits

from rgb_features import CODE, DIMENSION
from train_head import o
from head import count_fields

DATA = CODE/'data/public_cpr'
FEATURES = DATA/'rgb-domain-features-v1'
OUTPUT = DATA/'rgb-evaluation-v8'
LABELS = [1, 2, 5, 6, 7, 9]


def freeze():
    if OUTPUT.exists():
        raise FileExistsError('Preserve RGB expert experiment')
    manifest = o.parent.read(FEATURES/'manifest.json')
    profile = o.parent.read(FEATURES/'profile.json')
    o.parent.check_hashes(profile['implementation_sources'])
    source_files = [Path(__file__).resolve(), CODE/'experiments/public_cpr_fullpose/rgb_features.py']
    base = DATA/'rgb-evaluation-v6'
    chosen = o.parent.read(base/'selection.json')['selected']
    p = {'schema': 'cpr-rgb-expert-protocol-v1', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'feature_dimension': DIMENSION, 'features_sha256': o.parent.digest(FEATURES/'features.npz'),
        'manifest_sha256': o.parent.digest(FEATURES/'manifest.json'), 'profile_sha256': o.parent.digest(FEATURES/'profile.json'),
        'implementation_sources': {path.relative_to(CODE).as_posix(): o.parent.digest(path) for path in source_files},
        'base_head': (base/chosen['checkpoint']).relative_to(CODE).as_posix(), 'base_head_sha256': chosen['checkpoint_sha256'],
        'base_protocol': (base/'protocol.json').relative_to(CODE).as_posix(), 'base_selection_sha256': o.parent.digest(base/'selection.json'),
        'targets': LABELS, 'selection': 'Train groups only. Leave-one-public-error-combination-out, no record/frame crossing. C=.003/.03/.3/3; validation-only OOF thresholds .1.. .9. Prioritize OOF recall>=.8 and specificity>=.5, maximize F1 then balanced accuracy; within .02 F1 prefer fewer coefficients.',
        'feature_scopes': {'1/2': 'grayscale hand RGB means/std768', '5/6/7': 'own-YOLO sampled body geometry + unanchored motion305', '9': 'manikin-relative contact24 or contact + manikin-context RGB792'},
        'fusion': 'Replace the six specified independent binary decisions; retain v6 for 3/4/8/11/13. Same rule for single and composite inputs. All 26 records retained.',
        'raw_gates': o.parent.read(DATA/'rgb-video-acceptance-v4/acceptance-protocol.json')['quality_gates'],
        'training': 'small binary heads only, no backbone updates, no new annotation, original non-test groups only',
        'prior_exposure': 'v4/v6/v7 engineering regression inspected; not independent blind evidence', 'maximum_head_parameters': 70000}
    o.write(OUTPUT/'protocol.json', p)
    print('RGB expert selection protocol frozen', flush=True)


def checked():
    p = o.parent.read(OUTPUT/'protocol.json')
    o.parent.check_hashes(p['implementation_sources'])
    if o.parent.digest(FEATURES/'features.npz') != p['features_sha256'] or o.parent.digest(FEATURES/'profile.json') != p['profile_sha256']:
        raise ValueError('RGB features changed')
    if o.parent.digest(CODE/p['base_head']) != p['base_head_sha256']:
        raise ValueError('Body head changed')
    return p


def binary_metrics(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    recall = tp/max(tp+fn, 1); specificity = tn/max(tn+fp, 1)
    return {'f1': 2*tp/max(2*tp+fp+fn, 1), 'recall': recall, 'specificity': specificity,
            'balanced_accuracy': (recall+specificity)/2, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def train():
    p = checked()
    if (OUTPUT/'experts.pt').exists():
        raise FileExistsError('Preserve RGB expert fit')
    with np.load(FEATURES/'features.npz', allow_pickle=False) as f:
        x, y, split, ids, labels = [f[k].copy() for k in ['features', 'targets', 'splits', 'record_ids', 'label_ids']]
    manifest = {r['record_id']: r for r in o.parent.read(FEATURES/'manifest.json')['records']}
    fit, val = split == 'train', split == 'validation'
    xf, xv = x[fit].astype(np.float64), x[val].astype(np.float64)
    yf, yv = y[fit], y[val]
    groups = np.asarray([','.join(map(str, sorted(manifest[str(key)]['original_labels']))) for key in ids[fit]])
    scopes = {'hand': np.r_[np.arange(384), np.arange(768, 1152)], 'body': np.arange(1536, 1841),
              'contact': np.arange(1841, 1865), 'contact_visual': np.r_[np.arange(384, 768), np.arange(1152, 1536), np.arange(1841, 1865)]}
    experts, rows = [], []
    for label in LABELS:
        k = list(labels).index(label)
        target = yf[:, k]
        candidates = []
        names = ['hand'] if label in [1, 2] else (['body'] if label in [5, 6, 7] else ['contact', 'contact_visual'])
        if len(np.unique(groups[target])) < 2 or len(np.unique(groups[~target])) < 2:
            raise ValueError('Insufficient independent combinations for error '+str(label))
        for name in names:
            indices = scopes[name]
            values = xf[:, indices]
            for strength in [.003, .03, .3, 3.]:
                oof = np.full(len(values), np.nan)
                for train, test in LeaveOneGroupOut().split(values, target, groups):
                    if len(np.unique(target[train])) != 2:
                        raise ValueError('Combination holdout left only one training class')
                    mean, scale = values[train].mean(0), values[train].std(0); scale[scale < 1e-6] = 1.
                    with threadpool_limits(limits=2):
                        model = LogisticRegression(C=strength, class_weight='balanced', solver='liblinear', max_iter=2000).fit(
                            np.clip((values[train]-mean)/scale, -6, 6), target[train])
                    oof[test] = model.predict_proba(np.clip((values[test]-mean)/scale, -6, 6))[:, 1]
                if not np.isfinite(oof).all():
                    raise ValueError('OOF predictions incomplete')
                for threshold in np.linspace(.1, .9, 17):
                    metric = binary_metrics(oof >= threshold, target)
                    candidates.append({'scope': name, 'C': strength, 'threshold': float(threshold), 'oof': metric,
                        'preferred': metric['recall'] >= .8 and metric['specificity'] >= .5, 'dimension': len(indices)})
        preferred = [r for r in candidates if r['preferred']]
        best = max(preferred or candidates, key=lambda r: (r['oof']['f1'], r['oof']['balanced_accuracy']))
        comparable = [r for r in (preferred or candidates) if r['oof']['f1'] >= best['oof']['f1']-.02]
        best = max(comparable, key=lambda r: (-r['dimension'], r['oof']['f1'], r['oof']['balanced_accuracy'], -abs(r['threshold']-.5), -r['C']))
        indices = scopes[best['scope']]
        values = xf[:, indices]
        mean, scale = values.mean(0), values.std(0); scale[scale < 1e-6] = 1.
        with threadpool_limits(limits=2):
            model = LogisticRegression(C=best['C'], class_weight='balanced', solver='liblinear', max_iter=2000).fit(np.clip((values-mean)/scale, -6, 6), target)
        expert = {'label_id': label, 'indices': torch.from_numpy(indices), 'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),
            'weight': torch.from_numpy(model.coef_[0]), 'bias': float(model.intercept_[0]), 'threshold': best['threshold']}
        experts.append(expert)
        validation_probability = model.predict_proba(np.clip((xv[:, indices]-mean)/scale, -6, 6))[:, 1]
        rows.append({'label_id': label, 'selected': best, 'all_candidate_metrics': candidates,
            'heldout_validation': binary_metrics(validation_probability >= best['threshold'], yv[:, k]),
            'validation_support': int(yv[:, k].sum()), 'training_support': int(target.sum())})
        print('Expert', label, best['scope'], 'OOF', best['oof'], 'validation', rows[-1]['heldout_validation'], flush=True)
    base = torch.load(CODE/p['base_head'], map_location='cpu', weights_only=True)
    fields = count_fields(base['components'])+count_fields(experts)
    if fields > p['maximum_head_parameters']:
        raise ValueError('Combined body/RGB small-head budget exceeded')
    state = {'schema': 'cpr-rgb-specialists-v1', 'experts': experts, 'protocol_sha256': o.parent.digest(OUTPUT/'protocol.json'),
        'feature_profile_sha256': p['profile_sha256'], 'feature_dimension': DIMENSION, 'combined_scalar_fields': fields}
    torch.save(state, OUTPUT/'experts.pt')
    o.write(OUTPUT/'selection.json', {'experts': rows, 'checkpoint_sha256': o.parent.digest(OUTPUT/'experts.pt'),
        'combined_scalar_fields': fields, 'train_records': int(fit.sum()), 'heldout_validation_records': int(val.sum()),
        'OOF_groups': 'entire public error combinations, including all repetitions', 'test_rows_used_for_fit_or_thresholds': 0,
        'main_network_training': False})
    print('RGB specialist heads frozen; combined fields', fields, flush=True)


def predict_experts(x, state):
    result = {}
    for expert in state['experts']:
        c = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in expert.items()}
        z = np.clip((np.asarray(x)[:, c['indices']]-c['mean'])/c['scale'], -6, 6)
        probability = 1/(1+np.exp(-np.clip(z @ c['weight']+c['bias'], -60, 60)))
        result[c['label_id']] = (probability, probability >= c['threshold'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze', 'train'])
    globals()[parser.parse_args().stage]()
