"""Bounded small-head search. No fitting/selection access to test features."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from head import CODE, SCHEMA, FullPoseHead, count_fields, load
from descriptor import INPUT_DIMENSION
sys.path.insert(0, str(CODE/'experiments/public_rgb_optimization'))
import optimize_head as o
from quality_head import export_classifier

DATA = CODE/'data/public_cpr'
SOURCE = DATA/'fullpose-v2'
OUTPUT = DATA/'rgb-evaluation-v6'
REPORT = CODE/'test-results/public-cpr-fullpose-v1'
SEED = 20261022


def freeze():
    if OUTPUT.exists():
        raise FileExistsError('Preserve existing experiment')
    audit = o.parent.read(SOURCE/'extraction-audit.json')
    p = o.checked(DATA/'rgb-evaluation-v4')
    feature = o.parent.read(SOURCE/'feature-plan.json')
    if feature['descriptor_sha256'] != o.parent.digest(Path(__file__).with_name('descriptor.py')):
        raise ValueError('Descriptor changed after extraction')
    feature_profile = hashlib.sha256(json.dumps({'base': p['feature_profile_hash'], 'extra': feature['extra_profile_hash']}, sort_keys=True).encode()).hexdigest()
    sources = [Path(__file__).resolve(), Path(__file__).with_name('head.py'), Path(__file__).with_name('descriptor.py')]
    protected = o.parent.read(DATA/'rgb-video-acceptance-v4/model-lock.json')['protected_files']
    for path in [DATA/'rgb-video-acceptance-v4/evaluation.json', DATA/'rgb-evaluation-v5/evaluation.json', SOURCE/'feature-plan.json', SOURCE/'extraction-audit.json']:
        protected[path.relative_to(CODE).as_posix()] = o.parent.digest(path)
    protocol = {'schema': 'cpr-fullpose-head-protocol-v1', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'source_revision': p['source_revision'], 'feature_profile_hash': feature_profile, 'base_feature_profile_hash': p['feature_profile_hash'],
        'extra_profile_hash': feature['extra_profile_hash'], 'input_dimension': INPUT_DIMENSION, 'split_counts': p['split_counts'],
        'gates': p['gates'], 'raw_test_reservations': p['raw_test_reservations'], 'source_split_sha256': p['split_sha256'],
        'development_sha256': audit['development_sha256'], 'locked_test_sha256': audit['test_sha256'],
        'implementation_sources': {path.relative_to(CODE).as_posix(): o.parent.digest(path) for path in sources},
        'protected_sources': protected,
        'candidate_policy': '4 tree heads (full/invariant x32/64), 3 balanced clipped linear heads C=.03/.1/1, MLP32, 4 best-linear/tree32 mixtures weights .5/.75, plus old v4 control. Each evaluated independently or with one train-only learned cardinality classifier; <=70000 scalar fields.',
        'cardinality_policy': 'Multinomial logistic C=.1 over train-only normalized features; predicts 0..4 errors from visual features, no filename/ground-truth metadata. Top-k validation-threshold-adjusted logits. Applied identically to all single/composite inputs.',
        'selection_policy': 'Validation gates single>=.85/composite>=.8/min recall>=.6/zero Correct FP, <=70000 fields; rank worst-cohort F1, mean F1, then smaller cohort-balanced Brier and field count. Pick exactly one before test.',
        'promotion_policy': 'All unchanged 386-record gates and 26-video gates must pass. Same fixed engineering regression, not blind evidence. Never change thresholds after test.',
        'training': 'only small heads; 738 fit and 196 validation. No collection, no manual labels, no backbone training.',
        'prior_test_exposure': 'v4 386-record and 26-video metrics inspected; v5 rejected. This stage adds hand and original-time features using unchanged public labels/splits.'}
    o.write(OUTPUT/'protocol.json', protocol)
    print('Fullpose small-head protocol frozen', flush=True)


def checked():
    p = o.parent.read(OUTPUT/'protocol.json')
    o.parent.check_hashes(p['implementation_sources']); o.parent.check_hashes(p['protected_sources'])
    for name, key in [('development', 'development_sha256'), ('locked-test', 'locked_test_sha256')]:
        if o.parent.digest(SOURCE/(name+'.npz')) != p[key]:
            raise ValueError('Feature bundle changed')
    return p


def measured(probability, truth, cohorts, predictions):
    return {name: o.parent.metrics(predictions[mask].astype(float), truth[mask]) for name, mask in
        [('all', np.ones(len(truth), bool)), ('single', cohorts == 'single'), ('composite', cohorts == 'composite')]}


def train():
    p = checked()
    if (OUTPUT/'selection.json').exists() or list(OUTPUT.glob('*.pt')):
        raise FileExistsError('Preserve previous fitting')
    torch.set_num_threads(4)
    with np.load(SOURCE/'development.npz', allow_pickle=False) as f:
        x, y, split, cohorts, ids = [f[k].copy() for k in ['features', 'targets', 'splits', 'cohorts', 'record_ids']]
    if set(map(str, ids)) & set(p['raw_test_reservations']):
        raise ValueError('Reserved test data leaked')
    fit, val = split == 'train', split == 'validation'
    xf, xv = x[fit].astype(np.float64), x[val].astype(np.float64)
    yf, yv, cv = y[fit], y[val], cohorts[val]
    target = np.column_stack([yf, ~yf.any(1)])
    mean, scale = xf.mean(0), xf.std(0); scale[scale < 1e-6] = 1.
    zf, zv = np.clip((xf-mean)/scale, -6, 6), np.clip((xv-mean)/scale, -6, 6)
    with threadpool_limits(limits=4):
        card = LogisticRegression(C=.1, class_weight='balanced', max_iter=2000, random_state=SEED).fit(zf, yf.sum(1).astype(int))
    card_state = {'indices': torch.arange(INPUT_DIMENSION), 'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),
                  'weight': torch.from_numpy(card.coef_), 'bias': torch.from_numpy(card.intercept_), 'classes': torch.from_numpy(card.classes_)}
    rows, components, probabilities = [], {}, {}
    invariant = np.r_[np.arange(783), np.arange(783+408, INPUT_DIMENSION)]

    def register(name, parts, fixed_op=None):
        state = {'schema': SCHEMA, 'components': parts, 'input_dimension': INPUT_DIMENSION, 'label_ids': list(o.parent.LABEL_IDS),
            'source_revision': p['source_revision'], 'feature_profile_hash': p['feature_profile_hash'],
            'protocol_sha256': o.parent.digest(OUTPUT/'protocol.json'), 'prediction_rule': 'independent'}
        probability = FullPoseHead(state).probability(xv)
        op = fixed_op or o.parent.fit_quality_head.tune_operating_point(probability, yv)
        state['operating_point'] = op
        for mode in ['independent', 'learned_cardinality']:
            current = {**state, 'prediction_rule': mode}
            if mode == 'learned_cardinality':
                current['cardinality'] = card_state
            fields = count_fields({'components': parts, 'cardinality': current.get('cardinality')})
            if fields > 70000:
                continue
            model = FullPoseHead(current)
            pred = model.predict(xv, probability)
            metrics = measured(probability, yv, cv, pred)
            single, composite = metrics['single']['macro_f1'], metrics['composite']['macro_f1']
            recall = min(r['recall'] for r in metrics['all']['per_label'])
            eligible = single >= .85 and composite >= .8 and recall >= .6 and metrics['all']['correct_false_alarm_rate'] == 0
            candidate_name = name+'_'+mode
            path = OUTPUT/(candidate_name+'.pt'); torch.save(current, path)
            restored, _ = load(path, p['feature_profile_hash'], o.parent.digest(OUTPUT/'protocol.json'))
            np.testing.assert_array_equal(restored.probability(xv), probability)
            np.testing.assert_array_equal(restored.predict(xv), pred)
            brier = float(np.mean([((probability[cv == c, :11]-yv[cv == c])**2).mean() for c in ['single', 'composite']]))
            row = {'name': candidate_name, 'base_name': name, 'checkpoint': path.name, 'checkpoint_sha256': o.parent.digest(path),
                'scalar_fields': fields, 'prediction_rule': mode, 'eligible': bool(eligible), 'operating_point': op,
                'validation_cohorts': metrics, 'validation_brier': brier,
                'validation_rank': [min(single, composite), (single+composite)/2, -brier, -fields]}
            rows.append(row)
            print(candidate_name, 'validation', round(single, 6), round(composite, 6), 'fields', fields, 'eligible', eligible, flush=True)
        components[name] = parts; probabilities[name] = probability

    old = torch.load(DATA/'rgb-evaluation-v4/geometry32_linear0.1_0.75.pt', map_location='cpu', weights_only=True)
    control = [{**old['components'][0]}, {**old['components'][1], 'indices': torch.arange(783), 'clip': None}]
    register('v4_control', control, old['operating_point'])
    for strength in [.03, .1, 1.]:
        weights, bias = [], []
        with threadpool_limits(limits=4):
            for label in range(12):
                learner = LogisticRegression(C=strength, class_weight='balanced', solver='liblinear', max_iter=2000, random_state=SEED).fit(zf, target[:, label])
                weights.append(learner.coef_[0]); bias.append(learner.intercept_[0])
        linear = {'kind': 'linear', 'mixture_weight': 1., 'indices': torch.arange(INPUT_DIMENSION), 'mean': torch.from_numpy(mean),
            'scale': torch.from_numpy(scale), 'clip': 6., 'weight1': torch.from_numpy(np.stack(weights)), 'bias1': torch.tensor(bias, dtype=torch.float64)}
        register('linear'+str(strength), [linear])
    for view, indices in [('full', np.arange(INPUT_DIMENSION)), ('invariant', invariant)]:
        for trees in [32, 64]:
            classifiers = []
            with threadpool_limits(limits=4):
                for label in range(12):
                    learner = HistGradientBoostingClassifier(max_iter=trees, max_leaf_nodes=7, max_depth=3, min_samples_leaf=8,
                        l2_regularization=1., learning_rate=.1, early_stopping=False, random_state=SEED).fit(xf[:, indices], target[:, label])
                    exported = export_classifier(learner)
                    for tree in exported['trees']:
                        tree['feature'] = torch.from_numpy(indices[tree['feature'].numpy()].astype(np.int32))
                    classifiers.append(exported)
            register(view+str(trees), [{'kind': 'tree', 'mixture_weight': 1., 'classifiers': classifiers}])
    # One small, CPU-only MLP. Checkpoint selection uses validation loss only.
    torch.manual_seed(SEED)
    net = torch.nn.Sequential(torch.nn.Linear(INPUT_DIMENSION, 32), torch.nn.ReLU(), torch.nn.Dropout(.2), torch.nn.Linear(32, 12))
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    tx, ty = torch.tensor(zf, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    vx = torch.tensor(zv, dtype=torch.float32)
    vy = torch.tensor(np.column_stack([yv, ~yv.any(1)]), dtype=torch.float32)
    positive = torch.tensor(np.sqrt((len(target)-target.sum(0))/np.maximum(target.sum(0), 1)), dtype=torch.float32)
    best, saved, stale = float('inf'), None, 0
    for epoch in range(800):
        net.train(); optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(tx), ty, pos_weight=positive); loss.backward(); optimizer.step()
        net.eval()
        with torch.inference_mode():
            value = float(torch.nn.functional.binary_cross_entropy_with_logits(net(vx), vy, pos_weight=positive))
        if value < best-1e-5:
            best, saved, stale = value, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= 80:
            break
    mlp = {'kind': 'mlp', 'mixture_weight': 1., 'indices': torch.arange(INPUT_DIMENSION), 'mean': torch.from_numpy(mean),
        'scale': torch.from_numpy(scale), 'clip': 6., 'weight1': saved['0.weight'], 'bias1': saved['0.bias'], 'weight2': saved['3.weight'], 'bias2': saved['3.bias']}
    register('mlp32', [mlp])
    best_linear = max([r for r in rows if r['base_name'].startswith('linear')], key=lambda r: tuple(r['validation_rank']))['base_name']
    for tree in ['full32', 'invariant32']:
        for weight in [.5, .75]:
            register(tree+'_'+best_linear+'_'+str(weight), [{**components[tree][0], 'mixture_weight': weight}, {**components[best_linear][0], 'mixture_weight': 1-weight}])
    eligible = [r for r in rows if r['eligible'] and r['base_name'] != 'v4_control']
    chosen = max(eligible, key=lambda r: tuple(r['validation_rank'])) if eligible else None
    o.write(OUTPUT/'selection.json', {'selected': chosen, 'all_candidates': rows, 'protocol_sha256': o.parent.digest(OUTPUT/'protocol.json'),
        'test_read_during_fitting_or_selection': False, 'fitted_records': int(fit.sum()), 'validation_records': int(val.sum()),
        'normalization_fitted_on_train_only': True, 'raw_reserved_in_fit_or_tuning': 0, 'backbone_training': False})
    print('Frozen development winner', None if chosen is None else chosen['name'], flush=True)


def evaluate():
    p = checked(); s = o.parent.read(OUTPUT/'selection.json'); chosen = s['selected']
    if chosen is None or (OUTPUT/'evaluation.json').exists():
        raise ValueError('No selected challenger, or evaluation already frozen')
    model, state = load(OUTPUT/chosen['checkpoint'], p['feature_profile_hash'], o.parent.digest(OUTPUT/'protocol.json'))
    with np.load(SOURCE/'locked-test.npz', allow_pickle=False) as data:
        x, y, cohorts, ids = [data[k].copy() for k in ['features', 'targets', 'cohorts', 'record_ids']]
    probability = model.probability(x); predictions = model.predict(x, probability)
    all_metrics = measured(probability, y, cohorts, predictions)
    reserved = np.asarray([str(k) in p['raw_test_reservations'] for k in ids])
    all_metrics['reserved26'] = o.parent.metrics(predictions[reserved].astype(float), y[reserved])
    g = p['gates']
    checks = {'single_f1': all_metrics['single']['macro_f1'] >= g['single_macro_f1'], 'composite_f1': all_metrics['composite']['macro_f1'] >= g['composite_macro_f1'],
        'all_label_recall': min(r['recall'] for r in all_metrics['all']['per_label']) >= g['minimum_label_recall'],
        'normal_false_alarm': all_metrics['all']['correct_false_alarm_rate'] <= g['correct_false_alarm_rate'],
        'all_386_retained': len(x) == 386, 'all_26_reserved_retained': int(reserved.sum()) == 26, 'head_budget': chosen['scalar_fields'] <= 70000}
    with (OUTPUT/'test-predictions.npz').open('xb') as handle:
        np.savez_compressed(handle, scores=probability, predictions=predictions, targets=y, record_ids=ids)
    result = {'evaluations': all_metrics, 'quality_checks': checks, 'public_keypoint_quality_passed': all(checks.values()),
        'raw_video_quality_passed': False, 'runtime_release_approved': False, 'independent_blind_test': False,
        'checkpoint_sha256': chosen['checkpoint_sha256'], 'protocol_sha256': o.parent.digest(OUTPUT/'protocol.json')}
    o.write(OUTPUT/'evaluation.json', result)
    print({k: (v['macro_f1'], v['exact_match']) for k, v in all_metrics.items()}, checks, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze', 'train', 'evaluate'])
    globals()[parser.parse_args().stage]()
