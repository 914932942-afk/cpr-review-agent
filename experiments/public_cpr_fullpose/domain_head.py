"""Source-balanced small heads; selection explicitly checks validation domains."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import train_head as t
from head import FullPoseHead, SCHEMA, INPUT_DIMENSION, count_fields, load
from quality_head import export_classifier

CODE, DATA, SOURCE, o = t.CODE, t.DATA, t.SOURCE, t.o
OUTPUT = DATA/'rgb-evaluation-v7'


def checked():
    p = o.parent.read(OUTPUT/'protocol.json')
    o.parent.check_hashes(p['implementation_sources']); o.parent.check_hashes(p['protected_sources'])
    for name, key in [('development', 'development_sha256'), ('locked-test', 'locked_test_sha256')]:
        if o.parent.digest(SOURCE/(name+'.npz')) != p[key]:
            raise ValueError('Fullpose features changed')
    return p


def freeze():
    if OUTPUT.exists():
        raise FileExistsError('Preserve experiment')
    p = t.checked()
    sources = dict(p['implementation_sources'])
    sources[Path(__file__).resolve().relative_to(CODE).as_posix()] = o.parent.digest(Path(__file__))
    p = {**p, 'frozen_utc': datetime.now(timezone.utc).isoformat(), 'implementation_sources': sources,
        'candidate_policy': '6 clipped linear heads (C=.03/.1/1 x domain/label-domain weights), 6 trees (full/invariant32 and full64 x both weighting modes), 8 tree32/best-linear mixtures weights .5/.75. Independent multi-label decision only.',
        'selection_policy': 'Same global validation gates, zero normal FP, <=70000 fields; rank minimum F1 across primary-single/primary-composite/supplemental-composite, then global worst-cohort F1, mean F1, smaller Brier and fields. Choose exactly one before regression tests.',
        'threshold_policy': 'Two validation-only alternatives: legacy and domain-balanced per-class F1 grid .1.. .95 with zero Correct false alarms. Select with the same predeclared domain rank.',
        'cause_hypothesis': 'Primary and supplemental capture domains have unequal representation; balance training losses without using source IDs as runtime model inputs.',
        'previous_results': 'v6 public-keypoint gates passed, raw26 F1 .597; v4 raw26 F1 .627. No test samples may be fitted.'}
    o.write(OUTPUT/'protocol.json', p)
    print('Domain-balanced development protocol frozen', flush=True)


def sample_weights(domains, target, mode):
    w = np.ones(len(target))
    for domain in np.unique(domains):
        rows = domains == domain
        w[rows] = 1/rows.sum()
        if mode == 'label_domain' and len(np.unique(target[rows])) == 2:
            for value in [0, 1]:
                mask = rows & (target == value)
                w[mask] = .5/mask.sum()
    return w/w.mean()


def balanced_thresholds(prob, y, domains):
    result = []
    normal = ~y.any(1)
    for k in range(11):
        choices = []
        for threshold in np.linspace(.1, .95, 35):
            pred = prob[:, k] >= threshold
            if (pred & normal).any():
                continue
            scores = []
            for domain in np.unique(domains):
                mask = domains == domain
                truth = y[mask, k].astype(bool); proposed = pred[mask]
                tp = (proposed & truth).sum(); fp = (proposed & ~truth).sum(); fn = (~proposed & truth).sum()
                scores.append(2*tp/max(2*tp+fp+fn, 1))
            choices.append((np.mean(scores), -abs(threshold-.5), threshold))
        result.append(float(max(choices)[2]) if choices else 1.)
    return {'thresholds': result, 'normal_gate': None}


def train():
    p = checked()
    if list(OUTPUT.glob('*.pt')):
        raise FileExistsError('Preserve fit')
    torch.set_num_threads(4)
    with np.load(SOURCE/'development.npz', allow_pickle=False) as f:
        x, y, splits, cohorts, ids = [f[k].copy() for k in ['features', 'targets', 'splits', 'cohorts', 'record_ids']]
    fit, val = splits == 'train', splits == 'validation'
    xf, xv = x[fit].astype(np.float64), x[val].astype(np.float64)
    yf, yv, cv = y[fit], y[val], cohorts[val]
    df = np.asarray(['supplemental' if str(k).startswith('Sup') else 'primary' for k in ids[fit]])
    dv = np.asarray(['supplemental' if str(k).startswith('Sup') else 'primary' for k in ids[val]])
    mean, scale = xf.mean(0), xf.std(0); scale[scale < 1e-6] = 1.
    zf = np.clip((xf-mean)/scale, -6, 6)
    target = np.column_stack([yf, ~yf.any(1)])
    rows, parts_by_name = [], {}
    def register(name, parts):
        state = {'schema': SCHEMA, 'components': parts, 'input_dimension': INPUT_DIMENSION, 'label_ids': list(o.parent.LABEL_IDS),
            'source_revision': p['source_revision'], 'feature_profile_hash': p['feature_profile_hash'],
            'protocol_sha256': o.parent.digest(OUTPUT/'protocol.json'), 'prediction_rule': 'independent'}
        fields = count_fields({'components': parts, 'cardinality': None})
        if fields > 70000:
            return
        prob = FullPoseHead(state).probability(xv)
        options = [o.parent.fit_quality_head.tune_operating_point(prob, yv), balanced_thresholds(prob, yv, dv)]
        proposals = []
        for method, op in zip(['legacy', 'domain_balanced'], options):
            current = {**state, 'operating_point': op}
            pred = FullPoseHead(current).predict(xv, prob)
            metrics = t.measured(prob, yv, cv, pred)
            domains = {name: o.parent.metrics(pred[mask].astype(float), yv[mask]) for name, mask in [
                ('primary_single', (dv == 'primary') & (cv == 'single')),
                ('primary_composite', (dv == 'primary') & (cv == 'composite')),
                ('supplemental_composite', (dv == 'supplemental') & (cv == 'composite'))] if mask.any()}
            single, composite = metrics['single']['macro_f1'], metrics['composite']['macro_f1']
            eligible = single >= .85 and composite >= .8 and metrics['all']['correct_false_alarm_rate'] == 0 and min(r['recall'] for r in metrics['all']['per_label']) >= .6
            brier = float(np.mean([((prob[dv == domain, :11]-yv[dv == domain])**2).mean() for domain in np.unique(dv)]))
            rank = [min(r['macro_f1'] for r in domains.values()), min(single, composite), (single+composite)/2, -brier, -fields]
            proposals.append({'method': method, 'op': op, 'eligible': bool(eligible), 'rank': rank, 'metrics': metrics, 'domains': domains})
        choices = [r for r in proposals if r['eligible']]
        chosen = max(choices or proposals, key=lambda r: tuple(r['rank']))
        state['operating_point'] = chosen['op']
        path = OUTPUT/(name+'.pt'); torch.save(state, path)
        restored, _ = load(path, p['feature_profile_hash'], o.parent.digest(OUTPUT/'protocol.json'))
        np.testing.assert_array_equal(restored.probability(xv), prob)
        row = {'name': name, 'checkpoint': path.name, 'checkpoint_sha256': o.parent.digest(path), 'scalar_fields': fields,
            'eligible': bool(choices), 'validation_rank': chosen['rank'], 'validation_cohorts': chosen['metrics'],
            'validation_domains': chosen['domains'], 'operating_point': chosen['op'], 'threshold_method': chosen['method']}
        rows.append(row); parts_by_name[name] = parts
        print(name, 'single/composite/domain', chosen['metrics']['single']['macro_f1'], chosen['metrics']['composite']['macro_f1'], chosen['rank'][0], flush=True)
    for mode in ['domain', 'label_domain']:
        for c in [.03, .1, 1.]:
            weights, bias = [], []
            with threadpool_limits(limits=4):
                for k in range(12):
                    learner = LogisticRegression(C=c, solver='liblinear', max_iter=2000, random_state=20261022).fit(zf, target[:, k], sample_weight=sample_weights(df, target[:, k], mode))
                    weights.append(learner.coef_[0]); bias.append(learner.intercept_[0])
            register(f'linear_{mode}_{c}', [{'kind': 'linear', 'mixture_weight': 1., 'indices': torch.arange(INPUT_DIMENSION),
                'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale), 'clip': 6.,
                'weight1': torch.from_numpy(np.stack(weights)), 'bias1': torch.tensor(bias, dtype=torch.float64)}])
        invariant = np.r_[np.arange(783), np.arange(783+408, INPUT_DIMENSION)]
        for name, indices, trees in [('full32', np.arange(INPUT_DIMENSION), 32), ('invariant32', invariant, 32), ('full64', np.arange(INPUT_DIMENSION), 64)]:
            outputs = []
            with threadpool_limits(limits=4):
                for k in range(12):
                    learner = HistGradientBoostingClassifier(max_iter=trees, max_leaf_nodes=7, max_depth=3, min_samples_leaf=8, l2_regularization=1.,
                        learning_rate=.1, early_stopping=False, random_state=20261022).fit(xf[:, indices], target[:, k], sample_weight=sample_weights(df, target[:, k], mode))
                    item = export_classifier(learner)
                    for tree in item['trees']:
                        tree['feature'] = torch.from_numpy(indices[tree['feature'].numpy()].astype(np.int32))
                    outputs.append(item)
            register(name+'_'+mode, [{'kind': 'tree', 'mixture_weight': 1., 'classifiers': outputs}])
    best_linear = max([r for r in rows if r['name'].startswith('linear') and r['eligible']], key=lambda r: tuple(r['validation_rank']))['name']
    for mode in ['domain', 'label_domain']:
        for name in ['full32', 'invariant32']:
            for weight in [.5, .75]:
                tree_name = name+'_'+mode
                register(tree_name+'_'+best_linear+'_'+str(weight), [{**parts_by_name[tree_name][0], 'mixture_weight': weight},
                    {**parts_by_name[best_linear][0], 'mixture_weight': 1-weight}])
    chosen = max([r for r in rows if r['eligible']], key=lambda r: tuple(r['validation_rank']))
    o.write(OUTPUT/'selection.json', {'selected': chosen, 'all_candidates': rows, 'test_read_during_fitting_or_selection': False,
        'protocol_sha256': o.parent.digest(OUTPUT/'protocol.json'), 'fitted_records': int(fit.sum()), 'validation_records': int(val.sum()),
        'normalization_fitted_on_train_only': True, 'raw_reserved_in_fit_or_tuning': 0, 'source_ID_used_only_for_loss_weights_and_validation_groups': True})
    print('Selected domain-balanced head', chosen['name'], flush=True)


def evaluate():
    original = t.OUTPUT, t.checked
    try:
        t.OUTPUT, t.checked = OUTPUT, checked
        t.evaluate()
    finally:
        t.OUTPUT, t.checked = original


def evaluate_raw():
    import evaluate_raw as r
    original = r.OUTPUT, r.checked
    try:
        r.OUTPUT, r.checked = OUTPUT, checked
        r.main()
    finally:
        r.OUTPUT, r.checked = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze', 'train', 'evaluate', 'evaluate_raw'])
    globals()[parser.parse_args().stage]()
