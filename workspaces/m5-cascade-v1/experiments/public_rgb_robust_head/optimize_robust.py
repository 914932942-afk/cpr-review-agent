"""One bounded development-only follow-up; test is a veto, not a selector."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from robust_head import CODE, SCHEMA, RobustHead, count_fields, load
sys.path.insert(0, str(CODE/'experiments/public_rgb_dense_head'))
import optimize_dense as d

o = d.o
BASE = d.OUTPUT
OUTPUT = o.DATA/'rgb-evaluation-v5'
REPORT = CODE/'test-results/public-rgb-optimization-v5'
SEED = 20261022


def prepare():
    if OUTPUT.exists():
        raise FileExistsError('Preserve existing experiment')
    parent = o.checked(BASE)
    selected = o.parent.read(BASE/'selection.json')['selected']
    protected = dict(parent['protected_sources'])
    for name in ['protocol.json', 'selection.json', 'evaluation.json', 'inference-manifest.json', selected['checkpoint']]:
        path = BASE/name
        protected[path.relative_to(CODE).as_posix()] = o.parent.digest(path)
    pointer = o.DATA/'current-offline-candidate.json'
    OUTPUT.mkdir(parents=True)
    for name in ['development.npz', 'locked-test.npz', 'record-splits.json']:
        shutil.copyfile(BASE/name, OUTPUT/name)
    source_files = [Path(__file__).resolve(), Path(__file__).with_name('robust_head.py'),
                    Path(__file__).with_name('infer_robust_video.py')]
    protocol = {**parent, 'purpose': 'Train-only standardized feature clipping and bounded mixture search',
        'frozen_at': datetime.now(timezone.utc).isoformat(),
        'parent_protocol': (BASE/'protocol.json').relative_to(CODE).as_posix(),
        'parent_protocol_sha256': o.parent.digest(BASE/'protocol.json'),
        'protected_sources': protected,
        'implementation_sources': {**parent['implementation_sources'], **{p.relative_to(CODE).as_posix(): o.parent.digest(p) for p in source_files}},
        'candidate_configs': [{'C': c, 'zscore_clip': clip, 'tree_weight': w} for c in [.03, .1]
                              for clip in [None, 3., 6.] for w in [.6, .75, .9]],
        'candidate_policy': '18 train-only balanced logistic heads/mixtures plus an exact v4 control; existing geometry tree reused. No backbone training or new labels.',
        'head_training': {'normalization': 'train mean/std only; constant scale set to one',
                          'linear': 'liblinear, balanced class weights, max_iter=2000',
                          'clipping': 'apply the identical declared zscore bound during fitting and inference'},
        'selection_rule': 'Validation gates unchanged, normal false alarms zero, worst-cohort F1 >= v4 minus .005. Among eligible challengers choose minimum cohort-balanced Brier, then larger worst-cohort F1, then fewer fields. Test a challenger only if its Brier is at least .5% below v4.',
        'promotion_rule': {
            'all_original_gates': True, 'single_composite_and_all_macro_f1': 'each >= v4 (tolerance 1e-12)',
            'normal_false_alarm_count': '<= v4', 'minimum_overall_label_recall': '>= v4',
            'reserved_26_f1_and_exact_matches': 'both >= v4',
            'material_gain': 'at least .002 absolute all-cohort or composite F1, OR .02 reserved F1, OR one fewer normal false alarm',
            'runtime_and_replay': 'new contracts, exact selected replay and existing C13 raw-video smoke must pass',
            'no_second_test_choice': 'one selected challenger only; failure leaves v4 current'},
        'candidate_pointer_before': o.parent.read(pointer), 'candidate_pointer_before_sha256': o.parent.digest(pointer),
        'previous_test_exposure': 'v1/v3/v4 engineering test metrics already inspected. Development fitting/threshold/selection do not decode test targets. Reused test is a deployment veto, not independent blind evidence.',
        'new_model_fitting_in_this_stage': True}
    o.write(OUTPUT/'protocol.json', protocol)
    print('Frozen 18 configurations, exact control and strict nonregression promotion rule', flush=True)


def state_for(parts, protocol):
    return {'schema': SCHEMA, 'components': parts, 'input_dimension': 783, 'feature_indices': torch.arange(783),
        'label_ids': list(o.parent.LABEL_IDS), 'source_revision': protocol['source_revision'],
        'feature_profile_hash': protocol['feature_profile_hash'], 'protocol_sha256': o.parent.digest(OUTPUT/'protocol.json'),
        'score_semantics': 'Uncalibrated scores with validation-selected thresholds; not medical confidence'}


def choose_challenger(rows, control):
    eligible = [r for r in rows if r['name'] != 'v4_control' and r['eligible']
                and r['validation_rank'][0] >= control['validation_rank'][0]-.005]
    winner = min(eligible, key=lambda r: (r['validation_brier'], -r['validation_rank'][0], r['scalar_fields'])) if eligible else None
    if winner is not None and winner['validation_brier'] <= control['validation_brier']*.995:
        return winner
    return None


def train():
    protocol = o.checked(OUTPUT)
    if (OUTPUT/'selection.json').exists() or list(OUTPUT.glob('*.pt')):
        raise FileExistsError('Preserve fitting attempt')
    torch.set_num_threads(4)
    with np.load(OUTPUT/'development.npz', allow_pickle=False) as f:
        x, y, splits, cohorts, ids = [f[k].copy() for k in ['features', 'targets', 'splits', 'cohorts', 'record_ids']]
    if {o.parent.group_id(str(k)) for k in ids} & {o.parent.group_id(k) for k in protocol['raw_test_reservations']}:
        raise ValueError('Reserved recording leaked into development')
    fit, val = splits == 'train', splits == 'validation'
    xf, xv = x[fit].astype(np.float64), x[val].astype(np.float64)
    yf, yv, cv = y[fit], y[val], cohorts[val]
    mean, scale = xf.mean(0), xf.std(0)
    scale[scale < 1e-6] = 1.
    target = np.column_stack([yf, ~yf.any(1)])
    old_selected = o.parent.read(BASE/'selection.json')['selected']
    old_state = torch.load(BASE/old_selected['checkpoint'], map_location='cpu', weights_only=True)
    tree = old_state['components'][0]
    rows = []

    def register(name, parts, config, fixed_op=None, numerical_difference=0.):
        state = state_for(parts, protocol)
        probability = RobustHead(state).probability(xv)
        if fixed_op is None:
            op = o.parent.fit_quality_head.tune_operating_point(probability, yv)
        else:
            op = fixed_op
        measured = o.summarize(probability, yv, cv, op['thresholds'], op['normal_gate'])
        single, composite = measured['single']['macro_f1'], measured['composite']['macro_f1']
        eligible = (single >= .85 and composite >= .8 and measured['all']['correct_false_alarm_rate'] == 0
                    and min(r['recall'] for r in measured['all']['per_label']) >= .6 and count_fields(parts) <= 70000)
        state['operating_point'] = op
        path = OUTPUT/(name+'.pt')
        torch.save(state, path)
        restored, _ = load(path, protocol['feature_profile_hash'], o.parent.digest(OUTPUT/'protocol.json'))
        if not np.array_equal(restored.probability(xv), probability):
            raise AssertionError('Export replay differs')
        row = {'name': name, 'config': config, 'checkpoint': path.name, 'checkpoint_sha256': o.parent.digest(path),
            'scalar_fields': count_fields(parts), 'eligible': bool(eligible), 'operating_point': op,
            'threshold_method': 'fixed_v4_control' if fixed_op is not None else 'legacy_validation',
            'validation_cohorts': measured, 'validation_rank': [min(single, composite), (single+composite)/2],
            'validation_brier': d.brier(probability, yv, cv), 'export_max_difference': numerical_difference}
        rows.append(row)
        print(name, 'validation F1', round(single, 6), round(composite, 6), 'Brier', round(row['validation_brier'], 8), flush=True)
        return row

    control_parts = [{**c} for c in old_state['components']]
    control_parts[1]['zscore_clip'] = None
    control = register('v4_control', control_parts, {'reuse': 'exact v4 weights and thresholds'}, old_state['operating_point'])
    if abs(control['validation_brier']-old_selected['validation_brier']) > 1e-14:
        raise AssertionError('Control differs from v4')
    models = {}
    for config in protocol['candidate_configs']:
        strength, bound, tree_weight = config['C'], config['zscore_clip'], config['tree_weight']
        key = (strength, bound)
        if key not in models:
            normalized = (xf-mean)/scale
            validation_normalized = (xv-mean)/scale
            if bound is not None:
                normalized = np.clip(normalized, -bound, bound)
                validation_normalized = np.clip(validation_normalized, -bound, bound)
            weights, biases, reference = [], [], []
            with threadpool_limits(limits=4):
                for k in range(12):
                    learner = LogisticRegression(C=strength, class_weight='balanced', solver='liblinear', max_iter=2000, random_state=SEED)
                    learner.fit(normalized, target[:, k])
                    if int(learner.n_iter_[0]) >= 2000:
                        raise RuntimeError('Linear solver did not converge')
                    weights.append(learner.coef_[0]); biases.append(learner.intercept_[0])
                    reference.append(learner.decision_function(validation_normalized))
            linear = {'kind': 'linear', 'mixture_weight': 1., 'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),
                      'weight1': torch.from_numpy(np.stack(weights)), 'bias1': torch.tensor(biases, dtype=torch.float64), 'zscore_clip': bound}
            portable = RobustHead(state_for([{**tree, 'mixture_weight': 0.}, linear], protocol)).probability(xv)
            reference = 1/(1+np.exp(-np.clip(np.stack(reference, 1), -60, 60)))
            diff = float(np.abs(portable-reference).max())
            if diff > 1e-12:
                raise AssertionError('Training/inference numerical mismatch')
            models[key] = (linear, diff)
        linear, diff = models[key]
        parts = [{**tree, 'mixture_weight': tree_weight}, {**linear, 'mixture_weight': 1-tree_weight}]
        name = f'linear{strength}_clip{bound}_tree{tree_weight}'
        register(name, parts, config, numerical_difference=diff)
    chosen = choose_challenger(rows, control)
    o.write(OUTPUT/'selection.json', {'selected': chosen, 'control': control, 'all_candidates': rows,
        'protocol_sha256': o.parent.digest(OUTPUT/'protocol.json'), 'test_read_during_fit_or_selection': False,
        'normalization_fitted_on_train_only': True, 'fitted_records': int(fit.sum()), 'tuning_records': int(val.sum()),
        'validation_challenger_found': chosen is not None, 'no_main_network_training': True})
    o.write(OUTPUT/'fit-access-audit.json', {'passed': True, 'npz_files_decoded_by_fitter': ['development.npz'],
        'test_npz_decoded_by_fitter': False, 'reserved_rgb_in_fit_or_tuning': 0,
        'locked_test_integrity_hashed_before_fit': True, 'record_counts': dict(Counter(splits.tolist()))})
    o.parent.check_hashes(protocol['protected_sources'])
    print('Frozen challenger:', None if chosen is None else chosen['name'], flush=True)


def evaluate():
    selection = o.parent.read(OUTPUT/'selection.json')
    if selection['selected'] is None:
        raise ValueError('No development improvement; do not open test')
    original = o.parent.load
    try:
        o.parent.load = load
        o.parent.evaluate(OUTPUT)
    finally:
        o.parent.load = original


def compare_metrics(old, new):
    a, b = old['evaluations'], new['evaluations']
    minimum = lambda v: min(r['recall'] for r in v['all']['per_label'])
    matches = lambda e: sum(r['exact_match'] for r in e['reserved_per_record'])
    old_fp = a['all']['correct_false_alarm_rate']*a['all']['original_correct_records']
    new_fp = b['all']['correct_false_alarm_rate']*b['all']['original_correct_records']
    tol = 1e-12
    checks = {'original_gates': all(new['quality_checks'].values()),
        **{name+'_macro_f1_no_regression': b[name]['macro_f1']+tol >= a[name]['macro_f1'] for name in ['all', 'single', 'composite']},
        'normal_false_alarms_no_regression': new_fp <= old_fp+tol,
        'minimum_label_recall_no_regression': minimum(b)+tol >= minimum(a),
        'reserved_macro_f1_no_regression': b['rgb_reserved_public_keypoints']['macro_f1']+tol >= a['rgb_reserved_public_keypoints']['macro_f1'],
        'reserved_exact_match_no_regression': matches(new) >= matches(old),
        'material_gain': (b['all']['macro_f1']-a['all']['macro_f1'] >= .002
                         or b['composite']['macro_f1']-a['composite']['macro_f1'] >= .002
                         or b['rgb_reserved_public_keypoints']['macro_f1']-a['rgb_reserved_public_keypoints']['macro_f1'] >= .02
                         or new_fp <= old_fp-1+tol)}
    return checks


def compare():
    p = o.checked(OUTPUT)
    old, new = [o.parent.read(folder/'evaluation.json') for folder in [BASE, OUTPUT]]
    checks = compare_metrics(old, new)
    o.write(OUTPUT/'promotion-comparison.json', {'metric_checks': checks, 'metric_promotion_eligible': all(checks.values()),
        'contracts_and_raw_smoke_still_required': True, 'candidate_pointer_modified': False,
        'protocol_sha256': o.parent.digest(OUTPUT/'protocol.json'),
        'same_exposed_engineering_test': True, 'promotion_policy_frozen_before_fitting': bool(p['promotion_rule'])})
    print(checks, flush=True)


def export():
    p = o.checked(OUTPUT)
    if not o.parent.read(OUTPUT/'promotion-comparison.json')['metric_promotion_eligible']:
        raise ValueError('Failed nonregression; candidate not eligible for adoption')
    chosen = o.parent.read(OUTPUT/'selection.json')['selected']
    old = o.parent.read(BASE/'inference-manifest.json')
    exclude = {old['head'], old['protocol'], 'data/public_cpr/rgb-evaluation-v4/selection.json', old['entrypoint'],
               'experiments/public_rgb_dense_head/compact_head.py'}
    pinned = [r for r in old['pinned_files'] if r['path'] not in exclude]
    sources = [OUTPUT/chosen['checkpoint'], OUTPUT/'protocol.json', OUTPUT/'selection.json', OUTPUT/'record-splits.json',
               Path(__file__).resolve(), Path(__file__).with_name('robust_head.py'), Path(__file__).with_name('infer_robust_video.py'),
               CODE/'experiments/public_rgb_dense_head/compact_head.py']
    unique = {r['path']: r for r in pinned}
    for path in sources:
        key = path.relative_to(CODE).as_posix()
        unique[key] = {'path': key, 'sha256': o.parent.digest(path)}
    manifest = {**old, 'frozen_at': datetime.now(timezone.utc).isoformat(), 'head': (OUTPUT/chosen['checkpoint']).relative_to(CODE).as_posix(),
        'protocol': (OUTPUT/'protocol.json').relative_to(CODE).as_posix(), 'pinned_files': list(unique.values()),
        'head_kind': 'robust_compact', 'entrypoint': 'experiments/public_rgb_robust_head/infer_robust_video.py',
        'disclosure': p['previous_test_exposure'], 'runtime_release_approved': False, 'raw_video_acceptance_pending': True}
    o.write(OUTPUT/'inference-manifest.json', manifest)
    o.parent.load_manifest(OUTPUT/'inference-manifest.json')
    print('Isolated manifest exported; current pointer unchanged', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'compare', 'export'])
    globals()[parser.parse_args().stage]()
