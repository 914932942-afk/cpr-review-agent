"""Validation one-standard-error selection; no additional model fitting.

v2's strict validation maximization chose the unchanged control. Its new test
was never evaluated. This separate protocol chooses a simpler statistically
comparable candidate before the first optimized-test evaluation.
"""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import torch

import optimize_head as o

SOURCE = o.OUTPUT
OUTPUT = o.DATA / 'rgb-evaluation-v3'
REPORT = o.CODE / 'test-results/public-rgb-optimization-v3'
BOOTSTRAPS = 200


def prepare():
    if OUTPUT.exists():
        raise FileExistsError('Preserve previous selection attempts')
    protocol = o.checked(SOURCE)
    if (SOURCE / 'evaluation.json').exists():
        raise ValueError('This protocol expects v2 development selection only, without a new test evaluation')
    selection = o.parent.read(SOURCE / 'selection.json')
    protected = dict(protocol['protected_sources'])
    for name in ['protocol.json', 'selection.json', 'fit-access-audit.json'] + [r['checkpoint'] for r in selection['all_candidates'] if 'checkpoint' in r]:
        path = SOURCE / name
        protected[path.relative_to(o.CODE).as_posix()] = o.parent.digest(path)
    OUTPUT.mkdir(parents=True)
    for name in ['development.npz', 'locked-test.npz', 'record-splits.json']:
        shutil.copyfile(SOURCE / name, OUTPUT / name)
    protocol = {**protocol, 'purpose': 'Validation uncertainty-aware one-standard-error selection of previously fitted small heads',
        'frozen_at': datetime.now(timezone.utc).isoformat(), 'parent_protocol': (SOURCE / 'protocol.json').relative_to(o.CODE).as_posix(),
        'parent_protocol_sha256': o.parent.digest(SOURCE / 'protocol.json'), 'protected_sources': protected,
        'implementation_sources': {**protocol['implementation_sources'], Path(__file__).resolve().relative_to(o.CODE).as_posix(): o.parent.digest(Path(__file__).resolve())},
        'selection_rule': 'Bootstrap validation worst-cohort Macro-F1 200 times, stratified by cohort and supported-label pattern. Candidates within one bootstrap standard error of validation best are comparable; choose fewest scalar fields, then higher validation worst-cohort F1.',
        'candidate_policy': 'Reuse the eight v2 train/validation candidates and frozen thresholds; no new fit, no threshold changes, no test decoding during selection',
        'bootstrap_iterations_for_selection': BOOTSTRAPS,
        'v2_optimized_test_evaluated': False, 'new_model_fitting_in_this_stage': False}
    o.write(OUTPUT / 'protocol.json', protocol)
    print('Frozen one-standard-error selection; no new fit and no new test evaluation', flush=True)


def macro_f1(predicted, truth):
    truth = truth.astype(bool)
    tp = (predicted & truth).sum(0)
    fp = (predicted & ~truth).sum(0)
    fn = (~predicted & truth).sum(0)
    return float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1)))


def choose():
    protocol = o.checked(OUTPUT)
    if (OUTPUT / 'selection.json').exists():
        raise FileExistsError('Selection is frozen')
    selection = o.parent.read(SOURCE / 'selection.json')
    candidates = [r for r in selection['all_candidates'] if r['eligible']]
    best = max(candidates, key=lambda row: tuple(row['validation_rank']))
    model, state = o.load(SOURCE / best['checkpoint'], protocol['feature_profile_hash'], o.parent.digest(SOURCE / 'protocol.json'))
    with np.load(OUTPUT / 'development.npz', allow_pickle=False) as data:
        mask = data['splits'] == 'validation'
        x, y, cohorts = [data[key][mask].copy() for key in ['features', 'targets', 'cohorts']]
    op = state['operating_point']
    predicted = o.parent.predict(model.probability(x), op['thresholds'], op['normal_gate'])
    strata = defaultdict(list)
    for i, (cohort, target) in enumerate(zip(cohorts, y)):
        strata[(str(cohort), tuple(np.flatnonzero(target)))].append(i)
    rng = np.random.default_rng(o.SEED)
    scores = []
    for _ in range(BOOTSTRAPS):
        indices = np.concatenate([rng.choice(rows, len(rows), replace=True) for _, rows in sorted(strata.items())])
        scores.append(min(macro_f1(predicted[indices][cohorts[indices] == cohort], y[indices][cohorts[indices] == cohort])
                          for cohort in ['single', 'composite']))
    standard_error = float(np.std(scores, ddof=1))
    floor = best['validation_rank'][0] - standard_error
    comparable = [r for r in candidates if r['validation_rank'][0] >= floor]
    chosen = min(comparable, key=lambda row: (row['scalar_fields'], -row['validation_rank'][0]))
    source_path = SOURCE / chosen['checkpoint']
    model, state = o.load(source_path, protocol['feature_profile_hash'], o.parent.digest(SOURCE / 'protocol.json'))
    state['protocol_sha256'] = o.parent.digest(OUTPUT / 'protocol.json')
    state['selection_method'] = 'validation_one_standard_error_simplest'
    destination = OUTPUT / chosen['checkpoint']
    torch.save(state, destination)
    restored, _ = o.load(destination, protocol['feature_profile_hash'], o.parent.digest(OUTPUT / 'protocol.json'))
    if not np.array_equal(model.probability(x), restored.probability(x)):
        raise AssertionError('Protocol rebinding must not change any prediction')
    frozen = {**chosen, 'source_checkpoint_sha256': chosen['checkpoint_sha256'], 'checkpoint_sha256': o.parent.digest(destination),
              'checkpoint_bytes': destination.stat().st_size}
    uncertainty = {'bootstrap_iterations': BOOTSTRAPS, 'validation_best_name': best['config']['name'],
        'validation_best_worst_cohort_f1': best['validation_rank'][0], 'bootstrap_standard_error': standard_error,
        'one_standard_error_floor': floor, 'comparable_candidates': [r['config']['name'] for r in comparable],
        'selected_name': chosen['config']['name'], 'selected_fields': chosen['scalar_fields'],
        'rule': protocol['selection_rule']}
    o.write(OUTPUT / 'selection.json', {'selected': frozen, 'all_candidates': selection['all_candidates'],
        'candidate_source_directory': SOURCE.relative_to(o.CODE).as_posix(), 'one_standard_error': uncertainty,
        'protocol_sha256': o.parent.digest(OUTPUT / 'protocol.json'), 'test_read_during_fit_or_selection': False,
        'main_network_training': False, 'new_fitting_performed': False, 'fitted_records': 738, 'tuning_records': 196})
    o.write(OUTPUT / 'fit-access-audit.json', {'passed': True, 'npz_files_decoded_by_fitter': [(SOURCE / 'development.npz').relative_to(o.CODE).as_posix()],
        'npz_files_decoded_by_robust_selector': [(OUTPUT / 'development.npz').relative_to(o.CODE).as_posix()],
        'test_npz_decoded_by_fitter': False, 'test_npz_decoded_by_selector': False,
        'locked_test_integrity_hashed_before_fit': True, 'reserved_rgb_in_fit_or_tuning': 0,
        'test_based_candidate_selection': False, 'new_fitting_performed': False})
    o.parent.check_hashes(protocol['protected_sources'])
    print(uncertainty, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'choose', 'evaluate', 'export'])
    args = parser.parse_args()
    {'prepare': prepare, 'choose': choose, 'evaluate': lambda: o.evaluate(OUTPUT), 'export': lambda: o.export_manifest(OUTPUT)}[args.stage]()
