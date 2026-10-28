"""One frozen selected-head evaluation on all 26 real-video feature bundles."""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from train_head import CODE, DATA, SOURCE, OUTPUT, REPORT, o, checked
from head import load


def main():
    p = checked()
    selection = o.parent.read(OUTPUT/'selection.json')
    chosen = selection['selected']
    if chosen is None or (OUTPUT/'raw-evaluation.json').exists():
        raise ValueError('No challenger or prior raw evaluation already frozen')
    root = DATA/'fullpose-raw-v1'
    manifest = o.parent.read(root/'feature-manifest.json')
    feature_plan = o.parent.read(root/'feature-plan.json')
    o.parent.check_hashes(feature_plan['implementation_sources'])
    if manifest['records'] != 26 or manifest['labels_read'] is not False:
        raise ValueError('Raw feature manifest is incomplete or contaminated')
    plan = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'selection_sha256': o.parent.digest(OUTPUT/'selection.json'),
        'checkpoint_sha256': chosen['checkpoint_sha256'], 'raw_feature_manifest_sha256': o.parent.digest(root/'feature-manifest.json'),
        'source_sha256': o.parent.digest(Path(__file__).resolve()), 'gate_source': 'v4 unchanged raw acceptance protocol',
        'gates': o.parent.read(DATA/'rgb-video-acceptance-v4/acceptance-protocol.json')['quality_gates'],
        'no_test_time_threshold_or_model_choice': True, 'independent_blind_test': False}
    o.write(OUTPUT/'raw-evaluation-plan.json', plan)
    model, state = load(OUTPUT/chosen['checkpoint'], p['feature_profile_hash'], o.parent.digest(OUTPUT/'protocol.json'))
    ids, features = [], []
    for row in manifest['rows']:
        path = root/row['feature_file']
        if o.parent.digest(path) != row['feature_sha256'] or row['labels_used'] is not False:
            raise ValueError('Raw feature changed or contains label-derived inputs')
        with np.load(path, allow_pickle=False) as data:
            features.append(data['features'][0].copy())
        ids.append(row['record_id'])
    if set(ids) != set(p['raw_test_reservations']) or len(ids) != 26:
        raise ValueError('Raw recording scope changed')
    x = np.stack(features)
    probability = model.probability(x); predictions = model.predict(x, probability)
    targets = {r['record_id']: r['original_labels'] for r in o.parent.read(DATA/'rgb-video-acceptance-v4/targets.json')}
    y = np.asarray([[label in targets[k] for label in o.parent.LABEL_IDS] for k in ids], dtype=bool)
    metrics = o.parent.metrics(predictions.astype(float), y)
    ranked = o.parent.metrics(probability[:, :11], y)
    metrics['mAP'] = ranked['mAP']
    for row, rank in zip(metrics['per_label'], ranked['per_label']):
        row['average_precision'] = rank['average_precision']
    g = plan['gates']
    checks = {'full_classification_coverage': len(x) == 26, 'single_macro_f1': metrics['macro_f1'] >= g['single_macro_f1'],
        'correct_false_alarm': metrics['correct_false_alarm_rate'] <= g['correct_false_alarm_rate'],
        'minimum_label_recall': min(r['recall'] for r in metrics['per_label']) >= g['minimum_label_recall']}
    result = {'evaluated_utc': datetime.now(timezone.utc).isoformat(), 'raw_metrics': metrics, 'quality_checks': checks,
        'raw_video_quality_passed': all(checks.values()), 'runtime_release_approved': False, 'records': 26,
        'body_cache_reused': 'Existing verified real-video v4 perception, not public ground-truth keypoints',
        'hand_features': 'New frozen AlphaPose68 readout from 16 original RGB frames per video; actor bound to cached body track',
        'no_new_manual_data_or_labels': True, 'no_threshold_changes_after_selection': True,
        'rows': [{'record_id': key, 'target': targets[key], 'predicted': [label for label, yes in zip(o.parent.LABEL_IDS, pred) if yes],
                  'exact_match': bool(np.array_equal(pred, target))} for key, pred, target in zip(ids, predictions, y)]}
    with (OUTPUT/'raw-predictions.npz').open('xb') as handle:
        np.savez_compressed(handle, features=x, scores=probability, predictions=predictions, targets=y, record_ids=np.asarray(ids))
    o.write(OUTPUT/'raw-evaluation.json', result)
    print('Raw 26', metrics['macro_f1'], 'exact', sum(r['exact_match'] for r in result['rows']), checks, flush=True)
    print([(r['id'], r['f1'], r['recall']) for r in metrics['per_label']], flush=True)


if __name__ == '__main__':
    main()
