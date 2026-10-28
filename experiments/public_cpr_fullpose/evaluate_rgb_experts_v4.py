"""Frozen RGB refinement acceptance; no changes to learned heads or thresholds."""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

import train_rgb_experts_v4 as d
from train_head import o
from head import count_fields


def main():
    p = d.checked()
    output, root, data = d.OUTPUT, d.FEATURES, d.DATA
    if (output/'raw-evaluation.json').exists():
        raise FileExistsError('Preserve first frozen RGB expert regression')
    selection = o.parent.read(output/'selection.json')
    if o.parent.digest(output/'experts.pt') != selection['checkpoint_sha256']:
        raise ValueError('Selected RGB heads changed')
    experts = torch.load(output/'experts.pt', map_location='cpu', weights_only=True)
    if experts['protocol_sha256'] != o.parent.digest(output/'protocol.json'):
        raise ValueError('RGB head/protocol binding mismatch')
    raw = o.parent.read(root/'raw26/manifest.json')
    profile = o.parent.read(root/'profile.json')
    o.parent.check_hashes(profile['implementation_sources'])
    o.write(output/'raw-evaluation-plan.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': o.parent.digest(Path(__file__).resolve()), 'checkpoint_sha256': selection['checkpoint_sha256'],
        'raw_manifest_sha256': o.parent.digest(root/'raw26/manifest.json'), 'fixed_gates': p['raw_gates'],
        'no_test_time_choices_or_training': True, 'scope': 'All original26 videos; failed feature extraction precludes quality acceptance'})
    jobs = o.parent.read(data/'rgb-video-acceptance-v4/jobs.json')
    targets = {r['record_id']: r['original_labels'] for r in o.parent.read(data/'rgb-video-acceptance-v4/targets.json')}
    cache = {r['record_id']: r for r in raw['records']}
    with np.load(data/'rgb-evaluation-v6/raw-predictions.npz', allow_pickle=False) as base:
        base_rows = {str(k): i for i, k in enumerate(base['record_ids'])}
        probabilities = base['scores'].copy(); old_predictions = base['predictions'].copy()
        base_ids = list(map(str, base['record_ids']))
    if set(cache) != set(base_ids) or raw['failures']:
        o.write(output/'raw-evaluation.json', {'raw_video_quality_passed': False, 'runtime_release_approved': False,
            'completed_records': len(cache), 'failed_records': len(raw['failures']), 'raw_metrics': None,
            'quality_checks': {'full_classification_coverage': False}, 'failures': raw['failures']})
        print('Raw RGB coverage failed', len(cache), raw['failures'], flush=True)
        return
    ids = [j['record_id'] for j in jobs]
    x = []
    for key in ids:
        row = cache[key]; path = root/'raw26'/row['feature_file']
        if o.parent.digest(path) != row['feature_sha256']:
            raise ValueError('Raw RGB features changed')
        with np.load(path, allow_pickle=False) as bundle:
            x.append(bundle['features'][0].copy())
    x = np.stack(x)
    base_score = np.stack([probabilities[base_rows[key]] for key in ids])
    predicted = np.stack([old_predictions[base_rows[key]] for key in ids])
    scores = base_score.copy()
    indices = list(o.parent.LABEL_IDS)
    probabilities_by_expert = d.predict_experts(x, experts)
    for label, (probability, decisions) in probabilities_by_expert.items():
        k = indices.index(label)
        predicted[:, k] = decisions & (base_score[:, 11] < experts['normal_veto_threshold'])
        scores[:, k] = probability
    y = np.asarray([[label in targets[key] for label in indices] for key in ids], dtype=bool)
    metrics = o.parent.metrics(predicted.astype(float), y)
    ranking = o.parent.metrics(scores[:, :11], y)
    metrics['mAP'] = ranking['mAP']
    for row, rank in zip(metrics['per_label'], ranking['per_label']):
        row['average_precision'] = rank['average_precision']
    g = p['raw_gates']
    checks = {'full_classification_coverage': len(ids) == 26, 'single_macro_f1': metrics['macro_f1'] >= g['single_macro_f1'],
        'minimum_label_recall': min(r['recall'] for r in metrics['per_label']) >= g['minimum_label_recall'],
        'normal_false_alarm': metrics['correct_false_alarm_rate'] <= g['correct_false_alarm_rate'],
        'head_budget': experts['combined_scalar_fields'] <= 70000}
    result = {'evaluated_utc': datetime.now(timezone.utc).isoformat(), 'raw_video_quality_passed': all(checks.values()),
        'quality_checks': checks, 'raw_metrics': metrics, 'records': 26, 'completed_records': 26, 'failed_records': 0,
        'exact_matches': int((predicted == y).all(1).sum()), 'normal_veto_threshold': experts['normal_veto_threshold'],
        'combined_scalar_fields': experts['combined_scalar_fields'], 'runtime_release_approved': False,
        'body_perception': 'Previously verified real-video v4 body track and v6 sparse-fullpose cache',
        'rgb_perception': 'New actual MP4-decoded contact/context images, actor bound to M1, frozen DINO; no public keypoints at inference',
        'test_used_for_fitting_or_thresholds': False, 'independent_blind_test': False,
        'rows': [{'record_id': key, 'original_labels': targets[key], 'predicted': [label for label, yes in zip(indices, pred) if yes],
                  'exact_match': bool(np.array_equal(pred, truth)), 'RGB_usable_frames': cache[key]['audit']['frames']} for key, pred, truth in zip(ids, predicted, y)]}
    with (output/'raw-predictions.npz').open('xb') as handle:
        np.savez_compressed(handle, rgb_features=x, scores=scores, predictions=predicted, targets=y, record_ids=np.asarray(ids))
    o.write(output/'raw-evaluation.json', result)
    print('Frozen RGB-assisted raw26', metrics['macro_f1'], 'exact', result['exact_matches'], 'normal', metrics['correct_false_alarm_rate'], checks, flush=True)
    print([(r['id'], r['f1'], r['recall']) for r in metrics['per_label']], flush=True)


if __name__ == '__main__':
    main()

