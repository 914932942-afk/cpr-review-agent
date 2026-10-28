"""Freeze the passing v4 pipeline and all 26 public recordings before inference."""
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys

from common import CODE, DATA, MODEL, COLLECTION, OUTPUT, REPORT, read, digest, write_new


def main():
    if OUTPUT.exists():
        raise FileExistsError('Do not overwrite a frozen acceptance run')
    sys.path.insert(0, str(CODE/'experiments/public_rgb_dense_head'))
    import optimize_dense as d
    from compact_head import load
    protocol = d.o.checked(MODEL)
    pointer = read(DATA/'current-offline-candidate.json')
    if pointer['manifest'] != (MODEL/'inference-manifest.json').relative_to(CODE).as_posix():
        raise ValueError('Expected current v4 candidate')
    manifest = d.o.parent.load_manifest(MODEL/'inference-manifest.json')
    selected = read(MODEL/'selection.json')['selected']
    _, state = load(CODE/manifest['head'], protocol['feature_profile_hash'], digest(MODEL/'protocol.json'))
    collection = read(COLLECTION/'collection-manifest.json')
    records = {r['record_id']: r for r in read(DATA/'run/records.json')}
    split = {r['record_id']: r['application_split'] for r in read(MODEL/'record-splits.json')}
    from baseline import read_annotations
    annotations, _ = read_annotations(DATA/'assets/ann.tar.gz', 'ch0')
    protected = {**protocol['protected_sources'], **protocol['implementation_sources'],
                 **{r['path']: r['sha256'] for r in manifest['pinned_files']}}
    paths = [MODEL/'protocol.json', MODEL/'selection.json', MODEL/'evaluation.json', MODEL/'inference-manifest.json',
             COLLECTION/'collection-manifest.json', DATA/'run/records.json', DATA/'assets/ann.tar.gz']
    for path in paths:
        protected[path.relative_to(CODE).as_posix()] = digest(path)
    app = read(CODE/'test-results/public-head-validation/application-source-before.json')
    for name, expected in app.items():
        if digest(CODE/name) != expected:
            raise ValueError('Application differs from known source: '+name)
        protected[name] = expected
    jobs, targets = [], []
    for item in collection['records']:
        key = item['record_id']
        source = records[key]
        if source['original_labels'] != item['original_labels'] or annotations[key]['original_labels'] != item['original_labels']:
            raise ValueError('Public annotation mismatch: '+key)
        if split[key] != 'test' or source['expected_frames'] != item['expected_frames']:
            raise ValueError('Training leakage or frame mismatch: '+key)
        video = COLLECTION/item['video']
        if digest(video) != item['video_sha256']:
            raise ValueError('Video hash mismatch: '+key)
        source_manifest = COLLECTION/item['source_manifest']
        if digest(source_manifest) != item['source_manifest_sha256']:
            raise ValueError('Source media manifest changed: '+key)
        protected[source_manifest.relative_to(CODE).as_posix()] = item['source_manifest_sha256']
        jobs.append({'record_id': key, 'key': key.replace('/', '--'), 'video': video.relative_to(CODE).as_posix(),
                     'video_sha256': item['video_sha256'], 'expected_frames': item['expected_frames']})
        targets.append({'record_id': key, 'original_labels': item['original_labels'], 'label_names': item['label_names']})
    if len(jobs) != 26 or len({j['record_id'] for j in jobs}) != 26 or {j['record_id'] for j in jobs} != set(protocol['raw_test_reservations']):
        raise ValueError('Expected all 26 reserved recordings, with no substitutions')
    if sum(j['expected_frames'] for j in jobs) != 11215:
        raise ValueError('Expected 11215 original frames')
    manifest_hash = digest(MODEL/'inference-manifest.json')
    lock = {'schema': 'cpr-v4-model-lock-v1', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'manifest': pointer['manifest'], 'manifest_sha256': manifest_hash, 'head': manifest['head'],
        'head_sha256': digest(CODE/manifest['head']), 'operating_point': state['operating_point'],
        'input_profile': manifest['input_profile'], 'protected_files': protected,
        'pointer_before': pointer, 'pointer_before_sha256': digest(DATA/'current-offline-candidate.json'),
        'release_approved': False}
    write_new(OUTPUT/'model-lock.json', lock)
    write_new(OUTPUT/'jobs.json', jobs)
    write_new(OUTPUT/'targets.json', targets)
    source_files = [Path(__file__).resolve(), Path(__file__).with_name('common.py'), Path(__file__).with_name('run_batch.py')]
    gates = protocol['gates']
    acceptance = {'schema': 'cpr-raw-video-acceptance-v1', 'frozen_utc': lock['frozen_utc'],
        'model_lock_sha256': digest(OUTPUT/'model-lock.json'), 'jobs_sha256': digest(OUTPUT/'jobs.json'),
        'targets_sha256': digest(OUTPUT/'targets.json'), 'record_count': 26, 'frame_count': 11215,
        'implementation_sources': {p.relative_to(CODE).as_posix(): digest(p) for p in source_files},
        'entrypoint': pointer['entrypoint'], 'manifest': pointer['manifest'],
        'execution_gates': {'completed_records': 26, 'expected_total_frames': 11215,
                            'hash_and_frame_integrity': True, 'frozen_feature_replay_tolerance': 1e-5},
        'quality_gates': {'single_macro_f1': gates['single_macro_f1'],
                          'correct_false_alarm_rate': gates['correct_false_alarm_rate'],
                          'minimum_label_recall': gates['minimum_label_recall'], 'classification_coverage': gates['classification_coverage']},
        'threshold_policy': 'Use the exact v4 checkpoint thresholds and preprocessing; no training, retuning, rejection-based sample removal or test-guided fallback',
        'timeout_seconds_per_record': 900, 'inference_attempts_per_record': 1,
        'process_policy': 'Serial GPU jobs, isolated child per video. No reference-record argument; targets used only by post-run aggregation.',
        'metrics_policy': 'All 26 attempted. Full raw F1 is reported only if all succeed; failures remain in coverage and preclude acceptance. Correct false alarms use the two original Correct clips.',
        'composite_raw_video_evaluation': 'not available in this frozen collection; not claimed passed',
        'independent_unseen_test_claim': False,
        'disclosure': 'All 26 are test-only for v4. Earlier engineering models and four earlier RGB tests used/exposed some recordings. No cross-person or clinical claim.',
        'on_failure': 'Keep the v4 artifacts frozen; report failed quality or execution without changing scope, labels, model or thresholds',
        'runtime_release_approved': False}
    write_new(OUTPUT/'acceptance-protocol.json', acceptance)
    packages = {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'numpy', 'ultralytics', 'mmaction2', 'mmcv-lite', 'scikit-learn']}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'], capture_output=True, text=True, check=True)
    write_new(OUTPUT/'environment.json', {'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'packages': packages, 'gpu': gpu.stdout.strip(),
        'timing': 'per-record wall time includes process/model setup; analysis_seconds excludes model initialization; GPU peak is PyTorch allocated bytes only'})
    write_new(OUTPUT/'candidate-pointer-before.json', pointer)
    REPORT.mkdir(parents=True, exist_ok=True)
    print('Frozen exact v4, 26 held-out videos and 11215 frames; no model/threshold changes', flush=True)


if __name__ == '__main__':
    main()
