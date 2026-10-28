"""Strictly versioned raw-video acceptance, independent of the application."""
from pathlib import Path
import hashlib
import json

CODE = Path(__file__).resolve().parents[2]
DATA = CODE/'data/public_cpr'
MODEL = DATA/'rgb-evaluation-v4'
COLLECTION = DATA/'rgb-acceptance-v1'
OUTPUT = DATA/'rgb-video-acceptance-v4'
REPORT = CODE/'test-results/public-rgb-video-acceptance-v4'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def within_code(relative):
    path = (CODE/relative).resolve()
    if not path.is_relative_to(CODE.resolve()) or not path.is_file():
        raise ValueError('Invalid or missing CODE-relative file: '+str(relative))
    return path


def verify_lock(include_videos=False):
    protocol = read(OUTPUT/'acceptance-protocol.json')
    lock = read(OUTPUT/'model-lock.json')
    if digest(OUTPUT/'model-lock.json') != protocol['model_lock_sha256']:
        raise ValueError('Model lock changed')
    for source, expected in protocol['implementation_sources'].items():
        if digest(within_code(source)) != expected:
            raise ValueError('Acceptance implementation changed: '+source)
    for source, expected in lock['protected_files'].items():
        if digest(within_code(source)) != expected:
            raise ValueError('Frozen pipeline or source changed: '+source)
    if digest(OUTPUT/'jobs.json') != protocol['jobs_sha256'] or digest(OUTPUT/'targets.json') != protocol['targets_sha256']:
        raise ValueError('Frozen jobs/targets changed')
    if include_videos:
        for job in read(OUTPUT/'jobs.json'):
            if digest(within_code(job['video'])) != job['video_sha256']:
                raise ValueError('Video changed: '+job['record_id'])
    return protocol, lock


def validate_result(job, result, manifest_hash):
    if (result.get('status') != 'completed_experimental' or result.get('video_sha256') != job['video_sha256']
        or result.get('manifest_sha256') != manifest_hash or result.get('decoded_frames') != job['expected_frames']
        or result.get('model_or_threshold_modified') is not False or result.get('runtime_release_approved') is not False
        or result.get('public_label_verification') is not None or result.get('repeat_feature_max_difference', 1.) > 1e-5):
        raise ValueError('Raw analysis output violates the frozen contract')
    labels = result.get('predicted_original_label_ids', [])
    supported = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]
    if labels != sorted(set(labels)) or any(x not in supported for x in labels):
        raise ValueError('Invalid predicted labels')
    if [r['original_label_id'] for r in result.get('scores', [])] != supported:
        raise ValueError('Missing or reordered output scores')
