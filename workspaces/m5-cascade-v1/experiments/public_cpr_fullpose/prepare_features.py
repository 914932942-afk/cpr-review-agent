"""Recover public Halpe68 points without changing annotations or recording split."""
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

import numpy as np

CODE = Path(__file__).resolve().parents[2]
DATA = CODE/'data/public_cpr'
OUTPUT = DATA/'fullpose-v2'
sys.path.insert(0, str(CODE/'experiments/public_cpr_baseline'))
from baseline import RestrictedUnpickler, canonical_path, digest
from descriptor import describe, PROFILE, INPUT_DIMENSION


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve fullpose extraction attempt')
    archive = DATA/'assets/Keypoints.tar.gz'
    if digest(archive) != 'daf93b63af7663ef07988e2dcdc04400029fc1a82ed1afe101ff93a6c0a62441':
        raise ValueError('Public keypoint archive changed')
    records = json.loads((DATA/'rgb-evaluation-v4/record-splits.json').read_text(encoding='utf-8'))
    wanted = {r['record_id']: r for r in records}
    source = {r['record_id']: r for r in json.loads((DATA/'run/records.json').read_text(encoding='utf-8'))}
    OUTPUT.mkdir()
    profile_hash = hashlib.sha256(json.dumps(PROFILE, sort_keys=True).encode()).hexdigest()
    write_new(OUTPUT/'feature-plan.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'profile': PROFILE, 'extra_profile_hash': profile_hash, 'input_dimension': INPUT_DIMENSION,
        'descriptor_sha256': digest(Path(__file__).with_name('descriptor.py')),
        'archive_sha256': digest(archive), 'original_split_sha256': digest(DATA/'rgb-evaluation-v4/record-splits.json'),
        'labels_and_splits_unchanged': True, 'no_model_fit_in_extraction': True})
    found, quality = {}, []
    (OUTPUT/'skeletons').mkdir()
    with tarfile.open(archive, mode='r|gz') as bundle:
        for member in bundle:
            if not member.isfile() or PurePosixPath(member.name).name not in ['train_keypoints.pkl', 'test_keypoints.pkl', 'all_errors_keypoints.pkl']:
                continue
            print('Recover public full68', member.name, flush=True)
            payload = RestrictedUnpickler(bundle.extractfile(member)).load()
            items = payload.get('annotations') if isinstance(payload, dict) else payload
            for item in items:
                key = canonical_path(item['frame_dir'])
                if key not in wanted:
                    continue
                if key in found:
                    raise ValueError('Duplicate record')
                p = np.asarray(item['keypoint'], dtype=np.float32)
                q = np.clip(np.asarray(item['keypoint_score'], dtype=np.float32), 0, 1)
                if p.shape != (1, source[key]['expected_frames'], 68, 2) or q.shape != p.shape[:3]:
                    raise ValueError('Unexpected full68 shape')
                with np.load(DATA/'run'/source[key]['record_file'], allow_pickle=False) as old:
                    np.testing.assert_array_equal(p[:, :, :17], old['keypoint'])
                    np.testing.assert_array_equal(q[:, :, :17], old['keypoint_score'])
                labels = np.asarray(item['label']).reshape(-1)
                if labels.size == 14 and np.isin(labels, [0, 1]).all():
                    labels = np.flatnonzero(labels)
                if sorted(map(int, labels)) != sorted(source[key]['original_labels']):
                    raise ValueError('Public labels do not match')
                extra, audit = describe(p[0], q[0])
                found[key] = extra
                quality.append({'record_id': key, **audit})
                filename = hashlib.sha256(key.encode()).hexdigest()+'.npz'
                np.savez_compressed(OUTPUT/'skeletons'/filename, keypoint=p, keypoint_score=q,
                                    img_shape=np.asarray(item['img_shape']), record_id=np.asarray(key))
                if len(found) % 100 == 0:
                    print('Full68 descriptors', len(found), '/1320', flush=True)
            del items, payload
            gc.collect()
    if set(found) != set(wanted):
        raise ValueError('Missing full68 public records')
    for name in ['development', 'locked-test']:
        with np.load(DATA/f'rgb-evaluation-v4/{name}.npz', allow_pickle=False) as old:
            result = {k: old[k].copy() for k in old.files}
        extra = np.stack([found[str(key)] for key in result['record_ids']])
        result['features'] = np.concatenate([result['features'], extra], 1)
        result['extra_profile_hash'] = np.asarray(profile_hash)
        np.savez_compressed(OUTPUT/(name+'.npz'), **result)
    write_new(OUTPUT/'extraction-audit.json', {'records': len(found), 'input_dimension': INPUT_DIMENSION,
        'record_counts': dict(Counter(r['application_split'] for r in records)), 'quality': quality,
        'development_sha256': digest(OUTPUT/'development.npz'), 'test_sha256': digest(OUTPUT/'locked-test.npz'),
        'no_new_labels': True, 'first17_verified_exactly': True})
    print('Full68 and motion descriptors complete; no training or evaluation performed', flush=True)


if __name__ == '__main__':
    main()
