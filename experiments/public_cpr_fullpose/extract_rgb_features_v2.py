"""Image-derived development features and separate label-free raw feature cache."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np

from rgb_features_v2 import CODE, MODEL, PROFILE, RGBRuntime, digest
from descriptor import sample_indices
from prepare_features import write_new

DATA = CODE/'data/public_cpr'
ROOT = DATA/'rgb-domain-features-v2'
SUBSET = DATA/'rgb-domain-training-v1/rgb-subset-v2'


def source_files():
    files = [Path(__file__).resolve(), Path(__file__).with_name('rgb_features_v2.py')]
    return {p.relative_to(CODE).as_posix(): digest(p) for p in files}


def development():
    if ROOT.exists():
        raise FileExistsError('Preserve feature extraction')
    source = json.loads((DATA/'rgb-domain-training-v1/source-integrity.json').read_text(encoding='utf-8'))
    if source['verified'] is not True:
        raise ValueError('Official RGB source part not verified')
    manifest = json.loads((SUBSET/'manifest.json').read_text(encoding='utf-8'))
    groups = json.loads((DATA/'rgb-evaluation-v4/record-splits.json').read_text(encoding='utf-8'))
    group_map = {r['record_id']: r for r in groups}
    write_new(ROOT/'profile.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'profile': PROFILE,
        'implementation_sources': source_files(), 'model_source_sha256': digest(MODEL/'source.json'),
        'source_subset_sha256': digest(SUBSET/'manifest.json'), 'source_integrity_sha256': digest(DATA/'rgb-domain-training-v1/source-integrity.json'),
        'all_backbones_frozen': True, 'no_test_groups_allowed': True})
    runtime = RGBRuntime()
    good, bad = [], []
    (ROOT/'records').mkdir()
    for i, row in enumerate(manifest['records'], 1):
        key = row['record_id']
        original = group_map[key]
        if original['application_split'] not in ['train', 'validation'] or original['application_split'] != row['split']:
            raise ValueError('RGB group leakage')
        if sorted(original['original_labels']) != sorted(row['original_labels']):
            raise ValueError('Annotation mismatch')
        started = time.perf_counter()
        try:
            images = []
            for image in row['images']:
                path = SUBSET/image['filename']
                if digest(path) != image['sha256']:
                    raise ValueError('Source JPEG changed')
                images.append(cv2.imread(str(path)))
            import hashlib
            reference_path = DATA/'fullpose-v2/skeletons'/(hashlib.sha256(key.encode()).hexdigest()+'.npz')
            with np.load(reference_path, allow_pickle=False) as source_pose:
                indices = np.asarray([im['frame_ordinal'] for im in row['images']])
                reference_p = source_pose['keypoint'][0, indices, :17].copy()
                reference_q = source_pose['keypoint_score'][0, indices, :17].copy()
            feature, audit = runtime.extract(images, reference_p, reference_q)
            path = ROOT/'records'/(key.replace('/', '--')+'.npz')
            np.savez_compressed(path, features=feature[None])
            good.append({**row, 'feature_file': path.relative_to(ROOT).as_posix(), 'feature_sha256': digest(path), 'audit': audit})
            print(f'[{i}/{len(manifest["records"])}] RGB development {key} {row["split"]} {time.perf_counter()-started:.1f}s', flush=True)
        except Exception as error:
            bad.append({'record_id': key, 'error': type(error).__name__+': '+str(error)})
            print('RGB development extraction failed', key, bad[-1]['error'], flush=True)
    if not good:
        raise ValueError('No usable RGB development records')
    features = np.concatenate([np.load(ROOT/r['feature_file'])['features'] for r in good])
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]
    targets = np.asarray([[label in r['original_labels'] for label in labels] for r in good], dtype=bool)
    np.savez_compressed(ROOT/'features.npz', features=features, targets=targets,
        splits=np.asarray([r['split'] for r in good]), record_ids=np.asarray([r['record_id'] for r in good]), label_ids=np.asarray(labels))
    write_new(ROOT/'manifest.json', {'records': good, 'perception_failures': bad, 'usable_records': len(good),
        'no_test_groups': True, 'no_new_manual_labels': True, 'feature_bundle_sha256': digest(ROOT/'features.npz')})
    print('RGB development features complete', len(good), 'failures', len(bad), flush=True)


def read_video_samples(path, expected):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    indices = set(sample_indices(expected).tolist())
    frames, i = [], 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if i in indices:
                frames.append(image)
            i += 1
    finally:
        cap.release()
    if i != expected or len(frames) != len(indices):
        raise ValueError('Incomplete raw video sampling')
    return frames


def raw():
    output = ROOT/'raw26'
    if output.exists():
        raise FileExistsError('Preserve raw RGB feature extraction')
    profile = json.loads((ROOT/'profile.json').read_text(encoding='utf-8'))
    for path, expected in profile['implementation_sources'].items():
        if digest(CODE/path) != expected:
            raise ValueError('Development and raw input implementations differ')
    jobs = json.loads((DATA/'rgb-video-acceptance-v4/jobs.json').read_text(encoding='utf-8'))
    write_new(output/'plan.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'profile_sha256': digest(ROOT/'profile.json'),
        'jobs_sha256': digest(DATA/'rgb-video-acceptance-v4/jobs.json'), 'labels_available_to_feature_extractor': False})
    runtime = RGBRuntime()
    rows, failures = [], []
    for i, job in enumerate(jobs, 1):
        path = CODE/job['video']
        if digest(path) != job['video_sha256']:
            raise ValueError('Raw video bytes changed')
        try:
            started = time.perf_counter()
            frames = read_video_samples(path, job['expected_frames'])
            body_folder = DATA/'rgb-video-acceptance-v4/runs'/job['key']
            body_result = json.loads((body_folder/'inference.json').read_text(encoding='utf-8'))
            if body_result['video_sha256'] != job['video_sha256'] or body_result['public_label_verification'] is not None:
                raise ValueError('M1 reference is not label-independent video perception')
            with np.load(body_folder/'inference.npz', allow_pickle=False) as source_pose:
                indices = sample_indices(job['expected_frames'])
                reference_p = source_pose['keypoint'][0, indices].copy()
                reference_q = source_pose['keypoint_score'][0, indices].copy()
            features, audit = runtime.extract(frames, reference_p, reference_q)
            target = output/(job['video_sha256']+'.npz')
            np.savez_compressed(target, features=features[None])
            rows.append({'record_id': job['record_id'], 'video_sha256': job['video_sha256'], 'feature_file': target.name,
                         'feature_sha256': digest(target), 'wall_seconds': time.perf_counter()-started, 'audit': audit})
            print(f'[{i}/26] new RGB representation {job["record_id"]}', flush=True)
        except Exception as error:
            failures.append({'record_id': job['record_id'], 'error': type(error).__name__+': '+str(error)})
            print('Raw RGB feature failed', job['record_id'], failures[-1]['error'], flush=True)
    write_new(output/'manifest.json', {'records': rows, 'failures': failures, 'classification_scope': 26,
        'labels_read': False, 'same_input_transform_as_development': True})
    print('Raw RGB cache complete', len(rows), 'failures', len(failures), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['development', 'raw'])
    globals()[parser.parse_args().stage]()
