"""Public non-test shape features and independent raw26 feature regression cache."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import cv2
import numpy as np
from hand_shape import CODE,PROFILE,extract
from baseline import digest
from descriptor import sample_indices
from prepare_features import write_new
from extract_rgb_features_v3 import read_video_samples

DATA=CODE/'data/public_cpr'; ROOT=DATA/'hand-shape-v1'


def development():
    if ROOT.exists():raise FileExistsError('Preserve hand-shape feature pass')
    manifest=json.loads((DATA/'rgb-domain-features-v5/manifest.json').read_text(encoding='utf-8'))
    paths=[Path(__file__).resolve(),Path(__file__).with_name('hand_shape.py')]
    write_new(ROOT/'profile.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'profile':PROFILE,
        'implementation_sources':{p.relative_to(CODE).as_posix():digest(p) for p in paths},
        'source_manifest_sha256':digest(DATA/'rgb-domain-features-v5/manifest.json'),'no_new_labels':True})
    features=[]; rows=[]
    for row in manifest['records']:
        key=row['record_id']; partial=row.get('is_partial_label_control',False)
        folder=DATA/'rgb-primary-controls-v1' if row.get('is_primary_control') else (DATA/'rgb-partial-controls-v1' if partial else DATA/'rgb-domain-prefix-v2/rgb-subset-v1')
        frames=[cv2.imread(str(folder/im['filename'])) for im in row['images']]
        if any(frame is None for frame in frames):raise ValueError('Public source image unavailable')
        path=DATA/'rgb-partial-controls-v1'/row['pose_file'] if partial else DATA/'fullpose-v2/skeletons'/(hashlib.sha256(key.encode()).hexdigest()+'.npz')
        with np.load(path,allow_pickle=False) as body:
            indices=np.asarray([im['frame_ordinal'] for im in row['images']]); p=body['keypoint'][0,indices,:17]; q=body['keypoint_score'][0,indices,:17]
        vector,audit=extract(frames,p,q);features.append(vector);rows.append({'record_id':key,**audit})
        print('Public hand shape',key,audit['usable_frames'],flush=True)
    np.savez_compressed(ROOT/'development.npz',features=np.stack(features),record_ids=np.asarray([r['record_id'] for r in rows]),available=np.asarray([r['available'] for r in rows]))
    write_new(ROOT/'development.json',{'rows':rows,'feature_sha256':digest(ROOT/'development.npz')})


def raw():
    profile=json.loads((ROOT/'profile.json').read_text(encoding='utf-8'))
    for path,expected in profile['implementation_sources'].items():
        if digest(CODE/path)!=expected:raise ValueError('Hand shape runtime changed')
    if (ROOT/'raw.npz').exists():raise FileExistsError('Preserve raw hand-shape extraction')
    jobs=json.loads((DATA/'rgb-video-acceptance-v4/jobs.json').read_text(encoding='utf-8'))
    write_new(ROOT/'raw-plan.json',{'jobs_sha256':digest(DATA/'rgb-video-acceptance-v4/jobs.json'),'labels_read':False,
        'profile_sha256':digest(ROOT/'profile.json'),'scope':'All26 actual MP4 inputs, no new landmarks or reference labels'})
    features=[];rows=[]
    for job in jobs:
        video=CODE/job['video']
        if digest(video)!=job['video_sha256']:raise ValueError('Raw video bytes changed')
        frames=read_video_samples(video,job['expected_frames'])
        root=DATA/'rgb-video-acceptance-v4/runs'/job['key']
        body_json=json.loads((root/'inference.json').read_text(encoding='utf-8'))
        if body_json['video_sha256']!=job['video_sha256'] or body_json['public_label_verification'] is not None:raise ValueError('Unverified M1 track')
        with np.load(root/'inference.npz',allow_pickle=False) as body:
            indices=sample_indices(job['expected_frames']); p=body['keypoint'][0,indices];q=body['keypoint_score'][0,indices]
        vector,audit=extract(frames,p,q);features.append(vector);rows.append({'record_id':job['record_id'],**audit})
        print('Raw hand shape',job['record_id'],audit['usable_frames'],flush=True)
    np.savez_compressed(ROOT/'raw.npz',features=np.stack(features),record_ids=np.asarray([r['record_id'] for r in rows]),available=np.asarray([r['available'] for r in rows]))
    write_new(ROOT/'raw.json',{'rows':rows,'feature_sha256':digest(ROOT/'raw.npz'),'labels_read':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['development','raw']);globals()[parser.parse_args().stage]()
