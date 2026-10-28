"""Cold MP4 -> frozen perception -> small heads -> reliability-guarded results.

This offline CLI has no reference-record/target-label argument and cannot release
the application. All learned weights and interpretation sources are manifest bound.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import torch

from head import CODE,load
from train_head import o
from baseline import digest
import infer_video
from compact_head import load as load_compact_body
from raw_features import Runtime as FullRuntime
from rgb_features_v3 import RGBRuntime
from extract_rgb_features_v3 import read_video_samples
from descriptor import sample_indices
from mechanics import describe as mechanics_features
from train_mechanics_v3 import predict as predict_mechanics
from train_rgb_experts_v5 import predict_experts
from hand_shape import extract as shape_features
from fuse_hand_shape import predict as predict_hands,HAND_INDICES
from fuse_cardinality import counts
from reliability_fusion import guarded

DATA=CODE/'data/public_cpr'; FINAL=DATA/'rgb-evaluation-v15'


def analyze_body(video,output,manifest,reference):
    # Same isolated loader adapter as the frozen v4 26-video inference command.
    original=infer_video.load
    try:
        infer_video.load=load_compact_body
        return infer_video.analyze(video,output,manifest,reference)
    finally:
        infer_video.load=original


def validate_bundle(path):
    manifest=o.parent.read(path)
    if manifest['schema']!='cpr-offline-candidate-bundle-v1':raise ValueError('Unsupported candidate bundle')
    o.parent.check_hashes(manifest['pinned_sources_and_assets'])
    if manifest['active_scalar_fields_including_fallback']>70000:raise ValueError('Candidate exceeds small-head budget')
    return manifest


def analyze(video,output,manifest_path):
    if not video.is_file():raise ValueError('Input video not found')
    if output.exists() or output.with_suffix('.npz').exists() or output.with_name(output.stem+'-body.json').exists():
        raise FileExistsError('Preserve previous cold inference')
    manifest=validate_bundle(manifest_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    body_json=output.with_name(output.stem+'-body.json')
    started=time.perf_counter()
    # No labels or record identifiers are provided to body/hand/RGB perception.
    body=analyze_body(video,body_json,CODE/manifest['body_manifest'],None)
    full_runtime=FullRuntime()
    full,full_arrays,full_audit=full_runtime.extract(video,body_json,body_json.with_suffix('.npz'))
    del full_runtime;torch.cuda.empty_cache()
    source=o.parent.read(DATA/'rgb-evaluation-v12/protocol.json')
    base_protocol=o.parent.read(CODE/source['base_protocol'])
    base,_=load(CODE/source['base_head'],base_protocol['feature_profile_hash'],digest(CODE/source['base_protocol']))
    base_probability=base.probability(full[None]);scores=base_probability.copy();eligible=base.predict(full[None],base_probability)
    with np.load(body_json.with_suffix('.npz'),allow_pickle=False) as f:
        points=f['keypoint'][0].copy();confidence=f['keypoint_score'][0].copy()
    motion=mechanics_features(points,confidence)
    mechanical=torch.load(CODE/source['mechanics_head'],map_location='cpu',weights_only=True)
    for label,(probability,prediction) in predict_mechanics(motion[None],mechanical).items():
        col=list(o.parent.LABEL_IDS).index(label);scores[:,col]=probability;eligible[:,col]=prediction
    indices=sample_indices(len(points));frames=read_video_samples(video,len(points))
    rgb_runtime=RGBRuntime();rgb,rgb_audit=rgb_runtime.extract(frames,points[indices],confidence[indices])
    del rgb_runtime;torch.cuda.empty_cache()
    rgb_state=torch.load(DATA/'rgb-evaluation-v12/experts.pt',map_location='cpu',weights_only=True)
    thresholds=np.asarray(base.state['operating_point']['thresholds'],float)[None].copy()
    for e in mechanical['experts']:thresholds[:,list(o.parent.LABEL_IDS).index(e['label_id'])]=e['threshold']
    for label,(probability,prediction) in predict_experts(rgb[None],rgb_state).items():
        col=list(o.parent.LABEL_IDS).index(label);scores[:,col]=probability
        eligible[:,col]=prediction & (base_probability[:,11]<rgb_state['normal_veto_threshold'])
        thresholds[:,col]=next(e['threshold'] for e in rgb_state['experts'] if e['label_id']==label)
    shape,shape_audit=shape_features(frames,points[indices],confidence[indices])
    hand=torch.load(DATA/'rgb-evaluation-v14/hands.pt',map_location='cpu',weights_only=True)
    if shape_audit['available']:
        hand_input=np.r_[shape,rgb[HAND_INDICES]][None]
        for label,(probability,prediction) in predict_hands(hand_input,hand).items():
            col=list(o.parent.LABEL_IDS).index(label);scores[:,col]=probability;eligible[:,col]=prediction
            thresholds[:,col]=next(e['threshold'] for e in hand['experts'] if e['label_id']==label)
    cardinality=torch.load(DATA/'rgb-evaluation-v13/cardinality.pt',map_location='cpu',weights_only=True)
    number=counts(np.r_[full[:783],motion][None],cardinality)
    reliability=o.parent.read(FINAL/'protocol.json')['reliability']
    qualified=np.asarray([reliability[str(k)]['qualified'] for k in o.parent.LABEL_IDS])
    pred=guarded(eligible,scores[:,:11],thresholds,number,qualified)
    label_ids=[label for label,yes in zip(o.parent.LABEL_IDS,pred[0]) if yes]
    result={'schema':'cpr-offline-analysis-v2','status':'completed_experimental','video_sha256':digest(video),
        'bundle_sha256':digest(manifest_path),'decoded_frames':body['decoded_frames'],'reported_fps':body['reported_fps'],
        'predicted_original_label_ids':label_ids,'supported_label_ids':list(o.parent.LABEL_IDS),
        'estimated_supported_error_count':int(number[0]),'reported_error_count':len(label_ids),
        'reliability_sources':reliability,'hand_shape_audit':shape_audit,'rgb_audit':rgb_audit,'fullpose_audit':full_audit,
        'scores':[{'original_label_id':label,'uncalibrated_score':float(score),'threshold':float(threshold)}
            for label,score,threshold in zip(o.parent.LABEL_IDS,scores[0,:11],thresholds[0])],
        'analysis_seconds':time.perf_counter()-started,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'public_label_verification':None,'model_or_threshold_modified':False,'runtime_release_approved':False,
        'empty_error_list_semantics':'No supported error reported; not proof of correct CPR.',
        'scope':'Single-view ch0 public-video engineering profile. No clinical or cross-person guarantee.'}
    with output.with_suffix('.npz').open('xb') as f:
        np.savez_compressed(f,fullpose_features=full[None],rgb_features=rgb[None],shape_features=shape[None],
            mechanics_features=motion[None],scores=scores,eligible=eligible,thresholds=thresholds,predicted_count=number,predictions=pred)
    o.write(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--video',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--manifest',type=Path,default=FINAL/'bundle-manifest-v3.json');args=parser.parse_args()
    result=analyze(args.video,args.output,args.manifest)
    print(json.dumps({'status':result['status'],'labels':result['predicted_original_label_ids'],'seconds':result['analysis_seconds']},ensure_ascii=False),flush=True)
