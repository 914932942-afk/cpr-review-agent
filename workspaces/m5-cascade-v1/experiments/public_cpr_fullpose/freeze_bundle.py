"""Freeze the entire runnable pipeline and count fallback coefficients honestly."""
from datetime import datetime,timezone
from pathlib import Path
import torch
from head import CODE,count_fields
from train_head import o

DATA=CODE/'data/public_cpr'; FINAL=DATA/'rgb-evaluation-v15'


def main():
    if (FINAL/'bundle-manifest.json').exists():raise FileExistsError('Preserve final runnable manifest')
    metrics=o.parent.read(FINAL/'raw-evaluation.json')
    if not metrics['raw_video_quality_passed']:raise ValueError('Raw quality gates have not passed')
    rgb=torch.load(DATA/'rgb-evaluation-v12/experts.pt',map_location='cpu',weights_only=True)
    hand=torch.load(DATA/'rgb-evaluation-v14/hands.pt',map_location='cpu',weights_only=True)
    card=torch.load(DATA/'rgb-evaluation-v13/cardinality.pt',map_location='cpu',weights_only=True)
    fields=rgb['combined_scalar_fields']+count_fields(hand['experts'])+count_fields(card)
    if fields>70000:raise ValueError('Complete pipeline including hand fallback exceeds budget')
    pinned={}
    manifests=[DATA/'rgb-evaluation-v4/inference-manifest.json',DATA/'rgb-evaluation-v6/protocol.json',DATA/'rgb-evaluation-v12/protocol.json',
        DATA/'rgb-evaluation-v13/protocol.json',DATA/'rgb-evaluation-v14/protocol.json',FINAL/'protocol.json',DATA/'rgb-domain-features-v5/profile.json']
    for path in manifests:
        value=o.parent.read(path);pinned[path.relative_to(CODE).as_posix()]=o.parent.digest(path)
        for key in ['implementation_sources','protected_sources','implementation_and_data','pinned_sources_and_assets']:
            for name,sha in value.get(key,{}).items():
                if name in pinned and pinned[name]!=sha:raise ValueError('Manifest chains disagree: '+name)
                pinned[name]=sha
        for item in value.get('pinned_files',[]):pinned[item['path']]=item['sha256']
    base_protocol=o.parent.read(DATA/'rgb-evaluation-v12/protocol.json')
    for key in ['base_head','mechanics_head']:pinned[base_protocol[key]]=o.parent.digest(CODE/base_protocol[key])
    for relative in ['data/public_cpr/rgb-evaluation-v12/experts.pt','data/public_cpr/rgb-evaluation-v14/hands.pt',
        'data/public_cpr/rgb-evaluation-v13/cardinality.pt','data/public_cpr/weights/dinov2-small/source.json',
        'data/public_cpr/weights/dinov2-small/config.json','data/public_cpr/weights/dinov2-small/preprocessor_config.json',
        'data/public_cpr/weights/dinov2-small/model.safetensors']:
        pinned[relative]=o.parent.digest(CODE/relative)
    for name in ['infer_candidate.py','freeze_bundle.py','raw_features.py','head.py','descriptor.py','rgb_features_v3.py',
        'mechanics.py','hand_shape.py','fuse_hand_shape.py','fuse_cardinality.py','reliability_fusion.py','train_mechanics_v3.py','train_rgb_experts_v5.py']:
        relative='experiments/public_cpr_fullpose/'+name;pinned[relative]=o.parent.digest(CODE/relative)
    o.parent.check_hashes(pinned)
    manifest={'schema':'cpr-offline-candidate-bundle-v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
        'entrypoint':'experiments/public_cpr_fullpose/infer_candidate.py','body_manifest':'data/public_cpr/rgb-evaluation-v4/inference-manifest.json',
        'pinned_sources_and_assets':pinned,'supported_label_ids':list(o.parent.LABEL_IDS),
        'active_scalar_fields_including_fallback':fields,'maximum_fields':70000,
        'budget_audit':'Includes both original RGB hand classifiers used as fallback. Earlier v14/v15 experimental counter excluded these inactive-on26 fallback classifiers; corrected complete count still passes70000.',
        'original_26_raw_gates_passed':True,'cold_video_validation_pending':True,'runtime_release_approved':False,
        'training':'Public original labels only; frozen backbones, small heads. No manual collection or annotation.',
        'scope':'Offline single-view ch0 engineering regression; no composite RGB or independent blind accuracy claim.'}
    o.write(FINAL/'bundle-manifest.json',manifest)
    print('Frozen runnable candidate; sources',len(pinned),'complete fields including fallback',fields,flush=True)


if __name__=='__main__':main()
