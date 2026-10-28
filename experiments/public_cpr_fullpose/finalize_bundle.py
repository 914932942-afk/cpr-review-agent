"""Add transitive feature helpers and runtime versions to the final cold-run lock."""
from datetime import datetime,timezone
import importlib.metadata
import sys
from head import CODE
from train_head import o


def main():
    root=CODE/'data/public_cpr/rgb-evaluation-v15';original=o.parent.read(root/'bundle-manifest.json')
    pinned=dict(original['pinned_sources_and_assets'])
    for name in ['extract_rgb_features_v3.py','prepare_features.py','train_head.py','train_rgb_experts_v3.py']:
        relative='experiments/public_cpr_fullpose/'+name;pinned[relative]=o.parent.digest(CODE/relative)
    relative='experiments/public_cpr_fullpose/finalize_bundle.py';pinned[relative]=o.parent.digest(CODE/relative)
    o.parent.check_hashes(pinned)
    versions={name:importlib.metadata.version(name) for name in ['numpy','torch','torchvision','opencv-python','transformers','scikit-learn','Pillow','timm','mmcv-lite','mmaction2']}
    result={**original,'bundle_revision':2,'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'preliminary_manifest_sha256':o.parent.digest(root/'bundle-manifest.json'),'pinned_sources_and_assets':pinned,
        'runtime':{'python':sys.version,'packages':versions},'launch_requirement':'Pass --manifest data/public_cpr/rgb-evaluation-v15/bundle-manifest-v2.json explicitly.'}
    o.write(root/'bundle-manifest-v2.json',result)
    print('Final bundle2 pinned',len(pinned),'sources/assets; complete head fields',result['active_scalar_fields_including_fallback'],flush=True)


if __name__=='__main__':main()
