"""Versioned loader-only correction; frozen models and decision rules unchanged."""
from pathlib import Path
from datetime import datetime,timezone
from head import CODE
from train_head import o


def main():
    root=CODE/'data/public_cpr/rgb-evaluation-v15';source=root/'bundle-manifest-v2.json'
    original=o.parent.read(source);pinned=dict(original['pinned_sources_and_assets'])
    for name in ['infer_candidate_v2.py','repair_bundle_entrypoint.py']:
        path=Path(__file__).with_name(name);pinned[path.relative_to(CODE).as_posix()]=o.parent.digest(path)
    o.parent.check_hashes(pinned)
    manifest={**original,'bundle_revision':3,'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'entrypoint':'experiments/public_cpr_fullpose/infer_candidate_v2.py','pinned_sources_and_assets':pinned,
        'predecessor_sha256':o.parent.digest(source),'launch_requirement':'Default manifest or explicit --manifest data/public_cpr/rgb-evaluation-v15/bundle-manifest-v3.json',
        'loader_fix':'Use the unchanged compact-head adapter required by the existing v4 body manifest. No recognition weight, threshold or fusion-rule changes.'}
    o.write(root/'bundle-manifest-v3.json',manifest)
    print('Frozen bundle3 with compact-body loader adapter',flush=True)


if __name__=='__main__':main()
