"""Cache only video features. Never load labels, selected head, or predictions."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from raw_features import CODE, Runtime, digest
from prepare_features import write_new

DATA = CODE/'data/public_cpr'
ROOT = DATA/'fullpose-raw-v1'


def main():
    if ROOT.exists():
        raise FileExistsError('Preserve raw feature extraction attempt')
    jobs = json.loads((DATA/'rgb-video-acceptance-v4/jobs.json').read_text(encoding='utf-8'))
    sources = [Path(__file__).resolve(), Path(__file__).with_name('raw_features.py'), Path(__file__).with_name('descriptor.py')]
    write_new(ROOT/'feature-plan.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'jobs_sha256': digest(DATA/'rgb-video-acceptance-v4/jobs.json'),
        'implementation_sources': {p.relative_to(CODE).as_posix(): digest(p) for p in sources},
        'kind': 'label-independent feature caching; no head selection or evaluation', 'records': 26})
    runtime = Runtime()
    rows = []
    for i, job in enumerate(jobs, 1):
        started = time.perf_counter()
        folder = DATA/'rgb-video-acceptance-v4/runs'/job['key']
        features, values, audit = runtime.extract(CODE/job['video'], folder/'inference.json', folder/'inference.npz')
        path = ROOT/(job['video_sha256']+'.npz')
        np.savez_compressed(path, features=features[None], **values)
        row = {'record_id': job['record_id'], 'video_sha256': job['video_sha256'], 'feature_file': path.name,
               'feature_sha256': digest(path), 'wall_seconds': time.perf_counter()-started, **audit}
        write_new(ROOT/(job['video_sha256']+'.json'), row)
        rows.append(row)
        print(f'[{i}/26] full68 feature ready {job["record_id"]} {row["wall_seconds"]:.1f}s matchRMSE={audit["maximum_actor_match_rmse_pixels"]:.4f}', flush=True)
    write_new(ROOT/'feature-manifest.json', {'rows': rows, 'records': len(rows), 'labels_read': False,
        'backbone_training': False, 'old_v4_perception_cache_reused_with_video_and_model_hash_checks': True})
    print('26 raw full68 feature caches complete; no label evaluation', flush=True)


if __name__ == '__main__':
    main()
