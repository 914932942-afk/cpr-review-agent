"""Training-only observability probe before committing a new modality."""
import hashlib
import json
import cv2
import numpy as np
from hand_gesture import Runtime,CODE


def main():
    data=CODE/'data/public_cpr'
    rows=json.loads((data/'rgb-domain-features-v5/manifest.json').read_text(encoding='utf-8'))['records']
    runtime=Runtime()
    try:
        for target in [1,2,13]:
            row=next(r for r in rows if r['split']=='train' and target in r['original_labels'])
            base=(data/'rgb-primary-controls-v1' if row.get('is_primary_control') else data/'rgb-domain-prefix-v2/rgb-subset-v1')
            selected=row['images']; frames=[cv2.imread(str(base/v['filename'])) for v in selected]
            key=hashlib.sha256(row['record_id'].encode()).hexdigest()
            with np.load(data/'fullpose-v2/skeletons'/(key+'.npz'),allow_pickle=False) as f:
                indices=np.asarray([v['frame_ordinal'] for v in selected]); p=f['keypoint'][0,indices,:17]; q=f['keypoint_score'][0,indices,:17]
            feature,audit=runtime.extract(frames,p,q)
            print(row['record_id'],row['original_labels'],json.dumps(audit),flush=True)
    finally: runtime.close()


if __name__=='__main__': main()
