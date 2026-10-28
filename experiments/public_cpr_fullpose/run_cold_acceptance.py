"""Isolated label-independent cold video runs, then compare frozen predictions."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import json
import numpy as np
from head import CODE
from train_head import o

DATA=CODE/'data/public_cpr'; ROOT=DATA/'rgb-evaluation-v15/cold-runs'


def main():
    jobs={r['record_id']:r for r in o.parent.read(DATA/'rgb-video-acceptance-v4/jobs.json')}
    chosen=['SupC00/r00/ch0','SupC02/r00/ch0','SupC09/r01/ch0']
    manifest=DATA/'rgb-evaluation-v15/bundle-manifest-v2.json'
    if ROOT.exists():raise FileExistsError('Preserve cold-run attempt')
    o.write(ROOT/'plan.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'records':chosen,
        'roles':['original-Correct','hand-specific error','contact-location error previously harmed by cardinality cap'],
        'manifest_sha256':o.parent.digest(manifest),'inference_receives_ground_truth':False,'maximum_score_difference':1e-5,
        'purpose':'Cold reconstruction from actual video bytes, independent processes. This verifies cache replay, not an unseen test.'})
    expected=np.load(DATA/'rgb-evaluation-v15/raw-predictions.npz',allow_pickle=False)
    index={str(k):i for i,k in enumerate(expected['record_ids'])};results=[]
    for key in chosen:
        job=jobs[key];folder=ROOT/job['key'];folder.mkdir();output=folder/'inference.json'
        command=[sys.executable,'-X','utf8',str(CODE/'experiments/public_cpr_fullpose/infer_candidate.py'),'--video',str(CODE/job['video']),
            '--output',str(output),'--manifest',str(manifest)]
        started=time.perf_counter();row={'record_id':key,'input_video_sha256':job['video_sha256'],'labels_provided_to_inference':False}
        print('COLD START',key,flush=True)
        with (folder/'process.txt').open('x',encoding='utf-8') as log:
            process=subprocess.run(command,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,timeout=900,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1'})
        row['exit_code']=process.returncode;row['wall_seconds']=time.perf_counter()-started
        if process.returncode==0:
            value=o.parent.read(output)
            with np.load(output.with_suffix('.npz'),allow_pickle=False) as f:
                difference=float(np.max(np.abs(f['scores'][0]-expected['scores'][index[key]])))
                match=bool(np.array_equal(f['predictions'][0],expected['predictions'][index[key]]))
            row.update({'decoded_frames':value['decoded_frames'],'score_max_difference':difference,'predictions_identical':match,
                'no_reference_labels':value['public_label_verification'] is None,'video_integrity':value['video_sha256']==job['video_sha256'],
                'reported_labels':value['predicted_original_label_ids']})
            row['passed']=match and difference<=1e-5 and row['no_reference_labels'] and row['video_integrity'] and value['decoded_frames']==job['expected_frames']
        else:row['passed']=False
        o.write(folder/'result.json',row);results.append(row);print('COLD FINISH',key,row,flush=True)
        if not row['passed']:break
    o.write(ROOT/'summary.json',{'rows':results,'planned':3,'completed':len(results),'passed':len(results)==3 and all(r['passed'] for r in results),
        'manifest_sha256':o.parent.digest(manifest),'runtime_release_approved':False})


if __name__=='__main__':main()
