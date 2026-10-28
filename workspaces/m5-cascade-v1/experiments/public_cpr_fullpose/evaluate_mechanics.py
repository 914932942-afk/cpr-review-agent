"""One frozen mechanics regression on all386 keypoint and26 RGB-body inputs."""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import train_mechanics as d
from mechanics import describe
from train_head import o


def summary(pred, scores, targets):
    measured = o.parent.metrics(pred.astype(float), targets)
    ranking = o.parent.metrics(scores[:, :11], targets)
    measured['mAP'] = ranking['mAP']
    for row, rank in zip(measured['per_label'], ranking['per_label']): row['average_precision'] = rank['average_precision']
    return measured


def main():
    p = o.parent.read(d.ROOT/'protocol.json'); o.parent.check_hashes(p['implementation_sources'])
    if (d.ROOT/'evaluation.json').exists(): raise FileExistsError('Preserve first frozen regression')
    selection = o.parent.read(d.ROOT/'selection.json')
    if o.parent.digest(d.ROOT/'experts.pt') != selection['sha256']: raise ValueError('Head changed')
    head = torch.load(d.ROOT/'experts.pt', map_location='cpu', weights_only=True)
    if head['protocol_sha256'] != o.parent.digest(d.ROOT/'protocol.json'): raise ValueError('Protocol changed')
    o.write(d.ROOT/'evaluation-plan.json', {'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'head_sha256':selection['sha256'],'implementation_sha256':o.parent.digest(Path(__file__).resolve()),
        'unchanged_raw_gates':o.parent.read(d.DATA/'rgb-video-acceptance-v4/acceptance-protocol.json')['quality_gates'],
        'no_raw_time_selection_or_tuning':True,'scope':'replace errors5/6 only in frozen v6 body model, no clipping to one label'})
    result = {'quality_passed':False,'runtime_release_approved':False,'new_manual_labels':False,'scopes':{}}
    with np.load(d.ROOT/'features.npz',allow_pickle=False) as f:
        lookup={str(k):v.copy() for k,v in zip(f['record_ids'],f['features'])}
    for scope,file in [('public_keypoints','test-predictions.npz'),('raw_video','raw-predictions.npz')]:
        with np.load(d.DATA/'rgb-evaluation-v6'/file,allow_pickle=False) as f:
            ids,scores,pred,y=[f[k].copy() for k in ['record_ids','scores','predictions','targets']]
        if scope == 'public_keypoints':
            features=np.stack([lookup[str(k)] for k in ids])
        else:
            values=[]
            for key in ids:
                path=d.DATA/'rgb-video-acceptance-v4/runs'/str(key).replace('/','--')/'inference.npz'
                with np.load(path,allow_pickle=False) as body: values.append(describe(body['keypoint'][0],body['keypoint_score'][0]))
            features=np.stack(values)
        for label,(probability,decisions) in d.predict(features,head).items():
            index=list(o.parent.LABEL_IDS).index(label); scores[:,index]=probability; pred[:,index]=decisions
        metrics=summary(pred,scores,y)
        result['scopes'][scope]={'metrics':metrics,'exact_matches':int((pred==y).all(1).sum()),'records':len(y)}
        with (d.ROOT/(scope+'-predictions.npz')).open('xb') as handle:
            np.savez_compressed(handle,record_ids=ids,scores=scores,predictions=pred,targets=y,features=features)
        print(scope,metrics['macro_f1'],'normal',metrics['correct_false_alarm_rate'],[(r['id'],round(r['f1'],3),r['recall']) for r in metrics['per_label'] if r['id'] in [5,6]],flush=True)
    g=o.parent.read(d.ROOT/'evaluation-plan.json')['unchanged_raw_gates']; raw=result['scopes']['raw_video']['metrics']
    result['raw_checks']={'f1':raw['macro_f1']>=g['single_macro_f1'],'recall':min(r['recall'] for r in raw['per_label'])>=g['minimum_label_recall'],
        'normal':raw['correct_false_alarm_rate']<=g['correct_false_alarm_rate'],'all26':raw['records']==26}
    result['raw_quality_passed']=all(result['raw_checks'].values())
    o.write(d.ROOT/'evaluation.json',result)


if __name__=='__main__': main()
