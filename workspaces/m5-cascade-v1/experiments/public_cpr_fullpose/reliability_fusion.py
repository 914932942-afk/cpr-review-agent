"""Protect validated independent evidence from a weaker count estimator."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from head import CODE
from train_head import o

DATA=CODE/'data/public_cpr'; ROOT=DATA/'rgb-evaluation-v15'; SOURCE=DATA/'rgb-evaluation-v14'


def reliability(metric):
    if 'tp' in metric:
        tp=metric['tp'];fp=metric['fp'];recall=metric['recall']
    else:
        tp=metric['recall']*metric['support'];fp=metric['predicted_positive']-tp;recall=metric['recall']
    precision=tp/max(tp+fp,1)
    return {'precision':float(precision),'recall':float(recall),'qualified':bool(precision>=.8 and recall>=.8)}


def guarded(eligible,scores,thresholds,count,qualified):
    eligible=np.asarray(eligible,bool);scores=np.asarray(scores,float);thresholds=np.asarray(thresholds,float)
    count=np.asarray(count);qualified=np.asarray(qualified,bool)
    if eligible.ndim!=2 or scores.shape!=eligible.shape or thresholds.shape!=eligible.shape or count.shape!=(len(scores),) or qualified.shape!=(scores.shape[1],):
        raise ValueError('Fusion contract dimensions differ')
    if not np.isfinite(scores).all() or not np.isfinite(thresholds).all() or (count<0).any() or (count>4).any():
        raise ValueError('Invalid evidence or count input')
    p=np.clip(scores,1e-9,1-1e-9);t=np.clip(thresholds,1e-9,1-1e-9)
    margin=np.log(p/(1-p))-np.log(t/(1-t))
    result=eligible & qualified[None,:]
    for row,k in enumerate(count):
        remaining=max(int(k)-int(result[row].sum()),0)
        weak=np.flatnonzero(eligible[row] & ~qualified)
        selected=weak[np.argsort(-margin[row,weak],kind='stable')][:remaining]
        result[row,selected]=True
    return result


def freeze():
    if ROOT.exists():raise FileExistsError('Preserve reliability fusion experiment')
    sources=[Path(__file__).resolve(),SOURCE/'protocol.json',SOURCE/'selection.json',SOURCE/'hands.pt',
        DATA/'rgb-evaluation-v12/selection.json',DATA/'rgb-evaluation-v12/experts.pt',
        DATA/'rgb-evaluation-v6/selection.json',DATA/'body-mechanics-v3/selection.json',DATA/'rgb-evaluation-v13/cardinality.pt']
    measured={}
    base=o.parent.read(DATA/'rgb-evaluation-v6/selection.json')['selected']['operating_point']['validation_metrics']['per_label']
    for metric in base:measured[metric['id']]={'source':'Original held-out public validation196',**reliability(metric)}
    for expert in o.parent.read(DATA/'rgb-evaluation-v12/selection.json')['experts']:
        measured[expert['label_id']]={'source':'Leave-entire-error-combination-out RGB training predictions',**reliability(expert['selected']['oof'])}
    for expert in o.parent.read(SOURCE/'selection.json')['experts']:
        measured[expert['label_id']]={'source':'Leave-entire-error-combination-out local-shape training predictions',**reliability(expert['selected']['oof'])}
    mechanics=next(r for r in o.parent.read(DATA/'body-mechanics-v3/selection.json')['experts'] if r['label_id']==6)
    measured[6]={'source':'Original held-out public validation196',**reliability(mechanics['selected']['metric'])}
    o.write(ROOT/'protocol.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'protected_sources':{p.relative_to(CODE).as_posix():o.parent.digest(p) for p in sources},
        'reliability':{str(k):v for k,v in measured.items()},'qualification_rule':'Development precision>=.80 AND recall>=.80, same threshold for all labels',
        'decision':'Retain all qualified independent positive evidence. Use learned0..4 supported-error count only to limit additional low-reliability positives. Never force a negative candidate positive or remove qualified evidence.',
        'no_test_labels_used_for_reliability':True,'no_changed_model_or_individual_thresholds':True,
        'unchanged_raw_gates':o.parent.read(DATA/'rgb-video-acceptance-v4/acceptance-protocol.json')['quality_gates'],
        'scope':'Same26-video engineering regression, previously inspected. Not an independent blind test.',
        'runtime_release_approved':False})
    print('Reliability fusion frozen; qualified labels',[k for k,v in measured.items() if v['qualified']],flush=True)


def checked():
    p=o.parent.read(ROOT/'protocol.json');o.parent.check_hashes(p['protected_sources']);return p


def evaluate():
    p=checked()
    if (ROOT/'raw-evaluation.json').exists():raise FileExistsError('Preserve first reliability-fusion regression')
    with np.load(SOURCE/'raw-predictions.npz',allow_pickle=False) as f:
        ids,scores,eligible,thresholds,count=[f[k].copy() for k in ['record_ids','scores','eligible','thresholds','predicted_count']]
    qualification=np.asarray([p['reliability'][str(k)]['qualified'] for k in o.parent.LABEL_IDS])
    pred=guarded(eligible,scores[:,:11],thresholds,count,qualification)
    with np.load(SOURCE/'raw-predictions.npz',allow_pickle=False) as f:y=f['targets'].copy()
    metric=o.parent.metrics(pred.astype(float),y);g=p['unchanged_raw_gates']
    fields=o.parent.read(SOURCE/'selection.json')['combined_fields']
    checks={'coverage':len(ids)==26,'f1':metric['macro_f1']>=g['single_macro_f1'],'recall':min(r['recall'] for r in metric['per_label'])>=g['minimum_label_recall'],
        'normal':metric['correct_false_alarm_rate']<=g['correct_false_alarm_rate'],'head_budget':fields<=70000}
    result={'raw_metrics':metric,'quality_checks':checks,'raw_video_quality_passed':all(checks.values()),'exact_matches':int((pred==y).all(1).sum()),
        'combined_fields':fields,'runtime_release_approved':False,'independent_blind_test':False,'no_new_manual_labels':True,
        'protocol_sha256':o.parent.digest(ROOT/'protocol.json'),
        'rows':[{'record_id':str(key),'predicted':[label for label,yes in zip(o.parent.LABEL_IDS,a) if yes],
            'target':[label for label,yes in zip(o.parent.LABEL_IDS,b) if yes],'exact':bool(np.array_equal(a,b))} for key,a,b in zip(ids,pred,y)]}
    with (ROOT/'raw-predictions.npz').open('xb') as f:np.savez_compressed(f,record_ids=ids,scores=scores,predictions=pred,targets=y,eligible=eligible,thresholds=thresholds,predicted_count=count)
    o.write(ROOT/'raw-evaluation.json',result)
    print('Reliability-guarded raw26',metric['macro_f1'],'exact',result['exact_matches'],'normal',metric['correct_false_alarm_rate'],checks,flush=True)
    print([(r['id'],r['f1'],r['recall']) for r in metric['per_label']],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','evaluate']);globals()[parser.parse_args().stage]()
