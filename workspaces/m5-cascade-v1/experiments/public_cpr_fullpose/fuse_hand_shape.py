"""Small hand-shape/semantic heads and a learned upper bound on error count."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits
from head import CODE,count_fields
from train_head import o
from train_rgb_experts_v3 import binary_metrics
from fuse_cardinality import counts

DATA=CODE/'data/public_cpr'; ROOT=DATA/'rgb-evaluation-v14'; BASE=DATA/'rgb-evaluation-v12'; SHAPE=DATA/'hand-shape-v1'
HAND_INDICES=np.r_[np.arange(384),np.arange(768,1152)]


def freeze():
    if ROOT.exists():raise FileExistsError('Preserve local-shape experiment')
    files=[Path(__file__).resolve(),SHAPE/'profile.json',SHAPE/'development.npz',DATA/'rgb-domain-features-v5/features.npz',
        BASE/'experts.pt',BASE/'protocol.json',DATA/'rgb-evaluation-v13/cardinality.pt']
    o.write(ROOT/'protocol.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'protected_sources':{p.relative_to(CODE).as_posix():o.parent.digest(p) for p in files},
        'fit':'Original public non-test training rows only; leave entire error combinations out together; no frame or repeat crossing',
        'grid':'HOG576 or HOG576+semantic768; clipped balanced logistic C=.003/.03/.3/3; train-OOF thresholds .1.. .9 plus .95/.975',
        'selection':'Prefer OOF recall>=.8, specificity>=.8, zero original13/12 negative-control alarms; otherwise minimum control FP then maximum F1. Within .02 F1 prefer fewer fields.',
        'labels':'Original labels unchanged. Pure13/12 are negatives only for error1/2, never labelled Correct.',
        'unavailable':'If fewer than8 M1-observable wrist frames preserve the old RGB decisions, not fabricated negative evidence.',
        'fusion':'Replace hand1/2; retain other v12 decisions. Learned v13 count is an upper bound; it may not force any error below its individual trained threshold. Same rule for all inputs.',
        'maximum_fields':70000,'raw_gates':o.parent.read(DATA/'rgb-video-acceptance-v4/acceptance-protocol.json')['quality_gates'],
        'prior_exposure':'Engineering regression, not independent blind evidence','runtime_release_approved':False})


def checked():
    p=o.parent.read(ROOT/'protocol.json');o.parent.check_hashes(p['protected_sources']);return p


def joined(shape_file,rgb_file):
    with np.load(shape_file,allow_pickle=False) as f:
        shape={str(k):(v.copy(),bool(a)) for k,v,a in zip(f['record_ids'],f['features'],f['available'])}
    with np.load(rgb_file,allow_pickle=False) as f:
        ids=f['record_ids'].copy(); rgb=f['features'].copy() if 'features' in f else f['rgb_features'].copy()
    x=np.stack([np.r_[shape[str(k)][0],v[HAND_INDICES]] for k,v in zip(ids,rgb)])
    return x,np.asarray([shape[str(k)][1] for k in ids]),ids


def train():
    p=checked()
    if (ROOT/'hands.pt').exists():raise FileExistsError('Preserve selected shape heads')
    x,available,ids=joined(SHAPE/'development.npz',DATA/'rgb-domain-features-v5/features.npz')
    with np.load(DATA/'rgb-domain-features-v5/features.npz',allow_pickle=False) as f:
        y,splits,labels=[f[k].copy() for k in ['targets','splits','label_ids']]
    rows={r['record_id']:r for r in o.parent.read(DATA/'rgb-domain-features-v5/manifest.json')['records']}
    fit=(splits=='train') & available; val=(splits=='validation') & available
    xf=x[fit].astype(float); groups=np.asarray(['+'.join(map(str,sorted(rows[str(k)]['original_labels']))) for k in ids[fit]])
    control=np.asarray([rows[str(k)]['original_labels'] in [[12],[13]] for k in ids[fit]])
    experts=[];selection=[]
    for label in [1,2]:
        target=y[fit,list(labels).index(label)]
        if len(np.unique(groups[target]))<2:raise ValueError('Insufficient independent positive groups for hand label')
        candidates=[];saved={}
        for scope,indices in [('shape',np.arange(576)),('shape_semantic',np.arange(1344))]:
            values=xf[:,indices]
            for strength in [.003,.03,.3,3.]:
                prob=np.full(len(values),np.nan)
                for a,b in LeaveOneGroupOut().split(values,target,groups):
                    if len(np.unique(target[a]))!=2:raise ValueError('Fold lacks a class')
                    mean,scale=values[a].mean(0),values[a].std(0);scale[scale<1e-6]=1
                    with threadpool_limits(limits=2):
                        model=LogisticRegression(C=strength,class_weight='balanced',solver='liblinear',max_iter=2500).fit(np.clip((values[a]-mean)/scale,-6,6),target[a])
                    prob[b]=model.predict_proba(np.clip((values[b]-mean)/scale,-6,6))[:,1]
                if not np.isfinite(prob).all():raise ValueError('Incomplete OOF hand probabilities')
                for threshold in [*np.linspace(.1,.9,17),.95,.975]:
                    metric=binary_metrics(prob>=threshold,target);fp=int((prob[control]>=threshold).sum())
                    candidates.append({'scope':scope,'C':strength,'threshold':float(threshold),'oof':metric,'control_fp':fp,'dimension':len(indices),
                        'preferred':metric['recall']>=.8 and metric['specificity']>=.8 and fp==0})
                saved[(scope,strength)]=(indices,prob)
        preferred=[r for r in candidates if r['preferred']]
        pool=preferred or [r for r in candidates if r['control_fp']==min(v['control_fp'] for v in candidates)]
        best=max(pool,key=lambda r:(r['oof']['f1'],r['oof']['balanced_accuracy']))
        best=max([r for r in pool if r['oof']['f1']>=best['oof']['f1']-.02],key=lambda r:(-r['dimension'],r['oof']['f1'],r['oof']['balanced_accuracy'],-abs(r['threshold']-.5)))
        indices,_=saved[(best['scope'],best['C'])]; values=xf[:,indices]
        mean,scale=values.mean(0),values.std(0);scale[scale<1e-6]=1
        with threadpool_limits(limits=2):
            model=LogisticRegression(C=best['C'],class_weight='balanced',solver='liblinear',max_iter=2500).fit(np.clip((values-mean)/scale,-6,6),target)
        experts.append({'label_id':label,'indices':torch.from_numpy(indices),'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale),
            'weight':torch.from_numpy(model.coef_[0]),'bias':float(model.intercept_[0]),'threshold':best['threshold']})
        prediction=model.predict_proba(np.clip((x[val][:,indices]-mean)/scale,-6,6))[:,1]>=best['threshold']
        selection.append({'label_id':label,'selected':best,'all_candidates':candidates,'heldout_validation':binary_metrics(prediction,y[val,list(labels).index(label)])})
        print('Selected hand-shape expert',label,best,'validation',selection[-1]['heldout_validation'],flush=True)
    old=torch.load(BASE/'experts.pt',map_location='cpu',weights_only=True)
    card=torch.load(DATA/'rgb-evaluation-v13/cardinality.pt',map_location='cpu',weights_only=True)
    fields=old['combined_scalar_fields']-count_fields([e for e in old['experts'] if e['label_id'] in [1,2]])+count_fields(experts)+count_fields(card)
    if fields>p['maximum_fields']:raise ValueError('Final active model exceeds small-head budget')
    torch.save({'schema':'cpr-local-hand-heads-v1','experts':experts,'feature_dimension':1344,'combined_fields':fields,
        'protocol_sha256':o.parent.digest(ROOT/'protocol.json')},ROOT/'hands.pt')
    o.write(ROOT/'selection.json',{'experts':selection,'checkpoint_sha256':o.parent.digest(ROOT/'hands.pt'),'combined_fields':fields,
        'fit_records':int(fit.sum()),'unobservable_training_records_not_fitted':int(((splits=='train') & ~available).sum()),'test_used_for_fitting':False})


def predict(x,state):
    if x.ndim!=2 or x.shape[1]!=1344 or not np.isfinite(x).all():raise ValueError('Invalid shape/semantic input')
    output={}
    for e in state['experts']:
        z=np.clip((x[:,e['indices'].numpy()]-e['mean'].numpy())/e['scale'].numpy(),-6,6)
        probability=1/(1+np.exp(-np.clip(z @ e['weight'].numpy()+e['bias'],-60,60)))
        output[e['label_id']]=(probability,probability>=e['threshold'])
    return output


def evaluate():
    p=checked();selection=o.parent.read(ROOT/'selection.json')
    if (ROOT/'raw-evaluation.json').exists():raise FileExistsError('Preserve frozen shape regression')
    if o.parent.digest(ROOT/'hands.pt')!=selection['checkpoint_sha256']:raise ValueError('Shape head changed')
    state=torch.load(ROOT/'hands.pt',map_location='cpu',weights_only=True)
    x,available,ids=joined(SHAPE/'raw.npz',BASE/'raw-predictions.npz')
    with np.load(BASE/'raw-predictions.npz',allow_pickle=False) as f: scores=f['scores'].copy();eligible=f['predictions'].copy()
    source=o.parent.read(BASE/'protocol.json');body=torch.load(CODE/source['base_head'],map_location='cpu',weights_only=True)
    thresholds=np.tile(np.asarray(body['operating_point']['thresholds']), (len(ids),1))
    rgb=torch.load(BASE/'experts.pt',map_location='cpu',weights_only=True)
    mechanics=torch.load(CODE/source['mechanics_head'],map_location='cpu',weights_only=True)
    for e in [*rgb['experts'],*mechanics['experts']]:thresholds[:,list(o.parent.LABEL_IDS).index(e['label_id'])]=e['threshold']
    for label,(probability,decisions) in predict(x,state).items():
        col=list(o.parent.LABEL_IDS).index(label);scores[available,col]=probability[available];eligible[available,col]=decisions[available]
        thresholds[available,col]=next(e['threshold'] for e in state['experts'] if e['label_id']==label)
    with np.load(DATA/'rgb-evaluation-v6/raw-predictions.npz',allow_pickle=False) as f: body_lookup={str(k):v[:783].copy() for k,v in zip(f['record_ids'],f['features'])}
    with np.load(DATA/'body-mechanics-v3/raw_video-predictions.npz',allow_pickle=False) as f: motion={str(k):v.copy() for k,v in zip(f['record_ids'],f['features'])}
    card=torch.load(DATA/'rgb-evaluation-v13/cardinality.pt',map_location='cpu',weights_only=True)
    number=counts(np.stack([np.r_[body_lookup[str(k)],motion[str(k)]] for k in ids]),card)
    probability=np.clip(scores[:,:11],1e-9,1-1e-9); t=np.clip(thresholds,1e-9,1-1e-9)
    margin=np.log(probability/(1-probability))-np.log(t/(1-t));pred=np.zeros_like(eligible)
    for row,k in enumerate(number):
        candidates=np.flatnonzero(eligible[row]);order=candidates[np.argsort(-margin[row,candidates],kind='stable')]
        pred[row,order[:k]]=True
    with np.load(BASE/'raw-predictions.npz',allow_pickle=False) as f:y=f['targets'].copy()
    metric=o.parent.metrics(pred.astype(float),y);g=p['raw_gates']
    checks={'coverage':len(ids)==26,'f1':metric['macro_f1']>=g['single_macro_f1'],'recall':min(r['recall'] for r in metric['per_label'])>=g['minimum_label_recall'],
        'normal':metric['correct_false_alarm_rate']<=g['correct_false_alarm_rate'],'head_budget':state['combined_fields']<=70000}
    result={'raw_metrics':metric,'quality_checks':checks,'raw_video_quality_passed':all(checks.values()),'exact_matches':int((pred==y).all(1).sum()),
        'combined_fields':state['combined_fields'],'raw_shape_available':int(available.sum()),'runtime_release_approved':False,'independent_blind_test':False,
        'rows':[{'record_id':str(key),'predicted':[label for label,yes in zip(o.parent.LABEL_IDS,a) if yes],
            'target':[label for label,yes in zip(o.parent.LABEL_IDS,b) if yes]} for key,a,b in zip(ids,pred,y)]}
    with (ROOT/'raw-predictions.npz').open('xb') as f:np.savez_compressed(f,record_ids=ids,scores=scores,predictions=pred,targets=y,eligible=eligible,thresholds=thresholds,predicted_count=number)
    o.write(ROOT/'raw-evaluation.json',result)
    print('Local-shape fused raw26',metric['macro_f1'],'exact',result['exact_matches'],'normal',metric['correct_false_alarm_rate'],checks,flush=True)
    print([(r['id'],r['f1'],r['recall']) for r in metric['per_label']],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train','evaluate']);globals()[parser.parse_args().stage]()
