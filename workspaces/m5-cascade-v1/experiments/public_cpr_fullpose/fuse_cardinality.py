"""Learned supported-error cardinality; never assume a clip contains one error."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import warnings
import numpy as np
import torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from head import CODE, count_fields
from train_head import o

DATA=CODE/'data/public_cpr'
ROOT=DATA/'rgb-evaluation-v13'
BASE=DATA/'rgb-evaluation-v12'


def features():
    with np.load(DATA/'body-mechanics-v3/features.npz',allow_pickle=False) as f:
        lookup={str(key):value.copy() for key,value in zip(f['record_ids'],f['features'])}
    with np.load(DATA/'fullpose-v2/development.npz',allow_pickle=False) as f:
        ids,y,splits,cohorts=[f[k].copy() for k in ['record_ids','targets','splits','cohorts']]
        x=np.concatenate([f['features'][:,:783],np.stack([lookup[str(k)] for k in ids])],1)
    return x,y,splits,cohorts,ids


def freeze():
    if ROOT.exists(): raise FileExistsError('Preserve cardinality fusion experiment')
    x,y,splits,cohorts,ids=features()
    files=[Path(__file__).resolve(),BASE/'experts.pt',BASE/'mechanics6.pt',BASE/'protocol.json',DATA/'fullpose-v2/development.npz',DATA/'body-mechanics-v3/features.npz']
    o.write(ROOT/'protocol.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'implementation_and_data':{p.relative_to(CODE).as_posix():o.parent.digest(p) for p in files},
        'input':'Frozen PoseC3D512 + body271 + mechanics176; original train738 fit, validation196 selection; no test labels/images in fit',
        'targets':'Number of supported original error labels, learned from unchanged public annotations. 0..4 allowed. No original-Correct relabelling and no single-error assumption.',
        'grid':'train-only ANOVA top64/128/256, standardized multinomial logistic C=.003/.03/.3; class-balanced',
        'selection':'Require all original-Correct validation clips count0 if possible; then maximize minimum single/composite count accuracy, then overall accuracy; prefer fewer fields',
        'decision':'Predict count from body inputs; select that many threshold-adjusted score logits from frozen v12. Same rule for every input; no filename, target or dataset-single flag.',
        'maximum_combined_fields':70000,'raw_gates':o.parent.read(DATA/'rgb-video-acceptance-v4/acceptance-protocol.json')['quality_gates'],
        'scope':'Engineering regression previously inspected, not independent blind evidence','runtime_release_approved':False})


def checked():
    p=o.parent.read(ROOT/'protocol.json'); o.parent.check_hashes(p['implementation_and_data']); return p


def train():
    p=checked()
    if (ROOT/'cardinality.pt').exists(): raise FileExistsError('Preserve cardinality selection')
    x,y,splits,cohorts,ids=features(); fit=splits=='train'; val=splits=='validation'
    yf=y[fit].sum(1).astype(int); yv=y[val].sum(1).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); merit,_=f_classif(x[fit],yf)
    merit=np.nan_to_num(merit,nan=-1e30,posinf=1e30,neginf=-1e30)
    order=np.argsort(-merit,kind='stable'); candidates=[]; states={}
    base_state=torch.load(BASE/'experts.pt',map_location='cpu',weights_only=True)
    for width in [64,128,256]:
        indices=np.sort(order[:width]); xf=x[fit][:,indices]; xv=x[val][:,indices]
        mean,scale=xf.mean(0),xf.std(0); scale[scale<1e-6]=1
        for strength in [.003,.03,.3]:
            with threadpool_limits(limits=2):
                model=LogisticRegression(C=strength,class_weight='balanced',max_iter=3000,random_state=20261021).fit(np.clip((xf-mean)/scale,-6,6),yf)
            pred=model.predict(np.clip((xv-mean)/scale,-6,6)); correct=pred==yv
            errors=int((pred[yv==0]!=0).sum())
            single=float(correct[cohorts[val]=='single'].mean()); composite=float(correct[cohorts[val]=='composite'].mean())
            state={'indices':torch.from_numpy(indices),'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale),
                'weight':torch.from_numpy(model.coef_),'bias':torch.from_numpy(model.intercept_),'classes':torch.from_numpy(model.classes_)}
            fields=base_state['combined_scalar_fields']+count_fields(state)
            name=f'k{width}_C{strength}'; states[name]=state
            candidates.append({'name':name,'normal_count_errors':errors,'single_count_accuracy':single,'composite_count_accuracy':composite,
                'count_accuracy':float(correct.mean()),'fields':fields,'within_budget':fields<=70000})
    allowed=[r for r in candidates if r['within_budget']]
    if not allowed: raise ValueError('No within-budget cardinality candidate')
    best=max(allowed,key=lambda r:(-r['normal_count_errors'],min(r['single_count_accuracy'],r['composite_count_accuracy']),r['count_accuracy'],-r['fields']))
    state={**states[best['name']],'schema':'cpr-supported-count-v1','protocol_sha256':o.parent.digest(ROOT/'protocol.json')}
    torch.save(state,ROOT/'cardinality.pt')
    o.write(ROOT/'selection.json',{'selected':best,'candidates':candidates,'checkpoint_sha256':o.parent.digest(ROOT/'cardinality.pt'),
        'test_rows_used_for_fit_or_thresholds':0,'main_network_training':False})
    print('Selected learned supported-count model',best,flush=True)


def counts(x,state):
    if np.asarray(x).ndim!=2 or x.shape[1]!=959 or not np.isfinite(x).all(): raise ValueError('Invalid cardinality feature contract')
    z=np.clip((x[:,state['indices'].numpy()]-state['mean'].numpy())/state['scale'].numpy(),-6,6)
    logits=z @ state['weight'].numpy().T+state['bias'].numpy()
    return state['classes'].numpy()[logits.argmax(1)].astype(int)


def decide(scores,thresholds,cardinality):
    probability=np.clip(np.asarray(scores)[:,:11],1e-9,1-1e-9)
    t=np.clip(np.asarray(thresholds),1e-9,1-1e-9)
    margins=np.log(probability/(1-probability))-np.log(t/(1-t))
    order=np.argsort(-margins,axis=1,kind='stable'); result=np.zeros_like(probability,dtype=bool)
    for row,k in enumerate(cardinality):
        if k<0 or k>4: raise ValueError('Unsupported predicted count')
        result[row,order[row,:k]]=True
    return result


def evaluate():
    p=checked(); selection=o.parent.read(ROOT/'selection.json')
    if (ROOT/'raw-evaluation.json').exists(): raise FileExistsError('Preserve first frozen cardinality regression')
    if o.parent.digest(ROOT/'cardinality.pt')!=selection['checkpoint_sha256']: raise ValueError('Selected cardinality head changed')
    state=torch.load(ROOT/'cardinality.pt',map_location='cpu',weights_only=True)
    # First construct and predict counts without reading any raw targets.
    with np.load(DATA/'rgb-evaluation-v6/raw-predictions.npz',allow_pickle=False) as f:
        lookup={str(k):v[:783].copy() for k,v in zip(f['record_ids'],f['features'])}
    with np.load(DATA/'body-mechanics-v3/raw_video-predictions.npz',allow_pickle=False) as f:
        motion={str(k):v.copy() for k,v in zip(f['record_ids'],f['features'])}
    with np.load(BASE/'raw-predictions.npz',allow_pickle=False) as f:
        ids=f['record_ids'].copy(); scores=f['scores'].copy()
    x=np.stack([np.r_[lookup[str(k)],motion[str(k)]] for k in ids]); number=counts(x,state)
    body_protocol=o.parent.read(BASE/'protocol.json')
    base=torch.load(CODE/body_protocol['base_head'],map_location='cpu',weights_only=True)
    thresholds=np.asarray(base['operating_point']['thresholds'],float)
    experts=torch.load(BASE/'experts.pt',map_location='cpu',weights_only=True)
    for expert in experts['experts']: thresholds[list(o.parent.LABEL_IDS).index(expert['label_id'])]=expert['threshold']
    mechanics=torch.load(CODE/body_protocol['mechanics_head'],map_location='cpu',weights_only=True)
    for expert in mechanics['experts']: thresholds[list(o.parent.LABEL_IDS).index(expert['label_id'])]=expert['threshold']
    pred=decide(scores,thresholds,number)
    with np.load(BASE/'raw-predictions.npz',allow_pickle=False) as f: y=f['targets'].copy()
    metric=o.parent.metrics(pred.astype(float),y); g=p['raw_gates']
    checks={'coverage':len(ids)==26,'f1':metric['macro_f1']>=g['single_macro_f1'],'recall':min(r['recall'] for r in metric['per_label'])>=g['minimum_label_recall'],
        'normal':metric['correct_false_alarm_rate']<=g['correct_false_alarm_rate'],'head_budget':selection['selected']['fields']<=70000}
    result={'raw_metrics':metric,'quality_checks':checks,'raw_video_quality_passed':all(checks.values()),'exact_matches':int((pred==y).all(1).sum()),
        'runtime_release_approved':False,'combined_scalar_fields':selection['selected']['fields'],'no_new_labels':True,'independent_blind_test':False,
        'rows':[{'record_id':str(key),'predicted_count':int(k),'predicted':[label for label,yes in zip(o.parent.LABEL_IDS,v) if yes],
            'target':[label for label,yes in zip(o.parent.LABEL_IDS,t) if yes]} for key,k,v,t in zip(ids,number,pred,y)]}
    with (ROOT/'raw-predictions.npz').open('xb') as handle: np.savez_compressed(handle,record_ids=ids,predictions=pred,scores=scores,targets=y,predicted_count=number)
    o.write(ROOT/'raw-evaluation.json',result)
    print('Count-fused raw26',metric['macro_f1'],'exact',result['exact_matches'],'normal',metric['correct_false_alarm_rate'],checks,flush=True)
    print([(r['id'],r['f1'],r['recall']) for r in metric['per_label']],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['freeze','train','evaluate']); globals()[parser.parse_args().stage]()
