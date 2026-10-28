"""Small-head-only development search; unchanged public data and fixed gates."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / 'experiments/public_rgb_optimization'))
import optimize_head as o
from compact_head import CompactHead, count_fields, load

BASE = o.DATA / 'rgb-evaluation-v3'
OUTPUT = o.DATA / 'rgb-evaluation-v4'
REPORT = CODE / 'test-results/public-rgb-optimization-v4'
SEED = 20261022


def prepare():
    if OUTPUT.exists(): raise FileExistsError('Preserve prior experiments')
    parent = o.checked(BASE)
    selected = o.parent.read(BASE/'selection.json')['selected']
    protected = dict(parent['protected_sources'])
    for name in ['protocol.json','selection.json','evaluation.json',selected['checkpoint']]:
        path=BASE/name;protected[path.relative_to(CODE).as_posix()]=o.parent.digest(path)
    OUTPUT.mkdir(parents=True)
    for name in ['development.npz','locked-test.npz','record-splits.json']: shutil.copyfile(BASE/name,OUTPUT/name)
    protocol={**parent,'purpose':'Compact linear/MLP and budgeted tree mixtures; frozen visual features only',
        'frozen_at':datetime.now(timezone.utc).isoformat(),'parent_protocol_sha256':o.parent.digest(BASE/'protocol.json'),
        'parent_protocol':(BASE/'protocol.json').relative_to(CODE).as_posix(),'protected_sources':protected,
        'implementation_sources':{**parent['implementation_sources'],**{p.relative_to(CODE).as_posix():o.parent.digest(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('compact_head.py'),Path(__file__).with_name('infer_compact_video.py')]}},
        'candidate_policy':'Three balanced linear heads C=.1,1,10; MLP32/64; two tree32 branches; existing tree64 control; eight mixtures of best validation linear or MLP32 with tree32 at weights .5/.75. Oversized candidates ineligible.',
        'candidate_configs':([{'kind':'linear','C':c} for c in [.1,1.,10.]]+[{'kind':'mlp','hidden':h} for h in [32,64]]+[{'kind':'tree32','branch':b} for b in ['combined','geometry']]+[{'kind':'existing_control64'}]+[{'kind':'mixture','dense':d,'tree':t,'tree_weight':w} for d in ['best_validation_linear','mlp32'] for t in ['combined32','geometry32'] for w in [.5,.75]]),
        'head_budget_definition':'All tree-node fields, dense weights/biases, train normalization statistics and mixture weights <=70000; no stored training-example bank',
        'head_training':{'mlp_hidden':[32,64],'epochs':800,'patience':80,'learning_rate':.001,'weight_decay':.01,'dropout':.2,'positive_weight':'train-derived square root','normal_auxiliary_loss_weight':.25},
        'selection_rule':'Keep validation single>=.85, composite>=.80, min label recall>=.60, zero Correct false alarms and <=70000 fields. Estimate one bootstrap SE of best worst-cohort F1; among comparable heads choose smallest cohort-balanced Brier score, then fewer fields.',
        'new_model_fitting_in_this_stage':True,'previous_test_exposure':'v1 and v3 engineering test metrics inspected. v2 test was not evaluated. Current fitting, normalization, thresholds and selection use development only; reused test is not blind evidence.'}
    o.write(OUTPUT/'protocol.json',protocol)
    print('Frozen compact-head search; visual backbone, recordings and gates unchanged',flush=True)


def base_state(protocol,components):
    return {'schema':'cpr-compact-head-v1','components':components,'input_dimension':783,'feature_indices':torch.arange(783),
        'label_ids':list(o.parent.LABEL_IDS),'source_revision':protocol['source_revision'],'feature_profile_hash':protocol['feature_profile_hash'],
        'protocol_sha256':o.parent.digest(OUTPUT/'protocol.json'),'score_semantics':'Uncalibrated compact-head scores; validation-selected thresholds'}


def brier(probability,truth,cohorts):
    return float(np.mean([((probability[cohorts==c,:11]-truth[cohorts==c])**2).mean() for c in ['single','composite']]))


def macro_f1(pred,truth):
    truth=truth.astype(bool);tp=(pred&truth).sum(0);fp=(pred&~truth).sum(0);fn=(~pred&truth).sum(0)
    return float(np.mean(2*tp/np.maximum(2*tp+fp+fn,1)))


def train():
    protocol=o.checked(OUTPUT)
    if (OUTPUT/'selection.json').exists() or list(OUTPUT.glob('*.pt')): raise FileExistsError('Preserve fitting attempt')
    torch.set_num_threads(4)
    with np.load(OUTPUT/'development.npz',allow_pickle=False) as f:
        x,y,splits,cohorts,ids=[f[k].copy() for k in ['features','targets','splits','cohorts','record_ids']]
    if {o.parent.group_id(str(k)) for k in ids}&{o.parent.group_id(k) for k in protocol['raw_test_reservations']}: raise ValueError('Reserved recording leakage')
    fit=splits=='train';val=splits=='validation';xf=x[fit].astype(np.float64);xv=x[val].astype(np.float64);yf=y[fit];yv=y[val];cv=cohorts[val]
    mean=xf.mean(0);scale=xf.std(0);scale[scale<1e-6]=1.
    normal_fit=(xf-mean)/scale;normal_val=(xv-mean)/scale
    target=np.column_stack([yf,~yf.any(1)]);validation_target=np.column_stack([yv,~yv.any(1)])
    components={};candidates=[];probabilities={}

    def register(name, parts, extra=None):
        state=base_state(protocol,parts);fields=count_fields(parts)
        if fields>70000: candidates.append({'name':name,'eligible':False,'scalar_fields':fields,'reason':'head budget'});return None
        probability=CompactHead(state).probability(xv)
        choices=o.options(probability,yv,cv);eligible=[r for r in choices if r['eligible']]
        chosen=max(eligible or choices,key=lambda r:tuple(r['validation_rank']))
        state['operating_point']={k:chosen[k] for k in ['thresholds','normal_gate','validation_metrics']}
        path=OUTPUT/(name+'.pt');torch.save(state,path)
        restored,_=load(path,protocol['feature_profile_hash'],o.parent.digest(OUTPUT/'protocol.json'))
        if not np.array_equal(restored.probability(xv),probability): raise AssertionError('Reload mismatch')
        row={'name':name,'config':{'name':name},'eligible':bool(eligible),'scalar_fields':fields,'checkpoint':path.name,'checkpoint_sha256':o.parent.digest(path),'checkpoint_bytes':path.stat().st_size,
            'operating_point':state['operating_point'],'threshold_method':chosen['method'],'validation_cohorts':chosen['validation_cohorts'],'validation_rank':chosen['validation_rank'],
            'validation_brier':brier(probability,yv,cv),'training_scope':'small head only','details':extra or {}}
        candidates.append(row);components[name]=parts;probabilities[name]=probability
        print(name,'validation single/composite',row['validation_cohorts']['single']['macro_f1'],row['validation_cohorts']['composite']['macro_f1'],'fields',fields,flush=True)
        return row

    for strength in [.1,1.,10.]:
        weights=[];bias=[]
        with threadpool_limits(limits=4):
            for k in range(12):
                model=LogisticRegression(C=strength,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=SEED)
                model.fit(normal_fit,target[:,k]);weights.append(model.coef_[0]);bias.append(model.intercept_[0])
        component={'kind':'linear','mixture_weight':1.,'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale),'weight1':torch.from_numpy(np.stack(weights)),'bias1':torch.tensor(bias,dtype=torch.float64)}
        register('linear'+str(strength),[component])
    positive=np.sqrt((len(yf)-target.sum(0))/np.maximum(target.sum(0),1)).astype(np.float32)
    tx=torch.tensor(normal_fit,dtype=torch.float32);ty=torch.tensor(target,dtype=torch.float32);vx=torch.tensor(normal_val,dtype=torch.float32);vy=torch.tensor(validation_target,dtype=torch.float32)
    for hidden in [32,64]:
        torch.manual_seed(SEED)
        model=nn.Sequential(nn.Linear(783,hidden),nn.ReLU(),nn.Dropout(.2),nn.Linear(hidden,12))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        def loss(logits,truth):
            values=torch.nn.functional.binary_cross_entropy_with_logits(logits,truth,pos_weight=torch.from_numpy(positive),reduction='none')
            return values[:,:11].mean()+.25*values[:,11].mean()
        best=float('inf');best_state=None;stale=0;best_epoch=0
        for epoch in range(1,801):
            model.train();optimizer.zero_grad(set_to_none=True);value=loss(model(tx),ty);value.backward();optimizer.step();model.eval()
            with torch.inference_mode():validation_loss=float(loss(model(vx),vy))
            if not np.isfinite(validation_loss):raise ValueError('Nonfinite head loss')
            if validation_loss<best-1e-5:best=validation_loss;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0;best_epoch=epoch
            else:stale+=1
            if stale>=80:break
        model.load_state_dict(best_state);model.eval()
        component={'kind':'mlp','mixture_weight':1.,'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale),'weight1':best_state['0.weight'],'bias1':best_state['0.bias'],'weight2':best_state['3.weight'],'bias2':best_state['3.bias']}
        with torch.inference_mode():reference=model(vx).sigmoid().numpy()
        difference=float(np.abs(CompactHead(base_state(protocol,[component])).probability(xv)-reference).max())
        if difference>1e-5:raise AssertionError('MLP export mismatch')
        register('mlp'+str(hidden),[component],{'epochs':epoch,'best_epoch':best_epoch,'portable_vs_training_max_difference':difference})
    for branch in ['combined','geometry']:
        config={**o.CONFIGS[0],'branches':[branch],'trees':32}
        outputs,reference=o.fit_candidate(config,xf,target,cohorts[fit],xv)
        component={'kind':'tree','mixture_weight':1.,'classifiers':outputs}
        if np.abs(CompactHead(base_state(protocol,[component])).probability(xv)-reference).max()>1e-10:raise AssertionError('Tree export mismatch')
        register(branch+'32',[component])
    control_path=o.DATA/'rgb-evaluation-v2/control64.pt'
    control=torch.load(control_path,map_location='cpu',weights_only=True)
    register('control64',[{'kind':'tree','mixture_weight':1.,'classifiers':control['classifiers']}])
    best_linear=max([r for r in candidates if r['name'].startswith('linear')],key=lambda r:tuple(r['validation_rank']))['name']
    for dense in [best_linear,'mlp32']:
        for tree in ['combined32','geometry32']:
            for tree_weight in [.5,.75]:
                parts=[{**components[tree][0],'mixture_weight':tree_weight},{**components[dense][0],'mixture_weight':1-tree_weight}]
                register(tree+'_'+dense+'_'+str(tree_weight),parts)
    eligible=[r for r in candidates if r['eligible']]
    if not eligible:o.write(OUTPUT/'development-failure.json',{'all_candidates':candidates});raise RuntimeError('No eligible development candidate')
    best=max(eligible,key=lambda r:tuple(r['validation_rank']));op=best['operating_point'];pred=o.parent.predict(probabilities[best['name']],op['thresholds'],op['normal_gate'])
    strata=defaultdict(list)
    for i,(cohort,truth) in enumerate(zip(cv,yv)):strata[(str(cohort),tuple(np.flatnonzero(truth)))].append(i)
    rng=np.random.default_rng(SEED);boot=[]
    for _ in range(200):
        idx=np.concatenate([rng.choice(rows,len(rows),replace=True) for _,rows in sorted(strata.items())]);boot.append(min(macro_f1(pred[idx][cv[idx]==c],yv[idx][cv[idx]==c]) for c in ['single','composite']))
    se=float(np.std(boot,ddof=1));floor=best['validation_rank'][0]-se;comparable=[r for r in eligible if r['validation_rank'][0]>=floor]
    chosen=min(comparable,key=lambda r:(r['validation_brier'],r['scalar_fields']))
    o.write(OUTPUT/'selection.json',{'selected':chosen,'all_candidates':candidates,'protocol_sha256':o.parent.digest(OUTPUT/'protocol.json'),'test_read_during_fit_or_selection':False,'main_network_training':False,
        'normalization_fitted_on_train_only':True,'fitted_records':738,'tuning_records':196,'validation_best_name':best['name'],'bootstrap_standard_error':se,'comparable_floor':floor,'comparable_names':[r['name'] for r in comparable]})
    o.write(OUTPUT/'fit-access-audit.json',{'passed':True,'npz_files_decoded_by_fitter':[(OUTPUT/'development.npz').relative_to(CODE).as_posix()],'test_npz_decoded_by_fitter':False,'reserved_rgb_in_fit_or_tuning':0,'locked_test_integrity_hashed_before_fit':True})
    o.parent.check_hashes(protocol['protected_sources'])
    print('Frozen validation/Brier winner',chosen['name'],chosen['validation_brier'],flush=True)


def evaluate():
    original=o.parent.load;o.parent.load=load
    try:o.parent.evaluate(OUTPUT)
    finally:o.parent.load=original


def export():
    protocol=o.checked(OUTPUT);chosen=o.parent.read(OUTPUT/'selection.json')['selected'];collection=o.parent.read(o.parent.COLLECTION/'collection-manifest.json');old=o.parent.read(CODE/'experiments/public_cpr_improved/inference-manifest.json')
    pinned=[r for r in old['pinned_files'] if r['path'] not in ['data/public_cpr/improvement-v2/combined-64.pt','data/public_cpr/improvement-v2/protocol.json']]
    for path in [OUTPUT/chosen['checkpoint'],OUTPUT/'protocol.json',OUTPUT/'selection.json',OUTPUT/'record-splits.json',Path(__file__).resolve(),Path(__file__).with_name('compact_head.py'),Path(__file__).with_name('infer_compact_video.py')]:pinned.append({'path':path.relative_to(CODE).as_posix(),'sha256':o.parent.digest(path)})
    manifest={**old,'frozen_at':datetime.now(timezone.utc).isoformat(),'head':(OUTPUT/chosen['checkpoint']).relative_to(CODE).as_posix(),'protocol':(OUTPUT/'protocol.json').relative_to(CODE).as_posix(),'pinned_files':pinned,'head_kind':'compact','entrypoint':'experiments/public_rgb_dense_head/infer_compact_video.py',
        'public_samples':{r['video_sha256']:{'record_id':r['record_id'],'video':(o.parent.COLLECTION/r['video']).relative_to(CODE).as_posix(),'role':'previously_exposed_engineering_regression'} for r in collection['records']},
        'disclosure':protocol['previous_test_exposure'],'runtime_release_approved':False,'raw_video_acceptance_pending':True}
    manifest.pop('raw_test_record',None);manifest.pop('raw_test_video_sha256',None)
    o.write(OUTPUT/'inference-manifest.json',manifest);o.parent.load_manifest(OUTPUT/'inference-manifest.json')
    print('Compact offline manifest validated; web/Agent unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','evaluate','export']);args=parser.parse_args();globals()[args.stage]()
