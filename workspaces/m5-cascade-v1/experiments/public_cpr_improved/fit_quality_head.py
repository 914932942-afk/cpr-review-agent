"""Frozen splits + geometric descriptors + bounded small tree head."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from improve import GATES, LABEL_IDS, REVISION, SOURCE, digest, metrics, predict, read, tune_operating_point, write_json
from geometry import PROFILE
from quality_head import export_classifier,load

BASE=(Path(__file__).resolve().parents[2] / "data/public_cpr/improvement-v1")
GEOMETRY=Path(__file__).resolve().parent/"geometry-bundle.npz"


def prepare(output):
    if (output/"protocol.json").exists():raise FileExistsError("Protocol already frozen")
    source=read(BASE/"protocol.json")
    with np.load(GEOMETRY,allow_pickle=False) as f:
        lookup={str(k):v for k,v in zip(f["record_ids"],f["features"])}
        geometry_hash=str(f["profile_hash"])
    profile={"base_encoder_profile_hash":source["encoder_profile_hash"],"geometry":PROFILE,
             "geometry_profile_hash":geometry_hash,"feature_order":"PoseC3D512 then COCO17 geometry271",
             "input_dimension":783,"training_scope":"classification head only; all perception backbones frozen"}
    profile_hash=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    for name in ("development","locked-test"):
        with np.load(BASE/f"{name}.npz",allow_pickle=False) as f:
            data={k:f[k].copy() for k in f.files}
        data["features"]=np.concatenate([data["features"],np.stack([lookup[str(k)] for k in data["record_ids"]])],1)
        data["profile_hash"]=np.array(profile_hash)
        np.savez_compressed(output/f"{name}.npz",**data)
    (output/"record-splits.json").write_bytes((BASE/"record-splits.json").read_bytes())
    configs=[{"features":features,"trees":trees} for features in ("geometry","combined") for trees in (32,64)]
    protocol={**source,"parent_protocol_sha256":digest(BASE/"protocol.json"),"feature_profile":profile,
              "feature_profile_hash":profile_hash,"gates":GATES,"candidate_configs":configs,
              "development_sha256":digest(output/"development.npz"),"locked_test_sha256":digest(output/"locked-test.npz"),
              "development_search_disclosure":"Linear/MLP, nearest-neighbor and Nystroem alternatives were explored on development only. Test data were not evaluated during this search.",
              "head_budget_definition":"All exported scalar node fields plus biases, including integer topology and boolean flags, <=70000. No hidden support-vector/training-example bank."}
    write_json(output/"protocol.json",protocol)
    print("Frozen geometry protocol",protocol["split_counts"],"profile",profile_hash,flush=True)


def train(output):
    if (output/"selection.json").exists():raise FileExistsError("Model selection is frozen")
    protocol=read(output/"protocol.json")
    if digest(output/"development.npz")!=protocol["development_sha256"]:raise ValueError("Development changed")
    with np.load(output/"development.npz",allow_pickle=False) as f:
        if not np.array_equal(f['label_ids'],LABEL_IDS) or str(f['profile_hash'])!=protocol['feature_profile_hash']:
            raise ValueError('Development label order or feature profile mismatch')
        x=f["features"].copy();y=f["targets"].copy();fit=f["splits"]=="train";val=f["splits"]=="validation";ids=f['record_ids'].copy()
    xf=x[fit];yf=y[fit]
    if 'augmented_training_sha256' in protocol:
        if digest(output/'augmented-training.npz')!=protocol['augmented_training_sha256']:raise ValueError('Training augmentation changed')
        with np.load(output/'augmented-training.npz',allow_pickle=False) as f:
            if not set(map(str,f['record_ids'])).issubset(set(map(str,ids[fit]))):raise ValueError('Augmentation contains held-out records')
            xf=np.concatenate([xf,f['features']]);yf=np.concatenate([yf,f['targets']])
    target=np.column_stack([yf,~yf.any(1)])
    candidates=[]
    for config in protocol["candidate_configs"]:
        started=time.perf_counter();mask=np.arange(512,x.shape[1]) if config["features"]=="geometry" else np.arange(x.shape[1])
        outputs=[];reference=[]
        with threadpool_limits(limits=4):
            for label in range(12):
                head=HistGradientBoostingClassifier(max_iter=config["trees"],max_leaf_nodes=7,max_depth=3,
                    min_samples_leaf=8,l2_regularization=1.,learning_rate=.1,early_stopping=False,random_state=20261021)
                head.fit(xf[:,mask],target[:,label])
                reference.append(head.predict_proba(x[val][:,mask])[:,1]);outputs.append(export_classifier(head))
        probability=np.stack(reference,1);operating=tune_operating_point(probability,y[val])
        state={"schema":"public-cpr-geometry-head-v1","input_dimension":len(mask),"feature_indices":torch.from_numpy(mask),
               "classifiers":outputs,"label_ids":list(LABEL_IDS),"source_revision":REVISION,
               "feature_profile_hash":protocol["feature_profile_hash"],"protocol_sha256":digest(output/"protocol.json"),
               "operating_point":operating,"score_semantics":"uncalibrated classifier scores with validation-selected operating thresholds"}
        field_count=12+sum(t.numel() for o in outputs for tree in o["trees"] for t in tree.values())
        if field_count>GATES["maximum_head_parameters"]:raise AssertionError("Head scalar budget exceeded")
        path=output/f"{config['features']}-{config['trees']}.pt";torch.save(state,path)
        restored,_=load(path,protocol["feature_profile_hash"],digest(output/"protocol.json"))
        difference=float(np.abs(restored.probability(x[val][:,mask])-probability).max())
        if difference>1e-10:raise AssertionError(f"Portable predictor differs from sklearn: {difference}")
        row={"config":config,"scalar_fields":field_count,"checkpoint":path.name,"checkpoint_sha256":digest(path),
             "checkpoint_bytes":path.stat().st_size,"seconds":time.perf_counter()-started,
             "portable_vs_training_max_difference":difference,"operating_point":operating}
        if protocol.get('raw_development_record'):
            raw_mask=ids[val]==protocol['raw_development_record']
            raw_pred=predict(probability[raw_mask],operating['thresholds'],operating['normal_gate'])
            row['raw_development_exact_match']=bool((raw_pred==y[val][raw_mask]).all())
        candidates.append(row)
        print(config,"val macroF1",operating["validation_metrics"]["macro_f1"],"export difference",difference,"fields",field_count,flush=True)
    eligible=[c for c in candidates if c.get('raw_development_exact_match',True)]
    if not eligible:
        write_json(output/'development-failure.json',{'all_candidates':candidates,'test_opened':False})
        raise RuntimeError('No candidate passed the disclosed raw development example; test remains locked')
    winner=max(eligible,key=lambda c:(c["operating_point"]["validation_metrics"]["macro_f1"],-c["scalar_fields"]))
    write_json(output/"selection.json",{"selected":winner,"all_candidates":candidates,"protocol_sha256":digest(output/"protocol.json"),
                                      "test_read_during_fit_or_selection":False,"main_network_training":False})


def evaluate(output):
    if (output/"evaluation.json").exists():raise FileExistsError("Preserve first frozen test evaluation")
    protocol=read(output/"protocol.json");selection=read(output/"selection.json");chosen=selection["selected"]
    if digest(output/"locked-test.npz")!=protocol["locked_test_sha256"]:raise ValueError("Test changed")
    path=output/chosen["checkpoint"]
    if digest(path)!=chosen["checkpoint_sha256"]:raise ValueError("Selected model changed")
    head,state=load(path,protocol["feature_profile_hash"],digest(output/"protocol.json"))
    with np.load(output/"locked-test.npz",allow_pickle=False) as f:
        if not np.array_equal(f['label_ids'],LABEL_IDS) or str(f['profile_hash'])!=protocol['feature_profile_hash']:
            raise ValueError('Test label order or feature profile mismatch')
        x=f["features"].copy();y=f["targets"].copy();cohorts=f["cohorts"].copy();ids=f["record_ids"].copy()
    p=head.probability(x[:,state["feature_indices"].numpy()])
    op=state["operating_point"];pred=predict(p,op["thresholds"],op["normal_gate"])
    evaluations={}
    for name,mask in (("all",np.ones(len(y),bool)),("single",cohorts=="single"),("composite",cohorts=="composite")):
        m=metrics(pred[mask].astype(float),y[mask]);rank=metrics(p[mask,:11],y[mask])
        m["mAP"]=rank["mAP"]
        for entry,r in zip(m["per_label"],rank["per_label"]):entry["average_precision"]=r["average_precision"]
        evaluations[name]=m
    quality={"single_f1":evaluations["single"]["macro_f1"]>=GATES["single_macro_f1"],
             "composite_f1":evaluations["composite"]["macro_f1"]>=GATES["composite_macro_f1"],
             "correct_false_alarm":evaluations["single"]["correct_false_alarm_rate"]<=GATES["correct_false_alarm_rate"],
             "all_label_recall":min(r["recall"] for r in evaluations["all"]["per_label"])>=GATES["minimum_label_recall"],
             "coverage":True,"head_budget":chosen["scalar_fields"]<=GATES["maximum_head_parameters"]}
    from baseline import load_checked_head
    old,old_meta=load_checked_head(SOURCE/"mlp-head.pt",protocol["encoder_profile_hash"])
    with torch.inference_mode():old_p=old(torch.from_numpy(x[:,:512])).sigmoid().numpy()
    old_train={r["record_id"] for r in read(BASE/"record-splits.json") if r["split"]=="train"}
    comparison={}
    for cohort in ("single","composite"):
        mask=(cohorts==cohort)&np.array([str(k) not in old_train for k in ids])
        comparison[cohort]={"old":metrics(old_p[mask],y[mask],old_meta["threshold"]),"improved":metrics(pred[mask].astype(float),y[mask])}
    np.savez_compressed(output/"test-predictions.npz",scores=p,predictions=pred,targets=y,record_ids=ids)
    result={"quality_checks":quality,"evaluations":evaluations,"public_keypoint_quality_passed":all(quality.values()),
            "full_coverage":True,"rejected_test_records":0,"test_used_for_selection":False,
            "old_model_same_test_comparison":comparison,"raw_video_quality_passed":False,"step_passed":False,
            "protocol_sha256":digest(output/"protocol.json"),"selection_sha256":digest(output/"selection.json")}
    write_json(output/"evaluation.json",result)
    print(json.dumps({"quality":quality,"summary":{k:{n:m[n] for n in ('records','macro_f1','micro_f1','correct_false_alarm_rate')} for k,m in evaluations.items()}},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["prepare","train","evaluate"])
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();globals()[args.stage](args.output)


if __name__=="__main__":main()
