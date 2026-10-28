"""Application protocol: public composite labels, frozen encoder, small heads.

The earlier zero-shot-composition protocol remains unchanged on disk. This is
a declared new application protocol, NOT a reproduction of that benchmark.
Test observations from the earlier experiment are explicitly disclosed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

EXPERIMENT=(Path(__file__).resolve().parents[2] / "experiments/public_cpr_baseline")
sys.path.insert(0,str(EXPERIMENT))
from baseline import LABEL_IDS, LABEL_NAMES, REVISION, digest, metrics, write_json

SOURCE=(Path(__file__).resolve().parents[2] / "data/public_cpr/run")
SEED=20261021
CONFIGS=[{"name":f"{kind}_{weight}","hidden":hidden,"weight":weight}
         for kind,hidden in (("linear",0),("mlp64",64),("mlp128",128)) for weight in ("none","sqrt")]
GATES={"single_macro_f1":.85,"composite_macro_f1":.80,"correct_false_alarm_rate":.10,
       "minimum_label_recall":.60,"classification_coverage":1.,"maximum_head_parameters":70000,
       "raw_public_labelled_clips_minimum":1}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def grouped_split(records):
    result=[dict(r) for r in records]
    groups=defaultdict(list)
    for r in result:
        if r["original_partition"]=="official_composite_test":
            groups[tuple(r["original_labels"])].append(r)
        else:
            r["application_split"]=r["split"] if r["split"] in {"train","validation"} else "test"
            r["test_cohort"]="single"
        # The first available RGB archive begins with this class. Reserve its
        # two recordings before fitting any new model, for a labelled raw test.
        if r["record_id"].startswith("SupC02/"):
            r["application_split"]="test"
            r["test_cohort"]="single"
    rng=np.random.default_rng(SEED)
    for pattern in sorted(groups):
        group=sorted(groups[pattern],key=lambda r:r["record_id"])
        if len(group) not in {8,12}:
            raise ValueError("Unexpected public composite pattern size")
        order=rng.permutation(len(group))
        for position,index in enumerate(order):
            group[index]["application_split"]="test" if position<2 else "validation" if position<4 else "train"
            group[index]["test_cohort"]="composite"
    # Extra existing Correct examples stabilize the operating point. They are
    # moved from original training, never from the original test partition.
    normals=sorted([r for r in result if r["application_split"]=="train" and r["original_labels"]==[0]],key=lambda r:r["record_id"])
    for index in np.random.default_rng(SEED+1).permutation(len(normals))[:4]:
        normals[index]["application_split"]="validation"
    ids=[r["record_id"].rsplit("/",1)[0] for r in result]
    if len(set(ids))!=len(ids):
        raise ValueError("Recording group duplicates; camera variants must never cross folds")
    for r in result:
        if r["original_partition"]=="official_single_test" and r["application_split"]!="test":
            raise ValueError("Original single test cannot move into development")
    return result


def prepare(output):
    if (output/"protocol.json").exists():
        raise FileExistsError("Protocol already frozen; do not silently rewrite its partition")
    records=grouped_split(read(SOURCE/"records.json"))
    with np.load(SOURCE/"feature-bundle.npz",allow_pickle=False) as f:
        if list(f["record_ids"])!=[r["record_id"] for r in records] or not np.array_equal(f["label_ids"],LABEL_IDS):
            raise ValueError("Frozen feature/record order differs")
        feature=f["features"].copy();target=f["targets"].copy();profile=str(f["profile_hash"])
    source_audit=read(SOURCE/"data-audit.json")
    if source_audit["missing_keypoint_records"] or source_audit["rejected_records"]:
        raise ValueError("Source input audit is incomplete")
    if read(SOURCE/"input-duplicate-audit.json")["exact_duplicate_groups"]:
        raise ValueError("Exact input duplicates need grouping before a new protocol")
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"record-splits.json",records)
    for name,is_test in (("development",False),("locked-test",True)):
        mask=np.array([(r["application_split"]=="test")==is_test for r in records])
        np.savez_compressed(output/f"{name}.npz",features=feature[mask],targets=target[mask],
                            record_ids=np.array([r["record_id"] for r in records])[mask],
                            splits=np.array([r["application_split"] for r in records])[mask],
                            cohorts=np.array([r["test_cohort"] for r in records])[mask],
                            label_ids=np.array(LABEL_IDS),profile_hash=np.array(profile))
    protocol={"schema":1,"source_revision":REVISION,"source_bundle_sha256":digest(SOURCE/"feature-bundle.npz"),
              "source_records_sha256":digest(SOURCE/"records.json"),"split_sha256":digest(output/"record-splits.json"),
              "development_sha256":digest(output/"development.npz"),"locked_test_sha256":digest(output/"locked-test.npz"),
              "seed":SEED,"encoder_profile_hash":profile,"gates":GATES,"candidate_configs":CONFIGS,
              "split_counts":dict(Counter(r["application_split"] for r in records)),
              "test_cohorts":dict(Counter(r["test_cohort"] for r in records if r["application_split"]=="test")),
              "normal_counts":dict(Counter(r["application_split"] for r in records if r["original_labels"]==[0])),
              "protocol_change":"Public composite patterns may occur in development and test, but complete recordings are disjoint. This is application recognition, not unseen-composition generalization.",
              "previous_test_exposure":"The earlier baseline reported aggregate metrics on these public samples. This partition is frozen before the new model fit, but is not claimed to be a never-inspected external blind test.",
              "human_collection_or_annotation":False,"main_network_training":False,"uses_multiview_fusion":False,
              "subject_disjoint_claim":False,"source_view":"ch0","raw_test_reservations":["SupC02/r00/ch0","SupC02/r01/ch0"]}
    write_json(output/"protocol.json",protocol)
    print(json.dumps(protocol,ensure_ascii=False,indent=2),flush=True)


class ApplicationHead(nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.hidden=hidden
        self.register_buffer("mean",torch.zeros(512))
        self.register_buffer("scale",torch.ones(512))
        self.layers=nn.Linear(512,12) if hidden==0 else nn.Sequential(nn.Linear(512,hidden),nn.ReLU(),nn.Dropout(.2),nn.Linear(hidden,12))

    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=512 or not torch.isfinite(x).all():
            raise ValueError("Requires finite [N,512] profile-compatible features")
        return self.layers((x-self.mean)/self.scale)


def predict(probability,thresholds,normal_gate):
    # No rejected examples: all records remain in every quality denominator.
    errors=probability[:,:11]>=np.asarray(thresholds)
    if normal_gate is not None:
        errors &= (probability[:,11]<normal_gate)[:,None]
    return errors


def tune_operating_point(probability,target):
    normal=~target.any(1)
    candidates=[]
    for normal_gate in (None,.3,.5,.7,.9):
        thresholds=[]
        for k in range(11):
            choices=[]
            for threshold in np.linspace(.1,.95,35):
                pred=predict(probability,[threshold]*11,normal_gate)[:,k]
                if (pred & normal).any():
                    continue
                truth=target[:,k].astype(bool)
                tp=(pred & truth).sum();fp=(pred & ~truth).sum();fn=(~pred & truth).sum()
                score=2*tp/max(2*tp+fp+fn,1)
                choices.append((float(score),-abs(threshold-.5),float(threshold)))
            thresholds.append(max(choices)[2] if choices else 1.)
        pred=predict(probability,thresholds,normal_gate)
        result=metrics(pred.astype(float),target,.5)
        candidates.append((result["macro_f1"],result["micro_f1"],normal_gate,thresholds,result))
    best=max(candidates,key=lambda r:(r[0],r[1]))
    return {"thresholds":best[3],"normal_gate":best[2],"validation_metrics":best[4]}


def train(output):
    torch.set_num_threads(4)
    protocol=read(output/"protocol.json")
    if (output/"selection.json").exists():
        raise FileExistsError("Selection is already frozen")
    if digest(output/"development.npz")!=protocol["development_sha256"]:
        raise ValueError("Development data changed")
    with np.load(output/"development.npz",allow_pickle=False) as f:
        x=torch.tensor(f["features"]);y=torch.tensor(f["targets"])
        fit=f["splits"]=="train";val=f["splits"]=="validation"
    xf,yf=x[fit],y[fit];xv,yv=x[val],y[val]
    results=[]
    for config in protocol["candidate_configs"]:
        torch.manual_seed(SEED)
        model=ApplicationHead(config["hidden"])
        model.mean.copy_(xf.mean(0));model.scale.copy_(xf.std(0,unbiased=False).clamp_min(1e-4))
        target=torch.cat([yf,(yf.sum(1)==0).float()[:,None]],1)
        target_val=torch.cat([yv,(yv.sum(1)==0).float()[:,None]],1)
        weight=(len(yf)-yf.sum(0))/yf.sum(0).clamp_min(1)
        weight=weight.sqrt() if config["weight"]=="sqrt" else torch.ones_like(weight)
        normal_weight=(len(yf)-target[:,-1].sum())/target[:,-1].sum()
        positive_weight=torch.cat([weight,normal_weight[None]])
        def loss(logits,truth):
            values=torch.nn.functional.binary_cross_entropy_with_logits(logits,truth,pos_weight=positive_weight,reduction="none")
            return values[:,:11].mean()+.25*values[:,11].mean()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.003)
        best=float("inf");state=None;stale=0;history=[];started=time.perf_counter()
        for epoch in range(1,1001):
            model.train();opt.zero_grad(set_to_none=True)
            train_loss=loss(model(xf),target);train_loss.backward();opt.step()
            model.eval()
            with torch.inference_mode():value=float(loss(model(xv),target_val))
            if not np.isfinite(value):raise ValueError("Non-finite development loss")
            history.append(value)
            if value<best-1e-5:
                best=value;state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0;best_epoch=epoch
            else:stale+=1
            if stale>=60:break
        model.load_state_dict(state);model.eval()
        with torch.inference_mode():p=model(xv).sigmoid().numpy()
        operating=tune_operating_point(p,yv.numpy())
        checkpoint={"schema":1,"hidden":config["hidden"],"head_state":model.state_dict(),"label_ids":list(LABEL_IDS),
                    "encoder_profile_hash":protocol["encoder_profile_hash"],"source_revision":REVISION,
                    "protocol_sha256":digest(output/"protocol.json"),"operating_point":operating,
                    "score_semantics":"uncalibrated scores; validation-selected decision thresholds", "config":config}
        path=output/(config["name"]+".pt");torch.save(checkpoint,path)
        restored,_=load_head(path,protocol["encoder_profile_hash"],checkpoint["protocol_sha256"])
        with torch.inference_mode():
            if not torch.equal(model(xv),restored(xv)):raise AssertionError("Head reload changed output")
        row={"config":config,"parameters":sum(v.numel() for v in model.parameters()),"best_epoch":best_epoch,
             "seconds":time.perf_counter()-started,"checkpoint":path.name,"checkpoint_sha256":digest(path),
             "operating_point":operating,"validation_loss_history":history}
        results.append(row)
        print(config["name"],"validation macroF1",round(operating["validation_metrics"]["macro_f1"],4),"seconds",round(row["seconds"],2),flush=True)
    winner=max(results,key=lambda r:(r["operating_point"]["validation_metrics"]["macro_f1"],-r["parameters"]))
    if winner["parameters"]>GATES["maximum_head_parameters"]:raise AssertionError("Head exceeds lightweight budget")
    write_json(output/"selection.json",{"protocol_sha256":digest(output/"protocol.json"),"selected":winner,"all_candidates":results,
                                      "test_read_during_fit_or_selection":False,"visual_backbone_trained":False})


def load_head(path,profile_hash,protocol_hash):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    if (checkpoint.get("schema")!=1 or checkpoint.get("label_ids")!=list(LABEL_IDS) or
        checkpoint.get("source_revision")!=REVISION or checkpoint.get("encoder_profile_hash")!=profile_hash or
        checkpoint.get("protocol_sha256")!=protocol_hash):
        raise ValueError("Incompatible head/profile/label order/protocol")
    model=ApplicationHead(checkpoint["hidden"]).eval()
    model.load_state_dict(checkpoint["head_state"],strict=True)
    return model,checkpoint


def evaluate(output):
    if (output/"evaluation.json").exists():
        raise FileExistsError("The frozen test was already evaluated; preserve the result")
    protocol=read(output/"protocol.json");selection=read(output/"selection.json")
    if digest(output/"locked-test.npz")!=protocol["locked_test_sha256"]:
        raise ValueError("Locked test changed")
    chosen=selection["selected"];path=output/chosen["checkpoint"]
    if digest(path)!=chosen["checkpoint_sha256"]:raise ValueError("Selected head changed after selection")
    model,checkpoint=load_head(path,protocol["encoder_profile_hash"],digest(output/"protocol.json"))
    with np.load(output/"locked-test.npz",allow_pickle=False) as f:
        x=torch.tensor(f["features"]);y=f["targets"].copy();cohorts=f["cohorts"].copy();ids=f["record_ids"].copy()
    with torch.inference_mode():p=model(x).sigmoid().numpy()
    operating=checkpoint["operating_point"];prediction=predict(p,operating["thresholds"],operating["normal_gate"])
    evaluations={name:metrics(prediction[mask].astype(float),y[mask]) for name,mask in
                 (("all",np.ones(len(y),bool)),("single",cohorts=="single"),("composite",cohorts=="composite"))}
    quality={"single_f1":evaluations["single"]["macro_f1"]>=GATES["single_macro_f1"],
             "composite_f1":evaluations["composite"]["macro_f1"]>=GATES["composite_macro_f1"],
             "correct_false_alarm":evaluations["single"]["correct_false_alarm_rate"]<=GATES["correct_false_alarm_rate"],
             "all_label_recall":min(r["recall"] for r in evaluations["all"]["per_label"])>=GATES["minimum_label_recall"],
             "coverage":True,"lightweight_head":chosen["parameters"]<=GATES["maximum_head_parameters"]}
    # Old checkpoint on exactly the same held-out cases is a fair model
    # comparison. It was not trained on any new-test composite recordings.
    from baseline import load_checked_head
    old,old_meta=load_checked_head(SOURCE/"mlp-head.pt",protocol["encoder_profile_hash"])
    old_rows=[]
    split=read(output/"record-splits.json")
    old_fit={r["record_id"] for r in split if r["split"]=="train"}
    fair=np.array([str(k) not in old_fit for k in ids])
    with torch.inference_mode():old_p=old(x).sigmoid().numpy()
    comparison={}
    for name,mask in (("single",(cohorts=="single")&fair),("composite",(cohorts=="composite")&fair)):
        comparison[name]={"records":int(mask.sum()),"old":metrics(old_p[mask],y[mask],old_meta["threshold"]),
                          "improved":metrics(prediction[mask].astype(float),y[mask])}
    # Discrete operating decisions above have no meaningful ranking AP. Store
    # genuine error-score AP separately, rather than calling binary AP mAP.
    ranking={name:metrics(p[mask,:11],y[mask])["mAP"] for name,mask in
             (("all",np.ones(len(y),bool)),("single",cohorts=="single"),("composite",cohorts=="composite"))}
    for name in evaluations:
        evaluations[name]["mAP"]=ranking[name]
    write_json(output/"evaluation.json",{"protocol_sha256":digest(output/"protocol.json"),"selection_sha256":digest(output/"selection.json"),
               "evaluations":evaluations,"quality_checks":quality,"public_keypoint_quality_passed":all(quality.values()),
               "raw_video_quality_passed":False,"step_passed":False,"old_model_same_test_comparison":comparison,
               "test_used_to_choose_model_or_thresholds":False,"full_coverage":True,"rejected_test_records":0,
               "test_prior_aggregate_exposure_disclosed":True})
    np.savez_compressed(output/"test-predictions.npz",scores=p,predictions=prediction,targets=y,record_ids=ids)
    print(json.dumps({"quality_checks":quality,"metrics":{k:{n:v[n] for n in ('records','macro_f1','micro_f1','correct_false_alarm_rate')} for k,v in evaluations.items()}},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["prepare","train","evaluate"])
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    {"prepare":prepare,"train":train,"evaluate":evaluate}[args.stage](args.output)


if __name__=="__main__":main()
