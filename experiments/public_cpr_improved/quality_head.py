"""Portable small tree classifier; only trusted locally-produced tensor states.

No sklearn dependency at inference and no executable model-object pickle.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

EXPERIMENT=(Path(__file__).resolve().parents[2] / "experiments/public_cpr_baseline")
sys.path.insert(0,str(EXPERIMENT))
from baseline import LABEL_IDS, REVISION


def export_classifier(classifier):
    trees=[]
    for iteration in classifier._predictors:
        n=iteration[0].nodes
        if n["is_categorical"].any():raise ValueError("Only numeric trees are supported")
        trees.append({"feature":torch.from_numpy(n["feature_idx"].astype(np.int32)),
                      "threshold":torch.from_numpy(n["num_threshold"].copy()),
                      "left":torch.from_numpy(n["left"].astype(np.int32)),
                      "right":torch.from_numpy(n["right"].astype(np.int32)),
                      "value":torch.from_numpy(n["value"].copy()),
                      "leaf":torch.from_numpy(n["is_leaf"].astype(bool)),
                      "missing_left":torch.from_numpy(n["missing_go_to_left"].astype(bool))})
    return {"bias":float(classifier._baseline_prediction[0,0]),"trees":trees}


class PortableHead:
    def __init__(self,state):
        self.dimension=state["input_dimension"]
        self.outputs=[]
        for output in state["classifiers"]:
            self.outputs.append({"bias":output["bias"],"trees":[{k:v.numpy() for k,v in tree.items()} for tree in output["trees"]]})

    def probability(self,feature):
        x=np.asarray(feature,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=self.dimension or not np.isfinite(x).all():
            raise ValueError("Invalid geometry/PoseC3D feature contract")
        result=[]
        for output in self.outputs:
            raw=np.full(len(x),output["bias"],dtype=np.float64)
            for tree in output["trees"]:
                node=np.zeros(len(x),dtype=np.int32)
                active=np.ones(len(x),bool)
                steps=0
                while active.any():
                    rows=np.flatnonzero(active)
                    at_leaf=tree["leaf"][node[rows]]
                    leaves=rows[at_leaf]
                    raw[leaves]+=tree["value"][node[leaves]]
                    active[leaves]=False
                    moving=rows[~at_leaf]
                    if len(moving):
                        current=node[moving]
                        value=x[moving,tree["feature"][current]]
                        go_left=np.where(np.isnan(value),tree["missing_left"][current],value<=tree["threshold"][current])
                        node[moving]=np.where(go_left,tree["left"][current],tree["right"][current])
                    steps+=1
                    if steps>16:raise ValueError("Malformed/cyclic tree")
            result.append(1/(1+np.exp(-np.clip(raw,-60,60))))
        return np.stack(result,1)


def load(path,profile_hash,protocol_hash):
    state=torch.load(path,map_location="cpu",weights_only=True)
    if (state.get("schema")!="public-cpr-geometry-head-v1" or state.get("label_ids")!=list(LABEL_IDS)
        or state.get("source_revision")!=REVISION or state.get("feature_profile_hash")!=profile_hash
        or state.get("protocol_sha256")!=protocol_hash or len(state.get("classifiers",[]))!=12):
        raise ValueError("Incompatible classifier labels/profile/protocol")
    return PortableHead(state),state
