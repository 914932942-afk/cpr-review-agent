"""Deterministic COCO17 descriptors: no learned extractor or extra annotations."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

JOINTS=17
PROFILE={"version":1,"layout":"COCO17","visible_score":.2,"maximum_internal_gap_frames":5,
         "origin":"mid hip; mid shoulder fallback","scale":"median shoulder-to-hip length",
         "statistics":["mean","std","p10","p90"],"time":"original frame ordinal, NOT BPM or physical rate"}
DIMENSION=271
ANGLES=((5,7,9),(6,8,10),(11,13,15),(12,14,16),(7,5,11),(8,6,12),(5,11,13),(6,12,14))
PAIRS=((9,10),(7,8),(5,6),(11,12),(9,11),(10,12),(9,13),(10,14))


def repair(points,scores):
    p=np.asarray(points,dtype=np.float64).copy();q=np.asarray(scores,dtype=np.float64)
    if p.ndim!=3 or p.shape[1:]!=(17,2) or q.shape!=p.shape[:2] or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Expected finite [T,17,2] coordinates and [T,17] scores")
    if len(p)<8:raise ValueError("Too short for a temporal descriptor")
    valid=q>=PROFILE["visible_score"]
    p[~valid]=np.nan
    interpolated=0
    for joint in range(17):
        positions=np.flatnonzero(valid[:,joint])
        for left,right in zip(positions[:-1],positions[1:]):
            gap=right-left-1
            if 0<gap<=PROFILE["maximum_internal_gap_frames"]:
                for offset in range(1,gap+1):
                    ratio=offset/(gap+1);p[left+offset,joint]=(1-ratio)*p[left,joint]+ratio*p[right,joint]
                    interpolated+=1
    return p,valid,interpolated


def statistics(value):
    if value.ndim==1:value=value[:,None]
    # Explicit support counts avoid filling long missing intervals with motion.
    result=[]
    for column in value.T:
        available=column[np.isfinite(column)]
        result.extend([float(available.mean()),float(available.std()),*np.percentile(available,[10,90]).tolist()] if len(available) else [0.]*4)
    return result


def describe(points,scores):
    p,observed,repaired=repair(points,scores)
    hip=(p[:,11]+p[:,12])/2;shoulder=(p[:,5]+p[:,6])/2
    hip_valid=np.isfinite(hip).all(1);shoulder_valid=np.isfinite(shoulder).all(1)
    use_hip=hip_valid.sum()>=8
    origin=hip if use_hip else shoulder
    scales=np.linalg.norm(hip-shoulder,axis=1)
    available=scales[np.isfinite(scales)&(scales>1e-6)]
    if len(available)<8:
        widths=np.linalg.norm(p[:,5]-p[:,6],axis=1)
        available=widths[np.isfinite(widths)&(widths>1e-6)]
    if not len(available) or np.isfinite(origin).all(1).sum()<8:
        return np.zeros(DIMENSION,dtype=np.float32),{"geometry_valid":False,"visible_joint_fraction":float(observed.mean()),
                "short_gap_points_repaired":repaired,"reason":"No reliable torso/shoulder reference; descriptor unavailable, not fabricated"}
    scale=float(np.median(available))
    normalized=(p-origin[:,None,:])/scale
    values=statistics(normalized.reshape(len(p),-1))
    velocity=np.diff(normalized,axis=0)
    for column in velocity.reshape(len(velocity),-1).T:
        a=column[np.isfinite(column)]
        values.extend([float(np.abs(a).mean()),float(a.std())] if len(a) else [0.,0.])
    angles=[]
    for a,b,c in ANGLES:
        u=normalized[:,a]-normalized[:,b];v=normalized[:,c]-normalized[:,b]
        denominator=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
        angle=np.sum(u*v,axis=1)/np.maximum(denominator,1e-8)
        angle[(denominator<1e-8)|~np.isfinite(denominator)]=np.nan
        angles.append(np.clip(angle,-1,1))
    values+=statistics(np.stack(angles,1))
    distances=np.stack([np.linalg.norm(normalized[:,a]-normalized[:,b],axis=1) for a,b in PAIRS],1)
    values+=statistics(distances)
    values.extend([1.,float(use_hip),float(not use_hip)])
    # Observability is reported, but not included as a label shortcut.
    vector=np.asarray(values,dtype=np.float32)
    if vector.shape!=(DIMENSION,) or not np.isfinite(vector).all():raise ValueError("Invalid geometric descriptor")
    return vector,{"geometry_valid":True,"visible_joint_fraction":float(observed.mean()),"short_gap_points_repaired":repaired,
                   "torso_scale_pixels":scale,"long_missing_points_preserved_as_missing":True}


def main():
    source=(Path(__file__).resolve().parents[2] / "data/public_cpr/run")
    output=Path(__file__).resolve().parent
    records=json.loads((source/"records.json").read_text(encoding="utf-8"));vectors=[];quality=[]
    for r in records:
        with np.load(source/r["record_file"],allow_pickle=False) as f:
            vector,q=describe(f["keypoint"][0],f["keypoint_score"][0])
        vectors.append(vector);quality.append(q)
    np.savez_compressed(output/"geometry-bundle.npz",features=np.stack(vectors),record_ids=np.array([r["record_id"] for r in records]),
                        profile_hash=np.array(hashlib.sha256(json.dumps(PROFILE,sort_keys=True).encode()).hexdigest()))
    (output/"geometry-audit.json").write_text(json.dumps({"profile":PROFILE,"records":len(records),"dimension":len(vectors[0]),"quality":quality},indent=2))
    print("Geometry feature bundle",len(records),len(vectors[0]),flush=True)


if __name__=="__main__":main()
