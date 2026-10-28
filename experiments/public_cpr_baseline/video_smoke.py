"""Raw-video interface smoke test; local unlabelled clips are NEVER training data."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline import FrozenEncoder, LABEL_IDS, LABEL_NAMES, digest, encoder_profile, load_checked_head, write_json

POSE_HASH="c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212"


def load_pose(path):
    from ultralytics import YOLO
    if digest(path) != POSE_HASH:
        raise ValueError("Unverified local YOLO-Pose weights")
    # This exact trusted older checkpoint contains an Ultralytics model object.
    # Restrict the compatibility override to this file and constructor only.
    original=torch.load
    def load(source,*args,**kwargs):
        if isinstance(source,(str,Path)) and Path(source).resolve()==path.resolve():
            kwargs.setdefault("weights_only",False)
        return original(source,*args,**kwargs)
    torch.load=load
    try:
        return YOLO(str(path))
    finally:
        torch.load=original


def decode_pose(video, model):
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    keypoints,counts,detected=[],[],0
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    height=width=None
    try:
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            if len(keypoints)>=3000:
                raise ValueError("Smoke test accepts at most 3000 sequential frames")
            height,width=frame.shape[:2]
            result=model.predict(frame,device="cuda:0",verbose=False)[0]
            count=0 if result.boxes is None else len(result.boxes)
            counts.append(count)
            if not count or result.keypoints is None:
                points=np.zeros((17,3),dtype=np.float32)
            else:
                index=int(result.boxes.conf.argmax())
                points=result.keypoints.data[index].cpu().numpy().astype(np.float32)
                detected+=1
            if points.shape!=(17,3) or not np.isfinite(points).all():
                raise ValueError("Invalid pose output")
            keypoints.append(points)
    finally:
        cap.release()
    if not keypoints or detected/len(keypoints)<.5:
        raise ValueError("Too few frames with a detected person; do not emit error scores")
    points=np.stack(keypoints)
    return {"keypoint":points[None,:,:,:2],"keypoint_score":np.clip(points[None,:,:,2],0,1),
            "img_shape":(height,width)}, {"decoded_frames":len(points),"reported_fps":fps,
            "frames_with_person":detected,"frames_with_multiple_people":sum(n>1 for n in counts),
            "keypoint_fraction_above_025":float((points[:,:,2]>.25).mean()),
            "person_selection":"highest detector box confidence; no verified trainee/manikin association"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--video",type=Path,action="append",required=True)
    parser.add_argument("--backbone",type=Path,required=True)
    parser.add_argument("--pose",type=Path,required=True)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--feature-only",action="store_true")
    args=parser.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    pose=load_pose(args.pose)
    encoder=FrozenEncoder(args.backbone)
    profile_path=args.run/"encoder-profile.json"
    head=None
    if not args.feature_only:
        profile=json.loads(profile_path.read_text(encoding="utf-8"))
        _,actual_profile_hash=encoder_profile(encoder)
        if profile["profile_hash"] != actual_profile_hash:
            raise ValueError("Current encoder preprocessing differs from the trained head profile")
        results=json.loads((args.run/"head-training-results.json").read_text(encoding="utf-8"))
        head,checkpoint=load_checked_head(args.run/f"{results['selected_by_validation']}-head.pt",profile["profile_hash"])
    reports=[]
    for video in args.video:
        started=time.perf_counter()
        record,details=decode_pose(video,pose)
        feature=encoder.encode([record])
        repeated=encoder.encode([record])
        error=float(np.abs(feature-repeated).max())
        if error>1e-5:
            raise AssertionError("Unstable repeated raw-video feature inference")
        report={"video":str(video.resolve()),"video_sha256":digest(video),**details,
                "feature_shape":list(feature.shape),"repeat_max_difference":error,
                "manual_label":None,"used_for_training":False,"recognition_accuracy_measured":False,
                "limitation":"Public training poses use AlphaPose Halpe68; this input uses YOLO COCO17. Their accuracy/domain compatibility is NOT established by an unlabelled smoke test."}
        np.savez_compressed(args.run/f"{video.stem}-input-smoke.npz",feature=feature,**record)
        if head is not None:
            with torch.inference_mode():
                scores=head(torch.from_numpy(feature)).sigmoid().numpy()[0]
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite raw-video head output")
            report["uncalibrated_scores"]=[{"original_label_id":i,"name":n,"score":float(s)} for i,n,s in zip(LABEL_IDS,LABEL_NAMES,scores)]
            report["head_kind"]=checkpoint["kind"]
            report["runtime_release_approved"]=False
        report["elapsed_seconds"]=time.perf_counter()-started
        reports.append(report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    write_json(args.run/("video-feature-preflight.json" if args.feature_only else "raw-video-smoke.json"),reports)


if __name__=="__main__":
    main()
