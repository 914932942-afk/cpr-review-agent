"""Deterministic local hand-shape evidence in the M1-selected wrist region."""
import cv2
import numpy as np
from head import CODE

DIMENSION=576
PROFILE={'version':1,'roi':'M1-selected distal wrist/forearm ROI, color-to-gray64; top12% overlay excluded',
    'shape':'HOG9 orientations,8x8 cell,8x8 block/stride,64x64 window, temporal mean',
    'dimension':DIMENSION,'minimum_usable_frames':8,'no_label_or_filename_inputs':True,
    'missing_semantics':'Unavailable shape evidence preserves original RGB decisions; never treated as a correct gesture'}


def crop(frame,points,scores):
    if not (np.asarray(scores)[[7,8,9,10]]>=.2).all(): return None
    wrists=points[[9,10]]; forearm=wrists-points[[7,8]]; palms=wrists+.3*forearm
    center=palms.mean(0); torso=np.linalg.norm(points[[5,6]].mean(0)-points[[11,12]].mean(0))
    radius=max(50.,.18*torso,.45*np.linalg.norm(forearm,axis=1).mean(),.75*np.linalg.norm(palms[0]-palms[1]))
    height,width=frame.shape[:2]
    x1,y1=np.maximum(center-radius,[0,height*.12]).astype(int); x2,y2=np.minimum(center+radius,[width,height]).astype(int)
    if x2-x1<8 or y2-y1<8:return None
    gray=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray,(64,64),interpolation=cv2.INTER_AREA)


def extract(frames,points,scores):
    if np.asarray(points).shape!=(len(frames),17,2) or np.asarray(scores).shape!=(len(frames),17):
        raise ValueError('M1 hand-ROI input dimensions differ')
    hog=cv2.HOGDescriptor((64,64),(8,8),(8,8),(8,8),9)
    values=[]
    for frame,p,q in zip(frames,points,scores):
        image=crop(frame,p,q)
        if image is not None:values.append(hog.compute(image).reshape(-1))
    vector=np.mean(values,axis=0) if values else np.zeros(DIMENSION)
    if vector.shape!=(DIMENSION,) or not np.isfinite(vector).all():raise ValueError('Invalid hand shape')
    return vector.astype(np.float32),{'usable_frames':len(values),'requested_frames':len(frames),'available':len(values)>=8,
        'shape_labels_used':False,'keypoints_fabricated':False}
