"""Frozen generic hand landmarks/gestures as observable evidence, not CPR labels."""
from pathlib import Path
import sys
import json
import cv2
import numpy as np
from head import CODE
from descriptor import stats
from baseline import digest

VENDOR=CODE/'data/public_cpr/optional-hand-runtime/vendor'
MODEL=CODE/'data/public_cpr/weights/gesture_recognizer_v1.task'
DIMENSION=120
GESTURES=['None','Closed_Fist','Open_Palm','Pointing_Up','Thumb_Down','Thumb_Up','Victory','ILoveYou']
PROFILE={'version':1,'model':'Official generic MediaPipe gesture_recognizer float16 version1, frozen',
    'hand_limit':2,'detection_confidence':.3,'presence_confidence':.5,'crop':'M1 wrist/forearm distal color384; exclude top12% overlay',
    'features':'five finger curl/distance pairs and angles20, eight gesture scores, hand count and observability; four temporal stats',
    'dimension':DIMENSION,'no_invented_landmarks':True,'labels':'No CPR labels or filenames accepted as feature inputs'}


def crop(frame, points, scores):
    if not (np.asarray(scores)[[7,8,9,10]]>=.2).all(): return None
    wrist=points[[9,10]]; forearm=wrist-points[[7,8]]; palms=wrist+.3*forearm
    center=palms.mean(0); torso=np.linalg.norm(points[[5,6]].mean(0)-points[[11,12]].mean(0))
    radius=max(50.,.18*torso,.45*np.linalg.norm(forearm,axis=1).mean(),.75*np.linalg.norm(palms[0]-palms[1]))
    height,width=frame.shape[:2]
    x1,y1=np.maximum(center-radius,[0,height*.12]).astype(int)
    x2,y2=np.minimum(center+radius,[width,height]).astype(int)
    if x2-x1<8 or y2-y1<8: return None
    return cv2.resize(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB),(384,384),interpolation=cv2.INTER_CUBIC)


def hand_values(points):
    p=np.asarray([[v.x,v.y] for v in points],float)
    scale=max(np.linalg.norm(p[9]-p[0]),1e-6); values=[]
    for a,b,c,e in [(1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]:
        u=p[a]-p[b]; v=p[c]-p[b]; w=p[b]-p[c]; z=p[e]-p[c]
        values.extend([np.linalg.norm(p[e]-p[a])/scale,np.linalg.norm(p[e]-p[0])/scale,
            u@v/max(np.linalg.norm(u)*np.linalg.norm(v),1e-6),w@z/max(np.linalg.norm(w)*np.linalg.norm(z),1e-6)])
    return np.asarray(values)


class Runtime:
    def __init__(self):
        source=json.loads(MODEL.with_suffix('.source.json').read_text(encoding='utf-8'))
        if digest(MODEL)!=source['sha256']: raise ValueError('Frozen generic hand model changed')
        sys.path.insert(0,str(VENDOR))
        import mediapipe as mp
        if mp.__version__!='0.10.32': raise ValueError('Pinned MediaPipe runtime required')
        from mediapipe.tasks.python.components.processors import ClassifierOptions
        self.mp=mp
        options=mp.tasks.vision.GestureRecognizerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.3,min_hand_presence_confidence=.5,
            canned_gesture_classifier_options=ClassifierOptions(max_results=-1,score_threshold=0))
        self.task=mp.tasks.vision.GestureRecognizer.create_from_options(options)

    def close(self): self.task.close()

    def extract(self,frames,points,scores):
        if np.asarray(points).shape!=(len(frames),17,2) or np.asarray(scores).shape!=(len(frames),17):
            raise ValueError('Required M1 actor-reference dimensions do not match')
        values=[]; observed=0; labels=[]
        for frame,p,q in zip(frames,points,scores):
            image=crop(frame,p,q); hands=[]; categories=[]
            if image is not None:
                result=self.task.recognize(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(image)))
                for landmarks,gestures in zip(result.hand_landmarks,result.gestures):
                    feature=hand_values(landmarks); probabilities={g.category_name:g.score for g in gestures}
                    hands.append(np.r_[feature,[probabilities.get(k,0.) for k in GESTURES]])
                    categories.append(max(probabilities,key=probabilities.get) if probabilities else 'None')
            if hands:
                observed+=1; values.append(np.r_[np.mean(hands,axis=0),len(hands),1.])
            else:
                values.append(np.r_[np.full(28,np.nan),0.,0.])
            labels.append(categories)
        feature=stats(np.stack(values))
        if feature.shape!=(DIMENSION,) or not np.isfinite(feature).all(): raise ValueError('Invalid hand descriptor')
        return feature,{'observed_frames':observed,'requested_frames':len(frames),'generic_categories_per_frame':labels,
            'CPR_classification_performed':False,'unavailable_semantics':'No visible hand is unavailable evidence, not a correct gesture'}
