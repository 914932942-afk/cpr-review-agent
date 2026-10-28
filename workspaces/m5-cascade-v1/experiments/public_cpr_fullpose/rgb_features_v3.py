"""Frozen RGB contact/context representation. No labels or file names as inputs."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from head import CODE
from descriptor import stats
sys.path.insert(0, str(CODE/'experiments/public_cpr_improved'))
from raw_pose_adapter import load_pose, RawPosePredictor
from geometry import describe
from baseline import digest

MODEL = CODE/'data/public_cpr/weights/dinov2-small'
DIMENSION = 1893
PROFILE = {'schema': 'cpr-rgb-contact-context-v3', 'samples': 16, 'minimum_usable_samples': 8, 'backbone': 'frozen DINOv2-small CLS384',
    'roi': 'distal hand crop extended .3 forearm lengths beyond wrists + manikin torso/contact crop; grayscale letterbox224; exclude top12% image overlay',
    'actor': 'bind to supplied M1 actor track using dimensionless COCO17 shape; no independent actor choice',
    'features': 'RGB1536 + body271 + motion34 + contact24 + arm-axis24 + hip/knee movement4',
    'dimension': DIMENSION, 'not_used': 'labels, timestamps, file names, faces as identity, public ground-truth skeletons'}


def actor_score(points, box):
    width, height = box[2]-box[0], box[3]-box[1]
    torso = points[[11, 12], :2].mean(0)-points[[5, 6], :2].mean(0)
    upright = abs(torso[1])/max(np.linalg.norm(torso), 1.)
    return float(np.log(max(height/max(width, 1.), .01))+.5*upright)


def crop_image(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = int(max(0, x1)), int(min(width, x2))
    y1, y2 = int(max(height*.12, y1)), int(min(height, y2))
    if x2-x1 < 8 or y2-y1 < 8:
        raise ValueError('Invalid visual ROI')
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(gray).convert('RGB')
    return ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=(128, 128, 128))


def roi_and_context(frame, actor, other, other_box):
    wrists = actor[[9, 10], :2]
    forearms = wrists-actor[[7, 8], :2]
    palms = wrists+.3*forearms
    center = palms.mean(0)
    torso = np.linalg.norm(actor[[5, 6], :2].mean(0)-actor[[11, 12], :2].mean(0))
    radius = max(50., .18*torso, .45*np.linalg.norm(forearms, axis=1).mean(), .75*np.linalg.norm(palms[0]-palms[1]))
    hand = crop_image(frame, [*(center-radius), *(center+radius)])
    reference_valid = other is not None and bool((other[[5, 6, 11, 12], 2] >= .2).all())
    if reference_valid:
        shoulder = other[[5, 6], :2].mean(0)
        hip = other[[11, 12], :2].mean(0)
        axis = hip-shoulder; length = max(np.linalg.norm(axis), 1.)
        longitudinal = axis/length
        lateral = np.array([-longitudinal[1], longitudinal[0]])
        delta = center-shoulder
        context = np.array([delta@longitudinal/length, delta@lateral/length,
            np.linalg.norm(wrists[0]-wrists[1])/length, np.linalg.norm(delta)/length,
            np.linalg.norm(other[5, :2]-other[6, :2])/length, 1.])
        points = np.concatenate([other[[5, 6, 11, 12], :2], wrists])
        low, high = points.min(0), points.max(0)
        padding = max(radius*.5, .15*np.linalg.norm(high-low))
        box = [*(low-padding), *(high+padding)]
    elif other_box is not None:
        context = np.zeros(6)
        box = [min(other_box[0], center[0]-radius), min(other_box[1], center[1]-radius),
               max(other_box[2], center[0]+radius), max(other_box[3], center[1]+radius)]
    else:
        context = np.zeros(6)
        box = [*(center-radius*2), *(center+radius*2)]
    return hand, crop_image(frame, box), context


class RGBRuntime:
    def __init__(self):
        from transformers import AutoModel
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.source = json.loads((MODEL/'source.json').read_text(encoding='utf-8'))
        for row in self.source['files']:
            if digest(MODEL/row['file']) != row['sha256']:
                raise ValueError('Frozen DINO model changed')
        self.model = AutoModel.from_pretrained(str(MODEL), local_files_only=True, use_safetensors=True, trust_remote_code=False).eval().requires_grad_(False).cuda()
        self.pose = load_pose(CODE/'data/public_cpr/weights/yolov8x-pose.pt')
        self.predictor = RawPosePredictor(overrides={**self.pose.overrides, 'conf': .25, 'batch': 1,
            'save': False, 'mode': 'predict', 'device': 'cuda:0', 'imgsz': 640, 'half': False, 'verbose': False}, _callbacks=self.pose.callbacks)
        cfg = json.loads((MODEL/'preprocessor_config.json').read_text(encoding='utf-8'))
        self.mean = torch.tensor(cfg['image_mean']).reshape(1, 3, 1, 1).cuda()
        self.scale = torch.tensor(cfg['image_std']).reshape(1, 3, 1, 1).cuda()

    def extract(self, images, reference_points, reference_scores):
        if np.asarray(reference_points).shape != (len(images), 17, 2) or np.asarray(reference_scores).shape != (len(images), 17):
            raise ValueError('M1 actor reference contract is required')
        crops, actors, contexts, margins, skipped = [], [], [], [], []
        for ordinal, frame in enumerate(images):
            result = self.pose.predict(frame, predictor=self.predictor, device='cuda:0', imgsz=640, half=False, verbose=False)[0]
            if not len(result.boxes):
                raise ValueError('No observable person')
            points = result.cpr_raw_keypoints.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            ranking = []
            for i, candidate in enumerate(points):
                good = (candidate[:, 2] >= .2) & (reference_scores[ordinal] >= .2)
                if good.sum() < 5:
                    continue
                a, b = candidate[good, :2], reference_points[ordinal][good]
                a, b = a-a.mean(0), b-b.mean(0)
                a /= max(np.sqrt((a*a).sum(1).mean()), 1.)
                b /= max(np.sqrt((b*b).sum(1).mean()), 1.)
                ranking.append((float(np.sqrt(((a-b)**2).mean())), i))
            ranking.sort()
            if not ranking or ranking[0][0] > .5:
                skipped.append({'sample': ordinal, 'reason': 'no actor matches M1 track'}); continue
            if len(ranking) > 1 and ranking[1][0]-ranking[0][0] < .015:
                skipped.append({'sample': ordinal, 'reason': 'ambiguous M1 track binding'}); continue
            index = ranking[0][1]
            actor = points[index]
            others = [i for i in range(len(points)) if i != index]
            other_index = max(others, key=lambda i: (boxes[i, 2]-boxes[i, 0])*(boxes[i, 3]-boxes[i, 1])) if others else None
            other = points[other_index] if other_index is not None else None
            other_box = boxes[other_index] if other_index is not None else None
            try:
                hand, scene, context = roi_and_context(frame, actor, other, other_box)
            except ValueError:
                skipped.append({'sample': ordinal, 'reason': 'unusable contact ROI'}); continue
            crops.extend([hand, scene]); actors.append(actor); contexts.append(context)
            margins.append(ranking[1][0]-ranking[0][0] if len(ranking) > 1 else 10.)
        if len(actors) < 8:
            raise ValueError('Too few usable RGB samples after actor/ROI checks')
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(crops), 8):
                values = np.stack([np.asarray(im).transpose(2, 0, 1) for im in crops[start:start+8]])
                tensor = torch.from_numpy(values.copy()).cuda().float()/255
                encoded = self.model(pixel_values=(tensor-self.mean)/self.scale).last_hidden_state[:, 0].float()
                encoded = torch.nn.functional.normalize(encoded, dim=1)
                embeddings.append(encoded.cpu().numpy())
        embeddings = np.concatenate(embeddings).reshape(len(actors), 2, 384)
        visual = np.concatenate([embeddings.mean(0).reshape(-1), embeddings.std(0).reshape(-1)])
        actors = np.stack(actors)
        body, quality = describe(actors[:, :, :2], np.clip(actors[:, :, 2], 0, 1))
        if not quality['geometry_valid']:
            raise ValueError('No body geometry for RGB context')
        scales = np.linalg.norm(actors[:, [5, 6], :2].mean(1)-actors[:, [11, 12], :2].mean(1), axis=1)
        torso_scale = max(float(np.median(scales)), 1.)
        unanchored_motion = actors[:, :, :2].std(0).reshape(-1)/torso_scale
        axes = []
        for a, b in [(5, 9), (6, 10), (5, 7), (6, 8), (7, 9), (8, 10)]:
            vector = actors[:, b, :2]-actors[:, a, :2]
            length = np.linalg.norm(vector, axis=1)
            tilt = np.abs(vector[:, 0])/np.maximum(length, 1.)
            tilt[(actors[:, a, 2] < .2) | (actors[:, b, 2] < .2) | (length < 1.)] = np.nan
            axes.append(tilt)
        arm_axis = stats(np.stack(axes, 1))
        hip = actors[:, [11, 12], 1].mean(1)
        knee = actors[:, [13, 14], 1].mean(1)
        relative_height = (knee-hip)/torso_scale
        visible = (actors[:, [11, 12, 13, 14], 2] >= .2).all(1)
        relative_height[~visible] = np.nan
        hip_motion = stats(relative_height[:, None])
        result = np.concatenate([visual, body, unanchored_motion, stats(np.stack(contexts)), arm_axis, hip_motion]).astype(np.float32)
        if result.shape != (DIMENSION,) or not np.isfinite(result).all():
            raise ValueError('Invalid frozen RGB representation')
        return result, {'requested_frames': len(images), 'frames': len(actors), 'skipped_samples': skipped,
            'manikin_reference_fraction': float(np.mean(np.asarray(contexts)[:, -1])),
            'minimum_actor_geometry_margin': float(min(margins)), 'geometry_quality': quality,
            'all_visual_backbone_parameters_frozen': all(not p.requires_grad for p in self.model.parameters()),
            'input_profile': PROFILE}
