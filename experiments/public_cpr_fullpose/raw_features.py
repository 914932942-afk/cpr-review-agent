"""Label-independent sparse full-hand readout, reusing verified body perception.

The 17-point body cache is identified by video bytes and the frozen v4 pipeline,
not by record labels. The original v4 classifier output is never read as input.
"""
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

from descriptor import sample_indices, hand_features, spectrum_features, PROFILE
from head import CODE
sys.path.insert(0, str(CODE/'experiments/public_cpr_improved'))
from alpha_refiner import AlphaRefiner
from raw_pose_adapter import load_pose, RawPosePredictor
from infer_video import load_manifest
from baseline import digest


class FullRefiner(AlphaRefiner):
    def __call__(self, frame, result):
        if len(result.boxes) == 0:
            return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crops, boxes = [], []
        for x1, y1, x2, y2 in result.boxes.xyxy.cpu().numpy():
            center, scale = self.bbox._box_to_center_scale(x1, y1, x2-x1, y2-y1, .75)
            matrix = self.transforms.get_affine_transform(center, scale, 0, [192, 256])
            image = cv2.warpAffine(rgb, matrix, (192, 256), flags=cv2.INTER_LINEAR)
            tensor = self.transforms.im_to_torch(image)
            tensor -= torch.tensor([.406, .457, .480])[:, None, None]
            crops.append(tensor); boxes.append(self.bbox._center_scale_to_box(center, scale))
        with torch.inference_mode():
            maps = self.model(torch.stack(crops).cuda()).cpu()
        points = []
        for heatmap, box in zip(maps, boxes):
            xy, score = self.transforms.heatmap_to_coord_simple(heatmap, box)
            body_xy, body_score = self.transforms.heatmap_to_coord_simple(heatmap[:17], box)
            np.testing.assert_array_equal(xy[:17], body_xy)
            np.testing.assert_array_equal(score[:17], body_score)
            points.append(np.concatenate([xy, score], 1).astype(np.float32))
        return points


class Runtime:
    def __init__(self):
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.manifest_path = CODE/'data/public_cpr/rgb-evaluation-v4/inference-manifest.json'
        manifest = load_manifest(self.manifest_path)
        self.pose = load_pose(CODE/manifest['detector'])
        self.refiner = FullRefiner()

    def extract(self, video, body_json, body_npz):
        import json
        body = json.loads(Path(body_json).read_text(encoding='utf-8'))
        if (body['video_sha256'] != digest(video) or body['manifest_sha256'] != digest(self.manifest_path)
            or body['public_label_verification'] is not None or body['model_or_threshold_modified'] is not False):
            raise ValueError('Unverified body-perception cache')
        with np.load(body_npz, allow_pickle=False) as cache:
            body_p, body_q, base = [cache[k].copy() for k in ['keypoint', 'keypoint_score', 'features']]
        length = body['decoded_frames']
        if body_p.shape != (1, length, 17, 2) or body_q.shape != (1, length, 17) or base.shape != (1, 783):
            raise ValueError('Invalid base cache dimensions')
        indices = sample_indices(length)
        selected = set(map(int, indices))
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        predictor = RawPosePredictor(overrides={**self.pose.overrides, 'conf': .25, 'batch': 1, 'save': False,
            'mode': 'predict', 'device': 'cuda:0', 'imgsz': 640, 'half': False, 'verbose': False}, _callbacks=self.pose.callbacks)
        full_p, full_q, matching = [], [], []
        ordinal = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if ordinal in selected:
                    detection = self.pose.predict(frame, predictor=predictor, device='cuda:0', imgsz=640, half=False, verbose=False)[0]
                    candidates = self.refiner(frame, detection)
                    scored = []
                    for index, candidate in enumerate(candidates):
                        good = (body_q[0, ordinal] >= .2) & (candidate[:17, 2] >= .2)
                        if good.sum() < 5:
                            continue
                        distance = float(np.sqrt(((candidate[:17, :2][good]-body_p[0, ordinal][good])**2).mean()))
                        scored.append((distance, index))
                    scored.sort()
                    torso = np.linalg.norm(body_p[0, ordinal, [5, 6]].mean(0)-body_p[0, ordinal, [11, 12]].mean(0))
                    if not scored or scored[0][0] > max(2., .05*torso):
                        raise ValueError('No sparse full68 actor matches the frozen body track')
                    if len(scored) > 1 and scored[1][0] <= scored[0][0]+1e-6:
                        raise ValueError('Ambiguous sparse actor match')
                    points = candidates[scored[0][1]]
                    full_p.append(points[:, :2]); full_q.append(np.clip(points[:, 2], 0, 1)); matching.append(scored[0][0])
                ordinal += 1
        finally:
            cap.release()
        if ordinal != length or len(full_p) != len(indices):
            raise ValueError('Incomplete video or hand sample sequence')
        points, confidence = np.stack(full_p), np.stack(full_q)
        hands, quality = hand_features(points, confidence)
        spectrum = spectrum_features(body_p[0], body_q[0], body['reported_fps'])
        feature = np.concatenate([base[0], hands, spectrum])
        return feature, {'sample_indices': indices, 'hand_keypoint': points, 'hand_keypoint_score': confidence}, {
            'video_sha256': body['video_sha256'], 'body_manifest_sha256': body['manifest_sha256'],
            'body_json_sha256': digest(body_json), 'body_npz_sha256': digest(body_npz),
            'decoded_frames': ordinal, 'sampled_hand_frames': len(indices), 'maximum_actor_match_rmse_pixels': max(matching),
            'geometry_quality': quality, 'old_prediction_used_as_feature': False, 'labels_used': False,
            'scope': 'sparse frozen Halpe68 hands + cached original-time body motion; no physical BPM'}
