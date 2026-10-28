"""Version-pinned adapter preserving model coordinates before presentation masking.

The upstream Ultralytics 8.3.0 Keypoints wrapper zeros xy below score 0.5.
The CPR feature contract uses xy together with continuous scores; its visibility
threshold is applied by the downstream descriptor, not by a display wrapper.
NMS, scaling and confidence scores are unchanged. No coordinates are invented.
"""
from pathlib import Path
import hashlib
import importlib.metadata

import numpy as np
import torch
from ultralytics.engine.results import Results
from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.utils import ops

VERSION = "8.3.0"
WEIGHTS = {
    "7a9bf952c2fc1c07d4a1da00dbb4cfab5ed71f89cbe8824d51a8f7a61cee5825",
    "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212",
}
PROFILE = {"schema": "cpr-raw-pose-v1", "ultralytics": VERSION,
           "layout": "COCO17", "coordinate_policy": "unmasked model xy and original confidence",
           "person_selection": "highest box confidence; experimental fixed-camera clips only",
           "image_size": 640, "device_precision": "float32"}


def select_active_track(candidates):
    """Associate detector boxes without labels; select sustained torso motion.

    This offline selector addresses the rescuer/manikin pair. Ambiguous moving
    people are not resolved by guessing. No public reference poses are inputs.
    """
    from scipy.optimize import linear_sum_assignment
    tracks = []
    for frame_index, detections in enumerate(candidates):
        live = [t for t in tracks if frame_index - t["last"] <= 5]
        assigned = set()
        if live and detections:
            cost = np.full((len(live), len(detections)), 2.)
            for i, track in enumerate(live):
                a = track["box"][:4]
                for j, detection in enumerate(detections):
                    b = detection["box"][:4]
                    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
                    union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection
                    cost[i, j] = 1 - intersection / max(union, 1.)
            rows, columns = linear_sum_assignment(cost)
            for i, j in zip(rows, columns):
                if cost[i, j] <= .8:
                    track = live[i]; detection = detections[j]
                    track["frames"][frame_index] = detection["points"]
                    track["motion_frames"][frame_index] = detection.get("tracking_points", detection["points"])
                    track["box"] = detection["box"]; track["last"] = frame_index
                    assigned.add(j)
        for index, detection in enumerate(detections):
            if index not in assigned:
                tracks.append({"frames": {frame_index: detection["points"]},
                               "motion_frames": {frame_index: detection.get("tracking_points", detection["points"])},
                               "box": detection["box"], "last": frame_index, "id": len(tracks)})
    eligible = []
    for track in tracks:
        coverage = len(track["frames"]) / len(candidates)
        if coverage < .8: continue
        points = np.zeros((len(candidates), 17, 3), dtype=np.float32)
        motion_points = np.zeros_like(points)
        for frame_index, value in track["frames"].items(): points[frame_index] = value
        for frame_index, value in track["motion_frames"].items(): motion_points[frame_index] = value
        joints = np.array([0, 5, 6, 11, 12])
        valid = (motion_points[1:, joints, 2] >= .5) & (motion_points[:-1, joints, 2] >= .5)
        movement = np.linalg.norm(np.diff(motion_points[:, joints, :2], axis=0), axis=2)
        movement = movement[valid]
        scale = np.linalg.norm(track["box"][2:4] - track["box"][:2])
        activity = float(np.median(movement) / max(scale, 1.)) if len(movement) else 0.
        eligible.append((activity, track["id"], points, coverage))
    eligible.sort(key=lambda row: (-row[0], row[1]))
    if not eligible or eligible[0][0] <= 0:
        raise ValueError("No persistent moving-person track; no classification emitted")
    if len(eligible) > 1 and eligible[0][0] < 1.2 * eligible[1][0]:
        raise ValueError("Ambiguous active-person tracks; no classification emitted")
    return eligible[0][2], {"selected_track_id": eligible[0][1],
        "selection": "IoU temporal association + dominant median torso/head motion",
        "motion_source": "YOLO raw keypoints, independent of downstream pose refiner",
        "track_coverage": eligible[0][3],
        "eligible_tracks": [{"track_id": r[1], "normalized_motion": r[0], "coverage": r[3]} for r in eligible]}


def make_result(image, path, names, boxes, keypoints):
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError("Expected [people,17,3] model keypoints")
    if not torch.isfinite(keypoints).all():
        raise ValueError("Non-finite model keypoints")
    raw = keypoints.detach().clone()
    result = Results(image, path=path, names=names, boxes=boxes,
                     keypoints=raw.clone())
    result.cpr_raw_keypoints = raw
    return result


class RawPosePredictor(PosePredictor):
    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(
            preds, self.args.conf, self.args.iou,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det,
            classes=self.args.classes, nc=len(self.model.names))
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for pred, orig_img, path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            keypoints = pred[:, 6:].reshape(-1, 17, 3)
            keypoints = ops.scale_coords(img.shape[2:], keypoints, orig_img.shape)
            results.append(make_result(orig_img, path, self.model.names, pred[:, :6], keypoints))
        return results


def load_pose(path):
    from ultralytics import YOLO
    if importlib.metadata.version("ultralytics") != VERSION:
        raise ValueError("Pose adapter requires the verified Ultralytics version " + VERSION)
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() not in WEIGHTS:
        raise ValueError("Unverified pose checkpoint")
    original = torch.load
    def load(source, *args, **kwargs):
        if isinstance(source, (str, Path)) and Path(source).resolve() == path:
            kwargs.setdefault("weights_only", False)
        return original(source, *args, **kwargs)
    # Compatibility for this known old, hash-verified model-object checkpoint.
    # Offline experiment is single-threaded; no web worker imports this module.
    torch.load = load
    try:
        model = YOLO(str(path))
    finally:
        torch.load = original
    return model


def decode_pose(video, model, maximum_frames=3000, track_person=False, refiner=None):
    import cv2
    if model.predictor is not None and not isinstance(model.predictor, RawPosePredictor):
        raise ValueError("Model already has an incompatible, masking predictor")
    predictor = RawPosePredictor(overrides={**model.overrides, "conf": .25,
        "batch": 1, "save": False, "mode": "predict", "device": "cuda:0",
        "imgsz": 640, "half": False, "verbose": False}, _callbacks=model.callbacks)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Cannot open video")
    points, counts, candidates = [], [], []
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    advertised_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or advertised_frames > maximum_frames:
        capture.release()
        raise ValueError("Unsupported video timing or frame budget")
    shape = None
    preserved = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if len(points) >= maximum_frames:
                raise ValueError("Video exceeds the offline verification frame limit")
            height, width = frame.shape[:2]
            if shape is not None and shape != (height, width):
                raise ValueError("Video dimensions change within a clip")
            shape = (height, width)
            result = model.predict(frame, predictor=predictor, device="cuda:0",
                                   imgsz=640, half=False, verbose=False)[0]
            count = len(result.boxes)
            counts.append(count)
            refined = refiner(frame, result) if refiner is not None else None
            candidates.append([{"points": refined[i] if refined is not None else result.cpr_raw_keypoints[i].cpu().numpy().astype(np.float32),
                                "tracking_points": result.cpr_raw_keypoints[i].cpu().numpy().astype(np.float32),
                                "box": result.boxes.data[i].cpu().numpy().astype(np.float32)}
                               for i in range(count)])
            if count:
                index = int(result.boxes.conf.argmax())
                raw = result.cpr_raw_keypoints[index].cpu().numpy().astype(np.float32)
                displayed = result.keypoints.data[index].cpu().numpy()
                preserved += int(((raw[:, 2] < .5) & (raw[:, 2] >= .2) &
                                  (displayed[:, :2] == 0).all(1) &
                                  (raw[:, :2] != 0).any(1)).sum())
                if refined is not None: raw = refined[index]
            else:
                raw = np.zeros((17, 3), dtype=np.float32)
            if raw.shape != (17, 3) or not np.isfinite(raw).all():
                raise ValueError("Invalid pose output")
            points.append(raw)
    finally:
        capture.release()
    detected = sum(count > 0 for count in counts)
    if advertised_frames > 0 and len(points) != advertised_frames:
        raise ValueError("Decoded frame count differs from the declared video; possible truncation")
    if len(points) < 8 or detected / len(points) < .5:
        raise ValueError("Insufficient observable person frames; no classification emitted")
    points = np.stack(points)
    track_audit = {}
    if track_person:
        points, track_audit = select_active_track(candidates)
    return {"keypoint": points[None, :, :, :2],
            "keypoint_score": np.clip(points[None, :, :, 2], 0, 1),
            "img_shape": (height, width)}, {
        "decoded_frames": len(points), "reported_fps": fps,
        "frames_with_person": detected,
        "frames_with_multiple_people": sum(n > 1 for n in counts),
        "low_score_coordinates_preserved": preserved,
        "person_selection": track_audit or PROFILE["person_selection"],
        "rate_semantics": "nominal container fps; no physical compression-rate measurement"}
