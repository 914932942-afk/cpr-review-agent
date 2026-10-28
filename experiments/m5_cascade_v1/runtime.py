"""The same sixteen observed ROIs/context, without unused DINO materialisation.

Actor association, crop validity and context statistics match rgb_features_v3.
No landmark interpolation, scene guessing or reduced sample support is introduced.
"""

import numpy as np
import torch
from rgb_features_v3 import CODE, PROFILE, RawPosePredictor, describe, load_pose, roi_and_context, stats


class ContextRuntime:
    def __init__(self):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.pose = load_pose(CODE / "data/public_cpr/weights/yolov8x-pose.pt")
        self.predictor = RawPosePredictor(
            overrides={
                **self.pose.overrides,
                "conf": 0.25,
                "batch": 1,
                "save": False,
                "mode": "predict",
                "device": "cuda:0",
                "imgsz": 640,
                "half": False,
                "verbose": False,
            },
            _callbacks=self.pose.callbacks,
        )

    def extract(self, images, reference_points, reference_scores):
        if np.asarray(reference_points).shape != (len(images), 17, 2) or np.asarray(
            reference_scores
        ).shape != (len(images), 17):
            raise ValueError("M1 actor reference contract is required")
        actors, contexts, margins, skipped = [], [], [], []
        for ordinal, frame in enumerate(images):
            result = self.pose.predict(
                frame, predictor=self.predictor, device="cuda:0", imgsz=640, half=False, verbose=False
            )[0]
            if not len(result.boxes):
                raise ValueError("No observable person")
            points = result.cpr_raw_keypoints.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            ranking = []
            for i, candidate in enumerate(points):
                good = (candidate[:, 2] >= 0.2) & (reference_scores[ordinal] >= 0.2)
                if good.sum() < 5:
                    continue
                a, b = candidate[good, :2], reference_points[ordinal][good]
                a, b = a - a.mean(0), b - b.mean(0)
                a /= max(np.sqrt((a * a).sum(1).mean()), 1.0)
                b /= max(np.sqrt((b * b).sum(1).mean()), 1.0)
                ranking.append((float(np.sqrt(((a - b) ** 2).mean())), i))
            ranking.sort()
            if not ranking or ranking[0][0] > 0.5:
                skipped.append({"sample": ordinal, "reason": "no actor matches M1 track"})
                continue
            if len(ranking) > 1 and ranking[1][0] - ranking[0][0] < 0.015:
                skipped.append({"sample": ordinal, "reason": "ambiguous M1 track binding"})
                continue
            index = ranking[0][1]
            actor = points[index]
            others = [i for i in range(len(points)) if i != index]
            other_index = (
                max(others, key=lambda i: (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
                if others
                else None
            )
            other = points[other_index] if other_index is not None else None
            other_box = boxes[other_index] if other_index is not None else None
            try:
                _, _, context = roi_and_context(frame, actor, other, other_box)
            except ValueError:
                skipped.append({"sample": ordinal, "reason": "unusable contact ROI"})
                continue
            actors.append(actor)
            contexts.append(context)
            margins.append(ranking[1][0] - ranking[0][0] if len(ranking) > 1 else 10.0)
        if len(actors) < 8:
            raise ValueError("Too few usable RGB samples after actor/ROI checks")
        actors = np.stack(actors)
        body, quality = describe(actors[:, :, :2], np.clip(actors[:, :, 2], 0, 1))
        if not quality["geometry_valid"]:
            raise ValueError("No body geometry for RGB context")
        scales = np.linalg.norm(actors[:, [5, 6], :2].mean(1) - actors[:, [11, 12], :2].mean(1), axis=1)
        torso_scale = max(float(np.median(scales)), 1.0)
        motion = actors[:, :, :2].std(0).reshape(-1) / torso_scale
        axes = []
        for a, b in [(5, 9), (6, 10), (5, 7), (6, 8), (7, 9), (8, 10)]:
            vector = actors[:, b, :2] - actors[:, a, :2]
            length = np.linalg.norm(vector, axis=1)
            tilt = np.abs(vector[:, 0]) / np.maximum(length, 1.0)
            tilt[(actors[:, a, 2] < 0.2) | (actors[:, b, 2] < 0.2) | (length < 1.0)] = np.nan
            axes.append(tilt)
        arm = stats(np.stack(axes, 1))
        hip = actors[:, [11, 12], 1].mean(1)
        knee = actors[:, [13, 14], 1].mean(1)
        height = (knee - hip) / torso_scale
        visible = (actors[:, [11, 12, 13, 14], 2] >= 0.2).all(1)
        height[~visible] = np.nan
        context = np.concatenate(
            [body, motion, stats(np.stack(contexts)), arm, stats(height[:, None])]
        ).astype(np.float32)
        if context.shape != (357,) or not np.isfinite(context).all():
            raise ValueError("Invalid context feature representation")
        return context, {
            "requested_frames": len(images),
            "frames": len(actors),
            "skipped_samples": skipped,
            "manikin_reference_fraction": float(np.mean(np.asarray(contexts)[:, -1])),
            "minimum_actor_geometry_margin": float(min(margins)),
            "geometry_quality": quality,
            "input_profile": PROFILE,
            "image_backbone_executed": False,
            "visual_feature_slots_materialised": False,
        }
