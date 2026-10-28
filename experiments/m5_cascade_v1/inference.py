"""Production inference body, separate from the module adapter and benchmark driver."""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from pipeline import frozen, rgb_readout


def local(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


algorithm = local("m5_dependency_algorithm", "algorithm.py")
context_runtime = local("m5_context_runtime", "runtime.py")


def infer(video, points, confidence):
    indices = frozen.sample_indices(len(points))
    frames = frozen.read_video_samples(video, len(points))
    shape, hand_audit = frozen.shape_features(frames, points[indices], confidence[indices])
    root = Path(__file__).resolve().parents[2]
    rgb_state = torch.load(
        root / "data/public_cpr/rgb-evaluation-v12/experts.pt", map_location="cpu", weights_only=True
    )
    hand_state = torch.load(
        root / "data/public_cpr/rgb-evaluation-v14/hands.pt", map_location="cpu", weights_only=True
    )
    plan = algorithm.compile_plan(rgb_state, hand_state, bool(hand_audit["available"]))
    if plan["dense_visual_required"]:
        runtime = frozen.RGBRuntime()
        rgb, audit = runtime.extract(frames, points[indices], confidence[indices])
        del runtime
        torch.cuda.empty_cache()
        arrays = rgb_readout(rgb[None], shape[None], np.array([hand_audit["available"]]))
        arrays["rgb_features"] = rgb[None]
        dense_crops = 2 * audit["frames"]
    else:
        runtime = context_runtime.ContextRuntime()
        context, audit = runtime.extract(frames, points[indices], confidence[indices])
        del runtime
        torch.cuda.empty_cache()
        arrays = algorithm.context_readout(
            context[None],
            shape[None],
            rgb_state,
            hand_state,
            plan,
            frozen.predict_experts,
            frozen.predict_hands,
        )
        # Save only computed features. No fabricated dense embeddings in artifacts.
        arrays.update(context_features=context[None], computed_rgb_feature_indices=np.arange(1536, 1893))
        dense_crops = 0
    arrays["shape_features"] = shape[None]
    return arrays, {
        "rgb_audit": audit,
        "hand_shape_audit": hand_audit,
        "selected_frame_indices": indices.tolist(),
        "sampling_profile": "uniform16-v15",
        "m5_cascade": {
            **plan,
            "dense_visual_crop_evaluations": dense_crops,
            "observed_frames_unchanged": True,
            "minimum_rgb_support": 8,
            "weights_and_thresholds_unchanged": True,
            "uncomputed_dense_features_persisted": False,
        },
    }
