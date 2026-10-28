"""Decompose the frozen v15 computation, preserving its features, heads and decisions.

RGB uses the shared actor track and geometry context. It is not a pose-free RGB model.
No fitting, public targets, source record identifiers or filename-derived decisions occur here.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/public_cpr_fullpose"))
import infer_candidate_v2 as frozen  # noqa: E402

LABELS = list(frozen.o.parent.LABEL_IDS)
DATA = ROOT / "data/public_cpr"
SCHEMA = "cpr-decomposed-stage-v1"
TOOLS = ["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def array_file(path, **values):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
    return {"name": path.name, "sha256": sha(path)}


def read_stage(path, tool, video_sha, bundle_sha, pipeline_sha):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != SCHEMA
        or value.get("tool_id") != tool
        or value.get("video_sha256") != video_sha
        or value.get("bundle_sha256") != bundle_sha
        or value.get("pipeline_sha256") != pipeline_sha
        or value.get("labels_used") is not False
    ):
        raise ValueError("STAGE_BINDING_MISMATCH")
    for artifact in value.get("files", []):
        target = (path.parent / artifact["name"]).resolve()
        if target.parent != path.parent.resolve() or sha(target) != artifact["sha256"]:
            raise ValueError("STAGE_ARTIFACT_CHANGED")
    return value


def load_arrays(path, value):
    target = Path(path).parent / value["files"][0]["name"]
    with np.load(target, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def skeleton_readout(full, motion):
    """Existing full-keypoint head + full-sequence mechanics6; no RGB refinement."""
    source = frozen.o.parent.read(DATA / "rgb-evaluation-v12/protocol.json")
    protocol = frozen.o.parent.read(ROOT / source["base_protocol"])
    base, _ = frozen.load(
        ROOT / source["base_head"], protocol["feature_profile_hash"], sha(ROOT / source["base_protocol"])
    )
    probability = base.probability(full)
    scores, eligible = probability.copy(), base.predict(full, probability)
    thresholds = np.tile(np.asarray(base.state["operating_point"]["thresholds"], float), (len(full), 1))
    mechanical = torch.load(ROOT / source["mechanics_head"], map_location="cpu", weights_only=True)
    for label, (p, decisions) in frozen.predict_mechanics(motion, mechanical).items():
        col = LABELS.index(label)
        scores[:, col], eligible[:, col] = p, decisions
        thresholds[:, col] = next(e["threshold"] for e in mechanical["experts"] if e["label_id"] == label)
    cardinality = torch.load(
        DATA / "rgb-evaluation-v13/cardinality.pt", map_location="cpu", weights_only=True
    )
    count = frozen.counts(np.concatenate([full[:, :783], motion], axis=1), cardinality)
    return {"scores": scores, "eligible": eligible, "thresholds": thresholds, "predicted_count": count}


def rgb_readout(rgb, shape, available):
    """Existing five-category RGB/context specialists, plus hand refinements when observable."""
    state = torch.load(DATA / "rgb-evaluation-v12/experts.pt", map_location="cpu", weights_only=True)
    hand = torch.load(DATA / "rgb-evaluation-v14/hands.pt", map_location="cpu", weights_only=True)
    labels = [e["label_id"] for e in state["experts"]]
    scores, decisions, thresholds = [], [], []
    for label, (probability, pred) in frozen.predict_experts(rgb, state).items():
        assert label == labels[len(scores)]
        scores.append(probability)
        decisions.append(pred)
        thresholds.append(next(e["threshold"] for e in state["experts"] if e["label_id"] == label))
    scores, decisions = np.stack(scores, axis=1), np.stack(decisions, axis=1)
    thresholds = np.tile(np.asarray(thresholds, float), (len(rgb), 1))
    override = np.zeros_like(decisions, dtype=bool)
    hand_input = np.concatenate([shape, rgb[:, frozen.HAND_INDICES]], axis=1)
    for label, (probability, pred) in frozen.predict_hands(hand_input, hand).items():
        col = labels.index(label)
        scores[available, col] = probability[available]
        decisions[available, col] = pred[available]
        thresholds[available, col] = next(e["threshold"] for e in hand["experts"] if e["label_id"] == label)
        override[available, col] = True
    return {
        "label_ids": np.asarray(labels),
        "scores": scores,
        "eligible": decisions,
        "thresholds": thresholds,
        "hand_override": override,
        "normal_veto_threshold": np.asarray([state["normal_veto_threshold"]]),
    }


def merge_readouts(skeleton, rgb):
    scores, eligible, thresholds = (skeleton[k].copy() for k in ["scores", "eligible", "thresholds"])
    for index, label in enumerate(rgb["label_ids"]):
        col = LABELS.index(int(label))
        scores[:, col], thresholds[:, col] = rgb["scores"][:, index], rgb["thresholds"][:, index]
        allowed = (skeleton["scores"][:, 11] < rgb["normal_veto_threshold"][0]) | rgb["hand_override"][
            :, index
        ]
        eligible[:, col] = rgb["eligible"][:, index] & allowed
    reliability = frozen.o.parent.read(DATA / "rgb-evaluation-v15/protocol.json")["reliability"]
    qualified = np.asarray([reliability[str(label)]["qualified"] for label in LABELS])
    pred = frozen.guarded(eligible, scores[:, :11], thresholds, skeleton["predicted_count"], qualified)
    return {
        "scores": scores,
        "eligible": eligible,
        "thresholds": thresholds,
        "predicted_count": skeleton["predicted_count"],
        "predictions": pred,
    }, reliability


def run_stage(tool, video, output, manifest_path, pipeline_sha, inputs):
    started = time.perf_counter()
    if tool not in TOOLS or output.exists() or output.with_suffix(".npz").exists():
        raise ValueError("INVALID_OR_EXISTING_STAGE_OUTPUT")
    manifest = frozen.validate_bundle(manifest_path)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.parent.mkdir(parents=True, exist_ok=True)
    video_sha, bundle_sha = sha(video), sha(manifest_path)
    value = {
        "schema": SCHEMA,
        "status": "completed_experimental",
        "tool_id": tool,
        "video_sha256": video_sha,
        "bundle_sha256": bundle_sha,
        "pipeline_sha256": pipeline_sha,
        "labels_used": False,
        "model_or_threshold_modified": False,
        "files": [],
        "dependencies": {},
        "supported_label_ids": [],
        "scores": [],
        "predicted_original_label_ids": [],
    }
    dependencies, arrays = {}, {}
    for required in {
        "pose.extract": [],
        "skeleton.predict": ["pose.extract"],
        "rgb.predict": ["pose.extract"],
        "fusion.predict": ["skeleton.predict", "rgb.predict"],
    }[tool]:
        path = Path(inputs[required])
        dependencies[required] = read_stage(path, required, video_sha, bundle_sha, pipeline_sha)
        arrays[required] = load_arrays(path, dependencies[required])
        value["dependencies"][required] = sha(path)
    if tool == "pose.extract":
        body_path = output.with_name(output.stem + "-body.json")
        body = frozen.analyze_body(video, body_path, ROOT / manifest["body_manifest"], None)
        runtime = frozen.FullRuntime()
        full, _, audit = runtime.extract(video, body_path, body_path.with_suffix(".npz"))
        del runtime
        torch.cuda.empty_cache()
        with np.load(body_path.with_suffix(".npz"), allow_pickle=False) as cache:
            points, confidence = cache["keypoint"][0].copy(), cache["keypoint_score"][0].copy()
        motion = frozen.mechanics_features(points, confidence)
        value["files"] = [
            array_file(
                output.with_suffix(".npz"),
                full=full[None],
                motion=motion[None],
                points=points,
                confidence=confidence,
            )
        ]
        for path in [body_path, body_path.with_suffix(".npz")]:
            value["files"].append({"name": path.name, "sha256": sha(path)})
        value.update(
            decoded_frames=body["decoded_frames"],
            reported_fps=body["reported_fps"],
            fullpose_audit=audit,
            shared_perception="Frozen v4 body track/features and sparse Halpe68 hands; discarded v4 classification is not evidence",
        )
    elif tool in {"skeleton.predict", "rgb.predict"}:
        source, perception = dependencies["pose.extract"], arrays["pose.extract"]
        value.update(
            decoded_frames=source["decoded_frames"],
            reported_fps=source["reported_fps"],
            shared_pose_sha256=value["dependencies"]["pose.extract"],
        )
        if tool == "skeleton.predict":
            result = skeleton_readout(perception["full"], perception["motion"])
            labels = LABELS
            value["fullpose_audit"] = source["fullpose_audit"]
        else:
            points, confidence = perception["points"], perception["confidence"]
            indices = frozen.sample_indices(len(points))
            frames = frozen.read_video_samples(video, len(points))
            runtime = frozen.RGBRuntime()
            rgb, audit = runtime.extract(frames, points[indices], confidence[indices])
            del runtime
            torch.cuda.empty_cache()
            shape, shape_audit = frozen.shape_features(frames, points[indices], confidence[indices])
            result = rgb_readout(rgb[None], shape[None], np.asarray([shape_audit["available"]]))
            labels = result["label_ids"].tolist()
            value.update(
                rgb_audit=audit,
                hand_shape_audit=shape_audit,
                modality="RGB/local shape plus shared actor and geometric context; not pose-free RGB",
            )
            result.update(rgb_features=rgb[None], shape_features=shape[None])
        value["files"] = [array_file(output.with_suffix(".npz"), **result)]
        value["supported_label_ids"] = labels
        value["scores"] = [
            {
                "original_label_id": label,
                "uncalibrated_score": float(result["scores"][0, i]),
                "threshold": float(result["thresholds"][0, i]),
            }
            for i, label in enumerate(labels)
        ]
        value["predicted_original_label_ids"] = [
            label for label, yes in zip(labels, result["eligible"][0]) if yes
        ]
        value["decision_scope"] = "Branch candidates before cross-branch veto/reliability fusion"
    else:
        skeleton, rgb = dependencies["skeleton.predict"], dependencies["rgb.predict"]
        if (
            skeleton["shared_pose_sha256"] != rgb["shared_pose_sha256"]
            or skeleton["decoded_frames"] != rgb["decoded_frames"]
            or skeleton["reported_fps"] != rgb["reported_fps"]
        ):
            raise ValueError("CROSS_TRACK_FUSION")
        merged, reliability = merge_readouts(arrays["skeleton.predict"], arrays["rgb.predict"])
        value["files"] = [array_file(output.with_suffix(".npz"), **merged)]
        labels = [label for label, yes in zip(LABELS, merged["predictions"][0]) if yes]
        value.update(
            schema="cpr-offline-analysis-v2",
            decoded_frames=skeleton["decoded_frames"],
            reported_fps=skeleton["reported_fps"],
            supported_label_ids=LABELS,
            predicted_original_label_ids=labels,
            estimated_supported_error_count=int(merged["predicted_count"][0]),
            reliability_sources=reliability,
            rgb_audit=rgb["rgb_audit"],
            hand_shape_audit=rgb["hand_shape_audit"],
            fullpose_audit=skeleton["fullpose_audit"],
            public_label_verification=None,
            scores=[
                {
                    "original_label_id": label,
                    "uncalibrated_score": float(merged["scores"][0, i]),
                    "threshold": float(merged["thresholds"][0, i]),
                }
                for i, label in enumerate(LABELS)
            ],
            scope="Same frozen v15 computation decomposed into four tools; whole-clip experimental evidence",
        )
    value["analysis_seconds"] = time.perf_counter() - started
    value["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(output, value)
    return value
