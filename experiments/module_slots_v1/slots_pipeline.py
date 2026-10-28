"""Real M2/M5 slots around the accepted M1 pipeline; the controller owns persistence."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "backend"),
    str(ROOT / "experiments/m1_fast_v1"),
    str(ROOT / "experiments/branch_pipeline_v1"),
]
from cpr_agent.modules.contracts import (  # noqa: E402
    CONTRACT_VERSION,
    LABEL_IDS,
    RGB_LABEL_IDS,
    ArtifactRef,
    Binding,
    FusionInput,
    MLResult,
    Prediction,
    QualityIndicators,
    Readout,
    RGBInput,
)
from cpr_agent.modules.registry import build_registry  # noqa: E402
from m1_fast_pipeline import run_stage as accepted_stage  # noqa: E402
from pipeline import array_file, frozen, load_arrays, read_stage, save, sha  # noqa: E402


def from_stage(raw, arrays, binding, prediction_id):
    return Prediction(
        prediction_id=prediction_id,
        binding=binding,
        supported_label_ids=raw["supported_label_ids"],
        flagged_label_ids=raw["predicted_original_label_ids"],
        score_type="uncalibrated_score",
        scores={s["original_label_id"]: s["uncalibrated_score"] for s in raw["scores"]},
        thresholds={s["original_label_id"]: s["threshold"] for s in raw["scores"]},
    )


def validate_result(result, binding, execution_id, labels, decision_key):
    if not isinstance(result, MLResult) or not isinstance(result.prediction, Prediction):
        raise ValueError("MODULE_RESULT_TYPE_INVALID")
    p = Prediction.model_validate(result.prediction)
    if p.binding != binding or p.prediction_id != execution_id or p.supported_label_ids != list(labels):
        raise ValueError("MODULE_RESULT_BINDING_MISMATCH")
    if p.score_type != "uncalibrated_score" or set(p.thresholds) != set(labels):
        raise ValueError("MODULE_SCORE_PROFILE_NOT_SUPPORTED")
    for key, value in result.arrays.items():
        if not isinstance(value, np.ndarray) or value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError("MODULE_ARRAY_INVALID: " + key)
    for key in ["scores", "thresholds", "eligible", decision_key]:
        a = result.arrays[key]
        if a.ndim != 2 or a.shape[0] != 1 or a.shape[1] < len(labels):
            raise ValueError("MODULE_ARRAY_LAYOUT_INVALID")
    flags = [k for i, k in enumerate(labels) if result.arrays[decision_key][0, i]]
    if result.arrays["eligible"].dtype != bool or result.arrays[decision_key].dtype != bool:
        raise ValueError("MODULE_DECISION_MASK_NOT_BOOLEAN")
    if flags != p.flagged_label_ids:
        raise ValueError("MODULE_DECISIONS_DISAGREE")
    for i, label in enumerate(labels):
        if (
            float(result.arrays["scores"][0, i]) != p.scores[label]
            or float(result.arrays["thresholds"][0, i]) != p.thresholds[label]
        ):
            raise ValueError("MODULE_SCORE_ARRAY_DISAGREEMENT")


def run_stage(tool, video, output, manifest_path, pipeline_path, inputs, clip_id, execution_id):
    import json

    import torch

    pipeline_sha = sha(pipeline_path)
    if tool not in {"rgb.predict", "fusion.predict"}:
        return accepted_stage(tool, video, output, manifest_path, pipeline_sha, inputs)
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError("Preserve prior module output")
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen.validate_bundle(manifest_path)
    config = json.loads(pipeline_path.read_text(encoding="utf-8"))
    registry = build_registry()
    selected = registry.validate_selection(config["module_slots"])
    video_sha, bundle_sha = sha(video), sha(manifest_path)
    sources, arrays, dependencies = {}, {}, {}
    for name in ["pose.extract"] if tool == "rgb.predict" else ["skeleton.predict", "rgb.predict"]:
        path = Path(inputs[name])
        sources[name] = read_stage(path, name, video_sha, bundle_sha, pipeline_sha)
        arrays[name] = load_arrays(path, sources[name])
        dependencies[name] = sha(path)
    module_id = "M5" if tool == "rgb.predict" else "M2"
    spec = registry.spec(module_id, selected[module_id])
    module = registry.create(module_id, selected[module_id])
    output.parent.mkdir(parents=True, exist_ok=True)
    if tool == "rgb.predict":
        source, perception = sources["pose.extract"], arrays["pose.extract"]
        pose_sha = dependencies["pose.extract"]
        binding = Binding(
            clip_id=clip_id,
            video_sha256=video_sha,
            bundle_sha256=bundle_sha,
            pipeline_sha256=pipeline_sha,
            actor_source_sha256=pose_sha,
        )
        profile_id = "single-view-v15-branches-v1:" + pipeline_sha
        request = RGBInput(
            binding,
            execution_id,
            ArtifactRef(
                artifact_id=video_sha, clip_id=clip_id, kind="media", profile_id=profile_id, sha256=video_sha
            ),
            ArtifactRef(
                artifact_id=pose_sha, clip_id=clip_id, kind="pose", profile_id=profile_id, sha256=pose_sha
            ),
            video,
            perception["points"],
            perception["confidence"],
            source["decoded_frames"],
            source["reported_fps"],
        )
        result = module.predict(request)
        labels = RGB_LABEL_IDS
        validate_result(result, binding, execution_id, labels, "eligible")
        indices = result.audit["selected_frame_indices"]
        if (
            not indices
            or any(type(i) is not int or not 0 <= i < source["decoded_frames"] for i in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError("MODULE_SAMPLE_INDICES_INVALID")
        if result.audit["rgb_audit"]["requested_frames"] != len(indices):
            raise ValueError("MODULE_SAMPLE_COUNT_INVALID")
        raw = dict(
            schema="cpr-decomposed-stage-v1",
            decoded_frames=source["decoded_frames"],
            reported_fps=source["reported_fps"],
            shared_pose_sha256=pose_sha,
            **result.audit,
            modality="RGB/local shape plus shared actor and geometric context; not pose-free RGB",
        )
    else:
        skeleton, rgb = sources["skeleton.predict"], sources["rgb.predict"]
        if (
            skeleton["shared_pose_sha256"] != rgb["shared_pose_sha256"]
            or skeleton["decoded_frames"] != rgb["decoded_frames"]
            or skeleton["reported_fps"] != rgb["reported_fps"]
        ):
            raise ValueError("CROSS_TRACK_FUSION")
        binding = Binding(
            clip_id=clip_id,
            video_sha256=video_sha,
            bundle_sha256=bundle_sha,
            pipeline_sha256=pipeline_sha,
            actor_source_sha256=skeleton["shared_pose_sha256"],
        )
        quality = QualityIndicators(
            rgb_usable_samples=rgb["rgb_audit"]["frames"],
            rgb_requested_samples=rgb["rgb_audit"]["requested_frames"],
            hand_shape_available=rgb["hand_shape_audit"]["available"],
        )
        request = FusionInput(
            binding,
            execution_id,
            Readout(
                from_stage(skeleton, arrays["skeleton.predict"], binding, dependencies["skeleton.predict"]),
                arrays["skeleton.predict"],
            ),
            Readout(
                from_stage(rgb, arrays["rgb.predict"], binding, dependencies["rgb.predict"]),
                arrays["rgb.predict"],
            ),
            quality,
        )
        result = module.fuse(request)
        labels = LABEL_IDS
        validate_result(result, binding, execution_id, labels, "predictions")
        raw = dict(
            schema="cpr-offline-analysis-v2",
            decoded_frames=skeleton["decoded_frames"],
            reported_fps=skeleton["reported_fps"],
            estimated_supported_error_count=int(result.arrays["predicted_count"][0]),
            rgb_audit=rgb["rgb_audit"],
            hand_shape_audit=rgb["hand_shape_audit"],
            fullpose_audit=skeleton["fullpose_audit"],
            public_label_verification=None,
            scope="Module slots with baseline implementations; whole-clip evidence",
            **result.audit,
        )
    module_receipt = {
        "module_id": module_id,
        "implementation_id": spec.implementation_id,
        "kind": spec.kind,
        "contract_version": CONTRACT_VERSION,
        "binding": binding.model_dump(),
        "prediction_id": execution_id,
    }
    executions = {module_id: module_receipt}
    if tool == "fusion.predict":
        executions.update(sources["rgb.predict"]["module_executions"])
    modified = any(e["kind"] == "improved" for e in executions.values())
    raw.update(
        status="completed_experimental",
        tool_id=tool,
        video_sha256=video_sha,
        bundle_sha256=bundle_sha,
        pipeline_sha256=pipeline_sha,
        labels_used=False,
        model_or_threshold_modified=modified,
        module_contract_version=CONTRACT_VERSION,
        module_executions=executions,
        dependencies=dependencies,
        supported_label_ids=list(labels),
        predicted_original_label_ids=result.prediction.flagged_label_ids,
        scores=[
            {
                "original_label_id": label,
                "uncalibrated_score": result.prediction.scores[label],
                "threshold": result.prediction.thresholds[label],
            }
            for label in labels
        ],
        files=[array_file(output.with_suffix(".npz"), **result.arrays)],
        analysis_seconds=time.perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    save(output, raw)
    return raw
