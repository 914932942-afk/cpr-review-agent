"""Optional M1 derived-feature processing on top of the immutable four-tool baseline."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/branch_pipeline_v1"))
from algorithm import Config, features  # noqa: E402
from pipeline import (  # noqa: E402
    LABELS,
    SCHEMA,
    array_file,
    frozen,
    load_arrays,
    read_stage,
    save,
    sha,
    skeleton_readout,
)
from pipeline import (  # noqa: E402
    run_stage as baseline_stage,
)

SELECTION = ROOT / "data/m1-optimization-v2/experiment-03/selection.json"


def selected_config():
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    decision = json.loads(SELECTION.with_name("decision.json").read_text(encoding="utf-8"))
    if decision.get("accepted") is not True or decision.get("selected") != selection.get("name"):
        raise ValueError("M1_SELECTION_NOT_ACCEPTED")
    return Config(**selection["config"])


def transform(full, points, confidence, config):
    geometry, motion, audit = features(points, confidence, config)
    updated = np.asarray(full).copy()
    if updated.shape != (1, 1509):
        raise ValueError("M1 full-feature layout mismatch")
    updated[:, 512:783] = geometry[None]
    return updated, motion[None], audit


def run_stage(tool, video, output, manifest_path, pipeline_sha, inputs):
    if tool != "skeleton.predict":
        return baseline_stage(tool, video, output, manifest_path, pipeline_sha, inputs)
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError("Preserve previous M1 output")
    started = time.perf_counter()
    frozen.validate_bundle(manifest_path)
    source_path = Path(inputs["pose.extract"])
    video_sha, bundle_sha = sha(video), sha(manifest_path)
    source = read_stage(source_path, "pose.extract", video_sha, bundle_sha, pipeline_sha)
    perception = load_arrays(source_path, source)
    config = selected_config()
    full, motion, audit = transform(
        perception["full"], perception["points"], perception["confidence"], config
    )
    result = skeleton_readout(full, motion)
    value = {
        "schema": SCHEMA,
        "status": "completed_experimental",
        "tool_id": tool,
        "video_sha256": video_sha,
        "bundle_sha256": bundle_sha,
        "pipeline_sha256": pipeline_sha,
        "labels_used": False,
        "model_or_threshold_modified": False,
        "feature_preprocessing_modified": True,
        "dependencies": {"pose.extract": sha(source_path)},
        "shared_pose_sha256": sha(source_path),
        "decoded_frames": source["decoded_frames"],
        "reported_fps": source["reported_fps"],
        "fullpose_audit": {**source["fullpose_audit"], "m1_stability": audit},
        "supported_label_ids": LABELS,
        "scores": [
            {
                "original_label_id": label,
                "uncalibrated_score": float(result["scores"][0, i]),
                "threshold": float(result["thresholds"][0, i]),
            }
            for i, label in enumerate(LABELS)
        ],
        "predicted_original_label_ids": [label for label, yes in zip(LABELS, result["eligible"][0]) if yes],
        "decision_scope": "M1 branch candidates before unchanged cross-branch fusion",
        "files": [array_file(output.with_suffix(".npz"), **result)],
        "analysis_seconds": time.perf_counter() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    save(output, value)
    return value
