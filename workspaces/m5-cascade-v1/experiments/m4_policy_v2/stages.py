"""Route adapter around unchanged M1/M2 stages. Body finalisation does not claim RGB fusion."""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/branch_pipeline_v1"))
from pipeline import LABELS, array_file, load_arrays, read_stage, save, sha  # noqa: E402
from slots_pipeline import run_stage as full_stage  # noqa: E402

BODY = [3, 4, 8, 11]


def run_stage(tool, video, output, manifest_path, pipeline_path, inputs, clip_id, execution_id):
    inputs = dict(inputs)
    route = inputs.pop("__route_profile__", "full-v15")
    if route not in ["full-v15", "body-qualified-v1"]:
        raise ValueError("M4_ROUTE_NOT_REGISTERED")
    if route != "body-qualified-v1" or tool != "fusion.predict":
        return full_stage(tool, video, output, manifest_path, pipeline_path, inputs, clip_id, execution_id)
    started = time.perf_counter()
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError("Preserve previous final output")
    profile = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if (
        profile["module_slots"]["M4"] != "consent-coverage-dag-v2"
        or profile["m4_routes"]["body-qualified-v1"]["supported_label_ids"] != BODY
    ):
        raise ValueError("M4_ROUTE_NOT_CERTIFIED")
    if set(inputs) != set(["skeleton.predict"]):
        raise ValueError("M4_BODY_DEPENDENCIES_INVALID")
    path = Path(inputs["skeleton.predict"])
    source = read_stage(path, "skeleton.predict", sha(video), sha(manifest_path), sha(pipeline_path))
    arrays = load_arrays(path, source)
    cols = [LABELS.index(k) for k in BODY]
    values = {
        "scores": arrays["scores"][:, cols],
        "thresholds": arrays["thresholds"][:, cols],
        "eligible": arrays["eligible"][:, cols],
        "predictions": arrays["eligible"][:, cols].copy(),
    }
    result = {
        "schema": "cpr-offline-analysis-v2",
        "status": "completed_experimental",
        "tool_id": tool,
        "analysis_route": "body-qualified-v1",
        "video_sha256": sha(video),
        "bundle_sha256": sha(manifest_path),
        "pipeline_sha256": sha(pipeline_path),
        "decoded_frames": source["decoded_frames"],
        "reported_fps": source["reported_fps"],
        "fullpose_audit": source["fullpose_audit"],
        "labels_used": False,
        "public_label_verification": None,
        "model_or_threshold_modified": False,
        "module_contract_version": "cpr-modules-v1",
        "module_executions": {},
        "dependencies": {"skeleton.predict": sha(path)},
        "supported_label_ids": BODY,
        "predicted_original_label_ids": [label for label, yes in zip(BODY, values["predictions"][0]) if yes],
        "scores": [
            {
                "original_label_id": label,
                "uncalibrated_score": float(values["scores"][0, i]),
                "threshold": float(values["thresholds"][0, i]),
            }
            for i, label in enumerate(BODY)
        ],
        "files": [array_file(output.with_suffix(".npz"), **values)],
        "scope": "Certified body-only scope; RGB/M2/M5 not executed, other categories not evaluated.",
        "m4_finalization": {
            "profile_id": route,
            "rgb_executed": False,
            "m2_executed": False,
            "certificate": "Qualified non-RGB outputs protected by the frozen guard and M2 scope lock.",
        },
        "analysis_seconds": time.perf_counter() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    save(output, result)
    return result
