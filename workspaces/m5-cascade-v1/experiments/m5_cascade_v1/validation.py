"""Web-safe verification of M5 scope/certificate metadata; no torch/numpy imports."""

import json
from pathlib import Path
from zipfile import ZipFile


def validate_visual_metadata(raw, settings):
    from .config import ROOT
    from .vision import VisionError

    profile = json.loads(settings.branch_manifest.read_text(encoding="utf-8"))
    if profile.get("module_slots", {}).get("M5") != "quality-dependency-cascade-v1":
        return
    certificate = json.loads((ROOT / "data/m5-cascade-v1/certificate.json").read_text(encoding="utf-8"))
    hand = raw.get("hand_shape_audit", {})
    rgb = raw.get("rgb_audit", {})
    audit = raw.get("m5_cascade", {})
    available = hand.get("available")
    if type(available) is not bool:
        raise VisionError("M5_QUALITY_GATE_INVALID")
    expected = certificate["available" if available else "unavailable"]
    if any(audit.get(k) != v for k, v in expected.items()):
        raise VisionError("M5_DEPENDENCY_CERTIFICATE_INVALID")
    count = rgb.get("frames")
    requested = rgb.get("requested_frames")
    if type(count) is not int or type(requested) is not int or not 8 <= count <= requested <= 16:
        raise VisionError("M5_VISUAL_SUPPORT_INVALID")
    if requested != min(16, raw.get("decoded_frames", 0)) or hand.get("requested_frames") != requested:
        raise VisionError("M5_SAMPLE_SUPPORT_CHANGED")
    usable = hand.get("usable_frames")
    if type(usable) is not int or not 0 <= usable <= requested or available != (usable >= 8):
        raise VisionError("M5_SHAPE_GATE_CHANGED")
    if audit.get("dense_visual_crop_evaluations") != (
        0 if expected["route"] == "dependency-pruned" else 2 * count
    ):
        raise VisionError("M5_VISUAL_COST_AUDIT_INVALID")
    if (
        audit.get("weights_and_thresholds_unchanged") is not True
        or audit.get("observed_frames_unchanged") is not True
        or audit.get("uncomputed_dense_features_persisted") is not False
    ):
        raise VisionError("M5_COMPUTATION_BOUNDARY_INVALID")


def validate_visual_archive(raw, artifact):
    from .vision import VisionError

    if raw.get("m5_cascade", {}).get("algorithm") != "quality-dependency-cascade-v1":
        return
    array = Path(artifact).parent / raw["files"][0]["name"]
    with ZipFile(array) as z:
        keys = {Path(n).stem for n in z.namelist()}
    pruned = raw["m5_cascade"]["route"] == "dependency-pruned"
    if pruned and (
        "rgb_features" in keys
        or not {"context_features", "computed_rgb_feature_indices", "shape_features"} <= keys
    ):
        raise VisionError("M5_UNCOMPUTED_FEATURE_PERSISTED")
    if not pruned and "rgb_features" not in keys:
        raise VisionError("M5_FALLBACK_FEATURE_MISSING")


def visual_summary(store, settings, prediction):
    from .branch_vision import artifact_path, validate_branch

    if (
        not prediction
        or prediction.analysis_route != "full-v15"
        or getattr(settings, "vision_pipeline", "monolithic") != "branches"
    ):
        return None
    profile = json.loads(settings.branch_manifest.read_text(encoding="utf-8"))
    if profile.get("module_slots", {}).get("M5") != "quality-dependency-cascade-v1":
        return None
    final = store.get_tool(prediction.execution_id)
    source = store.get_tool(final["request"]["input_bindings"]["rgb_receipt_id"])
    receipt = validate_branch(source["output"], store, settings)
    raw = json.loads(artifact_path(receipt, settings).read_text(encoding="utf-8"))
    audit = raw.get("m5_cascade")
    if not audit:
        return None
    return {
        "mode": "lightweight" if audit["route"] == "dependency-pruned" else "full_visual",
        "sampled_frames": raw["rgb_audit"]["frames"],
        "requested_frames": raw["rgb_audit"]["requested_frames"],
        "hand_details_available": raw["hand_shape_audit"]["available"],
        "scope_unchanged": True,
        "source_execution_id": receipt.execution_id,
    }
