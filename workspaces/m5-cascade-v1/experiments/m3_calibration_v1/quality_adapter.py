"""Controller-only provenance checks for M3 quality features. No ML imports or data writes."""

import json

from .branch_vision import artifact_path, validate_branch
from .modules.contracts import QualityIndicators


def bound_quality(store, settings, execution, output, raw):
    branch = {}
    for binding, tool in [("skeleton_receipt_id", "skeleton.predict"), ("rgb_receipt_id", "rgb.predict")]:
        source_id = execution["request"]["input_bindings"].get(binding)
        if not source_id:
            raise ValueError("M3_BRANCH_BINDING_MISSING")
        record = store.get_tool(source_id)
        if record["status"] != "succeeded" or record["request"]["tool_id"] != tool:
            raise ValueError("M3_BRANCH_NOT_SUCCEEDED")
        receipt = validate_branch(record["output"], store, settings)
        if (
            receipt.clip_id != output.clip_id
            or receipt.video_sha256 != output.video_sha256
            or receipt.bundle_sha256 != output.bundle_sha256
            or receipt.profile_id != output.profile_id
            or raw["dependencies"].get(tool) != receipt.artifact_sha256
        ):
            raise ValueError("M3_CROSS_SOURCE_QUALITY")
        branch[tool] = json.loads(artifact_path(receipt, settings).read_text(encoding="utf-8"))
    sk, rg = branch["skeleton.predict"], branch["rgb.predict"]
    if sk["shared_pose_sha256"] != rg["shared_pose_sha256"] or sk["decoded_frames"] != rg["decoded_frames"]:
        raise ValueError("M3_CROSS_ACTOR_QUALITY")
    audit = raw.get("fullpose_audit", {}).get("m1_stability", {})
    observed, frames = audit.get("original_observed_points"), audit.get("frames")
    pose = None
    if (
        type(observed) is int
        and type(frames) is int
        and frames == raw["decoded_frames"]
        and frames > 0
        and 0 <= observed <= frames * 17
    ):
        pose = observed / (frames * 17)
    sk_flags, rg_flags = set(sk["predicted_original_label_ids"]), set(rg["predicted_original_label_ids"])
    return QualityIndicators(
        pose_visible_fraction=pose,
        rgb_usable_samples=raw.get("rgb_audit", {}).get("frames"),
        rgb_requested_samples=raw.get("rgb_audit", {}).get("requested_frames"),
        hand_shape_available=raw.get("hand_shape_audit", {}).get("available"),
        extras={
            "disagreement_" + str(label): float((label in sk_flags) != (label in rg_flags))
            for label in rg["supported_label_ids"]
        },
    )
