"""Experimental, label-independent video -> pose -> frozen encoder -> small head.

Optional public-reference verification runs strictly after prediction. No API,
Agent tool or application release flag is enabled by this offline CLI.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from paths import CODE, DATA

sys.path.insert(0, str(CODE / "experiments/public_cpr_baseline"))
from baseline import FrozenEncoder, digest, encoder_profile, LABEL_IDS, LABEL_NAMES, write_json
from geometry import describe
from quality_head import load
from improve import predict
from raw_pose_adapter import load_pose, decode_pose
from alpha_refiner import AlphaRefiner


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cpr-offline-pipeline-v1":
        raise ValueError("Unsupported inference manifest")
    for item in manifest["pinned_files"]:
        actual = CODE / item["path"]
        if not actual.is_file() or digest(actual) != item["sha256"]:
            raise ValueError("Frozen asset/source mismatch: " + item["path"])
    return manifest


def analyze(video, output, manifest_path, reference_record=None):
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError("Analysis output exists; preserve prior results")
    if not video.is_file(): raise ValueError("Video file is missing")
    manifest = load_manifest(manifest_path)
    protocol_path = CODE / manifest["protocol"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    encoder = FrozenEncoder(CODE / manifest["encoder"])
    _, encoder_hash = encoder_profile(encoder)
    if encoder_hash != protocol["encoder_profile_hash"]:
        raise ValueError("Current encoder differs from classifier contract")
    head, state = load(CODE / manifest["head"], protocol["feature_profile_hash"], digest(protocol_path))
    pose = load_pose(CODE / manifest["detector"])
    refiner = AlphaRefiner()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    record, audit = decode_pose(video, pose, track_person=True, refiner=refiner)
    pose_feature = encoder.encode([record])
    geometry, quality = describe(record["keypoint"][0], record["keypoint_score"][0])
    if not quality["geometry_valid"]: raise ValueError("No usable body geometry; no classification emitted")
    features = np.concatenate([pose_feature, geometry[None]], 1)
    scores = head.probability(features[:, state["feature_indices"].numpy()])
    operating = state["operating_point"]
    predictions = predict(scores, operating["thresholds"], operating["normal_gate"])
    repeated = encoder.encode([record])
    repeat_difference = float(np.abs(pose_feature - repeated).max())
    if repeat_difference > 1e-5: raise AssertionError("Non-reproducible frozen encoding")
    predicted_ids = [i for i, yes in zip(LABEL_IDS, predictions[0]) if yes]
    video_hash = digest(video)
    result = {"schema": "cpr-offline-analysis-v1", "status": "completed_experimental",
        "video_sha256": video_hash, "manifest_sha256": digest(manifest_path),
        "adapter_profile": manifest["input_profile"], **audit, "geometry_quality": quality,
        "predicted_original_label_ids": predicted_ids,
        "scores": [{"original_label_id": i, "name": name, "uncalibrated_score": float(score),
                    "threshold": float(threshold)} for i, name, score, threshold in
                   zip(LABEL_IDS, LABEL_NAMES, scores[0, :11], operating["thresholds"])],
        "normal_gate": operating["normal_gate"], "repeat_feature_max_difference": repeat_difference,
        "alpha_dcn_zero_offset_max_difference": refiner.zero_offset_difference,
        "analysis_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "model_or_threshold_modified": False, "public_label_verification": None,
        "empty_error_list_semantics": "No supported error exceeded thresholds; not proof of correct CPR",
        "runtime_release_approved": False,
        "scope": "Offline engineering verification; no clinical, all-class raw-video or cross-person accuracy claim"}
    # Ground truth is deliberately unavailable to all stages above this line.
    if reference_record:
        registered = manifest["public_samples"].get(video_hash)
        if not registered or registered["record_id"] != reference_record:
            raise ValueError("Video/reference mismatch: only registered public samples may be scored")
        records = json.loads((DATA / "run/records.json").read_text(encoding="utf-8"))
        source = next(r for r in records if r["record_id"] == reference_record)
        if source["expected_frames"] != audit["decoded_frames"]:
            raise ValueError("Public source frame count mismatch")
        target = [i for i in LABEL_IDS if i in source["original_labels"]]
        result["public_label_verification"] = {"record_id": reference_record,
            "original_labels": source["original_labels"], "evaluated_labels": target,
            "exact_label_match": target == predicted_ids, "role": registered["role"],
            "labels_from": "Existing public CPR-Coach annotations; no manual labels"}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), **record, features=features,
                        scores=scores, predictions=predictions)
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-record", default=None)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("inference-manifest.json"))
    args = parser.parse_args()
    result = analyze(args.video, args.output, args.manifest, args.reference_record)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
