"""Step 2: recording-disjoint public RGB reservations and a refitted small head.

Only cached, frozen public-keypoint features are used here. No RGB error
recognition, perceptual backbone training or web/Agent release is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

CODE = Path(__file__).resolve().parents[2]
DATA = CODE / "data/public_cpr"
BASE = DATA / "improvement-v2"
COLLECTION = DATA / "rgb-acceptance-v1"
OUTPUT = DATA / "rgb-evaluation-v1"
REPORT = CODE / "test-results/public-rgb-protocol-v1"
IMPROVED = CODE / "experiments/public_cpr_improved"
sys.path.insert(0, str(IMPROVED))
from improve import LABEL_IDS, digest, metrics, predict, read
from quality_head import load
from infer_video import load_manifest
import fit_quality_head


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def group_id(record_id):
    parts = record_id.split("/")
    if len(parts) < 3 or any(p in ("", ".", "..") for p in parts) or parts[-1] not in {"ch0", "ch1", "ch2", "ch3"}:
        raise ValueError("Invalid canonical recording/view ID")
    return "/".join(parts[:-1])


def reserve_groups(records, reservations):
    wanted = {r["record_id"] for r in reservations}
    if len(wanted) != len(reservations) or not wanted.issubset({r["record_id"] for r in records}):
        raise ValueError("Duplicate or missing RGB reservation")
    reserved = {group_id(k) for k in wanted}
    result = []
    for record in records:
        row = dict(record)
        previous = row["application_split"]
        if previous not in {"train", "validation", "test"}:
            raise ValueError("Unsupported parent split")
        row["previous_application_split"] = previous
        row["recording_group"] = group_id(row["record_id"])
        row["reserved_for_rgb_acceptance"] = row["recording_group"] in reserved
        if row["reserved_for_rgb_acceptance"]:
            row["application_split"] = "test"
        result.append(row)
    validate_groups(result, reserved)
    return result


def validate_groups(records, reserved):
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Duplicate record IDs")
    membership = defaultdict(set)
    for row in records:
        group = group_id(row["record_id"])
        split = row["application_split"]
        membership[group].add(split)
        if group in reserved and split != "test":
            raise ValueError("Reserved recording leaked into development")
        if row.get("original_partition") == "official_single_test" and split != "test":
            raise ValueError("Original single-error test moved into development")
        if row.get("previous_application_split") == "test" and split != "test":
            raise ValueError("Historical test moved into development")
    if any(len(v) != 1 for v in membership.values()):
        raise ValueError("Camera/recording group crosses partitions")


def protected_sources():
    manifest_path = IMPROVED / "inference-manifest.json"
    manifest = load_manifest(manifest_path)
    protected = {item["path"]: item["sha256"] for item in manifest["pinned_files"]}
    additional = [manifest_path, BASE / "record-splits.json", BASE / "development.npz", BASE / "locked-test.npz",
                  BASE / "selection.json", COLLECTION / "collection-manifest.json", COLLECTION / "frozen-record-plan.json",
                  DATA / "assets/ann.tar.gz", DATA / "run/input-duplicate-audit.json"]
    for path in additional:
        protected[path.relative_to(CODE).as_posix()] = digest(path)
    application = read(CODE / "test-results/public-head-validation/application-source-before.json")
    for name, expected in application.items():
        key = Path(name).as_posix()
        if digest(CODE / name) != expected:
            raise ValueError("Application differs from the previously verified scaffold")
        protected[key] = expected
    return protected


def check_hashes(items):
    for name, expected in items.items():
        if digest(CODE / name) != expected:
            raise ValueError("Protected asset/source changed: " + name)


def checked_protocol(output):
    protocol = read(output / "protocol.json")
    check_hashes(protocol["protected_sources"])
    check_hashes(protocol["implementation_sources"])
    for key, name in (("split_sha256", "record-splits.json"), ("development_sha256", "development.npz"),
                      ("locked_test_sha256", "locked-test.npz")):
        if digest(output / name) != protocol[key]:
            raise ValueError("Frozen protocol input changed: " + name)
    return protocol


def load_parent_features(records, parent):
    rows = {}
    for name, expected in (("development", parent["development_sha256"]), ("locked-test", parent["locked_test_sha256"])):
        path = BASE / f"{name}.npz"
        if digest(path) != expected:
            raise ValueError("Parent features changed")
        with np.load(path, allow_pickle=False) as bundle:
            if not np.array_equal(bundle["label_ids"], LABEL_IDS) or str(bundle["profile_hash"]) != parent["feature_profile_hash"]:
                raise ValueError("Parent feature/class profile differs")
            for rid, feature, target in zip(bundle["record_ids"], bundle["features"], bundle["targets"]):
                if str(rid) in rows:
                    raise ValueError("Duplicate feature ID")
                rows[str(rid)] = (feature.copy(), target.copy())
    if set(rows) != {r["record_id"] for r in records}:
        raise ValueError("Missing/extra frozen features")
    features = np.stack([rows[r["record_id"]][0] for r in records])
    targets = np.stack([rows[r["record_id"]][1] for r in records])
    expected = np.asarray([[float(i in r["original_labels"]) for i in LABEL_IDS] for r in records], dtype=np.float32)
    if features.shape != (len(records), 783) or not np.isfinite(features).all() or not np.array_equal(targets, expected):
        raise ValueError("Feature dimensions, values or public targets differ")
    if any(not t.any() and r["original_labels"] != [0] for r, t in zip(records, targets)):
        raise ValueError("Unsupported depth-only error must not become Correct")
    signatures = [hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in features]
    if len(set(signatures)) != len(signatures):
        raise ValueError("Exact duplicate features require group investigation")
    return features, targets


def prepare(output):
    if output.exists():
        raise FileExistsError("Use a new protocol directory; never overwrite a prior attempt")
    parent = read(BASE / "protocol.json")
    collection = read(COLLECTION / "collection-manifest.json")
    fingerprint = read(CODE / "test-results/public-rgb-acceptance-v1/collection-fingerprint.json")
    if digest(COLLECTION / "collection-manifest.json") != fingerprint["sha256"] or not collection["step_1_complete"]:
        raise ValueError("Step 1 collection is not frozen/complete")
    records = read(BASE / "record-splits.json")
    source = {r["record_id"]: r for r in read(DATA / "run/records.json")}
    if digest(BASE / "record-splits.json") != parent["split_sha256"] or set(source) != {r["record_id"] for r in records}:
        raise ValueError("Parent splits/record universe changed")
    if any(r["original_labels"] != source[r["record_id"]]["original_labels"] for r in records):
        raise ValueError("Public annotations changed")
    for row in collection["records"]:
        original = source[row["record_id"]]
        if row["original_labels"] != original["original_labels"] or row["expected_frames"] != original["expected_frames"]:
            raise ValueError("RGB/public annotation mismatch")
        if row["recording_group"] != group_id(row["record_id"]):
            raise ValueError("RGB recording grouping differs")
        if digest(COLLECTION / row["video"]) != row["video_sha256"]:
            raise ValueError("Reserved RGB media changed")
    records = reserve_groups(records, collection["records"])
    features, targets = load_parent_features(records, parent)
    if read(DATA / "run/input-duplicate-audit.json")["exact_duplicate_groups"]:
        raise ValueError("Unresolved source-pose duplicates")
    splits = np.array([r["application_split"] for r in records])
    for name in ("train", "validation"):
        target = targets[splits == name]
        if not len(target) or (target.sum(0) == 0).any() or not (~target.any(1)).any():
            raise ValueError("Training/validation lacks a supported error or Correct")
    protected = protected_sources()
    output.mkdir(parents=True)
    write_new(output / "record-splits.json", records)
    for name, mask in (("development", splits != "test"), ("locked-test", splits == "test")):
        with (output / f"{name}.npz").open("xb") as handle:
            np.savez_compressed(handle, features=features[mask], targets=targets[mask], splits=splits[mask],
                record_ids=np.array([r["record_id"] for r in records])[mask],
                cohorts=np.array([r["test_cohort"] for r in records])[mask], label_ids=np.array(LABEL_IDS),
                profile_hash=np.array(parent["feature_profile_hash"]))
    audit = {"records": len(records), "recording_groups": len({group_id(r["record_id"]) for r in records}),
        "split_counts": dict(Counter(splits)),
        "normal_counts": dict(Counter(r["application_split"] for r in records if r["original_labels"] == [0])),
        "reserved_rgb_records": len(collection["records"]), "reserved_rgb_groups_all_test": True,
        "moved_to_test": dict(Counter(r["previous_application_split"] for r in records
             if r["application_split"] != r["previous_application_split"])),
        "existing_test_retained": True, "all_labels_unchanged": True, "cross_split_recording_overlap": [],
        "exact_feature_duplicates": 0, "subject_disjoint_claim": False}
    write_new(output / "split-audit.json", audit)
    protocol = {**parent, "schema": 2, "purpose": "Step 2: frozen recording reservations and small-head refit",
        "frozen_at": datetime.now(timezone.utc).isoformat(), "parent_protocol_sha256": digest(BASE / "protocol.json"),
        "parent_protocol": (BASE / "protocol.json").relative_to(CODE).as_posix(),
        "collection_manifest_sha256": digest(COLLECTION / "collection-manifest.json"),
        "split_sha256": digest(output / "record-splits.json"), "development_sha256": digest(output / "development.npz"),
        "locked_test_sha256": digest(output / "locked-test.npz"), "split_counts": audit["split_counts"],
        "normal_counts": audit["normal_counts"], "test_cohorts": dict(Counter(r["test_cohort"] for r in records if r["application_split"] == "test")),
        "candidate_configs": [{"features": "combined", "trees": 64}],
        "candidate_policy": "One inherited fixed configuration; no new architecture/hyperparameter search or test-driven selection",
        "development_search_disclosure": "Prior pipeline search is disclosed in the parent protocol; this refit uses only the inherited combined-64 configuration",
        "raw_test_reservations": [r["record_id"] for r in collection["records"]],
        "raw_video_inference_performed_in_step2": False, "training_features": "Unchanged public-keypoint frozen 512D PoseC3D +271D geometry features",
        "protected_sources": protected,
        "implementation_sources": {p.relative_to(CODE).as_posix(): digest(p) for p in
            [Path(__file__).resolve(), IMPROVED / "fit_quality_head.py"]},
        "previous_test_exposure": "Prior pipeline used 20 RGB recording skeletons for fit, 2 for tuning; 4 raw-video outputs and prior skeleton aggregates were inspected. New head excludes all 26 from fit/tuning. Not an independent external blind test.",
        "independent_unseen_test_claim": False, "raw_development_record": None,
        "threshold_scope": "Validation records only; never fit/tune on reserved RGB or locked test records"}
    write_new(output / "protocol.json", protocol)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def train(output):
    protocol = checked_protocol(output)
    if (output / "selection.json").exists() or list(output.glob("*.pt")):
        raise FileExistsError("Preserve existing fitting attempt")
    with np.load(output / "development.npz", allow_pickle=False) as bundle:
        ids = list(map(str, bundle["record_ids"]))
        if {group_id(k) for k in ids} & {group_id(k) for k in protocol["raw_test_reservations"]}:
            raise ValueError("Reserved RGB leaked into head fitting/tuning")
    actual_load = np.load
    accessed = []
    def development_only(path, *args, **kwargs):
        if Path(path).resolve() != (output / "development.npz").resolve():
            raise PermissionError("Fitting may only open the development feature bundle")
        accessed.append(Path(path).relative_to(CODE).as_posix())
        return actual_load(path, *args, **kwargs)
    with patch.object(fit_quality_head.np, "load", side_effect=development_only):
        fit_quality_head.train(output)
    check_hashes(protocol["protected_sources"])
    write_new(output / "fit-access-audit.json", {"passed": True, "npz_files_decoded_by_fitter": accessed,
        "test_npz_decoded_by_fitter": False, "locked_test_integrity_hashed_before_fit": True,
        "reserved_rgb_in_fit_or_tuning": 0,
        "training_scope": "Small tree classification head only; all visual backbones frozen"})


def evaluate(output):
    protocol = checked_protocol(output)
    if (output / "evaluation.json").exists():
        raise FileExistsError("First test evaluation is frozen")
    selection = read(output / "selection.json")
    if selection["protocol_sha256"] != digest(output / "protocol.json"):
        raise ValueError("Selection/protocol mismatch")
    chosen = selection["selected"]
    checkpoint = output / chosen["checkpoint"]
    if digest(checkpoint) != chosen["checkpoint_sha256"] or not read(output / "fit-access-audit.json")["passed"]:
        raise ValueError("Fitting/checkpoint audit failed")
    head, state = load(checkpoint, protocol["feature_profile_hash"], digest(output / "protocol.json"))
    with np.load(output / "locked-test.npz", allow_pickle=False) as bundle:
        if not np.array_equal(bundle["label_ids"], LABEL_IDS) or str(bundle["profile_hash"]) != protocol["feature_profile_hash"]:
            raise ValueError("Test profile/class order differs")
        x, y, ids, cohorts = [bundle[k].copy() for k in ("features", "targets", "record_ids", "cohorts")]
    scores = head.probability(x[:, state["feature_indices"].numpy()])
    op = state["operating_point"]
    predictions = predict(scores, op["thresholds"], op["normal_gate"])
    reserved = np.array([str(k) in protocol["raw_test_reservations"] for k in ids])
    evaluations = {}
    for name, mask in (("all", np.ones(len(y), bool)), ("single", cohorts == "single"),
                       ("composite", cohorts == "composite"), ("rgb_reserved_public_keypoints", reserved)):
        measured = metrics(predictions[mask].astype(float), y[mask])
        ranked = metrics(scores[mask, :11], y[mask])
        measured["mAP"] = ranked["mAP"]
        for row, rank in zip(measured["per_label"], ranked["per_label"]):
            row["average_precision"] = rank["average_precision"]
        evaluations[name] = measured
    gates = protocol["gates"]
    checks = {"single_f1": evaluations["single"]["macro_f1"] >= gates["single_macro_f1"],
        "composite_f1": evaluations["composite"]["macro_f1"] >= gates["composite_macro_f1"],
        "correct_false_alarm": evaluations["single"]["correct_false_alarm_rate"] <= gates["correct_false_alarm_rate"],
        "all_label_recall": min(r["recall"] for r in evaluations["all"]["per_label"]) >= gates["minimum_label_recall"],
        "all_386_retained": len(y) == protocol["split_counts"]["test"],
        "all_26_reserved_retained": int(reserved.sum()) == len(protocol["raw_test_reservations"]),
        "head_budget": chosen["scalar_fields"] <= gates["maximum_head_parameters"]}
    with (output / "test-predictions.npz").open("xb") as handle:
        np.savez_compressed(handle, scores=scores, predictions=predictions, targets=y, record_ids=ids)
    source_records = {r["record_id"]: r for r in read(output / "record-splits.json")}
    result = {"evaluations": evaluations, "quality_checks": checks, "public_keypoint_quality_passed": all(checks.values()),
        "split_and_fit_contract_passed": True, "raw_video_acceptance_performed": False, "raw_video_quality_passed": False,
        "runtime_release_approved": False, "public_keypoint_feature_source_not_raw_video": True,
        "protocol_sha256": digest(output / "protocol.json"), "selection_sha256": digest(output / "selection.json"),
        "evaluation_frozen_before_any_test_based_changes": True, "independent_unseen_test_claim": False,
        "reserved_per_record": [{"record_id": str(k), "original_labels": source_records[str(k)]["original_labels"],
             "target_supported_error_ids": [v for v, yes in zip(LABEL_IDS, target) if yes],
             "predicted_supported_error_ids": [v for v, yes in zip(LABEL_IDS, pred) if yes], "exact_match": bool(np.array_equal(target, pred))}
             for k, target, pred in zip(ids[reserved], y[reserved], predictions[reserved])]}
    write_new(output / "evaluation.json", result)
    print(json.dumps({"quality_checks": checks, "summary": {k: {n: v[n] for n in
        ("records", "macro_f1", "correct_false_alarm_rate")} for k, v in evaluations.items()}}, ensure_ascii=False, indent=2), flush=True)


def export_manifest(output):
    protocol = checked_protocol(output)
    evaluation = read(output / "evaluation.json")
    chosen = read(output / "selection.json")["selected"]
    old = read(IMPROVED / "inference-manifest.json")
    collection = read(COLLECTION / "collection-manifest.json")
    replaced = {"data/public_cpr/improvement-v2/combined-64.pt", "data/public_cpr/improvement-v2/protocol.json"}
    pinned = [item for item in old["pinned_files"] if item["path"] not in replaced]
    for path in [output / chosen["checkpoint"], output / "protocol.json", output / "selection.json", output / "record-splits.json",
                 COLLECTION / "collection-manifest.json", Path(__file__).resolve(), IMPROVED / "fit_quality_head.py"]:
        pinned.append({"path": path.relative_to(CODE).as_posix(), "sha256": digest(path)})
    manifest = {**old, "frozen_at": datetime.now(timezone.utc).isoformat(), "head": (output / chosen["checkpoint"]).relative_to(CODE).as_posix(),
        "protocol": (output / "protocol.json").relative_to(CODE).as_posix(), "pinned_files": pinned,
        "public_samples": {r["video_sha256"]: {"record_id": r["record_id"],
            "video": (COLLECTION / r["video"]).relative_to(CODE).as_posix(),
            "role": "heldout_recording_engineering_acceptance_prior_pipeline_exposed"} for r in collection["records"]},
        "disclosure": protocol["previous_test_exposure"], "runtime_release_approved": False,
        "raw_video_acceptance_pending": True, "public_keypoint_quality_passed": evaluation["public_keypoint_quality_passed"]}
    manifest.pop("raw_test_record", None)
    manifest.pop("raw_test_video_sha256", None)
    path = output / "inference-manifest.json"
    write_new(path, manifest)
    load_manifest(path)
    check_hashes(protocol["protected_sources"])
    print("New offline manifest validated; original default manifest unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "evaluate", "export"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate, "export": export_manifest}[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
