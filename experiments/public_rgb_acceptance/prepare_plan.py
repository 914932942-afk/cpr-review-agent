"""Freeze the public-label inventory before acquiring the remaining RGB data."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import json
import tarfile
import urllib.request
from common import CODE, DATA, ROOT, REPORT, REVISION, ARCHIVE, SIZE, SHA, ANN_SHA, LABELS, sha, read, write


def source_annotations(path, wanted):
    if sha(path) != ANN_SHA: raise ValueError("Public annotation archive changed")
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if PurePosixPath(member.name).name not in {"trainlist.txt","testlist.txt","all_errors_testlist.txt"}:
                continue
            if not member.isfile() or member.size > 2**20: raise ValueError("Unexpected annotation member")
            for line in archive.extractfile(member).read().decode("utf-8-sig").splitlines():
                parts = line.split()
                if not parts or parts[0] not in wanted: continue
                if parts[0] in result: raise ValueError("Repeated public annotation")
                result[parts[0]] = {"frame_count":int(parts[1]), "labels":list(map(int,parts[2:])),
                                    "annotation_member":member.name}
    if set(result) != wanted: raise ValueError("Selected public annotations missing")
    return result


def main():
    if (ROOT / "frozen-record-plan.json").exists(): raise FileExistsError("Record plan already frozen")
    rows = read(DATA / "improvement-v2/record-splits.json")
    selected = sorted([r for r in rows if r["record_id"].startswith("SupC") and
        r["record_id"].split("/")[0][4:].isdigit()], key=lambda r:r["record_id"])
    existing = [next(r for r in rows if r["record_id"] == rid) for rid in
                ["CPR_Dataset_S0/C13_S0/r11/ch0", "CPR_Dataset_S0/C13_S0/r12/ch0"]]
    if len(selected) != 24 or {r["original_labels"][0] for r in selected} != set(LABELS):
        raise ValueError("Unexpected supplemental annotation coverage")
    annotations = source_annotations(DATA / "assets/ann.tar.gz", {r["annotation_frame_dir"] for r in selected+existing})
    url = f"https://huggingface.co/api/datasets/ShunliWang/CPR-Coach/tree/{REVISION}?recursive=false&expand=false&limit=100"
    with urllib.request.urlopen(url, timeout=30) as response: entries = json.load(response)
    remote = next(r for r in entries if r["path"] == ARCHIVE)
    if remote["size"] != SIZE or remote["lfs"]["oid"] != SHA:
        raise ValueError("Immutable public RGB archive metadata mismatch")
    records = []
    for r in selected + existing:
        annotation = annotations[r["annotation_frame_dir"]]
        if annotation["labels"] != r["original_labels"] or annotation["frame_count"] != r["expected_frames"]:
            raise ValueError("Annotation/record mismatch: " + r["record_id"])
        current_split = r["application_split"]
        records.append({"record_id":r["record_id"], "recording_group":r["record_id"].rsplit("/",1)[0],
            "view":"ch0", "original_labels":r["original_labels"],
            "label_names":[LABELS[label] for label in r["original_labels"]],
            "expected_frames":r["expected_frames"], "annotation_frame_dir":r["annotation_frame_dir"],
            "annotation_member":annotation["annotation_member"], "current_head_split":current_split,
            "used_for_current_head_training":current_split == "train",
            "used_for_current_head_tuning":current_split == "validation",
            "prior_rgb_result_exposed":r["record_id"] in {"SupC02/r00/ch0","SupC02/r01/ch0",
                "CPR_Dataset_S0/C13_S0/r11/ch0","CPR_Dataset_S0/C13_S0/r12/ch0"},
            "current_use":"engineering_regression_only" if current_split in {"train","validation"} else "previously_exposed_regression",
            "independent_unseen_test_claim":False,
            "source_kind":"supplemental_archive" if r in selected else "existing_pinned_parquet_subset"})
    plan = {"schema":"public-rgb-record-plan-v1", "frozen_at":datetime.now(timezone.utc).isoformat(),
        "purpose":"Acquire public RGB + audit original labels; no training, inference or split changes in this step",
        "dataset_revision":REVISION, "archive":{"filename":ARCHIVE,"bytes":SIZE,"sha256":SHA,"remote_metadata":remote},
        "annotation_archive_sha256":ANN_SHA,"source_records_sha256":sha(DATA/"run/records.json"),
        "current_head_split_sha256":sha(DATA/"improvement-v2/record-splits.json"),
        "current_head_checkpoint_sha256":sha(DATA/"improvement-v2/combined-64.pt"),
        "records":records,"supplemental_records":len(selected),"existing_primary_records":len(existing),
        "total_frames":sum(r["expected_frames"] for r in records),
        "supplemental_frames":sum(r["expected_frames"] for r in selected),
        "label_record_counts":dict(Counter(str(label) for r in records for label in r["original_labels"])),
        "current_head_split_counts":dict(Counter(r["current_head_split"] for r in records)),
        "manual_collection":False,"new_manual_annotations":False,"model_training":False,
        "future_rule":"Any recording used as a future test must be excluded from head fitting and threshold/calibration selection before refitting. Keep all views of a recording together.",
        "scope":"Single-view clip labels only; no composite RGB, onset, depth or subject-disjoint evaluation claim."}
    ROOT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    write(ROOT/"frozen-record-plan.json",plan)
    write(REPORT/"record-plan-fingerprint.json",{"sha256":sha(ROOT/"frozen-record-plan.json"),"records":len(records)})
    print(json.dumps({"records":len(records),"frames":plan["total_frames"],"label_counts":plan["label_record_counts"],
                      "head_split_counts":plan["current_head_split_counts"],"archive_bytes":SIZE},indent=2),flush=True)


if __name__ == "__main__": main()
