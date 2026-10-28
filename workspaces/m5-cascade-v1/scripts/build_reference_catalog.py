"""Build a local reference index only from existing, publicly labelled CPR-Coach records."""

from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    target = ROOT / "data/reference-v1/catalog.json"
    if target.exists():
        raise FileExistsError("Reference catalogues are immutable; use a new version")
    archive = ROOT / "data/public_cpr/assets/ann.tar.gz"
    with tarfile.open(archive) as tar:
        lines = tar.extractfile("./ann/trainlist.txt").read().decode("utf-8").splitlines()
    line_map = {line.split()[0]: line for line in lines}
    entries = []
    for label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]:
        for take in ["r00", "r01"]:
            folder = ROOT / f"data/public_cpr/rgb-acceptance-v1/records/SupC{label:02d}--{take}--ch0"
            path = folder / "manifest.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            video = folder / "video.mp4"
            assert value["original_labels"] == [label] and value["view"] == "ch0"
            annotation = line_map[value["annotation_frame_dir"]]
            parts = annotation.split()
            assert int(parts[1]) == value["expected_frames"] and [int(x) for x in parts[2:]] == [label]
            assert sha(video) == value["video_sha256"] and value["whole_archive_hash_verified"]
            entries.append(
                {
                    "id": f"cpr-c{label:02d}-{take}-ch0",
                    "record_id": value["record_id"],
                    "label_id": label,
                    "dataset_label": value["label_names"][0],
                    "view": "ch0",
                    "media_rel": video.relative_to(ROOT).as_posix(),
                    "media_sha256": value["video_sha256"],
                    "manifest_rel": path.relative_to(ROOT).as_posix(),
                    "manifest_sha256": sha(path),
                    "annotation_member": "./ann/trainlist.txt",
                    "annotation_line": annotation,
                    "frames": value["video_validation"]["decoded_frames"],
                    "fps": value["video_validation"]["fps"],
                }
            )
    result = {
        "schema": "cpr-reference-catalog-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CPR-Coach",
        "dataset_url": "https://huggingface.co/datasets/ShunliWang/CPR-Coach",
        "repository_url": "https://github.com/Shunli-Wang/CPR-Coach",
        "dataset_revision": "b8667b8fd6a70ac63cc09fc6b763a9459b7c160a",
        "annotation_rel": "data/public_cpr/assets/ann.tar.gz",
        "annotation_sha256": sha(archive),
        "license_note": "The repository contains an MIT software license. A separate dataset-specific license is not declared in the pinned dataset card; do not infer a media redistribution grant from the software license. This catalogue references existing local research data; reference media is not bundled for redistribution.",
        "reference_notice": "Dataset labels describe research samples, not professional certification or proof of correct CPR. References cover whole clips and do not diagnose your practice.",
        "entries": entries,
    }
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": True,
                "entries": len(entries),
                "normal_label_entries": 2,
                "error_categories": 11,
                "new_labels": False,
                "new_media_downloads": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
