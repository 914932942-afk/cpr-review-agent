"""Create labelled-as-test-only invalid inputs; no new training samples or annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main():
    import cv2

    root = Path(__file__).resolve().parents[1]
    folder = root / "data/demo-cases-v1/controls"
    if folder.exists():
        raise FileExistsError("Preserve boundary controls; do not overwrite")
    source = root / "data/public_cpr/rgb-acceptance-v1/records/SupC00--r00--ch0/video.mp4"
    folder.mkdir(parents=True)
    target = folder / "case-05-too-short-3frames.mp4"
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        assert capture.isOpened()
        fps = capture.get(cv2.CAP_PROP_FPS)
        for index in range(3):
            ok, frame = capture.read()
            assert ok, "The source must contain three valid frames"
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                assert writer.isOpened()
            writer.write(frame)
    finally:
        capture.release()
        if writer:
            writer.release()
    check = cv2.VideoCapture(str(target))
    count = 0
    try:
        while check.read()[0]:
            count += 1
    finally:
        check.release()
    assert count == 3
    invalid = folder / "case-04-invalid-container.mp4"
    invalid.write_bytes(b"Intentional invalid-container UI test. This is not a video or a dataset sample.\n")

    def sha(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    value = {
        "kind": "boundary_test_controls",
        "not_dataset_samples": True,
        "for_training_or_accuracy_evaluation": False,
        "manual_annotations": False,
        "source_rel": source.relative_to(root).as_posix(),
        "source_sha256": sha(source),
        "short_video": {
            "path": target.relative_to(root).as_posix(),
            "sha256": sha(target),
            "decoded_frames": count,
            "fps": fps,
            "original_frame_indices": [0, 1, 2],
            "expected_boundary": "VIDEO_TOO_SHORT",
            "cpr_label": None,
        },
        "invalid_container": {
            "path": invalid.relative_to(root).as_posix(),
            "sha256": sha(invalid),
            "expected_http_status": 415,
            "expected_error_code": "INVALID_CONTAINER",
        },
    }
    (folder / "manifest.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "short_video_frames": count, "new_training_data": False}))


if __name__ == "__main__":
    main()
