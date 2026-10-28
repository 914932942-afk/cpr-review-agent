"""ML-environment subprocess. Owns the GPU lock and executes the unmodified frozen CLI in-process."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import cv2
from filelock import FileLock, Timeout


class InputRejected(ValueError):
    pass


def probe(video):
    cap = cv2.VideoCapture(str(video))
    frames, dimensions = 0, None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not 0 < fps <= 240:
            raise InputRejected("VIDEO_NOT_DECODABLE")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            shape = frame.shape[:2]
            if shape[0] * shape[1] > 3840 * 2160 or max(shape) > 3840 or min(shape) < 128:
                raise InputRejected("VIDEO_DIMENSIONS_UNSUPPORTED")
            if dimensions and shape != dimensions:
                raise InputRejected("VIDEO_DIMENSIONS_CHANGED")
            dimensions = shape
            frames += 1
            if frames > 900 or frames / fps > 45:
                raise InputRejected("VIDEO_TOO_LONG")
        if frames < 8:
            raise InputRejected("VIDEO_TOO_SHORT")
        advertised = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if advertised > 0 and abs(frames - advertised) > 1:
            raise InputRejected("VIDEO_DECODE_INCOMPLETE")
        return {
            "decoded_frames": frames,
            "fps": fps,
            "dimensions_hw": dimensions,
            "content_validation": "CPR content and camera applicability are not automatically guaranteed",
        }
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser()
    for arg in ("video", "output", "manifest", "lock"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    try:
        meta = probe(args.video)
        args.output.with_name("input-probe.json").write_text(json.dumps(meta), encoding="utf-8")
        # The lock lives in this process, so an orphan after a hard server stop still owns it.
        with FileLock(args.lock, timeout=0):
            root = Path(__file__).resolve().parents[1]
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            entry = (root / manifest["entrypoint"]).resolve()
            if not entry.is_relative_to(root / "experiments"):
                raise ValueError("Entrypoint outside experiments")
            sys.path.insert(0, str(entry.parent))
            sys.argv = [
                str(entry),
                "--video",
                str(args.video),
                "--output",
                str(args.output),
                "--manifest",
                str(args.manifest),
            ]
            runpy.run_path(str(entry), run_name="__main__")
    except Exception as exc:
        code = (
            str(exc)
            if isinstance(exc, InputRejected)
            else ("VISION_GPU_BUSY" if isinstance(exc, Timeout) else "VISION_PROCESS_FAILED")
        )
        args.output.with_name("worker-error.json").write_text(
            json.dumps({"code": code, "type": type(exc).__name__}), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
