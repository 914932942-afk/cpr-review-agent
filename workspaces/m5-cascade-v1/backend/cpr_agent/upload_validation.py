"""Bounded CPU preflight. It does not load models, use the GPU, or assess CPR content."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading

from .config import ROOT
from .store import DomainError

INPUT_ERRORS = {
    "VIDEO_TOO_SHORT": "This video is too short. Choose a complete clip with at least 8 frames.",
    "VIDEO_TOO_LONG": "This video exceeds 900 frames or 45 seconds. Choose a shorter complete practice clip.",
    "VIDEO_NOT_DECODABLE": "This video could not be decoded, or its frame rate is unsupported. Export a playable MP4, MOV or WebM.",
    "VIDEO_DIMENSIONS_UNSUPPORTED": "The video dimensions are unsupported. Use at least 128 pixels per side and at most 3840 × 2160 pixels.",
    "VIDEO_DIMENSIONS_CHANGED": "The frame dimensions change during this video. Export it at a fixed resolution.",
    "VIDEO_DECODE_INCOMPLETE": "The full video could not be decoded. Export a complete copy and try again.",
}


class UploadInspector:
    def __init__(self, settings):
        self.settings = settings
        self.capacity = threading.BoundedSemaphore(1)

    def inspect(self, path):
        if not self.capacity.acquire(blocking=False):
            raise DomainError(
                429, "UPLOAD_CHECK_BUSY", "Another upload is being checked. Please try again shortly."
            )
        try:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("CPR_", "DEEPSEEK_"))}
            try:
                result = subprocess.run(
                    [
                        str(self.settings.ml_python),
                        str(ROOT / "scripts/probe_upload.py"),
                        "--video",
                        str(path),
                    ],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    timeout=35,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except subprocess.TimeoutExpired as exc:
                raise DomainError(
                    503, "UPLOAD_CHECK_TIMEOUT", "Video checking timed out. Try a shorter clip."
                ) from exc
            except OSError as exc:
                raise DomainError(
                    503, "UPLOAD_CHECK_UNAVAILABLE", "Video checking is unavailable. Please try again."
                ) from exc
            try:
                if result.returncode != 0 or len(result.stdout) > 8192:
                    raise ValueError("Invalid probe response")
                value = json.loads(result.stdout)
                policy = hashlib.sha256((ROOT / "scripts/vision_worker.py").read_bytes()).hexdigest()
                if value["policy_sha256"] != policy:
                    raise ValueError("Input policy changed")
                if value["status"] == "rejected" and value.get("code") in INPUT_ERRORS:
                    raise DomainError(422, value["code"], INPUT_ERRORS[value["code"]])
                meta = value["metadata"]
                frames, fps = meta["decoded_frames"], meta["fps"]
                height, width = meta["dimensions_hw"]
                if (
                    value["status"] != "passed"
                    or type(frames) is not int
                    or not 8 <= frames <= 900
                    or not isinstance(fps, (int, float))
                    or not math.isfinite(fps)
                    or not 0 < fps <= 240
                    or frames / fps > 45
                    or type(height) is not int
                    or type(width) is not int
                    or min(height, width) < 128
                    or max(height, width) > 3840
                    or height * width > 3840 * 2160
                ):
                    raise ValueError("Invalid checked metadata")
                return {
                    "status": "passed",
                    "policy_sha256": policy,
                    "decoded_frames": frames,
                    "fps": fps,
                    "dimensions_hw": [height, width],
                    "duration_seconds": frames / fps,
                    "scope": "Container and decoding only; CPR content and camera applicability are not guaranteed.",
                }
            except (ValueError, KeyError, TypeError, OSError) as exc:
                raise DomainError(
                    503,
                    "UPLOAD_CHECK_UNAVAILABLE",
                    "The video check did not return a valid result. Please try again.",
                ) from exc
        finally:
            self.capacity.release()
