"""Pinned optional M1 runtime; shares the existing four-stage protocol and GPU lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from branch_worker import sha
from filelock import FileLock, Timeout
from vision_worker import probe


def main():
    parser = argparse.ArgumentParser()
    for key in ["video", "output", "manifest", "pipeline-manifest", "lock"]:
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument(
        "--tool", required=True, choices=["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"]
    )
    parser.add_argument("--inputs", default="{}")
    args = parser.parse_args()
    try:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(args.pipeline_manifest.read_text(encoding="utf-8"))
        if (
            manifest["schema"] != "cpr-branch-pipeline-manifest-v1"
            or sha(args.manifest) != manifest["model_bundle_sha256"]
        ):
            raise ValueError("PIPELINE_MODEL_MISMATCH")
        for name, expected in manifest["sources"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or sha(path) != expected:
                raise ValueError("PIPELINE_SOURCE_CHANGED")
        if args.tool == "pose.extract":
            probe(args.video)
        with FileLock(args.lock, timeout=0):
            sys.path.insert(0, str(root / "experiments/m1_fast_v1"))
            from m1_fast_pipeline import run_stage

            result = run_stage(
                args.tool,
                args.video,
                args.output,
                args.manifest,
                sha(args.pipeline_manifest),
                json.loads(args.inputs),
            )
            print(
                json.dumps(
                    {"status": result["status"], "tool": args.tool, "seconds": result["analysis_seconds"]}
                ),
                flush=True,
            )
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_name("worker-error.json").write_text(
            json.dumps(
                {
                    "code": "VISION_GPU_BUSY" if isinstance(exc, Timeout) else "M1_STAGE_FAILED",
                    "type": type(exc).__name__,
                }
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
