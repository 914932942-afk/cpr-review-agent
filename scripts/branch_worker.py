"""One isolated, resumable stage of the v15 decomposition; no API or reference labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from filelock import FileLock, Timeout
from vision_worker import probe


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
        pipeline = json.loads(args.pipeline_manifest.read_text(encoding="utf-8"))
        if (
            pipeline["schema"] != "cpr-branch-pipeline-manifest-v1"
            or sha(args.manifest) != pipeline["model_bundle_sha256"]
        ):
            raise ValueError("PIPELINE_MODEL_MISMATCH")
        for name, expected in pipeline["sources"].items():
            source = (root / name).resolve()
            if not source.is_relative_to(root) or sha(source) != expected:
                raise ValueError("PIPELINE_SOURCE_CHANGED")
        if args.tool == "pose.extract":
            probe(args.video)
        with FileLock(args.lock, timeout=0):
            sys.path.insert(0, str(root / "experiments/branch_pipeline_v1"))
            from pipeline import run_stage

            value = run_stage(
                args.tool,
                args.video,
                args.output,
                args.manifest,
                sha(args.pipeline_manifest),
                json.loads(args.inputs),
            )
            print(
                json.dumps(
                    {"status": value["status"], "tool": args.tool, "seconds": value["analysis_seconds"]}
                ),
                flush=True,
            )
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_name("worker-error.json").write_text(
            json.dumps(
                {
                    "code": "VISION_GPU_BUSY" if isinstance(exc, Timeout) else "BRANCH_STAGE_FAILED",
                    "type": type(exc).__name__,
                }
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
