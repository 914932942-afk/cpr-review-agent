"""One real module-stage subprocess with source pins, shared GPU lock and explicit request binding."""

import argparse
import json
import sys
from pathlib import Path

from branch_worker import sha
from filelock import FileLock
from vision_worker import probe


def main():
    parser = argparse.ArgumentParser()
    for key in ["video", "output", "manifest", "pipeline-manifest", "lock"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument(
        "--tool", required=True, choices=["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"]
    )
    parser.add_argument("--inputs", default="{}")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--execution-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.pipeline_manifest.read_text(encoding="utf-8"))
    if config.get("schema") != "cpr-branch-pipeline-manifest-v1" or config["model_bundle_sha256"] != sha(
        args.manifest
    ):
        raise ValueError("PIPELINE_MODEL_MISMATCH")
    for name, expected in config["sources"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError("PIPELINE_SOURCE_CHANGED")
    sys.path[:0] = [str(root / "backend"), str(root / "experiments/module_slots_v1")]
    from cpr_agent.modules.registry import build_registry

    build_registry().validate_selection(config["module_slots"])
    if args.tool == "pose.extract":
        probe(args.video)
    with FileLock(args.lock, timeout=0):
        from slots_pipeline import run_stage

        result = run_stage(
            args.tool,
            args.video,
            args.output,
            args.manifest,
            args.pipeline_manifest,
            json.loads(args.inputs),
            args.clip_id,
            args.execution_id,
        )
        print(
            json.dumps(
                {"tool": args.tool, "seconds": result["analysis_seconds"], "status": result["status"]}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
