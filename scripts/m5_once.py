"""One real M5-only process for fair timing, memory and readout comparison."""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "experiments/branch_pipeline_v1")]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "candidate"], required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--pose", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stress", action="store_true")
    args = p.parse_args()
    assert not args.output.exists()
    import numpy as np
    import torch
    from cpr_agent.modules.baselines import BaselineRGB
    from cpr_agent.modules.contracts import Binding

    spec = importlib.util.spec_from_file_location(
        "m5_inference", ROOT / "experiments/m5_cascade_v1/inference.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(args.pose, allow_pickle=False) as a:
        points, confidence = a["keypoint"][0].copy(), a["keypoint_score"][0].copy()
    if args.stress:
        confidence[:, [7, 8, 9, 10]] = np.minimum(confidence[:, [7, 8, 9, 10]], 0.19)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if args.mode == "baseline":
        binding = Binding(
            clip_id="benchmark",
            video_sha256=sha(args.video),
            bundle_sha256="b" * 64,
            pipeline_sha256="c" * 64,
            actor_source_sha256=sha(args.pose),
        )
        request = SimpleNamespace(
            points=points,
            confidence=confidence,
            video_path=args.video,
            sampling_profile="uniform16-v15",
            binding=binding,
            execution_id="benchmark",
        )
        result = BaselineRGB().predict(request)
        arrays, audit = result.arrays, result.audit
    else:
        arrays, audit = m.infer(args.video, points, confidence)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    out = {
        "mode": args.mode,
        "video_sha256": sha(args.video),
        "pose_sha256": sha(args.pose),
        "controlled_low_wrist_confidence": args.stress,
        "inference_seconds": elapsed,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "audit": audit,
        "labels_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "inference_seconds": elapsed,
                "cuda_peak_allocated_bytes": out["cuda_peak_allocated_bytes"],
                "route": audit.get("m5_cascade", {}).get("route", "full"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
