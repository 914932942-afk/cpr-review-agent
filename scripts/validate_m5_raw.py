"""Real pixel inference on the existing26 public videos; labels are not inference inputs."""

import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test-results/m5-cascade-v1/raw26"
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "experiments/branch_pipeline_v1")]
from pipeline import rgb_readout  # noqa: E402


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def load(p):
    with np.load(p, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "acceptance.json").exists()
    assert read(ROOT / "test-results/m5-cascade-v1/benchmark/acceptance.json")["passed"]
    spec = importlib.util.spec_from_file_location(
        "m5_raw_inference", ROOT / "experiments/m5_cascade_v1/inference.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    old = load(ROOT / "data/public_cpr/rgb-evaluation-v12/raw-predictions.npz")
    oldrgb = {str(k): v for k, v in zip(old["record_ids"], old["rgb_features"])}
    shapes = load(ROOT / "data/public_cpr/hand-shape-v1/raw.npz")
    oldshape = {
        str(k): (v, bool(b)) for k, v, b in zip(shapes["record_ids"], shapes["features"], shapes["available"])
    }
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rows = []
    with FileLock(ROOT / "data/vision-gpu.lock", timeout=0):
        for job in read(ROOT / "data/public_cpr/rgb-video-acceptance-v4/jobs.json"):
            dest = OUT / (job["key"] + ".json")
            if dest.exists():
                row = read(dest)
                new = load(dest.with_suffix(".npz"))
            else:
                video = ROOT / job["video"]
                assert sha(video) == job["video_sha256"]
                pose = load(
                    ROOT / "data/public_cpr/rgb-video-acceptance-v4/runs" / job["key"] / "inference.npz"
                )
                started = time.perf_counter()
                new, audit = m.infer(video, pose["keypoint"][0], pose["keypoint_score"][0])
                row = {
                    "record_id": job["record_id"],
                    "video_sha256": sha(video),
                    "seconds": time.perf_counter() - started,
                    "audit": audit,
                    "labels_used": False,
                }
                with dest.with_suffix(".npz").open("xb") as f:
                    np.savez_compressed(f, **new)
                dest.write_text(json.dumps(row, indent=2), encoding="utf-8")
            key = job["record_id"]
            shape, available = oldshape[key]
            baseline = rgb_readout(oldrgb[key][None], shape[None], np.array([available]))
            for k in [
                "label_ids",
                "scores",
                "thresholds",
                "eligible",
                "hand_override",
                "normal_veto_threshold",
            ]:
                assert np.allclose(new[k], baseline[k], atol=1e-6, rtol=0), (
                    key,
                    k,
                    float(np.max(abs(new[k].astype(float) - baseline[k].astype(float)))),
                )
            assert np.array_equal(new["eligible"], baseline["eligible"])
            assert np.allclose(new["context_features"][0], oldrgb[key][1536:], atol=1e-6, rtol=0), key
            assert row["audit"]["m5_cascade"]["route"] == "dependency-pruned" and "rgb_features" not in new
            rows.append(
                {
                    "record_id": key,
                    "max_score_difference": float(abs(new["scores"] - baseline["scores"]).max()),
                    "seconds": row["seconds"],
                }
            )
            print(
                "Actual pixel regression", len(rows), "/26", job["key"], round(row["seconds"], 2), flush=True
            )
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "passed": True,
        "actual_videos": len(rows),
        "exact_decisions": True,
        "max_score_difference": max(r["max_score_difference"] for r in rows),
        "all_contexts_equal_within_1e6": True,
        "all_routes_pruned": True,
        "all_observed_frames_and_minimum_support_preserved": True,
        "uncomputed_dense_features_not_persisted": True,
        "labels_used_at_inference": False,
        "public_dataset_only": True,
        "rows": rows,
        "scope": "Existing repeatedly inspected engineering26; not new independent blind data. Same16 frames and frozen perception/head math.",
    }
    (OUT / "acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
