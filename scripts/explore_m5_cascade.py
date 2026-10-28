"""Freeze protocol, certify feature dependencies and compare all public readout caches."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/m5-cascade-v1"
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "experiments/branch_pipeline_v1")]
from pipeline import frozen, rgb_readout  # noqa: E402


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def load(p):
    with np.load(p, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def main():
    DATA.mkdir(exist_ok=True)
    assert not (DATA / "offline.json").exists()
    spec = importlib.util.spec_from_file_location(
        "m5_algorithm", ROOT / "experiments/m5_cascade_v1/algorithm.py"
    )
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    rgb = torch.load(
        ROOT / "data/public_cpr/rgb-evaluation-v12/experts.pt", weights_only=True, map_location="cpu"
    )
    hand = torch.load(
        ROOT / "data/public_cpr/rgb-evaluation-v14/hands.pt", weights_only=True, map_location="cpu"
    )
    parent = ROOT / "workspaces/m4-refined-v2/data/m4-exploration-v2/pipeline-manifest.json"
    before = {
        "env_sha256": sha(ROOT / ".env"),
        "parent_profile_sha256": sha(parent),
        "parent_sources": read(parent)["sources"],
        "parent_release_manifest_sha256": sha(ROOT / "releases/m4-refined-v2/manifest.json"),
    }
    (DATA / "before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    protocol = {
        "algorithm": a.VERSION,
        "algorithm_sha256": sha(ROOT / "experiments/m5_cascade_v1/algorithm.py"),
        "runtime_sha256": sha(ROOT / "experiments/m5_cascade_v1/runtime.py"),
        "models": {
            p: sha(ROOT / p)
            for p in [
                "data/public_cpr/rgb-evaluation-v12/experts.pt",
                "data/public_cpr/rgb-evaluation-v14/hands.pt",
            ]
        },
        "correctness_gate": "Every active input dependency materialised. All40development +26 previously exposed raw cached readouts exactly equal; forced quality-unavailable readouts equal full fallback. No manual/new labels; no threshold/model changes.",
        "resource_gate": "Predeclared3 public clips x3 process-isolated paired repeats; median process wall and median CUDA peak each improve>=10%; no per-clip median wall regression>5%. Same16 input frames and same minimum8 support. One controlled low-wrist-confidence fallback paired run.",
        "bench_cases": ["SupC00--r00--ch0", "SupC01--r00--ch0", "SupC02--r00--ch0"],
        "runtime_gate": "All26 actual videos reproduce old context/readout within1e-6 and exactly the same decisions; full live system M1-M4, fallback certificate, UI, recovery checks mandatory.",
        "scope": "Exact conditional inference under accepted model feature layouts. Not new classifier accuracy, architecture training, or independent blind generalisation. Feature pruning is certified; adversarial images are not certified.",
    }
    (DATA / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results = []
    for name, feature_file, shape_file in [
        (
            "development40",
            "data/public_cpr/rgb-domain-features-v5/features.npz",
            "data/public_cpr/hand-shape-v1/development.npz",
        ),
        (
            "raw26",
            "data/public_cpr/rgb-evaluation-v12/raw-predictions.npz",
            "data/public_cpr/hand-shape-v1/raw.npz",
        ),
    ]:
        d = load(ROOT / feature_file)
        sh = load(ROOT / shape_file)
        features = d["features"] if "features" in d else d["rgb_features"]
        lookup = {str(k): (v, bool(b)) for k, v, b in zip(sh["record_ids"], sh["features"], sh["available"])}
        pruned = 0
        checks = 0
        for key, x in zip(d["record_ids"], features):
            shape, available = lookup[str(key)]
            for quality in [available, False]:
                base = rgb_readout(x[None], shape[None], np.array([quality]))
                plan = a.compile_plan(rgb, hand, quality)
                candidate = (
                    a.context_readout(
                        x[None, 1536:],
                        shape[None],
                        rgb,
                        hand,
                        plan,
                        frozen.predict_experts,
                        frozen.predict_hands,
                    )
                    if not plan["dense_visual_required"]
                    else rgb_readout(x[None], shape[None], np.array([quality]))
                )
                assert all(np.array_equal(base[k], candidate[k]) for k in base), str(key)
                checks += 1
            pruned += not a.compile_plan(rgb, hand, available)["dense_visual_required"]
        results.append(
            {
                "pool": name,
                "records": len(features),
                "prunable_observed_records": int(pruned),
                "readout_checks": checks,
                "exact_equal": True,
            }
        )
    # Counterfactual unused-feature corruption cannot alter any active head.
    rng = np.random.default_rng(20261027)
    for _ in range(200):
        x = rng.normal(size=(1, 1893)).astype(np.float32)
        shape = rng.normal(size=(1, 576)).astype(np.float32)
        base = rgb_readout(x, shape, np.array([True]))
        plan = a.compile_plan(rgb, hand, True)
        fast = a.context_readout(
            x[:, 1536:], shape, rgb, hand, plan, frozen.predict_experts, frozen.predict_hands
        )
        assert all(np.array_equal(base[k], fast[k]) for k in base)
    result = {
        "passed": True,
        "pools": results,
        "counterfactual_dependency_cases": 200,
        "models_or_thresholds_changed": False,
        "new_training": False,
        "new_manual_labels": False,
        "locked_test_opened": False,
        "live_acceptance_required": True,
    }
    (DATA / "offline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
