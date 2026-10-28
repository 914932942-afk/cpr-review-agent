"""Approve module construction only after frozen numerical and resource gates pass."""

import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/m5-cascade-v1"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (DATA / "decision.json").exists()
    for p in [
        DATA / "offline.json",
        ROOT / "test-results/m5-cascade-v1/benchmark/acceptance.json",
        ROOT / "test-results/m5-cascade-v1/raw26/acceptance.json",
        ROOT / "test-results/m5-cascade-v1/ablation.json",
    ]:
        assert read(p)["passed"]
    protocol = read(DATA / "protocol.json")
    assert sha(ROOT / "experiments/m5_cascade_v1/algorithm.py") == protocol["algorithm_sha256"]
    assert sha(ROOT / "experiments/m5_cascade_v1/runtime.py") == protocol["runtime_sha256"]
    assert all(
        sha(ROOT / p) == h for p, h in {**protocol["models"], **protocol["additional_sources"]}.items()
    )
    spec = importlib.util.spec_from_file_location(
        "m5_certificate", ROOT / "experiments/m5_cascade_v1/algorithm.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rgb = torch.load(
        ROOT / "data/public_cpr/rgb-evaluation-v12/experts.pt", weights_only=True, map_location="cpu"
    )
    hand = torch.load(
        ROOT / "data/public_cpr/rgb-evaluation-v14/hands.pt", weights_only=True, map_location="cpu"
    )
    cert = {
        "available": m.compile_plan(rgb, hand, True),
        "unavailable": m.compile_plan(rgb, hand, False),
        "model_sha256": protocol["models"],
        "proof": "Every active feature index is computed before that head is run. Same quality gates, coefficients, thresholds, sample indices and minimum support.",
    }
    (DATA / "certificate.json").write_text(json.dumps(cert, indent=2), encoding="utf-8")
    decision = {
        "accepted": True,
        "protocol_sha256": sha(DATA / "protocol.json"),
        "certificate_sha256": sha(DATA / "certificate.json"),
        "live_integration_acceptance_required": True,
        "weights_unchanged": True,
        "new_training": False,
    }
    (DATA / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision))


if __name__ == "__main__":
    main()
