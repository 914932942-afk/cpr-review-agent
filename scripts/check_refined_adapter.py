"""Real cached-readout adapter compatibility, negative contracts and CPU-only fusion timing."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "backend"),
    str(ROOT / "experiments/module_slots_v1"),
    str(ROOT / "experiments/m2_competence_v3"),
]
from cpr_agent.modules.baselines import prediction_from_arrays  # noqa: E402
from cpr_agent.modules.contracts import Binding, FusionInput, QualityIndicators, Readout  # noqa: E402
from evaluate_m2_quality import sliced  # noqa: E402
from slots_pipeline import LABEL_IDS, validate_result  # noqa: E402


def main():
    output = ROOT / "test-results/m1-m2-refined-v1/adapter.json"
    if output.exists():
        raise FileExistsError("Preserve accepted adapter evidence")
    spec = importlib.util.spec_from_file_location(
        "checked_m2_member", ROOT / "experiments/m2_competence_v3/module.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    member = module.QualityAwareFusion()
    with np.load(ROOT / "data/m2-quality-v1/experiment-01/raw-readouts.npz", allow_pickle=False) as data:
        source = {k: data[k].copy() for k in data.files}
    n = len(source["record_ids"])
    records = []
    for i in range(n):
        skeleton = sliced({k[3:]: v for k, v in source.items() if k.startswith("sk_")}, i, n)
        rgb = sliced({k[4:]: v for k, v in source.items() if k.startswith("rgb_")}, i, n)
        binding = Binding(
            clip_id="cache-contract-" + str(i),
            video_sha256="a" * 64,
            bundle_sha256="b" * 64,
            pipeline_sha256="c" * 64,
            actor_source_sha256="d" * 64,
        )
        # Explicit cache fixtures: binding UUIDs are synthetic; scores are from real feature extraction.
        quality = QualityIndicators(
            rgb_usable_samples=int(source["usable"][i]),
            rgb_requested_samples=int(source["requested"][i]),
            hand_shape_available=bool(source["hand_available"][i]),
        )
        # Construct the two readouts before the actual strict FusionInput binding validation.
        from types import SimpleNamespace

        stub = SimpleNamespace(binding=binding, execution_id="skeleton")
        sk = Readout(prediction_from_arrays(stub, skeleton, LABEL_IDS), skeleton)
        stub.execution_id = "rgb"
        rg = Readout(prediction_from_arrays(stub, rgb, rgb["label_ids"].tolist()), rgb)
        request = FusionInput(binding, "adapter-" + str(i), sk, rg, quality)
        result = member.fuse(request)
        validate_result(result, binding, request.execution_id, LABEL_IDS, "predictions")
        records.append(
            {
                "record_id": str(source["record_ids"][i]),
                "adapted_labels": result.audit["m2_fusion"]["adapted_label_ids"],
            }
        )
    negative = 0
    for kind in ["score_disagreement", "unsupported_score_type", "cross_video"]:
        try:
            if kind == "score_disagreement":
                bad = sk.prediction.model_copy(update={"scores": {**sk.prediction.scores, 1: 0.123456789}})
                member.fuse(replace(request, skeleton=Readout(bad, sk.arrays)))
            elif kind == "unsupported_score_type":
                bad = rg.prediction.model_copy(update={"score_type": "calibrated_probability"})
                member.fuse(replace(request, rgb=Readout(bad, rg.arrays)))
            else:
                bad = sk.prediction.model_copy(
                    update={"binding": binding.model_copy(update={"clip_id": "different-video"})}
                )
                replace(request, skeleton=Readout(bad, sk.arrays))
        except ValueError:
            negative += 1
        else:
            raise AssertionError("Invalid contract accepted: " + kind)
    samples = []
    for _ in range(100):
        started = time.perf_counter()
        member.fuse(request)
        samples.append((time.perf_counter() - started) * 1000)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "passed": True,
        "real_cached_readouts": n,
        "synthetic_binding_fixture": True,
        "negative_contract_checks": negative,
        "fusion_median_ms": float(np.median(samples)),
        "fusion_p95_ms": float(np.percentile(samples, 95)),
        "no_gpu_models_loaded": True,
        "records": records,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
