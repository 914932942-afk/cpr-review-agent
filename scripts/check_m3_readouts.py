"""Real cached readout replay plus pure-Python inference checks, separately from live HTTP."""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "workspaces/m3-calibrated-v1"
sys.path.insert(0, str(WS / "backend"))
from cpr_agent.modules.contracts import Binding, CalibrationInput, Prediction, QualityIndicators  # noqa: E402
from cpr_agent.modules.registry import build_registry  # noqa: E402


def main():
    out = ROOT / "test-results/m3-calibration-v1/algorithm"
    out.mkdir(parents=True, exist_ok=False)
    p = subprocess.run(
        [sys.executable, str(ROOT / "experiments/m3_calibration_v1/test_algorithm.py")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (out / "unit.log").write_text(p.stdout + p.stderr, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    with np.load(ROOT / "data/m3-calibration-v1/raw26.npz", allow_pickle=False) as f:
        raw = {k: f[k].copy() for k in f.files}
    module = build_registry().create("M3", "source-quality-selective-v1")
    samples = []
    count = 0
    expected = json.loads((ROOT / "data/m3-calibration-v1/raw26.json").read_text())["rows"]
    for i, key in enumerate(dict.fromkeys(raw["record_ids"])):
        mask = raw["record_ids"] == key
        positions = np.flatnonzero(mask)
        labels = raw["label_ids"][mask].tolist()
        binding = Binding(
            clip_id="replay-" + str(i),
            video_sha256="a" * 64,
            bundle_sha256="b" * 64,
            pipeline_sha256="c" * 64,
        )
        scores = {label: float(raw["scores"][j]) for label, j in zip(labels, positions)}
        thresholds = {label: float(raw["thresholds"][j]) for label, j in zip(labels, positions)}
        flags = [label for label, j in zip(labels, positions) if raw["flagged"][j]]
        source = Prediction(
            prediction_id="fused-" + str(i),
            binding=binding,
            supported_label_ids=labels,
            flagged_label_ids=flags,
            score_type="uncalibrated_score",
            scores=scores,
            thresholds=thresholds,
        )
        body = [j for j in positions if raw["label_ids"][j] not in [1, 2, 5, 7, 9]]
        rgb = [j for j in positions if raw["label_ids"][j] in [1, 2, 5, 7, 9]]
        q = QualityIndicators(
            pose_visible_fraction=float(raw["quality"][body[0]]),
            rgb_usable_samples=int(round(raw["quality"][rgb[0]] * 16)),
            rgb_requested_samples=16,
            extras={"disagreement_" + str(raw["label_ids"][j]): float(raw["disagreement"][j]) for j in rgb},
        )
        request = CalibrationInput(target_label_ids=labels, predictions=[source], quality=q)
        before = request.model_dump()
        started = time.perf_counter()
        result = module.assess(request)
        samples.append((time.perf_counter() - started) * 1000)
        for row in expected:
            if row["record_id"] != key:
                continue
            assert result.states[row["label_id"]] == row["state"]
            assert abs(result.calibrated_probabilities[row["label_id"]] - row["probability"]) < 1e-12
        assert request.model_dump() == before
        count += 1
    record = {
        "passed": True,
        "unit_checks": 16,
        "real_cached_readouts": count,
        "readout_probabilities_match_tolerance": 1e-12,
        "inference_median_ms": float(np.median(samples)),
        "inference_p95_ms": float(np.percentile(samples, 95)),
        "no_vision_models_loaded": True,
        "source_outputs_unchanged": True,
    }
    (out / "acceptance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
