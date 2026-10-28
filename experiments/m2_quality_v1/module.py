"""M2 adapter. Persistence, execution scheduling and M3 calibration remain outside this module."""

import hashlib
import json
import sys
from pathlib import Path

from cpr_agent.modules.baselines import prediction_from_arrays
from cpr_agent.modules.contracts import LABEL_IDS, RGB_LABEL_IDS, MLResult

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_algorithm import Config, fuse  # noqa: E402


class QualityAwareFusion:
    def __init__(self):
        folder = ROOT / "data/m2-quality-v1/experiment-01"
        protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
        selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        decision = json.loads((folder / "decision.json").read_text(encoding="utf-8"))

        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        if (
            not decision.get("accepted")
            or decision["selection_sha256"] != digest(folder / "selection.json")
            or selection["protocol_sha256"] != digest(folder / "protocol.json")
            or protocol["algorithm_sha256"] != digest(Path(__file__).with_name("fusion_algorithm.py"))
        ):
            raise ValueError("M2_SELECTION_NOT_ACCEPTED")
        self.config = Config(**selection["selected"])

    def fuse(self, request):
        import numpy as np

        for readout, labels in [(request.skeleton, LABEL_IDS), (request.rgb, RGB_LABEL_IDS)]:
            p, arrays = readout.prediction, readout.arrays
            if p.supported_label_ids != list(labels) or p.score_type != "uncalibrated_score":
                raise ValueError("M2_SOURCE_SCOPE_MISMATCH")
            if (
                arrays["scores"].ndim != 2
                or arrays["scores"].shape[0] != 1
                or arrays["thresholds"].shape != (1, len(labels))
                or arrays["eligible"].shape != (1, len(labels))
                or arrays["eligible"].dtype != bool
            ):
                raise ValueError("M2_SOURCE_LAYOUT_INVALID")
            for i, label in enumerate(labels):
                if (
                    float(arrays["scores"][0, i]) != p.scores[label]
                    or float(arrays["thresholds"][0, i]) != p.thresholds[label]
                    or bool(arrays["eligible"][0, i]) != (label in p.flagged_label_ids)
                ):
                    raise ValueError("M2_SOURCE_ARRAY_DISAGREEMENT")
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("M2_NONFINITE_INPUT")
        quality = request.quality
        arrays, audit = fuse(
            request.skeleton.arrays,
            request.rgb.arrays,
            quality.rgb_usable_samples,
            quality.rgb_requested_samples,
            quality.hand_shape_available,
            self.config,
        )
        return MLResult(
            prediction_from_arrays(request, arrays, LABEL_IDS, "predictions"),
            arrays,
            {"m2_fusion": audit},
        )
