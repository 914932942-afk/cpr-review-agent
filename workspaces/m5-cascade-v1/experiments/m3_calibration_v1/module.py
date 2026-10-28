"""M3 adapter: no Store, no tools, no labels and no vision inference in this module."""

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

from cpr_agent.modules.contracts import Assessment

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "cpr_m3_calibration_algorithm", Path(__file__).with_name("algorithm.py")
)
algorithm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = algorithm
spec.loader.exec_module(algorithm)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceQualityCalibration:
    def __init__(self):
        folder = ROOT / "data/m3-calibration-v1"
        decision, selection, protocol = [
            read(folder / name) for name in ["decision.json", "selection.json", "protocol.json"]
        ]
        if (
            not decision["accepted"]
            or decision["selection_sha256"] != sha(folder / "selection.json")
            or selection["protocol_sha256"] != sha(folder / "protocol.json")
            or protocol["algorithm_sha256"] != sha(Path(__file__).with_name("algorithm.py"))
        ):
            raise ValueError("M3_PARAMETERS_NOT_ACCEPTED")
        upstream = read(folder / "upstream.json")
        for name, expected in upstream["sources"].items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != expected:
                raise ValueError("M3_UPSTREAM_PROFILE_CHANGED")
        self.parameters = selection["parameters"]
        for cfg in self.parameters.values():
            if any(
                not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v)
                for v in cfg.values()
            ):
                raise ValueError("M3_INVALID_PARAMETERS")
            if (
                not 0.01 <= cfg["slope"] <= 25
                or not -8 <= cfg["bias"] <= 8
                or not 0 <= cfg["quality_penalty"] <= 3
                or not 0 <= cfg["disagreement_penalty"] <= 3
                or not 0.5 <= cfg["positive_threshold"] <= 1
                or not 0.5 <= cfg["negative_threshold"] <= 1
                or not 0 <= cfg["minimum_quality"] <= 1
            ):
                raise ValueError("M3_INVALID_PARAMETERS")

    def assess(self, request):
        states = {label: "not_evaluated" for label in request.target_label_ids}
        probabilities = {}
        if len(request.predictions) > 1:
            raise ValueError("M3_EXPECTS_ONE_FUSED_SOURCE")
        if request.predictions:
            source = request.predictions[0]
            if source.score_type != "uncalibrated_score":
                raise ValueError("M3_SCORE_DOMAIN_MISMATCH")
            quality = request.quality
            for label in request.target_label_ids:
                if label not in source.supported_label_ids:
                    continue
                if label not in source.thresholds:
                    raise ValueError("M3_THRESHOLD_MISSING")
                if label in algorithm.RGB_LABELS:
                    used, total = quality.rgb_usable_samples, quality.rgb_requested_samples
                    q = used / total if used is not None and total is not None and total > 0 else None
                    disagreement = quality.extras.get("disagreement_" + str(label))
                else:
                    q = quality.pose_visible_fraction
                    disagreement = 0
                if (
                    q is None
                    or disagreement is None
                    or q < max(0.5, self.parameters[algorithm.pool(label)]["minimum_quality"])
                ):
                    states[label] = "uncertain"
                    continue
                p = algorithm.probability(
                    source.scores[label], source.thresholds[label], label, q, disagreement, self.parameters
                )
                probabilities[label] = p
                states[label] = algorithm.review_state(
                    label in source.flagged_label_ids, p, q, self.parameters[algorithm.pool(label)]
                )
        return Assessment(
            states=states,
            source_prediction_ids=[p.prediction_id for p in request.predictions],
            calibration_profile=algorithm.VERSION,
            calibration_status="calibrated" if probabilities else "uncalibrated",
            calibrated_probabilities=probabilities,
        )
