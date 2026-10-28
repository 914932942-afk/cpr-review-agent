"""Source-pooled regularised calibration and selective review, independent of CV decisions.

Pure Python inference keeps numpy/scipy/torch out of the web process. Labels never
enter inference. Probabilities describe public dataset categories, not CPR competence.
"""

import math

LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13)
RGB_LABELS = (1, 2, 5, 7, 9)
BODY_LABELS = tuple(k for k in LABELS if k not in RGB_LABELS)
VERSION = "source-quality-selective-v1"


def logit(x):
    if not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError("M3_INVALID_SCORE")
    x = min(max(x, 1e-7), 1 - 1e-7)
    return math.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + math.exp(-max(-50, min(50, x))))


def pool(label):
    if label not in LABELS:
        raise ValueError("M3_UNSUPPORTED_LABEL")
    return "rgb" if label in RGB_LABELS else "body"


def probability(score, threshold, label, quality, disagreement, parameters):
    if not math.isfinite(quality) or not 0 <= quality <= 1 or disagreement not in (0, 1):
        raise ValueError("M3_INVALID_QUALITY")
    cfg = parameters[pool(label)]
    margin = logit(score) - logit(threshold)
    denominator = 1 + cfg["quality_penalty"] * (1 - quality) + cfg["disagreement_penalty"] * disagreement
    return sigmoid((cfg["slope"] * margin + cfg["bias"]) / denominator)


def review_state(flagged, calibrated, quality, parameters):
    confidence = calibrated if flagged else 1 - calibrated
    threshold = parameters["positive_threshold" if flagged else "negative_threshold"]
    if quality < parameters["minimum_quality"] or confidence < threshold:
        return "uncertain"
    return "model_flagged" if flagged else "not_flagged"
