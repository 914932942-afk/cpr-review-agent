"""Quality-gated disagreement fusion, without labels, source IDs or backbone updates."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    full_rgb_ratio: float = 0.875
    minimum_rgb_weight: float = 0.0

    def __post_init__(self):
        if not 0.5 < self.full_rgb_ratio <= 1 or not 0 <= self.minimum_rgb_weight <= 0.5:
            raise ValueError("M2_CONFIG_INVALID")


def logit(value):
    value = np.clip(value, 1e-9, 1 - 1e-9)
    return np.log(value / (1 - value))


def fuse(skeleton, rgb, usable, requested, hand_available, config=Config()):
    # Lazy import: the same frozen baseline and cardinality guard remain the comparator.
    from pipeline import LABELS, frozen, merge_readouts

    baseline, reliability = merge_readouts(skeleton, rgb)
    n = len(skeleton["scores"])
    if n != 1:
        raise ValueError("M2_EXPECTS_ONE_BOUND_CLIP")
    if not all(np.isfinite(a).all() for source in (skeleton, rgb) for a in source.values()):
        raise ValueError("M2_NONFINITE_INPUT")
    weights = np.ones(len(rgb["label_ids"]), dtype=float)
    audit = {
        "algorithm": "quality-gated-disagreement-v1",
        "quality_available": False,
        "adapted_label_ids": [],
        "rgb_weights": {},
        "score_semantics": "uncalibrated; adapted scores are threshold-relative margins mapped to [0,1]",
    }
    if usable is None or requested is None:
        return baseline, {**audit, "reason": "quality_unavailable_keep_frozen_baseline"}
    if type(usable) is not int or type(requested) is not int or not 0 <= usable <= requested or requested < 1:
        raise ValueError("M2_RGB_QUALITY_INVALID")
    ratio = usable / requested
    audit.update(quality_available=True, usable_ratio=ratio)
    # Below the existing eight-of-sixteen RGB support boundary, stop rather than fabricate evidence.
    if ratio < 0.5:
        raise ValueError("M2_RGB_QUALITY_BELOW_SUPPORTED_RANGE")
    quality_weight = config.minimum_rgb_weight + (1 - config.minimum_rgb_weight) * np.clip(
        (ratio - 0.5) / (config.full_rgb_ratio - 0.5), 0, 1
    )
    arrays = {k: v.copy() for k, v in baseline.items()}
    for j, label in enumerate(rgb["label_ids"]):
        label = int(label)
        col = LABELS.index(label)
        a = bool(skeleton["eligible"][0, col])
        b = bool(baseline["eligible"][0, col])
        # The existing hand specialist is itself separately quality-gated; preserve it when observable.
        protected_hand = label in (1, 2) and hand_available is True and bool(rgb["hand_override"][0, j])
        weight = 1.0 if protected_hand or a == b else float(quality_weight)
        weights[j] = weight
        if weight >= 1:
            continue
        if weight == 0:
            arrays["scores"][0, col] = skeleton["scores"][0, col]
            arrays["thresholds"][0, col] = skeleton["thresholds"][0, col]
            arrays["eligible"][0, col] = a
        else:
            ma = float(logit(skeleton["scores"][0, col]) - logit(skeleton["thresholds"][0, col]))
            mb = float(logit(rgb["scores"][0, j]) - logit(rgb["thresholds"][0, j]))
            # Eligibility includes frozen branch vetoes; a vetoed score cannot become positive evidence.
            ma = max(ma, 1e-9) if a else min(ma, -1e-9)
            mb = max(mb, 1e-9) if b else min(mb, -1e-9)
            margin = (1 - weight) * ma + weight * mb
            arrays["scores"][0, col] = 1 / (1 + np.exp(-np.clip(margin, -50, 50)))
            arrays["thresholds"][0, col] = 0.5
            arrays["eligible"][0, col] = margin >= 0
        audit["adapted_label_ids"].append(label)
    qualified = np.asarray([reliability[str(label)]["qualified"] for label in LABELS])
    arrays["predictions"] = frozen.guarded(
        arrays["eligible"],
        arrays["scores"][:, :11],
        arrays["thresholds"],
        arrays["predicted_count"],
        qualified,
    )
    audit["rgb_weights"] = {str(int(label)): float(w) for label, w in zip(rgb["label_ids"], weights)}
    audit["reason"] = (
        "quality_gated_conflicts" if audit["adapted_label_ids"] else "no_low_quality_unprotected_conflict"
    )
    return arrays, audit
