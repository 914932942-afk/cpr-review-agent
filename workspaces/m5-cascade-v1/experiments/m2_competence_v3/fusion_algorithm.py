"""Competence-protected conflict resolution with scope-constrained selection."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    full_rgb_ratio: float = 0.875
    minimum_rgb_weight: float = 0.0
    boundary_band: float = 0.35
    skeleton_advantage: float = 1.0
    protect_specialists: bool = True
    lock_unshared: bool = True

    def __post_init__(self):
        if not 0.5 < self.full_rgb_ratio <= 1 or not 0 <= self.minimum_rgb_weight <= 0.5:
            raise ValueError("M2_CONFIG_INVALID")
        if not 0 <= self.boundary_band <= 2 or not 0 < self.skeleton_advantage <= 4:
            raise ValueError("M2_BOUNDARY_CONFIG_INVALID")


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
        "algorithm": "competence-scope-disagreement-v3",
        "quality_available": False,
        "adapted_label_ids": [],
        "boundary_adapted_label_ids": [],
        "protected_specialist_label_ids": [],
        "scope_locked_label_ids": [int(k) for k in LABELS if k not in rgb["label_ids"]],
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
        ma = float(logit(skeleton["scores"][0, col]) - logit(skeleton["thresholds"][0, col]))
        mb = float(logit(rgb["scores"][0, j]) - logit(rgb["thresholds"][0, j]))
        ma = max(ma, 1e-9) if a else min(ma, -1e-9)
        mb = max(mb, 1e-9) if b else min(mb, -1e-9)
        rgb_gate_open = skeleton["scores"][0, 11] < rgb["normal_veto_threshold"][0] or bool(
            rgb["hand_override"][0, j]
        )
        class_band = config.boundary_band * (0.5 if reliability[str(label)]["qualified"] else 1.0)
        protected_specialist = config.protect_specialists and reliability[str(label)]["qualified"]
        if protected_specialist and a != b:
            audit["protected_specialist_label_ids"].append(label)
        if (
            not protected_hand
            and not protected_specialist
            and a != b
            and rgb_gate_open
            and abs(mb) <= class_band
            and abs(ma) >= abs(mb) + config.skeleton_advantage
        ):
            weight = min(weight, 0.5)
            audit["boundary_adapted_label_ids"].append(label)
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
    if config.lock_unshared:
        # Preserve frozen decisions outside RGB scope. Only shared weak labels compete for
        # the remaining cardinality budget; eligibility never creates a new positive.
        shared = np.isin(np.asarray(LABELS), rgb["label_ids"])
        predictions = baseline["predictions"].copy()
        predictions[:, shared] = arrays["eligible"][:, shared] & qualified[shared]
        fixed_count = int(predictions.sum())
        remaining = max(0, int(arrays["predicted_count"][0]) - fixed_count)
        candidates = np.flatnonzero(shared & ~qualified & arrays["eligible"][0])
        if len(candidates) and remaining:
            margin = logit(arrays["scores"][0, candidates]) - logit(arrays["thresholds"][0, candidates])
            order = np.argsort(-margin, kind="stable")[:remaining]
            predictions[0, candidates[order]] = True
        arrays["predictions"] = predictions
    audit["rgb_weights"] = {str(int(label)): float(w) for label, w in zip(rgb["label_ids"], weights)}
    audit["reason"] = (
        "quality_gated_conflicts" if audit["adapted_label_ids"] else "no_low_quality_unprotected_conflict"
    )
    return arrays, audit
