"""Bidirectional trusted-anchor repair for short weak-confidence tracker bursts.

High-confidence motion and long missing intervals are preserved. This algorithm
does not estimate physical depth/BPM or treat repaired points as observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ANGLES = ((5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16), (7, 5, 11), (8, 6, 12), (5, 11, 13), (6, 12, 14))
PAIRS = ((9, 10), (7, 8), (5, 6), (11, 12), (9, 11), (10, 12), (9, 13), (10, 14))


@dataclass(frozen=True)
class Config:
    visible: float = 0.2
    anchor_confidence: float = 0.45
    max_gap: int = 5
    max_gap_speed: float = 0.08  # torso lengths per original frame, not physical velocity
    residual_floor: float = 0.08
    relative_confidence: float = 0.5
    correction_strength: float = 1.0
    legacy_gap_compatibility: bool = True
    statistics_mix: float = 0
    burst_enabled: bool = True
    max_burst: int = 4
    burst_residual: float = 0.08
    anchor_velocity_difference: float = 0.06
    offset_deviation: float = 0.03


def repair_bursts(points, confidence, scale, config):
    """Correct short observed tracking jumps; never create anchors or repair long intervals."""
    repaired = points.copy()
    changed = np.zeros_like(confidence, dtype=bool)
    rejected_motion = 0
    for joint in range(17):
        weak = (confidence[:, joint] >= config.visible) & (confidence[:, joint] < config.anchor_confidence)
        transitions = np.diff(np.r_[False, weak, False].astype(int))
        starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
        for start, end in zip(starts, ends):
            if not 2 <= end - start <= config.max_burst or start < 3 or end + 2 >= len(points):
                continue
            # Two untouched observations on either side establish velocity and continuity.
            anchor_ids = [start - 2, start - 1, end, end + 1]
            if (confidence[anchor_ids, joint] < config.anchor_confidence).any():
                continue
            left_velocity = points[start - 1, joint] - points[start - 2, joint]
            right_velocity = points[end + 1, joint] - points[end, joint]
            if np.linalg.norm(left_velocity - right_velocity) / scale > config.anchor_velocity_difference:
                rejected_motion += 1
                continue
            speed = np.linalg.norm(points[end, joint] - points[start - 1, joint]) / scale / (end - start + 1)
            if speed > config.max_gap_speed:
                rejected_motion += 1
                continue
            ratio = np.arange(1, end - start + 1) / (end - start + 1)
            proposed = (1 - ratio[:, None]) * points[start - 1, joint] + ratio[:, None] * points[end, joint]
            distance = np.linalg.norm(points[start:end, joint] - proposed, axis=1)
            residual_vectors = points[start:end, joint] - proposed
            deviation = np.linalg.norm(residual_vectors - np.median(residual_vectors, axis=0), axis=1).max()
            # A translated tracker burst has a common offset and two abrupt boundaries.
            # Curved/partial true movement is left alone instead of being straightened.
            if (
                distance <= config.burst_residual * scale
            ).any() or deviation > config.offset_deviation * scale:
                continue
            mask = distance > config.burst_residual * scale
            repaired[start:end, joint][mask] = proposed[mask]
            changed[start:end, joint] = mask
    return repaired, changed, rejected_motion


def _scale(points, confidence, threshold):
    hip, shoulder = points[:, [11, 12]].mean(1), points[:, [5, 6]].mean(1)
    size = np.linalg.norm(hip - shoulder, axis=1)
    mask = (confidence[:, [5, 6, 11, 12]] >= threshold).all(1) & (size > 1e-6)
    if mask.sum() >= 8:
        return float(np.median(size[mask]))
    size = np.linalg.norm(points[:, 5] - points[:, 6], axis=1)
    mask = (confidence[:, [5, 6]] >= threshold).all(1) & (size > 1e-6)
    return float(np.median(size[mask])) if mask.sum() >= 8 else None


def stabilize(points, confidence, config=Config()):
    p, q = np.asarray(points, float).copy(), np.asarray(confidence, float).copy()
    if (
        p.ndim != 3
        or p.shape[1:] != (17, 2)
        or q.shape != p.shape[:2]
        or len(p) < 8
        or not np.isfinite(p).all()
        or not np.isfinite(q).all()
        or (q < 0).any()
        or (q > 1).any()
    ):
        raise ValueError("Expected finite original-time COCO17 coordinates and confidence in [0,1]")
    observed = q >= config.visible
    scale = _scale(p, q, config.anchor_confidence)
    corrected, reconstructed = np.zeros_like(observed), np.zeros_like(observed)
    burst_corrected = np.zeros_like(observed)
    burst_motion_rejections = 0
    if scale is not None:
        if config.burst_enabled:
            p, burst_corrected, burst_motion_rejections = repair_bursts(p, q, scale, config)
        offsets = np.asarray([-2, -1, 1, 2])
        neighbours, reliability = [], []
        for offset in offsets:
            shifted, scores = np.zeros_like(p), np.zeros_like(q)
            if offset < 0:
                shifted[-offset:], scores[-offset:] = p[:offset], q[:offset]
            else:
                shifted[:-offset], scores[:-offset] = p[offset:], q[offset:]
            neighbours.append(shifted)
            reliability.append(np.where(scores >= config.anchor_confidence, scores**2, 0))
        values = np.stack(neighbours, axis=2)
        weights = np.stack(reliability, axis=2) * np.exp(-(offsets**2) / 8)
        sw, st, stt = weights.sum(2), (weights * offsets).sum(2), (weights * offsets**2).sum(2)
        sy, sty = (
            (weights[..., None] * values).sum(2),
            (weights[..., None] * values * offsets[None, None, :, None]).sum(2),
        )
        determinant = sw * stt - st**2
        prediction = (stt[..., None] * sy - st[..., None] * sty) / np.maximum(determinant[..., None], 1e-12)
        slope = (sw[..., None] * sty - st[..., None] * sy) / np.maximum(determinant[..., None], 1e-12)
        residual = np.linalg.norm(
            values - prediction[:, :, None] - slope[:, :, None] * offsets[None, None, :, None], axis=3
        )
        residual[weights <= 0] = np.nan
        # A local line protects coherent periodic motion; only isolated weaker-confidence points move.
        ordered = np.sort(np.where(np.isfinite(residual), residual, np.inf), axis=2)
        count = (weights > 0).sum(2)
        middle = np.maximum((count - 1) // 2, 0)[..., None]
        mad = np.take_along_axis(ordered, middle, axis=2)[..., 0]
        limit = np.maximum(config.residual_floor * scale, 3 * mad)
        flank = np.minimum(np.max(weights[:, :, :2], axis=2), np.max(weights[:, :, 2:], axis=2))
        corrected = (
            observed
            & (count >= 3)
            & (determinant > 1e-9)
            & (q < config.relative_confidence * np.sqrt(flank))
            & (np.linalg.norm(p - prediction, axis=2) > limit)
        )
        p[corrected] = (1 - config.correction_strength) * p[
            corrected
        ] + config.correction_strength * prediction[corrected]
        for joint in range(17):
            anchors = np.flatnonzero(observed[:, joint])
            gaps = np.diff(anchors) - 1
            usable = (gaps > 0) & (gaps <= config.max_gap)
            for left, right in zip(anchors[:-1][usable], anchors[1:][usable]):
                gap = right - left - 1
                if not config.legacy_gap_compatibility:
                    if min(q[left, joint], q[right, joint]) < config.anchor_confidence:
                        continue
                    if (
                        np.linalg.norm(p[right, joint] - p[left, joint]) / scale / (right - left)
                        > config.max_gap_speed
                    ):
                        continue
                ratio = np.arange(1, gap + 1) / (gap + 1)
                p[left + 1 : right, joint] = (1 - ratio[:, None]) * p[left, joint] + ratio[:, None] * p[
                    right, joint
                ]
                q[left + 1 : right, joint] = min(q[left, joint], q[right, joint]) * 0.5
                reconstructed[left + 1 : right, joint] = True
    supported = observed | reconstructed
    p[~supported] = np.nan
    audit = {
        "schema": "cpr-m1-stability-v3",
        "implementation_revision": "m1-coherent-offset-v3",
        "config": asdict(config),
        "frames": len(p),
        "reference_available": scale is not None,
        "original_observed_points": int(observed.sum()),
        "isolated_points_corrected": int(corrected.sum()),
        "short_burst_points_corrected": int(burst_corrected.sum()),
        "burst_motion_rejections": int(burst_motion_rejections),
        "burst_rule": "2..4 weak observed frames; unchanged trusted anchors and bidirectional continuity",
        "short_gap_points_reconstructed": int(reconstructed.sum()),
        "unavailable_points_preserved": int((~supported).sum()),
        "reconstructed_points_are_observations": False,
        "time_basis": "original frame ordinal",
    }
    return p, q, supported, audit


def _stats(values, weights, mix):
    values = np.asarray(values).reshape(len(values), -1)
    weights = np.broadcast_to(weights, values.shape)
    if mix == 0:
        available = np.isfinite(values) & (weights > 0)
        complete = available.all(0)
        result = np.zeros((values.shape[1], 4), dtype=float)
        if complete.any():
            dense = values[:, complete]
            quantiles = np.percentile(dense, [10, 90], axis=0)
            result[complete] = np.stack([dense.mean(0), dense.std(0), *quantiles], axis=1)
        for column in np.flatnonzero(~complete):
            subset = values[available[:, column], column]
            if len(subset):
                result[column] = [subset.mean(), subset.std(), *np.percentile(subset, [10, 90])]
        return result.ravel()
    result = []
    for column, w in zip(values.T, weights.T):
        good = np.isfinite(column) & (w > 0)
        a, w = column[good], w[good]
        if not len(a):
            result.extend([0.0] * 4)
            continue
        ordinary = np.r_[a.mean(), a.std(), np.percentile(a, [10, 90])]
        if mix and np.ptp(w) > 1e-12:
            mean = np.average(a, weights=w)
            order = np.argsort(a, kind="stable")
            cumulative = np.cumsum(w[order]) - 0.5 * w[order]
            cumulative = (cumulative - cumulative[0]) / max(cumulative[-1] - cumulative[0], 1e-12)
            weighted = np.r_[
                mean,
                np.sqrt(np.average((a - mean) ** 2, weights=w)),
                np.interp([0.1, 0.9], cumulative, a[order]),
            ]
            ordinary = (1 - mix) * ordinary + mix * weighted
        result.extend(ordinary)
    return np.asarray(result)


def features(points, confidence, config=Config()):
    p, q, supported, audit = stabilize(points, confidence, config)
    hip, shoulder = p[:, [11, 12]].mean(1), p[:, [5, 6]].mean(1)
    use_hip = np.isfinite(hip).all(1).sum() >= 8
    origin = hip if use_hip else shoulder
    sizes = np.linalg.norm(hip - shoulder, axis=1)
    valid = sizes[np.isfinite(sizes) & (sizes > 1e-6)]
    if len(valid) < 8:
        sizes = np.linalg.norm(p[:, 5] - p[:, 6], axis=1)
        valid = sizes[np.isfinite(sizes) & (sizes > 1e-6)]
    if not len(valid) or np.isfinite(origin).all(1).sum() < 8:
        audit["geometry_available"] = False
        return np.zeros(271, np.float32), np.zeros(176, np.float32), audit
    scale = float(np.median(valid))
    normalized = (p - origin[:, None]) / scale
    mix = config.statistics_mix
    reliability = np.where(supported, np.maximum(q, 0.001), 0)
    xyw = np.repeat(reliability[:, :, None], 2, axis=2).reshape(len(p), -1)
    values = list(_stats(normalized.reshape(len(p), -1), xyw, mix))
    velocity = np.diff(normalized, axis=0).reshape(len(p) - 1, -1)
    vw = np.minimum(xyw[:-1], xyw[1:])
    for column, w in zip(velocity.T, vw.T):
        good = np.isfinite(column) & (w > 0)
        a, w = column[good], w[good]
        if not len(a):
            values.extend([0.0, 0.0])
            continue
        if mix == 0:
            values.extend([np.abs(a).mean(), a.std()])
        else:
            values.extend(
                [
                    (1 - mix) * np.abs(a).mean() + mix * np.average(np.abs(a), weights=w),
                    (1 - mix) * a.std()
                    + mix * np.sqrt(np.average((a - np.average(a, weights=w)) ** 2, weights=w)),
                ]
            )
    angles, aw = [], []
    for a, b, c in ANGLES:
        u, v = normalized[:, a] - normalized[:, b], normalized[:, c] - normalized[:, b]
        denominator = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        value = (u * v).sum(1) / np.maximum(denominator, 1e-8)
        value[(denominator < 1e-8) | ~np.isfinite(denominator)] = np.nan
        angles.append(np.clip(value, -1, 1))
        aw.append(reliability[:, [a, b, c]].min(1))
    values.extend(_stats(np.stack(angles, 1), np.stack(aw, 1), mix))
    distances = np.stack([np.linalg.norm(normalized[:, a] - normalized[:, b], axis=1) for a, b in PAIRS], 1)
    values.extend(_stats(distances, np.stack([reliability[:, [a, b]].min(1) for a, b in PAIRS], 1), mix))
    values.extend([1.0, float(use_hip), float(not use_hip)])
    geometry = np.asarray(values, np.float32)
    tilt, tw = [], []
    for a, b in [(5, 9), (6, 10), (5, 7), (6, 8), (7, 9), (8, 10)]:
        vector = p[:, b] - p[:, a]
        length = np.linalg.norm(vector, axis=1)
        value = np.abs(vector[:, 0]) / np.maximum(length, 1e-6)
        value[length < 1e-6] = np.nan
        tilt.append(value)
        tw.append(reliability[:, [a, b]].min(1))
    # Clip-constant centering preserves vertical hip motion, matching the baseline convention.
    medians = np.array(
        [
            np.median(p[np.isfinite(p[:, j, 1]), j, 1]) if np.isfinite(p[:, j, 1]).any() else 0
            for j in range(17)
        ]
    )
    centered = (p[:, :, 1] - medians) / scale
    pairs = [(13, 11), (14, 12), (15, 11), (16, 12)]
    relative = np.stack([(p[:, a, 1] - p[:, b, 1]) / scale for a, b in pairs], 1)
    motion = np.r_[
        _stats(np.stack(tilt, 1), np.stack(tw, 1), mix),
        _stats(centered, reliability, mix),
        _stats(relative, np.stack([reliability[:, [a, b]].min(1) for a, b in pairs], 1), mix),
        _stats(np.abs(np.diff(centered, axis=0)), np.minimum(reliability[:-1], reliability[1:]), mix),
    ].astype(np.float32)
    if (
        geometry.shape != (271,)
        or motion.shape != (176,)
        or not np.isfinite(geometry).all()
        or not np.isfinite(motion).all()
    ):
        raise ValueError("M1 derived-feature contract failed")
    audit.update(
        geometry_available=True,
        torso_scale_pixels=scale,
        modified_components=["COCO17 geometry271", "mechanics176"],
        unchanged_components=[
            "PoseC3D512",
            "sparse hand632",
            "spectrum94",
            "classification weights and thresholds",
        ],
    )
    return geometry, motion, audit
