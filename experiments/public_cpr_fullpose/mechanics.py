"""Image-axis arm tilt and unanchored motion; no physical-depth inference."""
import numpy as np
from head import CODE
from geometry import repair, statistics

DIMENSION = 176


def describe(points, scores):
    p, visible, repaired = repair(points, scores)
    hip = (p[:, 11]+p[:, 12])/2
    shoulder = (p[:, 5]+p[:, 6])/2
    scales = np.linalg.norm(hip-shoulder, axis=1)
    valid = scales[np.isfinite(scales) & (scales > 1e-6)]
    if len(valid) < 8:
        widths = np.linalg.norm(p[:, 5]-p[:, 6], axis=1)
        valid = widths[np.isfinite(widths) & (widths > 1e-6)]
    if not len(valid):
        return np.zeros(DIMENSION, np.float32)
    scale = float(np.median(valid))
    tilt = []
    for a, b in [(5, 9), (6, 10), (5, 7), (6, 8), (7, 9), (8, 10)]:
        v = p[:, b]-p[:, a]; length = np.linalg.norm(v, axis=1)
        value = np.abs(v[:, 0])/np.maximum(length, 1e-6)
        value[length < 1e-6] = np.nan
        tilt.append(value)
    # Removing a clip-constant origin preserves vertical hip movement, unlike
    # subtracting each frame's hip origin. Still invariant to camera translation.
    centered = (p[:, :, 1]-np.nanmedian(p[:, :, 1], axis=0))/scale
    relative = np.stack([(p[:, a, 1]-p[:, b, 1])/scale for a, b in [(13, 11), (14, 12), (15, 11), (16, 12)]], 1)
    speed = np.diff(centered, axis=0)
    # Absolute interframe motion does not claim a clinical frequency or depth.
    values = np.asarray(statistics(np.stack(tilt, 1))+statistics(centered)+statistics(relative)+statistics(np.abs(speed)), np.float32)
    if values.shape != (DIMENSION,) or not np.isfinite(values).all():
        raise ValueError('Invalid mechanics representation')
    return values
