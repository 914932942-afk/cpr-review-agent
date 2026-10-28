"""Halpe68 hand shape + nominal motion spectrum; deterministic, no new labels."""
import numpy as np

HAND_SAMPLES = 16
EXTRA_DIMENSION = 726
INPUT_DIMENSION = 783 + EXTRA_DIMENSION
PROFILE = {'schema': 'cpr-fullpose-descriptor-v1', 'layout': 'Halpe68: body26, left hand26:47, right hand47:68',
    'hand_samples': HAND_SAMPLES, 'hand_visibility': .15, 'body_visibility': .2,
    'hand_features': '632D: extended body-relative coordinates408, local finger shape200, between-hand distances24',
    'temporal_features': '94D: original-time motion spectra; nominal FPS, not physical BPM',
    'dimension': EXTRA_DIMENSION, 'missing': 'mask low confidence; unavailable summaries zero, no invented landmarks'}


def sample_indices(length):
    if length < 8:
        raise ValueError('Too few frames')
    return np.unique(np.linspace(0, length-1, min(length, HAND_SAMPLES)).round().astype(int))


def stats(values):
    result = []
    for column in np.asarray(values).reshape(len(values), -1).T:
        valid = column[np.isfinite(column)]
        result.extend([valid.mean(), valid.std(), *np.percentile(valid, [10, 90])] if len(valid) else [0.]*4)
    return np.asarray(result, dtype=np.float32)


def hand_features(points, confidence):
    p = np.asarray(points, dtype=np.float64).copy()
    q = np.asarray(confidence, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (68, 2) or q.shape != p.shape[:2] or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Expected finite [samples,68,2] and scores')
    visible = q >= .15
    visible[:, :26] = q[:, :26] >= .2
    p[~visible] = np.nan
    hip = (p[:, 11]+p[:, 12])/2
    shoulder = (p[:, 5]+p[:, 6])/2
    distance = np.linalg.norm(hip-shoulder, axis=1)
    available = distance[np.isfinite(distance) & (distance > 1e-6)]
    if not len(available):
        available = np.linalg.norm(p[:, 5]-p[:, 6], axis=1)
        available = available[np.isfinite(available) & (available > 1e-6)]
    if not len(available):
        return np.zeros(632, dtype=np.float32), {'hand_visible_fraction': float(visible[:, 26:].mean()), 'torso_reference_valid': False}
    scale = float(np.median(available))
    origin = np.where(np.isfinite(hip).all(1)[:, None], hip, shoulder)
    normalized = (p-origin[:, None])/scale
    world = stats(normalized[:, 17:])
    local = []
    for start in [26, 47]:
        h = p[:, start:start+21]
        palm_scale = np.linalg.norm(h[:, 5]-h[:, 17], axis=1)
        palm_scale[palm_scale < 1e-6] = np.nan
        shape = []
        for base in [1, 5, 9, 13, 17]:
            chain = h[:, [0, base, base+1, base+2, base+3]]
            bones = np.diff(chain, axis=1)
            total = np.linalg.norm(bones, axis=2).sum(1)
            total[total < 1e-6] = np.nan
            shape.append(np.linalg.norm(chain[:, -1]-chain[:, 0], axis=1)/total)
            for joint in range(3):
                u, v = bones[:, joint], bones[:, joint+1]
                denominator = np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1)
                denominator[denominator < 1e-8] = np.nan
                shape.append(np.clip((u*v).sum(1)/denominator, -1, 1))
            shape.append(np.linalg.norm(h[:, base+3]-h[:, 9], axis=1)/palm_scale)
        local.append(stats(np.stack(shape, 1)))
    cross = stats(np.stack([np.linalg.norm(p[:, 26+i]-p[:, 47+i], axis=1)/scale for i in [0, 4, 8, 12, 16, 20]], 1))
    result = np.concatenate([world, *local, cross])
    if result.shape != (632,) or not np.isfinite(result).all():
        raise ValueError('Invalid hand descriptor')
    return result, {'hand_visible_fraction': float(visible[:, 26:].mean()), 'torso_reference_valid': True}


def spectrum_features(points, confidence, fps=25.):
    p = np.asarray(points, dtype=np.float64)
    q = np.asarray(confidence, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (17, 2) or q.shape != p.shape[:2] or not np.isfinite(p).all() or not np.isfinite(q).all() or fps <= 0:
        raise ValueError('Expected full original-time COCO17 sequence')
    hip = (p[:, 11]+p[:, 12])/2
    shoulder = (p[:, 5]+p[:, 6])/2
    sizes = np.linalg.norm(hip-shoulder, axis=1)
    valid = sizes[(q[:, [5, 6, 11, 12]] >= .2).all(1) & (sizes > 1e-6)]
    scale = float(np.median(valid)) if len(valid) else 1.
    values, peaks = [], []
    for joint in [0, 5, 6, 7, 8, 9, 10, 11, 12]:
        good = q[:, joint] >= .2
        for axis in range(2):
            # Long gaps are excluded by selecting the longest visible run.
            bounds = np.flatnonzero(np.diff(np.r_[False, good, False]))
            runs = list(zip(bounds[::2], bounds[1::2]))
            left, right = max(runs, key=lambda v: v[1]-v[0]) if runs else (0, 0)
            if right-left < 32:
                values.extend([0.]*5); continue
            signal = p[left:right, joint, axis]/scale
            t = np.linspace(0, 1, len(signal))
            signal = signal - (signal[0]+t*(signal[-1]-signal[0]))
            amplitude = float(signal.std())
            power = np.abs(np.fft.rfft((signal-signal.mean())*np.hanning(len(signal))))**2
            frequency = np.fft.rfftfreq(len(signal), d=1/fps)
            band = (frequency >= .25) & (frequency <= min(5., fps/3))
            f, a = frequency[band], power[band]
            if not len(f) or a.sum() < 1e-12 or amplitude < 1e-6:
                values.extend([0.]*5); continue
            a = a/a.sum()
            peak = float(f[a.argmax()])
            values.extend([peak, float(a.max()), float(-(a*np.log(a+1e-12)).sum()/max(np.log(len(a)), 1)), float((a*f).sum()), amplitude])
            if amplitude >= .005:
                peaks.append(peak)
    values.extend([float(np.median(peaks)), float(np.std(peaks)), float(np.min(peaks)), float(np.max(peaks))] if peaks else [0.]*4)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (94,) or not np.isfinite(result).all():
        raise ValueError('Invalid spectral descriptor')
    return result


def describe(full_points, full_scores, fps=25.):
    idx = sample_indices(len(full_points))
    hands, quality = hand_features(np.asarray(full_points)[idx], np.asarray(full_scores)[idx])
    temporal = spectrum_features(np.asarray(full_points)[:, :17], np.asarray(full_scores)[:, :17], fps)
    return np.concatenate([hands, temporal]), quality
