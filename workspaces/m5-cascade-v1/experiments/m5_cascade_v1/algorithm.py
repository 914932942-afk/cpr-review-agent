"""Compile active classifier dependencies before materialising expensive features.

Exact conditional computation, not score approximation or a new trained model.
Quality gates choose the same experts as the frozen baseline. Uncomputed feature
slots are never passed to a head that depends on them, and are not stored as data.
"""

import numpy as np

VERSION = "quality-dependency-cascade-v1"
LABELS = (1, 2, 5, 7, 9)
DIMENSION = 1893
VISUAL_DIMENSION = 1536
SHAPE_DIMENSION = 576
HAND_VISUAL_INDICES = np.r_[np.arange(384), np.arange(768, 1152)]


def indices(expert, maximum):
    values = expert["indices"]
    values = values.numpy() if hasattr(values, "numpy") else np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or not len(values):
        raise ValueError("M5_INVALID_EXPERT_INDICES")
    if len(set(values.tolist())) != len(values) or np.any(values < 0) or np.any(values >= maximum):
        raise ValueError("M5_EXPERT_DEPENDENCY_OUT_OF_RANGE")
    return values


def compile_plan(rgb_state, hand_state, shape_available):
    if type(shape_available) is not bool:
        raise ValueError("M5_QUALITY_GATE_NOT_BOOLEAN")
    rgb = {int(e["label_id"]): e for e in rgb_state["experts"]}
    hand = {int(e["label_id"]): e for e in hand_state["experts"]}
    if (
        tuple(rgb) != LABELS
        or set(hand) != {1, 2}
        or len(rgb) != len(rgb_state["experts"])
        or len(hand) != len(hand_state["experts"])
    ):
        raise ValueError("M5_EXPERT_SCOPE_CHANGED")
    if rgb_state["feature_dimension"] != DIMENSION:
        raise ValueError("M5_FEATURE_LAYOUT_CHANGED")
    dependencies = set()
    active = {}
    for label in LABELS:
        # Check inactive models as well: a malformed saved bundle must fail closed.
        rgb_deps = indices(rgb[label], DIMENSION)
        if label in hand:
            hand_deps = indices(hand[label], SHAPE_DIMENSION + len(HAND_VISUAL_INDICES))
        if shape_available and label in hand:
            semantic = hand_deps[hand_deps >= SHAPE_DIMENSION] - SHAPE_DIMENSION
            dependencies.update(HAND_VISUAL_INDICES[semantic].tolist())
            active[str(label)] = "hand-shape"
        else:
            dependencies.update(rgb_deps.tolist())
            active[str(label)] = "rgb-context"
    required = sorted(dependencies)
    needs_visual = any(i < VISUAL_DIMENSION for i in required)
    return {
        "algorithm": VERSION,
        "active_experts": active,
        "shape_available": shape_available,
        "required_rgb_indices": required,
        "dense_visual_required": needs_visual,
        "route": "full-rgb-fallback" if needs_visual else "dependency-pruned",
        "certificate": "All input indices of each active expert are materialised; baseline quality gates and thresholds unchanged.",
    }


def context_readout(context, shape, rgb_state, hand_state, plan, predict_experts, predict_hands):
    context = np.asarray(context)
    shape = np.asarray(shape)
    expected = compile_plan(rgb_state, hand_state, plan["shape_available"])
    if expected != plan or plan["dense_visual_required"]:
        raise ValueError("M5_PRUNING_CERTIFICATE_INVALID")
    if (
        context.shape != (1, DIMENSION - VISUAL_DIMENSION)
        or shape.shape != (1, SHAPE_DIMENSION)
        or not np.isfinite(context).all()
        or not np.isfinite(shape).all()
    ):
        raise ValueError("M5_CONTEXT_LAYOUT_INVALID")
    # Internal layout adapter only. Its unused prefix is never persisted or read.
    layout = np.zeros((1, DIMENSION), dtype=np.float32)
    layout[:, VISUAL_DIMENSION:] = context
    active_rgb = [
        e for e in rgb_state["experts"] if plan["active_experts"][str(int(e["label_id"]))] == "rgb-context"
    ]
    if any(np.any(indices(e, DIMENSION) < VISUAL_DIMENSION) for e in active_rgb):
        raise ValueError("M5_UNMATERIALISED_FEATURE_DEPENDENCY")
    values = predict_experts(layout, {**rgb_state, "experts": active_rgb})
    active_hand = [
        e for e in hand_state["experts"] if plan["active_experts"][str(int(e["label_id"]))] == "hand-shape"
    ]
    if any(np.any(indices(e, 1344) >= SHAPE_DIMENSION) for e in active_hand):
        raise ValueError("M5_UNMATERIALISED_HAND_SEMANTICS")
    hand_layout = np.zeros((1, 1344), dtype=np.float32)
    hand_layout[:, :SHAPE_DIMENSION] = shape
    if active_hand:
        values.update(predict_hands(hand_layout, {**hand_state, "experts": active_hand}))
    scores = np.stack([values[k][0] for k in LABELS], axis=1)
    eligible = np.stack([values[k][1] for k in LABELS], axis=1)
    threshold = {int(e["label_id"]): float(e["threshold"]) for e in active_rgb + active_hand}
    return {
        "label_ids": np.asarray(LABELS),
        "scores": scores,
        "eligible": eligible,
        "thresholds": np.asarray([[threshold[k] for k in LABELS]]),
        "hand_override": np.asarray([[plan["active_experts"][str(k)] == "hand-shape" for k in LABELS]]),
        "normal_veto_threshold": np.asarray([rgb_state["normal_veto_threshold"]]),
    }
