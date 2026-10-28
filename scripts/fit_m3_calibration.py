"""Finite grouped post-hoc calibration; freeze choices before held-out engineering checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/m3-calibration-v1"
POOLS = {"rgb": [1, 2, 5, 7, 9], "body": [3, 4, 6, 8, 11, 13]}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(name, value):
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def load():
    with np.load(OUT / "prepared.npz", allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def take(data, mask):
    return {k: v[mask] for k, v in data.items()}


def margin(data):
    return logit(np.clip(data["scores"], 1e-7, 1 - 1e-7)) - logit(np.clip(data["thresholds"], 1e-7, 1 - 1e-7))


def predict(data, config):
    denominator = (
        1
        + config["quality_penalty"] * (1 - data["quality"])
        + config["disagreement_penalty"] * data["disagreement"]
    )
    return expit((config["slope"] * margin(data) + config["bias"]) / denominator)


def fit(data, q, d, ridge, temperature_only=False):
    m, y = margin(data), data["targets"].astype(float)
    denominator = 1 + q * (1 - data["quality"]) + d * data["disagreement"]

    def objective(x):
        bias = 0 if temperature_only else x[1]
        z = (np.exp(x[0]) * m + bias) / denominator
        return float(np.mean(np.logaddexp(0, z) - y * z) + ridge * (x[0] ** 2 + bias**2))

    result = minimize(
        objective,
        np.zeros(1 if temperature_only else 2),
        method="L-BFGS-B",
        bounds=[(-3, 3)] if temperature_only else [(-3, 3), (-8, 8)],
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("M3_CALIBRATION_FIT_FAILED")
    return {
        "slope": float(np.exp(result.x[0])),
        "bias": 0.0 if temperature_only else float(result.x[1]),
        "quality_penalty": q,
        "disagreement_penalty": d,
        "ridge": ridge,
    }


def oof(data, q, d, ridge, temperature_only=False):
    result = np.zeros(len(data["targets"]))
    groups = data["groups"]
    folds = GroupKFold(n_splits=min(5, len(set(groups))))
    for train, val in folds.split(groups, groups=groups):
        assert not set(groups[train]) & set(groups[val])
        cfg = fit(take(data, train), q, d, ridge, temperature_only)
        result[val] = predict(take(data, val), cfg)
    return result


def keep(data, p, policy):
    confidence = np.where(data["flagged"], p, 1 - p)
    threshold = np.where(data["flagged"], policy["positive_threshold"], policy["negative_threshold"])
    return (data["quality"] >= policy["minimum_quality"]) & (confidence >= threshold)


def metrics(data, p, accepted=None):
    y = data["targets"].astype(bool)
    p = np.asarray(p)
    accepted = np.ones(len(y), bool) if accepted is None else accepted
    wrong = data["flagged"] != y
    ece = 0.0
    for i in range(10):
        mask = (p >= i / 10) & (p < (i + 1) / 10 if i < 9 else p <= 1)
        if mask.any():
            ece += mask.mean() * abs(float(p[mask].mean() - y[mask].mean()))
    positive = data["flagged"].astype(bool)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(
            np.mean(-y.astype(float) * np.log(np.clip(p, 1e-9, 1)) - (~y) * np.log(np.clip(1 - p, 1e-9, 1)))
        ),
        "ece10": float(ece),
        "coverage": float(accepted.mean()),
        "flagged_coverage": float(accepted[positive].mean()) if positive.any() else 1.0,
        "full_decision_error": float(wrong.mean()),
        "selective_error": float(wrong[accepted].mean()) if accepted.any() else None,
        "wrong_positions": int(wrong.sum()),
        "wrong_accepted": int((wrong & accepted).sum()),
        "wrong_deferred": int((wrong & ~accepted).sum()),
        "uncertain_positions": int((~accepted).sum()),
        "positions": len(y),
        "records": len(set(data["record_ids"])),
    }


def policy_fit(data, p):
    choices = []
    for pos in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for neg in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            for quality in [0.0, 0.5, 0.65, 0.8]:
                cfg = {"positive_threshold": pos, "negative_threshold": neg, "minimum_quality": quality}
                accepted = keep(data, p, cfg)
                m = metrics(data, p, accepted)
                if m["coverage"] < 0.75 or m["flagged_coverage"] < 0.65:
                    continue
                if any(accepted[data["label_ids"] == label].mean() < 0.5 for label in set(data["label_ids"])):
                    continue
                choices.append((m["selective_error"], -m["coverage"], pos + neg + quality, cfg))
    if not choices:
        raise ValueError("M3_NO_SUPPORTED_REVIEW_POLICY")
    return min(choices, key=lambda x: x[:3])[-1]


def select():
    data = load()
    write(
        "protocol.json",
        {
            "algorithm_sha256": sha(ROOT / "experiments/m3_calibration_v1/algorithm.py"),
            "prepared_sha256": sha(OUT / "prepared.npz"),
            "evaluator_sha256": sha(Path(__file__)),
            "candidates": {
                "quality_penalty": [0, 1, 3],
                "rgb_disagreement_penalty": [0, 1, 3],
                "ridge": [0.001, 0.01, 0.1],
            },
            "fit": "Few-parameter positive-slope shared logistic/temperature calibration per actual score source. Grouped post-hoc OOF; all stress variants of a recording stay in the same fold. Body uses original validation196 subdivided; RGB uses original paired32+6. No independent-blind claim.",
            "selection": "OOF Brier across declared conditions; ties prefer smaller penalties and stronger ridge. Policy from same OOF, coverage>=.75, flagged coverage>=.65, each-label coverage>=.5. No validation/raw parameter choice.",
            "validation_gates": "Each pool clean Brier <= temperature baseline+.015; overall Brier <= raw-.005; selective coverage>=.60 and flagged coverage>=.40; overall selective error <= full baseline error. Raw26 Brier <= raw+.02, coverage>=.60, flagged coverage>=.40, raw CV predictions unchanged; value >=1 deferred erroneous decision or >=.005 Brier improvement over temperature baseline.",
            "public_data_only": True,
            "locked_test_opened": False,
            "new_visual_training": False,
        },
    )
    selections = {}
    report = {}
    outputs = {}
    for name, labels in POOLS.items():
        train = take(data, (data["splits"] == "train") & np.isin(data["label_ids"], labels))
        baseline = oof(train, 0, 0, 0.01, True)
        candidates = []
        for q in [0, 1, 3]:
            for d in [0, 1, 3] if name == "rgb" else [0]:
                for ridge in [0.001, 0.01, 0.1]:
                    pp = oof(train, q, d, ridge)
                    candidates.append(
                        (float(np.mean((pp - train["targets"]) ** 2)), q + d, -ridge, q, d, ridge, pp)
                    )
        chosen = min(candidates, key=lambda x: x[:3])
        _, _, _, q, d, ridge, pp = chosen
        cfg = fit(train, q, d, ridge)
        policy = policy_fit(train, pp)
        selections[name] = {**cfg, **policy}
        report[name] = {
            "temperature_only_oof": metrics(train, baseline),
            "selected_oof": metrics(train, pp, keep(train, pp, policy)),
            "candidates": [{"brier": v[0], "q": v[3], "d": v[4], "ridge": v[5]} for v in candidates],
            "temperature_parameters": fit(train, 0, 0, 0.01, True),
        }
        outputs[name + "__oof"] = pp
    write("training.json", report)
    write(
        "selection.json",
        {
            "selected_before_validation": True,
            "protocol_sha256": sha(OUT / "protocol.json"),
            "parameters": selections,
            "upstream_sources": read(OUT / "prepared-audit.json")["upstream_sources"],
        },
    )
    np.savez_compressed(OUT / "oof.npz", **outputs)
    print(
        json.dumps(
            {
                k: {
                    "parameters": selections[k],
                    "oof_brier": v["selected_oof"]["brier"],
                    "temperature_brier": v["temperature_only_oof"]["brier"],
                    "coverage": v["selected_oof"]["coverage"],
                }
                for k, v in report.items()
            }
        ),
        flush=True,
    )


def validate():
    selected = read(OUT / "selection.json")
    assert selected["protocol_sha256"] == sha(OUT / "protocol.json")
    assert read(OUT / "protocol.json")["algorithm_sha256"] == sha(
        ROOT / "experiments/m3_calibration_v1/algorithm.py"
    )
    data = load()
    training = read(OUT / "training.json")
    report = {}
    parts = []
    for name, labels in POOLS.items():
        d = take(data, (data["splits"] == "validation") & np.isin(data["label_ids"], labels))
        cfg = selected["parameters"][name]
        p = predict(d, cfg)
        base = predict(d, training[name]["temperature_parameters"])
        accepted = keep(d, p, cfg)
        report[name] = {
            c: {
                "raw": metrics(take(d, d["conditions"] == c), d["scores"][d["conditions"] == c]),
                "temperature": metrics(take(d, d["conditions"] == c), base[d["conditions"] == c]),
                "selected": metrics(
                    take(d, d["conditions"] == c), p[d["conditions"] == c], accepted[d["conditions"] == c]
                ),
            }
            for c in set(d["conditions"])
        }
        parts.append((d, p, base, accepted))
    joined = {k: np.concatenate([v[0][k] for v in parts]) for k in data}
    p = np.concatenate([v[1] for v in parts])
    base = np.concatenate([v[2] for v in parts])
    accepted = np.concatenate([v[3] for v in parts])
    checks = {
        "each_pool_clean_brier_supported": all(
            report[k]["clean"]["selected"]["brier"] <= report[k]["clean"]["temperature"]["brier"] + 0.015
            for k in POOLS
        ),
        "brier_better_than_raw": metrics(joined, p)["brier"]
        <= metrics(joined, joined["scores"])["brier"] - 0.005,
        "coverage_supported": all(
            metrics(v[0], v[1], v[3])["coverage"] >= 0.6
            and metrics(v[0], v[1], v[3])["flagged_coverage"] >= 0.4
            for v in parts
        ),
        "selective_error_not_worse": metrics(joined, p, accepted)["selective_error"]
        <= metrics(joined, p)["full_decision_error"],
        "value": metrics(joined, p, accepted)["wrong_deferred"] >= 1
        or metrics(joined, p)["brier"] <= metrics(joined, base)["brier"] - 0.005,
    }
    write(
        "validation.json",
        {
            "checks": checks,
            "pools": report,
            "overall": {
                "raw": metrics(joined, joined["scores"]),
                "temperature": metrics(joined, base),
                "selected": metrics(joined, p, accepted),
            },
            "previously_inspected_engineering_split": True,
        },
    )
    write(
        "validation-decision.json",
        {"passed": all(checks.values()), "checks": checks, "selection_sha256": sha(OUT / "selection.json")},
    )
    print(
        json.dumps(
            {
                "checks": checks,
                "overall": metrics(joined, p, accepted),
                "temperature_brier": metrics(joined, base)["brier"],
                "raw_brier": metrics(joined, joined["scores"])["brier"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["select", "validate"])
    globals()[parser.parse_args().stage]()
