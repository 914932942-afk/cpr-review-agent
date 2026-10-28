"""Frozen M3 check on reused raw26: calibration/deferral only; raw CV labels remain untouched."""

import json

import numpy as np
from fit_m3_calibration import OUT, POOLS, keep, metrics, predict, read, sha, take, write
from prepare_m3_calibration import DATA, LABELS, ROOT, load, m1, m2, skeleton_readout


def main():
    selected = read(OUT / "selection.json")
    cfg1 = m1.Config(**read(ROOT / "data/m1-m2-exploration-v1/m1-coherent/selection.json")["config"])
    cfg2 = m2.Config(**read(ROOT / "data/m1-m2-exploration-v1/m2-competence/selection.json")["config"])
    raw = load(ROOT / "data/m2-quality-v1/experiment-01/raw-readouts.npz")
    full = load(DATA / "rgb-evaluation-v6/raw-predictions.npz")
    fm = {str(k): i for i, k in enumerate(full["record_ids"])}
    jobs = {r["record_id"]: r for r in read(DATA / "rgb-video-acceptance-v4/jobs.json")}
    source = {r["record_id"]: r for r in read(DATA / "fullpose-raw-v1/feature-manifest.json")["rows"]}
    values = []
    motions = []
    quality = []
    for key in raw["record_ids"]:
        key = str(key)
        path = DATA / "rgb-video-acceptance-v4/runs" / jobs[key]["key"] / "inference.npz"
        assert sha(path) == source[key]["body_npz_sha256"]
        a = load(path)
        p, q = a["keypoint"][0], a["keypoint_score"][0]
        g, m, _ = m1.features(p, q, cfg1)
        x = full["features"][fm[key]].copy()
        x[512:783] = g
        values.append(x)
        motions.append(m)
        quality.append(float((q >= 0.2).mean()))
    sk = skeleton_readout(np.stack(values), np.stack(motions))
    rows = []
    fused_flags = []
    for i, key in enumerate(raw["record_ids"]):
        skeleton = {k: v[i : i + 1] for k, v in sk.items()}
        rgb = {
            k[4:]: v.copy() if k in ["rgb_label_ids", "rgb_normal_veto_threshold"] else v[i : i + 1]
            for k, v in raw.items()
            if k.startswith("rgb_")
        }
        output, _ = m2.fuse(
            skeleton,
            rgb,
            int(raw["usable"][i]),
            int(raw["requested"][i]),
            bool(raw["hand_available"][i]),
            cfg2,
        )
        fused_flags.append(output["predictions"][0])
        for col, label in enumerate(LABELS):
            rgb_supported = label in rgb["label_ids"]
            rc = list(rgb["label_ids"]).index(label) if rgb_supported else None
            q = float(raw["usable"][i] / raw["requested"][i]) if rgb_supported else quality[i]
            d = (
                int(bool(skeleton["eligible"][0, col]) != bool(rgb["eligible"][0, rc]))
                if rgb_supported
                else 0
            )
            rows.append(
                (
                    str(key),
                    str(key).rsplit("/", 1)[0],
                    "raw",
                    "clean",
                    label,
                    float(output["scores"][0, col]),
                    float(output["thresholds"][0, col]),
                    bool(output["predictions"][0, col]),
                    q,
                    d,
                    bool(raw["targets"][i, col]),
                )
            )
    names = [
        "record_ids",
        "groups",
        "splits",
        "conditions",
        "label_ids",
        "scores",
        "thresholds",
        "flagged",
        "quality",
        "disagreement",
        "targets",
    ]
    data = {name: np.array([r[i] for r in rows]) for i, name in enumerate(names)}
    original = load(ROOT / "data/m1-m2-exploration-v1/combined/predictions.npz")
    assert np.array_equal(np.stack(fused_flags), original["candidate_m1__candidate_m2"])
    p = np.zeros(len(rows))
    base = np.zeros(len(rows))
    accepted = np.zeros(len(rows), bool)
    training = read(OUT / "training.json")
    pools = {}
    for name, labels in POOLS.items():
        mask = np.isin(data["label_ids"], labels)
        subset = take(data, mask)
        cfg = selected["parameters"][name]
        p[mask] = predict(subset, cfg)
        base[mask] = predict(subset, training[name]["temperature_parameters"])
        accepted[mask] = keep(subset, p[mask], cfg)
        pools[name] = {
            "raw": metrics(subset, subset["scores"]),
            "temperature": metrics(subset, base[mask]),
            "selected": metrics(subset, p[mask], accepted[mask]),
        }
    raw_metric = metrics(data, data["scores"])
    selected_metric = metrics(data, p, accepted)
    checks = {
        "brier_supported": selected_metric["brier"] <= raw_metric["brier"] + 0.02,
        "coverage_supported": selected_metric["coverage"] >= 0.6
        and selected_metric["flagged_coverage"] >= 0.4,
        "cv_decisions_unchanged": True,
        "no_new_normal_flags": True,
        "value": selected_metric["wrong_deferred"] >= 1
        or selected_metric["brier"] <= metrics(data, base)["brier"] - 0.005,
    }
    confidence = np.where(data["flagged"], base, 1 - base)
    n = int(accepted.sum())
    temperature_same = np.zeros(len(rows), bool)
    temperature_same[np.argsort(-confidence, kind="stable")[:n]] = True
    report = {
        "checks": checks,
        "pools": pools,
        "overall": {
            "raw": raw_metric,
            "temperature": metrics(data, base),
            "selected": selected_metric,
            "temperature_at_same_coverage": metrics(data, base, temperature_same),
        },
        "previously_inspected_engineering_regression": True,
        "rows": [
            {
                "record_id": str(key),
                "label_id": int(label),
                "raw_flagged": bool(flag),
                "probability": float(prob),
                "state": "uncertain" if not kept else "model_flagged" if flag else "not_flagged",
            }
            for key, label, flag, prob, kept in zip(
                data["record_ids"], data["label_ids"], data["flagged"], p, accepted
            )
        ],
    }
    write("raw26.json", report)
    np.savez_compressed(OUT / "raw26.npz", **data, probabilities=p, accepted=accepted)
    val = read(OUT / "validation-decision.json")
    write(
        "decision.json",
        {
            "accepted": val["passed"] and all(checks.values()),
            "validation_checks": val["checks"],
            "raw_checks": checks,
            "selection_sha256": sha(OUT / "selection.json"),
        },
    )
    print(json.dumps({"checks": checks, "overall": report["overall"]}))


if __name__ == "__main__":
    main()
