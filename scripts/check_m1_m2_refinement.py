"""Factorial replay of frozen choices on 26 already-inspected real-video feature caches."""

import json
from pathlib import Path

import numpy as np
from evaluate_m2_quality import load, metric
from explore_m1_coherent import burst, common
from explore_m2_competence import new, prediction

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/m1-m2-exploration-v1/combined"


def main():
    OUT.mkdir(exist_ok=False)
    data = ROOT / "data/public_cpr"
    cfg1 = burst.Config(**json.loads((OUT.parent / "m1-coherent/selection.json").read_text())["config"])
    cfg2 = new.Config(**json.loads((OUT.parent / "m2-competence/selection.json").read_text())["config"])
    cached = load(ROOT / "data/m2-quality-v1/experiment-01/raw-readouts.npz")
    full = load(data / "rgb-evaluation-v6/raw-predictions.npz")
    lookup = {str(k): i for i, k in enumerate(full["record_ids"])}
    jobs = {r["record_id"]: r for r in json.loads((data / "rgb-video-acceptance-v4/jobs.json").read_text())}
    source = {
        r["record_id"]: r
        for r in json.loads((data / "fullpose-raw-v1/feature-manifest.json").read_text())["rows"]
    }
    values = {"accepted_m1": [], "candidate_m1": []}
    motion = {k: [] for k in values}
    audits = []
    hashes = {}
    for key in cached["record_ids"]:
        key = str(key)
        path = data / "rgb-video-acceptance-v4/runs" / jobs[key]["key"] / "inference.npz"
        assert common.sha(path) == source[key]["body_npz_sha256"]
        hashes[path.relative_to(ROOT).as_posix()] = common.sha(path)
        a = load(path)
        for name, algorithm, cfg in [
            ("accepted_m1", common.old, common.old.Config()),
            ("candidate_m1", burst, cfg1),
        ]:
            g, m, audit = algorithm.features(a["keypoint"][0], a["keypoint_score"][0], cfg)
            row = full["features"][lookup[key]].copy()
            row[512:783] = g
            values[name].append(row)
            motion[name].append(m)
            if name == "candidate_m1":
                audits.append({"record_id": key, **audit})
    skeletons = {
        name: common.skeleton_readout(np.stack(v), np.stack(motion[name])) for name, v in values.items()
    }
    for k, v in skeletons["accepted_m1"].items():
        np.testing.assert_allclose(v, cached["sk_" + k], atol=1e-8)
    rg = {k[4:]: v for k, v in cached.items() if k.startswith("rgb_")}
    results, predictions = {}, {}
    for m1, sk in skeletons.items():
        for m2, cfg in [("accepted_m2", None), ("candidate_m2", cfg2)]:
            name = m1 + "__" + m2
            p, _ = prediction((sk, rg, cached["usable"], cached["requested"], cached["hand_available"]), cfg)
            predictions[name] = p
            results[name] = metric(p, cached["targets"])
    base = predictions["accepted_m1__accepted_m2"]
    final = predictions["candidate_m1__candidate_m2"]
    checks = {
        "m1_alone_no_f1_regression": results["candidate_m1__accepted_m2"]["macro_f1"]
        >= results["accepted_m1__accepted_m2"]["macro_f1"],
        "combined_no_f1_regression": results["candidate_m1__candidate_m2"]["macro_f1"]
        >= results["accepted_m1__candidate_m2"]["macro_f1"],
        "no_additional_normal_alarm": results["candidate_m1__candidate_m2"]["normal_false_alarms"]
        <= results["accepted_m1__accepted_m2"]["normal_false_alarms"],
        "26_complete": len(base) == 26,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": results,
        "source_hashes": hashes,
        "scope": "Already-inspected cached video features, not fresh decoding, blind or cross-person validation.",
        "m1_audits": audits,
        "rows": [
            {
                "record_id": str(key),
                "before": [k for k, yes in zip(common.helper.LABELS, base[i]) if yes],
                "after": [k for k, yes in zip(common.helper.LABELS, final[i]) if yes],
                "scores": skeletons["candidate_m1"]["scores"][i].tolist(),
            }
            for i, key in enumerate(cached["record_ids"])
        ],
    }
    with (OUT / "raw26.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    np.savez_compressed(OUT / "predictions.npz", record_ids=cached["record_ids"], **predictions)
    print(
        json.dumps(
            {
                "checks": checks,
                "f1": {k: v["macro_f1"] for k, v in results.items()},
                "natural_burst_repairs": sum(a["short_burst_points_corrected"] for a in audits),
            }
        )
    )


if __name__ == "__main__":
    main()
