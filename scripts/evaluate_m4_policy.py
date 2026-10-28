"""Counterfactual public-readout equivalence and predeclared-scope planning checks."""

import importlib.util
import json
import sys

import numpy as np
from prepare_m3_calibration import DATA, ROOT, load, m1, m2, read, rgb_readout, sha, skeleton_readout

OUT = ROOT / "data/m4-scheduling-v1"
spec = importlib.util.spec_from_file_location("m4_algorithm", ROOT / "experiments/m4_policy_v1/algorithm.py")
algorithm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = algorithm
spec.loader.exec_module(algorithm)
LABELS = list(algorithm.LABELS)
BODY = list(algorithm.BODY_LABELS)


def main():
    if (OUT / "decision.json").exists():
        raise FileExistsError("Preserve first M4 offline acceptance")
    assert read(OUT / "protocol.json")["algorithm_sha256"] == sha(
        ROOT / "experiments/m4_policy_v1/algorithm.py"
    )
    params = read(OUT / "parameters.json")
    cfg1 = m1.Config(**read(ROOT / "data/m1-m2-exploration-v1/m1-coherent/selection.json")["config"])
    cfg2 = m2.Config(**read(ROOT / "data/m1-m2-exploration-v1/m2-competence/selection.json")["config"])
    paired = load(ROOT / "data/m2-quality-v1/prepared/development.npz")
    full = load(DATA / "fullpose-v2/development.npz")
    fm = {str(k): i for i, k in enumerate(full["record_ids"])}
    metadata = {r["record_id"]: r for r in read(DATA / "run/records.json")}
    x = []
    motions = []
    for key in paired["record_ids"]:
        key = str(key)
        row = metadata[key]
        p = DATA / "run" / row["record_file"]
        assert sha(p) == row["record_file_sha256"]
        a = load(p)
        g, m, _ = m1.features(a["keypoint"][0], a["keypoint_score"][0], cfg1)
        v = full["features"][fm[key]].copy()
        v[512:783] = g
        x.append(v)
        motions.append(m)
    sk = skeleton_readout(np.stack(x), np.stack(motions))
    rg = rgb_readout(paired["rgb_features"], paired["hand_features"], paired["hand_available"])
    cols = [LABELS.index(k) for k in BODY]
    equivalence = []

    def sliced(source, i, n):
        return {
            k: v.copy() if k in ["label_ids", "normal_veto_threshold"] else v[i : i + 1].copy()
            for k, v in source.items()
        }

    for i, key in enumerate(paired["record_ids"]):
        fused, _ = m2.fuse(
            sliced(sk, i, len(x)),
            sliced(rg, i, len(x)),
            int(paired["usable"][i]),
            int(paired["requested"][i]),
            bool(paired["hand_available"][i]),
            cfg2,
        )
        assert np.array_equal(fused["predictions"][:, cols], sk["eligible"][i : i + 1, cols])
        assert np.array_equal(fused["scores"][:, cols], sk["scores"][i : i + 1, cols])
        assert np.array_equal(fused["thresholds"][:, cols], sk["thresholds"][i : i + 1, cols])
        equivalence.append(str(key))
    # Structural non-interference across arbitrary branch scores, with the real frozen guard.
    rng = np.random.default_rng(20261027)
    for _ in range(200):
        a = {k: v[0:1].copy() for k, v in sk.items()}
        b = sliced(rg, 0, len(x))
        a["scores"] = rng.uniform(0.001, 0.999, (1, 12))
        a["eligible"] = a["scores"][:, :11] >= a["thresholds"]
        a["predicted_count"][:] = rng.integers(0, 5)
        b["scores"] = rng.uniform(0.001, 0.999, (1, 5))
        b["eligible"] = b["scores"] >= b["thresholds"]
        fused, _ = m2.fuse(a, b, 8, 16, False, cfg2)
        assert np.array_equal(fused["predictions"][:, cols], a["eligible"][:, cols])
    cached = load(ROOT / "data/m3-calibration-v1/raw26.npz")
    raw_full = load(DATA / "rgb-evaluation-v6/raw-predictions.npz")
    lookup = {str(k): i for i, k in enumerate(raw_full["record_ids"])}
    jobs = {r["record_id"]: r for r in read(DATA / "rgb-video-acceptance-v4/jobs.json")}
    raw_metadata = {r["record_id"]: r for r in read(DATA / "fullpose-raw-v1/feature-manifest.json")["rows"]}
    values = []
    motions = []
    ids = list(dict.fromkeys(cached["record_ids"]))
    for key in ids:
        path = DATA / "rgb-video-acceptance-v4/runs" / jobs[str(key)]["key"] / "inference.npz"
        assert sha(path) == raw_metadata[str(key)]["body_npz_sha256"]
        a = load(path)
        g, m, _ = m1.features(a["keypoint"][0], a["keypoint_score"][0], cfg1)
        v = raw_full["features"][lookup[str(key)]].copy()
        v[512:783] = g
        values.append(v)
        motions.append(m)
    sk = skeleton_readout(np.stack(values), np.stack(motions))
    task_rows = []
    baseline_cost = selected_cost = 0.0
    baseline_calls = selected_calls = 0
    for i, key in enumerate(ids):
        for label in BODY:
            j = np.flatnonzero((cached["record_ids"] == key) & (cached["label_ids"] == label))[0]
            col = LABELS.index(label)
            assert bool(sk["eligible"][i, col]) == bool(cached["flagged"][j])
            assert abs(sk["scores"][i, col] - cached["scores"][j]) <= 1e-10
        for targets in [[k] for k in LABELS] + [LABELS]:
            obs = {
                "target_label_ids": targets,
                "analysis_mode": "adaptive",
                "available_actions": ["run_full_pipeline", "run_skeleton", "STOP"],
                "available_profile_ids": ["full-v15", "body-qualified-v1"],
                "executed_actions": [],
                "failed_actions": [],
                "remaining_budget_seconds": 900,
                "decoded_frames": jobs[str(key)]["expected_frames"],
            }
            chosen = algorithm.choose(obs, params)
            assert chosen["action"] != "STOP"
            profile = chosen["profile_id"]
            assert set(targets) <= set(algorithm.COVERAGE[profile])
            row = chosen["audit"]["selected"]
            full_cost = sum(
                algorithm.estimate(tool, obs["decoded_frames"], params["cost_model"])[0]
                for tool in algorithm.PATHS["full-v15"]
            )
            baseline_cost += full_cost
            selected_cost += row["expected_seconds"]
            baseline_calls += 4
            selected_calls += len(row["missing_tools"])
            task_rows.append(
                {
                    "record_id": str(key),
                    "targets": targets,
                    "route": profile,
                    "missing_tools": row["missing_tools"],
                }
            )
    expanded = algorithm.choose(
        {
            "target_label_ids": LABELS,
            "analysis_mode": "adaptive",
            "available_actions": ["run_full_pipeline", "STOP"],
            "available_profile_ids": ["full-v15"],
            "executed_actions": ["pose.extract", "skeleton.predict", "fusion.predict"],
            "covered_label_ids": BODY,
            "failed_actions": [],
            "remaining_budget_seconds": 900,
            "decoded_frames": 375,
        },
        params,
    )
    assert expanded["audit"]["selected"]["missing_tools"] == ["rgb.predict", "fusion.predict"]
    report = {
        "passed": True,
        "paired_public_records_exact": len(equivalence),
        "raw_video_cache_records_exact": len(ids),
        "arbitrary_branch_property_cases": 200,
        "scope_equivalence": "Same scores/flags for all four certified body classes; all other goals use unchanged full M1/M2/M3. No inferred negatives for skipped RGB classes.",
        "task_mix": "Predeclared 11 single-class requests plus one all11 request for each raw26; counts are hypothetical requests, not new videos.",
        "tasks": len(task_rows),
        "fixed_full_tool_calls": baseline_calls,
        "policy_tool_calls": selected_calls,
        "estimated_seconds_fixed": baseline_cost,
        "estimated_seconds_policy": selected_cost,
        "timing_claim": "Cost replay only, using six historical actual traces; not new measured GPU execution latency. Live routed/expansion tests remain required.",
        "expansion_missing_tools": expanded["audit"]["selected"]["missing_tools"],
        "rows": task_rows,
    }
    (OUT / "offline-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "protocol_sha256": sha(OUT / "protocol.json"),
                "parameters_sha256": sha(OUT / "parameters.json"),
                "live_acceptance_required": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k not in ["rows", "scope_equivalence", "task_mix"]}))


if __name__ == "__main__":
    main()
