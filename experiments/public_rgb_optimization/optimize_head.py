"""Bounded development-only improvement of the frozen-feature CPR head.

The preceding test is already exposed. No independent blind-test claim is
made. Recordings, gates and feature extraction stay fixed; fitting/selection
may only decode the development feature bundle.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "experiments/public_rgb_protocol"))
import rgb_protocol as parent
from quality_head import PortableHead, export_classifier, load

DATA = CODE / "data/public_cpr"
BASE = DATA / "rgb-evaluation-v1"
OUTPUT = DATA / "rgb-evaluation-v2"
REPORT = CODE / "test-results/public-rgb-optimization-v2"
SEED = 20261022
CONFIGS = [
    {"name": "control64", "branches": ["combined"], "trees": 64, "leaves": 7, "depth": 3, "minimum_leaf": 8, "l2": 1., "rate": .1, "balanced": False},
    {"name": "regularized64", "branches": ["combined"], "trees": 64, "leaves": 7, "depth": 3, "minimum_leaf": 8, "l2": 3., "rate": .1, "balanced": False},
    {"name": "smooth64", "branches": ["combined"], "trees": 64, "leaves": 7, "depth": 3, "minimum_leaf": 12, "l2": 1., "rate": .1, "balanced": False},
    {"name": "shallow88", "branches": ["combined"], "trees": 88, "leaves": 5, "depth": 3, "minimum_leaf": 8, "l2": 3., "rate": .08, "balanced": False},
    {"name": "deeper48", "branches": ["combined"], "trees": 48, "leaves": 9, "depth": 4, "minimum_leaf": 8, "l2": 3., "rate": .1, "balanced": False},
    {"name": "balanced64", "branches": ["combined"], "trees": 64, "leaves": 7, "depth": 3, "minimum_leaf": 8, "l2": 1., "rate": .1, "balanced": True},
    {"name": "pose_geometry32", "branches": ["pose", "geometry"], "trees": 32, "leaves": 7, "depth": 3, "minimum_leaf": 8, "l2": 1., "rate": .1, "balanced": False},
    {"name": "combined_geometry32", "branches": ["combined", "geometry"], "trees": 32, "leaves": 7, "depth": 3, "minimum_leaf": 8, "l2": 1., "rate": .1, "balanced": False},
]
METHODS = ["legacy_validation", "bootstrap_balanced", "bootstrap_balanced_shrink25"]


def write(path, value):
    parent.write_new(path, value)


def prepare(output):
    if output.exists():
        raise FileExistsError("Preserve previous attempts; use a new output directory")
    original = parent.checked_protocol(BASE)
    protected = dict(original["protected_sources"])
    for name in ("protocol.json", "selection.json", "evaluation.json", "inference-manifest.json", "record-splits.json", "development.npz", "locked-test.npz", "combined-64.pt"):
        path = BASE / name
        protected[path.relative_to(CODE).as_posix()] = parent.digest(path)
    output.mkdir(parents=True)
    for name in ("record-splits.json", "development.npz", "locked-test.npz"):
        shutil.copyfile(BASE / name, output / name)
    protocol = {**original, "purpose": "Small-head optimization using development only; unchanged recording reservations and gates",
        "frozen_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "parent_protocol": (BASE / "protocol.json").relative_to(CODE).as_posix(),
        "parent_protocol_sha256": parent.digest(BASE / "protocol.json"), "candidate_configs": CONFIGS,
        "threshold_methods": METHODS, "bootstrap_iterations": 21,
        "bootstrap_strata": "Validation cohort and public supported-label pattern; replacement within each stratum",
        "selection_rule": "Eligible: validation single>=.85, composite>=.80, normal false alarms=0, min overall label recall>=.60, head<=70000. Rank by min(single,composite) F1, then mean F1, then smaller head.",
        "test_observed_before_optimization": True,
        "previous_test_exposure": "v1 test metrics and reserved-clip errors were inspected before this development-only optimization. Same test is a disclosed engineering regression, not independent blind evidence.",
        "independent_unseen_test_claim": False, "protected_sources": protected,
        "implementation_sources": {**original["implementation_sources"], Path(__file__).resolve().relative_to(CODE).as_posix(): parent.digest(Path(__file__).resolve())},
        "candidate_policy": "Eight declared lightweight candidates; no test-driven candidate or threshold choice",
        "threshold_scope": "Validation only; stable methods balance single/composite F1 per class, bootstrap, optionally shrink thresholds toward 0.5, and enforce validation Correct guard",
        "development_search_disclosure": "Bounded eight-candidate/three-threshold-method search frozen before fits; no test decoding before selection"}
    write(output / "protocol.json", protocol)
    print("Frozen", len(CONFIGS), "candidates; split/gates unchanged", protocol["split_counts"], flush=True)


def checked(output):
    return parent.checked_protocol(output)


def feature_mask(branch):
    if branch == "pose":
        return np.arange(512)
    if branch == "geometry":
        return np.arange(512, 783)
    if branch == "combined":
        return np.arange(783)
    raise ValueError("Unknown branch")


def merge_branch(output, branch, mask, weight):
    output["bias"] += weight * branch["bias"]
    for tree in branch["trees"]:
        copied = {name: value.clone() for name, value in tree.items()}
        copied["feature"] = torch.from_numpy(mask[copied["feature"].numpy()].astype(np.int32))
        copied["value"] *= weight
        output["trees"].append(copied)


def best_balanced_thresholds(probability, truth, cohorts):
    normal = ~truth.any(1)
    grid = np.linspace(.1, .95, 35)
    result = []
    for label in range(11):
        predicted = probability[:, label, None] >= grid[None]
        utility = np.zeros(len(grid))
        for cohort in ("single", "composite"):
            selected = cohorts == cohort
            target = truth[selected, label, None].astype(bool)
            proposed = predicted[selected]
            tp = (proposed & target).sum(0)
            fp = (proposed & ~target).sum(0)
            fn = (~proposed & target).sum(0)
            utility += .5 * 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        valid = ~predicted[normal].any(0)
        choices = [(float(utility[i]), -abs(float(value) - .5), float(value)) for i, value in enumerate(grid) if valid[i]]
        result.append(max(choices)[2] if choices else 1.)
    return np.asarray(result)


def stable_thresholds(probability, truth, cohorts, iterations=21):
    strata = defaultdict(list)
    for i, (target, cohort) in enumerate(zip(truth, cohorts)):
        strata[(str(cohort), tuple(np.flatnonzero(target)))].append(i)
    rng = np.random.default_rng(SEED)
    bootstrapped = []
    for _ in range(iterations):
        selected = np.concatenate([rng.choice(rows, len(rows), replace=True) for _, rows in sorted(strata.items())])
        bootstrapped.append(best_balanced_thresholds(probability[selected], truth[selected], cohorts[selected]))
    distribution = np.stack(bootstrapped)
    return np.median(distribution, axis=0), distribution.std(0)


def summarize(probability, truth, cohorts, thresholds, normal_gate):
    predicted = parent.predict(probability, thresholds, normal_gate)
    return {name: parent.metrics(predicted[mask].astype(float), truth[mask]) for name, mask in
        (("all", np.ones(len(truth), bool)), ("single", cohorts == "single"), ("composite", cohorts == "composite"))}


def options(probability, truth, cohorts):
    legacy = parent.fit_quality_head.tune_operating_point(probability, truth)
    median, deviation = stable_thresholds(probability, truth, cohorts)
    choices = []
    for method in METHODS:
        if method == "legacy_validation":
            thresholds, gate = legacy["thresholds"], legacy["normal_gate"]
        else:
            shrink = .25 if method.endswith("shrink25") else 0.
            thresholds = median * (1 - shrink) + .5 * shrink
            normal = ~truth.any(1)
            if normal.any():
                thresholds = np.maximum(thresholds, np.nextafter(probability[normal, :11].max(0), np.inf))
            thresholds, gate = thresholds.tolist(), None
        measured = summarize(probability, truth, cohorts, thresholds, gate)
        single, composite = measured["single"]["macro_f1"], measured["composite"]["macro_f1"]
        eligible = (single >= .85 and composite >= .8 and measured["all"]["correct_false_alarm_rate"] == 0
                    and min(row["recall"] for row in measured["all"]["per_label"]) >= .6)
        choices.append({"method": method, "thresholds": thresholds, "normal_gate": gate,
            "validation_metrics": measured["all"], "validation_cohorts": measured,
            "bootstrap_threshold_std": deviation.tolist() if method != "legacy_validation" else None,
            "eligible": eligible, "validation_rank": [min(single, composite), (single + composite) / 2]})
    return choices


def fit_candidate(config, xfit, target, cohorts, xval):
    outputs = [{"bias": 0., "trees": []} for _ in range(12)]
    reference = np.zeros((len(xval), 12))
    weights = None
    if config["balanced"]:
        weights = np.ones(len(xfit))
        for name in np.unique(cohorts):
            mask = cohorts == name
            weights[mask] = len(xfit) / (len(np.unique(cohorts)) * mask.sum())
    with threadpool_limits(limits=4):
        for branch in config["branches"]:
            mask = feature_mask(branch)
            weight = 1. / len(config["branches"])
            for label in range(12):
                learner = HistGradientBoostingClassifier(max_iter=config["trees"], max_leaf_nodes=config["leaves"],
                    max_depth=config["depth"], min_samples_leaf=config["minimum_leaf"], l2_regularization=config["l2"],
                    learning_rate=config["rate"], early_stopping=False, random_state=SEED)
                learner.fit(xfit[:, mask], target[:, label], sample_weight=weights)
                reference[:, label] += weight * learner.decision_function(xval[:, mask])
                merge_branch(outputs[label], export_classifier(learner), mask, weight)
    expected = 1 / (1 + np.exp(-np.clip(reference, -60, 60)))
    return outputs, expected


def train(output):
    protocol = checked(output)
    if (output / "selection.json").exists() or list(output.glob("*.pt")):
        raise FileExistsError("Never overwrite an existing fitting attempt")
    with np.load(output / "development.npz", allow_pickle=False) as bundle:
        x, y, splits, cohorts, ids = [bundle[k].copy() for k in ("features", "targets", "splits", "cohorts", "record_ids")]
        if str(bundle["profile_hash"]) != protocol["feature_profile_hash"] or not np.array_equal(bundle["label_ids"], parent.LABEL_IDS):
            raise ValueError("Feature/class contract differs")
    reserved = {parent.group_id(k) for k in protocol["raw_test_reservations"]}
    if {parent.group_id(str(k)) for k in ids} & reserved:
        raise ValueError("Reserved recording leaked into development")
    fitting, validation = splits == "train", splits == "validation"
    target = np.column_stack([y[fitting], ~y[fitting].any(1)])
    candidates = []
    for config in protocol["candidate_configs"]:
        start = time.perf_counter()
        outputs, reference = fit_candidate(config, x[fitting], target, cohorts[fitting], x[validation])
        fields = 12 + sum(t.numel() for label in outputs for tree in label["trees"] for t in tree.values())
        if fields > protocol["gates"]["maximum_head_parameters"]:
            candidates.append({"config": config, "eligible": False, "scalar_fields": fields, "reason": "head budget"})
            continue
        state = {"schema": "public-cpr-geometry-head-v1", "input_dimension": 783,
            "feature_indices": torch.arange(783), "classifiers": outputs, "label_ids": list(parent.LABEL_IDS),
            "source_revision": protocol["source_revision"], "feature_profile_hash": protocol["feature_profile_hash"],
            "protocol_sha256": parent.digest(output / "protocol.json"),
            "score_semantics": "Uncalibrated small-head scores; validation-only stable thresholds"}
        probability = PortableHead(state).probability(x[validation])
        difference = float(np.abs(probability - reference).max())
        if difference > 1e-10:
            raise AssertionError("Branch export differs from fitted predictor")
        threshold_options = options(probability, y[validation], cohorts[validation])
        eligible_options = [option for option in threshold_options if option["eligible"]]
        chosen = max(eligible_options or threshold_options, key=lambda row: tuple(row["validation_rank"]))
        state["operating_point"] = {key: chosen[key] for key in ("thresholds", "normal_gate", "validation_metrics")}
        state["decision_method"] = chosen["method"]
        path = output / (config["name"] + ".pt")
        torch.save(state, path)
        restored, _ = load(path, protocol["feature_profile_hash"], parent.digest(output / "protocol.json"))
        if not np.array_equal(restored.probability(x[validation]), probability):
            raise AssertionError("Checkpoint reload differs")
        row = {"config": config, "eligible": bool(eligible_options), "scalar_fields": fields,
            "checkpoint": path.name, "checkpoint_sha256": parent.digest(path), "checkpoint_bytes": path.stat().st_size,
            "seconds": time.perf_counter() - start, "portable_vs_training_max_difference": difference,
            "operating_point": state["operating_point"], "threshold_method": chosen["method"],
            "validation_cohorts": chosen["validation_cohorts"], "validation_rank": chosen["validation_rank"],
            "threshold_options": threshold_options}
        candidates.append(row)
        print(config["name"], "validation single/composite", row["validation_cohorts"]["single"]["macro_f1"],
              row["validation_cohorts"]["composite"]["macro_f1"], "method", chosen["method"], "fields", fields, flush=True)
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        write(output / "development-failure.json", {"all_candidates": candidates, "test_decoded": False})
        raise RuntimeError("No candidate passed development gates; test remains unscored")
    chosen = max(eligible, key=lambda row: (*row["validation_rank"], -row["scalar_fields"]))
    write(output / "selection.json", {"selected": chosen, "all_candidates": candidates,
        "protocol_sha256": parent.digest(output / "protocol.json"), "test_read_during_fit_or_selection": False,
        "main_network_training": False, "fitted_records": int(fitting.sum()), "tuning_records": int(validation.sum())})
    write(output / "fit-access-audit.json", {"passed": True, "npz_files_decoded_by_fitter": [(output / "development.npz").relative_to(CODE).as_posix()],
        "test_npz_decoded_by_fitter": False, "locked_test_integrity_hashed_before_fit": True,
        "reserved_rgb_in_fit_or_tuning": 0, "test_based_candidate_selection": False})
    parent.check_hashes(protocol["protected_sources"])
    print("Frozen validation winner:", chosen["config"]["name"], chosen["threshold_method"], flush=True)


def evaluate(output):
    checked(output)
    parent.evaluate(output)


def export_manifest(output):
    checked(output)
    parent.export_manifest(output)
    # Parent exporter binds head, protocol, source labels and all 26 videos.
    # The new protocol additionally binds this optimizer's implementation.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "evaluate", "export"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate, "export": export_manifest}[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
