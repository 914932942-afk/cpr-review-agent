"""Goal-constrained dependency-DAG planning with conservative empirical costs.

Pure inference: expected gain is newly covered requested categories, not a promised
accuracy gain. A completed uncertain result does not justify repeating the same DAG.
"""

import math

VERSION = "goal-cost-dag-v1"
BODY_LABELS = (3, 4, 8, 11)
LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13)
PATHS = {
    "full-v15": ("pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"),
    "body-qualified-v1": ("pose.extract", "skeleton.predict", "fusion.predict"),
}
COVERAGE = {"full-v15": LABELS, "body-qualified-v1": BODY_LABELS}
ACTIONS = {"full-v15": "run_full_pipeline", "body-qualified-v1": "run_skeleton"}


def estimate(stage, frames, cost_model, observed=None):
    cfg = cost_model[stage]
    scale = max(1.0, (frames or 900) / cfg["reference_frames"]) if stage == "pose.extract" else 1.0
    load = 1.0
    if observed and "pose.extract" in observed and stage != "pose.extract":
        load = max(
            1.0, min(3.0, observed["pose.extract"] / max(cost_model["pose.extract"]["median_seconds"], 0.1))
        )
    return cfg["median_seconds"] * scale * load, cfg["upper_seconds"] * scale * load


def choose(observation, parameters):
    targets = set(observation["target_label_ids"])
    if not targets or not targets <= set(LABELS):
        raise ValueError("M4_INVALID_TARGETS")
    available = set(observation["available_actions"])
    legal_profiles = set(observation["available_profile_ids"])
    budget = observation["remaining_budget_seconds"]
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("M4_INVALID_BUDGET")
    covered = set(observation.get("covered_label_ids", []))
    assessment = observation.get("assessment") or {}
    audit = {
        "algorithm": VERSION,
        "requested_labels": sorted(targets),
        "covered_labels": sorted(covered),
        "gain_semantics": "Newly covered requested categories; not predicted accuracy improvement",
        "candidates": [],
    }

    def stop(code, message):
        return {"action": "STOP", "profile_id": None, "reason": message, "stop_code": code, "audit": audit}

    if observation["analysis_mode"] == "existing_only":
        return stop("EXISTING_ONLY", "Use only the bound saved evidence; do not start vision.")
    if targets <= covered:
        states = assessment.get("states", {})
        unresolved = any(states.get(str(k), states.get(k)) == "uncertain" for k in targets)
        return stop(
            "NO_INDEPENDENT_INFORMATION" if unresolved else "GOAL_COVERED",
            "The requested scope is already covered. Existing tools cannot add independent evidence; retain any uncertain states.",
        )
    if observation.get("failed_actions"):
        return stop("TOOL_FAILED", "A required tool failed; do not silently repeat it.")
    cached = set(observation.get("executed_actions", []))
    candidates = []
    for profile, path in PATHS.items():
        if (
            profile not in legal_profiles
            or ACTIONS[profile] not in available
            or not targets <= set(COVERAGE[profile])
        ):
            continue
        # An existing final from a narrower route cannot satisfy a newly expanded scope.
        missing = [stage for stage in path if stage not in cached or stage == "fusion.predict"]
        if not missing:
            continue
        expected = upper = 0.0
        for stage in missing:
            mid, hi = estimate(
                stage,
                observation.get("decoded_frames"),
                parameters["cost_model"],
                observation.get("observed_tool_seconds"),
            )
            expected += mid
            upper += hi
        gain = len(targets - covered)
        row = {
            "profile_id": profile,
            "missing_tools": missing,
            "expected_seconds": expected,
            "upper_seconds": upper,
            "new_coverage_gain": gain,
            "gain_per_expected_second": gain / max(expected, 0.01),
            "budget_feasible": upper + parameters["closing_reserve_seconds"] <= budget,
            "coverage_certificate": parameters["route_certificates"][profile],
        }
        audit["candidates"].append(row)
        if row["budget_feasible"]:
            candidates.append(row)
    if not candidates:
        return stop(
            "M4_BUDGET_INSUFFICIENT" if audit["candidates"] else "NO_SUPPORTED_ROUTE",
            "No verified route can cover this request within the remaining conservative budget.",
        )
    winner = max(
        candidates,
        key=lambda r: (r["gain_per_expected_second"], -r["upper_seconds"], -len(r["missing_tools"])),
    )
    audit["selected"] = winner
    return {
        "action": ACTIONS[winner["profile_id"]],
        "profile_id": winner["profile_id"],
        "stop_code": None,
        "reason": "Choose the verified dependency closure with the highest requested-coverage gain per estimated second; respect the conservative cost bound and reuse only approved cached stages.",
        "audit": audit,
    }
