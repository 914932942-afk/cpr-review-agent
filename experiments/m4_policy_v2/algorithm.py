"""Consent-bounded maximum-coverage DAG scheduling, with prefix re-evaluation.

Coverage means available supported categories, never classifier correctness.
Only the same two previously certified routes are admissible. No new model/data.
"""

import math

VERSION = "consent-coverage-dag-v2"
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
    budget = observation["remaining_budget_seconds"]
    allow_partial = observation.get("allow_partial", False)
    if not targets or not targets <= set(LABELS):
        raise ValueError("M4_INVALID_TARGETS")
    if not math.isfinite(budget) or budget < 0 or type(allow_partial) is not bool:
        raise ValueError("M4_INVALID_BUDGET_OR_CONSENT")
    available = set(observation["available_actions"])
    legal = set(observation["available_profile_ids"])
    covered = set(observation.get("covered_label_ids", []))
    cached = set(observation.get("executed_actions", []))
    failed = set(observation.get("failed_actions", []))
    committed = observation.get("committed_partial_route")
    if committed not in (None, "body-qualified-v1") or (committed and not allow_partial):
        raise ValueError("M4_INVALID_COMMITTED_SCOPE")
    audit = {
        "algorithm": VERSION,
        "requested_labels": sorted(targets),
        "covered_labels": sorted(covered),
        "allow_partial": allow_partial,
        "committed_partial_route": committed,
        "gain_semantics": "New requested category coverage, not accuracy or confidence gain",
        "selection_order": "Maximum requested coverage, then minimum marginal expected cost",
        "remaining_budget_seconds": budget,
        "candidates": [],
    }

    def stop(code, message):
        return {"action": "STOP", "profile_id": None, "reason": message, "stop_code": code, "audit": audit}

    if observation["analysis_mode"] == "existing_only":
        return stop("EXISTING_ONLY", "Use only the selected saved evidence; do not start vision.")
    if targets <= covered:
        states = (observation.get("assessment") or {}).get("states", {})
        uncertain = any(states.get(str(k), states.get(k)) == "uncertain" for k in targets)
        return stop(
            "NO_INDEPENDENT_INFORMATION" if uncertain else "GOAL_COVERED",
            "The requested scope is covered. Do not repeat the same evidence to manufacture certainty.",
        )
    if observation.get("partial_result_delivered"):
        return stop(
            "PARTIAL_SCOPE_DELIVERED",
            "The explicitly permitted partial review is ready. List unanalysed targets and offer a separate expansion.",
        )
    # A successful cached node is useful only when its ancestors also exist.
    # Runtime additionally validates every identity/hash and actual dependency DAG.
    if ("skeleton.predict" in cached or "rgb.predict" in cached) and "pose.extract" not in cached:
        raise ValueError("M4_INVALID_CACHE_PREFIX")
    if (cached & failed) - {"fusion.predict"}:
        raise ValueError("M4_FAILED_CACHE_CONFLICT")
    for profile, path in PATHS.items():
        supplied = targets & set(COVERAGE[profile])
        gain = len(supplied - covered)
        if (
            profile not in legal
            or ACTIONS[profile] not in available
            or not gain
            or (committed and committed != profile)
            or (not allow_partial and not targets <= set(COVERAGE[profile]))
        ):
            continue
        missing = [s for s in path if s not in cached or s == "fusion.predict"]
        if failed & set(missing):
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
        row = {
            "profile_id": profile,
            "missing_tools": missing,
            "expected_seconds": expected,
            "upper_seconds": upper,
            "new_coverage_gain": gain,
            "covered_requested_labels": sorted(supplied | (targets & covered)),
            "uncovered_requested_labels": sorted(targets - covered - supplied),
            "partial_scope": not targets <= set(COVERAGE[profile]) | covered,
            "budget_feasible": upper + parameters["closing_reserve_seconds"] <= budget,
            "coverage_certificate": parameters["route_certificates"][profile],
        }
        audit["candidates"].append(row)
    feasible = [r for r in audit["candidates"] if r["budget_feasible"]]
    if not feasible:
        return stop(
            "TOOL_FAILED"
            if failed
            else "M4_BUDGET_INSUFFICIENT"
            if audit["candidates"]
            else "NO_SUPPORTED_ROUTE",
            "No approved route can add requested coverage within this budget; no failed stage is retried.",
        )
    winner = max(
        feasible,
        key=lambda r: (
            r["new_coverage_gain"],
            -r["expected_seconds"],
            -r["upper_seconds"],
            -len(r["missing_tools"]),
        ),
    )
    audit["selected"] = winner
    return {
        "action": ACTIONS[winner["profile_id"]],
        "profile_id": winner["profile_id"],
        "reason": "Maximise requested coverage under the conservative budget and approved dependencies; prefer the cheaper equivalent route. Partial scope requires explicit user consent and is never widened within this run.",
        "stop_code": None,
        "audit": audit,
    }
