"""M4 observation construction and dependency-safe plans; numerical policy is separate."""

from .modules.contracts import Assessment, PolicyObservation
from .vision import final_tool, media_context, validate_receipt

PATHS = {
    "full-v15": ["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"],
    "body-qualified-v1": ["pose.extract", "skeleton.predict", "fusion.predict"],
}


def observe(store, settings, goal, receipts, failed, assessment, run, remaining):
    covered = []
    frames = None
    if final_tool(settings) in receipts:
        pred = validate_receipt(store.get_tool(receipts[final_tool(settings)])["output"], store, settings)
        covered = pred.supported_label_ids
        frames = pred.decoded_frames
    else:
        _, asset, _ = media_context(store, settings, run["clip_id"])
        for candidate in [
            asset.get("inspection"),
            asset.get("media_probe"),
            asset.get("preflight"),
            asset.get("validation"),
            asset,
        ]:
            if isinstance(candidate, dict) and type(candidate.get("decoded_frames")) is int:
                frames = candidate["decoded_frames"]
                break
    available = ["STOP"]
    if goal.analysis_mode != "existing_only" and not set(goal.target_label_ids) <= set(covered):
        available = ["run_full_pipeline", "run_skeleton", "STOP"]
    observed = {
        tool: store.get_tool(key)["duration_ms"] / 1000
        for tool, key in receipts.items()
        if store.get_tool(key)["run_id"] == run["id"] and store.get_tool(key).get("duration_ms") is not None
    }
    turn = store.get("turn", run["turn_id"])
    delivered = bool(
        goal.allow_partial
        and final_tool(settings) in receipts
        and store.get_tool(receipts[final_tool(settings)])["run_id"] == run["id"]
        and not set(goal.target_label_ids) <= set(covered)
    )
    return PolicyObservation(
        allow_partial=goal.allow_partial,
        partial_result_delivered=delivered,
        committed_partial_route=turn.get("m4_committed_partial_route"),
        target_label_ids=goal.target_label_ids,
        analysis_mode=goal.analysis_mode,
        assessment=Assessment.model_validate(assessment) if assessment else None,
        available_actions=available,
        available_profile_ids=list(PATHS) if len(available) > 1 else [],
        executed_actions=list(receipts),
        failed_actions=list(failed),
        remaining_budget_seconds=remaining,
        covered_label_ids=covered,
        decoded_frames=frames,
        observed_tool_seconds=observed,
    )


def missing_tools(profile, receipts, covered, targets, allow_partial=False):
    if profile not in PATHS:
        raise ValueError("M4_UNKNOWN_ROUTE")
    if (
        profile == "body-qualified-v1"
        and not set(targets) <= set([3, 4, 8, 11])
        and not (allow_partial and set(targets) & {3, 4, 8, 11})
    ):
        raise ValueError("M4_ROUTE_CANNOT_COVER_GOAL")
    return [
        tool
        for tool in PATHS[profile]
        if tool not in receipts or (tool == "fusion.predict" and not set(targets) <= set(covered))
    ]
