"""Controller-only approved dependency reuse. Cache identity never comes from user text."""

import json


def enabled(settings):
    if settings.vision_pipeline != "branches":
        return False
    return (
        json.loads(settings.branch_manifest.read_text(encoding="utf-8")).get("module_slots", {}).get("M4")
        == "goal-cost-dag-v1"
    )


def closure(store, final_id):
    output = {}
    pending = [final_id]
    visited = set()
    while pending:
        key = pending.pop()
        if key in visited:
            continue
        visited.add(key)
        if len(visited) > 4:
            raise ValueError("M4_LINEAGE_NOT_A_STAGE_DAG")
        source = store.get_tool(key)
        if source["status"] != "succeeded":
            raise ValueError("M4_CACHE_NOT_SUCCEEDED")
        tool = source["request"]["tool_id"]
        if tool in output and output[tool] != key:
            raise ValueError("M4_CACHE_AMBIGUOUS_STAGE")
        output[tool] = key
        for name in ["pose_receipt_id", "skeleton_receipt_id", "rgb_receipt_id"]:
            if name in source["request"]["input_bindings"]:
                pending.append(source["request"]["input_bindings"][name])
    return output


def reusable(store, settings, request, source):
    if source["run_id"] == request["run_id"]:
        return True
    anchor = request["input_bindings"].get("reuse_source_execution_id")
    if not enabled(settings) or not anchor:
        return False
    run = store.get_run(request["run_id"])
    turn = store.get("turn", run["turn_id"])
    bound = (turn.get("context_snapshot") or {}).get("source") or {}
    if (
        bound.get("execution_id") != anchor
        or bound.get("video_sha256") != request["input_bindings"].get("video_sha256")
        or run["clip_id"] != source["request"]["clip_id"]
    ):
        return False
    from .vision import validate_receipt

    previous = store.get_tool(anchor)
    verified = validate_receipt(previous["output"], store, settings)
    return verified.clip_id == run["clip_id"] and source["id"] in closure(store, anchor).values()


def hydrate(store, settings, previous):
    from .branch_vision import validate_branch

    result = closure(store, previous.execution_id)
    for tool, key in result.items():
        if tool == "fusion.predict":
            continue
        receipt = validate_branch(store.get_tool(key)["output"], store, settings)
        if (
            receipt.clip_id != previous.clip_id
            or receipt.profile_id != previous.profile_id
            or receipt.video_sha256 != previous.video_sha256
        ):
            raise ValueError("M4_CACHE_SOURCE_MISMATCH")
    return result
