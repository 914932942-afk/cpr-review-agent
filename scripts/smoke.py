"""Exercise the running HTTP server, not TestClient. Exit 0 only for a connected 12-node path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://127.0.0.1:8000")
args = parser.parse_args()


def wait(client, rid, predicate):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        run = client.get(f"/api/runs/{rid}").raise_for_status().json()
        if predicate(run):
            return run
        time.sleep(0.1)
    raise AssertionError(f"Run timed out: {run}")


with httpx.Client(base_url=args.base_url, timeout=10) as client:
    assert client.get("/api/health").raise_for_status().json()["status"] == "ok"
    demo = client.post("/api/demo/session").raise_for_status().json()
    start = (
        client.post(
            f"/api/sessions/{demo['session']['id']}/turns",
            json={"message": "Review this video"},
            headers={"Idempotency-Key": str(uuid4())},
        )
        .raise_for_status()
        .json()
    )
    rid = start["run_id"]
    paused = wait(
        client, rid, lambda r: r["status"] == "waiting_user" and (r["wait"] or {}).get("interrupt_id")
    )
    client.post(
        f"/api/runs/{rid}/resume",
        headers={"Idempotency-Key": str(uuid4())},
        json={
            "kind": "user_answer",
            "wait_id": paused["wait"]["wait_id"],
            "interrupt_id": paused["wait"]["interrupt_id"],
            "revision": paused["revision"],
            "goal": {"intent": "analyse", "target_label_ids": [4, 5]},
        },
    ).raise_for_status()
    run = wait(client, rid, lambda r: r["status"] in {"completed", "failed", "cancelled", "rejected"})
    events = client.get(f"/api/runs/{rid}/events").raise_for_status().json()["events"]
    tools = client.get(f"/api/runs/{rid}/tools").raise_for_status().json()
    nodes = sorted({e["node_id"] for e in events if e["node_id"]})
    assert nodes == [f"N{i:02}" for i in range(1, 13)], nodes
    assert len(tools) == 4 and all(t["status"] == "succeeded" for t in tools)
    assert run["error_code"] == "MODULES_NOT_IMPLEMENTED"
    assert run["response"]["workflow_reached_end"] and not run["response"]["classification_claims"]
    result = {
        "passed": True,
        "run_id": rid,
        "nodes": nodes,
        "tool_count": len(tools),
        "workflow_reached_end": True,
        "business_status": run["status"],
        "error_code": run["error_code"],
    }
    output = Path(__file__).resolve().parents[1] / "test-results"
    output.mkdir(exist_ok=True)
    (output / "http-smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
