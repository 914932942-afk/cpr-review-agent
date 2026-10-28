"""Restart, approved cross-run lineage reuse, and actual global GPU exclusion for M4."""

import json
import time
from pathlib import Path
from uuid import uuid4

import httpx
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test-results/m4-scheduling-v1/restart.json"


def main():
    if OUT.exists():
        raise FileExistsError("Preserve M4 restart evidence")
    state = json.loads((OUT.parent / "http/state.json").read_text())
    runtime = ROOT / "workspaces/m4-routing-v1/data/m4-scheduling-v1/runtime"
    with httpx.Client(base_url="http://127.0.0.1:8011", timeout=30) as c:

        def get(p):
            return c.get("/api" + p).raise_for_status().json()

        def post(p, v):
            return (
                c.post("/api" + p, json=v, headers={"Idempotency-Key": str(uuid4())})
                .raise_for_status()
                .json()
            )

        def wait(rid):
            until = time.monotonic() + 60
            while True:
                r = get("/runs/" + rid)
                if r["status"] in ["completed", "partially_completed", "failed", "rejected", "cancelled"]:
                    return r
                assert time.monotonic() < until
                time.sleep(0.2)

        for row in state.values():
            assert get("/runs/" + row["run_id"])["status"] == "completed"
            assert get("/reports/" + row["report_id"] + "/evidence")["status"] == "available"
        body = state["body"]
        expanded = get("/runs/" + body["expand_run_id"])
        assert expanded["response"]["prediction"]["analysis_route"] == "full-v15"
        assert get("/reports/" + body["expand_report_id"] + "/evidence")["status"] == "available"
        request = {
            "goal": {
                "intent": "analyse",
                "target_label_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
                "analysis_mode": "adaptive",
            },
            "context": {
                "source_report_id": body["expand_report_id"],
                "reference": {"scope": "explicit", "source": "viewed_report"},
            },
        }
        reused = wait(post("/sessions/" + body["session_id"] + "/turns", request)["run_id"])
        assert (
            reused["status"] == "completed"
            and reused["response"]["reused_result"]
            and get("/runs/" + reused["id"] + "/tools") == []
        )
        assert (
            reused["response"]["prediction"]["execution_id"]
            == expanded["response"]["prediction"]["execution_id"]
        )
        clip = post("/clips", {"asset_id": body["asset_id"]})
        session = post("/sessions", {"clip_id": clip["id"]})
        with FileLock(ROOT / "data/vision-gpu.lock", timeout=0):
            rid = post(
                "/sessions/" + session["id"] + "/turns",
                {"goal": {"intent": "analyse", "target_label_ids": [4]}},
            )["run_id"]
            blocked = wait(rid)
            assert blocked["status"] == "failed" and not (blocked.get("response") or {}).get("prediction")
        tools = get("/runs/" + rid + "/tools")
        assert (
            len(tools) == 1
            and tools[0]["request"]["tool_id"] == "pose.extract"
            and tools[0]["status"] == "failed"
        )
        logs = list((runtime / "analysis" / tools[0]["id"]).rglob("process.log"))
        assert len(logs) == 1
        log = logs[0].read_text(encoding="utf-8", errors="replace")
        assert "filelock" in log.lower() and "Timeout" in log and str(ROOT / "data/vision-gpu.lock") in log
        assert not list(logs[0].parent.glob("inference*.npz"))
        result = {
            "passed": True,
            "restored_body_and_expanded_lineage": True,
            "restored_full_report": True,
            "same_profile_completed_result_reused_without_tools": True,
            "reuse_run_id": reused["id"],
            "shared_gpu_lock_enforced": True,
            "blocked_attempt_run_id": rid,
            "new_model_executions": 0,
            "provider_calls": 0,
        }
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
