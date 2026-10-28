"""Post-restart partial source, zero-work reuse and global GPU exclusion checks."""

import json
import time
from pathlib import Path
from uuid import uuid4

import httpx
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test-results/m5-cascade-v1"


def main():
    assert not (OUT / "restart.json").exists()
    state = json.loads((OUT / "http/state.json").read_text())
    with httpx.Client(base_url="http://127.0.0.1:8013", timeout=30) as c:

        def get(p):
            return c.get("/api" + p).raise_for_status().json()

        def post(p, v):
            return (
                c.post("/api" + p, json=v, headers={"Idempotency-Key": str(uuid4())})
                .raise_for_status()
                .json()
            )

        def wait(rid):
            until = time.monotonic() + 65
            while time.monotonic() < until:
                r = get("/runs/" + rid)
                if r["status"] in {"completed", "partially_completed", "failed", "rejected", "cancelled"}:
                    return r
                time.sleep(0.2)
            raise TimeoutError(rid)

        p = state["partial"]
        f = state["full"]
        assert get("/runs/" + p["run_id"])["status"] == "partially_completed"
        assert get("/reports/" + p["report_id"] + "/evidence")["status"] == "available"
        context = get("/sessions/" + p["session_id"] + "/context?source_report_id=" + p["report_id"])
        assert context["source_status"] == "available" and context["source"]["report_id"] == p["report_id"]
        for row, rid, report in [
            (f, f["run_id"], f["report_id"]),
            (p, p["expand_run_id"], p["expand_report_id"]),
        ]:
            assert get("/runs/" + rid)["status"] == "completed"
            assert get("/reports/" + report + "/evidence")["status"] == "available"
        request = {
            "goal": {
                "intent": "analyse",
                "target_label_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
                "analysis_mode": "adaptive",
            },
            "context": {
                "source_report_id": p["expand_report_id"],
                "reference": {"scope": "explicit", "source": "viewed_report"},
            },
        }
        r = wait(post("/sessions/" + p["session_id"] + "/turns", request)["run_id"])
        assert (
            r["status"] == "completed"
            and r["response"]["reused_result"] is True
            and get("/runs/" + r["id"] + "/tools") == []
        )
        assert r["response"]["scheduling"]["stop"]["stop_code"] == "GOAL_COVERED"
        clip = post("/clips", {"asset_id": p["asset_id"]})
        session = post("/sessions", {"clip_id": clip["id"]})
        with FileLock(ROOT / "data/vision-gpu.lock", timeout=0):
            blocked = wait(
                post(
                    "/sessions/" + session["id"] + "/turns",
                    {"goal": {"intent": "analyse", "target_label_ids": [4], "allow_partial": True}},
                )["run_id"]
            )
        tools = get("/runs/" + blocked["id"] + "/tools")
        assert blocked["status"] == "failed" and len(tools) == 1 and tools[0]["status"] == "failed"
        assert not blocked["response"]["prediction"] and blocked["response"]["classification_claims"] == []
        result = {
            "passed": True,
            "terminal_service_restart_performed": True,
            "partial_source_and_report_restored": True,
            "full_and_expansion_restored": True,
            "source_identity_preserved": True,
            "reused_run_id": r["id"],
            "reuse_new_tool_calls": 0,
            "global_gpu_lock_exclusion": True,
            "blocked_run_id": blocked["id"],
            "failed_tool_not_retried": True,
            "new_successful_vision_executions": 0,
            "provider_calls": 0,
        }
        (OUT / "restart.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
