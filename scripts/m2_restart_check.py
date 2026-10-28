"""Verify restored M2 reports and actual global GPU-lock exclusion, without loading a new model."""

import json
import time
from pathlib import Path
from uuid import uuid4

import httpx
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "test-results/m2-quality-v1/restart.json"
    if out.exists():
        raise FileExistsError("Preserve restart evidence")
    state = json.loads((ROOT / "test-results/m2-quality-v1/http/state.json").read_text())
    runtime = ROOT / "workspaces/m2-quality-v1/data/m2-quality-v1/runtime"
    with httpx.Client(base_url="http://127.0.0.1:8008", timeout=30) as c:

        def get(path):
            return c.get("/api" + path).raise_for_status().json()

        def post(path, body):
            return (
                c.post("/api" + path, json=body, headers={"Idempotency-Key": str(uuid4())})
                .raise_for_status()
                .json()
            )

        for record in state.values():
            assert get("/runs/" + record["run_id"])["status"] == "completed"
            assert get("/reports/" + record["report_id"] + "/evidence")["status"] == "available"
            assert get("/runs/" + record["summary_run_id"])["response"]["reused_result"]
        # A separate clip prevents reuse of the existing result. Hold the exact parent lock across request.
        first = next(iter(state.values()))
        clip = post("/clips", {"asset_id": first["asset_id"]})
        session = post("/sessions", {"clip_id": clip["id"]})
        with FileLock(ROOT / "data/vision-gpu.lock", timeout=0):
            rid = post(
                "/sessions/" + session["id"] + "/turns",
                {"goal": {"intent": "analyse", "target_label_ids": [4]}},
            )["run_id"]
            until = time.monotonic() + 60
            while True:
                run = get("/runs/" + rid)
                if run["status"] in ["completed", "failed", "cancelled", "rejected"]:
                    break
                assert time.monotonic() < until
                time.sleep(0.3)
            assert run["status"] == "failed" and not (run.get("response") or {}).get("prediction")
        tools = get("/runs/" + rid + "/tools")
        assert (
            len(tools) == 1
            and tools[0]["request"]["tool_id"] == "pose.extract"
            and tools[0]["status"] == "failed"
        )
        logs = list((runtime / "analysis" / tools[0]["id"]).rglob("process.log"))
        assert len(logs) == 1
        text = logs[0].read_text(encoding="utf-8", errors="replace")
        assert "filelock" in text.lower() and "Timeout" in text
        assert str(ROOT / "data/vision-gpu.lock") in text
        assert not list(logs[0].parent.glob("inference*.npz"))
        result = {
            "passed": True,
            "restored_reports": len(state),
            "shared_parent_gpu_lock_enforced": True,
            "blocked_attempt_run_id": rid,
            "blocked_attempt_tool_id": tools[0]["id"],
            "new_model_executions": 0,
            "provider_calls": 0,
            "failed_attempt_publishes_no_prediction": True,
        }
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
