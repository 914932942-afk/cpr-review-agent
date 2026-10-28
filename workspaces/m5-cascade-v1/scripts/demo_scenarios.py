"""Repeatable local HTTP demo: real inference, reuse, invalid input, cancellation and restart."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from demo_preflight import GRAPH, PIPELINE, ROOT, SAMPLE, TOOLS, require, sha

TERMINAL = {"completed", "partially_completed", "failed", "rejected", "cancelled"}


def local_url(value):
    parsed = urlsplit(value)
    require(
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and parsed.port is not None
        and not parsed.username
        and not parsed.password
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment,
        "Use an explicit local HTTP URL with a port",
    )
    return value.rstrip("/")


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


class DemoClient:
    def __init__(self, client):
        self.client = client

    def get(self, path):
        return self.client.get(path).raise_for_status().json()

    def post(self, path, body, *, key=None):
        return (
            self.client.post(path, json=body, headers={"Idempotency-Key": key or str(uuid4())})
            .raise_for_status()
            .json()
        )

    def wait(self, rid, *, paused=False, predicate=None, seconds=30):
        deadline, last = time.monotonic() + seconds, None
        while True:
            run = self.get(f"/api/runs/{rid}")
            if predicate and predicate(run):
                return run
            if not predicate and (
                (paused and run["status"] == "waiting_user" and (run["wait"] or {}).get("interrupt_id"))
                or (not paused and run["status"] in TERMINAL)
            ):
                return run
            if predicate and run["status"] in TERMINAL:
                raise ValueError("Request ended before the expected boundary: " + run["status"])
            if paused and run["status"] in TERMINAL:
                raise ValueError("Request ended before confirmation: " + run["status"])
            current = (run["status"], run.get("error_code"))
            if current != last:
                print(json.dumps({"run_id": rid, "status": current[0]}), flush=True)
                last = current
            require(time.monotonic() < deadline, "Timed out waiting for run " + rid)
            time.sleep(0.25)

    def upload(self, name, content):
        asset = (
            self.client.post("/api/assets", files={"file": (name, content, "video/mp4")})
            .raise_for_status()
            .json()
        )
        clip = self.post("/api/clips", {"asset_id": asset["id"]})
        session = self.post("/api/sessions", {"clip_id": clip["id"]})
        return {"session_id": session["id"], "clip_id": clip["id"], "asset_id": asset["id"], "name": name}

    def submit(self, ws, *, intent="analyse", labels=None, report=None, message=None, key=None):
        body = (
            {"message": message}
            if message
            else {
                "goal": {
                    "intent": intent,
                    "target_label_ids": labels if labels is not None else [4],
                    "analysis_mode": "existing_only" if intent in {"explain", "summarise"} else "adaptive",
                }
            }
        )
        body["context"] = {
            "source_report_id": report,
            "reference": {
                "scope": "explicit",
                "source": "viewed_report" if report else "current_clip",
            },
        }
        return self.post(f"/api/sessions/{ws['session_id']}/turns", body, key=key)["run_id"]

    def followup(self, ws, source, intent, labels):
        rid = self.submit(ws, intent=intent, labels=labels, report=source["report_id"])
        done = self.wait(rid)
        require(done["status"] == "completed", "Saved-result follow-up failed")
        require(self.get(f"/api/runs/{rid}/tools") == [], "Follow-up unexpectedly ran vision")
        require(done["response"]["prediction"] == source["response"]["prediction"], "Prediction changed")
        explanation = self.get(f"/api/reports/{done['report_id']}/explanation")
        require(explanation["status"] == "available", "Explanation source did not verify")
        require(explanation["review"]["scope_label_ids"] == labels, "Explanation scope changed")
        return done


def health_check(api):
    health, caps = api.get("/api/health"), api.get("/api/capabilities")
    require(health["status"] == "ok" and health["llm"] == "disabled", "Demo must keep DeepSeek disabled")
    require(health["graph_version"] == GRAPH and caps["context_binding"] == "bound-v1", "Unexpected graph")
    require(caps["tools"] == TOOLS and len(caps["nodes"]) == 12, "Unexpected vision tools/nodes")
    require(
        set(caps["module_slots"]) == {"M2", "M3", "M4", "M5"}
        and all(v["kind"] == "baseline" for v in caps["module_slots"].values()),
        "Reserved modules changed",
    )


def powershell_json(command):
    require(os.name == "nt", "This demo process audit currently targets Windows")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    require(result.returncode == 0 and result.stdout.strip(), "Local process audit failed")
    return json.loads(result.stdout.strip())


def worker_pid(execution_id):
    require(re.fullmatch(r"[0-9a-f-]{36}", execution_id) is not None, "Invalid execution ID")
    result = powershell_json(
        "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and "
        f"$_.CommandLine -like '*module_slots_worker.py*--execution-id {execution_id}*' "
        "} | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    )
    pids = result if isinstance(result, list) else [result]
    require(len(pids) == 1 and type(pids[0]) is int, "Cannot identify exactly one started worker")
    return pids[0]


def server_pid(api):
    port = urlsplit(str(api.client.base_url)).port
    return powershell_json(
        f"Get-NetTCPConnection -LocalPort {int(port)} -State Listen | "
        "Select-Object -First 1 -ExpandProperty OwningProcess | ConvertTo-Json -Compress"
    )


def process_gone(pid):
    require(type(pid) is int and pid > 0, "Invalid worker PID")
    if os.name == "nt":
        command = f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 1 }} else {{ exit 0 }}"
        return (
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", command], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def prepare(api, output):
    require(
        not (output / "prepared.json").exists(), "Preserve previous demo records; use a new output folder"
    )
    checks = []
    # No-video and invalid-upload paths terminate without any classifier output.
    empty = {"session_id": api.post("/api/sessions", {})["id"]}
    failed = api.wait(api.submit(empty))
    require(
        failed["status"] in {"rejected", "failed"} and not (failed.get("response") or {}).get("prediction"),
        "No-video request must not produce a prediction",
    )
    invalid = []
    for name, data, code in [
        ("notes.txt", b"not video", "MEDIA_TYPE"),
        ("empty.mp4", b"", "INVALID_CONTAINER"),
        ("broken.mp4", b"not an mp4", "INVALID_CONTAINER"),
    ]:
        response = api.client.post("/api/assets", files={"file": (name, data, "video/mp4")})
        require(
            response.status_code == 415 and response.json()["error"]["error_code"] == code,
            "Invalid upload was not rejected as expected",
        )
        invalid.append({"file": name, "http_status": response.status_code, "code": code})
    checks.append("No video and invalid container uploads produce no prediction")

    real_record = output / "real-video.json"
    if real_record.exists():
        prior = json.loads(real_record.read_text(encoding="utf-8"))
        ws, run = prior["workspace"], api.get("/api/runs/" + prior["run_id"])
        require(run["status"] == "completed", "Saved real run unavailable; cannot silently rerun inference")
    else:
        with (ROOT / SAMPLE).open("rb") as stream:
            ws = api.upload("demo-training.mp4", stream)
        rid = api.submit(ws)
        run = api.wait(rid, seconds=850)
        require(run["status"] == "completed", "Real video failed: " + str(run.get("error_code")))
        write_new(real_record, {"workspace": ws, "run_id": rid})
    prediction = run["response"]["prediction"]
    executions = api.get(f"/api/runs/{run['id']}/tools")
    require([e["request"]["tool_id"] for e in executions] == TOOLS, "Incomplete real pipeline")
    require(
        all(e["status"] == "succeeded" and not e["output"]["is_stub"] for e in executions),
        "A stage failed or used placeholders",
    )
    require(prediction["video_sha256"] == sha(ROOT / SAMPLE), "Wrong video used")
    require(prediction["profile_id"].endswith(sha(ROOT / PIPELINE)), "Wrong algorithm profile used")
    # This is a known regression reference, never passed to the classifier or Agent as input.
    oracle = json.loads(
        (ROOT / "test-results/m1-stability-v1/oracle/raw26-with-scores.json").read_text(encoding="utf-8")
    )
    expected = next(r for r in oracle["rows"] if r["record_id"] == "SupC04/r00/ch0")
    delta = max(
        abs(row["uncalibrated_score"] - score)
        for row, score in zip(prediction["scores"], expected["scores"], strict=True)
    )
    require(
        delta <= 1e-6 and prediction["predicted_label_ids"] == expected["predicted"],
        "Regression output changed",
    )
    require(
        api.get(f"/api/reports/{run['report_id']}/evidence")["status"] == "available", "Evidence unavailable"
    )
    checks.append("Known public clip completes all four real stages with frozen-profile output")

    followup = api.followup(ws, run, "explain", [4])
    summary = api.followup(ws, followup, "summarise", [4])
    html = api.client.get(f"/api/reports/{summary['report_id']}/export").raise_for_status().text
    require("Evidence-constrained review" in html, "Readable summary export is missing")
    checks.append("Explanation and summary/export reuse the exact original prediction with zero vision tools")

    # A syntactically MP4-like header still requires actual decode validation.
    broken_ws = api.upload("decode-failure.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\0" * 24)
    broken_run = api.wait(api.submit(broken_ws), seconds=90)
    require(
        broken_run["status"] == "failed" and not (broken_run.get("response") or {}).get("prediction"),
        "Undecodable input produced a prediction",
    )
    require(
        not (broken_run.get("response") or {}).get("classification_claims"), "Failed input showed findings"
    )
    checks.append("MP4 magic bytes alone do not bypass decoder validation; failure shows no findings")

    # Cancel a real worker after its start receipt, not a fake long-running tool.
    with (ROOT / SAMPLE).open("rb") as stream:
        cancel_ws = api.upload("cancel-check.mp4", stream)
    cancel_id = api.submit(cancel_ws)
    worker = None
    until = time.monotonic() + 30
    while worker is None:
        events = api.get(f"/api/runs/{cancel_id}/events")["events"]
        worker = next((e["payload"] for e in events if e["kind"] == "vision.stage_started"), None)
        require(time.monotonic() < until, "Real worker did not start before cancellation check")
        if worker is None:
            time.sleep(0.1)
    pid = worker_pid(worker["execution_id"])
    api.post(f"/api/runs/{cancel_id}/cancel", {})
    cancelled = api.wait(cancel_id)
    require(
        cancelled["status"] == "cancelled" and cancelled["response"] is None,
        "Cancelled run published findings",
    )
    require(cancelled["report_id"] is None, "Cancelled run published a report")
    require(process_gone(pid), "Cancelled worker process is still running")
    checks.append("Cancel terminates the actual ML subprocess and publishes no report")

    # Disabled LLM enters N03; deliberately leave it durable across a real service restart.
    pending_id = api.submit(
        ws, report=summary["report_id"], message="Explain the same finding after restart."
    )
    pending = api.wait(pending_id, paused=True)
    turn = api.get(f"/api/turns/{pending['turn_id']}")
    require(turn["llm_resolution"]["code"] == "LLM_DISABLED", "Unexpected model call")
    prepared = {
        "passed": True,
        "base_url": str(api.client.base_url).rstrip("/"),
        "checks": checks,
        "workspace": ws,
        "analysis_run_id": run["id"],
        "analysis_report_id": run["report_id"],
        "explain_run_id": followup["id"],
        "summary_run_id": summary["id"],
        "failed_decode_run_id": broken_run["id"],
        "failed_decode_code": broken_run["error_code"],
        "cancelled_run_id": cancel_id,
        "cancelled_worker_pid": pid,
        "invalid_inputs": invalid,
        "no_video_run_id": failed["id"],
        "pending_run_id": pending_id,
        "pending_turn_id": pending["turn_id"],
        "pending_snapshot": turn["context_snapshot"],
        "server_pid_before_restart": server_pid(api),
        "maximum_score_difference": delta,
        "prediction": prediction,
        "real_completed_analyses": 1,
        "real_cancelled_worker_starts": 1,
        "real_llm_calls": 0,
        "new_training": False,
        "new_manual_labels": False,
        "requires_actual_service_restart_before_resume": True,
    }
    write_new(output / "prepared.json", prepared)
    print(
        json.dumps(
            {
                "passed": True,
                "phase": "prepare",
                "pending_run_id": pending_id,
                "next": "Restart start_demo.ps1, then run --phase resume with this output directory",
            }
        ),
        flush=True,
    )


def resume(api, output):
    require(not (output / "acceptance.json").exists(), "Preserve previous final acceptance")
    prepared = json.loads((output / "prepared.json").read_text(encoding="utf-8"))
    require(
        server_pid(api) != prepared["server_pid_before_restart"],
        "Actually restart the demo service before resume",
    )
    require(str(api.client.base_url).rstrip("/") == prepared["base_url"], "Demo server URL changed")
    pending = api.wait(prepared["pending_run_id"], paused=True)
    turn = api.get(f"/api/turns/{pending['turn_id']}")
    require(
        turn["context_snapshot"] == prepared["pending_snapshot"], "Waiting context changed across restart"
    )
    wait = pending["wait"]
    body = {
        "wait_id": wait["wait_id"],
        "interrupt_id": wait["interrupt_id"],
        "revision": pending["revision"],
        "goal": {"intent": "explain", "target_label_ids": [4]},
        "context_confirmation": {"snapshot_id": wait["context_snapshot_id"], "use_bound_source": True},
    }
    key = str(uuid4())
    api.post(f"/api/runs/{pending['id']}/resume", body, key=key)
    done = api.wait(pending["id"])
    require(done["status"] == "completed", "Resumed review did not complete")
    require(api.get(f"/api/runs/{done['id']}/tools") == [], "Resume unexpectedly ran vision")
    require(done["response"]["prediction"] == prepared["prediction"], "Resume changed the source prediction")
    repeated = api.post(f"/api/runs/{pending['id']}/resume", body, key=key)
    require(repeated["run_id"] == done["id"], "Repeated resume did not reuse the same request")
    ws = prepared["workspace"]
    summary = api.get("/api/runs/" + prepared["summary_run_id"])
    require(
        api.get(f"/api/reports/{summary['report_id']}/explanation")["status"] == "available",
        "Saved summary could not be verified after restart",
    )
    history = api.get(f"/api/sessions/{ws['session_id']}/history")
    require(history["latest_recap"]["run_id"] == summary["id"], "Saved summary pointer was lost")
    final_summary = api.followup(ws, done, "summarise", [4])
    checks = prepared["checks"] + [
        "Waiting confirmation survives restart with identical video/report/scope snapshot",
        "Resume retries are idempotent and issue no new vision calls",
        "Saved summary remains verified after restart and can be continued",
    ]
    write_new(
        output / "acceptance.json",
        {
            **prepared,
            "checks": checks,
            "resumed_run_id": done["id"],
            "final_summary_run_id": final_summary["id"],
            "final_summary_report_id": final_summary["report_id"],
            "ready_for_browser_demo": True,
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "phase": "resume",
                "checks": checks,
                "new_vision_calls_during_resume": 0,
                "real_llm_calls": 0,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8005", type=local_url)
    parser.add_argument("--phase", choices=["prepare", "resume"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.base_url, timeout=30, follow_redirects=False, trust_env=False) as client:
        api = DemoClient(client)
        health_check(api)
        (prepare if args.phase == "prepare" else resume)(api, args.output_dir)


if __name__ == "__main__":
    main()
