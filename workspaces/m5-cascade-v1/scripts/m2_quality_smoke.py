"""Actual HTTP, video decoding and four real ML processes. Reference targets read only after commit."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspaces/m2-quality-v1"
OUT = ROOT / "test-results/m2-quality-v1/http"
RUNTIME = WORKSPACE / "data/m2-quality-v1/runtime"
LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]
TOOLS = ["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "acceptance.json").exists():
        raise FileExistsError("Preserve accepted M2 HTTP evidence")
    state_path = OUT / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def save():
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    with httpx.Client(base_url="http://127.0.0.1:8008", timeout=60) as client:

        def get(path):
            return client.get("/api" + path).raise_for_status().json()

        def post(path, body):
            return (
                client.post("/api" + path, json=body, headers={"Idempotency-Key": str(uuid4())})
                .raise_for_status()
                .json()
            )

        caps = get("/capabilities")
        assert caps["llm"]["mode"] == "disabled"
        assert caps["module_slots"]["M2"]["kind"] == "improved"
        assert all(caps["module_slots"][k]["kind"] == "baseline" for k in ["M3", "M4", "M5"])
        rows = []
        for case in ["SupC04--r00--ch0", "SupC00--r00--ch0"]:
            if case not in state:
                video = ROOT / "data/public_cpr/rgb-acceptance-v1/records" / case / "video.mp4"
                with video.open("rb") as f:
                    asset = (
                        client.post("/api/assets", files={"file": ("m2-" + case + ".mp4", f, "video/mp4")})
                        .raise_for_status()
                        .json()
                    )
                assert asset["validation_level"] == "decoded_before_session"
                clip = post("/clips", {"asset_id": asset["id"]})
                session = post("/sessions", {"clip_id": clip["id"]})
                created = post(
                    "/sessions/" + session["id"] + "/turns",
                    {"goal": {"intent": "analyse", "target_label_ids": LABELS}},
                )
                state[case] = {
                    "asset_id": asset["id"],
                    "clip_id": clip["id"],
                    "session_id": session["id"],
                    "run_id": created["run_id"],
                    "name": asset["name"],
                }
                save()
            record = state[case]
            rid = record["run_id"]
            previous = None
            deadline = time.monotonic() + 900
            while True:
                run = get("/runs/" + rid)
                tools = get("/runs/" + rid + "/tools")
                marker = (
                    run["status"] + ":" + ",".join(t["request"]["tool_id"] + "=" + t["status"] for t in tools)
                )
                if marker != previous:
                    print(case, marker, flush=True)
                    previous = marker
                if run["status"] in ["completed", "failed", "rejected", "cancelled", "partially_completed"]:
                    break
                assert time.monotonic() < deadline, "Real M2 request exceeded deadline"
                time.sleep(0.6)
            assert run["status"] == "completed", {"status": run["status"], "error": run["error_code"]}
            assert [t["request"]["tool_id"] for t in tools] == TOOLS
            assert all(t["status"] == "succeeded" and t["output"]["is_stub"] is False for t in tools)
            raw = json.loads((RUNTIME / tools[-1]["output"]["artifact_rel"]).read_text())
            assert raw["labels_used"] is False and raw["public_label_verification"] is None
            assert raw["module_executions"]["M2"]["implementation_id"] == "quality-gated-disagreement-v1"
            assert (
                raw["module_executions"]["M2"]["kind"] == "improved"
                and raw["module_executions"]["M5"]["kind"] == "baseline"
            )
            assert raw["m2_fusion"]["algorithm"] == "quality-gated-disagreement-v1"
            assert raw["model_or_threshold_modified"] is True
            # The baseline oracle is deliberately consulted only after inference has committed.
            oracle = json.loads(
                (ROOT / "test-results/m1-stability-v1/oracle/raw26-with-scores.json").read_text()
            )
            expected = next(r for r in oracle["rows"] if r["record_id"] == case.replace("--", "/"))
            prediction = run["response"]["prediction"]
            assert prediction["predicted_label_ids"] == expected["predicted"]
            max_delta = max(
                abs(s["uncalibrated_score"] - v) for s, v in zip(prediction["scores"], expected["scores"])
            )
            assert max_delta <= 1e-6
            assert get("/reports/" + run["report_id"] + "/evidence")["status"] == "available"
            assert client.get("/api/reports/" + run["report_id"] + "/export").status_code == 200
            if not record.get("summary_run_id"):
                record["summary_run_id"] = post(
                    "/sessions/" + record["session_id"] + "/turns",
                    {
                        "goal": {
                            "intent": "summarise",
                            "analysis_mode": "existing_only",
                            "target_label_ids": LABELS,
                        },
                        "context": {
                            "source_report_id": run["report_id"],
                            "reference": {"scope": "explicit", "source": "viewed_report"},
                        },
                    },
                )["run_id"]
                save()
            deadline = time.monotonic() + 30
            while True:
                summary = get("/runs/" + record["summary_run_id"])
                if summary["status"] in ["completed", "failed", "rejected", "cancelled"]:
                    break
                assert time.monotonic() < deadline
                time.sleep(0.2)
            assert summary["status"] == "completed" and summary["response"]["reused_result"]
            assert get("/runs/" + summary["id"] + "/tools") == []
            for checked in [run, summary]:
                assert not any(
                    e["payload"].get("called") is True
                    for e in get("/runs/" + checked["id"] + "/events")["events"]
                )
            record.update(
                report_id=run["report_id"],
                predicted_labels=prediction["predicted_label_ids"],
                maximum_score_delta=max_delta,
                m2_audit=raw["m2_fusion"],
                tools=[{"id": t["request"]["tool_id"], "ms": t["duration_ms"]} for t in tools],
            )
            save()
            rows.append(record)
        comparison = get(
            "/review-comparison?left_report_id="
            + rows[0]["report_id"]
            + "&right_report_id="
            + rows[1]["report_id"]
        )
        assert comparison["status"] == "available" and comparison["common_label_ids"] == LABELS
        result = {
            "passed": True,
            "real_http_and_ml_processes": True,
            "mocked_tools": False,
            "new_vision_executions": len(rows) * 4,
            "provider_calls": 0,
            "records": rows,
            "m3_m4_m5": "unchanged baselines",
            "m1": "accepted optimized implementation",
            "source_selection_sha256": hashlib.sha256(
                (ROOT / "data/m2-quality-v1/experiment-01/selection.json").read_bytes()
            ).hexdigest(),
        }
        (OUT / "acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("M2 actual HTTP, both cold videos, evidence, report and summary reuse passed", flush=True)


if __name__ == "__main__":
    main()
