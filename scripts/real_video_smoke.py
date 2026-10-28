"""Actual HTTP + actual ML subprocess acceptance; expected labels are read only after inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve previous acceptance"
    started = time.monotonic()
    with httpx.Client(base_url=args.base_url, timeout=30) as client:

        def post(path, body):
            return (
                client.post(path, json=body, headers={"Idempotency-Key": str(uuid4())})
                .raise_for_status()
                .json()
            )

        def wait(rid, predicate=None):
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                run = client.get(f"/api/runs/{rid}").raise_for_status().json()
                if (
                    predicate(run)
                    if predicate
                    else run["status"]
                    in {"completed", "failed", "rejected", "cancelled", "partially_completed"}
                ):
                    return run
                time.sleep(0.4)
            raise AssertionError(run)

        def turn(sid, goal=None, message=""):
            return post(f"/api/sessions/{sid}/turns", {"message": message, "goal": goal})["run_id"]

        health = client.get("/api/health").raise_for_status().json()
        assert health["vision"] == "v15-experimental" and health["llm"] == "disabled"
        caps = client.get("/api/capabilities").raise_for_status().json()
        source = ROOT / "data/public_cpr/rgb-acceptance-v1/records/SupC02--r00--ch0/video.mp4"
        with source.open("rb") as video:
            asset = (
                client.post("/api/assets", files={"file": ("training-clip.mp4", video, "video/mp4")})
                .raise_for_status()
                .json()
            )
        clip = post("/api/clips", {"asset_id": asset["id"]})
        sid = post("/api/sessions", {"clip_id": clip["id"]})["id"]
        rid = turn(sid, message="Help me review this training clip.")
        print(json.dumps({"run_id": rid, "session_id": sid, "stage": "submitted"}), flush=True)
        paused = wait(rid, lambda r: r["status"] == "waiting_user" and r["wait"].get("interrupt_id"))
        post(
            f"/api/runs/{rid}/resume",
            {
                "kind": "user_answer",
                "revision": paused["revision"],
                "wait_id": paused["wait"]["wait_id"],
                "interrupt_id": paused["wait"]["interrupt_id"],
                "goal": {"intent": "analyse", "target_label_ids": caps["experimental_labels"]},
            },
        )
        run = wait(rid)
        assert run["status"] == "completed", run
        prediction = run["response"]["prediction"]
        tools = client.get(f"/api/runs/{rid}/tools").raise_for_status().json()
        assert len(tools) == 1 and tools[0]["status"] == "succeeded" and not tools[0]["output"]["is_stub"]
        assert prediction["video_sha256"] == asset["sha256"]
        # Regression comparison after the label-independent pipeline has finished.
        reference = json.loads(
            (
                ROOT / "data/public_cpr/rgb-evaluation-v15/cold-runs-v2/SupC02--r00--ch0/inference.json"
            ).read_text(encoding="utf-8")
        )
        assert prediction["predicted_label_ids"] == reference["predicted_original_label_ids"]
        assert prediction["decoded_frames"] == reference["decoded_frames"]
        score_diff = max(
            abs(x["uncalibrated_score"] - y["uncalibrated_score"])
            for x, y in zip(prediction["scores"], reference["scores"], strict=True)
        )
        assert score_diff < 1e-5
        events = client.get(f"/api/runs/{rid}/events").raise_for_status().json()["events"]
        nodes = sorted({e["node_id"] for e in events if e["kind"] == "node.enter"})
        assert nodes == [f"N{i:02}" for i in range(1, 13)]
        followups = []
        for intent in ("explain", "summarise", "verify"):
            next_id = turn(sid, {"intent": intent, "target_label_ids": [2], "analysis_mode": "existing_only"})
            result = wait(next_id)
            assert result["status"] == "completed" and result["response"]["reused_result"]
            assert client.get(f"/api/runs/{next_id}/tools").json() == []
            assert result["response"]["prediction"]["execution_id"] == prediction["execution_id"]
            followups.append({"intent": intent, "run_id": next_id, "tool_calls": 0})
        export = client.get(f"/api/reports/{run['report_id']}/export").raise_for_status().text
        assert "Scaffold only" not in export and "Experimental clip-level" in export
        media = client.get(f"/api/assets/{asset['id']}/media", headers={"Range": "bytes=0-31"})
        assert media.status_code == 206 and len(media.content) == 32
        summary = {
            "passed": True,
            "mode": "actual_http_and_actual_ml_subprocess",
            "run_id": rid,
            "session_id": sid,
            "asset_id": asset["id"],
            "clip_id": clip["id"],
            "report_id": run["report_id"],
            "nodes": nodes,
            "tool_count": 1,
            "predicted_labels": prediction["predicted_label_ids"],
            "video_sha256": prediction["video_sha256"],
            "bundle_sha256": prediction["bundle_sha256"],
            "decoded_frames": prediction["decoded_frames"],
            "maximum_score_difference": score_diff,
            "followups": followups,
            "playback_range_passed": True,
            "export_passed": True,
            "wall_seconds": time.monotonic() - started,
            "scope": "Integration regression of one existing public clip, not a new accuracy benchmark.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
