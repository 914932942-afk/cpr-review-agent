"""Real HTTP and four real subprocess stages, compared with a frozen M1 cache oracle; labels are read after inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-first-run")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier acceptance")
    cases = ["SupC04--r00--ch0"]
    started = time.monotonic()
    records = []
    with httpx.Client(base_url=args.base_url, timeout=30) as client:

        def get(path):
            return client.get(path).raise_for_status().json()

        def post(path, body):
            return (
                client.post(path, json=body, headers={"Idempotency-Key": str(uuid4())})
                .raise_for_status()
                .json()
            )

        health = get("/api/health")
        assert health["vision_pipeline"] == "branches" and health["llm"] == "disabled"
        assert get("/api/capabilities")["tools"] == TOOLS
        for case in cases:
            video = ROOT / "data/public_cpr/rgb-acceptance-v1/records" / case / "video.mp4"
            case_started = time.monotonic()
            resumed = bool(args.resume_first_run and case == cases[0])
            if resumed:
                prior = get(f"/api/runs/{args.resume_first_run}")
                assert prior["status"] == "completed"
                assert (
                    prior["response"]["prediction"]["video_sha256"]
                    == __import__("hashlib").sha256(video.read_bytes()).hexdigest()
                )
                rid = prior["id"]
                session = {"id": prior["session_id"]}
                asset = {"id": prior["response"]["prediction"]["asset_id"]}
            else:
                with video.open("rb") as stream:
                    asset = (
                        client.post("/api/assets", files={"file": ("practice.mp4", stream, "video/mp4")})
                        .raise_for_status()
                        .json()
                    )
                clip = post("/api/clips", {"asset_id": asset["id"]})
                session = post("/api/sessions", {"clip_id": clip["id"]})
                created = post(
                    f"/api/sessions/{session['id']}/turns",
                    {"goal": {"intent": "analyse", "target_label_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]}},
                )
                rid = created["run_id"]
            print(json.dumps({"stage": "submitted", "case": case, "run_id": rid}), flush=True)
            limit, last_stage = time.monotonic() + 600, None
            while True:
                run = get(f"/api/runs/{rid}")
                executions = get(f"/api/runs/{rid}/tools")
                running = [
                    t["request"]["tool_id"] for t in executions if t["status"] in {"running", "queued"}
                ]
                if running and running[-1] != last_stage:
                    last_stage = running[-1]
                    print(json.dumps({"stage": last_stage, "case": case}), flush=True)
                if run["status"] in {"completed", "failed", "rejected", "cancelled", "partially_completed"}:
                    break
                assert run["response"] is None, "Intermediate branch output must not become a final finding"
                assert time.monotonic() < limit, run
                time.sleep(0.5)
            assert run["status"] == "completed", run
            executions = get(f"/api/runs/{rid}/tools")
            assert [t["request"]["tool_id"] for t in executions] == TOOLS
            assert all(t["status"] == "succeeded" and t["output"]["is_stub"] is False for t in executions)
            assert (
                executions[1]["request"]["input_bindings"]["pose_receipt_id"]
                == executions[2]["request"]["input_bindings"]["pose_receipt_id"]
                == executions[0]["id"]
            )
            prediction = run["response"]["prediction"]
            # Reference labels/scores are loaded only after final inference has been committed.
            oracle = json.loads(
                (ROOT / "test-results/m1-stability-v1/oracle/raw26-with-scores.json").read_text(
                    encoding="utf-8"
                )
            )
            reference = next(row for row in oracle["rows"] if row["record_id"] == case.replace("--", "/"))
            difference = max(
                abs(a["uncalibrated_score"] - b) for a, b in zip(prediction["scores"], reference["scores"])
            )
            assert prediction["predicted_label_ids"] == reference["predicted"]
            assert difference <= 1e-6, difference
            stage_differences = {}
            for index, field in [(1, "skeleton_scores"), (2, "rgb_scores")]:
                raw_path = (
                    ROOT / "data/m1-optimization-v2/runtime" / executions[index]["output"]["artifact_rel"]
                )
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                assert raw["labels_used"] is False and raw["model_or_threshold_modified"] is False
                delta = max(
                    abs(score["uncalibrated_score"] - expected)
                    for score, expected in zip(raw["scores"], reference[field])
                )
                assert delta <= 1e-6, (field, delta)
                if index == 1:
                    assert raw["fullpose_audit"]["m1_stability"]["implementation_revision"] == "m1-fast-v1"
                stage_differences[field] = delta
            evidence = get(f"/api/reports/{run['report_id']}/evidence")
            assert evidence["status"] == "available", evidence
            m1 = evidence["review"]["skeleton_processing"]
            assert m1["method"] == "cpr-m1-stability-v3"
            assert m1["reconstructed_points_are_observations"] is False
            assert evidence["review"]["video_sha256"] == prediction["video_sha256"]
            followup_id = post(
                f"/api/sessions/{session['id']}/turns",
                {"goal": {"intent": "summarise", "analysis_mode": "existing_only"}},
            )["run_id"]
            until = time.monotonic() + 15
            while True:
                following = get(f"/api/runs/{followup_id}")
                if following["status"] in {"completed", "failed", "rejected", "cancelled"}:
                    break
                assert time.monotonic() < until
                time.sleep(0.2)
            assert following["status"] == "completed" and following["response"]["reused_result"]
            assert get(f"/api/runs/{followup_id}/tools") == []
            assert get(f"/api/sessions/{session['id']}/history")["items"][0]["run_id"] == followup_id
            record = {
                "case": case,
                "passed": True,
                "run_id": rid,
                "session_id": session["id"],
                "asset_id": asset["id"],
                "report_id": run["report_id"],
                "summary_run_id": followup_id,
                "predicted_labels": prediction["predicted_label_ids"],
                "maximum_score_difference": difference,
                "inference_resumed_for_acceptance": resumed,
                "branch_score_differences": stage_differences,
                "m1_processing": m1,
                "video_sha256": prediction["video_sha256"],
                "profile_id": prediction["profile_id"],
                "stage_cost_ms": {t["request"]["tool_id"]: t["duration_ms"] for t in executions},
                "wall_seconds_including_upload_and_followup": time.monotonic() - case_started,
            }
            records.append(record)
            args.output.with_name("http-progress.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )
            print(json.dumps({"stage": "verified", **record}), flush=True)
        assert get("/api/health")["llm"] == "disabled"
    result = {
        "passed": True,
        "mode": "actual_http_four_real_stage_subprocesses_m1_fast",
        "cases": records,
        "training_performed": False,
        "manual_labels_added": False,
        "real_llm_calls": 0,
        "wall_seconds": time.monotonic() - started,
        "scope": "One existing public clip; M1 optimization/cache/runtime parity, not new model accuracy evidence",
        "cost_scope": "Measured cold process launches and validation per tool; shared perception counted once; no machine-cold or speedup claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps({"passed": True, "cases": len(records), "wall_seconds": result["wall_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
