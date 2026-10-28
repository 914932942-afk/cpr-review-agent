"""Prepare verified local presentation cases. Reuse saved results; one new full analysis only."""

from __future__ import annotations

import argparse
import json
from uuid import uuid4

import httpx
from demo_preflight import ROOT, TOOLS, preflight, require, sha
from demo_scenarios import DemoClient, local_url, write_new

LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]
OUTPUT = ROOT / "test-results/demo-cases-v1/http"
STATE = OUTPUT / "run-state.json"


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=local_url, default="http://127.0.0.1:8007")
    parser.add_argument("--allow-vision", action="store_true")
    args = parser.parse_args()
    require(args.allow_vision, "Use --allow-vision for the one new normal clip and boundary probe")
    require(not (OUTPUT / "acceptance.json").exists(), "Preserve accepted demo cases")
    preflight()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state = read(STATE.relative_to(ROOT)) if STATE.exists() else {}

    def save():
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    controls = read("data/demo-cases-v1/controls/manifest.json")
    for item in [controls["short_video"], controls["invalid_container"]]:
        require(sha(ROOT / item["path"]) == item["sha256"], "Boundary control changed")
    old = read("test-results/product-demo-v2/browser/acceptance.json")
    next_practice = read("test-results/practice-continuity-v1/ui/acceptance.json")
    reference = read("test-results/reference-v1/live/acceptance.json")
    continuity = read("test-results/practice-continuity-v1/live/acceptance.json")
    cached = {r["record_id"]: r for r in read("test-results/module-slots-v1/raw26-acceptance.json")["rows"]}
    sources = {}
    for folder in ["SupC00--r00--ch0", "SupC04--r00--ch0", "SupC05--r00--ch0"]:
        relative = f"data/public_cpr/rgb-acceptance-v1/records/{folder}"
        meta = read(relative + "/manifest.json")
        require(sha(ROOT / relative / "video.mp4") == meta["video_sha256"], "Public sample changed")
        sources[meta["record_id"]] = {
            "record_id": meta["record_id"],
            "original_label_ids": meta["original_labels"],
            "video_rel": relative + "/video.mp4",
            "video_sha256": meta["video_sha256"],
            "manifest_rel": relative + "/manifest.json",
            "manifest_sha256": sha(ROOT / relative / "manifest.json"),
            "frames": meta["video_validation"]["decoded_frames"],
            "fps": meta["video_validation"]["fps"],
        }
    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        api = DemoClient(client)
        caps = api.get("/api/capabilities")
        require(
            caps["context_binding"] == "bound-v1" and caps["tools"] == TOOLS,
            "Use the real four-stage product service",
        )

        def submit(case, ws, *, report=None, existing=False):
            entry = state.setdefault(case, {})
            if "key" not in entry:
                entry["key"] = str(uuid4())
                save()
            body = {
                "goal": {
                    "intent": "analyse",
                    "target_label_ids": LABELS,
                    "analysis_mode": "existing_only" if existing else "adaptive",
                },
                "context": {
                    "source_report_id": report,
                    "reference": {
                        "scope": "explicit",
                        "source": "viewed_report" if report else "current_clip",
                    },
                },
            }
            result = api.post(f"/api/sessions/{ws['session_id']}/turns", body, key=entry["key"])
            entry["run_id"] = result["run_id"]
            save()
            return api.wait(entry["run_id"], seconds=850)

        def checked(run_id, record, *, expected_tools=None):
            run = api.get("/api/runs/" + run_id)
            require(run["status"] == "completed", "A prepared success case must be completed")
            prediction = run["response"]["prediction"]
            require(
                prediction["video_sha256"] == sources[record]["video_sha256"],
                "Prediction belongs to another video",
            )
            require(
                [c["label_id"] for c in run["response"]["classification_claims"]] == LABELS,
                "Demonstrate the complete supported scope",
            )
            expected = cached[record]
            delta = max(
                abs(s["uncalibrated_score"] - v)
                for s, v in zip(prediction["scores"], expected["scores"], strict=True)
            )
            require(
                prediction["predicted_label_ids"] == expected["predicted"] and delta <= 1e-6,
                "Frozen regression output changed",
            )
            tools = api.get(f"/api/runs/{run_id}/tools")
            if expected_tools is not None:
                require([t["request"]["tool_id"] for t in tools] == expected_tools, "Unexpected vision tools")
                require(
                    all(
                        t["status"] == "succeeded" and not (t.get("output") or {}).get("is_stub")
                        for t in tools
                    ),
                    "No placeholder or failed stages allowed",
                )
            evidence = api.get(f"/api/reports/{run['report_id']}/evidence")
            require(evidence["status"] == "available", "Saved source cannot be verified")
            require(
                api.get(f"/api/reports/{run['report_id']}/explanation")["status"] == "available",
                "Saved review unavailable",
            )
            return {
                "run_id": run_id,
                "report_id": run["report_id"],
                "scope_label_ids": LABELS,
                "model_flagged_ids": prediction["predicted_label_ids"],
                "source_execution_id": prediction["execution_id"],
                "maximum_score_difference_from_cached_regression": delta,
                "new_tools_for_run": [
                    {
                        "tool": t["request"]["tool_id"],
                        "status": t["status"],
                        "duration_ms": t.get("duration_ms"),
                    }
                    for t in tools
                ],
            }

        bent = checked(old["analysis_run_id"], "SupC04/r00/ch0", expected_tools=TOOLS)
        original_tilt = api.get("/api/runs/" + next_practice["analysis_run_id"])
        tilted = submit(
            "tilted", next_practice["workspace"], report=original_tilt["report_id"], existing=True
        )
        tilt_check = checked(tilted["id"], "SupC05/r00/ch0", expected_tools=[])
        require(
            tilted["response"]["prediction"] == original_tilt["response"]["prediction"],
            "Expanding display scope changed the saved prediction",
        )

        normal_entry = state.setdefault("normal", {})
        if "workspace" not in normal_entry:
            with (ROOT / sources["SupC00/r00/ch0"]["video_rel"]).open("rb") as stream:
                normal_entry["workspace"] = api.upload("case-03-normal-labelled.mp4", stream)
            save()
        normal = submit("normal", normal_entry["workspace"])
        normal_check = checked(normal["id"], "SupC00/r00/ch0", expected_tools=TOOLS)
        require(
            normal_check["model_flagged_ids"] == [],
            "Normal-label case does not match its engineering reference",
        )

        invalid_path = ROOT / controls["invalid_container"]["path"]
        with invalid_path.open("rb") as stream:
            invalid = client.post("/api/assets", files={"file": (invalid_path.name, stream, "video/mp4")})
        require(
            invalid.status_code == 415 and invalid.json()["error"]["error_code"] == "INVALID_CONTAINER",
            "Invalid upload was not rejected",
        )

        short_entry = state.setdefault("short", {})
        if "workspace" not in short_entry:
            with (ROOT / controls["short_video"]["path"]).open("rb") as stream:
                short_entry["workspace"] = api.upload("case-05-too-short-3frames.mp4", stream)
            save()
        short = submit("short", short_entry["workspace"])
        failed_tools = api.get(f"/api/runs/{short['id']}/tools")
        require(
            short["status"] == "failed" and not (short.get("response") or {}).get("prediction"),
            "Short input must not produce a prediction",
        )
        require(not (short.get("response") or {}).get("classification_claims"), "Short input showed findings")
        require(
            len(failed_tools) == 1
            and failed_tools[0]["request"]["tool_id"] == "pose.extract"
            and failed_tools[0]["status"] == "failed",
            "Short input must stop at input validation",
        )
        probe_logs = list(
            (ROOT / "data/product-demo-v2/runtime/analysis" / failed_tools[0]["id"]).glob("*/process.log")
        )
        require(
            any(
                "InputRejected: VIDEO_TOO_SHORT" in p.read_text(encoding="utf-8", errors="replace")
                for p in probe_logs
            ),
            "The boundary must fail for VIDEO_TOO_SHORT, not another worker error",
        )

        refs = api.get(f"/api/runs/{reference['run_id']}/references")
        require(
            refs["status"] == "available" and [e["label_id"] for e in refs["items"]] == [0, 4],
            "Saved reference case unavailable",
        )
        link = api.get(f"/api/sessions/{continuity['workspace']['session_id']}/continuity")
        current = api.get(f"/api/sessions/{continuity['workspace']['session_id']}/context")
        require(
            link["selection"]["status"] == "available"
            and link["selection"]["label_ids"] == [4]
            and current["source"] is None,
            "Starting focus case is no longer isolated",
        )
        for rid in [tilted["id"], normal["id"], short["id"]]:
            require(
                not any(
                    e["payload"].get("called") is True for e in api.get(f"/api/runs/{rid}/events")["events"]
                ),
                "Preparing cases must not call a provider",
            )
        cases = [
            {
                "id": "C01",
                "title": "弯臂：单项发现",
                "kind": "public_video",
                "workspace": old["workspace"],
                "source": sources["SupC04/r00/ch0"],
                "verified": bent,
            },
            {
                "id": "C02",
                "title": "手臂倾斜：切换错误类型",
                "kind": "public_video",
                "workspace": next_practice["workspace"],
                "source": sources["SupC05/r00/ch0"],
                "verified": tilt_check,
            },
            {
                "id": "C03",
                "title": "正常标签：未标记的表达边界",
                "kind": "public_video",
                "workspace": normal_entry["workspace"],
                "source": sources["SupC00/r00/ch0"],
                "verified": normal_check,
            },
            {
                "id": "C04",
                "title": "损坏容器：上传时拒绝",
                "kind": "invalid_control",
                "source": controls["invalid_container"],
                "verified": {"http_status": 415, "error_code": "INVALID_CONTAINER", "new_vision_tools": 0},
            },
            {
                "id": "C05",
                "title": "三帧视频：分析前置检查失败",
                "kind": "derived_boundary_control",
                "workspace": short_entry["workspace"],
                "source": controls["short_video"],
                "verified": {
                    "run_id": short["id"],
                    "status": short["status"],
                    "error_code": short["error_code"],
                    "has_prediction": False,
                    "tool_attempts": 1,
                    "full_classifier_run": False,
                },
            },
            {
                "id": "C06",
                "title": "有来源的参考对照",
                "kind": "saved_agent_workflow",
                "workspace": old["workspace"],
                "verified": {
                    "run_id": reference["run_id"],
                    "reference_ids": [e["id"] for e in refs["items"]],
                    "replay_provider_calls": 0,
                },
            },
            {
                "id": "C07",
                "title": "跨次目标记忆：新视频仍未分析",
                "kind": "saved_agent_workflow",
                "workspace": continuity["workspace"],
                "verified": {
                    "run_id": continuity["run_id"],
                    "starting_focus": [4],
                    "current_source": None,
                    "replay_provider_calls": 0,
                },
            },
        ]
        value = {
            "schema": "presentation-demo-cases-v1",
            "passed": True,
            "base_url": args.base_url,
            "dataset_url": "https://huggingface.co/datasets/ShunliWang/CPR-Coach",
            "revision": "b8667b8fd6a70ac63cc09fc6b763a9459b7c160a",
            "cases": cases,
            "new_full_analyses": 1,
            "new_failed_input_probes": 1,
            "new_provider_calls": 0,
            "new_training": False,
            "new_annotations": False,
            "case_selection_is_engineering_demo_not_blind_evaluation": True,
        }
        write_new(OUTPUT / "acceptance.json", value)
        write_new(ROOT / "data/demo-cases-v1/cases.json", value)
        print(json.dumps({"passed": True, "cases": len(cases), "new_full_analyses": 1, "provider_calls": 0}))


if __name__ == "__main__":
    main()
