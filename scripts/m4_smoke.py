"""Actual cold/expanded video acceptance of M4; no mocks and no inference labels."""

import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test-results/m4-scheduling-v1/http"
LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "acceptance.json").exists():
        raise FileExistsError("Preserve accepted M4 HTTP evidence")
    path = OUT / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {}

    def save():
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    with httpx.Client(base_url="http://127.0.0.1:8011", timeout=60) as c:

        def get(p):
            return c.get("/api" + p).raise_for_status().json()

        def post(p, v):
            return (
                c.post("/api" + p, json=v, headers={"Idempotency-Key": str(uuid4())})
                .raise_for_status()
                .json()
            )

        def wait(rid, seconds=900):
            until = time.monotonic() + seconds
            last = None
            while True:
                r = get("/runs/" + rid)
                tools = get("/runs/" + rid + "/tools")
                mark = (
                    r["status"] + ":" + ",".join(t["request"]["tool_id"] + "=" + t["status"] for t in tools)
                )
                if mark != last:
                    print(rid, mark, flush=True)
                    last = mark
                if r["status"] in ["completed", "partially_completed", "failed", "rejected", "cancelled"]:
                    break
                assert time.monotonic() < until
                time.sleep(0.5)
            assert r["status"] in ["completed", "partially_completed"], {
                "status": r["status"],
                "error": r["error_code"],
                "events": get("/runs/" + rid + "/events")["events"][-5:],
            }
            return r, tools

        def turn(row, intent, targets, mode="adaptive", source=None):
            body = {"goal": {"intent": intent, "target_label_ids": targets, "analysis_mode": mode}}
            if source:
                body["context"] = {
                    "source_report_id": source,
                    "reference": {"scope": "explicit", "source": "viewed_report"},
                }
            return post("/sessions/" + row["session_id"] + "/turns", body)["run_id"]

        def upload(case, name, targets):
            p = ROOT / "data/public_cpr/rgb-acceptance-v1/records" / case / "video.mp4"
            with p.open("rb") as f:
                asset = (
                    c.post(
                        "/api/assets", files={"file": ("m4-" + name + "-" + case + ".mp4", f, "video/mp4")}
                    )
                    .raise_for_status()
                    .json()
                )
            clip = post("/clips", {"asset_id": asset["id"]})
            session = post("/sessions", {"clip_id": clip["id"]})
            row = {
                "session_id": session["id"],
                "clip_id": clip["id"],
                "asset_id": asset["id"],
                "name": asset["name"],
                "public_case": case,
            }
            row["run_id"] = turn(row, "analyse", targets)
            return row

        caps = get("/capabilities")
        assert (
            caps["module_slots"]["M4"]["kind"] == "improved"
            and caps["module_slots"]["M5"]["kind"] == "baseline"
            and caps["llm"]["mode"] == "disabled"
        )
        if "body" not in state:
            state["body"] = upload("SupC04--r00--ch0", "body", [4])
            save()
        body = state["body"]
        r, tools = wait(body["run_id"])
        p = r["response"]["prediction"]
        a = r["response"]["module_assessment"]
        assert (
            r["status"] == "completed"
            and p["analysis_route"] == "body-qualified-v1"
            and p["supported_label_ids"] == [3, 4, 8, 11]
        )
        assert [t["request"]["tool_id"] for t in tools] == [
            "pose.extract",
            "skeleton.predict",
            "fusion.predict",
        ]
        assert [x["label_id"] for x in r["response"]["classification_claims"]] == [4]
        assert r["response"]["scheduling"]["plan"]["profile_id"] == "body-qualified-v1"
        assert all(t["status"] == "succeeded" and t["output"]["is_stub"] is False for t in tools)
        # Truth/reference opened only after live output commit.
        oracle = json.loads((ROOT / "data/m3-calibration-v1/raw26.json").read_text())["rows"]
        expected = [v for v in oracle if v["record_id"] == "SupC04/r00/ch0"]
        assert p["predicted_label_ids"] == [
            v["label_id"] for v in expected if v["label_id"] in [3, 4, 8, 11] and v["raw_flagged"]
        ]
        assert (
            abs(
                a["calibrated_probabilities"]["4"]
                - next(v["probability"] for v in expected if v["label_id"] == 4)
            )
            < 1e-7
        )
        assert (
            get("/reports/" + r["report_id"] + "/evidence")["status"] == "available"
            and get("/reports/" + r["report_id"] + "/explanation")["status"] == "available"
        )
        body.update(
            report_id=r["report_id"],
            states=a["states"],
            raw_flags=p["predicted_label_ids"],
            actual_tools=[t["request"]["tool_id"] for t in tools],
            actual_seconds=sum(t["duration_ms"] / 1000 for t in tools),
        )
        save()
        if "uncovered_run_id" not in body:
            body["uncovered_run_id"] = turn(body, "analyse", [5], "existing_only", body["report_id"])
            save()
        uncovered, utools = wait(body["uncovered_run_id"], 30)
        assert (
            uncovered["status"] == "partially_completed"
            and utools == []
            and uncovered["response"]["target_states"] == {"5": "not_evaluated"}
            and uncovered["response"]["classification_claims"] == []
        )
        if "expand_run_id" not in body:
            body["expand_run_id"] = turn(body, "analyse", LABELS, "adaptive", body["report_id"])
            save()
        expanded, etools = wait(body["expand_run_id"])
        assert expanded["status"] == "completed" and [t["request"]["tool_id"] for t in etools] == [
            "rgb.predict",
            "fusion.predict",
        ]
        assert (
            expanded["response"]["prediction"]["analysis_route"] == "full-v15"
            and expanded["response"]["prediction"]["supported_label_ids"] == LABELS
        )
        assert expanded["response"]["reused_result"] is False
        assert expanded["response"]["prediction"]["predicted_label_ids"] == [
            v["label_id"] for v in expected if v["raw_flagged"]
        ]
        for t in etools:
            assert t["request"]["input_bindings"]["reuse_source_execution_id"] == p["execution_id"]
        assert get("/reports/" + body["report_id"] + "/evidence")["status"] == "available"
        assert c.get("/api/reports/" + expanded["report_id"] + "/export").status_code == 200
        body.update(
            expand_report_id=expanded["report_id"],
            expand_tools=[t["request"]["tool_id"] for t in etools],
            expand_seconds=sum(t["duration_ms"] / 1000 for t in etools),
        )
        save()
        if "full" not in state:
            state["full"] = upload("SupC03--r00--ch0", "full", LABELS)
            save()
        full = state["full"]
        fr, ftools = wait(full["run_id"])
        assert [t["request"]["tool_id"] for t in ftools] == [
            "pose.extract",
            "skeleton.predict",
            "rgb.predict",
            "fusion.predict",
        ]
        expected = [v for v in oracle if v["record_id"] == "SupC03/r00/ch0"]
        assert fr["response"]["prediction"]["predicted_label_ids"] == [
            v["label_id"] for v in expected if v["raw_flagged"]
        ]
        assert fr["response"]["module_assessment"]["states"]["2"] == "uncertain"
        if "verify_run_id" not in full:
            full["verify_run_id"] = turn(full, "verify", [2], "adaptive", fr["report_id"])
            save()
        vr, vtools = wait(full["verify_run_id"], 30)
        assert vtools == [] and vr["response"]["target_states"]["2"] == "uncertain"
        assert vr["response"]["scheduling"]["stop"]["stop_code"] == "NO_INDEPENDENT_INFORMATION"
        if "summary_run_id" not in full:
            full["summary_run_id"] = turn(full, "summarise", LABELS, "existing_only", fr["report_id"])
            save()
        sr, stools = wait(full["summary_run_id"], 30)
        assert stools == [] and sr["response"]["target_states"]["2"] == "uncertain"
        full.update(
            report_id=fr["report_id"],
            states=fr["response"]["module_assessment"]["states"],
            actual_tools=[t["request"]["tool_id"] for t in ftools],
            actual_seconds=sum(t["duration_ms"] / 1000 for t in ftools),
        )
        save()
        for row in state.values():
            for key, value in row.items():
                if key == "run_id" or key.endswith("_run_id"):
                    assert not any(
                        e["payload"].get("called") is True
                        for e in get("/runs/" + value + "/events")["events"]
                    )
        result = {
            "passed": True,
            "real_cold_videos": 2,
            "real_analyses": 3,
            "vision_tool_executions": 9,
            "provider_calls": 0,
            "body_route_skips_rgb": True,
            "scope_expansion_reuses_only_approved_pose_and_skeleton": True,
            "uncertain_covered_goal_does_not_repeat_tools": True,
            "existing_only_uncovered_is_not_negative": True,
            "frozen_requested_outputs_match": True,
            "parent_artifacts_preserved": True,
            "records": state,
        }
        (OUT / "acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            "M4 cold focused/full routes, cached expansion, uncovered scope and uncertainty STOP passed",
            flush=True,
        )


if __name__ == "__main__":
    main()
