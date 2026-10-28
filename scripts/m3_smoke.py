"""Cold video HTTP acceptance and follow-up uncertainty preservation; no mocked inference."""

import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test-results/m3-calibration-v1/http"
LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "acceptance.json").exists():
        raise FileExistsError("Preserve M3 HTTP acceptance")
    state_path = OUT / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def save():
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    with httpx.Client(base_url="http://127.0.0.1:8010", timeout=60) as c:

        def get(path):
            return c.get("/api" + path).raise_for_status().json()

        def post(path, data):
            return (
                c.post("/api" + path, json=data, headers={"Idempotency-Key": str(uuid4())})
                .raise_for_status()
                .json()
            )

        def wait(rid, deadline=900):
            until = time.monotonic() + deadline
            old = None
            while True:
                run = get("/runs/" + rid)
                tools = get("/runs/" + rid + "/tools")
                marker = (
                    run["status"] + ":" + ",".join(t["request"]["tool_id"] + "=" + t["status"] for t in tools)
                )
                if marker != old:
                    print(rid, marker, flush=True)
                    old = marker
                if run["status"] in ["completed", "failed", "cancelled", "rejected", "partially_completed"]:
                    break
                assert time.monotonic() < until
                time.sleep(0.5)
            assert run["status"] == "completed", run
            return run, tools

        caps = get("/capabilities")
        assert (
            caps["llm"]["mode"] == "disabled"
            and caps["module_slots"]["M3"]["implementation_id"] == "source-quality-selective-v1"
        )
        for case in ["SupC03--r00--ch0", "SupC00--r00--ch0"]:
            if case not in state:
                path = ROOT / "data/public_cpr/rgb-acceptance-v1/records" / case / "video.mp4"
                with path.open("rb") as f:
                    asset = (
                        c.post("/api/assets", files={"file": ("m3-" + case + ".mp4", f, "video/mp4")})
                        .raise_for_status()
                        .json()
                    )
                clip = post("/clips", {"asset_id": asset["id"]})
                session = post("/sessions", {"clip_id": clip["id"]})
                rid = post(
                    "/sessions/" + session["id"] + "/turns",
                    {"goal": {"intent": "analyse", "target_label_ids": LABELS}},
                )["run_id"]
                state[case] = {
                    "asset_id": asset["id"],
                    "clip_id": clip["id"],
                    "session_id": session["id"],
                    "run_id": rid,
                    "name": asset["name"],
                }
                save()
            row = state[case]
            run, tools = wait(row["run_id"])
            assert [t["request"]["tool_id"] for t in tools] == [
                "pose.extract",
                "skeleton.predict",
                "rgb.predict",
                "fusion.predict",
            ]
            assert all(t["status"] == "succeeded" and t["output"]["is_stub"] is False for t in tools)
            response = run["response"]
            assessment = response["module_assessment"]
            assert (
                assessment["calibration_profile"] == "source-quality-selective-v1"
                and assessment["calibration_status"] == "calibrated"
            )
            # Open the previously frozen engineering oracle only after live inference commits.
            oracle = json.loads((ROOT / "data/m3-calibration-v1/raw26.json").read_text())
            expected = [r for r in oracle["rows"] if r["record_id"] == case.replace("--", "/")]
            assert len(expected) == 11
            for value in expected:
                label = str(value["label_id"])
                assert assessment["states"][label] == value["state"]
                assert abs(assessment["calibrated_probabilities"][label] - value["probability"]) <= 1e-7
            assert response["prediction"]["predicted_label_ids"] == [
                r["label_id"] for r in expected if r["raw_flagged"]
            ]
            assert get("/reports/" + run["report_id"] + "/evidence")["status"] == "available"
            explanation = get("/reports/" + run["report_id"] + "/explanation")
            assert explanation["status"] == "available"
            if case.startswith("SupC03"):
                assert assessment["states"]["2"] == "uncertain"
                assert (
                    next(e for e in explanation["review"]["entries"] if e["label_id"] == 2)["review_state"]
                    == "uncertain"
                )
            else:
                assert not response["prediction"]["predicted_label_ids"]
            assert c.get("/api/reports/" + run["report_id"] + "/export").status_code == 200
            followups = []
            for intent, scope in [("explain", [2]), ("summarise", LABELS)]:
                key = intent + "_run_id"
                if key not in row:
                    row[key] = post(
                        "/sessions/" + row["session_id"] + "/turns",
                        {
                            "goal": {
                                "intent": intent,
                                "analysis_mode": "existing_only",
                                "target_label_ids": scope,
                            },
                            "context": {
                                "source_report_id": run["report_id"],
                                "reference": {"scope": "explicit", "source": "viewed_report"},
                            },
                        },
                    )["run_id"]
                    save()
                follow, ftools = wait(row[key], 30)
                assert ftools == [] and follow["response"]["reused_result"]
                assert (
                    follow["response"]["module_assessment"]["calibration_profile"]
                    == "source-quality-selective-v1"
                )
                if case.startswith("SupC03"):
                    assert follow["response"]["target_states"]["2"] == "uncertain"
                assert get("/reports/" + follow["report_id"] + "/explanation")["status"] == "available"
                followups.append(follow["id"])
            for rid in [run["id"], *followups]:
                assert not any(
                    e["payload"].get("called") is True for e in get("/runs/" + rid + "/events")["events"]
                )
            row.update(
                report_id=run["report_id"],
                raw_flags=response["prediction"]["predicted_label_ids"],
                states=assessment["states"],
                probabilities=assessment["calibrated_probabilities"],
                tools=[{"id": t["request"]["tool_id"], "ms": t["duration_ms"]} for t in tools],
            )
            save()
        (OUT / "acceptance.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "cold_videos": 2,
                    "real_vision_executions": 8,
                    "provider_calls": 0,
                    "uncertainty_retained_in_followups": True,
                    "raw_cv_decisions_unchanged": True,
                    "records": list(state.values()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "M3 real-video, calibrated/uncertain review, exports and no-vision follow-ups passed", flush=True
        )


if __name__ == "__main__":
    main()
