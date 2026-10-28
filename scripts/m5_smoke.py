"""Real API/video validation. No model mocks, labels at inference or provider calls."""

import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test-results/m5-cascade-v1/http"
LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]
BODY = [3, 4, 8, 11]
TERMINAL = {"completed", "partially_completed", "failed", "rejected", "cancelled"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "acceptance.json").exists()
    path = OUT / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {}

    def save():
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    with httpx.Client(base_url="http://127.0.0.1:8013", timeout=60) as c:

        def get(p):
            return c.get("/api" + p).raise_for_status().json()

        def post(p, v, key=None):
            return (
                c.post("/api" + p, json=v, headers={"Idempotency-Key": key or str(uuid4())})
                .raise_for_status()
                .json()
            )

        def turn(row, targets, seconds=900, partial=False, source=None, mode="adaptive"):
            body = {
                "goal": {
                    "intent": "analyse",
                    "target_label_ids": targets,
                    "analysis_mode": mode,
                    "allow_partial": partial,
                    "analysis_budget_seconds": seconds,
                }
            }
            if source:
                body["context"] = {
                    "source_report_id": source,
                    "reference": {"scope": "explicit", "source": "viewed_report"},
                }
            return post("/sessions/" + row["session_id"] + "/turns", body)["run_id"]

        def wait(rid):
            until = time.monotonic() + 950
            last = None
            while time.monotonic() < until:
                r = get("/runs/" + rid)
                tools = get("/runs/" + rid + "/tools")
                marker = (
                    r["status"] + ":" + ",".join(t["request"]["tool_id"] + "=" + t["status"] for t in tools)
                )
                if marker != last:
                    print(rid, marker, flush=True)
                    last = marker
                if r["status"] in TERMINAL:
                    return r, tools
                time.sleep(0.6)
            raise TimeoutError(rid)

        def create(name, case):
            p = ROOT / "data/public_cpr/rgb-acceptance-v1/records" / case / "video.mp4"
            with p.open("rb") as f:
                a = (
                    c.post("/api/assets", files={"file": ("m4v2-" + name + ".mp4", f, "video/mp4")})
                    .raise_for_status()
                    .json()
                )
            clip = post("/clips", {"asset_id": a["id"]})
            session = post("/sessions", {"clip_id": clip["id"]})
            return dict(
                session_id=session["id"], asset_id=a["id"], clip_id=clip["id"], name=a["name"], case=case
            )

        def case(name, record, targets, seconds, partial, expected):
            if name not in state:
                row = create(name, record)
                state[name] = row
                save()
                row["run_id"] = turn(row, targets, seconds, partial)
                save()
            row = state[name]
            r, tools = wait(row["run_id"])
            assert r["status"] == expected, (
                r.get("error_code"),
                get("/runs/" + row["run_id"] + "/events")["events"][-6:],
            )
            row.update(
                report_id=r["report_id"],
                tools=[t["request"]["tool_id"] for t in tools],
                seconds=sum(t.get("duration_ms", 0) / 1000 for t in tools),
            )
            save()
            return row, r, tools

        caps = get("/capabilities")
        assert (
            caps["module_slots"]["M5"]["implementation_id"] == "quality-dependency-cascade-v1"
            and caps["module_slots"]["M5"]["kind"] == "improved"
        )
        assert caps["module_slots"]["M4"]["implementation_id"] == "consent-coverage-dag-v2"
        rejected, rr, rt = case("complete-limited", "SupC04--r00--ch0", LABELS, 180, False, "rejected")
        assert not rt and rr["error_code"] == "M4_BUDGET_INSUFFICIENT"
        row, r, tools = case("partial", "SupC04--r00--ch0", LABELS, 180, True, "partially_completed")
        assert row["tools"] == ["pose.extract", "skeleton.predict", "fusion.predict"]
        assert r["response"]["prediction"]["supported_label_ids"] == BODY
        assert r["response"]["scheduling"]["partial_scope_authorised"] is True
        assert r["response"]["scheduling"]["stop"]["stop_code"] == "PARTIAL_SCOPE_DELIVERED"
        assert sorted(int(k) for k, v in r["response"]["target_states"].items() if v == "not_evaluated") == [
            1,
            2,
            5,
            6,
            7,
            9,
            13,
        ]
        assert sorted(v["label_id"] for v in r["response"]["classification_claims"]) == BODY
        hist = r["response"]["scheduling"]["history"]
        assert len(hist) == 4 and all(d["profile_id"] == "body-qualified-v1" for d in hist[:-1])
        assert all(t["request"]["tool_id"] != "rgb.predict" for t in tools)
        if "uncovered_run_id" not in row:
            row["uncovered_run_id"] = turn(row, [5], source=row["report_id"], mode="existing_only")
            save()
        u, ut = wait(row["uncovered_run_id"])
        assert not ut and u["response"]["target_states"] == {"5": "not_evaluated"}
        full, fr, ft = case("full", "SupC03--r00--ch0", LABELS, 900, False, "completed")
        assert full["tools"] == ["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"]
        assert fr["response"]["prediction"]["analysis_route"] == "full-v15"
        assert fr["response"]["target_states"]["2"] == "uncertain"
        assert len(fr["response"]["scheduling"]["history"]) == 5
        if "expand_run_id" not in row:
            row["expand_run_id"] = turn(row, LABELS, source=row["report_id"])
            save()
        ex, et = wait(row["expand_run_id"])
        assert ex["status"] == "completed"
        assert [t["request"]["tool_id"] for t in et] == ["rgb.predict", "fusion.predict"]
        assert ex["response"]["reused_result"] is False
        row["expand_report_id"] = ex["report_id"]
        save()
        # Compare current outputs to old accepted actual live raw receipts (not labels).
        oldstate = json.loads((ROOT / "test-results/m4-exploration-v2/http/state.json").read_text())
        with httpx.Client(base_url="http://127.0.0.1:8012", timeout=15) as previous:
            op = (
                previous.get("/api/runs/" + oldstate["partial"]["expand_run_id"])
                .raise_for_status()
                .json()["response"]
            )
            of = previous.get("/api/runs/" + oldstate["full"]["run_id"]).raise_for_status().json()["response"]
        for current, base, scope in [
            (r["response"], op, BODY),
            (ex["response"], op, LABELS),
            (fr["response"], of, LABELS),
        ]:
            for label in scope:
                x = next(s for s in current["prediction"]["scores"] if s["original_label_id"] == label)
                y = next(s for s in base["prediction"]["scores"] if s["original_label_id"] == label)
                assert x == y, (label, x, y)
                assert current["target_states"][str(label)] == base["target_states"][str(label)]
                assert (
                    current["module_assessment"]["calibrated_probabilities"][str(label)]
                    == base["module_assessment"]["calibrated_probabilities"][str(label)]
                )
        assert get("/reports/" + row["report_id"] + "/evidence")["status"] == "available"
        assert r["response"].get("visual_processing") is None
        for current, current_tools in [(fr, ft), (ex, et)]:
            summary = current["response"]["visual_processing"]
            assert (
                summary["mode"] == "lightweight"
                and summary["sampled_frames"] >= 8
                and summary["scope_unchanged"] is True
            )
            rgb = next(t for t in current_tools if t["request"]["tool_id"] == "rgb.predict")
            artifact = (
                ROOT / "workspaces/m5-cascade-v1/data/m5-cascade-v1/runtime" / rgb["output"]["artifact_rel"]
            )
            raw = json.loads(artifact.read_text(encoding="utf-8"))
            assert raw["m5_cascade"]["dense_visual_crop_evaluations"] == 0
            assert raw["module_executions"]["M5"]["implementation_id"] == "quality-dependency-cascade-v1"
        result = dict(
            passed=True,
            actual_video_analyses=3,
            actual_successful_tool_calls=9,
            m5_actual_dependency_pruning=True,
            m1_m2_m3_states_unchanged=True,
            default_no_consent_zero_tools=True,
        )
        result["partial_scope"] = BODY
        result.update(
            exact_old_scores_and_m3_states=True,
            unanalysed_not_negative=True,
            per_stage_replanning=True,
            no_automatic_scope_widening=True,
            partial_full_expansion_only_missing=True,
            provider_calls=0,
        )
        (OUT / "acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
