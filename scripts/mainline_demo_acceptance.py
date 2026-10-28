"""Append real public-video demonstration records to the current accepted profile."""

import json
from pathlib import Path

import httpx
from demo_scenarios import DemoClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mainline-delivery-v1"
STATE = OUT / "demo-state.json"
ALL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]


def save(state):
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    with httpx.Client(base_url="http://127.0.0.1:8013", timeout=60, trust_env=False) as c:
        api = DemoClient(c)
        assert api.get("/api/health")["llm"] == "disabled"

        def workspace(key, sample, name):
            if key not in state:
                source = ROOT / "data/public_cpr/rgb-acceptance-v1/records" / sample / "video.mp4"
                state[key] = {
                    "workspace": api.upload(name, source.read_bytes()),
                    "source": str(source.relative_to(ROOT)),
                }
                save(state)
            return state[key]["workspace"]

        def run(key, ws, labels, intent="analyse", source=None):
            if key not in state:
                rid = api.submit(ws, intent=intent, labels=labels, report=source)
                state[key] = {"run_id": rid}
                save(state)
            rid = state[key]["run_id"]
            result = api.wait(rid, seconds=900)
            assert result["status"] == "completed", (key, result["status"], result.get("error_code"))
            state[key].update(
                report_id=result["report_id"], tools=[t["request"]["tool_id"] for t in api.get(f"/api/runs/{rid}/tools")]
            )
            save(state)
            return result

        one = workspace("D01", "SupC04--r00--ch0", "Demo 01 - Bent arms.mp4")
        full = run("D01-full", one, ALL)
        assert state["D01-full"]["tools"] == [
            "pose.extract",
            "skeleton.predict",
            "rgb.predict",
            "fusion.predict",
        ]
        explanation = run("D01-explain", one, [4], "explain", full["report_id"])
        recap = run("D01-recap", one, [4], "summarise", explanation["report_id"])
        reference = run("D01-reference", one, [4], "show_example", recap["report_id"])
        refs = api.get(f"/api/runs/{reference['id']}/references")
        assert refs["status"] == "available" and len(refs["items"]) >= 2
        state["references"] = refs
        if "D01-notebook" not in state:
            note = api.get(f"/api/sessions/{one['session_id']}/notebook")
            state["D01-notebook"] = api.post(
                f"/api/sessions/{one['session_id']}/notebook",
                {
                    "revision": note["revision"],
                    "note": "Next practice: review arm position, then compare the saved evidence.",
                    "items": [{"label_id": 4, "status": "next_focus"}],
                },
            )
            save(state)

        two = workspace("D02", "SupC03--r00--ch0", "Demo 02 - Review needed.mp4")
        if "D02-focus" not in state:
            state["D02-focus"] = api.post(
                f"/api/sessions/{two['session_id']}/continuity",
                {"revision": 0, "source_report_id": recap["report_id"]},
            )
            ctx = api.get(f"/api/sessions/{two['session_id']}/context")
            assert ctx["source"] is None and ctx["active_label_ids"] == [4]
            state["D02-context-before-analysis"] = ctx
            save(state)
        body = run("D02-body", two, [4])
        assert state["D02-body"]["tools"] == ["pose.extract", "skeleton.predict", "fusion.predict"]
        expanded = run("D02-full", two, ALL, source=body["report_id"])
        assert state["D02-full"]["tools"] == ["rgb.predict", "fusion.predict"]
        comparison = api.get(
            f"/api/review-comparison?left_report_id={full['report_id']}&right_report_id={expanded['report_id']}"
        )
        assert comparison["common_label_ids"] == ALL
        state["comparison"] = comparison

        three = workspace("D03", "SupC05--r00--ch0", "Demo 03 - Next practice.mp4")
        if "D03-focus" not in state:
            state["D03-focus"] = api.post(
                f"/api/sessions/{three['session_id']}/continuity",
                {
                    "revision": 0,
                    "source_notebook_session_id": one["session_id"],
                    "source_notebook_revision": state["D01-notebook"]["revision"],
                },
            )
            save(state)
        ctx = api.get(f"/api/sessions/{three['session_id']}/context")
        assert ctx["source"] is None and ctx["active_label_ids"] == [4]
        state["D03-context"] = ctx
        negatives = []
        for name in ["case-04-invalid-container.mp4", "case-05-too-short-3frames.mp4"]:
            content = (ROOT / "data/demo-cases-v1/controls" / name).read_bytes()
            response = c.post("/api/assets", files={"file": (name, content, "video/mp4")})
            assert response.status_code in {415, 422}, response.text
            negatives.append(
                {
                    "file": name,
                    "status": response.status_code,
                    "error": response.json()["error"]["error_code"],
                }
            )
        state["negative_inputs"] = negatives
        state["passed"] = True
        state["new_cv_calls"] = sum(len(state[k]["tools"]) for k in ["D01-full", "D02-body", "D02-full"])
        save(state)
        print(
            json.dumps(
                {
                    "passed": True,
                    "new_cv_calls": state["new_cv_calls"],
                    "new_manual_labels": False,
                    "new_training": False,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
