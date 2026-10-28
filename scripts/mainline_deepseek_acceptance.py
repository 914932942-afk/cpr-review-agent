"""Bounded live checks on saved current-profile evidence; never sends video to the provider."""

import json
from pathlib import Path
from uuid import uuid4

import httpx
from demo_scenarios import TERMINAL, DemoClient
from run_main import provider_values

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mainline-delivery-v1"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-live", action="store_true")
    args = parser.parse_args()
    assert args.allow_live, "Explicit live mode required"
    target = OUT / "deepseek-live.json"
    assert not target.exists(), "Preserve earlier real-call acceptance; do not repeat it automatically"
    state = json.loads((OUT / "demo-state.json").read_text(encoding="utf-8"))
    provider = provider_values(ROOT / ".env", {})
    base = provider.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    assert base in {"https://api.deepseek.com", "https://api.deepseek.com/v1"}
    # One read-only official availability query; no model list or credential is persisted.
    with httpx.Client(timeout=15, trust_env=False) as official:
        response = official.get(
            base + "/models", headers={"Authorization": "Bearer " + provider["DEEPSEEK_API_KEY"]}
        )
        assert response.status_code == 200, "Official model availability check failed"
        models = response.json().get("data", [])
        assert provider["DEEPSEEK_MODEL"] in {m["id"] for m in models}, "Configured model is not listed"
    result = {
        "model_list_verified": True,
        "model": provider["DEEPSEEK_MODEL"],
        "cases": [],
        "chat_requests": 0,
        "new_vision_calls": 0,
    }

    def save():
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    with httpx.Client(base_url="http://127.0.0.1:8013", timeout=45, trust_env=False) as c:
        api = DemoClient(c)
        assert api.get("/api/health")["llm"] == "deepseek"
        ws = state["D01"]["workspace"]
        original = api.get("/api/runs/" + state["D01-full"]["run_id"])

        def execute(name, message, report, *, intent=None, labels=None, clarify=False, duplicate=False):
            assert result["chat_requests"] + (1 if clarify else 2) <= 10
            body = {
                "message": message,
                "context": {
                    "source_report_id": report,
                    "reference": {"scope": "explicit", "source": "viewed_report"},
                },
            }
            path = f"/api/sessions/{ws['session_id']}/turns"
            key = str(uuid4())
            rid = api.post(path, body, key=key)["run_id"]
            run = api.wait(
                rid,
                seconds=60,
                predicate=lambda r: (
                    r["status"] in TERMINAL
                    or (r["status"] == "waiting_user" and (r["wait"] or {}).get("interrupt_id"))
                ),
            )
            turn = api.get("/api/turns/" + run["turn_id"])
            resolution = turn.get("llm_resolution") or {}
            checks = {
                "goal_called": resolution.get("called") is True,
                "no_cv": api.get(f"/api/runs/{rid}/tools") == [],
                "bound_source": turn["context_snapshot"]["source"]["report_id"] == report,
            }
            if clarify:
                checks["asks_confirmation"] = run["status"] == "waiting_user" and run["response"] is None
                if checks["asks_confirmation"]:
                    wait = run["wait"]
                    api.post(
                        f"/api/runs/{rid}/resume",
                        {
                            "wait_id": wait["wait_id"],
                            "interrupt_id": wait["interrupt_id"],
                            "revision": run["revision"],
                            "goal": {"intent": "explain", "target_label_ids": [4]},
                            "context_confirmation": {
                                "snapshot_id": wait["context_snapshot_id"],
                                "use_bound_source": True,
                            },
                        },
                    )
                    run = api.wait(rid, seconds=45)
                    checks["confirmed_scope"] = run["status"] == "completed" and run["grounded_review"][
                        "scope_label_ids"
                    ] == [4]
            else:
                checks["completed"] = run["status"] == "completed"
                checks["intent"] = (turn.get("resolved_goal") or {}).get("intent") == intent
                if intent != "show_example":
                    checks["scope"] = (run.get("grounded_review") or {}).get("scope_label_ids") == labels
            if intent != "show_example":
                checks["prediction_unchanged"] = (run.get("response") or {}).get("prediction") == original[
                    "response"
                ]["prediction"]
                checks["grounded_plan"] = (run.get("grounded_review") or {}).get(
                    "organisation_receipt", {}
                ).get("code") == "LLM_PLAN_VALIDATED"
            else:
                checks["reference_available"] = (
                    api.get(f"/api/runs/{rid}/references")["status"] == "available"
                )
            if duplicate:
                checks["duplicate_same_run"] = api.post(path, body, key=key)["run_id"] == rid
            events = api.get(f"/api/runs/{rid}/events")["events"]
            checks["one_goal_event"] = sum(e["kind"] == "llm.goal_resolution" for e in events) == 1
            turn = api.get("/api/turns/" + run["turn_id"])
            calls = int(bool(resolution.get("called"))) + int(
                bool((turn.get("explanation_resolution") or {}).get("called"))
            )
            result["chat_requests"] += calls
            record = {
                "name": name,
                "message": message,
                "run_id": rid,
                "report_id": run.get("report_id"),
                "status": run["status"],
                "checks": checks,
                "passed": all(checks.values()),
                "chat_requests": calls,
                "goal_receipt": resolution,
                "organisation_receipt": (run.get("grounded_review") or {}).get("organisation_receipt"),
            }
            result["cases"].append(record)
            save()
            print(
                json.dumps(
                    {"case": name, "passed": record["passed"], "checks": checks, "chat_requests": calls}
                ),
                flush=True,
            )
            assert record["passed"], "Live case did not pass; inspect sanitized acceptance before any repeat"
            return run

        explicit = execute(
            "explicit-zh",
            "请只解释这段训练视频中手臂弯曲这一项的已有结果，不要重新分析视频。",
            original["report_id"],
            intent="explain",
            labels=[4],
            duplicate=True,
        )
        execute(
            "bound-reference",
            "那这个问题的已有依据是什么？",
            explicit["report_id"],
            intent="explain",
            labels=[4],
        )
        execute(
            "summary-en",
            "Summarise the saved findings for the same categories; do not rerun analysis.",
            explicit["report_id"],
            intent="summarise",
            labels=[4],
        )
        execute("ambiguous-then-confirm", "解释一下这个问题。", original["report_id"], clarify=True)
        execute(
            "reference-request",
            "给我看一个手臂弯曲的参考视频，不要重新分析。",
            explicit["report_id"],
            intent="show_example",
            labels=[4],
        )
        result.update(
            passed=True,
            workspace=ws,
            full_run_id=original["id"],
            single_run_id=explicit["id"],
            single_report_id=explicit["report_id"],
            model_status=api.get("/api/capabilities")["llm"],
        )
        assert result["chat_requests"] <= 10
        save()
        print(
            json.dumps({"passed": True, "chat_requests": result["chat_requests"], "new_vision_calls": 0}),
            flush=True,
        )


if __name__ == "__main__":
    main()
