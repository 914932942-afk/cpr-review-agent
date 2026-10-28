"""Real HTTP replay of existing public-video results; no new CV, training or LLM calls."""

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
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    parser.add_argument("--phase", choices=["prepare", "resume"], required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / ("prepared.json" if args.phase == "prepare" else "acceptance.json")
    assert not target.exists(), "Preserve previous acceptance results"
    source = json.loads(
        (ROOT / "test-results/mainline-product-v1/ui-acceptance.json").read_text(encoding="utf-8")
    )
    sid, original_id = source["workspace"]["session_id"], source["run_id"]
    with httpx.Client(base_url=args.base_url, timeout=30) as client:

        def get(path):
            return client.get(path).raise_for_status().json()

        def post(body):
            return (
                client.post(
                    f"/api/sessions/{sid}/turns", json=body, headers={"Idempotency-Key": str(uuid4())}
                )
                .raise_for_status()
                .json()["run_id"]
            )

        def wait(rid, paused=False):
            until = time.monotonic() + 25
            while time.monotonic() < until:
                run = get(f"/api/runs/{rid}")
                if (
                    paused and run["status"] == "waiting_user" and (run["wait"] or {}).get("interrupt_id")
                ) or (not paused and run["status"] in {"completed", "rejected", "failed", "cancelled"}):
                    return run
                time.sleep(0.1)
            raise AssertionError(run)

        def followup(report, targets, scope="explicit"):
            run = wait(
                post(
                    {
                        "goal": {"intent": "explain", "target_label_ids": targets},
                        "context": {
                            "source_report_id": report,
                            "reference": {"scope": scope, "source": "viewed_report"},
                        },
                    }
                )
            )
            assert run["status"] == "completed", run
            assert get(f"/api/runs/{run['id']}/tools") == []
            assert run["response"]["reused_result"]
            assert run["response"]["prediction"] == original["response"]["prediction"]
            return run

        health = get("/api/health")
        assert (
            health["status"] == "ok"
            and health["llm"] == "disabled"
            and health["context_binding"] == "bound-v1"
        )
        original = get(f"/api/runs/{original_id}")
        assert original["status"] == "completed"
        if args.phase == "prepare":
            broad = followup(original["report_id"], [2, 4])
            narrow = followup(broad["report_id"], [4])
            inherited = followup(broad["report_id"], [], "previous_scope")
            assert inherited["response"]["review"]["displayed_label_ids"] == [2, 4]
            assert (
                inherited["review_context"]["source_report_id"] == broad["report_id"] != narrow["report_id"]
            )
            rid = post(
                {
                    "goal": {"intent": "explain"},
                    "context": {
                        "source_report_id": broad["report_id"],
                        "reference": {"scope": "single_finding", "source": "viewed_report"},
                    },
                }
            )
            paused = wait(rid, True)
            assert paused["wait"]["context_code"] == "CONTEXT_SCOPE_AMBIGUOUS"
            turn = get(f"/api/turns/{paused['turn_id']}")
            assert turn.get("resolved_goal") is None
            data = {
                "passed": True,
                "session_id": sid,
                "original_run_id": original_id,
                "broad_run_id": broad["id"],
                "broad_report_id": broad["report_id"],
                "narrow_run_id": narrow["id"],
                "inherited_run_id": inherited["id"],
                "pending_run_id": rid,
                "pending_turn_id": paused["turn_id"],
                "snapshot": turn["context_snapshot"],
                "wait": paused["wait"],
                "requires_server_restart_before_resume": True,
                "new_vision_calls": 0,
                "real_llm_calls": 0,
            }
        else:
            prepared = json.loads((args.output_dir / "prepared.json").read_text(encoding="utf-8"))
            paused = wait(prepared["pending_run_id"], True)
            assert get(f"/api/turns/{paused['turn_id']}")["context_snapshot"] == prepared["snapshot"]
            assert paused["wait"]["context_snapshot_id"] == prepared["wait"]["context_snapshot_id"]
            body = {
                "wait_id": paused["wait"]["wait_id"],
                "interrupt_id": paused["wait"]["interrupt_id"],
                "revision": paused["revision"],
                "goal": {"intent": "explain", "target_label_ids": [4]},
                "context_confirmation": {
                    "snapshot_id": paused["wait"]["context_snapshot_id"],
                    "use_bound_source": True,
                },
            }
            client.post(
                f"/api/runs/{paused['id']}/resume", json=body, headers={"Idempotency-Key": str(uuid4())}
            ).raise_for_status()
            done = wait(paused["id"])
            assert done["status"] == "completed", done
            assert done["response"]["prediction"] == original["response"]["prediction"]
            assert done["response"]["review"]["displayed_label_ids"] == [4]
            assert get(f"/api/runs/{done['id']}/tools") == []
            assert get(f"/api/reports/{done['report_id']}")["review_context"] == done["review_context"]
            assert get(f"/api/reports/{done['report_id']}/evidence")["status"] == "available"
            for rid in [
                prepared["broad_run_id"],
                prepared["narrow_run_id"],
                prepared["inherited_run_id"],
                done["id"],
            ]:
                assert get(f"/api/runs/{rid}/tools") == []
            data = {
                "passed": True,
                "mode": "actual_http_saved_real_video_context_and_restart",
                "session_id": sid,
                "original_run_id": original_id,
                "resumed_run_id": done["id"],
                "report_id": done["report_id"],
                "snapshot_preserved": True,
                "older_report_scope_preserved": True,
                "source_report_id": done["review_context"]["source_report_id"],
                "target_label_ids": [4],
                "new_vision_calls": 0,
                "real_llm_calls": 0,
                "new_training": False,
            }
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "passed": True,
                    "phase": args.phase,
                    "output": str(target),
                    "new_vision_calls": 0,
                    "real_llm_calls": 0,
                }
            )
        )


if __name__ == "__main__":
    main()
