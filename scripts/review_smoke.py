"""HTTP acceptance for saved real-video evidence and history, without new inference or LLM calls."""

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
    parser.add_argument(
        "--source-acceptance", type=Path, default=ROOT / "test-results/web-real-v1/http-acceptance.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve previous acceptance records"
    source = json.loads(args.source_acceptance.read_text(encoding="utf-8"))
    sid, original_run_id, report_id = source["session_id"], source["run_id"], source["report_id"]
    started = time.monotonic()
    with httpx.Client(base_url=args.base_url, timeout=20) as client:

        def get(path):
            return client.get(path).raise_for_status().json()

        health = get("/api/health")
        assert (
            health["status"] == "ok"
            and health["vision"] == "v15-experimental"
            and health["llm"] == "disabled"
        )
        original_report = get(f"/api/reports/{report_id}")
        original = get(f"/api/runs/{original_run_id}")
        assert original["status"] == "completed"
        prediction = original["response"]["prediction"]
        recovered = next(s for s in get("/api/workspaces") if s["session_id"] == sid)
        assert recovered["bundle_compatible"] and recovered["clip_id"] == source["clip_id"]
        evidence = get(f"/api/reports/{report_id}/evidence")
        assert evidence["status"] == "available", evidence
        assert evidence["review"]["source_execution_id"] == prediction["execution_id"]
        assert evidence["review"]["video_sha256"] == prediction["video_sha256"]
        followups = []
        for intent, scope in [("explain", [2, 4]), ("summarise", [])]:
            response = (
                client.post(
                    f"/api/sessions/{sid}/turns",
                    headers={"Idempotency-Key": str(uuid4())},
                    json={
                        "goal": {
                            "intent": intent,
                            "target_label_ids": scope,
                            "analysis_mode": "existing_only",
                        }
                    },
                )
                .raise_for_status()
                .json()
            )
            rid = response["run_id"]
            until = time.monotonic() + 15
            while True:
                run = get(f"/api/runs/{rid}")
                if run["status"] in {"completed", "failed", "rejected", "cancelled", "partially_completed"}:
                    break
                assert time.monotonic() < until, run
                time.sleep(0.1)
            assert run["status"] == "completed", run
            result = run["response"]
            assert result["reused_result"] and result["prediction"] == prediction
            assert get(f"/api/runs/{rid}/tools") == []
            assert result["review"]["source_run_id"] == original_run_id
            assert result["review"]["displayed_label_ids"] == (scope or prediction["supported_label_ids"])
            reopened = get(f"/api/reports/{run['report_id']}/evidence")
            assert reopened["status"] == "available" and reopened["review"] == result["review"]
            exported = client.get(f"/api/reports/{run['report_id']}/export").raise_for_status()
            assert "&quot;source_execution_id&quot;" in exported.text
            followups.append(
                {"intent": intent, "run_id": rid, "report_id": run["report_id"], "tool_calls": 0}
            )
        entries, cursor = [], None
        while True:
            page = (
                client.get(
                    f"/api/sessions/{sid}/history",
                    params={"limit": 2, **({"before_turn_id": cursor} if cursor else {})},
                )
                .raise_for_status()
                .json()
            )
            entries.extend(page["items"])
            if not page["has_more"]:
                break
            cursor = page["next_before_turn_id"]
        assert len({entry["turn_id"] for entry in entries}) == len(entries)
        assert {original_run_id, *(f["run_id"] for f in followups)} <= {e["run_id"] for e in entries}
        assert get(f"/api/reports/{report_id}") == original_report
        media = client.get(
            f"/api/assets/{source['asset_id']}/media", headers={"Range": "bytes=0-1023"}
        ).raise_for_status()
        assert media.status_code == 206 and len(media.content) == 1024
        assert get("/api/health")["llm"] == "disabled"
    record = {
        "passed": True,
        "scope": "Saved actual-video result reuse; no new model-accuracy evaluation",
        "source_run_id": original_run_id,
        "source_report_id": report_id,
        "session_id": sid,
        "source_execution_id": prediction["execution_id"],
        "video_sha256": prediction["video_sha256"],
        "bundle_sha256": prediction["bundle_sha256"],
        "history_entries": len(entries),
        "history_pagination_passed": True,
        "source_snapshot_unchanged": True,
        "source_playback_passed": True,
        "followups": followups,
        "new_inference_calls": 0,
        "llm_mode": "disabled",
        "real_llm_calls": 0,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
