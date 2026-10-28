"""Verify completed real M1 runs after restart, without running vision or an LLM."""

from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test-results/m1-optimization-v2/restart-acceptance.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserve restart acceptance")
    accepted = json.loads((OUTPUT.parent / "http-acceptance.json").read_text(encoding="utf-8"))
    assert accepted["passed"] and len(accepted["cases"]) == 1
    records = []
    with httpx.Client(base_url="http://127.0.0.1:8003", timeout=20) as client:

        def get(path):
            return client.get(path).raise_for_status().json()

        assert get("/api/health")["llm"] == "disabled"
        for case in accepted["cases"]:
            original = get(f"/api/runs/{case['run_id']}")
            assert original["status"] == "completed"
            assert original["response"]["prediction"]["predicted_label_ids"] == case["predicted_labels"]
            assert len(get(f"/api/runs/{case['run_id']}/tools")) == 4
            evidence = get(f"/api/reports/{case['report_id']}/evidence")
            assert evidence["status"] == "available"
            assert evidence["review"]["skeleton_processing"] == case["m1_processing"]
            snapshot = get(f"/api/reports/{case['report_id']}")
            history = get(f"/api/sessions/{case['session_id']}/history")
            assert case["run_id"] in [item["run_id"] for item in history["items"]]
            run_id = (
                client.post(
                    f"/api/sessions/{case['session_id']}/turns",
                    headers={"Idempotency-Key": str(uuid4())},
                    json={"goal": {"intent": "summarise", "analysis_mode": "existing_only"}},
                )
                .raise_for_status()
                .json()["run_id"]
            )
            deadline = time.monotonic() + 15
            while True:
                following = get(f"/api/runs/{run_id}")
                if following["status"] in {"completed", "failed", "cancelled", "rejected"}:
                    break
                assert time.monotonic() < deadline
                time.sleep(0.1)
            assert following["status"] == "completed" and following["response"]["reused_result"]
            assert following["response"]["prediction"] == original["response"]["prediction"]
            assert get(f"/api/runs/{run_id}/tools") == []
            assert get(f"/api/reports/{case['report_id']}") == snapshot
            media = client.get(f"/api/assets/{case['asset_id']}/media", headers={"Range": "bytes=0-1023"})
            assert media.status_code == 206 and len(media.content) == 1024
            exported = client.get(f"/api/reports/{case['report_id']}/export").raise_for_status()
            assert "skeleton_processing" in exported.text
            records.append(
                {
                    "case": case["case"],
                    "passed": True,
                    "summary_run_id": run_id,
                    "source_unchanged": True,
                    "evidence_revalidated": True,
                    "range_playback": True,
                    "export": True,
                    "new_vision_calls": 0,
                }
            )
    OUTPUT.write_text(
        json.dumps({"passed": True, "cases": records, "real_llm_calls": 0}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"passed": True, "recovered_cases": len(records), "new_vision_calls": 0}))


if __name__ == "__main__":
    main()
