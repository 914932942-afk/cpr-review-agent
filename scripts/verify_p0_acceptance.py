"""Verify the P0 user-flow guarantees without running inference.

The current frozen product already implements these behaviors in the workspace
frontend. This check keeps them visible as an explicit delivery gate while
leaving the model and pipeline artifacts untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "workspaces/m5-cascade-v1/frontend/src/App.tsx"
EVIDENCE = ROOT / "workspaces/m5-cascade-v1/frontend/src/EvidencePanel.tsx"
HISTORY = ROOT / "workspaces/m5-cascade-v1/frontend/src/HistoryPanel.tsx"
MANIFEST = ROOT / "data/mainline-delivery-v1/delivery-manifest.json"
OUT = ROOT / "data/mainline-delivery-v1/p0-acceptance.json"


def main() -> None:
    app = APP.read_text(encoding="utf-8")
    evidence = EVIDENCE.read_text(encoding="utf-8")
    history = HISTORY.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    checks = {
        "upload_metadata": all(
            token in app
            for token in (
                "fileCheck && fileCheck.asset_id",
                "fileCheck.decoded_frames",
                "fileCheck.duration_seconds",
                "fileCheck.dimensions_hw",
                "File compatibility only.",
            )
        ),
        "failure_recovery": all(
            token in app
            for token in (
                "M4_BUDGET_INSUFFICIENT",
                "Retry with more time",
                "Choose another video",
            )
        ) and "Review history" in history,
        "whole_clip_scope": all(
            token in app or token in evidence
            for token in (
                "Findings apply to the whole clip.",
                "the whole clip; it does not mark exact error times.",
            )
        ),
        "frozen_pipeline_contract": manifest["acceptance"].get("ux_polish_runtime_pipeline_unchanged") is True,
        "no_training_or_labels": manifest["acceptance"].get("new_training") is False
        and manifest["acceptance"].get("new_manual_labels") is False,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "metadata_fields": ["duration_seconds", "decoded_frames", "dimensions_hw"],
        "recovery_actions": ["retry_with_more_time", "choose_another_video", "open_review_history"],
        "evidence_scope": "whole_clip",
        "vision_calls_started": 0,
        "llm_calls_started": 0,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
