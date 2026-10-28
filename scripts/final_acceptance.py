"""Run the no-inference final product acceptance gate.

This gate checks the running service, frozen mainline evidence, demo cases,
negative input evidence, and UI acceptance records. It deliberately does not
start a vision tool or an LLM request.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mainline-delivery-v1"
MANIFEST = OUT / "delivery-manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    acceptance = manifest["acceptance"]
    files = manifest["files"]

    # The runtime gate is read-only and must stay in the default safe mode.
    with httpx.Client(base_url="http://127.0.0.1:8013", timeout=10, trust_env=False) as client:
        health = client.get("/api/health")
        capabilities = client.get("/api/capabilities")
        require(health.status_code == 200, f"health returned {health.status_code}")
        require(capabilities.status_code == 200, f"capabilities returned {capabilities.status_code}")
        h = health.json()
        c = capabilities.json()
        require(h.get("status") == "ok", "service is not healthy")
        require(h.get("llm") == "disabled", "default runtime must keep LLM disabled")
        require(h.get("graph_version") == manifest["graph_version"], "graph version drift")
        require(h.get("bundle_sha256") == manifest["model_bundle_sha256"], "model bundle drift")
        require(h.get("vision_pipeline") == "branches", "unexpected vision pipeline")
        require(c.get("context_binding") == "bound-v1", "context binding drift")
        for case in ("D01", "D02", "D03"):
            response = client.get(f"/demo/open/{case}")
            require(response.status_code == 200, f"demo case {case} returned {response.status_code}")

    # Frozen contract, existing UI acceptance, and public-data boundary.
    handoff = OUT / "module-contracts.json"
    require(handoff.exists(), "module handoff contract is missing")
    demo_state = json.loads((OUT / "demo-state.json").read_text(encoding="utf-8"))
    require(demo_state.get("passed") is True, "demo state is not marked passed")
    negatives = demo_state.get("negative_inputs", [])
    require(len(negatives) >= 2, "negative video cases are missing")
    require(all(item.get("status") in (415, 422) for item in negatives), "negative case status drift")
    ui = json.loads((OUT / "mainline-ui-acceptance.json").read_text(encoding="utf-8"))
    deepseek_ui = json.loads((OUT / "mainline-deepseek-ui-acceptance.json").read_text(encoding="utf-8"))
    require(ui.get("passed") is True, "offline UI acceptance failed")
    require(deepseek_ui.get("passed") is True, "DeepSeek UI acceptance failed")
    require(ui.get("browser_errors") == [], "offline UI has browser errors")
    require(deepseek_ui.get("browser_errors") == [], "DeepSeek UI has browser errors")
    require(ui.get("external_browser_requests") == 0, "offline UI made external requests")
    require(deepseek_ui.get("external_browser_requests") == 0, "DeepSeek UI made external requests")
    require(acceptance.get("new_training") is False, "new training was introduced")
    require(acceptance.get("new_manual_labels") is False, "new manual labels were introduced")

    # The two user-facing delivery documents are included in the manifest.
    for rel in ("docs/当前主线-演示与交付.html", "docs/当前主线-启动与交付.txt"):
        require(rel in files, f"delivery manifest does not track {rel}")
        require(sha256(ROOT / rel) == files[rel], f"delivery document hash drift: {rel}")

    result = {
        "passed": True,
        "graph_version": manifest["graph_version"],
        "default_llm_mode": "disabled",
        "demo_cases": ["D01", "D02", "D03"],
        "negative_video_cases": len(negatives),
        "offline_ui_browser_errors": len(ui.get("browser_errors", [])),
        "deepseek_ui_browser_errors": len(deepseek_ui.get("browser_errors", [])),
        "offline_ui_external_requests": ui.get("external_browser_requests"),
        "deepseek_ui_external_requests": deepseek_ui.get("external_browser_requests"),
        "new_training": False,
        "new_manual_labels": False,
        "vision_calls_started": 0,
        "llm_calls_started": 0,
    }
    output = OUT / "final-acceptance.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)
