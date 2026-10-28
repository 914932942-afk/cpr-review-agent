"""Read-only pre-presentation check. No uploads, new runs, training or provider probes."""

from __future__ import annotations

import argparse
import json

import httpx
from demo_preflight import ROOT, preflight, require, sha
from demo_scenarios import local_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=local_url, default="http://127.0.0.1:8007")
    args = parser.parse_args()
    dependencies = preflight()
    catalog = json.loads((ROOT / "data/demo-cases-v1/cases.json").read_text(encoding="utf-8"))
    checked = []
    with httpx.Client(base_url=args.base_url, timeout=30) as client:

        def get(path):
            return client.get("/api" + path).raise_for_status().json()

        require(get("/health")["status"] == "ok", "Start the product service on the selected port")
        require(get("/capabilities")["vision_pipeline"] == "branches", "Use the four-stage product profile")
        for case in catalog["cases"]:
            identity, saved = case["id"], case["verified"]
            if case["kind"] == "public_video":
                source = case["source"]
                path = (ROOT / source["video_rel"]).resolve()
                require(
                    path.is_relative_to(ROOT / "data/public_cpr") and sha(path) == source["video_sha256"],
                    identity + ": source video changed",
                )
                run = get("/runs/" + saved["run_id"])
                require(
                    run["status"] == "completed"
                    and run["response"]["prediction"]["predicted_label_ids"] == saved["model_flagged_ids"],
                    identity + ": saved result changed",
                )
                require(
                    get(f"/reports/{saved['report_id']}/evidence")["status"] == "available",
                    identity + ": source evidence unavailable",
                )
                require(
                    get(f"/reports/{saved['report_id']}/explanation")["status"] == "available",
                    identity + ": explanation unavailable",
                )
            elif case["kind"] in {"invalid_control", "derived_boundary_control"}:
                path = (ROOT / case["source"]["path"]).resolve()
                require(
                    path.is_relative_to(ROOT / "data/demo-cases-v1/controls")
                    and sha(path) == case["source"]["sha256"],
                    identity + ": boundary input changed",
                )
                if case["kind"] == "derived_boundary_control":
                    run = get("/runs/" + saved["run_id"])
                    require(
                        run["status"] == "failed"
                        and not (run.get("response") or {}).get("prediction")
                        and not (run.get("response") or {}).get("classification_claims"),
                        identity + ": boundary case no longer represents a clean failure",
                    )
                    tools = get(f"/runs/{saved['run_id']}/tools")
                    require(len(tools) == 1 and tools[0]["status"] == "failed", "C05: unexpected tool path")
                    logs = (ROOT / "data/product-demo-v2/runtime/analysis" / tools[0]["id"]).glob(
                        "*/process.log"
                    )
                    require(
                        any(
                            "InputRejected: VIDEO_TOO_SHORT"
                            in p.read_text(encoding="utf-8", errors="replace")
                            for p in logs
                        ),
                        "C05: the failed input must still have a verified VIDEO_TOO_SHORT receipt",
                    )
            else:
                require(
                    get(f"/runs/{saved['run_id']}/references")["status"] == "available",
                    identity + ": references unavailable",
                )
                if identity == "C07":
                    sid = case["workspace"]["session_id"]
                    require(
                        get(f"/sessions/{sid}/context")["source"] is None,
                        "C07: the practice has been analysed since preparation; choose an unanalysed practice for this scenario",
                    )
                    selection = get(f"/sessions/{sid}/continuity")["selection"]
                    require(
                        selection["status"] == "available" and selection["label_ids"] == [4],
                        "C07: the starting focus changed",
                    )
            checked.append(identity)
    print(
        json.dumps(
            {
                "passed": True,
                "cases_checked": checked,
                "dependency_pins": dependencies["pins"],
                "http_mutations": 0,
                "new_vision_calls": 0,
                "provider_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
