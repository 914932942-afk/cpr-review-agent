"""Canonical local launcher. Provider secrets stay in web settings, not CLI/ML env."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT / "workspaces/m5-cascade-v1"
PROVIDER_KEYS = ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL")


def provider_values(path, environment):
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key not in PROVIDER_KEYS:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            values[key] = value
    values.update({k: environment[k] for k in PROVIDER_KEYS if k in environment})
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8013)
    p.add_argument("--mode", choices=["disabled", "deepseek"], default="disabled")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if not 1024 <= args.port <= 65535:
        raise SystemExit("Use a local unprivileged port")
    sys.path.insert(0, str(WORKSPACE / "backend"))
    from cpr_agent.config import ROOT, Settings, read_environment

    if ROOT.resolve() != WORKSPACE.resolve():
        raise SystemExit("Refusing an unexpected application workspace")
    root_config = read_environment(PROJECT / ".env")
    ml_python = root_config.get("CPR_ML_PYTHON")
    if not ml_python:
        raise SystemExit("Configure CPR_ML_PYTHON in the project root .env")
    ml_path = Path(ml_python)
    if not ml_path.is_absolute():
        ml_path = (PROJECT / ml_path).resolve()
    if not ml_path.is_file():
        raise SystemExit("Configured ML Python does not exist")
    os.environ.update(
        {
            "CPR_ML_PYTHON": str(ml_path),
            "CPR_VISION_MODE": "real",
            "CPR_VISION_PIPELINE": "branches",
            "CPR_LLM_MODE": args.mode,
            "CPR_CONTEXT_MODE": "bound-v1",
            "CPR_UPLOAD_PREFLIGHT": "true",
            "CPR_REFERENCE_CATALOG": str(WORKSPACE / "data/reference-v1/catalog.json"),
            "CPR_DATA_DIR": str(WORKSPACE / "data/m5-cascade-v1/runtime"),
            "CPR_BRANCH_MANIFEST": str(WORKSPACE / "data/m5-cascade-v1/pipeline-manifest.json"),
            "CPR_VISION_MANIFEST": str(WORKSPACE / "data/public_cpr/web-integration-v1/bundle-manifest.json"),
            "CPR_SERVICE_BUDGET_SECONDS": "900",
        }
    )
    provider = provider_values(PROJECT / ".env", os.environ)
    os.environ.update(provider)
    settings = Settings.from_env()
    if args.mode == "deepseek" and not settings.deepseek_api_key:
        for k in PROVIDER_KEYS:
            os.environ.pop(k, None)
        raise SystemExit("DeepSeek key missing; configure it privately in the project root .env")
    # Settings capture the secret. It must not reach CV child processes or process args.
    for k in PROVIDER_KEYS:
        os.environ.pop(k, None)
    from cpr_agent.branch_vision import validate_pipeline

    validate_pipeline(settings)
    if args.check:
        print(
            json.dumps(
                {
                    "passed": True,
                    "mode": settings.llm_mode,
                    "key_configured": bool(settings.deepseek_api_key),
                    "provider_key_in_child_environment": "DEEPSEEK_API_KEY" in os.environ,
                    "graph_version": settings.graph_version,
                    "pipeline_sha256": validate_pipeline(settings),
                    "provider_calls": 0,
                }
            )
        )
        return
    import uvicorn
    from cpr_agent.api import create_app

    print(
        f"CPR mainline http://127.0.0.1:{args.port} | M1-M5 | LLM {args.mode} | root .env loaded privately",
        flush=True,
    )
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
