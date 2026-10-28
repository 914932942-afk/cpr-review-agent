"""Read-only local dependency and pin checks. Never probes DeepSeek or downloads assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from cpr_agent.config import ROOT, Settings
from filelock import FileLock

MODEL = Path("data/public_cpr/web-integration-v1/bundle-manifest.json")
PIPELINE = Path("data/module-slots-v1/pipeline-manifest.json")
SAMPLE = Path("data/public_cpr/rgb-acceptance-v1/records/SupC04--r00--ch0/video.mp4")
GRAPH = "cpr-review-12-v0.6-grounded"
TOOLS = ["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_pins(root, manifest, field):
    value = json.loads((root / manifest).read_text(encoding="utf-8"))
    for name, expected in value[field].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "Pinned path escapes project")
        require(path.is_file() and sha(path) == expected, "Pinned dependency missing or changed: " + name)
    return len(value[field])


def preflight(*, check_ml=False):
    settings = Settings.from_env()
    counts = {
        "model": check_pins(ROOT, MODEL, "pinned_sources_and_assets"),
        "module_profile": check_pins(ROOT, PIPELINE, "sources"),
    }
    require((ROOT / "frontend/dist/index.html").is_file(), "Build the frontend first")
    require((ROOT / SAMPLE).is_file(), "The accepted public sample is missing")
    require(settings.ml_python and settings.ml_python.is_file(), "The local ML Python is unavailable")
    require(shutil.disk_usage(ROOT).free >= 1024**3, "At least 1 GiB of free disk is required")
    with FileLock(ROOT / "data/vision-gpu.lock", timeout=0):
        pass
    ml = {"checked": False}
    if check_ml:
        probe = (
            "import json,torch,cv2; "
            "print(json.dumps({'cuda':torch.cuda.is_available(),'torch':torch.__version__,"
            "'opencv':cv2.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CPR_", "DEEPSEEK_"))}
        process = subprocess.run(
            [str(settings.ml_python), "-c", probe],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=45,
            env=env,
        )
        require(process.returncode == 0, "The ML dependency probe failed; inspect the local environment")
        ml = {"checked": True, **json.loads(process.stdout.strip())}
        require(ml["cuda"], "The demo real-inference profile requires the configured CUDA environment")
    return {
        "passed": True,
        "graph_version": GRAPH,
        "pins": counts,
        "pipeline_sha256": sha(ROOT / PIPELINE),
        "sample_sha256": sha(ROOT / SAMPLE),
        "free_disk_GiB": round(shutil.disk_usage(ROOT).free / 1024**3, 2),
        "ml": ml,
        "network_requests": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-ml", action="store_true")
    args = parser.parse_args()
    print(json.dumps(preflight(check_ml=args.check_ml)))


if __name__ == "__main__":
    main()
