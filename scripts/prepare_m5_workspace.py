"""Isolate M5 after offline/performance/pixel gates. Preserve numerical M1-M4."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workspaces/m4-refined-v2"
TARGET = ROOT / "workspaces/m5-cascade-v1"
DATA = ROOT / "data/m5-cascade-v1"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def patch(name, old, new):
    p = TARGET / name
    s = p.read_text(encoding="utf-8")
    assert s.count(old) == 1, (name, old[:50])
    p.write_text(s.replace(old, new), encoding="utf-8")


def main():
    assert not TARGET.exists(), "Preserve candidate workspace"
    for p in [
        DATA / "offline.json",
        ROOT / "test-results/m5-cascade-v1/benchmark/acceptance.json",
        ROOT / "test-results/m5-cascade-v1/raw26/acceptance.json",
        ROOT / "test-results/m5-cascade-v1/ablation.json",
    ]:
        assert read(p)["passed"]
    parent = SOURCE / "data/m4-exploration-v2/pipeline-manifest.json"
    profile = read(parent)
    assert all(sha(SOURCE / k) == h for k, h in profile["sources"].items())
    TARGET.mkdir()
    for folder in ["backend", "scripts", "tests", "fixtures", "frontend", "experiments", "data"]:
        shutil.copytree(
            SOURCE / folder,
            TARGET / folder,
            ignore=shutil.ignore_patterns(
                "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", "runtime"
            ),
        )
    for name in ["pyproject.toml", "README.md"]:
        shutil.copy2(ROOT / name, TARGET / name)
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "New-Item -ItemType Junction -Path '"
            + str(TARGET / "frontend/node_modules")
            + "' -Target '"
            + str(ROOT / "frontend/node_modules")
            + "' | Out-Null",
        ],
        check=True,
    )
    shutil.copytree(
        ROOT / "experiments/m5_cascade_v1",
        TARGET / "experiments/m5_cascade_v1",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (TARGET / "data/m5-cascade-v1").mkdir()
    for p in DATA.glob("*.json"):
        shutil.copy2(p, TARGET / "data/m5-cascade-v1" / p.name)
    for name in read(DATA / "protocol.json")["additional_sources"]:
        shutil.copy2(ROOT / name, TARGET / name)
    shutil.copy2(
        ROOT / "experiments/m5_cascade_v1/validation.py", TARGET / "backend/cpr_agent/m5_validation.py"
    )
    patch(
        "backend/cpr_agent/modules/members.py",
        "def register_members(registry):",
        "def _register_m4(registry):",
    )
    with (TARGET / "backend/cpr_agent/modules/members.py").open("a", encoding="utf-8") as f:
        f.write("""

def create_m5():
    spec=importlib.util.spec_from_file_location("cpr_m5_member",ROOT/"experiments/m5_cascade_v1/module.py")
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module.QualityDependencyRGB()

def register_members(registry):
    _register_m4(registry)
    registry.register(ModuleSpec("M5","quality-dependency-cascade-v1",create_m5,kind="improved",required_files=(
        "experiments/m5_cascade_v1/algorithm.py","experiments/m5_cascade_v1/runtime.py","experiments/m5_cascade_v1/inference.py","experiments/m5_cascade_v1/module.py",
        "backend/cpr_agent/m5_validation.py","data/m5-cascade-v1/protocol.json","data/m5-cascade-v1/certificate.json","data/m5-cascade-v1/decision.json")))
""")
    patch(
        "backend/cpr_agent/config.py",
        '                "cpr-review-12-v0.8-adaptive"',
        '                ("cpr-review-12-v0.9-cascade" if json.loads(branch_manifest.read_text(encoding="utf-8"))["module_slots"].get("M5")=="quality-dependency-cascade-v1" else "cpr-review-12-v0.8-adaptive")',
    )
    patch(
        "backend/cpr_agent/models.py",
        "    scheduling: dict[str, Any] | None = None",
        "    scheduling: dict[str, Any] | None = None\n    visual_processing: dict[str, Any] | None = None",
    )
    patch(
        "backend/cpr_agent/graph.py",
        '            response.scheduling = state.get("scheduling")',
        '            response.scheduling = state.get("scheduling")\n            from .m5_validation import visual_summary\n            response.visual_processing = visual_summary(self.store, self.settings, prediction)',
    )
    patch(
        "backend/cpr_agent/branch_vision.py",
        "    return receipt\n",
        '    if receipt.tool_id == "rgb.predict":\n        from .m5_validation import validate_visual_metadata, validate_visual_archive\n        validate_visual_metadata(raw, settings)\n        validate_visual_archive(raw, path)\n    return receipt\n',
    )
    p = TARGET / "backend/cpr_agent/modules/m5_rgb.py"
    p.write_text(
        p.read_text(encoding="utf-8").replace(
            "Lightweight RGB recognition and budgeted sampling.",
            "Quality-gated, dependency-aware lightweight visual cascade.",
        ),
        encoding="utf-8",
    )
    # Keep the accepted M4 policy and its conservative cost parameters byte-identical.
    profile["module_slots"]["M5"] = "quality-dependency-cascade-v1"
    profile["scope"] = (
        "M1-M4 numerical algorithms frozen; M5 certified conditional visual computation with full quality fallback. Same16 samples and heads; no new training."
    )
    (TARGET / "data/m5-cascade-v1/pending-profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    (TARGET / "workspace.json").write_text(
        json.dumps(
            {
                "workspace": str(TARGET),
                "parent_workspace": str(SOURCE),
                "parent_profile_sha256": sha(parent),
                "entrypoint": "scripts/start_m5.ps1",
                "url": "http://127.0.0.1:8013",
                "graph_version": "cpr-review-12-v0.9-cascade",
                "candidate_not_accepted": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Prepared M5 workspace " + str(TARGET))


if __name__ == "__main__":
    main()
