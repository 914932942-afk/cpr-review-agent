"""Build a new immutable, deterministic profile after explicit team-module registration."""

import argparse
import hashlib
import json
from pathlib import Path

from cpr_agent.config import ROOT
from cpr_agent.modules.contracts import CONTRACT_VERSION
from cpr_agent.modules.registry import DEFAULT_SELECTIONS, build_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-profile", type=Path)
    for key, value in DEFAULT_SELECTIONS.items():
        parser.add_argument("--" + key.lower())
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior module profiles; choose a new output")
    if args.base_profile and not args.base_profile.is_file():
        raise FileNotFoundError("Explicit base profile does not exist")
    base_path = args.base_profile or (ROOT / "data/module-slots-v1/pipeline-manifest.json")
    inherited = (
        json.loads(base_path.read_text(encoding="utf-8")).get("module_slots", DEFAULT_SELECTIONS)
        if base_path.is_file()
        else DEFAULT_SELECTIONS
    )
    selected = {key: getattr(args, key.lower()) or inherited[key] for key in DEFAULT_SELECTIONS}
    registry = build_registry()
    registry.validate_selection(selected)
    value = json.loads((ROOT / "data/m1-optimization-v2/pipeline-manifest.json").read_text(encoding="utf-8"))
    value.update(
        module_slots=selected,
        module_contract_version=CONTRACT_VERSION,
        worker_entrypoint="scripts/module_slots_worker.py",
        scope="Accepted M1 plus independently registered M2-M5; baseline selection is not algorithm completion",
    )
    files = set((ROOT / "backend/cpr_agent/modules").rglob("*.py"))
    files.update(
        ROOT / name
        for name in [
            "backend/cpr_agent/module_runtime.py",
            "backend/cpr_agent/models.py",
            "backend/cpr_agent/graph.py",
            "experiments/module_slots_v1/slots_pipeline.py",
            "scripts/module_slots_worker.py",
        ]
    )
    for key, implementation in selected.items():
        files.update(ROOT / name for name in registry.spec(key, implementation).required_files)
    for path in files:
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():
            raise ValueError("Module source/parameter must be a project file")
        with path.open("rb") as stream:
            value["sources"][path.relative_to(ROOT).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
    print(
        json.dumps(
            {"profile": str(args.output), "module_slots": selected, "source_pins": len(value["sources"])}
        )
    )


if __name__ == "__main__":
    main()
