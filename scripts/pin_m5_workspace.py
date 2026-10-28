"""Pin all runtime edits and the built UI before real integration starts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "workspaces/m5-cascade-v1"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    folder = WS / "data/m5-cascade-v1"
    target = folder / "pipeline-manifest.json"
    assert not target.exists()
    profile = json.loads((folder / "pending-profile.json").read_text(encoding="utf-8"))
    for name in [k for k in profile["sources"] if k.startswith("frontend/")]:
        profile["sources"].pop(name)
    for name in profile["sources"]:
        profile["sources"][name] = sha(WS / name)
    for folder in ["experiments/m5_cascade_v1", "frontend/src", "frontend/dist"]:
        for p in (WS / folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                profile["sources"][p.relative_to(WS).as_posix()] = sha(p)
    for name in ["protocol.json", "certificate.json", "decision.json"]:
        profile["sources"]["data/m5-cascade-v1/" + name] = sha(WS / "data/m5-cascade-v1" / name)
    for name in [
        "backend/cpr_agent/m5_validation.py",
        "scripts/m5_once.py",
        "scripts/benchmark_m5_cascade.py",
    ]:
        profile["sources"][name] = sha(WS / name)
    target.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    record = json.loads((WS / "workspace.json").read_text())
    record.update(
        profile_sha256=sha(target),
        module_slots=profile["module_slots"],
        runtime_data=str(WS / "data/m5-cascade-v1/runtime"),
    )
    (WS / "workspace.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"profile_sha256": sha(target), "pinned_files": len(profile["sources"])}))


if __name__ == "__main__":
    main()
