"""Pin candidate numerical/source contracts before any integration run."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "workspaces/m4-refined-v2"
DATA = ROOT / "data/m4-exploration-v2"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dest = WS / "data/m4-exploration-v2"
    profile_path = dest / "pipeline-manifest.json"
    if profile_path.exists():
        if "--revise-pending" not in sys.argv or (DATA / "acceptance.json").exists():
            raise FileExistsError("Preserve pinned candidate")
        backup = dest / ("pending-profile-" + sha(profile_path)[:12] + ".json")
        assert not backup.exists()
        backup.write_bytes(profile_path.read_bytes())
    profile = json.loads((dest / "pending-profile.json").read_text(encoding="utf-8"))
    params = json.loads((ROOT / "data/m4-scheduling-v1/parameters.json").read_text())
    params["algorithm"] = "consent-coverage-dag-v2"
    params["selection_scope"] = (
        "Complete-first lexicographic coverage; explicit partial consent; per-stage bounded replanning; terminal partial commitment."
    )
    (dest / "parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    protocol = json.loads((DATA / "protocol.json").read_text())
    assert protocol["algorithm_sha256"] == sha(WS / "experiments/m4_policy_v2/algorithm.py")
    protocol["parameters_sha256"] = sha(dest / "parameters.json")
    (dest / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    decision = {
        "accepted": True,
        "protocol_sha256": sha(dest / "protocol.json"),
        "offline_sha256": sha(DATA / "offline.json"),
        "live_acceptance_required": True,
    }
    (dest / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    for name in [k for k in profile["sources"] if k.startswith("frontend/")]:
        profile["sources"].pop(name)
    for name in list(profile["sources"]):
        assert (WS / name).is_file()
        profile["sources"][name] = sha(WS / name)
    for folder in ["experiments/m4_policy_v2", "frontend/src", "frontend/dist"]:
        for p in (WS / folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                profile["sources"][p.relative_to(WS).as_posix()] = sha(p)
    for name in ["protocol.json", "parameters.json", "decision.json"]:
        profile["sources"]["data/m4-exploration-v2/" + name] = sha(dest / name)
    for name in ["review_context.py", "store.py", "practice_continuity.py", "report_view.py"]:
        profile["sources"]["backend/cpr_agent/" + name] = sha(WS / "backend/cpr_agent" / name)
    profile["scope"] = (
        "Frozen M1-M3 and two certified routes; opt-in budget-bounded partial coverage; no automatic scope widening; stage-by-stage re-evaluation; M5baseline."
    )
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    record = json.loads((WS / "workspace.json").read_text())
    record.update(
        profile_sha256=sha(profile_path),
        entrypoint="scripts/start_m4_v2.ps1",
        module_slots=profile["module_slots"],
        runtime_data=str(dest / "runtime"),
    )
    (WS / "workspace.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"profile_sha256": sha(profile_path), "sources": len(profile["sources"])}))


if __name__ == "__main__":
    main()
