"""Pin only the intended pending M3 integration edits and the built product UI."""

import json

from prepare_m3_workspace import TARGET, read, sha


def main():
    data = TARGET / "data/m3-calibration-v1"
    path = data / "pipeline-manifest.json"
    profile = read(path)
    if (data / "profile-before-integration.json").exists():
        raise FileExistsError("Integration profile already frozen")
    allowed = {
        "backend/cpr_agent/modules/members.py",
        "backend/cpr_agent/module_runtime.py",
        "backend/cpr_agent/graph.py",
        "backend/cpr_agent/grounded_review.py",
        "backend/cpr_agent/m3_quality.py",
        "experiments/m3_calibration_v1/module.py",
    }
    for name, expected in profile["sources"].items():
        if sha(TARGET / name) != expected and name not in allowed:
            raise ValueError("Unexpected changed source: " + name)
    (data / "profile-before-integration.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    for name in allowed:
        profile["sources"][name] = sha(TARGET / name)
    for folder in ["frontend/src", "frontend/dist"]:
        for source in (TARGET / folder).rglob("*"):
            if source.is_file():
                profile["sources"][source.relative_to(TARGET).as_posix()] = sha(source)
    path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    info = read(TARGET / "workspace.json")
    info["profile_sha256"] = sha(path)
    info["entrypoint"] = "scripts/start_m3.ps1"
    (TARGET / "workspace.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({"profile_sha256": sha(path), "source_pins": len(profile["sources"])}))


if __name__ == "__main__":
    main()
