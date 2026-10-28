"""Pin the reviewed pending integration and built UI before M4 live acceptance."""

import json

from prepare_m4_workspace import TARGET, read, sha


def main():
    folder = TARGET / "data/m4-scheduling-v1"
    path = folder / "pipeline-manifest.json"
    if (folder / "profile-before-integration.json").exists():
        raise FileExistsError("Preserve frozen integration profile")
    profile = read(path)
    allowed = ["backend/cpr_agent/branch_vision.py", "backend/cpr_agent/m4_lineage.py"]
    for name, expected in profile["sources"].items():
        if name.startswith("frontend/"):
            continue
        if sha(TARGET / name) != expected and name not in allowed:
            raise ValueError("Unexpected integration change: " + name)
    (folder / "profile-before-integration.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    for name in allowed:
        profile["sources"][name] = sha(TARGET / name)
    for name in [key for key in profile["sources"] if key.startswith("frontend/")]:
        profile["sources"].pop(name)
    for area in ["frontend/src", "frontend/dist"]:
        for source in (TARGET / area).rglob("*"):
            if source.is_file():
                profile["sources"][source.relative_to(TARGET).as_posix()] = sha(source)
    path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    value = read(TARGET / "workspace.json")
    value["profile_sha256"] = sha(path)
    (TARGET / "workspace.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({"profile_sha256": sha(path), "pinned_sources": len(profile["sources"])}))


if __name__ == "__main__":
    main()
