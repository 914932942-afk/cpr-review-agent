"""Derive an isolated M3 app from the accepted M1/M2 release without mutating that release."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workspaces/m1-m2-refined-v1"
TARGET = ROOT / "workspaces/m3-calibrated-v1"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def patch(path, old, new):
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise ValueError("Unexpected patch target: " + str(path))
    path.write_text(text.replace(old, new), encoding="utf-8")


def main():
    if TARGET.exists():
        raise FileExistsError("Preserve existing M3 workspace")
    data = ROOT / "data/m3-calibration-v1"
    if not read(data / "decision.json")["accepted"]:
        raise ValueError("M3 offline acceptance required")
    original = SOURCE / "data/m1-m2-refined-v1/pipeline-manifest.json"
    profile = read(original)
    assert all(sha(SOURCE / p) == value for p, value in profile["sources"].items())
    upstream = {
        p: value
        for p, value in profile["sources"].items()
        if p.startswith(
            (
                "experiments/m1_fast_v1/",
                "experiments/m2_competence_v3/",
                "experiments/branch_pipeline_v1/",
                "data/m1-m2-exploration-v1/",
            )
        )
    }
    (data / "upstream.json").write_text(
        json.dumps(
            {
                "sources": upstream,
                "model_bundle_sha256": profile["model_bundle_sha256"],
                "accepted_parent_profile_sha256": sha(original),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def ignore(path, names):
        return [
            name
            for name in names
            if name in ["__pycache__", "node_modules", ".env", ".pytest_cache", ".ruff_cache"]
            or (Path(path).name == "m1-m2-refined-v1" and name == "runtime")
        ]

    shutil.copytree(SOURCE, TARGET, ignore=ignore)
    shutil.copytree(
        ROOT / "experiments/m3_calibration_v1",
        TARGET / "experiments/m3_calibration_v1",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ["protocol.json", "selection.json", "decision.json", "upstream.json"]:
        destination = TARGET / "data/m3-calibration-v1" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data / name, destination)
    shutil.copy2(
        ROOT / "experiments/m3_calibration_v1/quality_adapter.py", TARGET / "backend/cpr_agent/m3_quality.py"
    )
    members = TARGET / "backend/cpr_agent/modules/members.py"
    patch(members, "def register_members(registry):", "def _register_m2(registry):")
    with members.open("a", encoding="utf-8") as f:
        f.write("""

def create_m3():
    spec = importlib.util.spec_from_file_location("cpr_m3_member", ROOT / "experiments/m3_calibration_v1/module.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SourceQualityCalibration()


def register_members(registry):
    _register_m2(registry)
    registry.register(ModuleSpec("M3", "source-quality-selective-v1", create_m3, kind="improved", required_files=(
        "experiments/m3_calibration_v1/algorithm.py", "experiments/m3_calibration_v1/module.py", "backend/cpr_agent/m3_quality.py",
        "data/m3-calibration-v1/protocol.json", "data/m3-calibration-v1/selection.json", "data/m3-calibration-v1/decision.json", "data/m3-calibration-v1/upstream.json")))
""")
    patch(
        TARGET / "backend/cpr_agent/module_runtime.py",
        """        targets = goal.target_label_ids or sorted(""",
        """        if execution_id and self.implementation == "source-quality-selective-v1":
            from .m3_quality import bound_quality
            quality = bound_quality(self.store, self.settings, execution, output, raw)
        targets = goal.target_label_ids or sorted(""",
    )
    patch(
        TARGET / "backend/cpr_agent/graph.py",
        """                if state.get("module_assessment"):
""",
        """                if not state.get("module_assessment") and getattr(self.assessment, "implementation", None) == "source-quality-selective-v1":
                    checked_goal = goal.model_copy(update={"target_label_ids": [c.label_id for c in response.classification_claims]})
                    checked = self.assessment.evaluate(checked_goal, {final_tool(self.settings): prediction.execution_id})
                    state = {**state, "module_assessment": checked.model_dump(mode="json")}
                    self.store.event(state["run_id"], "module.call", {"module_id": "M3", "implementation_id": self.assessment.implementation, "source_prediction_ids": checked.source_prediction_ids}, "N11")
                if state.get("module_assessment"):
""",
    )
    patch(
        TARGET / "backend/cpr_agent/grounded_review.py",
        """    flagged = review.flagged_label_ids
""",
        """    if parsed.module_assessment and parsed.module_assessment.calibration_profile == "source-quality-selective-v1":
        for entry in entries:
            label = entry["label_id"]
            entry["review_state"] = parsed.module_assessment.states.get(label, "not_evaluated")
            entry["calibrated_estimate"] = parsed.module_assessment.calibrated_probabilities.get(label)
            if entry["review_state"] == "uncertain":
                entry["recorded_result"] += " The reliability check needs another look; this result is uncertain. The raw model output is unchanged."
    flagged = review.flagged_label_ids
""",
    )
    patch(
        TARGET / "backend/cpr_agent/grounded_review.py",
        """    value["content_sha256"] = digest(value)""",
        """    uncertain = [e["label_id"] for e in entries if e.get("review_state") == "uncertain"]
    if uncertain:
        value["uncertain_label_ids"] = uncertain
        value["summary"] += f" {len(uncertain)} categories need another look; raw flags have not changed."
        value["next_steps"].insert(0, "Replay the source clip and compare categories marked Needs another look. The current checks could not resolve them reliably.")
    value["content_sha256"] = digest(value)""",
    )
    profile["module_slots"]["M3"] = "source-quality-selective-v1"
    for name in [
        "backend/cpr_agent/modules/members.py",
        "backend/cpr_agent/module_runtime.py",
        "backend/cpr_agent/graph.py",
        "backend/cpr_agent/grounded_review.py",
        "backend/cpr_agent/m3_quality.py",
        "experiments/m3_calibration_v1/algorithm.py",
        "experiments/m3_calibration_v1/module.py",
        "data/m3-calibration-v1/protocol.json",
        "data/m3-calibration-v1/selection.json",
        "data/m3-calibration-v1/decision.json",
        "data/m3-calibration-v1/upstream.json",
    ]:
        profile["sources"][name] = sha(TARGET / name)
    profile["scope"] = (
        "Accepted frozen M1/M2 + post-hoc source/quality M3 calibration and review deferral. CV labels untouched. M4/M5 baseline."
    )
    destination = TARGET / "data/m3-calibration-v1/pipeline-manifest.json"
    destination.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    record = {
        "workspace": str(TARGET),
        "parent_workspace": str(SOURCE),
        "parent_profile_sha256": sha(original),
        "profile_sha256": sha(destination),
        "env_copied": False,
        "runtime_data": str(TARGET / "data/m3-calibration-v1/runtime"),
        "module_slots": profile["module_slots"],
    }
    (TARGET / "workspace.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
