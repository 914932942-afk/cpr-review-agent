"""Isolate M4 routing changes; preserve all accepted M1/M2/M3 numerical algorithms."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workspaces/m3-calibrated-v1"
TARGET = ROOT / "workspaces/m4-routing-v1"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def patch(name, old, new, count=1):
    path = TARGET / name
    text = path.read_text(encoding="utf-8")
    if text.count(old) != count:
        raise ValueError("Unexpected patch: " + name + " " + old[:60])
    path.write_text(text.replace(old, new), encoding="utf-8")


def main():
    if TARGET.exists() and (
        (TARGET / "data/m4-scheduling-v1/pipeline-manifest.json").exists()
        or (TARGET / "data/m4-scheduling-v1/runtime").exists()
        or read(TARGET / "workspace.json").get("workspace") != str(SOURCE)
    ):
        raise FileExistsError("Preserve existing M4 workspace")
    data = ROOT / "data/m4-scheduling-v1"
    if not read(data / "decision.json")["accepted"]:
        raise ValueError("M4 offline route acceptance required")
    parent = SOURCE / "data/m3-calibration-v1/pipeline-manifest.json"
    profile = read(parent)
    assert all(sha(SOURCE / p) == value for p, value in profile["sources"].items())
    shutil.copytree(
        SOURCE,
        TARGET,
        dirs_exist_ok=True,
        ignore=lambda path, names: [
            n
            for n in names
            if n in ["__pycache__", "node_modules", ".env", ".pytest_cache", ".ruff_cache"]
            or (n == "runtime" and Path(path).name in ["m3-calibration-v1", "m1-m2-refined-v1"])
        ],
    )
    shutil.copytree(
        ROOT / "experiments/m4_policy_v1",
        TARGET / "experiments/m4_policy_v1",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ["parameters.json", "protocol.json", "decision.json"]:
        path = TARGET / "data/m4-scheduling-v1" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data / name, path)
    for name, destination in [("lineage.py", "m4_lineage.py"), ("controller.py", "m4_controller.py")]:
        shutil.copy2(ROOT / "experiments/m4_policy_v1" / name, TARGET / "backend/cpr_agent" / destination)
    members = TARGET / "backend/cpr_agent/modules/members.py"
    patch(
        "backend/cpr_agent/modules/members.py",
        "def register_members(registry):",
        "def _register_m3(registry):",
    )
    with members.open("a", encoding="utf-8") as f:
        f.write("""

def create_m4():
    spec = importlib.util.spec_from_file_location("cpr_m4_member", ROOT / "experiments/m4_policy_v1/module.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GoalCostPolicy()

def register_members(registry):
    _register_m3(registry)
    registry.register(ModuleSpec("M4", "goal-cost-dag-v1", create_m4, kind="improved", required_files=(
        "experiments/m4_policy_v1/algorithm.py", "experiments/m4_policy_v1/module.py", "experiments/m4_policy_v1/stages.py",
        "backend/cpr_agent/m4_lineage.py", "backend/cpr_agent/m4_controller.py",
        "data/m4-scheduling-v1/protocol.json", "data/m4-scheduling-v1/parameters.json", "data/m4-scheduling-v1/decision.json")))
""")
    patch(
        "backend/cpr_agent/modules/contracts.py",
        """    remaining_budget_seconds: float = Field(ge=0)
""",
        """    remaining_budget_seconds: float = Field(ge=0)
    covered_label_ids: list[int] = Field(default_factory=list)
    decoded_frames: int | None = Field(default=None, ge=8, le=900)
    observed_tool_seconds: dict[str, FiniteFloat] = Field(default_factory=dict)
""",
    )
    patch(
        "backend/cpr_agent/modules/contracts.py",
        """    reason: str = Field(min_length=1, max_length=1000)
""",
        """    reason: str = Field(min_length=1, max_length=1000)
    stop_code: str | None = None
    audit: dict[str, Any] = Field(default_factory=dict)
""",
    )
    patch(
        "backend/cpr_agent/models.py",
        """    analysis_seconds: float = Field(ge=0)
    evidence_scope: Literal["whole_clip"] = "whole_clip"
""",
        """    analysis_seconds: float = Field(ge=0)
    evidence_scope: Literal["whole_clip"] = "whole_clip"
    analysis_route: Literal["full-v15", "body-qualified-v1"] = "full-v15"
""",
    )
    patch(
        "backend/cpr_agent/models.py",
        """        if sorted(self.supported_label_ids) != list(CANDIDATE_LABELS):""",
        """        expected_scope = list(CANDIDATE_LABELS) if self.analysis_route == "full-v15" else [3, 4, 8, 11]
        if sorted(self.supported_label_ids) != expected_scope:""",
    )
    patch(
        "backend/cpr_agent/models.py",
        """        if sorted(s.original_label_id for s in self.scores) != list(CANDIDATE_LABELS):""",
        """        if sorted(s.original_label_id for s in self.scores) != expected_scope:""",
    )
    patch(
        "backend/cpr_agent/models.py",
        """    module_assessment: Assessment | None = None
""",
        """    module_assessment: Assessment | None = None
    scheduling: dict[str, Any] | None = None
""",
    )
    patch(
        "backend/cpr_agent/config.py",
        """            graph_version="cpr-review-12-v0.6-grounded"
""",
        """            graph_version=("cpr-review-12-v0.7-routing" if is_slots and json.loads(branch_manifest.read_text(encoding="utf-8"))["module_slots"].get("M4") == "goal-cost-dag-v1" else "cpr-review-12-v0.6-grounded")
""",
    )
    patch("backend/cpr_agent/runtime.py", """SlotPolicy(settings)""", """SlotPolicy(settings, self.store)""")
    patch(
        "backend/cpr_agent/module_runtime.py",
        """class SlotPolicy:
    def __init__(self, settings):
        self.settings = settings""",
        """class SlotPolicy:
    def __init__(self, settings, store=None):
        self.settings, self.store = settings, store""",
    )
    patch(
        "backend/cpr_agent/module_runtime.py",
        """        result = Decision.model_validate(self.module.choose(observation))""",
        """        if self.implementation == "goal-cost-dag-v1":
            from .m4_controller import observe
            observation = observe(self.store, self.settings, goal, receipts, failed, assessment, run, observation.remaining_budget_seconds)
        result = Decision.model_validate(self.module.choose(observation))""",
    )
    patch(
        "backend/cpr_agent/graph.py",
        """    module_assessment: dict[str, Any]
""",
        """    module_assessment: dict[str, Any]
    scheduling: dict[str, Any]
""",
    )
    patch(
        "backend/cpr_agent/graph.py",
        """                "profile_id": decision.profile_id if decision else None,
""",
        """                "profile_id": decision.profile_id if decision else None,
                "stop_code": decision.stop_code if decision else None,
                "audit": decision.audit if decision else {},
""",
    )
    patch(
        "backend/cpr_agent/graph.py",
        """        if action == "STOP":
            return self._next(state, "N10", action=action)
""",
        """        if getattr(self.policy, "implementation", None) == "goal-cost-dag-v1":
            schedule = dict(state.get("scheduling", {}))
            if action == "STOP":
                schedule["stop"] = decision.model_dump(mode="json")
                updates = {"action": action, "scheduling": schedule}
                if decision.stop_code == "M4_BUDGET_INSUFFICIENT" and final_tool(self.settings) not in state["receipts"]:
                    updates.update(error_code="M4_BUDGET_INSUFFICIENT", final_status="rejected")
                return self._next(state, "N10", **updates)
            from .m4_controller import missing_tools
            covered = list((state.get("module_assessment") or {}).get("states", {}))
            covered = [int(k) for k in covered if (state.get("module_assessment") or {})["states"][k] != "not_evaluated"]
            missing = missing_tools(decision.profile_id, state["receipts"], covered, goal.target_label_ids)
            if not missing or any(tool in state["failed_tools"] for tool in missing):
                raise ValueError("M4_INVALID_OR_BLOCKED_PLAN")
            schedule["plan"] = decision.model_dump(mode="json")
            plan = [{"tool_id": tool, "execution_id": str(uuid5(NAMESPACE_URL, state["run_id"] + ":" + tool))} for tool in missing]
            return self._next(state, "N07", action=action, plan=plan, step_index=0, scheduling=schedule,
                              context={**state.get("context", {}), "reused_result": False})
        if action == "STOP":
            return self._next(state, "N10", action=action)
""",
    )
    patch(
        "backend/cpr_agent/graph.py",
        """        # Receipt IDs deliberately cannot masquerade as PoseArtifact / Prediction IDs.""",
        """        if getattr(self.policy, "implementation", None) == "goal-cost-dag-v1":
            bindings["route_profile"] = state["scheduling"]["plan"]["profile_id"]
            if bindings["route_profile"] == "body-qualified-v1":
                bindings.pop("rgb_receipt_id", None)
            anchor = state.get("context", {}).get("reuse_source_execution_id")
            if anchor:
                bindings["reuse_source_execution_id"] = anchor
        # Receipt IDs deliberately cannot masquerade as PoseArtifact / Prediction IDs.""",
    )
    # Body finalisation never dereferences a missing RGB receipt.
    patch(
        "backend/cpr_agent/graph.py",
        """                rgb_receipt_id=state["receipts"]["rgb.predict"],""",
        """                **({"rgb_receipt_id": state["receipts"]["rgb.predict"]} if "rgb.predict" in state["receipts"] else {}),""",
    )
    patch(
        "backend/cpr_agent/graph.py",
        """        failure = (
""",
        """        if self.settings.vision_mode == "real" and getattr(self.policy, "implementation", None) == "goal-cost-dag-v1" and goal and final_tool(self.settings) in state.get("receipts", {}) and not state.get("error_code"):
            current = validate_receipt(self.store.get_tool(state["receipts"][final_tool(self.settings)])["output"], self.store, self.settings)
            if not set(goal.target_label_ids) <= set(current.supported_label_ids):
                status = "partially_completed"
        failure = (
""",
    )
    patch(
        "backend/cpr_agent/graph.py",
        """        error = state.get("error_code") or (failure if status == "failed" else None)""",
        """        error = state.get("error_code") or (failure if status == "failed" else None)""",
    )
    patch(
        "backend/cpr_agent/graph.py",
        """            if response.prediction:
""",
        """            response.scheduling = state.get("scheduling")
            if response.prediction:
""",
    )
    patch(
        "backend/cpr_agent/context_graph.py",
        """                        context["reused_result"] = True
""",
        """                        context["reused_result"] = True
                        from .m4_lineage import enabled, hydrate
                        if enabled(self.settings) and analysis:
                            receipts.update(hydrate(self.store, self.settings, previous))
                            context["reuse_source_execution_id"] = previous.execution_id
""",
    )
    patch(
        "backend/cpr_agent/context_graph.py",
        """if self.settings.vision_mode != "real" or not (response or {}).get("review"):""",
        """if self.settings.vision_mode != "real" or not (response or {}).get("review") or not response.get("classification_claims"):""",
    )
    patch(
        "backend/cpr_agent/vision.py",
        """            for label in targets
""",
        """            for label in targets if label in score_map
""",
    )
    patch(
        "backend/cpr_agent/vision.py",
        """    return ReviewResponse(
""",
        """    missing = [label for label in targets if prediction and label not in prediction.supported_label_ids] if show else []
    if missing:
        message = "Some selected categories have not been analysed. Expand the review to run the needed tools. " + message if claims else "The selected categories have not been analysed. Expand the review to run the needed tools."
    return ReviewResponse(
""",
    )
    patch(
        "backend/cpr_agent/vision.py",
        """        target_states={str(c.label_id): c.state for c in claims}""",
        """        target_states={**{str(c.label_id): c.state for c in claims}, **{str(k): "not_evaluated" for k in missing}}""",
    )
    patch(
        "backend/cpr_agent/evidence.py",
        """        if prediction.profile_id.startswith("single-view-v15-branches-v1:"):""",
        """        if prediction.analysis_route == "body-qualified-v1":
            expected["analysis_route"] = "body-qualified-v1"
        if prediction.profile_id.startswith("single-view-v15-branches-v1:"):""",
    )
    branch = "backend/cpr_agent/branch_vision.py"
    patch(
        branch,
        """from .config import ROOT
""",
        """from .config import ROOT
from .m4_lineage import reusable
""",
    )
    patch(
        branch,
        """source["run_id"] != request["run_id"]""",
        """not reusable(store, settings, request, source)""",
    )
    patch(
        branch,
        """execution["run_id"] != request["run_id"]""",
        """not reusable(store, settings, request, execution)""",
    )
    patch(
        branch,
        """source["run_id"] != request.run_id""",
        """not reusable(self.store, settings, request.model_dump(), source)""",
    )
    patch(
        branch,
        """    if "module_slots" in profile:
        modified = validate_module_receipts(raw, settings, ["M2", "M5"], prediction.clip_id)""",
        """    if prediction.analysis_route == "body-qualified-v1":
        return validate_body_final(prediction, raw, store, settings, profile)
    if "module_slots" in profile:
        modified = validate_module_receipts(raw, settings, ["M2", "M5"], prediction.clip_id)""",
    )
    patch(
        branch,
        """        for binding, tool in DEPENDENCIES[request.tool_id].items():""",
        """        dependencies = DEPENDENCIES[request.tool_id]
        if request.tool_id == "fusion.predict" and request.input_bindings.get("route_profile") == "body-qualified-v1":
            dependencies = {"skeleton_receipt_id": "skeleton.predict"}
        for binding, tool in dependencies.items():""",
    )
    patch(
        branch,
        """        base = settings.data_dir / "analysis" / request.execution_id""",
        """        inputs["__route_profile__"] = request.input_bindings.get("route_profile", "full-v15")
        base = settings.data_dir / "analysis" / request.execution_id""",
    )
    patch(
        branch,
        """            stage_seconds = sum(
                self.store.get_tool(request.input_bindings[k])["output"]["analysis_seconds"]
                for k in ["skeleton_receipt_id", "rgb_receipt_id"]
            )""",
        """            stage_seconds = sum(
                self.store.get_tool(request.input_bindings[k])["output"]["analysis_seconds"]
                for k in ["skeleton_receipt_id", "rgb_receipt_id"] if k in request.input_bindings and self.store.get_tool(request.input_bindings[k])["run_id"] == request.run_id
            )""",
    )
    patch(
        branch,
        """            stage_seconds += self.store.get_tool(pose_id)["output"]["analysis_seconds"]""",
        """            if self.store.get_tool(pose_id)["run_id"] == request.run_id:
                stage_seconds += self.store.get_tool(pose_id)["output"]["analysis_seconds"]""",
    )
    patch(
        branch,
        """                supported_label_ids=raw["supported_label_ids"],""",
        """                supported_label_ids=raw["supported_label_ids"],
                analysis_route=raw.get("analysis_route", "full-v15"),""",
    )
    with (TARGET / branch).open("a", encoding="utf-8") as f:
        f.write("""

def validate_body_final(prediction, raw, store, settings, profile):
    if profile.get("module_slots", {}).get("M4") != "goal-cost-dag-v1" or profile.get("m4_routes", {}).get("body-qualified-v1", {}).get("supported_label_ids") != [3,4,8,11]:
        raise VisionError("M4_ROUTE_NOT_CERTIFIED")
    request = store.get_tool(prediction.execution_id)["request"]
    if request["input_bindings"].get("route_profile") != "body-qualified-v1" or set(raw.get("dependencies", {})) != {"skeleton.predict"} or raw.get("module_executions") != {} or raw.get("model_or_threshold_modified") is not False:
        raise VisionError("M4_BODY_RESULT_INVALID")
    source = store.get_tool(request["input_bindings"]["skeleton_receipt_id"])
    if not reusable(store, settings, request, source):
        raise VisionError("M4_BODY_SOURCE_NOT_APPROVED")
    receipt = validate_branch(source["output"], store, settings)
    if receipt.clip_id != prediction.clip_id or receipt.artifact_sha256 != raw["dependencies"]["skeleton.predict"] or receipt.tool_id != "skeleton.predict":
        raise VisionError("M4_BODY_SOURCE_MISMATCH")
    value = json.loads(artifact_path(receipt, settings).read_text(encoding="utf-8"))
    scores = [s for s in value["scores"] if s["original_label_id"] in [3,4,8,11]]
    flags = [k for k in value["predicted_original_label_ids"] if k in [3,4,8,11]]
    if raw["scores"] != scores or prediction.predicted_label_ids != flags or raw["decoded_frames"] != value["decoded_frames"] or raw["reported_fps"] != value["reported_fps"] or raw.get("fullpose_audit") != value.get("fullpose_audit") or raw.get("module_contract_version") != "cpr-modules-v1":
        raise VisionError("M4_BODY_READOUT_CHANGED")
""")
    patch(
        "backend/cpr_agent/m3_quality.py",
        """    branch = {}
""",
        """    if getattr(output, "analysis_route", "full-v15") == "body-qualified-v1":
        source = store.get_tool(execution["request"]["input_bindings"]["skeleton_receipt_id"])
        receipt = validate_branch(source["output"], store, settings)
        if receipt.artifact_sha256 != raw["dependencies"]["skeleton.predict"] or receipt.clip_id != output.clip_id:
            raise ValueError("M3_BODY_QUALITY_SOURCE_MISMATCH")
        audit = raw.get("fullpose_audit", {}).get("m1_stability", {})
        frames, observed = audit.get("frames"), audit.get("original_observed_points")
        pose = observed/(frames*17) if type(frames) is int and type(observed) is int and frames == raw["decoded_frames"] and frames > 0 and 0 <= observed <= frames*17 else None
        return QualityIndicators(pose_visible_fraction=pose)
    branch = {}
""",
    )
    patch(
        "scripts/module_slots_worker.py",
        """str(root / "experiments/module_slots_v1")]""",
        """str(root / "experiments/module_slots_v1"), str(root / "experiments/m4_policy_v1")]""",
    )
    patch(
        "scripts/module_slots_worker.py",
        """        from slots_pipeline import run_stage""",
        """        from stages import run_stage""",
    )
    changed = [
        "backend/cpr_agent/modules/members.py",
        "backend/cpr_agent/modules/contracts.py",
        "backend/cpr_agent/models.py",
        "backend/cpr_agent/config.py",
        "backend/cpr_agent/runtime.py",
        "backend/cpr_agent/module_runtime.py",
        "backend/cpr_agent/graph.py",
        "backend/cpr_agent/context_graph.py",
        "backend/cpr_agent/vision.py",
        "backend/cpr_agent/evidence.py",
        "backend/cpr_agent/branch_vision.py",
        "backend/cpr_agent/m3_quality.py",
        "backend/cpr_agent/m4_lineage.py",
        "backend/cpr_agent/m4_controller.py",
        "scripts/module_slots_worker.py",
    ]
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", *[str(TARGET / p) for p in changed]],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", *[str(TARGET / p) for p in changed], "--fix"],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )
    profile["module_slots"]["M4"] = "goal-cost-dag-v1"
    profile["m4_routes"] = {
        "full-v15": {"supported_label_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]},
        "body-qualified-v1": {"supported_label_ids": [3, 4, 8, 11]},
    }
    for name in changed + [
        "experiments/m4_policy_v1/algorithm.py",
        "experiments/m4_policy_v1/module.py",
        "experiments/m4_policy_v1/stages.py",
        "data/m4-scheduling-v1/parameters.json",
        "data/m4-scheduling-v1/protocol.json",
        "data/m4-scheduling-v1/decision.json",
    ]:
        profile["sources"][name] = sha(TARGET / name)
    profile["scope"] = (
        "Accepted M1-M3 numerical algorithms; M4 certified-scope DAG cost planning and bound cache reuse; M5 baseline. Partial output explicitly scoped."
    )
    destination = TARGET / "data/m4-scheduling-v1/pipeline-manifest.json"
    destination.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    record = {
        "workspace": str(TARGET),
        "parent_workspace": str(SOURCE),
        "parent_profile_sha256": sha(parent),
        "profile_sha256": sha(destination),
        "entrypoint": "scripts/start_m4.ps1",
        "graph_version": "cpr-review-12-v0.7-routing",
        "runtime_data": str(TARGET / "data/m4-scheduling-v1/runtime"),
        "env_copied": False,
        "module_slots": profile["module_slots"],
    }
    (TARGET / "workspace.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
