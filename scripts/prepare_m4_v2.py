"""Build an isolated candidate; accepted v1 remains byte-identical and runnable."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workspaces/m4-routing-v1"
TARGET = ROOT / "workspaces/m4-refined-v2"
DATA = ROOT / "data/m4-exploration-v2"
VERSION = "consent-coverage-dag-v2"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def patch(name, old, new, count=1):
    p = TARGET / name
    value = p.read_text(encoding="utf-8")
    if value.count(old) != count:
        raise ValueError("Unexpected patch: " + name + " " + old[:70])
    p.write_text(value.replace(old, new), encoding="utf-8")


def junction(source, target):
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "New-Item -ItemType Junction -Path '"
            + str(target)
            + "' -Target '"
            + str(source)
            + "' | Out-Null",
        ],
        check=True,
    )


def main():
    if TARGET.exists():
        raise FileExistsError("Preserve candidate workspace; do not overwrite")
    assert read(DATA / "offline.json")["passed"]
    parent = read(SOURCE / "data/m4-scheduling-v1/pipeline-manifest.json")
    assert all(sha(SOURCE / p) == h for p, h in parent["sources"].items())
    TARGET.mkdir()
    for folder in ["backend", "scripts", "tests", "fixtures", "frontend", "experiments"]:
        shutil.copytree(
            SOURCE / folder,
            TARGET / folder,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "node_modules", ".ruff_cache"),
        )
    for name in ["pyproject.toml", "README.md"]:
        shutil.copy2(ROOT / name, TARGET / name)
    junction(ROOT / "frontend/node_modules", TARGET / "frontend/node_modules")
    (TARGET / "data").mkdir()
    # Keep pinned inputs inside the workspace: path containment validation stays strict.
    # No parent runtime is copied, and only the new data directory owns runtime writes.
    for folder in (SOURCE / "data").iterdir():
        if folder.is_dir():
            shutil.copytree(
                folder, TARGET / "data" / folder.name, ignore=shutil.ignore_patterns("runtime", "__pycache__")
            )
    candidate = TARGET / "data/m4-exploration-v2"
    candidate.mkdir()
    for p in DATA.glob("*.json"):
        shutil.copy2(p, candidate / p.name)
    shutil.copytree(ROOT / "experiments/m4_policy_v2", TARGET / "experiments/m4_policy_v2")
    module = (
        (SOURCE / "experiments/m4_policy_v1/module.py")
        .read_text(encoding="utf-8")
        .replace("data/m4-scheduling-v1", "data/m4-exploration-v2")
    )
    (TARGET / "experiments/m4_policy_v2/module.py").write_text(module, encoding="utf-8")
    stages = (
        (SOURCE / "experiments/m4_policy_v1/stages.py")
        .read_text(encoding="utf-8")
        .replace("goal-cost-dag-v1", VERSION)
    )
    (TARGET / "experiments/m4_policy_v2/stages.py").write_text(stages, encoding="utf-8")
    for name in [
        "backend/cpr_agent/m4_lineage.py",
        "backend/cpr_agent/module_runtime.py",
        "backend/cpr_agent/graph.py",
        "backend/cpr_agent/branch_vision.py",
        "backend/cpr_agent/config.py",
        "backend/cpr_agent/modules/members.py",
    ]:
        p = TARGET / name
        p.write_text(
            p.read_text(encoding="utf-8")
            .replace("goal-cost-dag-v1", VERSION)
            .replace("experiments/m4_policy_v1/", "experiments/m4_policy_v2/")
            .replace("cpr-review-12-v0.7-routing", "cpr-review-12-v0.8-adaptive"),
            encoding="utf-8",
        )
    p = TARGET / "backend/cpr_agent/modules/members.py"
    p.write_text(
        p.read_text(encoding="utf-8").replace('"data/m4-scheduling-v1/', '"data/m4-exploration-v2/'),
        encoding="utf-8",
    )
    p = TARGET / "scripts/module_slots_worker.py"
    p.write_text(
        p.read_text(encoding="utf-8").replace("experiments/m4_policy_v1", "experiments/m4_policy_v2"),
        encoding="utf-8",
    )
    patch(
        "backend/cpr_agent/models.py",
        '    analysis_mode: Literal["adaptive", "existing_only"] = "adaptive"',
        '    analysis_mode: Literal["adaptive", "existing_only"] = "adaptive"\n    allow_partial: bool = False\n    analysis_budget_seconds: int | None = Field(default=None, ge=30, le=900)',
    )
    patch(
        "backend/cpr_agent/modules/contracts.py",
        "    covered_label_ids: list[int] = Field(default_factory=list)",
        '    allow_partial: bool = False\n    partial_result_delivered: bool = False\n    committed_partial_route: Literal["body-qualified-v1"] | None = None\n    covered_label_ids: list[int] = Field(default_factory=list)',
    )
    patch(
        "backend/cpr_agent/context_graph.py",
        '        decision = {\n            "reference":',
        """        # Only structured input or the confirmation form may grant partial-scope consent.
        if not confirmed and turn.get("goal") is None:
            goal = goal.model_copy(update={"allow_partial": False, "analysis_budget_seconds": None})
        if goal.intent in {Intent.ANALYSE, Intent.VERIFY} and goal.analysis_budget_seconds is not None:
            started = run["deadline"] - self.settings.service_budget_seconds
            self.store.update_run(run["id"], deadline=min(run["deadline"], started + goal.analysis_budget_seconds))
        decision = {
            "reference":""",
    )
    # Deadline anchor does not reset on replay/form reconfirmation.
    patch(
        "backend/cpr_agent/context_graph.py",
        '            started = run["deadline"] - self.settings.service_budget_seconds\n            self.store.update_run(run["id"], deadline=min(run["deadline"], started + goal.analysis_budget_seconds))',
        '            started = turn.get("m4_budget_started_at", run["deadline"] - self.settings.service_budget_seconds)\n            turn = {**turn, "m4_budget_started_at": started}\n            self.store.update_run(run["id"], deadline=min(run["deadline"], started + goal.analysis_budget_seconds))',
    )
    patch("backend/cpr_agent/m4_controller.py", "        and not failed\n", "")
    patch(
        "backend/cpr_agent/m4_controller.py",
        "    return PolicyObservation(",
        """    turn = store.get("turn", run["turn_id"])
    delivered = bool(
        goal.allow_partial and final_tool(settings) in receipts
        and store.get_tool(receipts[final_tool(settings)])["run_id"] == run["id"]
        and not set(goal.target_label_ids) <= set(covered)
    )
    return PolicyObservation(
        allow_partial=goal.allow_partial,
        partial_result_delivered=delivered,
        committed_partial_route=turn.get("m4_committed_partial_route"),""",
    )
    patch(
        "backend/cpr_agent/m4_controller.py",
        "def missing_tools(profile, receipts, covered, targets):",
        "def missing_tools(profile, receipts, covered, targets, allow_partial=False):",
    )
    patch(
        "backend/cpr_agent/m4_controller.py",
        'if profile == "body-qualified-v1" and not set(targets) <= set([3, 4, 8, 11]):',
        'if profile == "body-qualified-v1" and not set(targets) <= set([3, 4, 8, 11]) and not (allow_partial and set(targets) & {3,4,8,11}):',
    )
    patch(
        "backend/cpr_agent/graph.py",
        '            missing = missing_tools(decision.profile_id, state["receipts"], covered, goal.target_label_ids)',
        '            missing = missing_tools(decision.profile_id, state["receipts"], covered, goal.target_label_ids, goal.allow_partial)',
    )
    patch(
        "backend/cpr_agent/graph.py",
        '            schedule = dict(state.get("scheduling", {}))',
        """            schedule = dict(state.get("scheduling", {}))
            schedule["history"] = [*schedule.get("history", []), decision.model_dump(mode="json")][-6:]""",
    )
    patch(
        "backend/cpr_agent/graph.py",
        '            schedule["plan"] = decision.model_dump(mode="json")',
        """            selected = decision.audit.get("selected", {})
            if selected.get("partial_scope"):
                if not goal.allow_partial:
                    raise ValueError("M4_PARTIAL_SCOPE_NOT_AUTHORISED")
                run = self.store.get_run(state["run_id"])
                turn = self.store.get("turn", run["turn_id"])
                self.store.put("turn", {**turn, "m4_committed_partial_route": decision.profile_id})
                schedule["partial_scope_authorised"] = True
            schedule["plan"] = decision.model_dump(mode="json")""",
    )
    patch(
        "backend/cpr_agent/graph.py",
        '            state, "N07" if index < len(state["plan"]) else "N05", receipts=receipts, step_index=index',
        '            state, "N05" if getattr(self.policy, "implementation", None) == "'
        + VERSION
        + '" else "N07" if index < len(state["plan"]) else "N05", receipts=receipts, step_index=index',
    )
    # Pending profile is pinned only after all source/UI edits. It cannot be launched yet.
    parent["module_slots"]["M4"] = VERSION
    (candidate / "pending-profile.json").write_text(json.dumps(parent, indent=2), encoding="utf-8")
    (TARGET / "workspace.json").write_text(
        json.dumps(
            {
                "workspace": str(TARGET),
                "parent_workspace": str(SOURCE),
                "port": 8012,
                "graph_version": "cpr-review-12-v0.8-adaptive",
                "candidate_not_accepted": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Prepared isolated candidate: " + str(TARGET))


if __name__ == "__main__":
    main()
