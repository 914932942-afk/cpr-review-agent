"""New consent, budget, prefix and partial-status contracts; no vision execution."""

from types import SimpleNamespace

import pytest
from cpr_agent import graph as graph_module
from cpr_agent.graph import ReviewGraph
from cpr_agent.m4_controller import missing_tools
from cpr_agent.models import ReviewGoal
from cpr_agent.modules.contracts import Decision
from cpr_agent.review_context import ContextBinder
from cpr_agent.store import DomainError
from cpr_agent.vision import build_response
from pydantic import ValidationError
from test_m4_profile import prediction


def test_existing_goals_keep_complete_scope_by_default():
    goal = ReviewGoal(intent="analyse", target_label_ids=[4, 5])
    assert goal.allow_partial is False and goal.analysis_budget_seconds is None


@pytest.mark.parametrize("budget", [0, 29, 901, -1])
def test_requested_budget_is_bounded(budget):
    with pytest.raises(ValidationError):
        ReviewGoal(intent="analyse", target_label_ids=[4], analysis_budget_seconds=budget)


def test_partial_plan_requires_consent_and_useful_overlap():
    with pytest.raises(ValueError):
        missing_tools("body-qualified-v1", {}, [], [4, 5])
    with pytest.raises(ValueError):
        missing_tools("body-qualified-v1", {}, [], [5], True)
    assert missing_tools("body-qualified-v1", {"pose.extract": "p"}, [], [4, 5], True) == [
        "skeleton.predict",
        "fusion.predict",
    ]


def test_partial_selection_commits_scope_and_keeps_decision_history():
    graph = ReviewGraph.__new__(ReviewGraph)
    graph.settings = SimpleNamespace(vision_pipeline="branches", vision_mode="real")
    saved = []
    graph.store = SimpleNamespace(
        event=lambda *a, **kw: None,
        get_run=lambda rid: {"id": rid, "turn_id": "turn"},
        get=lambda *a: {"id": "turn"},
        put=lambda kind, v: saved.append((kind, v)),
    )
    decision = Decision(
        action="run_skeleton",
        profile_id="body-qualified-v1",
        reason="Explicit partial scope.",
        audit={"selected": {"partial_scope": True}},
    )
    graph.policy = SimpleNamespace(implementation="consent-coverage-dag-v2", decide=lambda *a: decision)
    state = {
        "run_id": "new",
        "goal": ReviewGoal(intent="analyse", target_label_ids=[4, 5], allow_partial=True).model_dump(),
        "receipts": {},
        "failed_tools": [],
        "context": {},
    }
    command = graph.n06(state)
    assert command.goto == "N07"
    assert saved[0][1]["m4_committed_partial_route"] == "body-qualified-v1"
    assert command.update["scheduling"]["partial_scope_authorised"] is True
    assert len(command.update["scheduling"]["history"]) == 1
    state["goal"]["allow_partial"] = False
    with pytest.raises(ValueError):
        graph.n06(state)


def test_every_successful_stage_reassesses_before_next_dispatch():
    graph = ReviewGraph.__new__(ReviewGraph)
    graph.settings = SimpleNamespace(vision_mode="stub", vision_pipeline="branches")
    execution = {
        "id": "pose",
        "status": "succeeded",
        "request": {"tool_id": "pose.extract", "run_id": "r", "clip_id": "c"},
        "output": {"is_stub": True, "execution_id": "pose"},
    }
    graph.store = SimpleNamespace(get_tool=lambda _: execution, get_run=lambda _: {"id": "r", "clip_id": "c"})
    graph.policy = SimpleNamespace(implementation="consent-coverage-dag-v2")
    state = {"run_id": "r", "execution_id": "pose", "receipts": {}, "step_index": 0, "plan": [{}, {}]}
    command = graph.n09(state)
    assert command.goto == "N05" and command.update["receipts"] == {"pose.extract": "pose"}


def test_partial_output_is_never_marked_complete(monkeypatch):
    graph = ReviewGraph.__new__(ReviewGraph)
    graph.settings = SimpleNamespace(vision_mode="real", vision_pipeline="branches")
    graph.policy = SimpleNamespace(implementation="consent-coverage-dag-v2")
    graph.store = SimpleNamespace(get_tool=lambda _: {"output": {}}, tools_for_run=lambda _: [])
    monkeypatch.setattr(
        graph_module, "validate_receipt", lambda *a: SimpleNamespace(supported_label_ids=[3, 4, 8, 11])
    )
    command = graph.n10(
        {
            "run_id": "r",
            "goal": ReviewGoal(intent="analyse", target_label_ids=[4, 5], allow_partial=True).model_dump(),
            "receipts": {"fusion.predict": "f"},
        }
    )
    assert command.update["final_status"] == "partially_completed" and command.update["error_code"] is None


@pytest.mark.parametrize(
    "status", ["completed", "partially_completed", "failed", "running", "rejected", "cancelled"]
)
def test_partial_source_is_eligible_but_unfinished_or_failed_source_is_not(status):
    session = {"id": "session", "clip_id": "clip", "bundle_id": "bundle"}
    response = build_response(ReviewGoal(intent="analyse", target_label_ids=[4, 5]), prediction()).model_dump(
        mode="json"
    )
    report = {"id": "report", "run_id": "run", "session_id": "session", "created_at": 1, "response": response}
    run = {
        "id": "run",
        "session_id": "session",
        "clip_id": "clip",
        "bundle_id": "bundle",
        "status": status,
        "report_id": "report",
        "response": response,
    }
    binder = ContextBinder(SimpleNamespace(get=lambda *a: report, get_run=lambda _: run), None)
    if status in {"completed", "partially_completed"}:
        assert binder.report(session, "report")[0]["id"] == "report"
    else:
        with pytest.raises(DomainError):
            binder.report(session, "report")
