"""Replace these adapters without moving API or graph responsibilities into algorithms."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings
from .models import BranchReceipt, ClipPrediction, PlaceholderReceipt, ReviewGoal, ToolRequest

TOOLS = {
    "pose.extract": [],
    "skeleton.predict": ["M1"],
    "rgb.predict": ["M5"],
    "fusion.predict": ["M2"],
}
MODULES = [
    {"id": "M1", "owner": "Jinlai GE", "name": "骨架质量处理与稳定识别", "location": "skeleton.predict"},
    {
        "id": "M2",
        "owner": "Wenbo Zhang",
        "name": "类别与质量感知的RGB—骨架输出级融合",
        "location": "fusion.predict",
    },
    {"id": "M3", "owner": "Wenxin Yan", "name": "质量条件置信度校准与拒判", "location": "N05"},
    {
        "id": "M4",
        "owner": "Mingmin Guo",
        "name": "目标与成本感知的Agent工具调度与停止决策",
        "location": "N06",
    },
    {"id": "M5", "owner": "Xiaoyu Li", "name": "轻量RGB识别与预算化采样", "location": "rgb.predict"},
]


@dataclass(frozen=True)
class GoalResolution:
    goal: ReviewGoal | None
    code: str
    called: bool = False
    duration_ms: float = 0
    reference: dict[str, str] | None = None

    def receipt(self):
        value = {
            "goal": self.goal.model_dump(mode="json") if self.goal else None,
            "code": self.code,
            "called": self.called,
            "duration_ms": self.duration_ms,
        }
        if self.reference is not None:
            value["reference"] = self.reference
        return value


class ModelGateway(Protocol):
    def parse_goal(self, message: str, context: dict[str, Any]) -> GoalResolution: ...
    def status(self) -> dict[str, Any]: ...


class DisabledModelGateway:
    """No heuristic pretending to be an LLM. Unstructured text enters the form fallback."""

    def __init__(self, key_configured=False):
        self.key_configured = key_configured

    def parse_goal(self, message: str, context: dict[str, Any]) -> GoalResolution:
        return GoalResolution(None, "LLM_DISABLED")

    def plan_explanation(self, facts):
        from .explanation_model import PlanResolution

        return PlanResolution(None, "LLM_DISABLED")

    def status(self):
        return {
            "mode": "disabled",
            "network_enabled": False,
            "key_configured": self.key_configured,
            "scope": "goal_parser_and_evidence_organisation",
            "connection_verified": False,
        }


class VisionTool(Protocol):
    def execute(self, request: ToolRequest) -> PlaceholderReceipt | BranchReceipt | ClipPrediction: ...


class StubVisionTool:
    def __init__(self, settings: Settings):
        self.settings = settings

    def execute(self, request: ToolRequest) -> PlaceholderReceipt:
        if request.tool_id not in TOOLS:
            raise ValueError("Tool is not registered")
        time.sleep(self.settings.stub_delay_seconds)
        if request.tool_id == self.settings.fail_tool:
            raise RuntimeError("Injected placeholder failure")
        return PlaceholderReceipt(
            execution_id=request.execution_id, tool_id=request.tool_id, module_ids=TOOLS[request.tool_id]
        )


class AssessmentPort(Protocol):
    def assess(self, goal: ReviewGoal, receipts: dict[str, str]) -> dict[str, str]: ...


class StubAssessment:
    def assess(self, goal: ReviewGoal, receipts: dict[str, str]) -> dict[str, str]:
        # Placeholder receipts are not Prediction objects and never become CPR evidence.
        return {str(label): "not_evaluated" for label in goal.target_label_ids}


class PolicyPort(Protocol):
    def choose(self, goal: ReviewGoal, receipts: dict[str, str], failed: list[str]) -> str: ...


class ScaffoldPolicy:
    """Fixed reachability policy for integration only. This is NOT the M4 research algorithm."""

    def choose(self, goal: ReviewGoal, receipts: dict[str, str], failed: list[str]) -> str:
        if goal.analysis_mode == "existing_only":
            return "STOP"
        for tool, action in [("skeleton.predict", "run_skeleton"), ("rgb.predict", "run_rgb")]:
            dependency_ok = tool != "skeleton.predict" or "pose.extract" not in failed
            if tool not in receipts and tool not in failed and dependency_ok:
                return action
        if {"skeleton.predict", "rgb.predict"} <= receipts.keys() and "fusion.predict" not in failed:
            if "fusion.predict" not in receipts:
                return "run_fusion"
        return "STOP"
