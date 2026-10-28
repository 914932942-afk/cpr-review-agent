"""Controller-only conversation contracts; independent of the frozen ML pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .models import CANDIDATE_LABELS, ResumeRequest, ReviewGoal, StrictModel, TurnCreate

ScopeReference = Literal["explicit", "previous_scope", "single_finding", "ambiguous"]
SourceReference = Literal["current_clip", "viewed_report", "ambiguous"]


class GoalReference(StrictModel):
    scope: ScopeReference = "explicit"
    source: SourceReference = "current_clip"


class ContextSelection(StrictModel):
    source_report_id: str | None = Field(default=None, min_length=1, max_length=120)
    reference: GoalReference = Field(default_factory=GoalReference)


class ContextTurnCreate(TurnCreate):
    context: ContextSelection | None = None


class ContextConfirmation(StrictModel):
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    use_bound_source: Literal[True]


class ContextResumeRequest(ResumeRequest):
    context_confirmation: ContextConfirmation | None = None


class SourceAnchor(StrictModel):
    session_id: str
    report_id: str
    report_run_id: str
    execution_id: str
    profile_id: str
    video_sha256: str
    bundle_sha256: str
    artifact_sha256: str
    label_ids: list[int]


class ContextSnapshot(StrictModel):
    version: Literal["bound-v1"] = "bound-v1"
    snapshot_id: str = ""
    session_id: str
    run_id: str
    clip_id: str | None
    asset_id: str | None = None
    video_name: str | None = None
    video_sha256: str | None = None
    bundle_id: str
    source: SourceAnchor | None = None
    source_status: Literal["available", "unavailable", "none"] = "none"
    source_error_code: str | None = None
    source_selection: Literal["viewed_report", "latest_confirmed", "none"] = "none"
    active_label_ids: list[int] = Field(default_factory=list)
    recent_goals: list[ReviewGoal] = Field(default_factory=list, max_length=4)
    recent_turn_ids: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def consistent(self):
        if any(label not in CANDIDATE_LABELS for label in self.active_label_ids):
            raise ValueError("Unknown context label")
        if (self.source is None) != (self.source_status == "none"):
            raise ValueError("Source status and anchor disagree")
        return self


class ContextReceipt(StrictModel):
    version: Literal["bound-v1"] = "bound-v1"
    snapshot_id: str
    session_id: str
    clip_id: str | None
    video_name: str | None
    video_sha256: str | None
    source_report_id: str | None
    source_execution_id: str | None
    source_status: str
    goal_confirmed: bool
    scope_reference: ScopeReference
    target_label_ids: list[int]
    result_execution_id: str | None = None
    result_report_id: str | None = None
    reused_result: bool = False
