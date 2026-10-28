from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .modules.contracts import Assessment

CANDIDATE_LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13)
TERMINAL = {"completed", "partially_completed", "rejected", "failed", "cancelled"}
NODE_NAMES = {
    "N01": "Load execution context",
    "N02": "Resolve and validate goal",
    "N03": "Await user input",
    "N04": "Prepare review context",
    "N05": "Assess available evidence",
    "N06": "Agent tool scheduling and planning",
    "N07": "Dispatch one tool step",
    "N08": "Await tool result",
    "N09": "Collect and validate result",
    "N10": "Freeze response scope",
    "N11": "Build and validate response",
    "N12": "Commit run",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Intent(StrEnum):
    ANALYSE = "analyse"
    VERIFY = "verify"
    EXPLAIN = "explain"
    EXAMPLE = "show_example"
    SUMMARY = "summarise"
    MEMORY = "manage_memory"
    FEEDBACK = "record_feedback"


class MemoryCommand(StrictModel):
    operation: Literal["remember_focus", "forget_preference", "list_preferences"]
    label_ids: list[int] = Field(default_factory=list, max_length=11)

    @model_validator(mode="after")
    def valid_labels(self):
        if any(i not in CANDIDATE_LABELS for i in self.label_ids):
            raise ValueError("Unknown candidate label")
        if self.operation == "remember_focus" and not self.label_ids:
            raise ValueError("remember_focus needs label_ids")
        self.label_ids = sorted(set(self.label_ids))
        return self


class OutputOptions(StrictModel):
    include_examples: bool = False
    include_summary: bool = True
    example_scope: Literal["general", "category"] = "general"


class ReviewGoal(StrictModel):
    schema_version: Literal["3.0.0"] = "3.0.0"
    intent: Intent
    target_label_ids: list[int] = Field(default_factory=list, max_length=11)
    analysis_mode: Literal["adaptive", "existing_only"] = "adaptive"
    allow_partial: bool = False
    analysis_budget_seconds: int | None = Field(default=None, ge=30, le=900)
    output_options: OutputOptions = Field(default_factory=OutputOptions)
    memory_commands: list[MemoryCommand] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def valid_goal(self):
        self.target_label_ids = sorted(set(self.target_label_ids))
        if any(i not in CANDIDATE_LABELS for i in self.target_label_ids):
            raise ValueError("Label is outside the scaffold routing catalogue")
        if self.intent in {Intent.ANALYSE, Intent.VERIFY} and not self.target_label_ids:
            raise ValueError("Analysis requires at least one target")
        if self.intent == Intent.EXAMPLE or self.output_options.include_examples:
            if self.output_options.example_scope == "category" and not self.target_label_ids:
                raise ValueError("Category example requires targets")
        if self.intent == Intent.MEMORY and not self.memory_commands:
            raise ValueError("Memory request needs an explicit command")
        return self


class SessionCreate(StrictModel):
    clip_id: str | None = None


class ClipCreate(StrictModel):
    asset_id: str
    start_seconds: float = Field(default=0, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def range_order(self):
        if self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must exceed start_seconds")
        return self


class TurnCreate(StrictModel):
    message: str = Field(default="", max_length=2000)
    goal: ReviewGoal | None = None

    @model_validator(mode="after")
    def has_input(self):
        if not self.message.strip() and self.goal is None:
            raise ValueError("Provide a message or a structured goal")
        return self


class ResumeRequest(StrictModel):
    kind: Literal["user_answer"] = "user_answer"
    wait_id: str
    interrupt_id: str
    revision: int = Field(ge=1)
    goal: ReviewGoal


class ToolRequest(StrictModel):
    execution_id: str
    run_id: str
    clip_id: str
    tool_id: Literal["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict", "video.analyze"]
    profile_id: str = Field(
        default="placeholder-v1",
        pattern=r"^(placeholder-v1|single-view-v15|single-view-v15-branches-v1:[0-9a-f]{64})$",
    )
    input_bindings: dict[str, str]
    request_hash: str
    attempt_token: str


class PlaceholderReceipt(StrictModel):
    output_kind: Literal["placeholder"] = "placeholder"
    is_stub: Literal[True] = True
    execution_id: str
    tool_id: str
    module_ids: list[str]
    message: str = "Contract exercised. No model, keypoints, features, scores or CPR evidence produced."


class ScaffoldResponse(StrictModel):
    mode: Literal["scaffold"] = "scaffold"
    kind: Literal["scaffold_receipt", "facts", "memory_receipt", "error"]
    title: str
    message: str
    workflow_reached_end: bool = True
    analysis_status: Literal["not_implemented", "not_requested"]
    classification_claims: list[Any] = Field(default_factory=list, max_length=0)
    target_states: dict[str, Literal["not_evaluated"]] = Field(default_factory=dict)
    memory_receipts: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ClipScore(StrictModel):
    original_label_id: int
    uncalibrated_score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)


class BranchReceipt(StrictModel):
    output_kind: Literal["branch_artifact"] = "branch_artifact"
    is_stub: Literal[False] = False
    execution_id: str
    tool_id: Literal["pose.extract", "skeleton.predict", "rgb.predict"]
    clip_id: str
    asset_id: str
    profile_id: str = Field(pattern=r"^single-view-v15-branches-v1:[0-9a-f]{64}$")
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_rel: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_seconds: float = Field(ge=0)


class ClipPrediction(StrictModel):
    output_kind: Literal["clip_prediction"] = "clip_prediction"
    is_stub: Literal[False] = False
    execution_id: str
    clip_id: str
    asset_id: str
    profile_id: str = Field(
        default="single-view-v15", pattern=r"^(single-view-v15|single-view-v15-branches-v1:[0-9a-f]{64})$"
    )
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_rel: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decoded_frames: int = Field(ge=8, le=900)
    reported_fps: float = Field(gt=0, le=240)
    predicted_label_ids: list[int]
    supported_label_ids: list[int]
    scores: list[ClipScore]
    analysis_seconds: float = Field(ge=0)
    evidence_scope: Literal["whole_clip"] = "whole_clip"
    analysis_route: Literal["full-v15", "body-qualified-v1"] = "full-v15"

    @model_validator(mode="after")
    def consistent(self):
        expected_scope = list(CANDIDATE_LABELS) if self.analysis_route == "full-v15" else [3, 4, 8, 11]
        if sorted(self.supported_label_ids) != expected_scope:
            raise ValueError("Unsupported label catalogue")
        if len(set(self.predicted_label_ids)) != len(self.predicted_label_ids):
            raise ValueError("Duplicate predictions")
        if not set(self.predicted_label_ids) <= set(self.supported_label_ids):
            raise ValueError("Prediction outside supported labels")
        if sorted(s.original_label_id for s in self.scores) != expected_scope:
            raise ValueError("Missing or duplicate scores")
        return self


class ModelClaim(StrictModel):
    label_id: int
    name: str
    state: Literal["model_flagged", "not_flagged"]
    uncalibrated_score: float = Field(ge=0, le=1)
    source_execution_id: str
    evidence_scope: Literal["whole_clip"] = "whole_clip"


class FindingEvidence(ModelClaim):
    threshold: float = Field(ge=0, le=1)
    score_meets_threshold: bool
    decision_note: str


class SkeletonProcessingEvidence(StrictModel):
    method: Literal["cpr-m1-stability-v3"]
    frames: int = Field(ge=8, le=900)
    geometry_available: bool
    original_observed_points: int = Field(ge=0)
    isolated_points_corrected: int = Field(ge=0)
    short_gap_points_reconstructed: int = Field(ge=0)
    unavailable_points_preserved: int = Field(ge=0)
    reconstructed_points_are_observations: Literal[False] = False


class ReviewEvidence(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    displayed_label_ids: list[int]
    assessed_label_ids: list[int]
    flagged_label_ids: list[int]
    source_run_id: str
    source_execution_id: str
    clip_id: str
    asset_id: str
    video_sha256: str
    bundle_sha256: str
    artifact_sha256: str
    evidence_scope: Literal["whole_clip"] = "whole_clip"
    nominal_duration_seconds: float = Field(gt=0)
    source_finished_utc: str | None = None
    rgb_usable_samples: int | None = Field(default=None, ge=0)
    rgb_requested_samples: int | None = Field(default=None, ge=0)
    hand_usable_samples: int | None = Field(default=None, ge=0)
    hand_requested_samples: int | None = Field(default=None, ge=0)
    hand_shape_available: bool | None = None
    skeleton_processing: SkeletonProcessingEvidence | None = None
    findings: list[FindingEvidence]


class ReviewResponse(StrictModel):
    mode: Literal["experimental"] = "experimental"
    kind: Literal["analysis_result", "facts", "memory_receipt", "error"]
    title: str
    message: str
    workflow_reached_end: bool = True
    analysis_status: Literal["completed_experimental", "not_requested", "unavailable"]
    classification_claims: list[ModelClaim] = Field(default_factory=list)
    target_states: dict[str, Literal["model_flagged", "not_flagged", "uncertain", "not_evaluated"]] = Field(
        default_factory=dict
    )
    prediction: ClipPrediction | None = None
    review: ReviewEvidence | None = None
    module_assessment: Assessment | None = None
    scheduling: dict[str, Any] | None = None
    visual_processing: dict[str, Any] | None = None
    reused_result: bool = False
    memory_receipts: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


IdempotencyKey = Annotated[str, Field(min_length=8, max_length=120)]
