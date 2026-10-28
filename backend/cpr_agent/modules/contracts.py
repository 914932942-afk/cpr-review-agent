"""Versioned module contracts; no web, database or ML dependency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

CONTRACT_VERSION = "cpr-modules-v1"
LABEL_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13)
RGB_LABEL_IDS = (1, 2, 5, 7, 9)
FindingState = Literal["model_flagged", "not_flagged", "uncertain", "not_evaluated"]
Action = Literal["run_full_pipeline", "run_skeleton", "run_rgb", "run_fusion", "STOP"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, revalidate_instances="always")


class Binding(ContractModel):
    clip_id: str = Field(min_length=1)
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pipeline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actor_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ArtifactRef(ContractModel):
    artifact_id: str = Field(min_length=1)
    clip_id: str
    kind: Literal["media", "pose", "feature"]
    profile_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Prediction(ContractModel):
    prediction_id: str = Field(min_length=1)
    binding: Binding
    supported_label_ids: list[int]
    flagged_label_ids: list[int]
    score_type: Literal["uncalibrated_score", "calibrated_probability", "logit"]
    scores: dict[int, FiniteFloat]
    thresholds: dict[int, FiniteFloat] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self):
        supported = set(self.supported_label_ids)
        if len(supported) != len(self.supported_label_ids) or not supported <= set(LABEL_IDS):
            raise ValueError("Invalid supported labels")
        if set(self.scores) != supported or not set(self.flagged_label_ids) <= supported:
            raise ValueError("Missing scores are not negative decisions")
        if (
            len(set(self.flagged_label_ids)) != len(self.flagged_label_ids)
            or not set(self.thresholds) <= supported
        ):
            raise ValueError("Invalid decisions/threshold scope")
        if self.score_type != "logit" and any(
            not 0 <= x <= 1 for x in [*self.scores.values(), *self.thresholds.values()]
        ):
            raise ValueError("Bounded score required")
        return self


class QualityIndicators(ContractModel):
    pose_visible_fraction: float | None = Field(default=None, ge=0, le=1)
    rgb_usable_samples: int | None = Field(default=None, ge=0)
    rgb_requested_samples: int | None = Field(default=None, ge=0)
    hand_shape_available: bool | None = None
    extras: dict[str, FiniteFloat] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent(self):
        if self.rgb_usable_samples is not None and self.rgb_requested_samples is not None:
            if self.rgb_usable_samples > self.rgb_requested_samples:
                raise ValueError("Usable sample count exceeds requested count")
        return self


class CalibrationInput(ContractModel):
    target_label_ids: list[int]
    predictions: list[Prediction]
    quality: QualityIndicators

    @model_validator(mode="after")
    def coherent(self):
        if not set(self.target_label_ids) <= set(LABEL_IDS):
            raise ValueError("Unknown target")
        if self.predictions and any(p.binding != self.predictions[0].binding for p in self.predictions):
            raise ValueError("Cross-input calibration")
        return self


class Assessment(ContractModel):
    states: dict[int, FindingState]
    source_prediction_ids: list[str]
    calibration_profile: str
    calibration_status: Literal["uncalibrated", "calibrated"]
    calibrated_probabilities: dict[int, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent(self):
        if not set(self.states) <= set(LABEL_IDS) or not set(self.calibrated_probabilities) <= set(
            self.states
        ):
            raise ValueError("Invalid assessment scope")
        if self.calibration_status == "uncalibrated" and self.calibrated_probabilities:
            raise ValueError("Raw scores cannot be represented as calibrated probabilities")
        if any(not 0 <= x <= 1 for x in self.calibrated_probabilities.values()):
            raise ValueError("Invalid calibrated probability")
        if any(self.states[label] == "not_evaluated" for label in self.calibrated_probabilities):
            raise ValueError("Unassessed category cannot have a calibrated probability")
        return self


class PolicyObservation(ContractModel):
    target_label_ids: list[int]
    analysis_mode: Literal["adaptive", "existing_only"]
    assessment: Assessment | None = None
    available_actions: list[Action]
    available_profile_ids: list[str]
    executed_actions: list[str] = Field(default_factory=list)
    failed_actions: list[str] = Field(default_factory=list)
    remaining_budget_seconds: float = Field(ge=0)


class Decision(ContractModel):
    action: Action
    profile_id: str | None = None
    reason: str = Field(min_length=1, max_length=1000)


@dataclass(frozen=True)
class Readout:
    prediction: Prediction
    arrays: dict[str, Any]


@dataclass(frozen=True)
class RGBInput:
    binding: Binding
    execution_id: str
    media: ArtifactRef
    pose: ArtifactRef
    video_path: Path
    points: Any
    confidence: Any
    decoded_frames: int
    reported_fps: float
    sampling_profile: str = "uniform16-v15"

    def __post_init__(self):
        if (
            self.media.clip_id != self.binding.clip_id
            or self.pose.clip_id != self.binding.clip_id
            or self.media.sha256 != self.binding.video_sha256
            or self.pose.sha256 != self.binding.actor_source_sha256
            or self.media.profile_id != self.pose.profile_id
            or not self.media.profile_id.endswith(self.binding.pipeline_sha256)
        ):
            raise ValueError("RGB media/pose binding mismatch")
        if getattr(self.points, "shape", None) != (self.decoded_frames, 17, 2) or getattr(
            self.confidence, "shape", None
        ) != (self.decoded_frames, 17):
            raise ValueError("RGB reference layout mismatch")


@dataclass(frozen=True)
class FusionInput:
    binding: Binding
    execution_id: str
    skeleton: Readout
    rgb: Readout
    quality: QualityIndicators

    def __post_init__(self):
        if self.skeleton.prediction.binding != self.binding or self.rgb.prediction.binding != self.binding:
            raise ValueError("Cross-video, cross-profile or cross-actor fusion")


@dataclass(frozen=True)
class MLResult:
    prediction: Prediction
    arrays: dict[str, Any]
    audit: dict[str, Any]
