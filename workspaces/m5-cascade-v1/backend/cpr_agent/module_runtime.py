"""Controller adapters for M3/M4. Algorithms receive data contracts, never a Store or Settings."""

import json
import time

from .evidence import file_hash, read_bound_artifact
from .modules.contracts import (
    Assessment,
    Binding,
    CalibrationInput,
    Decision,
    PolicyObservation,
    Prediction,
    QualityIndicators,
)
from .modules.registry import build_registry
from .vision import final_tool, validate_receipt


def selection(settings):
    if settings.vision_mode != "real" or settings.vision_pipeline != "branches":
        return None
    return json.loads(settings.branch_manifest.read_text(encoding="utf-8")).get("module_slots")


class SlotAssessment:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings
        self.implementation = selection(settings)["M3"]
        self.module = build_registry().create("M3", self.implementation)

    def evaluate(self, goal, receipts):
        predictions, quality = [], QualityIndicators()
        execution_id = receipts.get(final_tool(self.settings))
        if execution_id:
            execution = self.store.get_tool(execution_id)
            output = validate_receipt(execution["output"], self.store, self.settings)
            raw = read_bound_artifact(output, self.settings.data_dir)
            predictions = [
                Prediction(
                    prediction_id=output.execution_id,
                    binding=Binding(
                        clip_id=output.clip_id,
                        video_sha256=output.video_sha256,
                        bundle_sha256=output.bundle_sha256,
                        pipeline_sha256=file_hash(self.settings.branch_manifest),
                    ),
                    supported_label_ids=output.supported_label_ids,
                    flagged_label_ids=output.predicted_label_ids,
                    score_type="uncalibrated_score",
                    scores={s.original_label_id: s.uncalibrated_score for s in output.scores},
                    thresholds={s.original_label_id: s.threshold for s in output.scores},
                    evidence_ids=[output.artifact_sha256],
                )
            ]
            quality = QualityIndicators(
                rgb_usable_samples=raw.get("rgb_audit", {}).get("frames"),
                rgb_requested_samples=raw.get("rgb_audit", {}).get("requested_frames"),
                hand_shape_available=raw.get("hand_shape_audit", {}).get("available"),
            )
        if execution_id and self.implementation == "source-quality-selective-v1":
            from .m3_quality import bound_quality

            quality = bound_quality(self.store, self.settings, execution, output, raw)
        targets = goal.target_label_ids or sorted(
            {label for p in predictions for label in p.supported_label_ids}
        )
        request = CalibrationInput(target_label_ids=targets, predictions=predictions, quality=quality)
        result = Assessment.model_validate(self.module.assess(request))
        if set(result.states) != set(targets) or result.source_prediction_ids != [
            p.prediction_id for p in predictions
        ]:
            raise ValueError("MODULE_ASSESSMENT_BINDING_MISMATCH")
        if result.calibration_profile != self.implementation:
            raise ValueError("MODULE_ASSESSMENT_PROFILE_MISMATCH")
        covered = {label for p in predictions for label in p.supported_label_ids}
        if any(state != "not_evaluated" and label not in covered for label, state in result.states.items()):
            raise ValueError("MODULE_ASSESSMENT_WITHOUT_EVIDENCE")
        return result


class SlotPolicy:
    def __init__(self, settings, store=None):
        self.settings, self.store = settings, store
        self.implementation = selection(settings)["M4"]
        self.module = build_registry().create("M4", self.implementation)

    def decide(self, goal, receipts, failed, assessment, run):
        available = ["STOP"]
        if goal.analysis_mode != "existing_only" and final_tool(self.settings) not in receipts and not failed:
            available.insert(0, "run_full_pipeline")
        observation = PolicyObservation(
            target_label_ids=goal.target_label_ids,
            analysis_mode=goal.analysis_mode,
            assessment=Assessment.model_validate(assessment) if assessment else None,
            available_actions=available,
            available_profile_ids=["full-v15"] if len(available) > 1 else [],
            executed_actions=list(receipts),
            failed_actions=list(failed),
            remaining_budget_seconds=max(0, (run.get("deadline") or time.time()) - time.time()),
        )
        if self.implementation == "consent-coverage-dag-v2":
            from .m4_controller import observe

            observation = observe(
                self.store,
                self.settings,
                goal,
                receipts,
                failed,
                assessment,
                run,
                observation.remaining_budget_seconds,
            )
        result = Decision.model_validate(self.module.choose(observation))
        if result.action not in observation.available_actions:
            raise ValueError("MODULE_ACTION_NOT_AVAILABLE")
        if result.action == "STOP" and result.profile_id is not None:
            raise ValueError("MODULE_STOP_HAS_PROFILE")
        if result.action != "STOP" and result.profile_id not in observation.available_profile_ids:
            raise ValueError("MODULE_ACTION_PROFILE_NOT_AVAILABLE")
        return result
