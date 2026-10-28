"""Check profile pins/registration and optional schema export; never runs vision or an LLM."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cpr_agent.branch_vision import validate_pipeline
from cpr_agent.config import ROOT, Settings
from cpr_agent.modules.contracts import (
    Assessment,
    Binding,
    CalibrationInput,
    Decision,
    PolicyObservation,
    Prediction,
    QualityIndicators,
)
from cpr_agent.modules.registry import build_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=ROOT / "data/module-slots-v1/pipeline-manifest.json")
    parser.add_argument("--schemas", type=Path)
    args = parser.parse_args()
    settings = replace(
        Settings.from_env(), vision_mode="real", vision_pipeline="branches", branch_manifest=args.profile
    )
    digest = validate_pipeline(settings)
    selected = json.loads(args.profile.read_text(encoding="utf-8"))["module_slots"]
    registry = build_registry()
    assessment = Assessment.model_validate(
        registry.create("M3", selected["M3"]).assess(
            CalibrationInput(target_label_ids=[2], predictions=[], quality=QualityIndicators())
        )
    )
    assert (
        assessment.states == {2: "not_evaluated"}
        and assessment.source_prediction_ids == []
        and not assessment.calibrated_probabilities
    )
    decision = Decision.model_validate(
        registry.create("M4", selected["M4"]).choose(
            PolicyObservation(
                target_label_ids=[2],
                analysis_mode="existing_only",
                available_actions=["STOP"],
                available_profile_ids=[],
                remaining_budget_seconds=0,
            )
        )
    )
    assert decision.action == "STOP" and decision.profile_id is None
    if args.schemas:
        args.schemas.mkdir(parents=True, exist_ok=False)
        for cls in [
            Assessment,
            Binding,
            CalibrationInput,
            Decision,
            PolicyObservation,
            Prediction,
            QualityIndicators,
        ]:
            (args.schemas / (cls.__name__ + ".json")).write_text(
                json.dumps(cls.model_json_schema(), indent=2), encoding="utf-8"
            )
    print(
        json.dumps(
            {
                "passed": True,
                "pipeline_sha256": digest,
                "selected": selected,
                "scope": "Pins, registration and no-evidence/STOP contracts only; not algorithm-quality acceptance",
                "vision_calls": 0,
                "llm_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
