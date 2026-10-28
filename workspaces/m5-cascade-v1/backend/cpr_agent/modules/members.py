"""Accepted M2 registration in the isolated M2 workspace."""

import importlib.util
import sys
from pathlib import Path

from .registry import ModuleSpec

ROOT = Path(__file__).resolve().parents[3]


def create_m2():
    name = "cpr_m2_quality_member"
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/m2_competence_v3/module.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.QualityAwareFusion()


def _register_m2(registry):
    registry.register(
        ModuleSpec(
            "M2",
            "competence-scope-disagreement-v3",
            create_m2,
            kind="improved",
            required_files=(
                "experiments/m2_competence_v3/module.py",
                "experiments/m2_competence_v3/fusion_algorithm.py",
                "data/m1-m2-exploration-v1/m2-competence/protocol.json",
                "data/m1-m2-exploration-v1/m2-competence/selection.json",
                "data/m1-m2-exploration-v1/m2-competence/decision.json",
            ),
        )
    )


def create_m3():
    spec = importlib.util.spec_from_file_location(
        "cpr_m3_member", ROOT / "experiments/m3_calibration_v1/module.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SourceQualityCalibration()


def _register_m3(registry):
    _register_m2(registry)
    registry.register(
        ModuleSpec(
            "M3",
            "source-quality-selective-v1",
            create_m3,
            kind="improved",
            required_files=(
                "experiments/m3_calibration_v1/algorithm.py",
                "experiments/m3_calibration_v1/module.py",
                "backend/cpr_agent/m3_quality.py",
                "data/m3-calibration-v1/protocol.json",
                "data/m3-calibration-v1/selection.json",
                "data/m3-calibration-v1/decision.json",
                "data/m3-calibration-v1/upstream.json",
            ),
        )
    )


def create_m4():
    spec = importlib.util.spec_from_file_location(
        "cpr_m4_member", ROOT / "experiments/m4_policy_v2/module.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GoalCostPolicy()


def _register_m4(registry):
    _register_m3(registry)
    registry.register(
        ModuleSpec(
            "M4",
            "consent-coverage-dag-v2",
            create_m4,
            kind="improved",
            required_files=(
                "experiments/m4_policy_v2/algorithm.py",
                "experiments/m4_policy_v2/module.py",
                "experiments/m4_policy_v2/stages.py",
                "backend/cpr_agent/m4_lineage.py",
                "backend/cpr_agent/m4_controller.py",
                "data/m4-exploration-v2/protocol.json",
                "data/m4-exploration-v2/parameters.json",
                "data/m4-exploration-v2/decision.json",
            ),
        )
    )


def create_m5():
    spec = importlib.util.spec_from_file_location(
        "cpr_m5_member", ROOT / "experiments/m5_cascade_v1/module.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.QualityDependencyRGB()


def register_members(registry):
    _register_m4(registry)
    registry.register(
        ModuleSpec(
            "M5",
            "quality-dependency-cascade-v1",
            create_m5,
            kind="improved",
            required_files=(
                "experiments/m5_cascade_v1/algorithm.py",
                "experiments/m5_cascade_v1/runtime.py",
                "experiments/m5_cascade_v1/inference.py",
                "experiments/m5_cascade_v1/module.py",
                "backend/cpr_agent/m5_validation.py",
                "data/m5-cascade-v1/protocol.json",
                "data/m5-cascade-v1/certificate.json",
                "data/m5-cascade-v1/decision.json",
            ),
        )
    )
