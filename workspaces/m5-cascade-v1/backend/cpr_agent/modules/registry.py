"""Explicit registration. Configuration selects IDs, never arbitrary import paths."""

from dataclasses import dataclass
from typing import Callable

from .contracts import CONTRACT_VERSION

DEFAULT_SELECTIONS = {
    "M2": "baseline-v15",
    "M3": "baseline-uncalibrated-v1",
    "M4": "baseline-fixed-v1",
    "M5": "baseline-rgb-v15",
}
METHODS = {"M2": "fuse", "M3": "assess", "M4": "choose", "M5": "predict"}


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    implementation_id: str
    factory: Callable
    kind: str = "baseline"
    contract_version: str = CONTRACT_VERSION
    required_files: tuple[str, ...] = ()


class ModuleRegistry:
    def __init__(self):
        self.entries = {}

    def register(self, spec):
        key = (spec.module_id, spec.implementation_id)
        if spec.module_id not in METHODS or key in self.entries or spec.kind not in {"baseline", "improved"}:
            raise ValueError("MODULE_REGISTRATION_INVALID")
        if spec.contract_version != CONTRACT_VERSION:
            raise ValueError("MODULE_CONTRACT_VERSION_MISMATCH")
        self.entries[key] = spec

    def spec(self, module_id, implementation_id):
        if (module_id, implementation_id) not in self.entries:
            raise ValueError(f"MODULE_NOT_REGISTERED: {module_id}/{implementation_id}")
        return self.entries[module_id, implementation_id]

    def create(self, module_id, implementation_id):
        result = self.spec(module_id, implementation_id).factory()
        if not callable(getattr(result, METHODS[module_id], None)):
            raise ValueError("MODULE_METHOD_MISSING")
        return result

    def validate_selection(self, selection):
        if set(selection) != set(METHODS):
            raise ValueError("MODULE_SELECTION_REQUIRES_M2_M3_M4_M5")
        for module_id, implementation in selection.items():
            self.spec(module_id, implementation)
        return selection


def build_registry():
    from .baselines import BaselineCalibration, BaselineFusion, BaselineRGB, BaselineScheduling
    from .members import register_members

    registry = ModuleRegistry()
    for module_id, factory in [
        ("M2", BaselineFusion),
        ("M3", BaselineCalibration),
        ("M4", BaselineScheduling),
        ("M5", BaselineRGB),
    ]:
        registry.register(ModuleSpec(module_id, DEFAULT_SELECTIONS[module_id], factory))
    register_members(registry)
    return registry
