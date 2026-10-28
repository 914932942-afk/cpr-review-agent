"""M5 · Xiaoyu Li · Quality-gated, dependency-aware lightweight visual cascade."""

from typing import Protocol

from .contracts import MLResult, RGBInput


class RGBModule(Protocol):
    def predict(self, request: RGBInput) -> MLResult: ...
