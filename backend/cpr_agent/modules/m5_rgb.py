"""M5 · Xiaoyu Li · Lightweight RGB recognition and budgeted sampling."""

from typing import Protocol

from .contracts import MLResult, RGBInput


class RGBModule(Protocol):
    def predict(self, request: RGBInput) -> MLResult: ...
