"""M3 · Wenxin Yan · Quality-conditioned confidence calibration and rejection."""

from typing import Protocol

from .contracts import Assessment, CalibrationInput


class CalibrationModule(Protocol):
    def assess(self, request: CalibrationInput) -> Assessment: ...
