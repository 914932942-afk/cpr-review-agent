"""M2 · Wenbo Zhang · Class/quality-aware RGB–skeleton score fusion."""

from typing import Protocol

from .contracts import FusionInput, MLResult


class FusionModule(Protocol):
    def fuse(self, request: FusionInput) -> MLResult: ...
