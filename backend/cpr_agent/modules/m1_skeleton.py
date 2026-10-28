"""M1 · Jinlai GE · Skeleton quality processing and stable recognition."""

from typing import Protocol

from .contracts import ArtifactRef, Prediction


class SkeletonModule(Protocol):
    def predict(self, pose: ArtifactRef) -> Prediction: ...
