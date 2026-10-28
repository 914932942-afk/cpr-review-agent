"""M4 · Mingmin Guo · Goal/cost-aware Agent tool scheduling and stopping."""

from typing import Protocol

from .contracts import Decision, PolicyObservation


class SchedulingModule(Protocol):
    def choose(self, observation: PolicyObservation) -> Decision: ...
