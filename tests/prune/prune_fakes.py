# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

from bmiemg.prune.domain import DataActor, PruneStep


# ================================================================
# 1. Section: Fakes
# ================================================================
@dataclass
class FakeActor(DataActor):
    name: ClassVar[str] = "fake_actor"
    trace: tuple[str, ...] = ()


@dataclass
class TraceStep(PruneStep[FakeActor, FakeActor]):
    """Appends its label to every actor's trace, so the run order is observable."""

    name: ClassVar[str] = "trace"
    label: str

    def apply(self, actors: list[FakeActor]) -> list[FakeActor]:
        return [actor.copy_with(trace=(*actor.trace, self.label)) for actor in actors]


@dataclass
class SeedStep(PruneStep[FakeActor, FakeActor]):
    """Adds one fresh actor, since the pruner loads none without data files."""

    name: ClassVar[str] = "seed"

    def apply(self, actors: list[FakeActor]) -> list[FakeActor]:
        return [*actors, FakeActor()]
