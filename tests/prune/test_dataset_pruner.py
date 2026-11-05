# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pytest

from bmiemg.prune import DatasetPruner
from bmiemg.prune.domain import DataActor

from prune_fakes import FakeActor, SeedStep, TraceStep


# ================================================================
# 1. Section: Service wiring
# ================================================================
def test_run_applies_custom_steps_in_order(tmp_path) -> None:
    steps = [SeedStep(), TraceStep("a"), TraceStep("b"), TraceStep("c")]
    pruner = DatasetPruner(source_name=[], custom_steps=steps, _data_dir=tmp_path)

    actors = pruner.run(protocol_name=None)

    assert actors == [FakeActor(trace=("a", "b", "c"))]


def test_apply_feeds_each_step_the_previous_output() -> None:
    pruner = DatasetPruner(source_name=[])
    start: list[DataActor] = [FakeActor()]

    actors = pruner.apply(start, [TraceStep("x"), TraceStep("y")])

    assert actors == [FakeActor(trace=("x", "y"))]
    assert start == [FakeActor()], "steps must not mutate the input actors"


def test_run_without_protocol_or_steps_raises(tmp_path) -> None:
    pruner = DatasetPruner(source_name=[], _data_dir=tmp_path)
    with pytest.raises(ValueError):
        pruner.run(protocol_name=None)
