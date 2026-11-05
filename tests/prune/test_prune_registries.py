# ================================================================
# 0. Section: IMPORTS
# ================================================================
import bmiemg.prune  # noqa: F401

from dataclasses import dataclass
from bmiemg.prune.domain import PruneStep, StepRegistry
from bmiemg.prune.adapters.file_loaders import FileLoaderRegistry
from bmiemg.prune.adapters.exporters import Writer, WriterRegistry
from bmiemg.prune.adapters.file_loaders.file_loader import FileLoader

from contracts.discovery import registry_keys
from contracts.registry_contract import RegistryContract


# ================================================================
# 1. Section: Registry contracts
# ================================================================
class TestStepRegistry(RegistryContract):
    registry = StepRegistry

    def make_entry(self) -> type:
        @dataclass
        class Probe(PruneStep):
            name = "probe"

            def apply(self, actors):
                return actors

        return Probe


class TestFileLoaderRegistry(RegistryContract):
    registry = FileLoaderRegistry

    def make_entry(self) -> type:
        class Probe(FileLoader):
            def load_actors(self, path):
                raise NotImplementedError

        return Probe


class TestWriterRegistry(RegistryContract):
    registry = WriterRegistry

    def make_entry(self) -> type:
        # Registry.get builds with no args, so the probe needs a default timestamp.
        @dataclass
        class Probe(Writer):
            name = "probe"
            actor = "probe"
            timestamp: str = "t"

            def write(self, actor, dest_dir):
                pass

        return Probe


def test_prune_families_do_not_share_entries(register_temporarily) -> None:
    families = [StepRegistry, FileLoaderRegistry, WriterRegistry]
    for family in families:
        register_temporarily(family, f"only_in_{family.__name__}", object)
    for family in families:
        for other in families:
            key = f"only_in_{other.__name__}"
            assert (key in registry_keys(family)) is (
                family is other
            ), f"{key!r} leaked into {family.__name__}"
