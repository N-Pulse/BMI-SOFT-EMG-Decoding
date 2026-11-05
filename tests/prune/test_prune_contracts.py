# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pytest
import dataclasses

import prune_fakes  # noqa: F401
import bmiemg.prune  # noqa: F401

from pathlib import Path
from bmiemg.prune.adapters.file_loaders import FileLoaderRegistry
from bmiemg.prune.adapters.exporters import Writer, WriterRegistry
from bmiemg.prune.domain import DataActor, PruneStep, StepRegistry
from bmiemg.prune.adapters.file_loaders.file_loader import FileLoader

from contracts.assertions import (
    assert_implements,
    assert_is_dataclass,
    assert_name_classvar,
)
from contracts.discovery import leaf_subclasses, registry_keys
from prune_samples import STEP_SAMPLE_CONFIGS, WRITER_SAMPLE_CONFIGS
from contracts.samples import init_field_names, sample_for, sentinel_instance

STEP_KEYS = registry_keys(StepRegistry)
LOADER_KEYS = registry_keys(FileLoaderRegistry)
WRITER_KEYS = registry_keys(WriterRegistry)
ACTORS = leaf_subclasses(DataActor)


# ================================================================
# 1. Section: PruneStep
# ================================================================
@pytest.mark.parametrize("key", STEP_KEYS)
class TestPruneStepContract:
    def test_is_a_prune_step_dataclass(self, key: str) -> None:
        cls = StepRegistry.get_type(key)
        assert_is_dataclass(cls)
        assert_implements(cls, PruneStep, ["apply"])

    def test_name_matches_registry_key(self, key: str) -> None:
        assert_name_classvar(StepRegistry.get_type(key), expected=key)

    def test_from_config_builds_it(self, key: str) -> None:
        cls = StepRegistry.get_type(key)
        config = sample_for(STEP_SAMPLE_CONFIGS, key, "tests/prune/prune_samples.py")
        step = cls.from_config(dict(config))
        assert type(step) is cls

    def test_sample_config_keys_are_dataclass_fields(self, key: str) -> None:
        # Dataclass fields are the YAML keys: an unknown key means a stale sample or a renamed field.
        cls = StepRegistry.get_type(key)
        config = sample_for(STEP_SAMPLE_CONFIGS, key, "tests/prune/prune_samples.py")
        unknown = set(config) - set(init_field_names(cls))
        assert not unknown, f"{key!r} config keys {unknown} are not fields of {cls}"

    def test_apply_returns_a_list(self, key: str) -> None:
        cls = StepRegistry.get_type(key)
        config = sample_for(STEP_SAMPLE_CONFIGS, key, "tests/prune/prune_samples.py")
        assert isinstance(cls.from_config(dict(config)).apply([]), list)


# ================================================================
# 2. Section: DataActor
# ================================================================
@pytest.mark.parametrize("cls", ACTORS, ids=lambda cls: cls.__qualname__)
class TestDataActorContract:
    def test_is_a_dataclass(self, cls: type) -> None:
        assert_is_dataclass(cls)

    def test_has_name(self, cls: type) -> None:
        assert_name_classvar(cls)

    def test_copy_with_returns_a_changed_copy(self, cls: type) -> None:
        fields = init_field_names(cls)
        if not fields:
            pytest.skip(f"{cls.__qualname__} has no fields to change")
        original = sentinel_instance(cls)
        before = {f: getattr(original, f) for f in fields}
        changed, new_value = fields[0], object()

        copy = original.copy_with(**{changed: new_value})

        assert type(copy) is cls and copy is not original
        assert getattr(copy, changed) is new_value
        assert {
            f: getattr(original, f) for f in fields
        } == before, f"{cls.__qualname__}.copy_with mutated the original"


def test_actor_names_are_unique() -> None:
    names = [getattr(cls, "name", None) for cls in ACTORS]
    named = [name for name in names if name is not None]
    assert len(named) == len(set(named)), f"duplicate actor names: {named}"


# ================================================================
# 3. Section: FileLoader
# ================================================================
@pytest.mark.parametrize("key", LOADER_KEYS)
class TestFileLoaderContract:
    def test_key_is_a_lowercase_suffix(self, key: str) -> None:
        # Loader matches files by Path.suffix.lower(); any other key is never picked.
        assert (
            key == Path(f"file{key}").suffix.lower()
        ), f"FileLoader key {key!r} must be a lowercase file suffix like '.xdf'"

    def test_implements_file_loader(self, key: str) -> None:
        assert_implements(FileLoaderRegistry.get_type(key), FileLoader, ["load_actors"])

    def test_builds_without_arguments(self, key: str) -> None:
        assert isinstance(FileLoaderRegistry.get(key), FileLoader)


# ================================================================
# 4. Section: Writer
# ================================================================
@pytest.mark.parametrize("key", WRITER_KEYS)
class TestWriterContract:
    def test_is_a_writer_dataclass(self, key: str) -> None:
        cls = WriterRegistry.get_type(key)
        assert_is_dataclass(cls)
        assert_implements(cls, Writer, ["write"])

    def test_name_matches_registry_key(self, key: str) -> None:
        assert_name_classvar(WriterRegistry.get_type(key), expected=key)

    def test_actor_names_a_known_actor(self, key: str) -> None:
        cls = WriterRegistry.get_type(key)
        known = {getattr(actor, "name", None) for actor in ACTORS}
        assert cls.actor in known, (
            f"{cls.__qualname__}.actor is {cls.actor!r}; Exporter matches it "
            f"against DataActor.name, known: {sorted(n for n in known if n)}"
        )

    def test_from_config_builds_it(self, key: str) -> None:
        cls = WriterRegistry.get_type(key)
        config = sample_for(WRITER_SAMPLE_CONFIGS, key, "tests/prune/prune_samples.py")
        writer = cls.from_config(dict(config), timestamp="2026-01-01_00-00-00")
        assert type(writer) is cls
        assert "timestamp" in {f.name for f in dataclasses.fields(cls)}
