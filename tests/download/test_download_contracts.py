# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pytest

import bmiemg.download  # noqa: F401  # registers every strategy
from typing import ClassVar
from dataclasses import dataclass
from bmiemg.download import Downloader
from bmiemg.download.domain import DownloadSpec, DownloadStrategy, Registry

from contracts.discovery import generic_arg, registered_type, registry_keys
from contracts.assertions import assert_implements, assert_is_dataclass
from contracts.registry_contract import RegistryContract

STRATEGY_KEYS = registry_keys(Registry)


# ================================================================
# 1. Section: Fakes
# ================================================================
@dataclass
class FakeSpec(DownloadSpec):
    source: ClassVar[str] = "contract-fake"


@dataclass
class FakeStrategy(DownloadStrategy[FakeSpec]):
    calls: ClassVar[list[tuple[str, DownloadSpec]]] = []

    def validate(self, spec: FakeSpec) -> None:
        FakeStrategy.calls.append(("validate", spec))

    def fetch(self, spec: FakeSpec) -> None:
        FakeStrategy.calls.append(("fetch", spec))


@pytest.fixture
def fake_strategy(register_temporarily) -> type[FakeStrategy]:
    FakeStrategy.calls.clear()
    return register_temporarily(Registry, FakeSpec.source, FakeStrategy)


# ================================================================
# 2. Section: Registry
# ================================================================
class TestDownloadRegistry(RegistryContract):
    registry = Registry

    def make_entry(self) -> type:
        @dataclass
        class Probe(DownloadStrategy):
            def validate(self, spec):
                pass

            def fetch(self, spec):
                pass

        return Probe


# ================================================================
# 3. Section: DownloadStrategy
# ================================================================
@pytest.mark.parametrize("key", STRATEGY_KEYS)
class TestDownloadStrategyContract:
    def test_implements_download_strategy(self, key: str) -> None:
        assert_implements(
            registered_type(Registry, key), DownloadStrategy, ["validate", "fetch"]
        )

    def test_builds_without_arguments(self, key: str) -> None:
        assert isinstance(Registry.get(key), DownloadStrategy)

    def test_spec_source_maps_back_to_key(self, key: str) -> None:
        cls = registered_type(Registry, key)
        spec_type = generic_arg(cls, DownloadStrategy)
        assert isinstance(spec_type, type) and issubclass(
            spec_type, DownloadSpec
        ), f"{cls.__qualname__} must subclass DownloadStrategy[<its DownloadSpec>]"
        assert_is_dataclass(spec_type)
        assert getattr(spec_type, "source", None) == key, (
            f"{spec_type.__qualname__}.source is "
            f"{getattr(spec_type, 'source', None)!r} but its strategy "
            f"{cls.__qualname__} is registered as {key!r}: Downloader routes by "
            f"spec.source, so they must match"
        )


# ================================================================
# 4. Section: Downloader wiring
# ================================================================
def test_run_validates_then_fetches(fake_strategy) -> None:
    spec = FakeSpec(dest="unused")
    Downloader().run(spec)
    assert fake_strategy.calls == [("validate", spec), ("fetch", spec)]


def test_run_fetches_with_the_strategy_of_spec_source(fake_strategy) -> None:
    spec = FakeSpec(dest="unused")
    Downloader().run(spec)
    assert ("fetch", spec) in fake_strategy.calls


def test_run_unknown_source_raises() -> None:
    @dataclass
    class OrphanSpec(DownloadSpec):
        source: ClassVar[str] = "contract-orphan"

    with pytest.raises(ValueError, match="contract-orphan"):
        Downloader().run(OrphanSpec(dest="unused"))
