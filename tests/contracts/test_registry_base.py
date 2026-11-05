# ================================================================
# 0. Section: IMPORTS
# ================================================================
from bmiemg.utils import Registry

from contracts.registry_contract import RegistryContract
from contracts.discovery import registry_keys, temporary_entry


# ================================================================
# 1. Section: Throwaway families
# ================================================================
class _Thing: ...


class _ThingRegistry(Registry[_Thing]): ...


class _OtherRegistry(Registry[_Thing]): ...


# ================================================================
# 2. Section: Tests
# ================================================================
class TestSharedRegistry(RegistryContract):
    registry = _ThingRegistry

    def make_entry(self) -> type:
        return type("Probe", (_Thing,), {})


def test_each_family_has_its_own_entries() -> None:
    with temporary_entry(_ThingRegistry, "only_here", _Thing):
        assert "only_here" in registry_keys(_ThingRegistry)
        assert "only_here" not in registry_keys(_OtherRegistry)


def test_the_same_key_can_live_in_two_families() -> None:
    with (
        temporary_entry(_ThingRegistry, "shared", _Thing),
        temporary_entry(_OtherRegistry, "shared", _Thing),
    ):
        assert _OtherRegistry.get_type("shared") is _Thing
