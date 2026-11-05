# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pytest

from typing import ClassVar

from .discovery import (
    registry_entries,
    registry_keys,
    registered_type,
    temporary_entry,
)


# ================================================================
# 1. Section: Registry contract
# ================================================================
class RegistryContract:
    """Behaviour every registry family must honour, checked through register/get.

    Subclass it as `TestXRegistry`, set `registry`, and implement `make_entry` to return a fresh, no-arg constructible class acceptable to that registry.
    """

    registry: ClassVar[type]
    key: ClassVar[str] = "__contract_probe__"

    def make_entry(self) -> type:
        raise NotImplementedError

    def test_register_then_get_returns_an_instance(self) -> None:
        entry = self.make_entry()
        with temporary_entry(self.registry, self.key, entry):
            assert isinstance(self.registry.get(self.key), entry)
            assert registered_type(self.registry, self.key) is entry
            assert self.key in registry_keys(self.registry)
        assert self.key not in registry_keys(self.registry)

    def test_register_returns_the_class_unchanged(self) -> None:
        entry = self.make_entry()
        try:
            assert self.registry.register(self.key)(entry) is entry
        finally:
            registry_entries(self.registry).pop(self.key, None)

    def test_duplicate_key_raises(self) -> None:
        with temporary_entry(self.registry, self.key, self.make_entry()):
            with pytest.raises(ValueError, match=self.key):
                self.registry.register(self.key)(self.make_entry())

    def test_unknown_key_raises_naming_known_keys(self) -> None:
        with temporary_entry(self.registry, self.key, self.make_entry()):
            with pytest.raises(ValueError) as excinfo:
                self.registry.get("__missing__")
        message = str(excinfo.value)
        assert "__missing__" in message
        assert self.key in message, "the error must list the registered keys"

    def test_registered_keys_resolve(self) -> None:
        for key in registry_keys(self.registry):
            assert isinstance(registered_type(self.registry, key), type)
