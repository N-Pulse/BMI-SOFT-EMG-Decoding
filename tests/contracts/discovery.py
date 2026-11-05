# ================================================================
# 0. Section: IMPORTS
# ================================================================
import inspect

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, get_args, get_origin

from bmiemg.utils import Registry


# ================================================================
# 1. Section: Registry access
# ================================================================
def registry_entries(registry: type[Registry[Any]]) -> dict[str, type]:
    """Return the live key -> class mapping of a registry family."""
    return registry._entries


def registry_keys(registry: type[Registry[Any]]) -> list[str]:
    return sorted(registry.known_keys())


def registered_type(registry: type[Registry[Any]], key: str) -> type:
    return registry.get_type(key)


@contextmanager
def temporary_entry(
    registry: type[Registry[Any]], key: str, target: type
) -> Iterator[type]:
    """Register `target` under `key` through the public API, then remove it."""
    registry.register(key)(target)
    try:
        yield target
    finally:
        registry_entries(registry).pop(key, None)


# ================================================================
# 2. Section: Class discovery
# ================================================================
def all_subclasses(base: type) -> list[type]:
    found: list[type] = []
    stack = list(base.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in found:
            found.append(cls)
            stack.extend(cls.__subclasses__())
    return sorted(found, key=lambda cls: cls.__qualname__)


def leaf_subclasses(base: type) -> list[type]:
    """Concrete implementations: subclasses nobody else extends (intermediate bases are skipped)."""
    return [cls for cls in all_subclasses(base) if not cls.__subclasses__()]


def generic_arg(cls: type, base: type, index: int = 0) -> Any:
    """Return the `index`-th type argument `cls` passed to the generic `base`, or None."""
    for klass in inspect.getmro(cls):
        for orig in getattr(klass, "__orig_bases__", ()):
            if get_origin(orig) is base:
                args = get_args(orig)
                return args[index] if len(args) > index else None
    return None
