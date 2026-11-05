# ================================================================
# 0. Section: IMPORTS
# ================================================================
import inspect
import dataclasses

from typing import Any


# ================================================================
# 1. Section: Class-shape assertions
# ================================================================
def assert_is_dataclass(cls: type) -> None:
    assert dataclasses.is_dataclass(cls), f"{cls.__qualname__} must be a @dataclass"


def assert_name_classvar(cls: type, expected: str | None = None) -> None:
    """`name` must be a class-level string, not a dataclass field."""
    name = getattr(cls, "name", None)
    assert (
        isinstance(name, str) and name
    ), f'{cls.__qualname__} must define `name: ClassVar[str] = "..."`'
    field_names = {f.name for f in dataclasses.fields(cls)}
    assert (
        "name" not in field_names
    ), f"{cls.__qualname__}.name must be a ClassVar, not a dataclass field"
    if expected is not None:
        assert name == expected, (
            f"{cls.__qualname__}.name is {name!r} but it is registered as "
            f"{expected!r}: they must match"
        )


def assert_implements(cls: type, base: type, methods: list[str]) -> None:
    """Every abstract method is overridden and keeps the base's parameter names and kinds."""
    assert issubclass(cls, base), f"{cls.__qualname__} must subclass {base.__name__}"
    assert not inspect.isabstract(cls), (
        f"{cls.__qualname__} leaves abstract methods unimplemented: "
        f"{sorted(getattr(cls, '__abstractmethods__', ()))}"
    )
    for method in methods:
        assert getattr(cls, method) is not getattr(
            base, method
        ), f"{cls.__qualname__} must override {base.__name__}.{method}"
        expected = _param_shape(getattr(base, method))
        actual = _param_shape(getattr(cls, method))
        assert actual == expected, (
            f"{cls.__qualname__}.{method} signature {actual} does not match "
            f"{base.__name__}.{method} {expected}"
        )


def _param_shape(func: Any) -> list[tuple[str, Any]]:
    # Annotations are left out: implementations narrow generic types on purpose.
    return [(p.name, p.kind) for p in inspect.signature(func).parameters.values()]
