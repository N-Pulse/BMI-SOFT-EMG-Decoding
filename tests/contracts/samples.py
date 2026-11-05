# ================================================================
# 0. Section: IMPORTS
# ================================================================
import dataclasses

from typing import Any
from collections.abc import Mapping

import pytest


# ================================================================
# 1. Section: Sample lookups
# ================================================================
def sample_for(samples: Mapping[str, Any], key: str, table: str) -> Any:
    """Return the sample for `key`, failing with a pointer to the table to edit."""
    if key not in samples:
        pytest.fail(
            f"No sample config for registered key {key!r}: add a sample config "
            f"for {key!r} to {table}",
            pytrace=False,
        )
    return samples[key]


def sentinel_instance(cls: type) -> Any:
    """Build a dataclass instance whose fields hold unique sentinels,
    skipping __init__.

    Contracts about structure (copy_with, equality of untouched fields) need
    an instance but not meaningful data.
    """
    instance = object.__new__(cls)
    for field in dataclasses.fields(cls):
        object.__setattr__(instance, field.name, object())
    return instance


def init_field_names(cls: type) -> list[str]:
    return [f.name for f in dataclasses.fields(cls) if f.init]
