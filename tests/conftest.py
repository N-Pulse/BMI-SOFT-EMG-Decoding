# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pytest

from contextlib import ExitStack
from collections.abc import Callable, Iterator

from contracts.discovery import temporary_entry


# ================================================================
# 1. Section: Fixtures
# ================================================================
@pytest.fixture
def register_temporarily() -> Iterator[Callable[[type, str, type], type]]:
    """Register fakes for one test; every entry is removed afterwards."""
    with ExitStack() as stack:

        def register(registry: type, key: str, target: type) -> type:
            return stack.enter_context(temporary_entry(registry, key, target))

        yield register
