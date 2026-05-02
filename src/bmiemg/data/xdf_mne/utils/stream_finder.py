# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Any



# ================================================================
# 1. Section: Functions
# ================================================================
def find_stream(streams: list[dict], stream_type: str) -> dict:
    matches = [
        stream for stream in streams
        if info(stream, "type") == stream_type
    ]

    if not matches:
        available = [info(stream, "type") for stream in streams]
        raise ValueError(
            f"No stream with type {stream_type!r} found. "
            f"Available stream types: {available}"
        )

    if len(matches) > 1:
        names = [info(stream, "name") for stream in matches]
        print(f"Warning: multiple {stream_type!r} streams found: {names}. Using first.")

    return matches[0]



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def first(value: Any, default: Any = None) -> Any:
    if isinstance(value, list) and len(value) > 0:
        return value[0]
    return default

def info(stream: dict, key: str, default: Any = None) -> Any:
    return first(stream.get("info", {}).get(key, [default]), default)
