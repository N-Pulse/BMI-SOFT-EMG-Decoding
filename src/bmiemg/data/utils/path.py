# ================================================================
# 0. Section: IMPORTS
# ================================================================
from collections.abc import Callable
from datetime import datetime, date
from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
def get_xdf_paths(
    ignore_set: set,
    period: str,
    data_path: Path,
    cutoff_date: date
) -> list:
    period = period.lower()

    comparators: dict[str, Callable[[date, date], bool]] = {
        "before": lambda folder_date, cutoff: folder_date < cutoff,
        "after": lambda folder_date, cutoff: folder_date >= cutoff,
    }

    if period not in comparators:
        raise ValueError(f"{period} is not defined. Use 'before' or 'after'.")

    is_in_period = comparators[period]

    return [
        file
        for file in data_path.rglob("*.xdf")
        if file.is_file()
        and file.name not in ignore_set
        and (folder_date := get_date_from_parents(file)) is not None
        and is_in_period(folder_date, cutoff_date)
    ]


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_date_from_parents(path: Path) -> date | None:
    for parent in path.parents:
        try:
            return datetime.strptime(parent.name, "%Y-%m-%d").date()
        except ValueError:
            continue

    return None
