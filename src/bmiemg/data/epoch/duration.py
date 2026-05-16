# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np
import pandas as pd


# ================================================================
# 1. Section: Functions
# ================================================================
def average_movement_duration(raw: mne.io.RawArray, movement_labels: list) -> float:
    dur = []
    for lab in movement_labels:
        mov_duration = time_from_marker_to_next(raw, target_marker=lab)
        dur.append(mov_duration["duration_s"].mean())

    print(f"{np.mean(dur)} ± {np.std(dur)}")
    return np.mean(dur, axis=0)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def clean_marker(desc) -> str:
    return (
        str(desc)
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
        .strip()
    )

def time_from_marker_to_next(raw, target_marker: str = "32101") -> pd.DataFrame:
    annotations = raw.annotations

    onsets = np.asarray(annotations.onset, dtype=float)
    descriptions = np.array([clean_marker(d) for d in annotations.description])

    # Ensure annotations are ordered by time
    order = np.argsort(onsets)
    onsets = onsets[order]
    descriptions = descriptions[order]

    rows = []

    for i, desc in enumerate(descriptions):
        if desc != target_marker:
            continue

        # Skip if this is the last marker
        if i + 1 >= len(descriptions):
            continue

        start_time = onsets[i]
        next_time = onsets[i + 1]
        next_marker = descriptions[i + 1]

        rows.append(
            {
                "marker": desc,
                "start_time_s": start_time,
                "next_marker": next_marker,
                "next_time_s": next_time,
                "duration_s": next_time - start_time,
            }
        )

    return pd.DataFrame(rows)
