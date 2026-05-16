# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np

from dataclasses import dataclass

from .TriggerMap import TriggerMap
from .duration import average_movement_duration



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SignalPartitioner:
    trigger_map: TriggerMap

    def partition(
        self,
        raw: mne.io.RawArray,
        movement_duration: float = 5.0,
    ) -> mne.Epochs:
        # 1. Get all the movement labels present on the signal
        movement_labels = get_labels_at_position(
            raw,
            self.trigger_map.movement_id[0],
            self.trigger_map.movement_id[1]
        )

        # 2. Get only that signal
        event_id_map = build_event_dict(movement_labels)
        events, event_id = mne.events_from_annotations(
            raw=raw,
            event_id=event_id_map # type: ignore[arg-type]
        )

        # 2.1 Inform on trial duration
        average_movement_duration(raw, movement_labels)

        # 3. Build the epochs
        epochs = mne.Epochs(
                raw=raw,
                events=events,
                event_id=event_id,
                tmin=-0.5,
                tmax=movement_duration,
                baseline=(-0.5, 0),
                preload=True,
            )

        return epochs

    def group(
        self,
        epochs: mne.EpochsArray,
        rest_label: str = "rest",
    ) -> mne.EpochsArray:
        grouped_event_id = {
            group_name: idx + 1
            for idx, group_name in enumerate(self.trigger_map.target_code.keys())
        }

        rest_code = len(grouped_event_id) + 1
        grouped_event_id[rest_label] = rest_code

        code_to_label = {
            code: label
            for label, code in epochs.event_id.items()
        }

        X = epochs.get_data(copy=True)
        old_events = epochs.events.copy()

        keep_indices: list[int] = []
        new_event_codes: list[int] = []

        group_counts = {
            group_name: 0
            for group_name in grouped_event_id.keys()
        }

        for i, event in enumerate(old_events):
            old_code = int(event[-1])

            if old_code not in code_to_label:
                continue

            old_label = code_to_label[old_code]

            try:
                mov_code = movement_suffix(old_label)
            except ValueError:
                continue

            assigned_group = rest_label

            for group_name, mov_codes in self.trigger_map.target_code.items():
                if mov_code in mov_codes:
                    assigned_group = group_name
                    break

            keep_indices.append(i)
            new_event_codes.append(grouped_event_id[assigned_group])
            group_counts[assigned_group] += 1

        if len(keep_indices) == 0:
            raise ValueError(
                "No epochs could be grouped. Check marker format and epochs.event_id."
            )

        keep_indices_array = np.asarray(keep_indices, dtype=int)
        new_event_codes_array = np.asarray(new_event_codes, dtype=int)

        X_grouped = X[keep_indices_array]

        new_events = old_events[keep_indices_array].copy()
        new_events[:, -1] = new_event_codes_array

        present_codes = set(new_events[:, -1])

        grouped_event_id = {
            group_name: code
            for group_name, code in grouped_event_id.items()
            if code in present_codes
        }

        grouped_epochs = mne.EpochsArray(
            data=X_grouped,
            info=epochs.info.copy(),
            events=new_events,
            event_id=grouped_event_id,
            tmin=epochs.tmin,
            baseline=epochs.baseline,
            metadata=None,
        )

        return grouped_epochs


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_labels_at_position(signal: mne.io.RawArray, pos: int, key: int) -> list:
    key_str = str(key)

    return sorted({
        desc for desc in signal.annotations.description
        if str(desc)[pos:pos + len(key_str)] == key_str
    })

def build_event_dict(labels: list) -> dict[str, int]:
    return {label: int(label) for label in labels}

def get_movement_code(marker: str) -> int:
    marker = str(marker).strip()

    # Last two digits are the movement code
    return int(marker[-2:])

def clean_marker(label) -> str:
    return (
        str(label)
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
        .strip()
    )

def movement_suffix(label: str) -> int:
    """
    Example:
        32101 -> 1
        32111 -> 11
        32122 -> 22
    """
    label = clean_marker(label)

    if len(label) < 2:
        raise ValueError(f"Marker label too short: {label}")

    suffix = label[-2:]

    if not suffix.isdigit():
        raise ValueError(f"Marker suffix is not numeric: {label}")

    return int(suffix)
