# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

from dataclasses import dataclass

from .TriggerMap import TriggerMap



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SignalPartitioner:
    trigger_map: TriggerMap

    def partition(self, raw: mne.io.RawArray, movement_duration: float = 5.0) -> mne.Epoch:
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
