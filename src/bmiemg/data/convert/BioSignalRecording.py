# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class BioSignalRecording:
    eeg_raw: mne.io.RawArray
    emg_raw: mne.io.RawArray
    annotations: mne.Annotations

    def attach_annotations(self) -> dict[str, mne.io.RawArray]:
        eeg_annotated = attach(self.eeg_raw, self.annotations)
        emg_annotated = attach(self.emg_raw, self.annotations)

        return {
            "EEG": eeg_annotated,
            "EMG": emg_annotated
        }


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def attach(raw: mne.io.RawArray, annotations: mne.Annotations) -> mne.io.RawArray:
    raw_annotated = raw.copy()
    raw_annotated.set_annotations(annotations)

    return raw_annotated
