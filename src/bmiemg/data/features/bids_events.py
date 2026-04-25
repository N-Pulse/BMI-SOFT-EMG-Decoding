# ================================================================
# 0. Section: IMPORTS
# ================================================================
from mne.io.fiff.raw import Raw



# ================================================================
# 1. Section: Functions
# ================================================================
def get_event_labels(emg: Raw, tmin: int, tmax: int) -> list:
    # 0. Handles the extra
    extras = emg.annotations.extras
    if extras is None:
        extras = [None] * len(emg.annotations)

    # 1. Quick start the data
    visible_annotations = []
    params = zip(
        emg.annotations.onset,
        emg.annotations.duration,
        emg.annotations.description,
        extras,
    )

    # 2. Loops over every label
    for onset, duration, desc, extra in params:
        if onset > tmax or (onset + duration) < tmin:
            continue

        # 2.1 Define the labels
        if extra is not None and "lsl_code" in extra:
            lsl_code = str(extra["lsl_code"])
            event_label = lsl_code
        else:
            event_label = desc

        visible_annotations.append((onset, duration, event_label))

    return visible_annotations
