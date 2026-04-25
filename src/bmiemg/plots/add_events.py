# ================================================================
# 0. Section: IMPORTS
# ================================================================
from matplotlib import pyplot as plt

from matplotlib.axes import Axes



# ================================================================
# 1. Section: Functions
# ================================================================
def add_events(ax: Axes, visible_annotations: list) -> Axes:
    # 1. Define the label colors
    event_colors = get_section_colors(visible_annotations)

    #2. Adds the sections per area
    seen_labels = set()
    for onset, duration, event_label in visible_annotations:
        color = event_colors[event_label]
        label = event_label if event_label not in seen_labels else None
        seen_labels.add(event_label)

        if event_label[0] == '5':
            if duration > 0:
                ax.axvspan(
                    onset,
                    onset + duration,
                    alpha=0.25,
                    color=color,
                    label=label,
                )
            else:
                ax.axvline(
                    onset,
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                    label=label,
                )

    plt.legend()
    return ax


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_section_colors(visible_annotations: list) -> dict:
    unique_labels = list(dict.fromkeys(label for _, _, label in visible_annotations))
    cmap = plt.get_cmap("tab10", len(unique_labels))
    event_colors = {label: cmap(i) for i, label in enumerate(unique_labels)}

    return event_colors
