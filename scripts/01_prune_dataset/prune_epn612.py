# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from bmiemg.prune import DatasetPruner
from bmiemg.prune.domain.actors import Dataset, DatasetPartition, TrialData

# ================================================================
# 1. Section: INPUTS
# ================================================================
# Where scripts/00_download/download_epn612.py extracts the dataset.
SOURCE_NAME: list[str] = [
    "raw/emg_epn_612",
]

PROTOCOL_NAME = "epn612_adapted_v1"
PLOT_BALANCE = True
_PARTITIONS = ["train", "val", "test"]


# ================================================================
# 2. Section: PLOTS
# ================================================================
def plot_dataset_balance(dataset: Dataset) -> None:
    client = {p.partition_name: p for p in dataset.client}
    server = {p.partition_name: p for p in dataset.server}
    label_source = {**client, "test": server["test"]}

    _, axes = plt.subplots(1, 3, figsize=(16, 4))
    _plot_sizes(axes[0], client)
    _plot_bars(axes[1], _subject_totals(client), "subjects per partition")
    _plot_balance(axes[2], _movement_counts(label_source), "movement balance (code)")

    plt.suptitle("EPN-612 TTV dataset balance")
    plt.tight_layout()
    plt.show()


def _plot_sizes(ax, partitions: dict[str, DatasetPartition]) -> None:
    sizes = [len(partitions[p].trials) for p in _PARTITIONS]
    _plot_bars(ax, sizes, "partition sizes (trials)")


def _plot_bars(ax, sizes: list[int], title: str) -> None:
    ax.bar(_PARTITIONS, sizes, color=["C0", "C1", "C2"])
    for x, n in enumerate(sizes):
        ax.text(x, n, str(n), ha="center", va="bottom")
    ax.set_title(title)


def _plot_balance(ax, counts: dict, title: str) -> None:
    categories = sorted({c for part in counts.values() for c in part})
    bottom = np.zeros(len(_PARTITIONS))
    for category in categories:
        heights = np.array(
            [counts.get(p, Counter()).get(category, 0) for p in _PARTITIONS]
        )
        ax.bar(_PARTITIONS, heights, bottom=bottom, label=str(category))
        bottom += heights
    ax.set_title(title)
    ax.set_ylabel("trials")
    ax.legend(fontsize="x-small", ncol=2)


# A 612-subject stack is unreadable, so only count distinct subjects.
def _subject_totals(partitions: dict[str, DatasetPartition]) -> list[int]:
    return [len({t.subject_number for t in partitions[p].trials}) for p in _PARTITIONS]


def _movement_counts(partitions: dict[str, DatasetPartition]) -> dict:
    return {
        name: Counter(_dominant_movement(t) for t in part.trials)
        for name, part in partitions.items()
    }


def _dominant_movement(trial: TrialData) -> int:
    movements = trial.markers % 100
    labelled = Counter(movements[movements != 0].tolist())
    return int(labelled.most_common(1)[0][0]) if labelled else 0


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    pruner = DatasetPruner(
        source_name=SOURCE_NAME,
    )

    actors = pruner.run(PROTOCOL_NAME)
    pruner.export(actors, PROTOCOL_NAME)

    if PLOT_BALANCE and actors and isinstance(actors[0], Dataset):
        plot_dataset_balance(actors[0])
