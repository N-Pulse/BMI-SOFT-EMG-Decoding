"""Subject-disjoint, movement-balanced splits for EMG-EPN-612."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SOURCE_GROUPS = ("trainingJSON", "testingJSON")
SPLIT_NAMES = ("train", "test", "val")
MOVEMENTS = ("noGesture", "fist", "waveIn", "waveOut", "open", "pinch")

_TRIAL_PATTERN = re.compile(
    rb'"(?P<trial_id>idx_\d+)"\s*:\s*\{.{0,50000}?'
    rb'"gestureName"\s*:\s*"(?P<movement>[^"]+)"',
    flags=re.DOTALL,
)
_SAMPLING_RATE_PATTERN = re.compile(
    rb'"samplingFrequencyInHertz"\s*:\s*(?P<rate>[0-9]+(?:\.[0-9]+)?)'
)


@dataclass(frozen=True, slots=True)
class ParticipantSource:
    participant_id: str
    source_group: str
    source_user_id: str
    source_path: Path


@dataclass(frozen=True, slots=True)
class ParticipantSplits:
    """Selected participants and the data split assigned to each one."""

    participants: tuple[ParticipantSource, ...]
    assignments: dict[str, str]
    split_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SplitRecord:
    participant_id: str
    source_group: str
    source_user_id: str
    source_section: str
    trial_id: str
    movement: str
    split: str
    source_path: str
    source_sampling_rate_hz: float
    target_sampling_rate_hz: float
    source_adc_bits: int
    target_adc_bits: int
    analog_filter: str


def discover_participants(dataset_root: str | Path) -> list[ParticipantSource]:
    """Discover all 612 users while keeping the two source groups distinct."""

    root = Path(dataset_root)
    participants: list[ParticipantSource] = []
    for group in SOURCE_GROUPS:
        paths = sorted((root / group).glob("user*/*.json"), key=_user_path_key)
        if not paths:
            raise FileNotFoundError(f"No user JSON files under {root / group}")
        for path in paths:
            source_user = path.parent.name
            participants.append(
                ParticipantSource(
                    participant_id=f"{group}:{source_user}",
                    source_group=group,
                    source_user_id=source_user,
                    source_path=path,
                )
            )
    duplicate_ids = len(participants) - len({p.participant_id for p in participants})
    if duplicate_ids:
        raise ValueError(f"Found {duplicate_ids} duplicate participant identifiers")
    return participants


def select_participant_splits(
    dataset_root: str | Path,
    *,
    participant_id: str | None = None,
    max_participants: int | None = None,
    ratios: Mapping[str, float] | None = None,
    random_seed: int = 42,
) -> ParticipantSplits:
    """Select participants and create subject-disjoint split assignments.

    A single selected participant is returned as one ``all`` split. Its trials
    can then be divided reproducibly before their windows enter the datasets.
    Cohort runs assign whole participants to train, validation, and test splits.
    """

    if participant_id is not None and max_participants is not None:
        raise ValueError("participant_id and max_participants are mutually exclusive")
    if max_participants is not None and max_participants <= 0:
        raise ValueError("max_participants must be a positive integer")

    participants = discover_participants(dataset_root)
    if participant_id is not None:
        selected = tuple(
            participant
            for participant in participants
            if participant.participant_id == participant_id
        )
        if not selected:
            raise ValueError(f"EPN-612 participant {participant_id!r} was not found")
        return ParticipantSplits(selected, {participant_id: "all"}, ("all",))

    selected = tuple(participants[:max_participants])
    assignments = assign_participant_splits(
        selected,
        ratios=ratios,
        random_seed=random_seed,
    )
    return ParticipantSplits(selected, assignments, SPLIT_NAMES)


def assign_participant_splits(
    participants: Iterable[ParticipantSource],
    ratios: Mapping[str, float] | None = None,
    random_seed: int = 42,
) -> dict[str, str]:
    """Assign whole participants to splits, stratified by source group."""

    split_ratios = dict(ratios or {"train": 0.70, "test": 0.15, "val": 0.15})
    if set(split_ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Ratios must define exactly {SPLIT_NAMES}")
    if any(value < 0 for value in split_ratios.values()) or not np.isclose(
        sum(split_ratios.values()), 1.0
    ):
        raise ValueError("Split ratios must be non-negative and sum to one")

    grouped: dict[str, list[ParticipantSource]] = defaultdict(list)
    for participant in participants:
        grouped[participant.source_group].append(participant)
    rng = np.random.default_rng(random_seed)
    assignments: dict[str, str] = {}
    for group in sorted(grouped):
        group_participants = sorted(
            grouped[group], key=lambda item: _user_text_key(item.source_user_id)
        )
        permutation = rng.permutation(len(group_participants))
        shuffled = [group_participants[index] for index in permutation]
        counts = _largest_remainder_counts(len(shuffled), split_ratios)
        offset = 0
        for split in SPLIT_NAMES:
            for participant in shuffled[offset : offset + counts[split]]:
                assignments[participant.participant_id] = split
            offset += counts[split]
    return assignments


def read_training_trial_metadata(
    path: str | Path,
) -> tuple[float, list[tuple[str, str]]]:
    """Read only trial IDs and labels without decoding multi-megabyte signal arrays."""

    source_path = Path(path)
    document = source_path.read_bytes()
    training_start = document.find(b'"trainingSamples"')
    testing_start = document.find(b'"testingSamples"', training_start + 1)
    if training_start < 0 or testing_start < 0:
        raise ValueError(f"Missing training/testing sections in {source_path}")
    rate_match = _SAMPLING_RATE_PATTERN.search(document[:training_start])
    if rate_match is None:
        raise ValueError(f"Missing sampling rate in {source_path}")
    sampling_rate = float(rate_match.group("rate"))

    section = document[training_start:testing_start]
    trials = [
        (
            match.group("trial_id").decode("utf-8"),
            match.group("movement").decode("utf-8"),
        )
        for match in _TRIAL_PATTERN.finditer(section)
    ]
    if not trials:
        raise ValueError(f"No labelled training trials in {source_path}")
    unknown = sorted({movement for _, movement in trials} - set(MOVEMENTS))
    if unknown:
        raise ValueError(f"Unknown movements in {source_path}: {unknown}")
    if len({trial_id for trial_id, _ in trials}) != len(trials):
        raise ValueError(f"Duplicate training trial IDs in {source_path}")
    return sampling_rate, sorted(trials, key=lambda item: _trial_text_key(item[0]))


def build_split_records(
    participants: Iterable[ParticipantSource],
    assignments: Mapping[str, str],
    target_sampling_rate_hz: float = 2000.0,
    source_adc_bits: int = 8,
    target_adc_bits: int = 12,
) -> list[SplitRecord]:
    """Build one manifest row per labelled source trial."""

    records: list[SplitRecord] = []
    for participant in sorted(
        participants,
        key=lambda item: (item.source_group, _user_text_key(item.source_user_id)),
    ):
        try:
            split = assignments[participant.participant_id]
        except KeyError as error:
            raise ValueError(
                f"Missing split assignment for {participant.participant_id}"
            ) from error
        sampling_rate, trials = read_training_trial_metadata(participant.source_path)
        for trial_id, movement in trials:
            records.append(
                SplitRecord(
                    participant_id=participant.participant_id,
                    source_group=participant.source_group,
                    source_user_id=participant.source_user_id,
                    source_section="trainingSamples",
                    trial_id=trial_id,
                    movement=movement,
                    split=split,
                    source_path=str(participant.source_path),
                    source_sampling_rate_hz=sampling_rate,
                    target_sampling_rate_hz=target_sampling_rate_hz,
                    source_adc_bits=source_adc_bits,
                    target_adc_bits=target_adc_bits,
                    analog_filter="causal first-order 20-500 Hz band-pass",
                )
            )
    _validate_balance(records)
    return records


def write_split_manifest(records: Iterable[SplitRecord], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    if not rows:
        raise ValueError("Cannot write an empty split manifest")
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def summarize_split(records: Iterable[SplitRecord]) -> dict[str, object]:
    rows = list(records)
    user_sets: dict[str, set[str]] = defaultdict(set)
    movement_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        user_sets[row.split].add(row.participant_id)
        movement_counts[row.split][row.movement] += 1
        group_counts[row.split][row.source_group] += 1
    total_users = len({row.participant_id for row in rows})
    total_trials = len(rows)
    return {
        "total_users": total_users,
        "total_trials": total_trials,
        "splits": {
            split: {
                "users": len(user_sets[split]),
                "user_fraction": len(user_sets[split]) / total_users,
                "trials": sum(movement_counts[split].values()),
                "trial_fraction": sum(movement_counts[split].values()) / total_trials,
                "movements": dict(sorted(movement_counts[split].items())),
                "source_groups": dict(sorted(group_counts[split].items())),
            }
            for split in SPLIT_NAMES
        },
    }


def _validate_balance(records: list[SplitRecord]) -> None:
    if not records:
        raise ValueError("No split records were produced")
    participant_splits: dict[str, set[str]] = defaultdict(set)
    per_user: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        participant_splits[record.participant_id].add(record.split)
        per_user[record.participant_id][record.movement] += 1
    leaked = [user for user, splits in participant_splits.items() if len(splits) != 1]
    if leaked:
        raise ValueError(f"Participants leak across splits: {leaked[:5]}")
    expected = Counter(next(iter(per_user.values())))
    unbalanced = [user for user, counts in per_user.items() if counts != expected]
    if unbalanced:
        raise ValueError(f"Movement counts differ for participants: {unbalanced[:5]}")


def _largest_remainder_counts(
    total: int, ratios: Mapping[str, float]
) -> dict[str, int]:
    exact = {name: total * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(np.floor(exact[name])) for name in SPLIT_NAMES}
    remainder = total - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(exact[name] - counts[name]), SPLIT_NAMES.index(name)),
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _user_path_key(path: Path) -> tuple[int, str]:
    return _user_text_key(path.parent.name)


def _user_text_key(value: str) -> tuple[int, str]:
    suffix = value.removeprefix("user")
    return (int(suffix), value) if suffix.isdigit() else (2**63 - 1, value)


def _trial_text_key(value: str) -> tuple[int, str]:
    suffix = value.rpartition("_")[2]
    return (int(suffix), value) if suffix.isdigit() else (2**63 - 1, value)
