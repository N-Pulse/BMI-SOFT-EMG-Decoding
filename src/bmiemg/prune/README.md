# Prune Dataset
A pipeline that turns raw recordings into final files ready for training. Several protocols can live side by side, each producing a different dataset.

The class diagram is in [`docs/diagrams/class.puml`](../../../docs/diagrams/class.puml) (`make` renders it).

---

## 1. Architecture
The package follows a three-layer split:

| Layer | Folder | Contents |
|---|---|---|
| Service | `service/` | `DatasetPruner`: wires everything together and runs a protocol. No logic of its own. |
| Domain | `domain/` | `DataActor`s (the data), `PruneStep`s (the transformations), `Protocol`. No I/O. |
| Adapter | `adapters/` | `Source` (paths), `Loader` + `FileLoader`s (read files), `ProtocolLoader` (read YAML), `Exporter` + `Writer`s (write files). |

The pipeline is: **load files → actors → steps (in order) → actors → writers → files**.

Every extensible family (`PruneStep`, `FileLoader`, `Writer`) is plugged in through a registry built on `bmiemg.utils.Registry`. A class registers itself with a decorator under a string key, and the YAML or file suffix refers to that key. Adding a new implementation never requires touching the service or the loaders.

### Data actors
Actors are dataclasses carried between steps. Each step declares which actor types it takes and returns (`PruneStep[InT, OutT]`), so step order matters.

| Actor | Produced by | Contents |
|---|---|---|
| `XdfData` | `XdfLoader` | Raw pyxdf `streams` plus file metadata. |
| `TrialData` | `SignalUnwarpStep`, `Epn612JsonLoader` | `signal_ch` `(channels, time)`, per-sample `markers` `(1, time)`, `channel_names`, `sfreq`. |
| `DatasetPartition` | `TTVSplitStep` | One of `train`/`val`/`test` and its trials; `mne_dataset` builds an `mne` Raw. |
| `Dataset` | `TTVSplitStep` | `client` and `server` partition lists. |

`XdfData` and `TrialData` inherit from `MetadataActor`, which derives `session_number`, `subject_number` and `date` from the `bids_local` folder layout (`<date>/<subject>/<session>/<modality>/<file>`). Loaders for sources without that layout set `subject_id` instead, which `subject_number` returns first.

---

## 2. Protocols
Each protocol is a YAML file in `configs/prune/<protocol_name>.yaml`:

- `steps:` lists the steps in run order, keyed by each step's registry key with its constructor parameters (`{}` for none).
- `export:` lists the writers to run, keyed by the writer's registry key, with their parameters.

Steps whose parameters aren't plain YAML types (tuples, nested dataclasses) override `from_config` to convert them.

### raw_archive_v1
Turns the raw data from the acquisition team into a usable train/val/test dataset:

`remove_session` → `signal_split` → `protocol_definer` → `signal_unwarp` → `analogue_filter` → `marker_unification` → `build_epochs` → `ttv_split`, exported with `dataset`.

### epn612_adapted_v1
Adapts EMG-EPN-612 (Myo armband, signed 8-bit, 200 Hz, read by `Epn612JsonLoader` from `data/raw/emg_epn_612/`) to the board domain (signed 12-bit ADC, 2000 Hz, first-order 20-500 Hz front end) and splits it subject-disjoint 70/15/15:

`adc_normalize` (8 bits) → `resample` (2000 Hz, `padtype: line`) → `analogue_filter` (causal 20 Hz high-pass) → `analogue_filter` (causal 500 Hz low-pass) → `adc_quantize` (12 bits) → `marker_unification` → `ttv_split` (`group_by: subject`), exported with `dataset`. Trials stay whole: windowing is left to the model.

---

## 3. Steps
Every step is a `PruneStep`: it receives a list of `DataActor`s and returns the transformed list. Parameters are the step's own dataclass fields, so the attributes listed below are exactly the YAML keys.

### RemoveSessionStep - `remove_session` (`XdfData → XdfData`)
Drops whole recordings, keeping only actors whose `file_name` is not in the removal list. It strips corrupted, incomplete and kraken sessions before anything else runs.

- `files_to_remove: list[str]`: file names, without suffix, to drop.

### SignalSplitStep - `signal_split` (`XdfData → XdfData`)
Reduces each recording to one signal modality plus its markers. Marker streams are always kept. For every other stream it keeps only the channels whose type matches (EMG accepts `emg` and `aux`, EEG accepts `eeg`) and rewrites the stream's `time_series`, type and channel metadata. A stream left with no matching channel is dropped. It also sets `signal_type` on the actor.

- `signal: "emg" | "eeg"`: target modality. Any other value raises `ValueError`.

### ProtocolDefinerStep - `protocol_definer` (`XdfData → XdfData`)
Tags each actor with the acquisition protocol version (`protocol_version`, default `0` = undefined), derived from the recording date: `1 + number of cutoffs <= actor.date`. Actors without a date keep the default.

- `cuttoff_dates: list[date]`: boundary dates between protocol versions.

### SignalUnwarpStep - `signal_unwarp` (`XdfData → TrialData`)
Converts the XDF streams (one signal + one marker stream) into a `TrialData`. The signal is transposed to `(channels, time)`. Each marker is snapped to its nearest signal sample and held until the next marker, giving a per-sample marker array aligned with the signal.

- `background_marker: int = 0`: value used before the first marker.

### AnalogueFilterStep - `analogue_filter` (`TrialData → TrialData`)
Applies a Butterworth filter emulating the device's analogue front end. It uses zero-phase `sosfiltfilt` so the signal stays aligned with the markers. Cutoffs are clamped below Nyquist.

- `btype: "bandpass" | "bandstop" | "lowpass" | "highpass"`
- `cutoff: float | [low, high]`: a pair is required for band filters, a single value otherwise.
- `order: int = 1`
- `causal: bool = False`: filter forward only with `sosfilt`, like the real front end, instead of zero-phase `sosfiltfilt`. It adds the front end's phase lag.

### ResampleStep - `resample` (`TrialData → TrialData`)
Polyphase-resamples the signal (`scipy.signal.resample_poly`) to a target rate, with the up/down factors taken from the rate ratio. Markers are held from their source sample, so an onset at source index `i` lands on `i * up / down`. Actors already at the target rate pass through.

- `sfreq: float`: target sampling rate in Hz.
- `padtype: "constant" | "line" | "mean" | "median" | "maximum" | "minimum" = "constant"`: edge padding passed to `resample_poly`.

### AdcNormalizeStep - `adc_normalize` (`TrialData → TrialData`)
Maps signed ADC counts to fractional full scale (`counts / 2^(bits-1)`, in `[-1, 1)`), so signals from ADCs of different resolutions share one scale. Counts outside the signed range raise `ValueError`.

- `bits: int`: source ADC resolution, 2 to 32.

### AdcQuantizeStep - `adc_quantize` (`TrialData → TrialData`)
Quantizes fractional full scale to the signed counts of a target ADC: `rint(x * 2^(bits-1))`, saturated to `[-2^(bits-1), 2^(bits-1) - 1]`, stored as `int16` (`int32` above 16 bits).

- `bits: int`: target ADC resolution, 2 to 32.

### MarkerUnificationStep - `marker_unification` (`TrialData → TrialData`)
Rewrites every raw marker code into the canonical `XXDD` code (`XX` = phase, `DD` = movement), so recordings from different protocol versions share one marker vocabulary. The actor's `protocol_version` picks the source schema, and a missing schema raises `ValueError`. `0` stays background.

- `source_schemas: dict[int, SourceMarkerSchema]`: per protocol version, how to decode raw codes (`phase_pos`, `movement_slice`, `phase_map`, `movement_groups`, `special_triggers`).
- `target: CanonicalMarkerMap`: canonical `phases` and `movements` name → number.

### BuildEpochs - `build_epochs` (`TrialData | Dataset → same`)
Cuts each trial into one window per movement onset (a run of non-zero `DD`), `[onset + tmin, onset + tmax]` seconds. Windows running past either edge are dropped, like `mne.Epochs`. It works on a `Dataset` too, segmenting every partition.

- `tmin: float = -0.5`
- `tmax: float = 5.0`: must be greater than `tmin`.

### TTVSplitStep - `ttv_split` (`TrialData → Dataset`)
Shuffles trials (seeded) into train/val/test, optionally stratified so each subject and/or dominant movement is split by the same ratios. It produces one `Dataset`:
- `client`: `train`, `val` and `test`. The client test markers are zeroed out.
- `server`: `test` with its markers, for scoring.

- `train_ratio`, `val_ratio`, `test_ratio: float`: must sum to 1.
- `user_balance: bool = False`: stratify by subject.
- `movement_balance: bool = False`: stratify by dominant movement.
- `seed: int = 0`
- `group_by: "subject" | None = None`: assign whole subjects to one partition (subject-disjoint split). Subject counts per partition use largest-remainder rounding. It cannot be combined with `user_balance` or `movement_balance`.

---

## 4. File loaders
`Loader` walks `data/<source_name>/` recursively and dispatches each file to the `FileLoader` registered for its suffix. Files with unknown suffixes are skipped and reported. Each `FileLoader` implements `load_actors(path)`, returning every actor the file holds: one for a continuous recording, one per trial when a file stores several.

| Class | Suffix | Produces |
|---|---|---|
| `XdfLoader` | `.xdf` | `XdfData` |
| `Epn612JsonLoader` | `.json` | One `TrialData` per EMG-EPN-612 `trainingSamples` trial (8 channels, raw counts). `subject_id` is the `user<N>` number, plus 1000 under `testingJSON`. Markers are raw codes `3000M` (phase 3, `M` = gesture: 1 noGesture, 2 fist, 3 waveIn, 4 waveOut, 5 open, 6 pinch) over the `groundTruth` samples, or the whole trial when there is none (noGesture); `marker_unification` maps them to `XXDD` (in `epn612_adapted_v1`: waveIn → rotate wrist right, waveOut → rotate wrist left, noGesture → background `0`). |

## 5. Writers
`Exporter` runs each writer from the protocol's `export:` section on every actor whose `name` matches the writer's `actor`. A writer that finds no matching actor raises `ValueError`. Output goes to `data/dataset/<protocol_name>/`, and every file of one run shares a UTC timestamp suffix.

| Class | Key | Writes | Parameters |
|---|---|---|---|
| `DatasetWriter` | `dataset` | `Dataset` → `client/` and `server/` folders, one `<partition>_<timestamp>_raw.fif` per partition | `fmt: ".fif"`, `zip_groups: bool = True` (also zips each group folder) |

---

## 6. Usage
Run from the repo root through Poetry. Paths default to `data/` and `configs/prune/`.

```python
from bmiemg.prune import DatasetPruner

pruner = DatasetPruner(source_name=["bids"])   # folders under data/
actors = pruner.run("raw_archive_v1")           # load + apply steps
pruner.export(actors, "raw_archive_v1")         # write files (explicit)
```

`run` doesn't write anything, so call `export` when you want files. [`scripts/01_prune_dataset/prune_archive_raw.py`](../../../scripts/01_prune_dataset/prune_archive_raw.py) is a full example that also plots the split balance:

```bash
poetry run python scripts/01_prune_dataset/prune_archive_raw.py
```

To try steps without writing a YAML, pass them directly. `custom_steps` takes precedence over the protocol:

```python
from bmiemg.prune.domain.steps import RemoveSessionStep, SignalSplitStep, SignalUnwarpStep

pruner = DatasetPruner(
    source_name=["bids"],
    custom_steps=[RemoveSessionStep(files_to_remove=[]), SignalSplitStep(signal="emg"), SignalUnwarpStep()],
)
trials = pruner.run(None)
```

You can also run steps on actors you already have with `pruner.apply(actors, steps)`.

To make a new dataset, copy `configs/prune/raw_archive_v1.yaml` to `configs/prune/<new_name>.yaml`, set `protocol_name`, and edit the steps and writers.

---

## 7. Contributing new implementations
All three extension points work the same way: subclass the interface, register it with a decorator, and import the module in its package `__init__.py`. **The import is required.** Registration happens at import time, so a class that is never imported is unknown to the registry (`ValueError: Unknown key ...`).

Registry keys must be unique per registry, and a duplicate raises at import. Keep the `name` ClassVar equal to the registry key.

### New step (`domain/steps/`)
Steps belong to the domain layer, so no file, network or plotting I/O.

```python
# domain/steps/my_step.py
from dataclasses import dataclass
from typing import ClassVar

from ..actors import TrialData
from .prune_step import PruneStep
from .step_registry import StepRegistry


@StepRegistry.register("my_step")
@dataclass
class MyStep(PruneStep[TrialData, TrialData]):
    name: ClassVar[str] = "my_step"
    gain: float = 1.0

    def apply(self, actors: list[TrialData]) -> list[TrialData]:
        return [a.copy_with(signal_ch=a.signal_ch * self.gain) for a in actors]
```

1. Declare the input/output actor types in `PruneStep[InT, OutT]`. That fixes where the step can sit in a protocol.
2. Return new actors with `copy_with(...)` rather than mutating inputs.
3. Validate parameters in `__post_init__` so bad configs fail when the protocol loads.
4. Override `from_config` only if a parameter can't be built from plain YAML (tuples, nested dataclasses). See `AnalogueFilterStep` and `MarkerUnificationStep`.
5. Export it from `domain/steps/__init__.py`, then use `- my_step: {gain: 2.0}` in a protocol.
6. Document it in section 3 above.

### New actor (`domain/actors/`)
Subclass `DataActor` (or `MetadataActor` if it comes from a recording file) as a dataclass. Give it a `name: ClassVar[str]` if a writer needs to target it. Export it from `domain/actors/__init__.py`.

### New file loader (`adapters/file_loaders/`)

```python
@FileLoaderRegistry.register(".csv")
class CsvLoader(FileLoader):
    def load_actors(self, path: Path) -> list[DataActor]:
        ...
```

The key is the lowercase file suffix, dot included. Return every actor the file holds: one for a continuous recording, one per trial when the file stores several. Export it from `adapters/file_loaders/__init__.py`, and add the actor it produces if none fits.

### New writer (`adapters/exporters/`)

```python
@WriterRegistry.register("my_writer")
@dataclass
class MyWriter(Writer[Dataset]):
    name: ClassVar[str] = "my_writer"
    actor: ClassVar[str] = Dataset.name   # which actors it writes
    fmt: str = ".npz"

    def write(self, actor: Dataset, dest_dir: Path) -> None:
        ...  # use self.timestamp in file names
```

The pruner injects `timestamp`, and the remaining fields come from the YAML `export:` entry. Export it from `adapters/exporters/__init__.py`, then add `- my_writer: {fmt: .npz}` under `export:`.

### Checklist
- Update `docs/diagrams/class.puml` notes listing the implemented steps, loaders and writers.
- Update this README.
