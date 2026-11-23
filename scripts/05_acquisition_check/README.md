# Acquisition check

Checks one recording at a time. It confirms the expected EMG and EEG channels are present, that the EEG channels map to 10-20 positions, and that the signals look sane. The summary is printed and also saved as a `.txt` file.

Supported formats: XDF (`.xdf`) and BrainVision (`.vhdr` + `.eeg` + `.vmrk`; pass either the `.eeg` or the `.vhdr`). There is no dependency on `bmiemg`, only `numpy` and `pyxdf`.

## How to run

1. Open `check.py` and edit the constants in the `INPUTS` section (see below).
2. From the repository root:

```bash
poetry run python scripts/05_acquisition_check/check.py
```

3. Read the printed summary, or the saved copy at `OUTPUT_DIR/<file stem>_check.txt`.

To check another file, change `FILE_PATH` and run it again. The script only ever loads one file, which keeps RAM usage in check. An XDF file still has to be loaded whole (`pyxdf` has no lazy reading), so expect a few GB of RAM for long 177-channel recordings. A BrainVision file is memory-mapped and read one channel at a time.

## Inputs

| Constant | Meaning |
|---|---|
| `FILE_PATH` | Recording to check |
| `OUTPUT_DIR` | Folder where the `.txt` summary is saved |
| `N_EMG_CHANNELS` | N: the EMG channel count must match exactly |
| `M_EEG_CHANNELS` | M: the EEG channel count must match exactly |
| `P_EMG_DEAD_ALLOWED` | P: maximum number of dead EMG channels |
| `Q_EEG_DEAD_ALLOWED` | Q: maximum number of dead EEG channels |
| `EXPECTED_FS` | Expected sampling rate in Hz, `None` to skip the check |
| `EXPECTED_MARKER_CODES` | Marker codes that must be present, e.g. `["303", "304"]`; `None` skips the check |
| `PLOT` | Save the marker plot |
| `PLOT_WINDOW_S` | Window around each code's first marker, in seconds |
| `PLOT_EEG_CHANNELS` | EEG channels plotted below the EMG channels |
| `PLOT_MAX_CODES` | Maximum number of codes plotted (the most frequent first) |
| `EMG_PATTERNS` | Regexes (full match, case-insensitive) for EMG channel names, e.g. `AUX\d+`, `EMG\d+` |
| `EEG_PATTERNS` | Regexes for EEG names that are not 10-20 labels, e.g. `EEG\d+` |
| `IGNORE_PATTERNS` | Regexes for channels to skip (trigger, EOG, MISC) |
| `THRESHOLDS` | Quality thresholds, see below |

A channel is classified in this order: ignore pattern → EMG pattern → 10-20/10-10 label or EEG pattern → XDF channel type (`EMG`/`AUX` → EMG, `EEG` → EEG) → other. Channels classified as "other" are listed at the end of the report but not checked. Mastoids (`M1`, `M2`) count as EEG.

## What is checked

| Check | Fails when |
|---|---|
| Header consistency | The XDF header lists a different number of labels than the data has columns (missing labels show as `<unlabeled N>`), or the effective sampling rate differs from the nominal rate by more than 1% (dropped samples) |
| Sampling rate | It differs from `EXPECTED_FS` |
| Duplicate names | Two channels share a name |
| EMG / EEG count | The count is not exactly N / M |
| EEG positions | An EEG channel is numbered (`EEG001`, `Ch1`) or has a name that is not a 10-20/10-10 label |
| Dead channels | More than P EMG or Q EEG channels are dead |
| Markers | No markers were found, or a code in `EXPECTED_MARKER_CODES` is missing |
| Suspicious channels | Any channel looks noisy (warning only) |

## Markers

Markers come from the file when it has them: XDF marker streams (aligned to the signal by timestamp), or the BrainVision `.vmrk`. If the file has none, they are read from a trigger channel, with one marker each time its value changes to a non-zero code. The script tries the channel named `TRIGGER` first, then the last column, which is where eego writes the trigger. A column only counts as a trigger if it is integer-coded with at most 256 distinct values. When the trigger is found in the last column but not under the `TRIGGER` label, the run fails with a "labels are shifted" note. That is the case for P015: its header has 65 labels for 177 columns, so the trigger sits in column 177.

The summary ends with a table listing each code with its count and the times of its first and last occurrence.

## Marker plot

With `PLOT = True`, `<file stem>_markers.png` is saved next to the summary:

1. A raster of the marker times, coloured by code.
2. A 100 ms RMS envelope of the EMG channels and `PLOT_EEG_CHANNELS` over the whole run, with the markers overlaid.
3. One panel per code showing the raw channels (each normalised) in `PLOT_WINDOW_S` around that code's first marker.

## Quality features

Every feature is unit-free, so the same thresholds work for data in V or in µV.

| Feature | Meaning | Rule |
|---|---|---|
| `nan` | Fraction of non-finite samples | dead if ≥ `nan_fraction` |
| `flat` | Fraction of consecutive samples with the same value | dead if ≥ `flat_fraction` |
| `clip` | Fraction of samples sitting at the channel's min or max | dead if ≥ `clip_fraction` |
| `line` | Share of the 1 Hz–Nyquist power within ±1 Hz of `line_freq` | warning if ≥ `line_noise_ratio` |
| `z` | Robust z-score of log(std) against the other channels of the same kind | warning if \|z\| ≥ `robust_z` |
| `emgband` | Share of the 1 Hz–Nyquist power that falls in `emg_band` (EMG channels only) | warning if < `emg_min_band_ratio` |

The spectrum is a Welch estimate built from up to `welch_max_segments` 2 s Hann windows spread across the recording. `emgband` is low when low-frequency drift or motion dominates. That is common in raw, unfiltered EMG, so treat this warning as a reason to look at the channel, not as proof that it is broken.

## Files

| File | Content |
|---|---|
| `check.py` | Inputs, report and entry point |
| `loaders.py` | XDF and BrainVision readers returning a `Recording` |
| `checks.py` | Channel classification, 10-20 label set and quality features |
| `plots.py` | Marker plot |

The plot needs `matplotlib`, which the poetry environment already has.
