import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import mne
    import contextlib
    import io
    import gc

    import numpy as np
    import pandas as pd
    import marimo as mo
    from matplotlib import pyplot as plt

    from pathlib import Path
    from datetime import date, datetime
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    from bmiemg.data.convert import session_load, ChannelSplitter
    from bmiemg.data.epoch import SignalPartitioner, V2_TRIGGER_MAP, average_movement_duration
    from bmiemg.preprocessing import get_envelop

    return (
        ChannelSplitter,
        DecisionTreeClassifier,
        Path,
        SignalPartitioner,
        StratifiedKFold,
        V2_TRIGGER_MAP,
        contextlib,
        cross_val_score,
        date,
        datetime,
        get_envelop,
        io,
        mne,
        mo,
        np,
        pd,
        plt,
        session_load,
    )


@app.cell
def _():
    SESSIONS_TO_IGNORE = [

    ]
    SESSIONS_TO_IGNORE_SET = set(SESSIONS_TO_IGNORE)
    return (SESSIONS_TO_IGNORE_SET,)


@app.cell
def _(Path, SESSIONS_TO_IGNORE_SET, date, datetime, mo):
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA: Path = ROOT / "data" / "bids"

    CUTOFF_DATE = date(2025, 11, 21)


    def get_date_from_parents(path: Path) -> date | None:
        for parent in path.parents:
            try:
                return datetime.strptime(parent.name, "%Y-%m-%d").date()
            except ValueError:
                continue

        return None


    XDF_FILES = [
        file
        for file in DATA.rglob("*.xdf")
        if file.is_file()
        and file.name not in SESSIONS_TO_IGNORE_SET
        and (folder_date := get_date_from_parents(file)) is not None
        and folder_date >= CUTOFF_DATE
    ]

    with mo.redirect_stdout():
        print(f"Files for this demo: {len(XDF_FILES)}")
    return (XDF_FILES,)


@app.cell
def _(ChannelSplitter, XDF_FILES, contextlib, io, session_load):
    biotech_splitter = ChannelSplitter()

    emg_signals = []
    for file in XDF_FILES:
        # 1. Load the session
        print(f"Loading {file}")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            session = session_load(file)
            bio_signal = biotech_splitter.split(session)
            signal = bio_signal.attach_annotations()
            emg_signals.append(signal["EMG"])

        del(bio_signal)
        del(signal)
        del(session)
    return (emg_signals,)


@app.cell
def _(SignalPartitioner, V2_TRIGGER_MAP, emg_signals):
    epochs_group = []
    partinioner = SignalPartitioner(V2_TRIGGER_MAP)
    for emg_signal in emg_signals:
        epochs_group.append(partinioner.partition(emg_signal))
    return epochs_group, partinioner


@app.cell
def _(epochs_group, mne):
    merged_epochs = mne.concatenate_epochs(epochs_group)
    return (merged_epochs,)


@app.cell
def _(mne, np):
    # ================================================================
    # 0. Section: IMPORTS
    # ================================================================
    import warnings



    # ================================================================
    # 1. Section: Functions
    # ================================================================
    def group(
        epochs: mne.Epochs,
        target_code: dict[str, list[int]],
        rest_label: str = "rest",
    ) -> mne.EpochsArray:
        grouped_event_id = {
            group_name: idx + 1
            for idx, group_name in enumerate(target_code.keys())
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
                warnings.warn(f"Event code {old_code} not found in epochs.event_id.")
                continue

            old_label = code_to_label[old_code]

            try:
                mov_code = movement_suffix(old_label)
            except ValueError:
                warnings.warn(f"Could not parse movement code from marker: {old_label}")
                continue

            assigned_group = rest_label

            for group_name, mov_codes in target_code.items():
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

        for group_name, count in group_counts.items():
            if count == 0:
                warnings.warn(f"No epochs found for group: {group_name}")

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


    return (group,)


@app.cell
def _(group, merged_epochs, partinioner):
    grouped_epochs = group(merged_epochs, partinioner.trigger_map.target_code)
    print(grouped_epochs)

    clean_epochs = grouped_epochs.copy().drop_bad(
        reject={
            "emg": 500e-6,   # example threshold, in volts
        }
    )
    clean_epochs = grouped_epochs.copy().drop_bad(
        flat={
            "emg": 1e-9,
        }
    )

    grouped_epochs = clean_epochs.copy()
    return (grouped_epochs,)


@app.cell
def _(grouped_epochs, mo, plt):
    fig = grouped_epochs.plot(picks=grouped_epochs.ch_names, show=False)
    plt.close(fig)

    out = mo.mpl.interactive(fig)
    out
    return


@app.cell
def _(grouped_epochs, mo):
    window_slider = mo.ui.slider(
        start=0.020,
        stop=0.500,
        step=0.010,
        value=0.100,
        label="RMS window size (s)",
        show_value=True,
        debounce=True,
    )

    movement_labels = list(grouped_epochs.event_id.keys())
    movement_dropdown = mo.ui.dropdown(
        options=movement_labels,
        value=movement_labels[0],
        label="Movement",
        searchable=True,
    )

    channel_names = grouped_epochs.ch_names
    channel_dropdown = mo.ui.dropdown(
        options=channel_names,
        value=channel_names[0],
        label="Channel",
        searchable=True,
    )

    mode_names = [
        'nearest',
        'wrap',
        'mirror',
        'constant',
        'grid-wrap',
        'grid-wrap',
        'grid-constant',
        'grid-constant',
    ]
    mode_dropdown = mo.ui.dropdown(
        options=mode_names,
        value=mode_names[0],
        label="Modes",
        searchable=True,
    )

    controls = mo.hstack([
        window_slider,
        movement_dropdown,
        channel_dropdown,
        mode_dropdown
    ])

    controls
    return channel_dropdown, mode_dropdown, movement_dropdown, window_slider


@app.cell
def _(channel_dropdown, mode_dropdown, movement_dropdown, window_slider):
    window_s = float(window_slider.value)
    movement = movement_dropdown.value
    channel = channel_dropdown.value
    mode = mode_dropdown.value
    return channel, mode, movement, window_s


@app.cell
def _(channel, get_envelop, grouped_epochs, mode, movement, window_s):
    fig2 = grouped_epochs[movement].average(picks=[channel]).plot(picks="all", show=False)
    emg_envelop_epochs = get_envelop(
        grouped_epochs,
        window_s=window_s,
        mode=mode
    )
    fig3 = emg_envelop_epochs[movement].average(picks=[channel]).plot(picks="all", show=False)
    return emg_envelop_epochs, fig2, fig3


@app.cell
def _(fig2, fig3, mo):
    signal_stack = mo.vstack([
        mo.mpl.interactive(fig2),
        mo.mpl.interactive(fig3)
    ])

    signal_stack
    return


@app.cell
def _(np):
    # ================================================================
    # 0. Section: IMPORTS
    # ================================================================


    # ================================================================
    # 1. Section: Time-domain features
    # ================================================================
    def mav(x: np.ndarray) -> np.ndarray:
        """Mean absolute value."""
        return np.mean(np.abs(x), axis=2)


    def std(x: np.ndarray) -> np.ndarray:
        """Standard deviation."""
        return np.std(x, axis=2)


    def var(x: np.ndarray) -> np.ndarray:
        """Variance."""
        return np.var(x, axis=2)


    def maxav(x: np.ndarray) -> np.ndarray:
        """Maximum absolute value."""
        return np.max(np.abs(x), axis=2)


    def rms(x: np.ndarray) -> np.ndarray:
        """Root mean square."""
        return np.sqrt(np.mean(x**2, axis=2))


    def wl(x: np.ndarray) -> np.ndarray:
        """Waveform length."""
        return np.sum(np.abs(np.diff(x, axis=2)), axis=2)


    def ssc(x: np.ndarray) -> np.ndarray:
        """Slope sign changes."""
        dx = np.diff(x, axis=2)
        return np.sum((dx[:, :, :-1] * dx[:, :, 1:]) < 0, axis=2)


    def zc(x: np.ndarray) -> np.ndarray:
        """Zero crossings."""
        return np.sum(np.diff(np.signbit(x), axis=2), axis=2)


    def log_det(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Log detector.

        Uses abs(x) because EMG can be negative.
        """
        return np.exp(np.mean(np.log(np.abs(x) + eps), axis=2))


    def wamp(x: np.ndarray, threshold: float = 20e-6) -> np.ndarray:
        """
        Willison amplitude.

        Counts how often abs(diff) exceeds a threshold.
        Threshold assumes signal is in volts.
        """
        dx = np.abs(np.diff(x, axis=2))
        return np.sum(dx > threshold, axis=2)


    # ================================================================
    # 2. Section: Frequency-domain scalar features
    # ================================================================
    def fft_power(x: np.ndarray) -> np.ndarray:
        """
        One-sided FFT power spectrum.

        Returns:
            shape (n_epochs, n_channels, n_freqs)
        """
        fft = np.fft.rfft(x, axis=2)
        return np.abs(fft) ** 2


    def fft_freqs(x: np.ndarray, sfreq: float) -> np.ndarray:
        """
        One-sided FFT frequency vector.

        Returns:
            shape (n_freqs,)
        """
        return np.fft.rfftfreq(x.shape[2], d=1.0 / sfreq)


    def total_power(x: np.ndarray, sfreq: float) -> np.ndarray:
        """Total spectral power."""
        power = fft_power(x)
        return np.sum(power, axis=2)


    def mean_freq(x: np.ndarray, sfreq: float) -> np.ndarray:
        """Mean frequency."""
        power = fft_power(x)
        f = fft_freqs(x, sfreq)

        denom = np.sum(power, axis=2)
        denom = np.maximum(denom, 1e-12)

        return np.sum(power * f[None, None, :], axis=2) / denom


    def median_freq(x: np.ndarray, sfreq: float) -> np.ndarray:
        """Median frequency based on cumulative spectral power."""
        power = fft_power(x)
        f = fft_freqs(x, sfreq)

        cumulative_power = np.cumsum(power, axis=2)
        half_power = cumulative_power[:, :, -1:] / 2.0

        idx = np.argmax(cumulative_power >= half_power, axis=2)

        return f[idx]


    def peak_freq(x: np.ndarray, sfreq: float) -> np.ndarray:
        """Peak frequency."""
        power = fft_power(x)
        f = fft_freqs(x, sfreq)

        idx = np.argmax(power, axis=2)

        return f[idx]
    
    # ================================================================
    # 3. Section: Feature extraction
    # ================================================================
    TIME_FEATURE_FUNCTIONS = [
        mav,
        std,
        var,
        maxav,
        rms,
        wl,
        ssc,
        zc,
        log_det,
        wamp,
    ]


    FREQ_FEATURE_FUNCTIONS = [
        total_power,
        mean_freq,
        median_freq,
        peak_freq,
    ]


    def extract_emg_features(
        X: np.ndarray,
        sfreq: float,
    ) -> np.ndarray:
        """
        X shape:
            (n_epochs, n_channels, n_times)

        Returns:
            (n_epochs, n_features)

        If there are:
            6 channels
            10 time features
            4 frequency features

        Then:
            n_features = 6 * 14 = 84
        """
        feature_values = []

        print("Input X:", X.shape)

        for feature_func in TIME_FEATURE_FUNCTIONS:
            val = feature_func(X)

            if val.ndim != 2:
                raise ValueError(
                    f"{feature_func.__name__} returned shape {val.shape}, "
                    "expected (n_epochs, n_channels)."
                )

            print(feature_func.__name__, val.shape)
            feature_values.append(val)

        for feature_func in FREQ_FEATURE_FUNCTIONS:
            val = feature_func(X, sfreq)

            if val.ndim != 2:
                raise ValueError(
                    f"{feature_func.__name__} returned shape {val.shape}, "
                    "expected (n_epochs, n_channels)."
                )

            print(feature_func.__name__, val.shape)
            feature_values.append(val)

        features = np.concatenate(feature_values, axis=1)

        print("Final features:", features.shape)

        return features

    return FREQ_FEATURE_FUNCTIONS, TIME_FEATURE_FUNCTIONS, extract_emg_features


@app.cell
def _(emg_envelop_epochs, extract_emg_features):
    X_signal = emg_envelop_epochs.get_data()
    y = emg_envelop_epochs.events[:, -1]

    sfreq = emg_envelop_epochs.info["sfreq"]
    X_features = extract_emg_features(X_signal, sfreq)

    print(X_features.shape)
    print(y.shape)
    return X_features, y


@app.cell
def _():
    model = "pinch"
    return (model,)


@app.cell
def _(
    DecisionTreeClassifier,
    StratifiedKFold,
    X_features,
    cross_val_score,
    emg_envelop_epochs,
    mo,
    model,
    np,
    y,
):
    grasp_code = emg_envelop_epochs.event_id[model]

    y_binary = (y == grasp_code).astype(int)

    print(np.unique(y_binary, return_counts=True))

    clf3 = DecisionTreeClassifier(
        random_state=42,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores3 = cross_val_score(
        clf3,
        X_features,
        y_binary,
        cv=cv,
        scoring="accuracy",
    )

    with mo.redirect_stdout():
        print("Decision tree: grasp vs no grasp")
        print("Scores:", scores3)
        print("Mean accuracy:", scores3.mean())
    return clf3, cv, y_binary


@app.cell
def _(
    FREQ_FEATURE_FUNCTIONS,
    TIME_FEATURE_FUNCTIONS,
    X_features,
    clf3,
    grouped_epochs,
    pd,
    y_binary,
):
    rows = []

    for feature_func in TIME_FEATURE_FUNCTIONS:
        for ch_name in grouped_epochs.ch_names:
            rows.append({
                "feature_type": feature_func.__name__,
                "channel": ch_name,
                "feature": f"{feature_func.__name__}_{ch_name}",
            })

    for feature_func in FREQ_FEATURE_FUNCTIONS:
        for ch_name in grouped_epochs.ch_names:
            rows.append({
                "feature_type": feature_func.__name__,
                "channel": ch_name,
                "feature": f"{feature_func.__name__}_{ch_name}",
            })

    feature_info = pd.DataFrame(rows)

    print(len(feature_info))
    print(X_features.shape[1])

    clf3.fit(X_features, y_binary)
    feature_info["importance"] = clf3.feature_importances_
    return (feature_info,)


@app.cell
def _(X_features, clf3, feature_info, plt, y_binary):
    clf3.fit(X_features, y_binary)

    feature_type_importance = (
        feature_info
        .groupby("feature_type", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    print(feature_type_importance)

    top_n = 20

    plt.figure(figsize=(10, 6))

    plt.barh(
        feature_type_importance["feature_type"][::-1],
        feature_type_importance["importance"][::-1],
    )

    plt.xlabel("Total importance across channels")
    plt.title("Overall feature importance")
    plt.tight_layout()
    plt.show()
    return (feature_type_importance,)


@app.cell
def _(feature_info, plt):
    channel_importance = (
        feature_info
        .groupby("channel", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    print(channel_importance)

    plt.figure(figsize=(8, 5))

    plt.barh(
        channel_importance["channel"][::-1],
        channel_importance["importance"][::-1],
    )

    plt.xlabel("Total importance across features")
    plt.title("Channel importance")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_features, feature_info, feature_type_importance):
    top_feature_types = (
        feature_type_importance
        .head(7)["feature_type"]
        .tolist()
    )

    selected_indices = feature_info.index[
        feature_info["feature_type"].isin(top_feature_types)
    ].to_numpy()

    X_selected = X_features[:, selected_indices]

    print("Selected feature types:", top_feature_types)
    print("Original:", X_features.shape)
    print("Selected:", X_selected.shape)
    return (X_selected,)


@app.cell
def _(X_selected, clf3, cross_val_score, cv, mo, y_binary):
    scores_selected = cross_val_score(
        clf3,
        X_selected,
        y_binary,
        cv=cv,
        scoring="accuracy",
    )

    with mo.redirect_stdout():
        print("Decision tree with selected features")
        print("Scores:", scores_selected)
        print("Mean accuracy:", scores_selected.mean())
    return


@app.cell
def _(DecisionTreeClassifier, StratifiedKFold, cross_val_score, mne, np, y):
    def train_dt_all(
        epochs: mne.EpochsArray | mne.Epochs,
        model_name: str,
        features: np.ndarray
    ) -> tuple:
        model_code = epochs.event_id[model_name]
        y_binary = (y == model_code).astype(int) # 1 vs all

        clf = DecisionTreeClassifier(
            random_state=42,
        )
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        scores = cross_val_score(
            clf,
            features,
            y_binary,
            cv=cv,
            scoring="accuracy",
        )

        clf.fit(features, y_binary)

        return clf, scores

    return (train_dt_all,)


@app.cell
def _(X_selected, emg_envelop_epochs, model, train_dt_all):
    clf, scores = train_dt_all(emg_envelop_epochs, model, X_selected)

    print(f"Decision tree: {model} vs no {model}")
    print("Mean accuracy:", scores.mean())

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
