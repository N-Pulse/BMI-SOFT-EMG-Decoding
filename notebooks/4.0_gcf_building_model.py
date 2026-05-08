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
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    from bmiemg.data.convert import session_load, ChannelSplitter
    from bmiemg.data.epoch import SignalPartitioner, V1_TRIGGER_MAP, average_movement_duration
    from bmiemg.preprocessing import get_envelop

    return (
        ChannelSplitter,
        DecisionTreeClassifier,
        LogisticRegression,
        Path,
        SignalPartitioner,
        StandardScaler,
        V1_TRIGGER_MAP,
        contextlib,
        cross_val_score,
        date,
        datetime,
        get_envelop,
        io,
        make_pipeline,
        mne,
        mo,
        np,
        plt,
        session_load,
    )


@app.cell
def _():
    SESSIONS_TO_IGNORE = [
        "sub-05_ses-04_task-Down_run-01_raw.xdf"
    ]
    SESSIONS_TO_IGNORE_SET = set(SESSIONS_TO_IGNORE)
    return (SESSIONS_TO_IGNORE_SET,)


@app.cell
def _(Path, SESSIONS_TO_IGNORE_SET, date, datetime):
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
        and folder_date < CUTOFF_DATE
    ]
    return (XDF_FILES,)


@app.cell
def _(XDF_FILES, mo):
    with mo.redirect_stdout():
        print(f"Files for this demo: {len(XDF_FILES)}")
    return


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
def _(SignalPartitioner, V1_TRIGGER_MAP, emg_signals):
    epochs_group = []
    for emg_signal in emg_signals:
        partinioner = SignalPartitioner(V1_TRIGGER_MAP)
        epochs_group.append(partinioner.partition(emg_signal))
    return (epochs_group,)


@app.cell
def _(epochs_group, mne):
    merged_epochs = mne.concatenate_epochs(epochs_group)
    return (merged_epochs,)


@app.cell
def _(merged_epochs, mo, plt):
    fig1 = merged_epochs.plot(picks=merged_epochs.ch_names, show=False)
    plt.close(fig1)

    out = mo.mpl.interactive(fig1)
    out
    return


@app.cell
def _(merged_epochs, mo):
    window_slider = mo.ui.slider(
        start=0.020,
        stop=0.500,
        step=0.010,
        value=0.100,
        label="RMS window size (s)",
        show_value=True,
        debounce=True,
    )

    movement_labels = list(merged_epochs.event_id.keys())
    movement_dropdown = mo.ui.dropdown(
        options=movement_labels,
        value=movement_labels[0],
        label="Movement",
        searchable=True,
    )

    channel_names = merged_epochs.ch_names
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
def _(channel, get_envelop, merged_epochs, mode, movement, window_s):
    fig2 = merged_epochs[movement].average(picks=[channel]).plot(picks="all", show=False)
    emg_envelop_epochs = get_envelop(
        merged_epochs,
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
    # 1. Section: Functions
    # ================================================================
    def mav(x):
        """Mean absolute value (MAV)"""
        return np.mean(np.abs(x), axis=2)

    def std(x):
        """Standard Deviation (STD)"""
        return np.std(x, axis=2)

    def var(x):
        """Variance"""
        return np.var(x, axis=2)

    def maxav(x):
        """Maximum absolute Value (MaxAV)"""
        return np.max(np.abs(x), axis=2)

    def rms(x):
        """Root mean square (RMS)"""
        return np.sqrt(np.mean(x**2, axis=2))

    def wl(x):
        """Waveform length (WL)"""
        return np.sum(np.abs(np.diff(x, axis=2)), axis=2)

    def ssc(x):
        """Slope sign changes (SSC)"""
        return np.sum((np.diff(x, axis=2)[:-1] * np.diff(x, axis=2)[1:]) < 0, axis=2)

    def zc(x):
        """Zero Crossing (ZC)"""
        return np.sum(np.diff(np.sign(x), axis=2) != 0)

    def log_det(x):
        """Log detector"""
        return np.exp(1 / len(x) * np.sum(np.log(x), axis=2))

    def wamp(x):
        """Willison amplitude"""
        return np.sum((x > 0.2 * np.std(x)), axis=2)

    def fft_values(x):
        """Frequency domain features (FFT-based) - Value"""
        return np.fft.fft(x, axis=2)

    def fft_magnitude(x):
        """Frequency domain features (FFT-based) - Magntiude"""
        return np.abs(fft_values(x))

    def fft_power(x):
        """Frequency domain features (FFT-based) - Power"""
        return np.square(fft_magnitude(x))

    def freqs(x, srate: float = 1000.0):
        """
        Frequency domain features (FFT-based) - Frequency
        Assuming a sampling rate of 1000 Hz
        """
        return np.fft.fftfreq(x.shape[0], d=1/srate)

    def total_power(x):
        """Total power"""
        return np.sum(fft_power(x), axis=2)

    def mean_freq(x):
        """Mean frequency"""
        return np.sum(freqs(x) * fft_power(x), axis=2) / np.sum(fft_power(x), axis=2)

    def median_freq(x):
        """Median frequency"""
        return np.median(freqs(x) * fft_power(x), axis=2)

    def peak_freq(x):
        """Peak frequency"""
        return freqs(x)[np.argmax(fft_power(x), axis=2)]


    FEATURE_FUNCTIONS: list = [
        mav,
        std,
        var,
        maxav,
        rms,
        wl,
        #ssc,
        #zc,
        log_det,
        wamp,
        #fft_values,
        #fft_magnitude,
        #fft_power,
        #freqs,
        total_power,
        #mean_freq,
        #median_freq,
        #peak_freq,
    ]
    return (FEATURE_FUNCTIONS,)


@app.cell
def _(FEATURE_FUNCTIONS: list, np):
    def extract_emg_features(X: np.ndarray) -> np.ndarray:
        """
        X shape: (n_epochs, n_channels, n_times)

        Returns:
            features shape: (n_epochs, n_features)
        """
        feature_val = []
        print(X.shape)
        print()

        for feature_func in FEATURE_FUNCTIONS:
            #print(feature_func.__name__)
            val = feature_func(X)
            print(val.shape)
            feature_val.append(val)

        features = np.concatenate(feature_val, axis=1)
        print()

        return features

    return (extract_emg_features,)


@app.cell
def _(emg_envelop_epochs, extract_emg_features):
    X_signal = emg_envelop_epochs.get_data()
    y = emg_envelop_epochs.events[:, -1]

    X_features = extract_emg_features(X_signal)

    print(X_features.shape)
    print(y.shape)
    return X_features, y


@app.cell
def _(
    DecisionTreeClassifier,
    StandardScaler,
    X_features,
    cross_val_score,
    make_pipeline,
    mo,
    y,
):
    clf = make_pipeline(
        StandardScaler(),
        #LogisticRegression(max_iter=1000),
        DecisionTreeClassifier(),
    )

    scores = cross_val_score(clf, X_features, y, cv=5)

    with mo.redirect_stdout():
        print("A simple logistic regression will achieve:")
        print("Mean accuracy:", scores.mean())
    return


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    X_features,
    cross_val_score,
    make_pipeline,
    mo,
    y,
):
    clf_2 = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
        #DecisionTreeClassifier(),
    )

    scores_2 = cross_val_score(clf_2, X_features, y, cv=5)

    with mo.redirect_stdout():
        print("A simple logistic regression will achieve:")
        print("Mean accuracy:", scores_2.mean())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
