import marimo

__generated_with = "0.22.0"
app = marimo.App(
    width="medium",
    layout_file="layouts/3.0_gcf_epochs.slides.json",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Getting the building blocks
    To train a model we need more than just the signal. We need to partition it into the blocks of signal vs movement. Thelp us doing this task, we convert the `.xdf` file into the `mne.Raw`, since the `mne` package has some cool and usefull functions.
    """)
    return


@app.cell
def _():
    import mne

    import numpy as np
    import pandas as pd
    import marimo as mo
    from matplotlib import pyplot as plt

    from pathlib import Path

    from bmiemg.data.convert import session_load, ChannelSplitter
    from bmiemg.data.epoch import SignalPartitioner, V1_TRIGGER_MAP, average_movement_duration
    from bmiemg.preprocessing import get_envelop

    return (
        ChannelSplitter,
        Path,
        SignalPartitioner,
        V1_TRIGGER_MAP,
        get_envelop,
        mo,
        np,
        plt,
        session_load,
    )


@app.cell
def _(Path):
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA: Path = ROOT / "data" / "bids"

    FILE: Path = DATA / "sub-05/ses-02/sourcedata/sub-05_ses-02_task-Up_run-01_raw.xdf"
    return (FILE,)


@app.cell
def _(FILE: "Path", mo):
    with mo.redirect_stdout():
        print(f"File for this demo: {str(FILE.name)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first can just extract the emg signal out of the `.xdf
    """)
    return


@app.cell
def _(ChannelSplitter, FILE: "Path", session_load):
    # 1. Load the session
    session = session_load(FILE)
    biotech_splitter = ChannelSplitter()
    return biotech_splitter, session


@app.cell
def _(session):
    session.signal_stream.time_series.shape, session.marker_stream.time_series.shape
    return


@app.cell
def _(biotech_splitter, session):
    bio_signal = biotech_splitter.split(session)
    signal = bio_signal.attach_annotations()
    emg_signal = signal["EMG"]
    return (emg_signal,)


@app.cell
def _(emg_signal, mo):
    print(emg_signal)
    X = emg_signal.get_data()

    with mo.redirect_stdout():
        print(f"The data has the following shape: {X.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then it's time to separate by epochs, that is, the snippets where movement was present. For this session we will use the V1 Protocol, but this will need to get mapped at some point.
    """)
    return


@app.cell
def _(SignalPartitioner, V1_TRIGGER_MAP, emg_signal):
    partinioner = SignalPartitioner(V1_TRIGGER_MAP)
    emg_epochs = partinioner.partition(emg_signal)
    return (emg_epochs,)


@app.cell
def _(emg_epochs, mo, plt):
    fig1 = emg_epochs.plot(picks=emg_epochs.ch_names, show=False)
    plt.close(fig1)

    out = mo.mpl.interactive(fig1)
    out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also inspect by movement and see the average signal. However, this is very noisy, so we need to apply an envelop. We can achieve this with `get_envelop()` function
    """)
    return


@app.cell
def _(emg_epochs, mo):
    window_slider = mo.ui.slider(
        start=0.020,
        stop=0.500,
        step=0.010,
        value=0.100,
        label="RMS window size (s)",
        show_value=True,
        debounce=True,
    )

    movement_labels = list(emg_epochs.event_id.keys())
    movement_dropdown = mo.ui.dropdown(
        options=movement_labels,
        value=movement_labels[0],
        label="Movement",
        searchable=True,
    )

    channel_names = emg_epochs.ch_names
    channel_dropdown = mo.ui.dropdown(
        options=channel_names,
        value=channel_names[0],
        label="Channel",
        searchable=True,
    )

    controls = mo.hstack([
        window_slider,
        movement_dropdown,
        channel_dropdown,
    ])

    controls
    return channel_dropdown, movement_dropdown, window_slider


@app.cell
def _(channel_dropdown, movement_dropdown, window_slider):
    window_s = float(window_slider.value)
    movement = movement_dropdown.value
    channel = channel_dropdown.value
    return channel, movement, window_s


@app.cell
def _(channel, emg_epochs, get_envelop, movement, window_s):
    fig2 = emg_epochs[movement].average(picks=[channel]).plot(picks="all", show=False)
    emg_envelop_epochs = get_envelop(
        emg_epochs,
        window_s=window_s,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building a decoder
    Once the data is in this state, building a decoder becomes incredebily easy:
    """)
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
        print(X.shape)
        print()
        feature_val = []
        for feature_func in FEATURE_FUNCTIONS:
            #print(feature_func.__name__)
            val = feature_func(X)
            print(val.shape)
            feature_val.append(val)
        
        features = np.concatenate(feature_val, axis=1)

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
def _(X_features, mo, y):
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
        #DecisionTreeClassifier(),
    )

    scores = cross_val_score(clf, X_features, y, cv=5)

    with mo.redirect_stdout():
        print("A simple logistic regression will achieve:")
        print("Mean accuracy:", scores.mean())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
