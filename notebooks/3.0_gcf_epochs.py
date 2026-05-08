import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


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
    We can also inspect by movement and see the average signal
    """)
    return


@app.cell
def _(emg_epochs, mo):
    fig2 = emg_epochs["32101"].average(picks=["AUX8"]).plot(picks="all", show=False)
    mo.mpl.interactive(fig2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    However, this is very noisy, so we need to apply an envelop. We can achieve this with `get_envelop()` function
    """)
    return


@app.cell
def _(emg_epochs, get_envelop):
    emg_envelop_epochs = get_envelop(emg_epochs)
    return (emg_envelop_epochs,)


@app.cell
def _(emg_envelop_epochs, mo):
    fig3 = emg_envelop_epochs["32101"].average(picks=["AUX8"]).plot(picks="all", show=False)
    mo.mpl.interactive(fig3)
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
    def extract_emg_features(X: np.ndarray) -> np.ndarray:
        """
        X shape: (n_epochs, n_channels, n_times)

        Returns:
            features shape: (n_epochs, n_features)
        """

        rms = np.sqrt(np.mean(X ** 2, axis=2))
        mav = np.mean(np.abs(X), axis=2)
        var = np.var(X, axis=2)

        features = np.concatenate([rms, mav, var], axis=1)

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

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
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
