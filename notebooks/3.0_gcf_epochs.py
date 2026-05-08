import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import mne

    import numpy as np
    import pandas as pd
    import marimo as mo
    from matplotlib import pyplot as plt

    from pathlib import Path

    from bmiemg.data.convert import session_load, ChannelSplitter

    return ChannelSplitter, Path, mne, np, pd, plt, session_load


@app.cell
def _(Path):
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA: Path = ROOT / "data" / "bids"

    FILE: Path = DATA / "sub-05/ses-02/sourcedata/sub-05_ses-02_task-Up_run-01_raw.xdf"
    return (FILE,)


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
def _(emg_signal):
    movement_labels = sorted({
        desc for desc in emg_signal.annotations.description
        if str(desc).startswith("3")
    })
    print(movement_labels)
    return (movement_labels,)


@app.cell
def _(emg_signal, mne):
    events, event_id = mne.events_from_annotations(
        emg_signal,
        event_id={
            '32101': 32101,
            '32102': 32102,
            '32103': 32103,
            '32104': 32104,
            '32106': 32106,
            '32107': 32107,
            '32108': 32108,
            '32111': 32111,
            '32112': 32112,
            '32113': 32113,
            '32114': 32114,
            '32116': 32116,
        },
    )
    return event_id, events


@app.cell
def _(emg_signal, event_id, events, mne):
    raw_centered = emg_signal.copy().load_data()

    raw_centered.apply_function(
            lambda x: x - x.mean(),
            picks="all",
            channel_wise=True,
        )


    epochs = mne.Epochs(
            raw=emg_signal,
            events=events,
            event_id=event_id,
            tmin=-0.5,
            tmax=5.0,
            baseline=(-0.5, 0.0),
            preload=True,
        )
    return (epochs,)


@app.cell
def _(epochs):
    print(epochs.ch_names)
    print(epochs.get_channel_types())
    print(epochs.info)
    return


@app.cell
def _(epochs, plt):
    epochs.plot(picks=epochs.ch_names)
    plt.show()
    return


@app.cell
def _(np, pd):
    def clean_marker(desc) -> str:
        return (
            str(desc)
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
            .strip()
        )

    def time_from_marker_to_next(raw, target_marker: str = "32101") -> pd.DataFrame:
        annotations = raw.annotations

        onsets = np.asarray(annotations.onset, dtype=float)
        descriptions = np.array([clean_marker(d) for d in annotations.description])

        # Ensure annotations are ordered by time
        order = np.argsort(onsets)
        onsets = onsets[order]
        descriptions = descriptions[order]

        rows = []

        for i, desc in enumerate(descriptions):
            if desc != target_marker:
                continue

            # Skip if this is the last marker
            if i + 1 >= len(descriptions):
                continue

            start_time = onsets[i]
            next_time = onsets[i + 1]
            next_marker = descriptions[i + 1]

            rows.append(
                {
                    "marker": desc,
                    "start_time_s": start_time,
                    "next_marker": next_marker,
                    "next_time_s": next_time,
                    "duration_s": next_time - start_time,
                }
            )

        return pd.DataFrame(rows)

    return (time_from_marker_to_next,)


@app.cell
def _(emg_signal, movement_labels, np, time_from_marker_to_next):
    dur = []
    for lab in movement_labels:
        durations_32101 = time_from_marker_to_next(emg_signal, target_marker=lab)
        dur.append(durations_32101["duration_s"].mean())

    #durations_32101 = time_from_marker_to_next(emg_signal, target_marker="32101")

    print(f"{np.mean(dur)} ± {np.std(dur)}")
    return


@app.cell
def _(epochs):
    print(epochs)
    print(epochs.event_id)
    print(epochs.ch_names)
    print(epochs.get_channel_types())
    print(epochs.get_data().shape)
    return


@app.cell
def _(epochs):
    X = epochs.get_data()
    print(X.shape)
    return


@app.cell
def _(epochs, np):
    event_codes = epochs.events[:, -1]

    for label, code in epochs.event_id.items():
        count = np.sum(event_codes == code)
        print(label, count)
    return


@app.cell
def _(epochs, mne, np):
    from scipy.ndimage import uniform_filter1d


    def moving_rms_envelope(
        epochs: mne.Epochs,
        window_s: float = 0.200,
        picks: str | list[str] | None = "all",
    ) -> mne.Epochs:
        env = epochs.copy().load_data()

        sfreq = env.info["sfreq"]
        window_samples = max(1, int(round(window_s * sfreq)))

        def rms(x: np.ndarray) -> np.ndarray:
            return np.sqrt(
                uniform_filter1d(
                    x ** 2,
                    size=window_samples,
                    mode="reflect",
                )
            )

        env.apply_function(
            rms,
            picks=picks,
            channel_wise=True,
        )

        return env

    envelope_epochs_2 = moving_rms_envelope(
        epochs,
        window_s=0.100,
    )
    return (envelope_epochs_2,)


@app.cell
def _(envelope_epochs_2, plt):
    envelope_epochs_2.plot(picks=envelope_epochs_2.ch_names)
    plt.show()
    return


@app.cell
def _(envelope_epochs_2):
    envelope_epochs_2["32101"].average(picks=["all"]).plot(picks="all")
    return


@app.cell
def _(epochs):
    def _():
        print("Epoch channels:", epochs.ch_names)
        print("Epoch channel types:", epochs.get_channel_types())
        print("Epoch data shape:", epochs.get_data().shape)

        evoked = epochs

        print("Evoked channels:", evoked.ch_names)
        return print("Evoked data shape:", evoked.get_data().shape)


    _()
    return


@app.cell
def _(epochs):
    for movement_label in epochs.event_id:
        epochs[movement_label].average().plot()
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
def _(epochs, extract_emg_features):
    X_signal = epochs.get_data()
    y = epochs.events[:, -1]

    X_features = extract_emg_features(X_signal)

    print(X_features.shape)
    print(y.shape)
    return X_features, y


@app.cell
def _(X_features, y):
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    scores = cross_val_score(clf, X_features, y, cv=5)

    print(scores)
    print("Mean accuracy:", scores.mean())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
