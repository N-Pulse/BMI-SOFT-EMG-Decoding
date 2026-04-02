import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Notebook
    This document will be used for me to get used both with the marimo setup, but also with current code usage
    """)
    return


@app.cell
def _():
    import os
    import pyxdf

    import marimo as mo
    import numpy as np
    import polars as pl

    from pathlib import Path
    from scipy import signal

    return Path, mo, np, os, pl, pyxdf, signal


@app.cell
def _(Path):
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA: Path = ROOT / "data"

    SUBJECT = '008'
    SESSION = '001'
    return DATA, SESSION, SUBJECT


@app.cell
def _(np, signal):
    def notch_filter(df, fs=1000, freq=50.0, q=30.0, single=False):
        """Remove power line interference at 50 Hz (or 60 Hz for US)"""
        b, a = signal.iirnotch(freq, q, fs)
    
        filtered_df = df.copy()
        for col in df.columns:
            # Skip non-time-series columns
            if col == 'window_index' or col == 'label':
                continue

            if not single:
                # Apply filter only to columns that contain arrays
                filtered_df[col] = df[col].apply(
                    lambda x: signal.filtfilt(b, a, x) if isinstance(x, (list, np.ndarray)) and len(x) > 1 else x
                )
            else:
                filtered_df[col] = signal.filtfilt(b, a, df[col])
    
        return filtered_df


    def passband_filter(df, fs=1000, lowcut=20.0, highcut=300.0, order=4, single=False):
        """Bandpass filter for EMG signals (20-450 Hz)"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
    
        filtered_df = df.copy()
        for col in df.columns:
            # Skip non-time-series columns
            if col == 'window_index' or col == 'label':
                continue

            if not single:
                # Apply filter only to columns that contain arrays
                filtered_df[col] = df[col].apply(
                    lambda x: signal.filtfilt(b, a, x) if isinstance(x, (list, np.ndarray)) and len(x) > 1 else x
                )
            else:
                filtered_df[col] = signal.filtfilt(b, a, df[col])
    
        return filtered_df

    return notch_filter, passband_filter


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Exploration
    Here I will be exploring the data exploration setup. In the next cell you can inspect very quickly what a Kraken data loooks like
    """)
    return


@app.cell
def _(DATA: "Path", SESSION, SUBJECT, os, pl, pyxdf):
    xdf_dir = DATA / f"raw/sub-P{SUBJECT}/ses-S{SESSION}/emg/"
    xdf_file = f'sub-P{SUBJECT}_ses-S{SESSION}_task-Default_run-001_emg_kraken.xdf'
    data_xdf_path = os.path.join(xdf_dir, xdf_file)

    streams, header = pyxdf.load_xdf(data_xdf_path)
    channels = streams[1]['time_series']
    timestamps = streams[1]['time_stamps']

    data_df = pl.DataFrame(
        channels,
        schema=['Channel_1', 'Channel_2', 'Channel_3', 'Channel_4', 'Channel_5', 'Channel_6']
    )

    data_df = data_df.with_columns(
        pl.Series("Time", timestamps)
    )

    data_df
    return (data_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can then filter the dataset with a notch filter
    """)
    return


@app.cell
def _(data_df, notch_filter, passband_filter, pd):
    time_df = data_df['Time']
    channels_df = data_df.drop('Time')

    filtered1_channels = notch_filter(channels_df, single=True)
    filtered2_channels = passband_filter(filtered1_channels, single=True)

    filtered_data_df = pd.concat([time_df, filtered2_channels], axis=1)
    filtered_data_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
