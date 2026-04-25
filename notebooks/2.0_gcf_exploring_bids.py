import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Notebook
    This notebook will serve as a way to inspect the bids emg data format
    """)
    return


@app.cell
def _():
    import os
    import pyxdf

    import marimo as mo
    import numpy as np
    import polars as pl
    import pandas as pd

    from pathlib import Path
    from scipy import signal

    from bmiemg.preprocessing import notch_filter, passband_filter
    from bmiemg.snr import known_noise, SNR
    from bmiemg.plots import plot_stacked_channels

    return Path, mo, os, pd, pyxdf


@app.cell
def _(Path):
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA: Path = ROOT / "data" / "bids"

    SUBJECT = '05'
    SESSION = '01'
    return DATA, SESSION, SUBJECT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Exploration
    Here I will be exploring the data exploration setup. In the next cell you can inspect very quickly what the setup at Campus Biotech data loooks like fro EMG
    """)
    return


@app.cell
def _(DATA: "Path", SESSION, SUBJECT, os, pd, pyxdf):
    xdf_dir = DATA / f"raw/sub-P{SUBJECT}/ses-S{SESSION}/emg/"
    xdf_file = f'sub-P{SUBJECT}_ses-S{SESSION}_task-Default_run-001_emg_kraken.xdf'
    data_xdf_path = os.path.join(xdf_dir, xdf_file)

    streams, header = pyxdf.load_xdf(data_xdf_path)
    channels = streams[1]['time_series']
    timestamps = streams[1]['time_stamps']

    data_df = pd.DataFrame(
        channels,
        columns=['Channel_1', 'Channel_2', 'Channel_3', 'Channel_4', 'Channel_5', 'Channel_6']
    )
    data_df['Time'] = timestamps

    data_df
    return


if __name__ == "__main__":
    app.run()
