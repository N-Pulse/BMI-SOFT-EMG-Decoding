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
    import pandas as pd

    from pathlib import Path
    from scipy import signal

    from bmiemg.preprocessing import notch_filter, passband_filter
    from bmiemg.snr import known_noise, SNR
    from bmiemg.plots import plot_stacked_channels

    return (
        Path,
        known_noise,
        mo,
        notch_filter,
        os,
        passband_filter,
        pd,
        plot_stacked_channels,
        pyxdf,
    )


@app.cell
def _(Path):
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA: Path = ROOT / "data"

    SUBJECT = '008'
    SESSION = '001'
    return DATA, SESSION, SUBJECT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Exploration
    Here I will be exploring the data exploration setup. In the next cell you can inspect very quickly what a Kraken data loooks like
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
    return (data_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can then filter the dataset with a notch filter
    """)
    return


@app.cell
def _(data_df, notch_filter, passband_filter, pd, plot_stacked_channels):
    time_df = data_df['Time']
    channels_df = data_df.drop(columns='Time')

    filtered1_channels = notch_filter(channels_df, single=True)
    filtered2_channels = passband_filter(filtered1_channels, single=True)

    filtered_data_df = pd.concat([time_df, filtered2_channels], axis=1)
    filtered_data_df = filtered_data_df.drop(columns=["Channel_5", "Channel_6"])

    plot_stacked_channels(filtered_data_df)
    return (filtered_data_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can evaluate the performance of our flitering by analysing the SNR. In this case we use known-noise approach, where a part of the recording with signl is taken and is compared to a part of the recording that there is no signal (baseline noise)
    """)
    return


@app.cell
def _(filtered_data_df, known_noise, plot_stacked_channels):
    sig = filtered_data_df.loc[(filtered_data_df['Time'] >= 336120) & (filtered_data_df['Time'] < 336200)].copy()
    noise  = filtered_data_df.loc[(filtered_data_df['Time'] >= 336020) & (filtered_data_df['Time'] < 336100)].copy()
    plot_stacked_channels(sig)
    plot_stacked_channels(noise)

    snr = known_noise(filtered_data_df, (336120, 336200), (336020, 336100))
    snr.summary
    return


if __name__ == "__main__":
    app.run()
