import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import mne
    import pyxdf

    import numpy as np
    import marimo as mo
    import pandas as pd
    from matplotlib import pyplot as plt

    from pathlib import Path

    return Path, np, pyxdf


@app.cell
def _(Path):
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA: Path = ROOT / "data" / "bids"

    FILE: Path = DATA / "sub-05/ses-02/sourcedata/sub-05_ses-02_task-Up_run-01_raw.xdf"

    print(f"Does path exist? {FILE.exists()}")
    return (FILE,)


@app.cell
def _(FILE: "Path", pyxdf):
    streams, header = pyxdf.load_xdf(FILE)
    return (streams,)


@app.cell
def _(streams):
    print(type(streams[0]))
    return


@app.cell
def _(streams):
    channel_list = streams[1]["info"]["desc"][0]["channels"][0]["channel"]

    for ch in channel_list:
        print(ch["type"])
    return


@app.cell
def _(streams):
    for i, stream in enumerate(streams):
        keys = stream.keys()

        print(f"Stream {i}")
        print(f"  keys:  {keys}")
    return


@app.cell
def _(streams):
    for j, stream1 in enumerate(streams):
        info = stream1["info"]
        name = info["name"][0]
        stream_type = info["type"][0]
        sfreq = info["nominal_srate"][0]
        shape = getattr(stream1["time_series"], "shape", None)

        print(f"Stream {j}")
        print("  info keys")
        print(f"    name:  {name}")
        print(f"    type:  {stream_type}")
        print(f"    sfreq: {sfreq}")
        print(f"    shape: {shape}")
    return


@app.cell
def _(np):
    def print_keys(
        obj, 
        max_indent: int, 
        name: str = "root", 
        indent: int = 0, 
        show_list_content: bool = False,
        max_content: int = 10
    ):
        prefix = "    " * indent

        if isinstance(obj, dict) and indent < max_indent:
            print(f"{prefix}{name}: dict with keys {list(obj.keys())}")

            for key, value in obj.items():
                print_keys(
                    value, 
                    max_indent, 
                    name=key, 
                    indent=indent + 1, 
                    show_list_content=show_list_content,
                    max_content=max_content
                )

        elif (isinstance(obj, list) or isinstance(obj, np.ndarray)) and indent < max_indent:
            value = len(obj) if isinstance(obj, list) else obj.shape
            content = obj if len(obj) < max_content else np.nan
            print(f"{prefix}{name}: list with {value} items - {content}")

            if show_list_content:
                for i, value in enumerate(obj):
                    print_keys(
                        value, 
                        max_indent, 
                        name=f"[{i}]", 
                        indent=indent + 1, 
                        show_list_content=show_list_content,
                        max_content=max_content
                    )
        elif isinstance(obj, int):
            print(f"{prefix}{name}: {type(obj).__name__} - {obj}")
        else:
            print(f"{prefix}{name}: {type(obj).__name__}")

    return (print_keys,)


@app.cell
def _(print_keys, streams):
    for k, stream2 in enumerate(streams):
        if k == 0:
            continue
        print(f"Stream {k}")
        print_keys(stream2, 4, name="stream", indent=0)
    return


@app.cell
def _(print_keys, streams):
    for i1, stream3 in enumerate(streams):
        if i1 == 0:
            continue
        print(f"Stream {i1}")
        desc = stream3["info"]["desc"][0]
        print_keys(desc, 3, name="desc", indent=0)
        channels = stream3["info"]["desc"][0]["channels"][0]
        print_keys(channels, 3, name="channels", indent=0)
        acquisition = stream3["info"]["desc"][0]["acquisition"][0]
        print_keys(acquisition, 3, name="acquisition", indent=0)
    return


@app.cell
def _(print_keys, streams):
    for i2, stream4 in enumerate(streams):
        if i2 == 0:
            continue
        print(f"Stream {i2}")
        clock_offsets = stream4["footer"]["info"]["clock_offsets"][0]
        print_keys(clock_offsets, 2, name="clock_offsets", indent=0)
    return


@app.cell
def _(streams):
    streams[0]["time_stamps"]
    #streams[0]["time_series"]
    return


@app.cell
def _(streams):
    streams[1]["time_stamps"][:-1] - streams[1]["time_stamps"][1:]
    return


@app.cell
def _(np, streams):
    len(np.unique(streams[0]["time_series"]))
    return


@app.cell
def _():
    return


app._unparsable_cell(
    r"""
    CONVERSION_SPECIAL: dict = {
        9701: "resting state, eyes open",
        9702: "resting state, eyes closed",
        8888: "start LabRecorder",
        9999: "a test marker (to test befiore the experiment start)",
        8899: "experiment finished",
    }

    PHASE_CONVERSION: dict = {
        1: "cue",
        2: "prep",
        3: "move",
        4: "return",
        5: "iti",
    
    }

    def convert_marks(marks: np.ndarray, conversion: dict = CONVERSION_SPECIAL) -> np.ndarray:
        marks = marks.squeeze()
        converted = []

        for mark in marks:
            mark = int(mark)

            label = CONVERSION_SPECIAL.get(mark, mark)
            if isinstance(label, int):
            
            converted.append(label)

        return np.asarray(converted)
    """,
    name="_"
)


@app.cell
def _(convert_marks, streams):
    convert_marks(streams[0]["time_series"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
