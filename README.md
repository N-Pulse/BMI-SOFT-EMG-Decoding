<div align="center">
    <img src="https://shields.io/badge/python-3.13-green?logo=python&style=flat" alt="Python">
    <img src="https://shields.io/badge/Package_Manager-poetry-green?logo=poetry&style=flat" alt="Poetry">
</div>

# BMI-SOFT-Signal_Processing_ML

This repository is responsible for the handling of the EMG decoding pipeline: from the raw data to an effective decoder!

## Handling the Data
Because we have an in-house data acquisition pipeline we need a pipeline for Downloading and Loading the data. The downloading the data is currently being implemented and should follow the following architecture:

![image](images/class.png)

The loading the follows a `mne` based approach with simple inheritance from a Loader class

## The Downloader

The downloader lives in `src/bmiemg/download` and is split into three layers: a `service` orchestrator (`Downloader`), the I/O-free `domain` (specs, strategies, registry), and `adapters` that talk to a concrete source. Each source registers its own strategy, so adding one means writing an adapter, not touching the service.

Per-source setup lives with the code: see [`src/bmiemg/download/README.md`](src/bmiemg/download/README.md).

## CNN training and export

Run training from the repository root. Validation runs after every epoch; the
held-out test split is evaluated once at the end.

```bash
# One participant (complete trials are split 70/15/15)
poetry run python scripts/cnn/train.py --user trainingJSON:user1

# Subject-disjoint cohort split (use at least 5 participants)
poetry run python scripts/cnn/train.py --max-users 10

# Full dataset
poetry run python scripts/cnn/train.py
```

Common tuning options are available with `--help`, for example:

```bash
poetry run python scripts/cnn/train.py --max-users 20 --epochs 30 \
  --batch-size 64 --learning-rate 0.0005 --window-size 400 \
  --window-step 200 --model-name CNNv2
```

### CLI options

`--user` and `--max-users` are mutually exclusive. Defaults match `DEFAULT_CONFIG` in `scripts/cnn/train.py`.

| Option | Default | Purpose |
|---|---:|---|
| `--dataset-root PATH` | EPN-612 data directory | Override the dataset location |
| `--user ID` | All participants | Train one participant with trial-disjoint splits, e.g. `trainingJSON:user1` |
| `--max-users N` | All participants | Use the first `N` participants with subject-disjoint splits |
| `--window-size N` | `400` | Window length in adapted 2 kHz samples |
| `--window-step N` | `200` | Step between windows in adapted samples |
| `--max-windows-per-trial N` | `10` | Limit retained windows per trial |
| `--epochs N` | `10` | Number of training epochs |
| `--batch-size N` | `32` | Mini-batch size |
| `--learning-rate X` | `0.001` | Adam learning rate |
| `--device DEVICE` | `auto` | Use `auto`, `cpu`, `cuda`, or `mps` |
| `--model-name NAME` | `CNNv1` | Set the exported model name/version |
| `--output-dir PATH` | `artifacts/cnn` | Set the checkpoint directory |
| `--seed N` | `42` | Set the reproducibility seed |
| `-h`, `--help` | — | Show the current CLI reference |

Advanced parameters such as split ratios, board adaptation, TCNN filters, and Adam settings are edited directly in `DEFAULT_CONFIG`.

Every successful run exports a checkpoint to `artifacts/cnn` by default. The name records the model version, actual participant count, and window duration:

```text
artifacts/cnn/CNNv1_10users_200mswindow.pt
```

Use `--output-dir PATH` to change the directory. The checkpoint contains CPU weights (`state_dict`), model and experiment configuration, class order, input shape and sampling rate, split sizes, and final test metrics. It is the source artifact for subsequent quantization/conversion to the embedded deployment format; it is not directly loadable by a microcontroller.

Run the automated tests with:

```bash
poetry run pytest -q
```
