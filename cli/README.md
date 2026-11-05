# cli

Batch pipelines that chain the numbered entry scripts in `scripts/` into one reproducible job. They run on SLURM clusters (GPU or CPU-only) and locally with plain `bash`.

## Layout

```
cli/
├── common/env.sh        shared helpers (repo root, environment, device log, stage banners)
├── epn612/run.sh        EPN612 route
└── README.md
```

One folder per pipeline route, each with a single `run.sh` that sources `common/env.sh` and runs its stages in order, stopping on the first failure (`set -euo pipefail`).

## EPN612 route

| Order | Stage | Script | Skip with |
|---|---|---|---|
| 1 | download | `scripts/00_download/download_epn612.py` | `SKIP_DOWNLOAD=1` |
| 2 | prune | `scripts/01_prune_dataset/prune_epn612.py` | `SKIP_PRUNE=1` |
| 3 | train/evaluate | placeholder, not implemented yet | `SKIP_TRAIN=1` |

The dataset lands in `data/raw/emg_epn_612`. EPN612 comes from Zenodo, so no `.env` credentials are needed. Credentials (`MAKER_USERNAME`, `MAKER_APP_PASSWORD`, `MAKER_DAV_USER`) are only needed by routes that use the EPFL archive.

## Submit on SLURM

Run from the repo root. SLURM does not create the log directory, so create it first. The `#SBATCH` header holds placeholders for account, partition and mail address (nothing personal is committed), so pass them on the command line:

```bash
mkdir -p logs/epn612
sbatch --account=<account> --partition=<partition> --mail-user=<you@example.org> cli/epn612/run.sh
```

Command-line flags override the `#SBATCH` lines, so the same flags change any resource: `--time=08:00:00`, `--mem=32G`, `--cpus-per-task=16`, `--job-name=...`. To drop mail notifications use `--mail-type=NONE`.

Logs go to `logs/epn612/<jobname>_<jobid>.log` under the submit directory (`logs/` is git-ignored).

## Run on CPU only

`#SBATCH` lines cannot be conditional, so override on the command line: pick a CPU partition and cancel the GPU request.

```bash
sbatch --account=<account> --partition=<cpu-partition> --gpus-per-task=0 cli/epn612/run.sh
```

If your SLURM rejects `--gpus-per-task=0`, delete the `--gpus-per-task` line from the header instead. The job logs the detected device (`GPU (...)` or `CPU only`) at start.

## Run locally

```bash
bash cli/epn612/run.sh
```

## Skip stages

Set the toggles in the environment. They are forwarded by sbatch by default:

```bash
SKIP_DOWNLOAD=1 sbatch ... cli/epn612/run.sh
SKIP_DOWNLOAD=1 SKIP_PRUNE=1 bash cli/epn612/run.sh
```

## Environment

`common/env.sh` `cd`s to the repo root (`BMIEMG_ROOT`, resolved automatically, `SLURM_SUBMIT_DIR` under sbatch) and picks the Python environment:

- `BMIEMG_VENV=/path/to/venv` activates that virtualenv and uses `python`.
- Otherwise `poetry run python` is used (poetry must be on `PATH`).

Cluster modules (`module load ...`) are site-specific: uncomment and adapt the hook in `setup_env`.

## Add a new route

1. Create `cli/<route>/run.sh`, copying the EPN612 one (header, `set -euo pipefail`, `source cli/common/env.sh`, `setup_env`, `detect_device`).
2. Add one `run_stage "<name>" ${BMIEMG_PYTHON} scripts/<NN_stage>/<script>.py` per stage, in order, each behind a `SKIP_<STAGE>` toggle.
3. Point `--output` at `logs/<route>/%x_%j.log`.
4. Add a section for the route in this README. Put anything reusable in `common/`, never route-specific logic.
