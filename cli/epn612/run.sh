#!/usr/bin/env bash
#SBATCH --job-name=epn612
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/epn612/%x_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=CHANGE_ME

set -euo pipefail

# 0. Locate the repo root; under sbatch $0 is a spooled copy, so use the submit dir
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    BMIEMG_ROOT="${SLURM_SUBMIT_DIR}"
else
    BMIEMG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export BMIEMG_ROOT
# shellcheck disable=SC1091
source "${BMIEMG_ROOT}/cli/common/env.sh"

setup_env
log "job: ${SLURM_JOB_ID:-local} on $(hostname)"
detect_device

# 1. Stages
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
    skip_stage "download" "SKIP_DOWNLOAD=1"
else
    run_stage "download" ${BMIEMG_PYTHON} scripts/00_download/download_epn612.py
fi

if [[ "${SKIP_PRUNE:-0}" == "1" ]]; then
    skip_stage "prune" "SKIP_PRUNE=1"
else
    run_stage "prune" ${BMIEMG_PYTHON} scripts/01_prune_dataset/prune_epn612.py
fi

# PLACEHOLDER: model training and evaluation are not implemented yet.
# Replace with e.g. run_stage "train" ${BMIEMG_PYTHON} scripts/03_model/<script>.py
if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    skip_stage "train/evaluate" "SKIP_TRAIN=1"
else
    skip_stage "train/evaluate" "placeholder, not implemented yet"
fi

log "pipeline finished"
