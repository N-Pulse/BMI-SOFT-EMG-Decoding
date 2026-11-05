#!/usr/bin/env bash
# Shared helpers for batch pipelines. Source it, do not execute it:
#   source "${BMIEMG_ROOT}/cli/common/env.sh"
#
# Environment variables (all optional):
#   BMIEMG_ROOT      repo root override (default: resolved from this file)
#   BMIEMG_VENV      path to a virtualenv to activate instead of poetry
#
# After setup_env, BMIEMG_PYTHON holds the command prefix to run python.

# 1. Repo root
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BMIEMG_ROOT="${BMIEMG_ROOT:-$(cd "${_COMMON_DIR}/../.." && pwd)}"
export BMIEMG_ROOT

# 2. Logging and stage banners
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# run_stage <name> <command...>: banner, run, report elapsed time.
# A failing command aborts the caller because of `set -e`.
run_stage() {
    local name="$1"
    shift
    local start=$SECONDS
    printf '\n========== STAGE: %s ==========\n' "${name}"
    log "start: $*"
    "$@"
    log "done: ${name} ($((SECONDS - start))s)"
}

skip_stage() {
    printf '\n========== STAGE: %s (skipped: %s) ==========\n' "$1" "$2"
}

# 3. Environment
setup_env() {
    cd "${BMIEMG_ROOT}"

    # Cluster-specific: uncomment and adapt to your site.
    # module load gcc python/3.12 cuda

    if [[ -n "${BMIEMG_VENV:-}" ]]; then
        # shellcheck disable=SC1091
        source "${BMIEMG_VENV}/bin/activate"
        BMIEMG_PYTHON="python"
    elif command -v poetry >/dev/null 2>&1; then
        BMIEMG_PYTHON="poetry run python"
    else
        log "ERROR: BMIEMG_VENV is unset and poetry is not on PATH"
        return 1
    fi
    export BMIEMG_PYTHON
    log "python: ${BMIEMG_PYTHON}"
}

# 4. Device detection (informational, never fails)
detect_device() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        log "device: GPU ($(nvidia-smi -L | head -n1))"
        log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
        log "device: GPU requested (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}), nvidia-smi unavailable"
    else
        log "device: CPU only"
    fi
}
