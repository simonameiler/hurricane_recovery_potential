#!/bin/bash
#SBATCH --job-name=make_na_exposure
#SBATCH --output=logs/make_na_exp_%j.log
#SBATCH --error=logs/make_na_exp_%j.err
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail
set -x

module purge 2>/dev/null || true
source ~/.bashrc
conda activate climada_env

# --- repo + imports ---
REPO="$HOME/repos/hurricane_recovery_potential"
cd "$REPO"
mkdir -p logs
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# --- scratch for temps (faster I/O) ---
export TMPDIR="${SCRATCH:-/tmp}"
mkdir -p "$TMPDIR"

which python
python --version
python -u scripts/make_NA_exposure.py