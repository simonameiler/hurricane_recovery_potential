#!/usr/bin/env bash
#SBATCH -J hrp_haz_gori_parts
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
# NOTE: set the array range when submitting, e.g. --array=0-20

set -euo pipefail

# --- Load environment ---
module purge
# Try mambaforge, then miniforge3, then generic conda
if [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
  source "$HOME/mambaforge/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# Activate your env (adjust name as needed)
conda activate climada_env || true

# Repo root (keep your path)
REPO="/home/groups/bakerjw/smeiler/repos/hurricane_recovery_potential"
cd "$REPO"

# Ensure logs dir exists
mkdir -p logs

# Make your package modules importable without installing
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO"

# Prefer scratch for temp/intermediate writes
export TMPDIR="${SCRATCH:-/tmp}"
mkdir -p "$TMPDIR"

# Chunk size (events per chunk); override at submit time with:  sbatch --export=N=300 ...
N="${N:-250}"

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(command -v python)  $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-unset}"
echo "Chunk size (N): $N"

# --- Run (chunk-id defaults to SLURM_ARRAY_TASK_ID inside the script) ---
python -u scripts/make_haz_gori_chunks.py --chunk-size "$N"
