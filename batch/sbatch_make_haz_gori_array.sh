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
set -x

# --- activate env ---
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate climada_env
else
  echo "Could not locate conda.sh via 'conda info --base'." >&2
  exit 1
fi

# Activate your env (adjust name as needed)
conda activate climada_env || true

# Repo root (keep your path)
REPO="$HOME/repos/hurricane_recovery_potential"
mkdir -p logs
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# --- scratch for temps (faster I/O) ---
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
