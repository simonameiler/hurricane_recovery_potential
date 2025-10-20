#!/usr/bin/env bash
#SBATCH -J hrp_haz_gori_parts
#SBATCH -o /home/users/smeiler/repos/hurricane_recovery_potential/logs/%x-%A_%a.out
#SBATCH -e /home/users/smeiler/repos/hurricane_recovery_potential/logs/%x-%A_%a.err
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
# submit with: sbatch --array=0-20 --export=N=250 batch/sbatch_make_haz_gori_array.sh

set -euo pipefail
set -x

# optional: clean modules
module --force purge || true

# --- activate env ---
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate climada_env
else
  echo "Could not locate conda.sh via 'conda info --base'." >&2
  exit 1
fi

# Repo root
REPO="$HOME/repos/hurricane_recovery_potential"
cd "$REPO"
mkdir -p "$REPO/logs"

# Make repo importable
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Prefer scratch for temps (faster I/O)
export TMPDIR="${SCRATCH:-/tmp}"
mkdir -p "$TMPDIR"

# Keep threaded libs in check (since we use -c 4)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Chunk size (override with: sbatch --export=N=300 ...)
N="${N:-250}"

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(command -v python)  $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-unset}"
echo "Chunk size (N): $N"

# Run (chunk-id defaults to SLURM_ARRAY_TASK_ID inside the script)
python -u scripts/make_haz_gori_chunks.py --chunk-size "$N"
