#!/usr/bin/env bash
#SBATCH -J hrp_haz_all_events
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

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

# --- repo + imports ---
REPO="$HOME/repos/hurricane_recovery_potential"
cd "$REPO"
mkdir -p logs
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# --- scratch for temps (faster I/O) ---
export TMPDIR="${SCRATCH:-/tmp}"
mkdir -p "$TMPDIR"

# Thread control for numpy/scipy/etc
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(command -v python)  $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Memory allocated: 256G"
echo "TMPDIR: $TMPDIR"

python -u scripts/make_haz_gori.py

