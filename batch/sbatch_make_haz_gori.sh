#!/usr/bin/env bash
#SBATCH -J hrp_haz_subset
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --hint=nomultithread

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

# Optional: let you control subset size via env var SUBSET_N (if your script supports it)
# export SUBSET_N="${SUBSET_N:-100}"

which python
python --version
python -u scripts/make_haz_gori.py

