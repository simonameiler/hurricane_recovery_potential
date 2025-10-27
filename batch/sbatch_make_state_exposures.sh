#!/bin/bash
#SBATCH --job-name=state_exp
#SBATCH --output=logs/make_state_exposures_%j.log
#SBATCH --error=logs/make_state_exposures_%j.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

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

which python
python --version
python -u scripts/make_state_exposures.py