#!/usr/bin/env bash
#SBATCH -J hrp_haz_gori
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G

set -euo pipefail

# --- Load environment ---
module purge
# If you use conda/mamba on Sherlock:
if [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
  source "$HOME/mambaforge/etc/profile.d/conda.sh"
fi
# Activate your env (adjust name as needed)
conda activate climada_env || true

# Repo root (adjust path)
REPO="$HOME/path/to/hurricane_recovery_potential"
cd "$REPO"

# Ensure logs dir exists
mkdir -p logs

# Make your package modules importable without installing
export PYTHONPATH="$PYTHONPATH:$REPO"

# Optional: prefer scratch for any temp/intermediate writes
export TMPDIR="${SCRATCH:-/tmp}"
mkdir -p "$TMPDIR"

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(command -v python)  $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"

# --- Run ---
python scripts/make_haz_gori.py
