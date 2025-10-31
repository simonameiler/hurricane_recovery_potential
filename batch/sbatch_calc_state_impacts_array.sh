#!/usr/bin/env bash
# SBATCH script to run calc_state_impact.py for each state as an array job
# Adjust SBATCH directives below to match your cluster's configuration.

#SBATCH --job-name=calc_state_impacts
#SBATCH --output=logs/calc_state_impacts_%A_%a.out
#SBATCH --error=logs/calc_state_impacts_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

# Optional: uncomment and set partition/queue
##SBATCH --partition=short

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

REPO_DIR="/home/users/smeiler/repos/hurricane_recovery_potential"
SCRIPT="$REPO_DIR/scripts/calc_state_impact.py"
DATA_DIR="/home/groups/bakerjw/smeiler/climada_data/data"
EXP_DIR="$DATA_DIR/exposure/states"

cd "$REPO_DIR"
mkdir -p logs
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# --- scratch for temps (faster I/O) ---
export TMPDIR="${SCRATCH:-/tmp}"
mkdir -p "$TMPDIR"

# build an array of state names from filenames
mapfile -t STATE_FILES < <(ls -1 ${EXP_DIR}/*_exposure.hdf5 2>/dev/null || true)
if [ ${#STATE_FILES[@]} -eq 0 ]; then
  echo "No exposure files found in ${EXP_DIR}" >&2
  exit 1
fi

STATE_NAMES=()
for f in "${STATE_FILES[@]}"; do
  fname=$(basename "$f")
  st=${fname%%_exposure.hdf5}
  STATE_NAMES+=("$st")
done

# If not submitted as an array job, run sequentially
if [ -z "${SLURM_ARRAY_TASK_ID-}" ]; then
  echo "Running sequentially for all states"
  for st in "${STATE_NAMES[@]}"; do
    echo "Processing state: $st"
    python3 "$SCRIPT" --state "$st" --data-dir "$DATA_DIR"
  done
  exit 0
fi

# Array task: pick the state based on SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ $TASK_ID -ge ${#STATE_NAMES[@]} ]; then
  echo "SLURM_ARRAY_TASK_ID ($TASK_ID) out of range (0..$((${#STATE_NAMES[@]} - 1)))" >&2
  exit 2
fi

STATE=${STATE_NAMES[$TASK_ID]}

echo "SLURM task $TASK_ID processing state: $STATE"
python3 "$SCRIPT" --state "$STATE" --data-dir "$DATA_DIR"
