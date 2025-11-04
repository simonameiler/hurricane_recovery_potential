#!/usr/bin/env bash
# SBATCH script to run calc_state_impact.py per hazard chunk (array job: one task per chunk)
# Each task iterates over all states and computes impacts using that chunk's hazard.
# Output files are naturally separated by state and event_name (no conflicts across chunks).

#SBATCH --job-name=calc_impacts_per_chunk
#SBATCH --output=logs/calc_impacts_chunk_%A_%a.out
#SBATCH --error=logs/calc_impacts_chunk_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G

# Submit with: sbatch --array=0-N batch/sbatch_calc_impacts_per_haz_chunk.sh
# where N = number of chunk files - 1

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
HAZ_DIR="$DATA_DIR/hazard/tropical_cyclone/gori"
EXP_DIR="$DATA_DIR/exposure/states"
OUT_DIR="/home/groups/bakerjw/smeiler/climada_data/data/results/hrp_impacts_out"

cd "$REPO_DIR"
mkdir -p logs
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# --- scratch for temps (faster I/O) ---
export TMPDIR="${SCRATCH:-/tmp}"
mkdir -p "$TMPDIR"

# Discover chunk files (sorted by chunk number)
mapfile -t CHUNK_FILES < <(ls -1 ${HAZ_DIR}/tc_ncep_reanal_chunk*_N*.hdf5 2>/dev/null | sort -V || true)
if [ ${#CHUNK_FILES[@]} -eq 0 ]; then
  echo "No chunk HDF5 files found in ${HAZ_DIR}" >&2
  exit 1
fi

# Pick the chunk for this array task
TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ $TASK_ID -ge ${#CHUNK_FILES[@]} ]; then
  echo "SLURM_ARRAY_TASK_ID ($TASK_ID) out of range (0..$((${#CHUNK_FILES[@]} - 1)))" >&2
  exit 2
fi

CHUNK_FILE="${CHUNK_FILES[$TASK_ID]}"
CHUNK_BASENAME=$(basename "$CHUNK_FILE")

echo "================================================"
echo "Processing chunk: $CHUNK_BASENAME"
echo "Task ID: $TASK_ID"
echo "================================================"

# Discover all state exposure files
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

echo "Found ${#STATE_NAMES[@]} states to process with this chunk"

# Iterate over all states and compute impacts using this chunk's hazard
for st in "${STATE_NAMES[@]}"; do
  echo "  Processing state: $st with chunk $CHUNK_BASENAME"
  python3 "$SCRIPT" \
    --state "$st" \
    --data-dir "$DATA_DIR" \
    --haz-file "$CHUNK_BASENAME" \
    --out-dir "$OUT_DIR"
done

echo "Chunk $CHUNK_BASENAME completed for all states."
