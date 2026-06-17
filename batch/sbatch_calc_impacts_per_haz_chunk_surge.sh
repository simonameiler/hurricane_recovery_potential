#!/usr/bin/env bash
# Per-hazard-chunk impact calculation WITH dual scaling (surge-OFF + surge-ON).
# Reproduces the committed pipeline (chunked hazard) and, in the same impact
# computation, also exports the surge-ON results -- so off and on are apples-to-apples.
#
# Submit over all chunk files:
#     N=$(ls /home/groups/bakerjw/smeiler/climada_data/data/hazard/tropical_cyclone/gori/tc_ncep_reanal_chunk*_N*.hdf5 | wc -l)
#     sbatch --array=0-$((N-1)) batch/sbatch_calc_impacts_per_haz_chunk_surge.sh

#SBATCH --job-name=calc_impacts_chunk_surge
#SBATCH --output=logs/calc_impacts_chunk_surge_%A_%a.out
#SBATCH --error=logs/calc_impacts_chunk_surge_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G

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

# scaling matrices (in the repo data dir)
SCALING_OFF="$REPO_DIR/data/scaling_relative.npz"           # committed (surge off)
SCALING_ON="$REPO_DIR/data/scaling_relative_SURGE_ON.npz"   # surge on (upload this)
COUNTY_REGION="$REPO_DIR/data/county_region.csv"

# output dirs (kept separate from the committed hrp_impacts_out)
OUT_OFF="$DATA_DIR/results/hrp_impacts_out_recheck"         # off re-run (must reproduce committed)
OUT_ON="$DATA_DIR/results/hrp_impacts_out_SURGE_ON"         # surge on

cd "$REPO_DIR"
mkdir -p logs
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${SCRATCH:-/tmp}"
mkdir -p "$TMPDIR"

# Discover chunk files (sorted by chunk number)
mapfile -t CHUNK_FILES < <(ls -1 ${HAZ_DIR}/tc_ncep_reanal_chunk*_N*.hdf5 2>/dev/null | sort -V || true)
if [ ${#CHUNK_FILES[@]} -eq 0 ]; then
  echo "No chunk HDF5 files found in ${HAZ_DIR}" >&2
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ "$TASK_ID" -ge ${#CHUNK_FILES[@]} ]; then
  echo "SLURM_ARRAY_TASK_ID ($TASK_ID) out of range (0..$((${#CHUNK_FILES[@]} - 1)))" >&2
  exit 2
fi

CHUNK_FILE="${CHUNK_FILES[$TASK_ID]}"
CHUNK_BASENAME=$(basename "$CHUNK_FILE")
echo "Processing chunk: $CHUNK_BASENAME (task $TASK_ID)"

# Discover all state exposure files
mapfile -t STATE_FILES < <(ls -1 ${EXP_DIR}/*_exposure.hdf5 2>/dev/null || true)
if [ ${#STATE_FILES[@]} -eq 0 ]; then
  echo "No exposure files found in ${EXP_DIR}" >&2
  exit 1
fi
STATE_NAMES=()
for f in "${STATE_FILES[@]}"; do
  fname=$(basename "$f")
  STATE_NAMES+=("${fname%%_exposure.hdf5}")
done

# Iterate over all states; compute impact ONCE per state and export both scalings
for st in "${STATE_NAMES[@]}"; do
  echo "  state: $st  chunk: $CHUNK_BASENAME"
  python3 "$SCRIPT" \
    --state "$st" \
    --data-dir "$DATA_DIR" \
    --haz-file "$CHUNK_BASENAME" \
    --scaling-npz    "$SCALING_OFF" --out-dir    "$OUT_OFF" \
    --scaling-npz-on "$SCALING_ON"  --out-dir-on "$OUT_ON" \
    --county-region  "$COUNTY_REGION"
done

echo "Chunk $CHUNK_BASENAME completed for all states (off + on)."
