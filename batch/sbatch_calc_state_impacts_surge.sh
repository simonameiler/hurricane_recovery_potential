#!/usr/bin/env bash
# Run calc_state_impact.py for each state, producing BOTH the committed surge-OFF
# results and the surge-ON results in a single impact computation per state.
# Submit as an array over the 19 states:
#     sbatch --array=0-18 batch/sbatch_calc_state_impacts_surge.sh

#SBATCH --job-name=calc_impacts_surge
#SBATCH --output=logs/calc_impacts_surge_%A_%a.out
#SBATCH --error=logs/calc_impacts_surge_%A_%a.err
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
EXP_DIR="$DATA_DIR/exposure/states"

# scaling matrices (in the repo data dir)
SCALING_OFF="$REPO_DIR/data/scaling_relative.npz"            # committed (surge off)
SCALING_ON="$REPO_DIR/data/scaling_relative_SURGE_ON.npz"    # surge on (upload this)
COUNTY_REGION="$REPO_DIR/data/county_region.csv"

# output dirs
OUT_OFF="$DATA_DIR/results/hrp_impacts_out_recheck"          # surge-off re-run (validation)
OUT_ON="$DATA_DIR/results/hrp_impacts_out_SURGE_ON"          # surge-on

cd "$REPO_DIR"
mkdir -p logs
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${SCRATCH:-/tmp}"
mkdir -p "$TMPDIR"

# build state-name array from exposure filenames
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

run_one () {
  local st="$1"
  echo "Processing state: $st"
  python3 "$SCRIPT" \
    --state "$st" \
    --data-dir "$DATA_DIR" \
    --scaling-npz   "$SCALING_OFF" \
    --out-dir       "$OUT_OFF" \
    --scaling-npz-on "$SCALING_ON" \
    --out-dir-on     "$OUT_ON" \
    --county-region "$COUNTY_REGION"
}

# If not an array job, run all states sequentially
if [ -z "${SLURM_ARRAY_TASK_ID-}" ]; then
  echo "No SLURM_ARRAY_TASK_ID: running all ${#STATE_NAMES[@]} states sequentially"
  for st in "${STATE_NAMES[@]}"; do run_one "$st"; done
  exit 0
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ "$TASK_ID" -ge ${#STATE_NAMES[@]} ]; then
  echo "SLURM_ARRAY_TASK_ID ($TASK_ID) out of range (0..$((${#STATE_NAMES[@]} - 1)))" >&2
  exit 2
fi
run_one "${STATE_NAMES[$TASK_ID]}"
