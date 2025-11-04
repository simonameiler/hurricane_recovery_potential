#!/usr/bin/env bash
set -euo pipefail
set -x

# Submit calc_state_impacts as three array jobs with different memory sizes per state group.
# Groups chosen by exposure file size bins (user-provided):
#  - group A (keep 32G)
#  - group B (increase to 48G)
#  - group C (increase to 96G)

REPO="$HOME/repos/hurricane_recovery_potential"
DATA_DIR="/home/groups/bakerjw/smeiler/climada_data/data"
EXP_DIR="$DATA_DIR/exposure/states"
SBATCH_SCRIPT="$REPO/batch/sbatch_calc_state_impacts_array.sh"

if [ ! -d "$EXP_DIR" ]; then
  echo "Exposure directory not found: $EXP_DIR" >&2
  exit 1
fi

cd "$REPO"

# build ordered list of state stems (matching how the sbatch script builds STATE_NAMES)
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

# helper: normalize names for matching
norm() { echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d ' _-'; }

# user-provided groups (use normalized stems)
GROUP_A=(alabama connecticut delaware louisiana maine maryland massachusetts mississippi newhampshire newjersey rhodeisland southcarolina virginia)
GROUP_B=(georgia newyork northcarolina pennsylvania)
GROUP_C=(florida texas)

make_index_list() {
  # Accepts state names as arguments (e.g. make_index_list "alabama" "florida")
  local idx_list=()
  for want in "$@"; do
    wn=$(norm "$want")
    found=0
    for i in "${!STATE_NAMES[@]}"; do
      sn=$(norm "${STATE_NAMES[$i]}")
      if [ "$sn" = "$wn" ]; then
        idx_list+=("$i")
        found=1
        break
      fi
    done
    if [ $found -eq 0 ]; then
      echo "Warning: state '$want' not found in exposure list; skipping" >&2
    fi
  done
  # join with comma
  IFS=','; echo "${idx_list[*]}"; unset IFS
}

submit_group() {
  local idxs=$1
  local mem=$2
  local name=$3
  if [ -z "$idxs" ]; then
    echo "No indices for group $name; skipping"
    return
  fi
  echo "Submitting group $name indices=$idxs mem=$mem"
  sbatch --array=${idxs} --mem=${mem}G --job-name=calc_${name} "$SBATCH_SCRIPT"
}

IDX_A=$(make_index_list "${GROUP_A[@]}")
IDX_B=$(make_index_list "${GROUP_B[@]}")
IDX_C=$(make_index_list "${GROUP_C[@]}")

# submit with chosen memory sizes (GB)
submit_group "$IDX_A" 32 "groupA"
submit_group "$IDX_B" 48 "groupB"
submit_group "$IDX_C" 96 "groupC"

echo "Submitted grouped array jobs."
