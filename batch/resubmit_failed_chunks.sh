#!/usr/bin/env bash
# Helper script to detect failed/partial chunk jobs and resubmit them.
# Checks output logs for success markers and builds a list of failed chunk IDs to resubmit.

set -euo pipefail

REPO="$HOME/repos/hurricane_recovery_potential"
SBATCH_SCRIPT="$REPO/batch/sbatch_calc_impacts_per_haz_chunk.sh"
LOG_DIR="$REPO/logs"

cd "$REPO"

if [ ! -d "$LOG_DIR" ]; then
  echo "Log directory not found: $LOG_DIR" >&2
  exit 1
fi

# Collect failed chunk IDs
failed_chunks=()

for log in "$LOG_DIR"/calc_impacts_chunk_*_*.out; do
  [ -e "$log" ] || continue  # skip if glob doesn't match
  
  # Extract task ID from filename (calc_impacts_chunk_JOBID_TASKID.out)
  basename=$(basename "$log")
  if [[ "$basename" =~ calc_impacts_chunk_[0-9]+_([0-9]+)\.out ]]; then
    task_id="${BASH_REMATCH[1]}"
  else
    continue
  fi
  
  # Check if this chunk completed successfully
  if grep -q "completed for all states" "$log" 2>/dev/null; then
    : # success, skip
  else
    # Either partial, error, or disk full
    failed_chunks+=("$task_id")
  fi
done

if [ ${#failed_chunks[@]} -eq 0 ]; then
  echo "No failed chunks detected. All chunks completed successfully!"
  exit 0
fi

# Sort failed chunks numerically
IFS=$'\n' sorted_failed=($(sort -n <<<"${failed_chunks[*]}"))
unset IFS

echo "Detected ${#sorted_failed[@]} failed/partial chunks:"
printf '  %s\n' "${sorted_failed[@]}"

# Build comma-separated list for sbatch --array
array_spec=$(IFS=,; echo "${sorted_failed[*]}")

echo ""
echo "Resubmitting failed chunks with: sbatch --array=${array_spec}"
echo ""

# Resubmit
sbatch --array="${array_spec}" "$SBATCH_SCRIPT"

echo "Resubmitted. Check status with: squeue -u \$USER"
