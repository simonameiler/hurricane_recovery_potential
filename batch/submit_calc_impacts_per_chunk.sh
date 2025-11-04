#!/usr/bin/env bash
# Helper script to submit the per-chunk impact calculation array job.
# Discovers chunk HDF5 files and submits sbatch with the correct array range.

set -euo pipefail

REPO="$HOME/repos/hurricane_recovery_potential"
HAZ_DIR="/home/groups/bakerjw/smeiler/climada_data/data/hazard/tropical_cyclone/gori"
SBATCH_SCRIPT="$REPO/batch/sbatch_calc_impacts_per_haz_chunk.sh"

if [ ! -d "$HAZ_DIR" ]; then
  echo "Hazard directory not found: $HAZ_DIR" >&2
  exit 1
fi

cd "$REPO"

# Count chunk files
CHUNK_COUNT=$(ls -1 ${HAZ_DIR}/tc_ncep_reanal_chunk*_N*.hdf5 2>/dev/null | wc -l)

if [ "$CHUNK_COUNT" -eq 0 ]; then
  echo "No chunk HDF5 files found in ${HAZ_DIR}" >&2
  exit 1
fi

LAST=$((CHUNK_COUNT - 1))

echo "Found $CHUNK_COUNT chunk files in $HAZ_DIR"
echo "Submitting array job with range 0-${LAST}"

sbatch --array=0-${LAST} "$SBATCH_SCRIPT"

echo "Submitted. Check job status with: squeue -u \$USER"
