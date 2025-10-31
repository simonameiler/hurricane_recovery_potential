#!/usr/bin/env bash
# SBATCH script to run combine_aggregated_outputs.py on the cluster
# Adjust SBATCH directives to match your cluster.

#SBATCH --job-name=combine_aggr
#SBATCH --output=logs/combine_aggr_%A_%a.out
#SBATCH --error=logs/combine_aggr_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
set -x

# Activate conda env (same pattern as other batch scripts)
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate climada_env
else
  echo "Could not locate conda.sh via 'conda info --base'." >&2
  exit 1
fi

REPO_DIR="/home/users/smeiler/repos/hurricane_recovery_potential"
SCRIPT="$REPO_DIR/scripts/combine_aggregated_outputs.py"

cd "$REPO_DIR"
mkdir -p logs
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# scratch dir for temporary files
export TMPDIR="${SCRATCH:-/tmp}"
mkdir -p "$TMPDIR"

# Run the combine script. Optionally set BASE_OUT_DIR to point to the outputs on the cluster
# Example: BASE_OUT_DIR=/home/groups/bakerjw/smeiler/impacts_out
BASE_OUT_DIR="${BASE_OUT_DIR:-/home/groups/bakerjw/smeiler/climada_data/data/results/hrp_impacts_out}"

python3 "$SCRIPT" --base-out-dir "$BASE_OUT_DIR"
