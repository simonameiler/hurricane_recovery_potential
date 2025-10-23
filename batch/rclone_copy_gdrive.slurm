#!/bin/bash
#SBATCH --job-name=rclone_gdrive_copy
#SBATCH --output=rclone_copy_%j.log
#SBATCH --error=rclone_copy_%j.err
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# --- user inputs (edit these two lines) ---
FOLDER_ID="1vtOaAZRkjeX5QNURaLN_ai9mcx165-iv"
DEST_DIR="/home/groups/bakerjw/smeiler/climada_data/data/exposure"
REMOTE_NAME="gdrive_simona"   # the name you created with `rclone config`
# ------------------------------------------

set -euo pipefail

echo "[$(date)] Starting rclone copy"
echo "Folder ID: ${FOLDER_ID}"
echo "Destination: ${DEST_DIR}"
echo "Remote: ${REMOTE_NAME}"
echo

# Ensure config exists (must have been copied from your laptop already)
if [[ ! -f "$HOME/.config/rclone/rclone.conf" ]]; then
  echo "ERROR: $HOME/.config/rclone/rclone.conf not found."
  echo "Copy it from your laptop:  scp ~/.config/rclone/rclone.conf yourSUNetID@login.sherlock.stanford.edu:~/.config/rclone/"
  exit 1
fi

# Prefer user-installed rclone; adjust if you installed elsewhere
RCLONE="$HOME/bin/rclone"
if ! command -v "$RCLONE" >/dev/null 2>&1; then
  # fallback to whatever is in PATH
  RCLONE="$(command -v rclone || true)"
fi
if [[ -z "$RCLONE" ]]; then
  echo "ERROR: rclone not found. Put it in ~/bin or load a module."
  exit 1
fi
echo "Using rclone at: $RCLONE"
echo

mkdir -p "$DEST_DIR"

# Copy from Drive (by ID). Curly braces are required for IDs.
# Tweak transfers/checkers if network is slower/faster.
"$RCLONE" copy "${REMOTE_NAME}:{${FOLDER_ID}}" "$DEST_DIR" \
  --progress \
  --transfers=8 \
  --checkers=8 \
  --drive-chunk-size=128M \
  --contimeout=30s \
  --timeout=5m \
  --retries=5 \
  --low-level-retries=10 \
  --retries-sleep=20s

echo
echo "[$(date)] Copy completed. Starting integrity check..."
"$RCLONE" check "${REMOTE_NAME}:{${FOLDER_ID}}" "$DEST_DIR" \
  --one-way \
  --checkers=8 \
  --size-only=false \
  --download

echo "[$(date)] All done."
