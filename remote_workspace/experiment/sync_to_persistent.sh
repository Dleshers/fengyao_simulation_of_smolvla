#!/usr/bin/env bash
set -euo pipefail

SOURCE="${SOURCE:-/scratch0/fenzhang/projects/remote_handoff_gripper_lstm_work}"
DEST="${DEST:-/cs/student/project_msc/2025/rai/fenzhang/simulation_storage/remote_handoff_gripper_lstm_workspace}"

mkdir -p "$DEST"
rsync -a --info=progress2 \
  --exclude=.conda/ \
  --exclude=.venv/ \
  --exclude=.cache/ \
  --exclude=tmp/ \
  --exclude=external/downloads/ \
  "$SOURCE/" "$DEST/"

echo "Persistent workspace synchronized: $DEST"
