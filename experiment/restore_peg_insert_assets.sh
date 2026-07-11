#!/usr/bin/env bash
set -euo pipefail

# Restore the minimal Isaac 4.5 USD assets required by the manager-based
# peg-insert smoke test.
#
# This script downloads only public NVIDIA Isaac assets referenced by:
#   Isaac-Peg-Insert-Franka-IK-Rel-v0
#
# It intentionally stores assets under _runtime/persistent instead of Git.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/_runtime/remote_handoff_gripper_lstm_work}"
ASSET_ROOT="${ASSET_ROOT:-$RUNTIME_ROOT/persistent/assets/isaac_4_5_mirror}"
BASE_URL="${BASE_URL:-http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5}"

mkdir -p "$ASSET_ROOT/Isaac/IsaacLab/Factory"
mkdir -p "$ASSET_ROOT/Isaac/IsaacLab/Robots/FrankaEmika/Materials"
mkdir -p "$ASSET_ROOT/Isaac/IsaacLab/Robots/FrankaEmika/Props"
mkdir -p "$ASSET_ROOT/Isaac/Props/Mounts/SeattleLabTable"
mkdir -p "$ASSET_ROOT/Isaac/Props/Mounts/SeattleLabTable/Materials/Textures"

download_one() {
  local rel="$1"
  local out="$ASSET_ROOT/$rel"
  local url="$BASE_URL/$rel"
  if [[ -s "$out" && "${FORCE_DOWNLOAD:-0}" != "1" ]]; then
    echo "exists: $out"
    return 0
  fi
  echo "download: $url"
  curl -L --fail --retry 3 --retry-delay 2 --max-time 300 -o "$out" "$url"
  test -s "$out"
}

download_one "Isaac/IsaacLab/Factory/factory_peg_8mm.usd"
download_one "Isaac/IsaacLab/Factory/factory_hole_8mm.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Materials/Materials.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Props/instanceable_collision_meshes.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_hand.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_leftfinger.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link0.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link1.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link2.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link3.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link4.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link5.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link6.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link7.usd"
download_one "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_rightfinger.usd"
download_one "Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd"
download_one "Isaac/Props/Mounts/SeattleLabTable/table.usd"
download_one "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_BaseColor.png"
download_one "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_Metallic.png"
download_one "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_Normal.png"
download_one "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_Roughness.png"
download_one "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableParts_BaseColor.png"
download_one "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableParts_Metallic.png"
download_one "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableParts_Normal.png"
download_one "Isaac/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableParts_Roughness.png"

echo "Restored core peg-insert assets under: $ASSET_ROOT"
echo
echo "Next smoke command:"
echo "LOCAL_ISAAC_4_5_ASSET_ROOT=\"$ASSET_ROOT\" ENABLE_CAMERAS=0 TIMEOUT_SECONDS=420 NUM_STEPS=1 bash experiment/peg_insert_env_smoke.sh"
