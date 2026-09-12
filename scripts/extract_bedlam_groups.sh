#!/bin/bash
# Extract the 12 paper groups at 6 fps (frame stride 5) sequentially.
# Runs for hours (streams ~3 TB of tars from the NAS); log per group.
set -uo pipefail
cd "$(dirname "$0")/.."
GROUPS_12=(
  20221019_3-8_1000_highbmihand_static_suburb_d
  20221010_3-10_500_batch01hand_zoom_suburb_d
  20221018_1_250_batch01hand_zoom_suburb_b
  20221011_1_250_batch01hand_closeup_suburb_a
  20221011_1_250_batch01hand_closeup_suburb_c
  20221018_3-8_250_batch01hand_pitchDown52_stadium
  20221018_3-8_250_batch01hand_pitchUp52_stadium
  20221019_3-8_250_highbmihand_orbit_stadium
  20221013_3-10_500_batch01hand_static_highSchoolGym
  20221012_3-10_500_batch01hand_zoom_highSchoolGym
  20221010_3_1000_batch01hand
  20221024_3-10_100_batch01handhair_static_highSchoolGym
)
mkdir -p logs
for g in "${GROUPS_12[@]}"; do
  if [ -d "data/generated/bedlam_raw/$g/depth" ] && [ -f "outputs/logs/extract_$g.done" ]; then
    echo "skip $g (done)"; continue
  fi
  echo "$(date +%F_%T) start $g"
  uv run python scripts/extract_bedlam.py --group "$g" --frame-stride 5 \
      --modalities gt masks png depth > "outputs/logs/extract_$g.log" 2>&1 \
    && touch "outputs/logs/extract_$g.done" && echo "$(date +%F_%T) done $g" \
    || echo "$(date +%F_%T) FAILED $g (see outputs/logs/extract_$g.log)"
done
