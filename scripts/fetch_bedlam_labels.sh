#!/bin/bash
# Download the BEDLAM training labels (per-image SMPL / SMPL-X parameters in
# the camera frame) from the BEDLAM project server. These are NOT on the
# Hugging Face mirror, which only holds the image data (png/mp4/masks/gt).
#
# Requires a BEDLAM website account (https://bedlam.is.tue.mpg.de/).
# Credentials are asked for interactively and never stored.
#
# Usage: bash scripts/fetch_bedlam_labels.sh [smpl|smplx|all]   (default all)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/generated/bedlam_labels
mkdir -p "$OUT"
what=${1:-all}

urle() { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

read -r -p "BEDLAM website email: " email
read -r -s -p "BEDLAM website password: " password; echo
username=$(urle "$email"); password=$(urle "$password")

fetch() {  # fetch <sfile> <target>
    wget --post-data "username=$username&password=$password" \
        "https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=$1" \
        -O "$2" --no-check-certificate --continue
}

if [[ "$what" == smpl || "$what" == all ]]; then
    # SMPL (not -X) labels, as used by CameraHMR: one npz per sequence group
    fetch bedlam-labels-smpl.zip "$OUT/bedlam-labels-smpl.zip"
    unzip -o -q "$OUT/bedlam-labels-smpl.zip" -d "$OUT/smpl"
fi
if [[ "$what" == smplx || "$what" == all ]]; then
    # SMPL-X labels, as used by the official BEDLAM training code
    fetch bedlam_labels/all_npz_12_training.zip "$OUT/all_npz_12_training.zip"
    unzip -o -q "$OUT/all_npz_12_training.zip" -d "$OUT/smplx"
fi
unset password
echo "done:"; find "$OUT" -name '*.npz' | head; find "$OUT" -name '*.npz' | wc -l
