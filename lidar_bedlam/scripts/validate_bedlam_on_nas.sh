#!/bin/bash
# Run ON THE NAS (not over sshfs) from the bedlam directory:
#   cd /volume1/publicdatasets/bedlam && bash validate_bedlam_on_nas.sh
# Requires xxhsum >= 0.8.1. Extracts the per-modality checksum lists from
# be_imagedata_download.zip and validates each modality in turn.
set -euo pipefail
unzip -o -q be_imagedata_download.zip 'b0_checksums_*.xxh128'
for m in gt masks mp4 png depth; do
    echo "== validating $m"
    xxhsum -c "b0_checksums_${m}.xxh128" | tee "validate_${m}.log" | grep -v ': OK$' || true
done
echo "done; see validate_*.log"
