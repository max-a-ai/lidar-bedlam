#!/usr/bin/env bash
# Pull the small result files of every Helma run into outputs/helma/:
# metrics.jsonl + DONE per run and the re-evaluation JSONs (outputs/eval/).
#
#   bash lidar_bedlam/scripts/pull_helma_results.sh
set -euo pipefail
HOST=${HELMA_HOST:-helma}
REMOTE=${HELMA_REPO:-/hnvme/workspace/v103fe17-lidar-bedlam}
DEST=${1:-outputs/helma}
mkdir -p "$DEST/runs" "$DEST/eval"
rsync -az -e "ssh -4" --include='*/' --include='metrics.jsonl' --include='DONE' --include='config.json' --exclude='*' \
  --prune-empty-dirs "$HOST:$REMOTE/outputs/" "$DEST/runs/"
rsync -az -e "ssh -4" "$HOST:$REMOTE/outputs/eval/" "$DEST/eval/" 2>/dev/null || true
echo "runs: $(ls "$DEST/runs" | wc -l), eval files: $(ls "$DEST/eval" 2>/dev/null | wc -l)"
