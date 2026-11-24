#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== Starting Baloo Unified 4-Controller Weekend Benchmark ==="
echo "Timestamp: $(date)"
echo "Host: $(hostname)"
echo "PID: $$"

# Use virtual environment python if present, otherwise fall back to system python
if [ -f "$DIR/.venv/bin/python" ]; then
    PY_BIN="$DIR/.venv/bin/python"
else
    PY_BIN="python"
fi

OUTPUT_DIR="${DIR}/../evaluation_results/unified_weekend"
mkdir -p "$OUTPUT_DIR"

exec "$PY_BIN" "$DIR/evaluate_unified.py" \
    --controller all \
    --condition both \
    --num_trials 1000 \
    --num_detail 5 \
    --output_dir "$OUTPUT_DIR"
