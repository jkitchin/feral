#!/usr/bin/env bash
# Collect KKT matrices from ripopt CUTEst runs for FERAL benchmarking.
#
# Usage:
#   ./scripts/collect-kkt.sh [CUTEST_MAX_N]
#
# Requires: ../ripopt with CUTEst problems prepared.

set -euo pipefail

FERAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RIPOPT_DIR="$FERAL_DIR/../ripopt"
OUTPUT_DIR="$FERAL_DIR/data/matrices/kkt"
MAX_N="${1:-500}"

if [ ! -d "$RIPOPT_DIR" ]; then
    echo "Error: ripopt not found at $RIPOPT_DIR"
    exit 1
fi

if [ ! -f "$RIPOPT_DIR/cutest_suite/problem_list.txt" ]; then
    echo "Error: CUTEst problems not prepared (run prepare.sh in ripopt/cutest_suite first)"
    exit 1
fi

echo "Collecting KKT matrices (CUTEST_MAX_N=$MAX_N) → $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$RIPOPT_DIR"
CUTEST_MAX_N="$MAX_N" cargo run --bin collect_kkt --release --features cutest -- \
    --output "$OUTPUT_DIR"

N_MATRICES=$(find "$OUTPUT_DIR" -name '*.mtx' | wc -l | tr -d ' ')
echo ""
echo "Done: $N_MATRICES matrices collected in $OUTPUT_DIR"
