#!/usr/bin/env bash
# Assemble dev/context.md from current project state.
# Budget: ~350 lines. Truncate lower-priority items if needed.
#
# Pass --with-bench to re-run the full corpus benchmark (~3 minutes).
# By default the benchmark section is sourced from the latest session
# checkpoint instead, making the refresh complete in seconds.
set -euo pipefail

OUT="dev/context.md"
BUDGET=350
WITH_BENCH=0
for arg in "$@"; do
    case "$arg" in
        --with-bench) WITH_BENCH=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

{
echo "# FERAL Context (auto-generated)"
echo ""
echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Latest session checkpoint
echo "## Latest Session"
LATEST_SESSION=$(ls -1 dev/sessions/[0-9]*.md 2>/dev/null | sort | tail -1 || true)
if [ -n "${LATEST_SESSION:-}" ]; then
    echo "File: $LATEST_SESSION"
    echo '```'
    head -50 "$LATEST_SESSION"
    echo '```'
else
    echo "No sessions yet."
fi
echo ""

# Git status summary
echo "## Git Status"
echo '```'
git log --oneline -5
echo '```'
echo ""

# Cargo test summary
echo "## Test Status"
echo '```'
cargo test 2>&1 | tail -20
echo '```'
echo ""

# Benchmark output
echo "## Benchmark"
echo '```'
if [ "$WITH_BENCH" -eq 1 ]; then
    cargo run --bin bench --release 2>&1 | grep -v "^   Compiling\|^   Downloading\|^    Finished\|^     Running\|^     Locking\|^ Downloading"
else
    # Source bench results from the latest session checkpoint to avoid
    # the ~3 minute corpus walk. Re-run with --with-bench for fresh numbers.
    if [ -n "${LATEST_SESSION:-}" ] && grep -q '^## Benchmark Results' "$LATEST_SESSION"; then
        echo "(skipped: pass --with-bench to re-run; sourced from $LATEST_SESSION)"
        echo ""
        awk '/^## Benchmark Results/{flag=1; next} /^## /{flag=0} flag' "$LATEST_SESSION" | grep -v '^```$'
    else
        echo "(skipped: pass --with-bench to re-run; no session checkpoint with bench)"
    fi
fi
echo '```'
echo ""

# Recent decisions
echo "## Recent Decisions"
if [ -s dev/decisions.md ]; then
    tail -30 dev/decisions.md
else
    echo "None yet."
fi
echo ""

# Recent tried-and-rejected
echo "## Recent Tried-and-Rejected"
if [ -s dev/tried-and-rejected.md ]; then
    tail -20 dev/tried-and-rejected.md
else
    echo "None yet."
fi
echo ""

# Source file listing
echo "## Source Files"
echo '```'
find src -name '*.rs' | sort
echo '```'
echo ""

# Test file listing
echo "## Test Files"
echo '```'
find tests -name '*.rs' 2>/dev/null | sort
echo '```'

} > "$OUT"

# Truncate to budget
LINE_COUNT=$(wc -l < "$OUT")
if [ "$LINE_COUNT" -gt "$BUDGET" ]; then
    head -"$BUDGET" "$OUT" > "${OUT}.tmp"
    echo "" >> "${OUT}.tmp"
    echo "(truncated from $LINE_COUNT lines to $BUDGET line budget)" >> "${OUT}.tmp"
    mv "${OUT}.tmp" "$OUT"
fi

echo "Wrote $OUT ($(wc -l < "$OUT") lines)"
