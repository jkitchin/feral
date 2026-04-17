#!/usr/bin/env bash
# Clean-room invariant for feral-amd: the external `amd` crate
# (SuiteSparse AMD binding) appears ONLY in the oracle harness
# source under `crates/feral-amd/tests/data/amd_oracle/harness/`.
# It must never enter the feral workspace's dependency graph.
#
# Exit 0 if clean, 1 if violations found.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

violations=()

# 1) No `amd = "..."` or `amd = { ... }` in any Cargo.toml (including
#    the root workspace toml and every crate member).
while IFS= read -r -d '' manifest; do
    if grep -E '^[[:space:]]*amd[[:space:]]*=' "$manifest" >/dev/null; then
        violations+=("$manifest declares an 'amd' dependency")
    fi
done < <(find "$repo_root" -name Cargo.toml \
    -not -path '*/target/*' \
    -not -path '*/ref/*' \
    -not -path '*/tests/data/amd_oracle/harness/*' \
    -print0)

# 2) No `use amd::` or `extern crate amd;` in source files inside
#    feral-amd or the feral package's src/.
while IFS= read -r -d '' rs; do
    if grep -E '^(use[[:space:]]+amd(::|[[:space:]]*;)|extern[[:space:]]+crate[[:space:]]+amd)' "$rs" >/dev/null; then
        violations+=("$rs imports the 'amd' crate")
    fi
done < <(find "$repo_root/crates/feral-amd/src" "$repo_root/src" \
    -type f -name '*.rs' -print0 2>/dev/null)

# 3) Cargo.lock must not mention `name = "amd"` at the workspace root.
if [ -f "$repo_root/Cargo.lock" ]; then
    if grep -E '^name[[:space:]]*=[[:space:]]*"amd"$' "$repo_root/Cargo.lock" >/dev/null; then
        violations+=("Cargo.lock contains 'amd' package — it leaked into the workspace")
    fi
fi

if [ ${#violations[@]} -eq 0 ]; then
    echo "clean-room OK: 'amd' crate absent from feral workspace"
    exit 0
else
    echo "clean-room VIOLATION:"
    for v in "${violations[@]}"; do
        echo "  - $v"
    done
    echo
    echo "The 'amd' crate must live only in the oracle harness under"
    echo "crates/feral-amd/tests/data/amd_oracle/harness/ (not compiled)."
    exit 1
fi
