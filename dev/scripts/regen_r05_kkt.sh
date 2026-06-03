#!/usr/bin/env bash
# Regenerate the issue #64 arrow/bordered-KKT fixture.
#
# r05's iteration-0 IPM KKT is a *generated* matrix (not a SuiteSparse
# download), so it cannot go through fetch_large_matrices.sh. It is the
# regression fixture for the arrow-ordering catch: nested dissection
# blows the LDLᵀ factor up ~7-9× vs AMF/AMD on it.
#
# Produces tests/data/large/r05_kkt.mtx (gitignored). The regression
# test `tests/issue64_arrow_ordering.rs` skips cleanly when it is absent,
# so running this is optional and local-only.
#
# Requires a built pounce with FERAL and the r05.nl input. Override the
# paths via the env vars below if your checkout differs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${REPO_ROOT}/tests/data/large/r05_kkt.mtx"

POUNCE_BIN="${POUNCE_BIN:-${HOME}/projects/pounce/target/release/pounce}"
POUNCE_REPO="${POUNCE_REPO:-${HOME}/projects/pounce}"
R05_NL="${R05_NL:-${HOME}/Dropbox/projects/pounce-bench-data/lp/nl/r05.nl}"
CONVERTER="${CONVERTER:-${POUNCE_REPO}/benchmarks/mittelmann/feral_repro/jsonl_to_mtx.py}"

for f in "${POUNCE_BIN}" "${R05_NL}" "${CONVERTER}"; do
    if [[ ! -f "${f}" ]]; then
        echo "error: required input not found: ${f}" >&2
        echo "       set POUNCE_BIN / R05_NL / CONVERTER to override." >&2
        exit 1
    fi
done

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "dumping r05 iter-0 KKT via pounce..."
# pounce exits non-zero on "Maximum Number of Iterations Exceeded"
# (expected with max_iter=1); the iter-0 dump is what we need, so the
# solve status is ignored. The jsonl-existence check below is the real
# success gate.
"${POUNCE_BIN}" "${R05_NL}" --dump kkt:0 --dump-dir "${TMP}/dump" max_iter=1 >/dev/null || true

JSONL="${TMP}/dump/iter_000/kkt_solve_001.jsonl"
if [[ ! -f "${JSONL}" ]]; then
    echo "error: pounce did not produce ${JSONL}" >&2
    exit 1
fi

echo "converting jsonl -> MatrixMarket..."
python3 "${CONVERTER}" "${JSONL}" "${TMP}/r05"

mkdir -p "$(dirname "${DEST}")"
mv "${TMP}/r05_kkt.mtx" "${DEST}"
echo "wrote ${DEST} ($(wc -c <"${DEST}") bytes)"
echo "header:"
head -4 "${DEST}"
