#!/usr/bin/env bash
# Regenerate the issue #65 ill-conditioned-KKT fixtures.
#
# sawpath / twirism1 iteration-0 IPM KKTs are generated matrices (not
# SuiteSparse downloads), so they cannot go through fetch_large_matrices.sh.
# They are the regression fixtures for the inertia-guided MC64 scaling
# fallback: under default `Auto` scaling FERAL mis-factors sawpath's KKT
# (wrong, rank-deficient inertia with ~116 spurious zero pivots); the
# fallback re-runs with Mc64Symmetric and recovers the true (789,786,0).
#
# Produces tests/data/large/{sawpath,twirism1}_kkt.mtx (gitignored). The
# regression test `tests/issue65_mc64_fallback.rs` skips cleanly when they
# are absent, so running this is optional and local-only.
#
# Requires a built pounce with FERAL and the vanderbei .nl set. Override
# paths via the env vars below.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${REPO_ROOT}/tests/data/large"

POUNCE_BIN="${POUNCE_BIN:-${HOME}/projects/pounce/target/release/pounce}"
POUNCE_REPO="${POUNCE_REPO:-${HOME}/projects/pounce}"
NL_DIR="${NL_DIR:-${HOME}/Dropbox/projects/pounce-bench-data/vanderbei/nl}"
CONVERTER="${CONVERTER:-${POUNCE_REPO}/benchmarks/mittelmann/feral_repro/jsonl_to_mtx.py}"

for f in "${POUNCE_BIN}" "${CONVERTER}"; do
    if [[ ! -f "${f}" ]]; then
        echo "error: required input not found: ${f}" >&2
        echo "       set POUNCE_BIN / CONVERTER to override." >&2
        exit 1
    fi
done

mkdir -p "${DEST}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

for prob in sawpath twirism1; do
    nl="${NL_DIR}/${prob}.nl"
    if [[ ! -f "${nl}" ]]; then
        echo "error: ${nl} not found (set NL_DIR)" >&2
        exit 1
    fi
    echo "dumping ${prob} iter-0 KKT via pounce..."
    # pounce exits non-zero on max_iter=1; the iter-0 dump is what we need.
    "${POUNCE_BIN}" "${nl}" --no-sol max_iter=1 \
        --dump kkt:0 --dump-dir "${TMP}/${prob}" >/dev/null 2>&1 || true
    jsonl="${TMP}/${prob}/iter_000/kkt_solve_001.jsonl"
    if [[ ! -f "${jsonl}" ]]; then
        echo "error: pounce did not produce ${jsonl}" >&2
        exit 1
    fi
    python3 "${CONVERTER}" "${jsonl}" "${TMP}/${prob}" >/dev/null
    mv "${TMP}/${prob}_kkt.mtx" "${DEST}/${prob}_kkt.mtx"
    echo "wrote ${DEST}/${prob}_kkt.mtx ($(wc -c <"${DEST}/${prob}_kkt.mtx") bytes)"
done
