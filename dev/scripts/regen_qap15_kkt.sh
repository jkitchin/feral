#!/usr/bin/env bash
# Regenerate the issue #91 preprocessing-misfire fixture.
#
# qap15's conic-IPM iteration-0 KKT is a *generated* matrix (pounce-convex
# on qap15.mps), not a SuiteSparse download. It is the regression fixture
# for the `OrderingPreprocess::Auto` fill-verified race: the structural
# predicate mistakenly recommends LdltCompress on this quasi-definite KKT,
# inflating fill ~6x (7.16M -> 45.4M) and factor time ~20x (0.8s -> 15s).
#
# Produces tests/data/large/qap15_kkt.mtx (gitignored). The regression
# test `tests/issue91_preprocess_misfire.rs` skips cleanly when it is
# absent, so running this is optional and local-only.
#
# Requires the `pounce` Python package (linsol-dump-capable build),
# `highspy`, `scipy`, and qap15.mps.bz2. Override paths via env vars.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${REPO_ROOT}/tests/data/large/qap15_kkt.mtx"

PYTHON="${PYTHON:-python3}"
GEN="${REPO_ROOT}/dev/scripts/gen_qap15_kkt.py"
QAP15_MPS="${QAP15_MPS:-${HOME}/projects/pounce/benchmarks/lpopt/mps/qap15.mps.bz2}"

if [[ ! -f "${QAP15_MPS}" ]]; then
  echo "ERROR: qap15 MPS not found at ${QAP15_MPS} (set QAP15_MPS)." >&2
  exit 1
fi

QAP15_MPS="${QAP15_MPS}" OUT="${DEST}" "${PYTHON}" "${GEN}"
echo "Regenerated ${DEST}"
