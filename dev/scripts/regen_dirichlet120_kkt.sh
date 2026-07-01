#!/usr/bin/env bash
# Regenerate the issue #102 regression fixture: dirichlet120's conic iter-0 KKT.
#
# This is the POUNCE mittelmann problem whose parallel dense-front factorization
# deadlocked at 0 % CPU under PR #92's ordering (issue #102). The KKT triplets
# are feral-version-independent (feral only *factors* the matrix), so any POUNCE
# build that reaches the first factorization can dump it — feral 0.11.3 solves
# it fine and dumps the same matrix feral `main` deadlocks on.
#
# Produces tests/data/large/dirichlet120_kkt.mtx (gitignored). The regression
# test `tests/issue102_intrafront_deadlock.rs` skips cleanly when absent.
#
# Requires a built `pounce` binary and dirichlet120.nl. Override via env vars.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${REPO_ROOT}/tests/data/large/dirichlet120_kkt.mtx"

POUNCE_BIN="${POUNCE_BIN:-${HOME}/projects/pounce/target/release/pounce}"
NL="${NL:-${HOME}/projects/pounce/benchmarks/mittelmann/nl/dirichlet120.nl}"
PYTHON="${PYTHON:-python3}"
BIN_DUMP="$(mktemp -t dirichlet120_kkt.XXXX.bin)"

for f in "${POUNCE_BIN}" "${NL}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

# max_iter=0 still runs the initial least-squares equality-multiplier factor,
# which is the first (and dumped) `multi_solve`.
POUNCE_DBG_KKT_DUMP="${BIN_DUMP}" "${POUNCE_BIN}" "${NL}" print_level=0 max_iter=0 >/dev/null 2>&1 || true

[[ -s "${BIN_DUMP}" ]] || { echo "ERROR: no KKT dump produced" >&2; exit 1; }

# Binary dump (little-endian): u64 dim,nnz,nrhs; i64[nnz] airn (1-based lower);
# i64[nnz] ajcn; f64[nnz] vals; f64[dim*nrhs] rhs. Emit symmetric MatrixMarket.
OUT="${DEST}" BIN="${BIN_DUMP}" "${PYTHON}" - <<'PY'
import os, numpy as np
b = os.environ["BIN"]; out = os.environ["OUT"]
with open(b, "rb") as f:
    dim, nnz, _ = (int(x) for x in np.frombuffer(f.read(24), dtype="<u8"))
    airn = np.frombuffer(f.read(8 * nnz), dtype="<i8")
    ajcn = np.frombuffer(f.read(8 * nnz), dtype="<i8")
    vals = np.frombuffer(f.read(8 * nnz), dtype="<f8")
with open(out, "w") as o:
    o.write("%%MatrixMarket matrix coordinate real symmetric\n")
    o.write(f"{dim} {dim} {nnz}\n")
    o.writelines(f"{i} {j} {v:.17e}\n"
                 for i, j, v in zip(airn.tolist(), ajcn.tolist(), vals.tolist()))
print(f"wrote {out}: dim={dim} nnz={nnz}")
PY

rm -f "${BIN_DUMP}"
echo "Regenerated ${DEST}"
