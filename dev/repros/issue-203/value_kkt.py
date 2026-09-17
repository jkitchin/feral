"""Give the issue #203 KKT pattern block-aware values, for wall-clock timing.

kkt_pattern_repro.py writes diag 4 / offdiag 1, which is fine for symbolic
analysis and useless for numeric factorization: it is not a saddle point, so
the pivoting a real KKT drives never happens.

This rewrites the *same pattern* with values that make it the augmented system
it represents,

    K = [ H    J^T ]        rows/cols [0, n)   are variables
        [ J   -dI  ]        rows/cols [n, n+m) are constraints

with `H` strictly diagonally dominant and positive on the diagonal (so SPD)
and `d > 0`.

**The inertia oracle.** For `H` SPD and `d > 0`, the Schur complement
`-dI - J H^-1 J^T` is negative definite, so `K` has inertia `(n, m, 0)`
*whatever* `J` is — no rank assumption needed. That is an analytic oracle,
independent of any solver, and it is what the A/B harness gates on before it
reports a single timing.

Usage: python value_kkt.py IN.mtx OUT.mtx N_VARS
"""
import sys

import numpy as np

DELTA = 1e-2          # the -dI dual regularization
OFFDIAG = 0.37        # magnitude of every off-diagonal entry


def main(src, dst, n_vars):
    with open(src) as f:
        line = f.readline()
        while True:
            line = f.readline()
            if not line.startswith("%"):
                break
        N, _, nz = map(int, line.split())
        dat = np.loadtxt(f, usecols=(0, 1), dtype=np.int64)
    r = dat[:, 0] - 1
    c = dat[:, 1] - 1
    assert r.size == nz

    off = r != c
    # Alternate the sign so H is not an M-matrix and J has mixed signs.
    vals = np.where(off, OFFDIAG * np.where((r + c) % 2 == 0, 1.0, -1.0), 0.0)

    # Row sums of |off-diagonal| over the full symmetric matrix, so the
    # dominance below is a statement about K's rows, not the stored triangle.
    absrow = np.zeros(N)
    np.add.at(absrow, r[off], np.abs(vals[off]))
    np.add.at(absrow, c[off], np.abs(vals[off]))

    diag = r == c
    di = r[diag]
    # H: strictly dominant and positive. Dual block: exactly -DELTA.
    dval = np.where(di < n_vars, absrow[di] + 1.0, -DELTA)
    vals[diag] = dval

    with open(dst, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"% issue #203 pattern, saddle-point values; inertia = ({n_vars}, {N - n_vars}, 0)\n")
        f.write(f"{N} {N} {nz}\n")
        np.savetxt(f, np.column_stack([r + 1, c + 1, vals]), fmt="%d %d %.17g")
    print(f"{dst}  N={N} nnz={nz} n_vars={n_vars} inertia=({n_vars}, {N - n_vars}, 0)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
