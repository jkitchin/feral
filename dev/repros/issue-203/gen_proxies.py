"""Proxy KKTs for the two #67/#73 families that decided the Auto reroute.

The families themselves (dtoc2, cont5_1_l, nql180, YATP1NE, pinene) live in a
corpus that is not on this machine. These are *geometry-matched stand-ins* for
the two whose structure is public and unambiguous:

  dtoc   — discrete-time optimal control. A chain: step `t` couples only to
           `t+1`, through a **narrow** state vector. dtoc2 in #73 is
           n=104k, avg_deg 17.5.
  pde2d  — elliptic PDE-constrained control on a 2-D grid. cont5_1_l in #73
           is n=181k, avg_deg 6.96.

Both are written as the augmented system

    K = [ H    J^T ]     H strictly diagonally dominant positive (SPD)
        [ J   -dI  ]     d = 1e-2

so the inertia is analytically `(n_vars, m, 0)` — the Schur complement
`-dI - J H^-1 J^T` is negative definite whatever `J` is.

**These are proxies.** `external_benchmarks/chain_proxy/README.md` records a
case where geometry-matched stand-ins produced the wrong answer, and that
warning applies here in full. A result from this file is a hypothesis about
the real family, not a measurement of it.

Usage: python gen_proxies.py OUTDIR
"""
import sys
from pathlib import Path

import numpy as np

DELTA = 1e-2
OFFDIAG = 0.37


def write_kkt(path, n_vars, m, h_pairs, j_pairs):
    """`h_pairs` are (row, col) in [0, n_vars); `j_pairs` are (crow, col)."""
    N = n_vars + m
    hr = np.asarray(h_pairs[0], dtype=np.int64)
    hc = np.asarray(h_pairs[1], dtype=np.int64)
    jr = np.asarray(j_pairs[0], dtype=np.int64) + n_vars
    jc = np.asarray(j_pairs[1], dtype=np.int64)
    r = np.concatenate([hr, jr, np.arange(N)])
    c = np.concatenate([hc, jc, np.arange(N)])
    lo, hi = np.minimum(r, c), np.maximum(r, c)
    key = hi * N + lo
    _, keep = np.unique(key, return_index=True)
    r, c = hi[keep], lo[keep]

    off = r != c
    vals = np.where(off, OFFDIAG * np.where((r + c) % 2 == 0, 1.0, -1.0), 0.0)
    absrow = np.zeros(N)
    np.add.at(absrow, r[off], np.abs(vals[off]))
    np.add.at(absrow, c[off], np.abs(vals[off]))
    diag = r == c
    di = r[diag]
    vals[diag] = np.where(di < n_vars, absrow[di] + 1.0, -DELTA)

    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"% proxy KKT; inertia = ({n_vars}, {m}, 0)\n")
        f.write(f"{N} {N} {len(r)}\n")
        np.savetxt(f, np.column_stack([r + 1, c + 1, vals]), fmt="%d %d %.17g")
    deg = 2 * len(r) - N
    print(f"{path}  N={N} nnz={len(r)} n_vars={n_vars} avg_deg={deg / N:.2f}")


def dtoc(T, nx, nu, out):
    """Narrow chain: variables [x_0 u_0 x_1 u_1 ...], one dynamics block per
    step coupling (x_t, u_t, x_{t+1}). H is dense within each step's
    (x_t, u_t) block."""
    step = nx + nu
    n_vars = T * step
    m = (T - 1) * nx
    a, b = np.meshgrid(np.arange(step), np.arange(step), indexing="ij")
    t = np.arange(T)[:, None, None] * step
    hr = (t + a).ravel()
    hc = (t + b).ravel()

    # Row (t, i) of the dynamics touches x_t, u_t and x_{t+1}.
    ts = np.arange(T - 1)
    cols_local = np.concatenate([np.arange(step), step + np.arange(nx)])
    ci, cj = np.meshgrid(np.arange(nx), cols_local, indexing="ij")
    jr = (ts[:, None, None] * nx + ci).ravel()
    jc = (ts[:, None, None] * step + cj).ravel()
    write_kkt(out, n_vars, m, (hr, hc), (jr, jc))


def pde2d(k, out):
    """Elliptic control: state and control on a k x k grid, one 5-point
    equation per node coupling the state stencil and the local control."""
    idx = np.arange(k * k).reshape(k, k)
    ns = k * k
    n_vars = 2 * ns          # [state | control]
    m = ns
    # H: state-state diagonal plus state-control coupling at each node.
    hr = np.concatenate([np.arange(ns), ns + np.arange(ns), ns + np.arange(ns)])
    hc = np.concatenate([np.arange(ns), ns + np.arange(ns), np.arange(ns)])
    # J: 5-point Laplacian on the state, minus the local control.
    rows = [idx.ravel(), idx.ravel()]
    cols = [idx.ravel(), ns + idx.ravel()]
    for ax in range(2):
        a = np.take(idx, np.arange(k - 1), axis=ax).ravel()
        b = np.take(idx, np.arange(1, k), axis=ax).ravel()
        rows += [a, b]
        cols += [b, a]
    write_kkt(out, n_vars, m, (hr, hc), (np.concatenate(rows), np.concatenate(cols)))


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    # dtoc2 in #73: n=104k, avg_deg 17.5.
    dtoc(10400, 4, 2, out / "dtoc_proxy.mtx")
    # cont5_1_l in #73: n=181k, avg_deg 6.96.
    pde2d(246, out / "pde2d_proxy.mtx")
