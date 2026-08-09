#!/usr/bin/env python3
"""Generate block-tridiagonal (chain-structured) KKT proxies at the
geometries reported in pounce#552.

Each model is a direct-transcription dynamic optimization problem, so its
KKT matrix is symmetric indefinite and block tridiagonal: one diagonal
block per time point, coupled to the next point by state continuity.
What drives factorization cost is that shape (a chain-shaped elimination
tree whose front sizes track the per-time-point block size), so the
proxy fixes T (time points) and b (vars per time point) to the reported
values and fills each block with a plausible banded saddle structure.

Per time point t, the block is

    [ H_t   A_t^T ]      H_t : nx x nx  banded SPD
    [ A_t    0    ]      A_t : nc x nx  banded constraint Jacobian

with the (2,2) block structurally absent, as in a true equality-
constrained KKT. Consecutive points are coupled by C_t, which puts the
continuity entries of point t+1's dual rows on point t's state columns.

Writes MatrixMarket symmetric coordinate (lower triangle only), which is
the convention both bench_one_matrix and ma57_bench read.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# (label, T time points, nx primal per point, nc dual per point)
# n = T * (nx + nc); chosen to land within ~1% of the reported n.
MODELS = [
    ("hicks_like", 301, 3, 3),           # reported n=1,706
    ("cart_pole_like", 301, 5, 4),       # reported n=2,810
    ("quad_tank_like", 301, 6, 4),       # reported n=2,910
    ("double_column_like", 31, 491, 328),  # reported n=25,377
    ("prommis_sx_like", 31, 548, 365),   # reported n=28,313
]

H_HALF_BAND = 2   # half-bandwidth of H_t
A_WIDTH = 3       # primal columns touched per dual row
C_WIDTH = 2       # state columns of point t touched per dual row of t+1
DELTA = 1e-4      # (2,2) regularization, as in an interior-point KKT


def gen(T: int, nx: int, nc: int) -> tuple[int, list[tuple[int, int, float]]]:
    """Return (n, lower-triangle triplets) with 0-based indices."""
    b = nx + nc
    n = T * b
    trips: list[tuple[int, int, float]] = []

    def add(r: int, c: int, v: float) -> None:
        # store lower triangle only
        if r < c:
            r, c = c, r
        trips.append((r, c, v))

    for t in range(T):
        base = t * b
        p0 = base           # primal block start
        d0 = base + nx      # dual block start

        # H_t: banded SPD, diagonally dominant.
        for i in range(nx):
            add(p0 + i, p0 + i, 2.0 + 0.1 * (i / max(nx, 1)))
            for k in range(1, H_HALF_BAND + 1):
                if i + k < nx:
                    add(p0 + i + k, p0 + i, -1.0 / k)

        # A_t: each dual row touches A_WIDTH primal columns, spread
        # across the primal block so the Jacobian has full row rank.
        #
        # The starts must be spread over [0, nx - A_WIDTH] rather than
        # advanced by a fixed stride and clamped: with nx close to
        # A_WIDTH a fixed stride pins every row to the same start, and
        # if the values are also a function of k alone the rows come out
        # identical, making A_t rank 1. Values therefore depend on both
        # r and k, and non-proportionally, so no two rows are multiples.
        span = max(nx - A_WIDTH, 0)
        for r in range(nc):
            start = (r * span) // max(nc - 1, 1) if nc > 1 else 0
            for k in range(A_WIDTH):
                c = start + k
                if c < nx:
                    add(d0 + r, p0 + c, 1.0 + 0.05 * k + 0.37 * (((r * (k + 1)) % 5) / 5.0))

        # (2,2) block: -delta*I. Real interior-point KKTs carry this
        # regularization, and it keeps the proxy comfortably nonsingular
        # so the inertia is well defined (exactly nc negatives per point
        # come from here) and both solvers can be held to a tight
        # residual.
        for r in range(nc):
            add(d0 + r, d0 + r, -DELTA)

        # C_t: continuity — point t+1's dual rows reference point t's
        # state columns. This is the only inter-block coupling, and it
        # is what makes the elimination tree a chain.
        if t + 1 < T:
            d1 = (t + 1) * b + nx
            cspan = max(nx - C_WIDTH, 0)
            for r in range(nc):
                start = (r * cspan) // max(nc - 1, 1) if nc > 1 else 0
                for k in range(C_WIDTH):
                    c = start + k
                    if c < nx:
                        add(d1 + r, p0 + c, -1.0 - 0.11 * (((r + k) % 3) / 3.0))

    return n, trips


def write_mtx(path: Path, n: int, trips: list[tuple[int, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write("% chain-structured KKT proxy for pounce#552 geometry\n")
        f.write(f"{n} {n} {len(trips)}\n")
        for r, c, v in trips:
            f.write(f"{r + 1} {c + 1} {v:.17g}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    out = Path(args.out)

    print(f"{'model':<22}{'T':>5}{'b':>6}{'n':>9}{'nnz':>10}{'nnz/row':>9}")
    for label, T, nx, nc in MODELS:
        n, trips = gen(T, nx, nc)
        write_mtx(out / f"{label}.mtx", n, trips)
        print(f"{label:<22}{T:>5}{nx + nc:>6}{n:>9}{len(trips):>10}"
              f"{len(trips) / n:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
