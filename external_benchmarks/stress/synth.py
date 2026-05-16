#!/usr/bin/env python3
"""Synthetic stress matrices for feral.

Generates the `synth/*` rows from manifest.tsv as Matrix-Market
symmetric files under `matrices/synth/`. Each category is constructed
with a *known* property the solver should exhibit:

  rankdef_<n>_<k>      symmetric indefinite, rank n-k, k explicit zero
                        eigenvalues. Inertia oracle: (?, ?, k).
  near_singular_eps<p> diag-perturbed sym indef; min |pivot| ~ 10^-p.
                        Probes the 2x2 pivot threshold.
  ill_cond_e<p>        symmetric indefinite cond(A) ~ 10^p.
                        Probes residual quality on κ-stressed solves.
  deep_null_cascade_n  tridiagonal-like sym matrix where the first
                        n//2 diagonal entries are zero, forcing a long
                        chain of 2x2 pivots / null-pivot cascades.

The generators are seeded so output is bit-reproducible. Matrices are
written in MatrixMarket coordinate-symmetric format (lower triangle
only, matching the rest of feral's drivers).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

STRESS_DIR = Path(__file__).resolve().parent
MATRICES_DIR = STRESS_DIR / "matrices" / "synth"


def write_mtx_symmetric(path: Path, A: np.ndarray) -> None:
    """Write the lower triangle of symmetric A in MatrixMarket format."""
    n = A.shape[0]
    assert A.shape == (n, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols, vals = [], [], []
    for j in range(n):
        for i in range(j, n):
            v = A[i, j]
            if v != 0.0 or i == j:
                rows.append(i + 1)
                cols.append(j + 1)
                vals.append(v)
    with path.open("w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"{n} {n} {len(rows)}\n")
        for r, c, v in zip(rows, cols, vals):
            f.write(f"{r} {c} {v:.17e}\n")


def gen_rankdef(n: int, k: int, seed: int) -> np.ndarray:
    """Symmetric indef matrix with exactly k zero eigenvalues.

    Construction: pick a random orthonormal basis Q, set diagonal
    eigenvalues D = [d_1 .. d_{n-k}, 0, 0, ..., 0], with d_i drawn
    from a symmetric distribution so the matrix is indefinite.
    A = Q D Q^T.
    """
    rng = np.random.default_rng(seed)
    # Eigenvalues: half positive, half negative, k zeros at the tail.
    nz = n - k
    signs = np.where(rng.random(nz) < 0.5, -1.0, 1.0)
    mags = rng.uniform(0.5, 3.0, size=nz)
    eigs = np.concatenate([signs * mags, np.zeros(k)])
    # Random orthonormal basis.
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * eigs) @ Q.T
    # Symmetrize to kill floating-point asymmetry.
    return 0.5 * (A + A.T)


def gen_near_singular(n: int, eps_pow: int, seed: int) -> np.ndarray:
    """sym indef with a single pivot at scale 10^{-eps_pow}."""
    rng = np.random.default_rng(seed)
    signs = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    mags = rng.uniform(0.5, 3.0, size=n)
    eigs = signs * mags
    eigs[0] = 10.0 ** (-eps_pow)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * eigs) @ Q.T
    return 0.5 * (A + A.T)


def gen_ill_cond(n: int, cond_pow: int, seed: int) -> np.ndarray:
    """sym indef with cond(A) ~ 10^cond_pow, geometric eigenvalue spread."""
    rng = np.random.default_rng(seed)
    signs = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    # Eigenvalue magnitudes: geometric from 1.0 down to 10^-cond_pow.
    mags = np.logspace(0.0, -cond_pow, num=n)
    eigs = signs * mags
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * eigs) @ Q.T
    return 0.5 * (A + A.T)


def gen_deep_null_cascade(n: int) -> np.ndarray:
    """Symmetric matrix where the first n//2 diagonals are zero but
    each is paired with a strong off-diagonal one row down, forcing
    a long chain of 2x2 pivots through the BK kernel.

    A[i,i]     = 0   for i < n//2
    A[i,i]     = 1   for i >= n//2
    A[i+1, i]  = 1   for i < n-1
    A[i+2, i]  = 0.3 for i < n-2  (creates near-singular 2x2 blocks)
    """
    A = np.zeros((n, n))
    half = n // 2
    for i in range(n):
        A[i, i] = 0.0 if i < half else 1.0
    for i in range(n - 1):
        A[i + 1, i] = 1.0
        A[i, i + 1] = 1.0
    for i in range(n - 2):
        A[i + 2, i] = 0.3
        A[i, i + 2] = 0.3
    return A


GENERATORS = {
    "rankdef_5_2":         lambda: gen_rankdef(5, 2, seed=1),
    "rankdef_10_3":        lambda: gen_rankdef(10, 3, seed=2),
    "rankdef_50_5":        lambda: gen_rankdef(50, 5, seed=3),
    "rankdef_200_20":      lambda: gen_rankdef(200, 20, seed=4),
    "near_singular_eps9":  lambda: gen_near_singular(100, 9, seed=5),
    "near_singular_eps12": lambda: gen_near_singular(100, 12, seed=6),
    "ill_cond_e10":        lambda: gen_ill_cond(100, 10, seed=7),
    "ill_cond_e14":        lambda: gen_ill_cond(100, 14, seed=8),
    "deep_null_cascade_50":  lambda: gen_deep_null_cascade(50),
    "deep_null_cascade_200": lambda: gen_deep_null_cascade(200),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="generate only this single matrix name")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    names = [args.only] if args.only else list(GENERATORS)
    MATRICES_DIR.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for name in names:
        if name not in GENERATORS:
            print(f"  skip unknown synthetic {name}", flush=True)
            continue
        tgt = MATRICES_DIR / f"{name}.mtx"
        if tgt.exists() and not args.force:
            print(f"  skip existing {name}", flush=True)
            continue
        A = GENERATORS[name]()
        write_mtx_symmetric(tgt, A)
        print(f"  wrote {tgt.relative_to(STRESS_DIR.parent.parent)} "
              f"(n={A.shape[0]})", flush=True)
        n_new += 1
    print(f"\ndone: {n_new} matrices generated", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
