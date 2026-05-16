#!/usr/bin/env python3
"""Synthetic stress matrices for feral.

Generates the `synth/*` rows from manifest.tsv as Matrix-Market
symmetric files under `matrices/synth/`. Each category is constructed
with a *known* property the solver should exhibit:

  rankdef_<n>_<k>           symmetric indefinite, rank n-k, k zero
                             eigenvalues. Inertia oracle: (?, ?, k).
  rankdef_exact_<n>_<k>     same skeleton but with k *exact* IEEE 0.0
                             eigenvalues and a seed chosen to make the
                             null space maximally dispersed (issue #31
                             follow-up to #27).
  near_singular_eps<p>      diag-perturbed sym indef; min |pivot| ~ 10^-p.
                             Probes the 2x2 pivot threshold.
  ill_cond_e<p>             symmetric indefinite cond(A) ~ 10^p.
                             Probes residual quality on κ-stressed solves.
  deep_null_cascade_n       tridiagonal-like sym matrix where the first
                             n//2 diagonal entries are zero, forcing a
                             long chain of 2x2 pivots / null-pivot
                             cascades.
  saddle_rankdef_<n>_<k>_<r>  KKT block [H A^T; A 0] with H ≻ 0,
                             m = n − k constraints, rank(A) = m − r.
                             Inertia oracle (n, m − r, r).
  wide_frontal_<n>          bordered block diagonal that forces a single
                             supernode of width > 1000. Stresses the
                             sparse-to-dense crossover.
  mc64_resistant_<n>        rank-1 perturbation of a diagonally dominant
                             skeleton. MC64 matching succeeds but the
                             resulting scaling leaves cond(A) ~ 10^8.
  stokes_q1p0_<h>           Q1-P0 mixed-element Stokes saddle on an
                             h × h unit-square mesh with Dirichlet
                             velocity boundary. Inertia oracle
                             (n_u, n_p − 1, 1).

The generators are seeded so output is bit-reproducible. Matrices are
written in MatrixMarket coordinate-symmetric format (lower triangle
only, matching the rest of feral's drivers).

Math, oracle derivations, and pathology rationale live in
`dev/research/synthetic-generators-m4.md`.
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


def gen_rankdef_exact(n: int, k: int, seed: int) -> np.ndarray:
    """Rank-deficient sym indef with k *exact* IEEE 0.0 eigenvalues.

    Same Q D Q^T construction as gen_rankdef, but the trailing k
    eigenvalues are written as literal 0.0 (no random tiny values),
    and the chosen seed produces a dense null-space mixing so that
    partial-pivot BK cannot trivially detect the rank loss from the
    diagonal pattern. This is the explicit oracle requested in the
    issue #31 follow-up to issue #27.
    """
    rng = np.random.default_rng(seed)
    nz = n - k
    signs = np.where(rng.random(nz) < 0.5, -1.0, 1.0)
    mags = rng.uniform(0.5, 3.0, size=nz)
    eigs = np.concatenate([signs * mags, np.zeros(k, dtype=np.float64)])
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * eigs) @ Q.T
    return 0.5 * (A + A.T)


def gen_saddle_rankdef(n: int, k: int, r: int, seed: int) -> np.ndarray:
    """KKT-shape saddle [H A^T; A 0] with rank-deficient constraint.

    H ∈ R^{n × n} is SPD (M^T M + δI, M Gaussian, δ = 1e-2). A is
    m × n with m = n − k; A is first generated Gaussian then its SVD
    is rank-truncated to rank m − r (the trailing r singular values
    are zeroed). The resulting saddle has inertia (n, m − r, r). Total
    matrix size is n + m = 2n − k.
    """
    if k < 0 or r < 0 or k > n or r > n - k:
        raise ValueError(f"bad saddle params n={n} k={k} r={r}")
    rng = np.random.default_rng(seed)
    m = n - k
    M = rng.standard_normal((n, n))
    H = M.T @ M + 1e-2 * np.eye(n)
    A = rng.standard_normal((m, n))
    if r > 0:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S[m - r:] = 0.0  # drop smallest r singular values to enforce rank m-r
        A = (U * S) @ Vt
    N = n + m
    K = np.zeros((N, N))
    K[:n, :n] = H
    K[n:, :n] = A
    K[:n, n:] = A.T
    # bottom-right block is left zero by construction
    return 0.5 * (K + K.T)


def gen_wide_frontal(width: int, n_leaves: int, seed: int,
                     border_density: float = 0.05) -> np.ndarray:
    """Bordered block-diagonal that forces a single wide supernode.

    Layout: n_leaves leaf columns each connected only to a dense tail
    block of size width × width. The elimination tree is therefore
    n_leaves leaves all feeding a single supernode of width = width.

    The tail block is Q D Q^T with a balanced indefinite spectrum, then
    pruned: entries with |T_w[i,j]| < 1e-3 are zeroed to keep nnz
    moderate. Leaves are +1 on the diagonal; borders are short random
    rows of length width with `border_density` fraction nonzero.
    """
    rng = np.random.default_rng(seed)
    n = width + n_leaves
    # Build tail: Q diag(eigs) Q^T with balanced indefinite spectrum.
    Q, _ = np.linalg.qr(rng.standard_normal((width, width)))
    signs = np.where(rng.random(width) < 0.5, -1.0, 1.0)
    mags = rng.uniform(0.5, 3.0, size=width)
    eigs = signs * mags
    T = (Q * eigs) @ Q.T
    T = 0.5 * (T + T.T)
    # Prune small entries to keep nnz manageable; preserve diagonal.
    mask = np.abs(T) < 1e-3
    np.fill_diagonal(mask, False)
    T[mask] = 0.0
    # Assemble full matrix: leaves first, then tail.
    A = np.zeros((n, n))
    # Leaf diagonals: +1.
    for i in range(n_leaves):
        A[i, i] = 1.0
    # Each leaf gets a sparse border into the tail.
    n_nz_per_border = max(1, int(border_density * width))
    for i in range(n_leaves):
        cols = rng.choice(width, size=n_nz_per_border, replace=False)
        for c in cols:
            v = rng.uniform(-1.0, 1.0)
            A[n_leaves + c, i] = v
            A[i, n_leaves + c] = v
    A[n_leaves:, n_leaves:] = T
    return 0.5 * (A + A.T)


def gen_mc64_resistant(n: int, seed: int,
                       small_eig: float = 1e-8) -> np.ndarray:
    """Dense sym indef whose ill-conditioning survives diagonal scaling.

    Construction: A = Q D Q^T with Q a random orthonormal basis and D
    a balanced indefinite spectrum where exactly one eigenvalue is
    `small_eig` (default 1e-8) and the rest are O(1). The smallness
    is in the *spectral* sense, not the diagonal sense — A's diagonal
    entries are all O(1) because the eigenvector for the tiny
    eigenvalue is dense in the original basis.

    MC64 matching reports success on this matrix (every row has an
    O(1) diagonal that the matching can pick), and the symmetric
    scaling s_i = exp((u_i + v_i)/2) ≈ O(1). After scaling, the
    spectrum is essentially unchanged: cond(A) and cond(D_s A D_s)
    both ~ 1/small_eig. This is the regression target for "MC64
    succeeded but scaling did not equilibrate". A future iterative-
    refinement or auxiliary-equilibration gate should detect that the
    matrix is ill-conditioned even after scaling.

    Inertia is data-dependent (recorded in the generator log); for
    default parameters with n=200 and seed=601 it is (111, 89, 0).
    """
    rng = np.random.default_rng(seed)
    signs = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    mags = rng.uniform(0.5, 2.0, size=n)
    eigs = signs * mags
    eigs[0] = small_eig
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * eigs) @ Q.T
    return 0.5 * (A + A.T)


def gen_stokes_q1p0(h: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Q1-P0 mixed-element Stokes saddle on a unit-square h × h mesh.

    Velocity: bilinear Q1 on (h+1) × (h+1) nodes, 2 components, with
    homogeneous Dirichlet on the boundary (boundary velocity DOFs are
    eliminated). Pressure: piecewise constant on h × h elements.

    Saddle:
        K = [ A    B^T ]
            [ B     0  ]

    where A is the velocity Laplacian (block-diag in the two velocity
    components) and B is the discrete divergence.

    The Q1 Laplacian element stencil on the unit square is computed
    in closed form; pressure-velocity coupling uses standard mixed-
    element quadrature. Q1-P0 famously fails the LBB condition with
    *two* spurious pressure modes in 2D: the global constant and the
    "checkerboard" alternating mode. Therefore rank(B) = n_p − 2 and
    the saddle inertia is (n_u_free, n_p − 2, 2).

    Returns (K, (n_u_free, n_p, expected_zero=2)).
    """
    if h < 2:
        raise ValueError(f"need h >= 2, got {h}")
    nx = h + 1  # nodes per side
    n_p = h * h  # pressure dofs (one per element)

    def node(i: int, j: int) -> int:
        return j * nx + i

    n_nodes = nx * nx
    # Boundary node indices (one component).
    boundary = set()
    for i in range(nx):
        boundary.add(node(i, 0))
        boundary.add(node(i, nx - 1))
    for j in range(nx):
        boundary.add(node(0, j))
        boundary.add(node(nx - 1, j))
    # Free interior nodes per component (map to compact indices).
    free_nodes = [k for k in range(n_nodes) if k not in boundary]
    node_to_free = {k: idx for idx, k in enumerate(free_nodes)}
    n_u_per_comp = len(free_nodes)
    n_u = 2 * n_u_per_comp
    N = n_u + n_p

    # Element Q1 stencil on a unit square element of side 1/h. For
    # the bilinear element of side dx the local Laplacian is the
    # reference 4x4 matrix scaled by 1 (it is dimensionless after the
    # standard derivation). We use the textbook reference matrix.
    # K_local[a,b] = ∫∇φ_a · ∇φ_b on the reference square.
    KL = (1.0 / 6.0) * np.array([
        [ 4.0, -1.0, -2.0, -1.0],
        [-1.0,  4.0, -1.0, -2.0],
        [-2.0, -1.0,  4.0, -1.0],
        [-1.0, -2.0, -1.0,  4.0],
    ])
    # Element divergence (constant pressure × Q1 velocity component):
    # ∫ p · ∂_x v dx on the reference element gives the 1x4 vector
    # for each velocity component. For Q1 on a unit square, the
    # average of ∂_x φ_a over the element equals ±1/2 with sign
    # determined by the node's x-position. Same for ∂_y for the v
    # component.
    # Local node order: (0,0)=0, (1,0)=1, (1,1)=2, (0,1)=3
    Bx_local = 0.5 * np.array([-1.0,  1.0,  1.0, -1.0])  # u component
    By_local = 0.5 * np.array([-1.0, -1.0,  1.0,  1.0])  # v component

    # Assemble A (block diag in two velocity components).
    Au = np.zeros((n_u_per_comp, n_u_per_comp))
    Bx = np.zeros((n_p, n_u_per_comp))
    By = np.zeros((n_p, n_u_per_comp))
    for ej in range(h):
        for ei in range(h):
            elem = ej * h + ei
            # Local node ids in global numbering.
            gn = [
                node(ei,     ej    ),
                node(ei + 1, ej    ),
                node(ei + 1, ej + 1),
                node(ei,     ej + 1),
            ]
            # Velocity Laplacian: skip rows/cols on the Dirichlet
            # boundary (their dofs are eliminated).
            for a in range(4):
                ga = gn[a]
                if ga in boundary:
                    continue
                ia = node_to_free[ga]
                for b in range(4):
                    gb = gn[b]
                    if gb in boundary:
                        continue
                    ib = node_to_free[gb]
                    Au[ia, ib] += KL[a, b]
            # Divergence: pressure has no boundary DOFs, but velocity
            # DOFs on the boundary contribute 0 (their column drops).
            for a in range(4):
                ga = gn[a]
                if ga in boundary:
                    continue
                ia = node_to_free[ga]
                Bx[elem, ia] += Bx_local[a]
                By[elem, ia] += By_local[a]

    # Pack velocity as [u_free; v_free].
    A = np.zeros((n_u, n_u))
    A[:n_u_per_comp, :n_u_per_comp] = Au
    A[n_u_per_comp:, n_u_per_comp:] = Au
    B = np.zeros((n_p, n_u))
    B[:, :n_u_per_comp] = Bx
    B[:, n_u_per_comp:] = By

    K = np.zeros((N, N))
    K[:n_u, :n_u] = A
    K[n_u:, :n_u] = B
    K[:n_u, n_u:] = B.T
    # bottom-right zero block

    return 0.5 * (K + K.T), (n_u, n_p, 2)


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

# Issue #27 + #31 follow-up: exact-zero rank-deficient variant, saddle
# rankdef, wide-frontal, MC64-resistant, Stokes Q1-P0.
GENERATORS["rankdef_exact_50_5"] = (
    lambda: gen_rankdef_exact(50, 5, seed=301)
)
GENERATORS["rankdef_exact_100_10"] = (
    lambda: gen_rankdef_exact(100, 10, seed=302)
)
GENERATORS["saddle_rankdef_50_10_3"] = (
    lambda: gen_saddle_rankdef(50, 10, 3, seed=401)
)
GENERATORS["saddle_rankdef_100_20_5"] = (
    lambda: gen_saddle_rankdef(100, 20, 5, seed=402)
)
GENERATORS["wide_frontal_616"] = (
    lambda: gen_wide_frontal(width=600, n_leaves=16, seed=501)
)
GENERATORS["mc64_resistant_200"] = (
    lambda: gen_mc64_resistant(200, seed=601)
)
GENERATORS["stokes_q1p0_8"] = (
    lambda: gen_stokes_q1p0(8)[0]
)

# Parametric near-singular sweep (issue #31). Each `near_singular_eps_<p>`
# is a 100x100 sym indef matrix with exactly one eigenvalue at scale 10^-p,
# the remaining 99 eigenvalues uniform in [0.5, 3.0] with random sign. The
# seed varies with p so the sweep is not just rescaling of a single random
# basis. p ∈ {6..14} covers the regime from "well above" the BK threshold
# (1e-8) down to "two orders below" so we can pinpoint the detection bound.
for _p in range(6, 15):
    GENERATORS[f"near_singular_eps_{_p}"] = (
        lambda p=_p: gen_near_singular(100, p, seed=100 + p)
    )


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
