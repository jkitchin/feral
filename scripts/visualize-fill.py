#!/usr/bin/env python3
"""Visualize the fill-in story: original KKT pattern vs symbolic L
pattern under a poor (natural) ordering vs a fill-reducing ordering.

Pedagogical figure for the manuscript. Uses a pure-Python symbolic
LDL^T factorization (elimination-tree merge) so no external sparse-
direct dependency is required. The "fill-reducing" ordering is
SciPy's reverse Cuthill-McKee (RCM); FERAL's production AMD typically
gives similar or slightly better fill, but RCM is in stdlib SciPy and
shows the same dramatic contrast.

Outputs:

  manuscript/figures/fill-comparison.png   -- 3-panel side-by-side
  manuscript/figures/fill-A.png            -- original A only
  manuscript/figures/fill-L-natural.png    -- L under natural ordering
  manuscript/figures/fill-L-amd.png        -- L under RCM ordering
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.sparse
import scipy.sparse.csgraph

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MTX = ROOT / "data" / "matrices" / "kkt" / "KIRBY2" / "KIRBY2_0007.mtx"
FIG_DIR = ROOT / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_full_symmetric(path: Path) -> scipy.sparse.csc_matrix:
    """Read a Matrix Market symmetric file as a full symmetric CSC."""
    A = scipy.io.mmread(str(path))
    A = scipy.sparse.csc_matrix(A)
    A_T = A.T.tocsc()
    A_full = A + A_T
    A_full.setdiag(A.diagonal())
    A_full.eliminate_zeros()
    return A_full.tocsc()


def symbolic_ldlt(A: scipy.sparse.csc_matrix, perm: np.ndarray) -> tuple[scipy.sparse.csc_matrix, int]:
    """Return the lower-triangular L sparsity pattern of A[perm, perm] = LDL^T,
    plus the off-diagonal nnz count.

    Implements the elimination-tree merge: for each column k, the row
    pattern of L[k+1:, k] is the union of A[k+1:, k] (after permutation)
    and the row patterns of all children j of k in the elimination tree
    (filtered to indices > k). Children of k are the columns j < k whose
    minimum-row-index in L[:, j] is k.

    O(n + nnz(L)) time after the permutation; the cost-dominant operation
    is the per-column set merge.
    """
    n = A.shape[0]
    P = perm
    Pinv = np.empty(n, dtype=np.int64)
    Pinv[P] = np.arange(n)
    A_perm = A[P, :][:, P].tocsc()

    rows: list[set[int]] = [set() for _ in range(n)]
    parent = [-1] * n
    children: list[list[int]] = [[] for _ in range(n)]

    indptr = A_perm.indptr
    indices = A_perm.indices

    for k in range(n):
        col_start, col_end = indptr[k], indptr[k + 1]
        rows[k] = {int(i) for i in indices[col_start:col_end] if i > k}
        for j in children[k]:
            rows[k].update(i for i in rows[j] if i > k)
        if rows[k]:
            p = min(rows[k])
            parent[k] = p
            children[p].append(k)

    nnz_off = sum(len(s) for s in rows)
    row_idx: list[int] = []
    col_ptr = np.zeros(n + 1, dtype=np.int64)
    for k in range(n):
        col_ptr[k + 1] = col_ptr[k] + len(rows[k])
        row_idx.extend(sorted(rows[k]))
    L = scipy.sparse.csc_matrix(
        (np.ones(nnz_off, dtype=np.int8), np.array(row_idx, dtype=np.int64), col_ptr),
        shape=(n, n),
    )
    return L, nnz_off


def spy_panel(ax, M, title: str, n: int, dot_size: float, color: str = "#1B4F72") -> None:
    coo = M.tocoo()
    ax.scatter(coo.col, coo.row, s=dot_size, c=color, marker="s", linewidths=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#888")


def main(mtx_path: Path = DEFAULT_MTX) -> int:
    print(f"loading {mtx_path}", file=sys.stderr)
    A = load_full_symmetric(mtx_path)
    n = A.shape[0]
    A_lower_tmp = scipy.sparse.tril(A, k=-1)
    nnz_A_lower_strict = A_lower_tmp.nnz
    nnz_A_lower = scipy.sparse.tril(A, k=0).nnz
    print(f"  n = {n}, full nnz = {A.nnz}, strict-lower nnz = {nnz_A_lower}", file=sys.stderr)

    natural = np.arange(n, dtype=np.int64)
    rcm = np.asarray(scipy.sparse.csgraph.reverse_cuthill_mckee(A.tocsr(), symmetric_mode=True))

    print("symbolic factor: natural", file=sys.stderr)
    L_nat, nnz_L_nat = symbolic_ldlt(A, natural)
    print(f"  L nnz (off-diag) = {nnz_L_nat}", file=sys.stderr)

    print("symbolic factor: rcm", file=sys.stderr)
    L_rcm, nnz_L_rcm = symbolic_ldlt(A, rcm)
    print(f"  L nnz (off-diag) = {nnz_L_rcm}", file=sys.stderr)

    A_lower = scipy.sparse.tril(A, k=0).tocsc()
    fill_nat = nnz_L_nat / max(nnz_A_lower_strict, 1)
    fill_rcm = nnz_L_rcm / max(nnz_A_lower_strict, 1)

    base = mtx_path.stem
    dot = max(0.4, 4.0 - np.log10(n))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    spy_panel(axes[0], A_lower, f"original $A$ (lower)\n$n = {n}$, $\\mathrm{{nnz}} = {nnz_A_lower:,}$", n, dot)
    spy_panel(
        axes[1],
        L_nat,
        f"$L$ under natural ordering\n$\\mathrm{{nnz}}(L) = {nnz_L_nat:,}$ ({fill_nat:.1f}$\\times$ A)",
        n,
        dot,
        color="#C44536",
    )
    spy_panel(
        axes[2],
        L_rcm,
        f"$L$ under RCM ordering\n$\\mathrm{{nnz}}(L) = {nnz_L_rcm:,}$ ({fill_rcm:.1f}$\\times$ A)",
        n,
        dot,
        color="#2E7D9A",
    )
    fig.suptitle(f"Fill-in comparison on {base}", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fill-comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    for label, M, color, fname, title in [
        ("A", A_lower, "#1B4F72", "fill-A.png", f"$A$ (lower) on {base}: $n = {n}$, $\\mathrm{{nnz}} = {nnz_A_lower:,}$"),
        ("L-nat", L_nat, "#C44536", "fill-L-natural.png", f"$L$ natural ordering: $\\mathrm{{nnz}} = {nnz_L_nat:,}$"),
        ("L-rcm", L_rcm, "#2E7D9A", "fill-L-amd.png", f"$L$ RCM ordering: $\\mathrm{{nnz}} = {nnz_L_rcm:,}$"),
    ]:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        spy_panel(ax, M, title, n, dot * 1.2, color=color)
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote 4 figures to {FIG_DIR}", file=sys.stderr)
    print(f"  fill ratio: natural = {fill_nat:.2f}x, RCM = {fill_rcm:.2f}x", file=sys.stderr)
    return 0


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MTX
    sys.exit(main(arg))
