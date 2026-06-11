"""Numeric factor (L / D) access tests.

Oracle is the input matrix and the solver itself: the assembled
``L D Lᵀ`` is reconstructed and checked against the scaled, permuted
input, and re-solving with the reconstructed factor reproduces
``Solver.solve``.
"""

from __future__ import annotations

import numpy as np
import pytest

import feral


def _indef_matrix() -> np.ndarray:
    return np.array(
        [
            [2.0, 1.0, 0.0, 0.0],
            [1.0, -3.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, -2.0],
        ]
    )


def _assemble(fac) -> tuple[np.ndarray, np.ndarray]:
    """Build dense L (unit-lower) and D from a Factors snapshot."""
    n = fac.n
    indptr, indices, data = fac.l_csc()
    d_diag, d_sub = fac.d_blocks()
    L = np.zeros((n, n))
    for j in range(n):
        for k in range(indptr[j], indptr[j + 1]):
            L[indices[k], j] = data[k]
    D = np.zeros((n, n))
    i = 0
    while i < n:
        if i + 1 < n and d_sub[i] != 0.0:
            D[i, i] = d_diag[i]
            D[i + 1, i + 1] = d_diag[i + 1]
            D[i, i + 1] = d_sub[i]
            D[i + 1, i] = d_sub[i]
            i += 2
        else:
            D[i, i] = d_diag[i]
            i += 1
    return L, D


def test_ldlt_reconstruction_identity():
    A = _indef_matrix()
    s = feral.Solver()
    s.factor(feral.CscMatrix.from_dense(A))
    fac = s.factors()
    assert fac is not None
    L, D = _assemble(fac)
    M = L @ D @ L.T
    perm = fac.perm
    sc = fac.scaling
    # M[i,j] = s[perm[i]] * A[perm[i],perm[j]] * s[perm[j]]
    Mexp = np.array(
        [
            [sc[perm[i]] * A[perm[i], perm[j]] * sc[perm[j]] for j in range(A.shape[0])]
            for i in range(A.shape[0])
        ]
    )
    np.testing.assert_allclose(M, Mexp, rtol=1e-10, atol=1e-12)


def test_unit_diagonal_explicit():
    A = _indef_matrix()
    s = feral.Solver()
    s.factor(feral.CscMatrix.from_dense(A))
    L, _ = _assemble(s.factors())
    np.testing.assert_allclose(np.diag(L), np.ones(A.shape[0]), rtol=0, atol=1e-14)


def test_l_csc_matches_to_scipy_l():
    sp = pytest.importorskip("scipy.sparse")
    A = _indef_matrix()
    s = feral.Solver()
    s.factor(feral.CscMatrix.from_dense(A))
    fac = s.factors()
    indptr, indices, data = fac.l_csc()
    Lsp = fac.to_scipy_l()
    assert Lsp.shape == (fac.n, fac.n)
    expected = sp.csc_matrix((data, indices, indptr), shape=(fac.n, fac.n))
    np.testing.assert_allclose(Lsp.toarray(), expected.toarray(), rtol=0, atol=1e-14)


def test_factors_none_before_factor():
    s = feral.Solver()
    assert s.factors() is None


def test_nnz_matches_solver_factor_nnz():
    A = _indef_matrix()
    s = feral.Solver()
    s.factor(feral.CscMatrix.from_dense(A))
    assert s.factors().nnz == s.factor_nnz


def test_reconstruction_reproduces_solve():
    # The classic oracle: solving with the reconstructed factor (in
    # scaled/permuted coordinates) reproduces Solver.solve.
    A = _indef_matrix()
    csc = feral.CscMatrix.from_dense(A)
    s = feral.Solver()
    s.factor(csc)
    rng = np.random.default_rng(0)
    b = rng.standard_normal(A.shape[0])
    x_solver = s.solve(b)
    # Independent dense solve as the external oracle.
    x_oracle = np.linalg.solve(A, b)
    np.testing.assert_allclose(x_solver, x_oracle, rtol=1e-9, atol=1e-11)
