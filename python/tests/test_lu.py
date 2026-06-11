"""Unsymmetric LU basis-engine tests.

Oracles are external: dense solves are checked against
``numpy.linalg.solve``; the factor reconstruction is checked against the
``P A Q = L U`` identity from the returned permutations.
"""

from __future__ import annotations

import numpy as np
import pytest

import feral


def _dense_basis(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Diagonally dominant → well-conditioned, non-singular.
    A = rng.standard_normal((n, n))
    A += n * np.eye(n)
    return A


def test_dense_ftran_btran_residual():
    A = _dense_basis(8, 0)
    lu = feral.LuFactor(feral.LuMatrix.from_dense(A), force_dense=True)
    assert lu.is_dense
    rng = np.random.default_rng(1)
    b = rng.standard_normal(8)
    x = lu.ftran(b)
    np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-10, atol=1e-12)
    c = rng.standard_normal(8)
    y = lu.btran(c)
    np.testing.assert_allclose(y, np.linalg.solve(A.T, c), rtol=1e-10, atol=1e-12)


def test_sparse_ftran_btran_residual():
    sp = pytest.importorskip("scipy.sparse")
    n = 120
    S = (sp.random(n, n, density=0.03, random_state=3) + sp.eye(n) * 6).tocsc()
    lm = feral.LuMatrix(
        n,
        S.indptr.astype(np.int64),
        S.indices.astype(np.int64),
        S.data.astype(np.float64),
    )
    lu = feral.LuFactor(lm, force_dense=False)
    assert not lu.is_dense
    rng = np.random.default_rng(4)
    b = rng.standard_normal(n)
    x = lu.ftran(b)
    # Oracle: dense solve of the same matrix.
    np.testing.assert_allclose(x, np.linalg.solve(S.toarray(), b), rtol=1e-8, atol=1e-9)
    assert lu.factor_nnz is not None and lu.factor_nnz > 0


def test_auto_routing_matches_forced():
    # Small dense problem auto-routes to dense; both engines must agree.
    A = _dense_basis(5, 7)
    auto = feral.LuFactor(feral.LuMatrix.from_dense(A))
    forced_sparse = feral.LuFactor(feral.LuMatrix.from_dense(A), force_dense=False)
    rng = np.random.default_rng(8)
    b = rng.standard_normal(5)
    np.testing.assert_allclose(auto.ftran(b), forced_sparse.ftran(b), rtol=1e-10, atol=1e-12)


def test_perm_qcol_reconstruct_plu():
    A = _dense_basis(6, 11)
    lu = feral.LuFactor(feral.LuMatrix.from_dense(A), force_dense=True)
    L = lu.l_array()
    U = lu.u_array()
    p = lu.perm
    q = lu.qcol
    # P A Q = L U  →  A[p][:, q] == L @ U
    lhs = A[np.ix_(p, q)]
    np.testing.assert_allclose(lhs, L @ U, rtol=1e-9, atol=1e-11)


def test_update_keeps_solve_correct_dense():
    A = _dense_basis(6, 2)
    lu = feral.LuFactor(feral.LuMatrix.from_dense(A), force_dense=True)
    assert lu.updates_since_refactor == 0
    rng = np.random.default_rng(5)
    new_col = rng.standard_normal(6) + 6 * np.eye(6)[:, 2]
    A2 = A.copy()
    A2[:, 2] = new_col
    lu.update(2, new_col)
    assert lu.updates_since_refactor == 1
    b = rng.standard_normal(6)
    x = lu.ftran(b)
    np.testing.assert_allclose(x, np.linalg.solve(A2, b), rtol=1e-8, atol=1e-10)


def test_refactor_resets_update_count():
    A = _dense_basis(5, 9)
    lu = feral.LuFactor(feral.LuMatrix.from_dense(A), force_dense=True)
    rng = np.random.default_rng(10)
    lu.update(0, rng.standard_normal(5) + 5 * np.eye(5)[:, 0])
    assert lu.updates_since_refactor == 1
    lu.refactor(feral.LuMatrix.from_dense(A))
    assert lu.updates_since_refactor == 0


def test_singular_basis_raises():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
    with pytest.raises(feral.SingularBasisError):
        feral.LuFactor(feral.LuMatrix.from_dense(A), force_dense=True)


def test_singular_basis_is_factor_error():
    # Backward-compat: the new leaf is a FactorError subclass.
    A = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(feral.FactorError):
        feral.LuFactor(feral.LuMatrix.from_dense(A), force_dense=False)


def test_update_sparse_rejected_on_dense():
    A = _dense_basis(4, 1)
    lu = feral.LuFactor(feral.LuMatrix.from_dense(A), force_dense=True)
    with pytest.raises(ValueError):
        lu.update_sparse(0, np.array([0, 1], dtype=np.int64), np.array([1.0, 2.0]))


def test_matvec_roundtrip():
    A = _dense_basis(5, 6)
    lm = feral.LuMatrix.from_dense(A)
    rng = np.random.default_rng(12)
    x = rng.standard_normal(5)
    np.testing.assert_allclose(lm.matvec(x), A @ x, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(lm.matvec_transpose(x), A.T @ x, rtol=1e-12, atol=1e-13)
