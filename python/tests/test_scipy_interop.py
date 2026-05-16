"""scipy.sparse interop tests."""

from __future__ import annotations

import numpy as np
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")
spsolve = pytest.importorskip("scipy.sparse.linalg").spsolve

import feral  # noqa: E402


def random_symmetric_indefinite(n: int, density: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = scipy_sparse.random(n, n, density=density, format="csc", random_state=rng).toarray()
    A = (A + A.T) / 2.0
    # Force a saddle structure: first n//2 positive eigenvalues,
    # second half negative. Easier: add a diagonal that flips sign.
    diag_shift = np.zeros(n)
    diag_shift[: n // 2] = 5.0
    diag_shift[n // 2 :] = -5.0
    A += np.diag(diag_shift)
    return scipy_sparse.csc_matrix(A)


def test_from_scipy_full():
    A_sp = random_symmetric_indefinite(20)
    A = feral.from_scipy(A_sp, symmetric="full")
    assert A.n == 20


def test_solve_matches_scipy_dense():
    rng = np.random.default_rng(1)
    n = 30
    A_sp = random_symmetric_indefinite(n, density=0.25, seed=2)
    A = feral.from_scipy(A_sp, symmetric="full")
    b = rng.standard_normal(n)

    solver = feral.Solver()
    status, _ = solver.factor(A)
    assert status == feral.FactorStatus.SUCCESS
    x_feral = solver.solve_refined(A, b)

    # scipy reference solve via dense
    x_ref = np.linalg.solve(A_sp.toarray(), b)
    assert np.allclose(x_feral, x_ref, atol=1e-9, rtol=1e-9)


def test_to_scipy_roundtrip():
    A_sp = random_symmetric_indefinite(15, density=0.3, seed=3)
    A = feral.from_scipy(A_sp, symmetric="full")
    A_back = feral.to_scipy(A)
    assert np.allclose(A_back.toarray(), A_sp.toarray())


def test_from_scipy_lower_triangle():
    A = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 2.0, 4.0]])
    A_sp = scipy_sparse.csc_matrix(A)
    M = feral.from_scipy(A_sp, symmetric="lower")
    assert M.nnz == 5

    # Solve with the symmetrized matrix
    solver = feral.Solver()
    solver.factor(M)
    b = np.array([1.0, 2.0, 3.0])
    x = solver.solve(b)
    A_sym = A + A.T - np.diag(np.diag(A))
    assert np.allclose(A_sym @ x, b, atol=1e-10)
