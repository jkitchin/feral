"""Basic factor / solve / inertia checks."""

from __future__ import annotations

import numpy as np
import pytest

import feral


def spd_3x3() -> feral.CscMatrix:
    return feral.CscMatrix.from_dense(
        np.array(
            [
                [4.0, 1.0, 0.0],
                [1.0, 3.0, 2.0],
                [0.0, 2.0, 5.0],
            ]
        )
    )


def saddle_point_2x2() -> feral.CscMatrix:
    """[[1, 1], [1, -1]] has inertia (1, 1, 0)."""
    return feral.CscMatrix.from_dense(np.array([[1.0, 1.0], [1.0, -1.0]]))


def test_csc_construction_from_dense():
    A = spd_3x3()
    assert A.n == 3
    assert A.nnz == 5  # lower triangle: (0,0), (1,0), (1,1), (2,1), (2,2)


def test_csc_construction_from_triplet():
    rows = np.array([0, 1, 1, 2, 2], dtype=np.int64)
    cols = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    vals = np.array([4.0, 1.0, 3.0, 2.0, 5.0])
    A = feral.CscMatrix.from_triplet(3, rows, cols, vals)
    assert A.n == 3
    assert A.nnz == 5


def test_csc_rejects_upper_triangle_triplet():
    with pytest.raises(ValueError):
        feral.CscMatrix.from_triplet(
            3,
            np.array([0, 0], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),  # row 0 < col 1 → upper
            np.array([1.0, 1.0]),
        )


def test_factor_solve_spd():
    A = spd_3x3()
    solver = feral.Solver()
    status, inertia = solver.factor(A)
    assert status == feral.FactorStatus.SUCCESS
    assert inertia == feral.Inertia(3, 0, 0)
    b = np.array([1.0, 2.0, 3.0])
    x = solver.solve(b)
    assert A.relative_residual(x, b) < 1e-12


def test_factor_inertia_saddle_point():
    A = saddle_point_2x2()
    solver = feral.Solver()
    status, inertia = solver.factor(A)
    assert status == feral.FactorStatus.SUCCESS
    assert inertia.n_pos == 1
    assert inertia.n_neg == 1
    assert inertia.n_zero == 0


def test_factor_expected_inertia_match():
    A = spd_3x3()
    solver = feral.Solver()
    status, _ = solver.factor(A, expected_inertia=feral.Inertia(3, 0, 0))
    assert status == feral.FactorStatus.SUCCESS


def test_factor_expected_inertia_mismatch_does_not_invalidate_solve():
    A = spd_3x3()
    solver = feral.Solver()
    status, inertia = solver.factor(
        A, expected_inertia=feral.Inertia(2, 1, 0)
    )
    assert status == feral.FactorStatus.WRONG_INERTIA
    assert inertia == feral.Inertia(3, 0, 0)
    # Factor is still stored; solve proceeds.
    b = np.array([1.0, 2.0, 3.0])
    x = solver.solve(b)
    assert A.relative_residual(x, b) < 1e-12


def test_symbolic_reuse_across_refactor():
    A = spd_3x3()
    solver = feral.Solver()
    solver.factor(A)
    assert solver.symbolic_call_count == 1
    # Same pattern, new values
    A2 = feral.CscMatrix.from_dense(
        np.array(
            [
                [5.0, 1.0, 0.0],
                [1.0, 4.0, 2.0],
                [0.0, 2.0, 6.0],
            ]
        )
    )
    status, _ = solver.refactor(A2)
    assert status == feral.FactorStatus.SUCCESS
    assert solver.symbolic_call_count == 1


def test_refactor_pattern_drift_raises():
    A = spd_3x3()
    solver = feral.Solver()
    solver.factor(A)
    # Different sparsity pattern
    A2 = feral.CscMatrix.from_dense(np.eye(3))
    with pytest.raises(feral.PatternMismatch):
        solver.refactor(A2)


def test_multi_rhs_solve():
    A = spd_3x3()
    solver = feral.Solver()
    solver.factor(A)
    B = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    X = solver.solve(B)
    assert X.shape == (3, 2)
    for j in range(2):
        b = B[:, j]
        x = X[:, j]
        assert A.relative_residual(x, b) < 1e-12


def test_solve_refined_recovers_residual():
    A = spd_3x3()
    solver = feral.Solver()
    solver.factor(A)
    b = np.array([1.0, 2.0, 3.0])
    x = solver.solve_refined(A, b)
    assert A.relative_residual(x, b) < 1e-13


def test_solve_without_factor_raises():
    solver = feral.Solver()
    with pytest.raises(feral.SolveError):
        solver.solve(np.array([1.0, 2.0, 3.0]))


def test_inertia_dataclass_arithmetic():
    i = feral.Inertia(3, 2, 1)
    assert i.n == 6
    assert i.matches(feral.Inertia(3, 2, 1))
    assert not i.matches(feral.Inertia(3, 2, 0))
    assert tuple(i) == (3, 2, 1)


def test_estimate_condition():
    A = spd_3x3()
    solver = feral.Solver()
    solver.factor(A)
    cond = solver.estimate_condition_1norm(A)
    assert cond > 1.0
    assert np.isfinite(cond)


def test_repr():
    A = spd_3x3()
    assert "n=3" in repr(A)
    solver = feral.Solver()
    assert "Solver(" in repr(solver)
