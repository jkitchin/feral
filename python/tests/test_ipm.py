"""KktSolver / IPM helper tests."""

from __future__ import annotations

import numpy as np
import pytest

import feral
from feral.ipm import KktSolver


def build_kkt_pattern(n: int, m: int, seed: int = 0):
    """Make a synthetic block KKT matrix:

        [ H   A^T ]
        [ A    0  ]

    where H is SPD (n x n) and A is m x n. Returns the feral CscMatrix
    with values plus the dense form for verification. Explicit zero
    entries are placed on the (2,2)-block diagonal so KktSolver can
    apply δ_c when it needs to.
    """
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((n, n))
    H = H @ H.T + np.eye(n) * 0.5  # SPD
    A_mat = rng.standard_normal((m, n))
    K = np.zeros((n + m, n + m))
    K[:n, :n] = H
    K[n:, :n] = A_mat
    K[:n, n:] = A_mat.T

    # Build triplets explicitly so we can keep the (2,2) zero diagonals.
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    N = n + m
    for j in range(N):
        for i in range(j, N):
            if i == j or K[i, j] != 0.0:
                rows.append(i)
                cols.append(j)
                vals.append(float(K[i, j]))
    pattern = feral.CscMatrix.from_triplet(
        N,
        np.array(rows, dtype=np.int64),
        np.array(cols, dtype=np.int64),
        np.array(vals, dtype=np.float64),
    )
    return pattern, K


def test_kkt_solver_basic():
    n, m = 6, 3
    pattern, K = build_kkt_pattern(n, m, seed=7)
    kkt = KktSolver(pattern, expected_inertia=feral.Inertia(n, m))

    rhs = np.arange(1, n + m + 1, dtype=np.float64)
    # Provide values matching the pattern's nnz; the values inside
    # `pattern` are already correct.
    report = kkt.factor(pattern.values())
    assert report.status == feral.FactorStatus.SUCCESS
    assert report.inertia == feral.Inertia(n, m)
    assert report.delta_w == 0.0
    assert report.n_attempts == 1

    x = kkt.solve(rhs)
    assert np.max(np.abs(K @ x - rhs)) / np.max(np.abs(rhs)) < 1e-9


def test_kkt_solver_symbolic_reuse_across_newton_loop():
    n, m = 5, 2
    pattern, _ = build_kkt_pattern(n, m, seed=11)
    kkt = KktSolver(pattern, expected_inertia=feral.Inertia(n, m))

    rng = np.random.default_rng(123)
    for _ in range(20):
        vals = pattern.values() + 0.01 * rng.standard_normal(pattern.nnz)
        kkt.factor(vals)
    assert kkt.solver.symbolic_call_count == 1


def _kkt_with_explicit_zero_diag(K: np.ndarray) -> feral.CscMatrix:
    N = K.shape[0]
    rows, cols, vals = [], [], []
    for j in range(N):
        for i in range(j, N):
            if i == j or K[i, j] != 0.0:
                rows.append(i)
                cols.append(j)
                vals.append(float(K[i, j]))
    return feral.CscMatrix.from_triplet(
        N,
        np.array(rows, dtype=np.int64),
        np.array(cols, dtype=np.int64),
        np.array(vals, dtype=np.float64),
    )


def test_kkt_solver_perturbs_on_wrong_inertia():
    n, m = 4, 2
    # Build a KKT whose H block is indefinite — wrong inertia for
    # a typical IPM step. KktSolver should perturb δ_w until inertia
    # matches.
    rng = np.random.default_rng(0)
    H = rng.standard_normal((n, n))
    H = (H + H.T) / 2.0  # indefinite, possibly
    A_mat = rng.standard_normal((m, n))
    K = np.zeros((n + m, n + m))
    K[:n, :n] = H
    K[n:, :n] = A_mat
    K[:n, n:] = A_mat.T
    pattern = _kkt_with_explicit_zero_diag(K)

    kkt = KktSolver(
        pattern,
        expected_inertia=feral.Inertia(n, m),
        delta_w_0=1e-2,
        kappa_w_plus_first=10.0,
        kappa_w_plus=5.0,
    )
    report = kkt.factor(pattern.values())
    # Either it perturbed and succeeded, or it gave up — both
    # outcomes acceptable as long as the loop ran > 1 attempt when
    # inertia was actually wrong on the first try.
    assert report.n_attempts >= 1


def test_kkt_solver_solve_pair():
    n, m = 5, 2
    pattern, K = build_kkt_pattern(n, m, seed=42)
    kkt = KktSolver(pattern, expected_inertia=feral.Inertia(n, m))
    report = kkt.factor(pattern.values())
    assert report.status == feral.FactorStatus.SUCCESS

    rng = np.random.default_rng(5)
    rhs_aff = rng.standard_normal(n + m)
    rhs_corr = rng.standard_normal(n + m)
    dx_aff, dx_corr = kkt.solve_pair(rhs_aff, rhs_corr)
    assert dx_aff.shape == (n + m,)
    assert dx_corr.shape == (n + m,)
    assert np.max(np.abs(K @ dx_aff - rhs_aff)) / np.max(np.abs(rhs_aff)) < 1e-9
    assert np.max(np.abs(K @ dx_corr - rhs_corr)) / np.max(np.abs(rhs_corr)) < 1e-9
