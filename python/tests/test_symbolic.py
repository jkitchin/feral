"""Symbolic-analysis (`analyze` + `SymbolicAnalysis`) tests."""

from __future__ import annotations

import numpy as np
import pytest

import feral


def _spd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


def test_analyze_perm_is_permutation():
    A = _spd(10, 0)
    sym = feral.analyze(feral.CscMatrix.from_dense(A))
    perm = np.sort(sym.perm)
    np.testing.assert_array_equal(perm, np.arange(sym.n))
    # perm_inv is the inverse of perm.
    p = sym.perm
    pinv = sym.perm_inv
    assert np.array_equal(np.asarray(pinv)[p], np.arange(sym.n))


def test_etree_is_valid_forest():
    A = _spd(12, 1)
    sym = feral.analyze(feral.CscMatrix.from_dense(A))
    parent = sym.etree_parent
    root = sym.root_sentinel
    # Each non-root parent points to a strictly later node (etree
    # invariant parent[j] > j), and at least one root exists.
    n_roots = 0
    for j in range(sym.n):
        if parent[j] == root:
            n_roots += 1
        else:
            assert parent[j] > j
    assert n_roots >= 1


def test_nnz_estimate_bounds_actual():
    A = _spd(15, 2)
    csc = feral.CscMatrix.from_dense(A)
    sym = feral.analyze(csc)
    s = feral.Solver()
    s.factor(csc)
    # The symbolic prediction is a (slack-inflated) upper bound on the
    # realized factor nnz.
    assert sym.factor_nnz_estimate >= s.factor_nnz


def test_symbolic_only_needs_no_factor():
    # analyze() does no numeric work — it returns a structure with no
    # dependence on a Solver factor.
    A = _spd(8, 3)
    sym = feral.analyze(feral.CscMatrix.from_dense(A), ordering="amd")
    assert sym.num_supernodes >= 1
    assert sym.ordering == "amd"
    assert sym.col_counts.shape == (sym.n,)


def test_solver_symbolic_matches_analyze_dimensions():
    A = _spd(9, 4)
    csc = feral.CscMatrix.from_dense(A)
    s = feral.Solver(ordering="amd")
    s.factor(csc)
    sym_solver = s.symbolic()
    sym_free = feral.analyze(csc, ordering="amd")
    assert sym_solver is not None
    assert sym_solver.n == sym_free.n
    assert sym_solver.num_supernodes == sym_free.num_supernodes
    assert sym_solver.ordering == "amd"


def test_bad_ordering_raises():
    A = _spd(5, 5)
    with pytest.raises(ValueError):
        feral.analyze(feral.CscMatrix.from_dense(A), ordering="nope")
