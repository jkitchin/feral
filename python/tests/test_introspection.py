"""Introspection tests: knobs, counters, factor stats, profiles."""

from __future__ import annotations

import numpy as np
import pytest

import feral


def _indef(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M + M.T
    # Push to a definite-ish but indefinite spectrum with a few negatives.
    A += np.diag(rng.standard_normal(n))
    return A


def _spd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


def test_ordering_invariant_inertia():
    # Different fill-reducing orderings must produce the same inertia.
    A = _indef(14, 0)
    csc = feral.CscMatrix.from_dense(A)
    inertias = []
    for ordering in ("auto", "amd", "amf"):
        s = feral.Solver(ordering=ordering)
        _, inertia = s.factor(csc)
        inertias.append(inertia.as_tuple())
    assert len(set(inertias)) == 1


def test_pivot_magnitudes_present():
    A = _spd(10, 1)
    s = feral.Solver()
    assert s.min_pivot_magnitude is None  # no factor yet
    s.factor(feral.CscMatrix.from_dense(A))
    lo = s.min_pivot_magnitude
    hi = s.max_pivot_magnitude
    assert lo is not None and hi is not None
    assert 0.0 < lo <= hi


def test_mc64_counters_are_ints():
    A = _spd(8, 2)
    s = feral.Solver(scaling="mc64")
    s.factor(feral.CscMatrix.from_dense(A))
    for c in (
        s.mc64_fallback_count,
        s.mc64_scaling_fallback_count,
        s.mc64_retry_attempt_count,
        s.mc64_cache_hit_count,
    ):
        assert isinstance(c, int)
        assert c >= 0


def test_last_factor_stats_sane():
    A = _spd(12, 3)
    csc = feral.CscMatrix.from_dense(A)
    s = feral.Solver()
    assert s.last_factor_stats() is None
    s.factor(csc)
    fs = s.last_factor_stats()
    assert fs is not None
    assert fs.nnz_a == csc.nnz
    assert fs.nnz_l >= fs.nnz_a
    assert fs.fill_ratio >= 1.0
    assert isinstance(fs.inertia, feral.Inertia)
    assert 0.0 < fs.min_abs_pivot <= fs.max_abs_pivot


def test_scaling_info_kind():
    A = _spd(6, 4)
    s = feral.Solver(scaling="infnorm")
    assert s.scaling_info is None
    s.factor(feral.CscMatrix.from_dense(A))
    si = s.scaling_info
    assert si is not None
    assert si.kind in ("applied", "partial_singular", "mc64_fallback_to_infnorm", "not_applied")


def test_profile_report_requires_profiling():
    A = _spd(40, 5)
    csc = feral.CscMatrix.from_dense(A)
    s_off = feral.Solver(profiling=False)
    s_off.factor(csc)
    assert s_off.profile_report() is None

    s_on = feral.Solver(profiling=True)
    s_on.factor(csc)
    pr = s_on.profile_report()
    assert pr is not None
    assert pr.total_us >= 0
    # Symbolic profile is populated on a fresh (cache-miss) analysis.
    spr = s_on.symbolic_profile_report()
    assert spr is not None
    assert spr.total_us >= 0


def test_invalidate_factors():
    A = _spd(6, 6)
    s = feral.Solver()
    s.factor(feral.CscMatrix.from_dense(A))
    assert s.factors() is not None
    s.invalidate_factors()
    assert s.factors() is None


def test_invalidate_symbolic_forces_reanalysis():
    A = _spd(6, 7)
    csc = feral.CscMatrix.from_dense(A)
    s = feral.Solver()
    s.factor(csc)
    assert s.symbolic_call_count == 1
    # A pattern-identical refactor reuses the cached symbolic.
    s.refactor(csc)
    assert s.symbolic_call_count == 1
    # After invalidation the next factor re-runs symbolic analysis.
    s.invalidate_symbolic_cache()
    s.factor(csc)
    assert s.symbolic_call_count == 2
