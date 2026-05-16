"""SQD (symmetric quasi-definite) fast-path bindings.

Mirrors the Rust-side `tests/sqd_fast_path.rs` coverage at the Python
surface: the `sqd_mode=True` opt-in produces a correct factor on a
true SQD KKT, the `sqd_mode` getter reflects the configured value,
and contract violations surface as :class:`feral.SqdContractViolated`
(never a silent BK fallback). See `dev/research/sqd-fast-path-2026-05-16.md`
and Vanderbei 1995 Theorem 2.1.
"""

from __future__ import annotations

import numpy as np
import pytest

import feral


def test_sqd_mode_kwarg_and_getter():
    s = feral.Solver(sqd_mode=True)
    assert s.sqd_mode is True
    assert "sqd_mode=true" in repr(s)

    default_solver = feral.Solver()
    assert default_solver.sqd_mode is False


def test_sqd_mode_factor_kkt_4x4():
    # K = [[-1 0 1 0], [0 -2 1 1], [1 1 1 0], [0 1 0 1]] -- true SQD,
    # inertia (2, 2, 0). Same fixture as the Rust phase-(f) parity test.
    K = np.array(
        [
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -2.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    M = feral.CscMatrix.from_dense(K)
    s = feral.Solver(sqd_mode=True)
    status, inertia = s.factor(M)
    assert status == int(feral.FactorStatus.SUCCESS)
    assert inertia.as_tuple() == (2, 2, 0)

    b = np.array([1.0, 2.0, 3.0, 4.0])
    x = s.solve(b)
    assert np.max(np.abs(K @ x - b)) < 1e-12


def test_sqd_contract_violation_raises():
    # Near-zero diagonal pivot trips the SQD contract guard. The
    # solver must raise SqdContractViolated rather than silently
    # falling back to BK pivoting (the whole point of opt-in SQD).
    A_bad = np.array([[1e-20, 1.0], [1.0, 1e-20]])
    M = feral.CscMatrix.from_dense(A_bad)
    # scaling='none' so equilibration can't rescale the tiny pivot
    # away and hide the bug we want to test.
    s = feral.Solver(sqd_mode=True, scaling="none")
    with pytest.raises(feral.SqdContractViolated) as exc_info:
        s.factor(M)
    assert "column" in str(exc_info.value)


def test_sqd_exception_is_factor_error():
    # SqdContractViolated must inherit FactorError so existing
    # `except feral.FactorError:` blocks catch it.
    assert issubclass(feral.SqdContractViolated, feral.FactorError)
    assert issubclass(feral.SqdContractViolated, feral.FeralError)


def test_sqd_symbolic_cache_reuse_across_refactor():
    K = np.array(
        [
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -2.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    M = feral.CscMatrix.from_dense(K)
    s = feral.Solver(sqd_mode=True)
    s.factor(M)
    assert s.symbolic_call_count == 1
    # Refactor with the same pattern must not re-analyse.
    s.refactor(M)
    assert s.symbolic_call_count == 1
