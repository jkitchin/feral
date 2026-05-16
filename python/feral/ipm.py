"""Interior-point method helper layer.

:class:`KktSolver` wraps :class:`feral.Solver` with the Wächter–Biegler
2006 §3.1 perturbation-escalation loop that interior-point methods
need around an inertia-providing direct solver. Discrete-optimization
callers (discopt and friends) wire this into their Newton loop:

.. code-block:: python

    kkt = feral.ipm.KktSolver(pattern, expected_inertia=feral.Inertia(n, m))
    for k in range(max_iter):
        report = kkt.factor(values_this_iter)
        dx_aff, dx_corr = kkt.solve_pair(b_aff, b_corr)
        ...

The wrapped solver caches the symbolic factorization across all
refactor calls, so ``kkt.solver.symbolic_call_count`` stays at 1 over
the whole Newton run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import numpy as np

from . import (
    CscMatrix,
    FactorStatus,
    Inertia,
    PatternMismatch,
    Solver,
    WrongInertiaError,
)


@dataclass
class FactorReport:
    """Outcome of a single :meth:`KktSolver.factor` call."""

    status: FactorStatus
    inertia: Optional[Inertia]
    delta_w: float
    delta_c: float
    n_attempts: int
    factor_time_ms: float
    needs_refinement: bool


@dataclass
class KktSolver:
    """Inertia-corrected KKT factorization driver.

    Parameters
    ----------
    pattern : CscMatrix
        Initial KKT matrix carrying the sparsity pattern that will
        persist across Newton iterations. Its numeric values are used
        for the first :meth:`factor` call; subsequent calls overwrite
        them via :meth:`feral.CscMatrix.set_values`.
    expected_inertia : Inertia
        For an NLP with ``n`` primal variables and ``m`` equality
        constraints, use ``Inertia(n, m, 0)``.
    solver : Solver, optional
        Pre-configured solver; defaults to ``Solver()``.
    delta_w_min, delta_w_max, delta_w_0 : float
        (1,1)-block perturbation bounds and initial value. Defaults
        match Ipopt §3.1.
    kappa_w_plus_first, kappa_w_plus, kappa_w_minus : float
        Multipliers for escalating / de-escalating ``delta_w``. The
        first time it fires the escalation uses ``kappa_w_plus_first``;
        subsequent firings within the same iteration use ``kappa_w_plus``.
    delta_c : float
        Constraint regularization fired when the KKT is rank-deficient
        on the constraint block (zero pivots in the (2,2) corner).
    max_attempts : int
        Cap on perturbation retries per :meth:`factor` call.
    """

    pattern: CscMatrix
    expected_inertia: Inertia
    solver: Solver = field(default_factory=Solver)
    delta_w_min: float = 1e-20
    delta_w_max: float = 1e40
    delta_w_0: float = 1e-4
    kappa_w_plus_first: float = 100.0
    kappa_w_plus: float = 8.0
    kappa_w_minus: float = 1.0 / 3.0
    delta_c: float = 1e-8
    max_attempts: int = 8

    _last_delta_w: float = field(default=0.0, init=False)
    _diag_indices: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _constraint_diag_indices: Optional[np.ndarray] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # Cache positions of diagonal entries in self.pattern.values for
        # fast diagonal perturbation. CscMatrix stores lower triangle,
        # so the diagonal entry of column j is at the row in column j
        # whose row index equals j.
        n = self.pattern.n
        indptr = self.pattern.indptr()
        row_idx = self.pattern.row_idx()
        diag_pos = np.full(n, -1, dtype=np.int64)
        for j in range(n):
            s = int(indptr[j])
            e = int(indptr[j + 1])
            for k in range(s, e):
                if int(row_idx[k]) == j:
                    diag_pos[j] = k
                    break
        if (diag_pos < 0).any():
            missing = int(np.where(diag_pos < 0)[0][0])
            raise ValueError(
                f"pattern lacks an explicit diagonal entry at row/col {missing}; "
                "KktSolver needs every diagonal slot to apply δ_w and δ_c."
            )
        self._diag_indices = diag_pos
        # Constraint block diagonal indices: last m columns
        m = self.expected_inertia.n_neg
        self._constraint_diag_indices = diag_pos[n - m :].copy()

    def factor(self, values: np.ndarray) -> FactorReport:
        """Refactor the KKT with new values, applying the
        Wächter–Biegler perturbation escalation if inertia is wrong.

        ``values`` must have length ``self.pattern.nnz`` and must match
        the sparsity ordering of the pattern.
        """
        values = np.ascontiguousarray(values, dtype=np.float64)
        if values.shape != (self.pattern.nnz,):
            raise ValueError(
                f"values shape {values.shape} != (nnz={self.pattern.nnz},)"
            )
        original = self.pattern.values()  # copy
        delta_w = 0.0
        delta_c = 0.0
        first_perturb_this_call = True

        t0 = perf_counter()
        attempts = 0
        last_status: FactorStatus = FactorStatus.NUMERIC_FAILURE
        last_inertia: Optional[Inertia] = None
        while attempts < self.max_attempts:
            attempts += 1
            base = original if delta_w == 0.0 and delta_c == 0.0 else None
            if base is None:
                # Reconstruct values + diagonal perturbations.
                new_vals = original.copy()
                if delta_w != 0.0:
                    new_vals[self._diag_indices] += delta_w
                if delta_c != 0.0:
                    new_vals[self._constraint_diag_indices] -= delta_c
            else:
                new_vals = values
            self.pattern.set_values(new_vals)

            try:
                code, inertia = self.solver.refactor(
                    self.pattern, expected_inertia=self.expected_inertia
                )
            except PatternMismatch:
                # First call after construction — fall back to factor().
                code, inertia = self.solver.factor(
                    self.pattern, expected_inertia=self.expected_inertia
                )

            last_status = FactorStatus(code)
            last_inertia = inertia

            if last_status == FactorStatus.SUCCESS:
                break

            if last_status == FactorStatus.WRONG_INERTIA:
                # Escalate δ_w; if constraints are rank-deficient, kick δ_c.
                if (
                    inertia is not None
                    and inertia.n_zero > 0
                    and delta_c == 0.0
                ):
                    delta_c = self.delta_c
                    continue
                if delta_w == 0.0:
                    delta_w = max(self.delta_w_min, self._last_delta_w * self.kappa_w_minus)
                    if delta_w < self.delta_w_min:
                        delta_w = self.delta_w_0
                else:
                    factor = (
                        self.kappa_w_plus_first
                        if first_perturb_this_call
                        else self.kappa_w_plus
                    )
                    delta_w *= factor
                    first_perturb_this_call = False
                if delta_w > self.delta_w_max:
                    break
                continue

            if last_status == FactorStatus.SINGULAR:
                # Treat as the constraint-block-rank-deficient case;
                # kick δ_c and retry once, otherwise escalate δ_w.
                if delta_c == 0.0:
                    delta_c = self.delta_c
                    continue
                if delta_w == 0.0:
                    delta_w = self.delta_w_0
                else:
                    delta_w *= self.kappa_w_plus
                if delta_w > self.delta_w_max:
                    break
                continue

            break  # NUMERIC_FAILURE — give up

        wall_ms = (perf_counter() - t0) * 1000.0

        # Restore original values so caller sees the un-perturbed pattern.
        self.pattern.set_values(original)
        if delta_w > 0.0:
            self._last_delta_w = delta_w

        return FactorReport(
            status=last_status,
            inertia=last_inertia,
            delta_w=delta_w,
            delta_c=delta_c,
            n_attempts=attempts,
            factor_time_ms=wall_ms,
            needs_refinement=self.solver.needs_refinement,
        )

    def solve(self, rhs: np.ndarray, *, refine: bool = True) -> np.ndarray:
        """Solve ``KKT · x = rhs`` against the stored factor.

        ``rhs`` may be 1-D ``(n,)`` or 2-D ``(n, nrhs)``. With
        ``refine=True`` (default), iterative refinement runs against
        the unperturbed pattern values stored in ``self.pattern``.
        """
        if refine:
            return self.solver.solve_refined(self.pattern, rhs)
        return self.solver.solve(rhs)

    def solve_pair(
        self, rhs_aff: np.ndarray, rhs_corr: np.ndarray, *, refine: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mehrotra predictor-corrector solve: two RHS, one factor.

        Returns ``(dx_aff, dx_corr)``.
        """
        rhs_aff = np.asarray(rhs_aff, dtype=np.float64)
        rhs_corr = np.asarray(rhs_corr, dtype=np.float64)
        if rhs_aff.shape != rhs_corr.shape:
            raise ValueError(
                f"rhs_aff shape {rhs_aff.shape} != rhs_corr shape {rhs_corr.shape}"
            )
        stacked = np.column_stack([rhs_aff, rhs_corr])
        x = self.solve(stacked, refine=refine)
        return np.ascontiguousarray(x[:, 0]), np.ascontiguousarray(x[:, 1])
