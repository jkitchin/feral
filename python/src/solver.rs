//! `Solver`: the stateful sparse symmetric indefinite direct solver
//! (LDLᵀ with Bunch-Kaufman pivoting).

use feral::error::FeralError as RustFeralError;
use feral::inertia::Inertia as RustInertia;
use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus as RustFactorStatus, Solver as RustSolver};
use feral::scaling::ScalingStrategy;
use feral::sparse::csc::CscMatrix as RustCscMatrix;
use feral::symbolic::SupernodeParams;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::{
    array1_to_vec, ordering_from_str, ordering_to_str, quality_to_int, Inertia,
    STATUS_INTERRUPTED, STATUS_SINGULAR, STATUS_SUCCESS, STATUS_WRONG_INERTIA,
};
use crate::errors::{
    map_feral_err, DelayBudgetExceeded, NumericFailure, PatternMismatch, SingularError,
    SqdContractViolated,
};
use crate::factors::Factors;
use crate::introspect::{FactorStats, ProfileReport, ScalingInfo, SymbolicProfileReport};
use crate::matrix::CscMatrix;
use crate::symbolic::SymbolicAnalysis;

pub(crate) fn pick_scaling(name: &str) -> PyResult<ScalingStrategy> {
    match name {
        "auto" | "default" => Ok(ScalingStrategy::default()),
        "none" | "identity" => Ok(ScalingStrategy::Identity),
        "infnorm" | "inf_norm" | "equilibration" => Ok(ScalingStrategy::InfNorm),
        "mc64" | "mc64_symmetric" => Ok(ScalingStrategy::Mc64Symmetric),
        other => Err(PyValueError::new_err(format!(
            "unknown scaling '{other}'; valid options: auto, none, infnorm, mc64"
        ))),
    }
}

/// Stateful sparse symmetric indefinite direct solver.
///
/// Mirrors `feral::numeric::solver::Solver`. The solver owns its
/// quality-escalation state, a cached symbolic factorization (reused
/// across `factor`/`refactor` calls on matrices with the same sparsity
/// pattern — the IPM use case), and a rayon `ThreadPool` for the
/// parallel multifrontal driver.
///
/// Not thread-safe across concurrent `factor`/`solve` from multiple
/// Python threads; use one `Solver` per thread.
#[pyclass(module = "feral._feral", unsendable)]
pub struct Solver {
    pub(crate) inner: RustSolver,
    last_pattern: Option<(usize, usize, u64)>,
}

impl Solver {
    fn pattern_signature(m: &RustCscMatrix) -> (usize, usize, u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        m.col_ptr.hash(&mut h);
        m.row_idx.hash(&mut h);
        (m.n, m.row_idx.len(), h.finish())
    }
}

#[pymethods]
impl Solver {
    /// Construct a new solver with optional configuration.
    ///
    /// - `parallel`: dispatch the rayon-parallel multifrontal driver
    ///   when work warrants it. Default `True`.
    /// - `fma`: opt-in FMA dispatch on dense kernels. Default `False`.
    /// - `static_pivoting`: force-accept failing pivots in place rather
    ///   than delaying up the elimination tree. Default `False`.
    /// - `cascade_break_ratio`, `cascade_break_eps`: opt-in cascade-break
    ///   knobs. Both default `None` (off). See
    ///   `dev/research/cascade-break-l-perturbation-2026-05-15.md`.
    /// - `scaling`: one of `"auto"`, `"none"`, `"infnorm"`, `"mc64"`.
    ///   Default `"auto"`.
    /// - `pivot_threshold`: BK column-relative pivot threshold. Default
    ///   uses `NumericParams::default()` (MA27-style 1e-8).
    /// - `sqd_mode`: opt in to the symmetric-quasi-definite fast path
    ///   (Vanderbei 1995). Skips the BK 1x1-vs-2x2 pivot search; the
    ///   factor either completes under a stated diagonal-D contract or
    ///   raises :class:`SqdContractViolated` (loud, never a silent BK
    ///   fallback). Default `False`. See
    ///   `dev/research/sqd-fast-path-2026-05-16.md`.
    /// - `ordering`: fill-reducing ordering method — one of `"amd"`,
    ///   `"amf"`, `"metis"`, `"scotch"`, `"kahip"`, `"auto"`,
    ///   `"auto_race"`. Default `None` keeps the adaptive `Auto`
    ///   dispatch.
    /// - `mc64_cache`: reuse the MC64 matching computed at symbolic
    ///   time across refactors (the IPM use case). Default `None`
    ///   keeps the Rust default.
    /// - `profiling`: collect per-stage timing for `profile_report()`
    ///   and `symbolic_profile_report()`. Default `None` (off).
    /// - `partial_singular_warning`: emit a diagnostic when scaling
    ///   detects a structurally singular matrix. Default `None`.
    /// - `auto_cascade_break`: enable adaptive cascade-break with the
    ///   given `beta` threshold. Default `None` (off). Distinct from
    ///   the manual `cascade_break_ratio`/`cascade_break_eps` knobs.
    #[new]
    #[pyo3(signature = (
        *,
        parallel = true,
        fma = false,
        static_pivoting = false,
        cascade_break_ratio = None,
        cascade_break_eps = None,
        scaling = "auto",
        pivot_threshold = None,
        sqd_mode = false,
        ordering = None,
        mc64_cache = None,
        profiling = None,
        partial_singular_warning = None,
        auto_cascade_break = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        parallel: bool,
        fma: bool,
        static_pivoting: bool,
        cascade_break_ratio: Option<f64>,
        cascade_break_eps: Option<f64>,
        scaling: &str,
        pivot_threshold: Option<f64>,
        sqd_mode: bool,
        ordering: Option<&str>,
        mc64_cache: Option<bool>,
        profiling: Option<bool>,
        partial_singular_warning: Option<bool>,
        auto_cascade_break: Option<f64>,
    ) -> PyResult<Self> {
        let mut np = NumericParams::default();
        np.fma = fma;
        np.allow_delayed_pivots = !static_pivoting;
        np.cascade_break_ratio = cascade_break_ratio;
        np.cascade_break_eps = cascade_break_eps;
        np.scaling = pick_scaling(scaling)?;
        if let Some(pt) = pivot_threshold {
            np.bk.pivot_threshold = pt;
        }
        np.sqd_mode = sqd_mode;
        // Resolve the ordering string before constructing so a bad name
        // raises before any allocation.
        let ordering_method = ordering.map(ordering_from_str).transpose()?;
        let mut inner =
            RustSolver::with_params(np, SupernodeParams::default()).with_parallel(parallel);
        // The remaining knobs are only applied when explicitly set, so
        // the default constructor reproduces the prior behavior exactly.
        if let Some(m) = ordering_method {
            inner = inner.with_ordering(m);
        }
        if let Some(on) = mc64_cache {
            inner = inner.with_mc64_cache(on);
        }
        if let Some(on) = profiling {
            inner = inner.with_profiling(on);
        }
        if let Some(on) = partial_singular_warning {
            inner = inner.with_partial_singular_warning(on);
        }
        if let Some(beta) = auto_cascade_break {
            inner = inner.with_auto_cascade_break(beta);
        }
        Ok(Self {
            inner,
            last_pattern: None,
        })
    }

    /// Factor `A`. If `expected_inertia` is provided and disagrees with
    /// the actual inertia, returns `FactorStatus.WRONG_INERTIA` (the
    /// factor is still stored — `solve` will proceed). On fatal errors
    /// raises `NumericFailure`.
    #[pyo3(signature = (a, *, expected_inertia = None))]
    fn factor(
        &mut self,
        py: Python<'_>,
        a: &CscMatrix,
        expected_inertia: Option<&Inertia>,
    ) -> PyResult<(i32, Option<Inertia>)> {
        let expected_rust = expected_inertia.map(RustInertia::from);
        let sig = Self::pattern_signature(a.inner());
        let status = py.allow_threads(|| self.inner.factor(a.inner(), expected_rust));
        self.last_pattern = Some(sig);
        match status {
            RustFactorStatus::Success => Ok((
                STATUS_SUCCESS,
                self.inner.inertia().cloned().map(Into::into),
            )),
            RustFactorStatus::Singular => Ok((
                STATUS_SINGULAR,
                self.inner.inertia().cloned().map(Into::into),
            )),
            RustFactorStatus::WrongInertia {
                actual,
                expected: _,
            } => Ok((STATUS_WRONG_INERTIA, Some(actual.into()))),
            // Issue #194: unreachable from Python — these bindings expose
            // no way to arm the interrupt flag — but returned as its own
            // code rather than folded into NUMERIC_FAILURE, so if an
            // arming API is added later nothing here has to change.
            RustFactorStatus::Interrupted => Ok((STATUS_INTERRUPTED, None)),
            RustFactorStatus::FatalError(e) => Err(match e {
                RustFeralError::NumericallyRankDeficient => SingularError::new_err(format!("{e}")),
                RustFeralError::InvalidInput(s) => PyValueError::new_err(s),
                RustFeralError::SqdContractViolated { .. } => {
                    SqdContractViolated::new_err(format!("{e}"))
                }
                RustFeralError::DelayBudgetExceeded { .. } => {
                    DelayBudgetExceeded::new_err(format!("{e}"))
                }
                other => NumericFailure::new_err(format!("{other}")),
            }),
        }
    }

    /// Re-factor with new values on the same sparsity pattern. Raises
    /// `PatternMismatch` if the pattern differs from the previous
    /// `factor`/`refactor` call. The symbolic factorization is reused.
    #[pyo3(signature = (a, *, expected_inertia = None))]
    fn refactor(
        &mut self,
        py: Python<'_>,
        a: &CscMatrix,
        expected_inertia: Option<&Inertia>,
    ) -> PyResult<(i32, Option<Inertia>)> {
        let new_sig = Self::pattern_signature(a.inner());
        if let Some(old) = self.last_pattern {
            if old != new_sig {
                return Err(PatternMismatch::new_err(
                    "refactor called with a different sparsity pattern; \
                     use factor() instead, or build a new CscMatrix with the same pattern",
                ));
            }
        }
        self.factor(py, a, expected_inertia)
    }

    /// Solve `A · x = b` against the stored factor. `b` may be a 1-D
    /// array of length `n` or a 2-D `(n, nrhs)` array (one column per
    /// RHS). Returns a numpy array of the same shape as `b`.
    fn solve<'py>(&self, py: Python<'py>, b: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        // 2-D path
        if let Ok(arr2) = b.extract::<PyReadonlyArray2<'py, f64>>() {
            let view = arr2.as_array();
            let shape = view.shape();
            let n = shape[0];
            let nrhs = shape[1];
            // Pack column-major
            let mut buf = vec![0.0f64; n * nrhs];
            for j in 0..nrhs {
                for i in 0..n {
                    buf[j * n + i] = view[[i, j]];
                }
            }
            let out = py
                .allow_threads(|| self.inner.solve_many(&buf, nrhs))
                .map_err(map_feral_err)?;
            // Reshape column-major to (n, nrhs) row-major numpy
            let mut np_out = vec![0.0f64; n * nrhs];
            for j in 0..nrhs {
                for i in 0..n {
                    np_out[i * nrhs + j] = out[j * n + i];
                }
            }
            let arr = PyArray2::from_vec2_bound(
                py,
                &(0..n)
                    .map(|i| np_out[i * nrhs..(i + 1) * nrhs].to_vec())
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(arr.into_py(py));
        }
        // 1-D path
        let arr1: PyReadonlyArray1<'py, f64> = b.extract()?;
        let bs_vec = array1_to_vec(&arr1);
        let bs = bs_vec.as_slice();
        let x = py
            .allow_threads(|| self.inner.solve(bs))
            .map_err(map_feral_err)?;
        Ok(x.into_pyarray_bound(py).into_py(py))
    }

    /// Solve with iterative refinement against `a` and the stored
    /// factor. Same shape conventions as `solve`. Default `max_iter`
    /// and `tol` mirror the Rust defaults.
    #[pyo3(signature = (a, b))]
    fn solve_refined<'py>(
        &self,
        py: Python<'py>,
        a: &CscMatrix,
        b: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        if let Ok(arr2) = b.extract::<PyReadonlyArray2<'py, f64>>() {
            let view = arr2.as_array();
            let shape = view.shape();
            let n = shape[0];
            let nrhs = shape[1];
            let mut buf = vec![0.0f64; n * nrhs];
            for j in 0..nrhs {
                for i in 0..n {
                    buf[j * n + i] = view[[i, j]];
                }
            }
            let out = py
                .allow_threads(|| self.inner.solve_many_refined(a.inner(), &buf, nrhs))
                .map_err(map_feral_err)?;
            let mut np_out = vec![0.0f64; n * nrhs];
            for j in 0..nrhs {
                for i in 0..n {
                    np_out[i * nrhs + j] = out[j * n + i];
                }
            }
            let arr = PyArray2::from_vec2_bound(
                py,
                &(0..n)
                    .map(|i| np_out[i * nrhs..(i + 1) * nrhs].to_vec())
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(arr.into_py(py));
        }
        let arr1: PyReadonlyArray1<'py, f64> = b.extract()?;
        let bs_vec = array1_to_vec(&arr1);
        let bs = bs_vec.as_slice();
        let x = py
            .allow_threads(|| self.inner.solve_refined(a.inner(), bs))
            .map_err(map_feral_err)?;
        Ok(x.into_pyarray_bound(py).into_py(py))
    }

    /// Hager-Higham 1-norm condition estimate of `A`. Requires a
    /// stored factor — call `factor` first.
    fn estimate_condition_1norm(&self, py: Python<'_>, a: &CscMatrix) -> PyResult<f64> {
        py.allow_threads(|| self.inner.estimate_condition_1norm(a.inner()))
            .map_err(map_feral_err)
    }

    /// Two-stage quality escalation. Persists across `factor` calls
    /// until `reset_quality` reverts it. Returns `False` if both
    /// stages are exhausted.
    fn increase_quality(&mut self) -> bool {
        self.inner.increase_quality()
    }

    /// Revert every escalation applied by `increase_quality`, putting
    /// the solver back at `QualityLevel.BASELINE` with the scaling
    /// strategy and pivot threshold it was configured with.
    ///
    /// Returns `True` only if a parameter was actually restored to a
    /// different value; `False` both when the solver was already at
    /// baseline and when the escalation itself moved nothing. Either
    /// way the level is re-baselined and the ladder is armed again.
    ///
    /// Lets a caller bound the escalation's lifetime -- re-baselining
    /// at a major iteration, or on entering a restoration
    /// sub-problem -- instead of letting one hard factorization govern
    /// the rest of the solve (issue #192). The cached symbolic
    /// factorization is preserved.
    fn reset_quality(&mut self) -> bool {
        self.inner.reset_quality()
    }

    // ---- properties ----

    /// Inertia of the last successful factor, or `None`.
    #[getter]
    fn inertia(&self) -> Option<Inertia> {
        self.inner.inertia().cloned().map(Into::into)
    }

    /// Number of negative eigenvalues from the last factor. Returns
    /// `None` if no factor is stored (the Rust API panics; the Python
    /// binding returns `None` for safety).
    #[getter]
    fn num_negative_eigenvalues(&self) -> Option<usize> {
        self.inner.inertia().map(|i| i.negative)
    }

    /// Minimum eigenvalue of D over the last factor's pivots, or `None`.
    #[getter]
    fn min_diagonal(&self) -> Option<f64> {
        self.inner.min_diagonal()
    }

    /// Current quality-escalation level (matches `QualityLevel`
    /// IntEnum codes).
    #[getter]
    fn quality_level(&self) -> i32 {
        quality_to_int(self.inner.quality_level())
    }

    /// Always `True` for feral.
    #[getter]
    fn provides_inertia(&self) -> bool {
        self.inner.provides_inertia()
    }

    /// Total number of symbolic-analysis calls. Increments on the
    /// first `factor` and on every subsequent `factor` whose pattern
    /// differs from the cached one. Stays at 1 across pure IPM
    /// refactor loops.
    #[getter]
    fn symbolic_call_count(&self) -> usize {
        self.inner.symbolic_call_count()
    }

    /// True if the last factor's diagnostic flag flagged the result
    /// as benefiting from iterative refinement (e.g. cascade-break
    /// perturbations were applied).
    #[getter]
    fn needs_refinement(&self) -> bool {
        self.inner
            .factors()
            .map(|f| f.needs_refinement)
            .unwrap_or(false)
    }

    /// Current BK pivot threshold.
    #[getter]
    fn pivot_threshold(&self) -> f64 {
        self.inner.pivot_threshold()
    }

    /// Whether the parallel multifrontal driver is enabled.
    #[getter]
    fn parallel(&self) -> bool {
        self.inner.parallel()
    }

    /// Whether the SQD (symmetric quasi-definite) fast-path is enabled.
    #[getter]
    fn sqd_mode(&self) -> bool {
        self.inner.sqd_mode()
    }

    /// Total stored nonzeros in L + D after the last factor; `None`
    /// if no factor is stored.
    #[getter]
    fn factor_nnz(&self) -> Option<usize> {
        self.inner.factors().map(|f| f.factor_nnz())
    }

    /// String representation of the configured scaling strategy.
    #[getter]
    fn scaling(&self) -> String {
        format!("{:?}", self.inner.scaling_strategy())
    }

    /// Resolved fill-reducing ordering of the last factor (or cached
    /// symbolic analysis), e.g. `"amd"`, `"metis"`. `None` if nothing
    /// has been factored yet. Reflects the `Auto`-dispatch resolution,
    /// so this may differ from the `ordering` constructor argument.
    #[getter]
    fn ordering(&self) -> Option<&'static str> {
        self.inner
            .symbolic()
            .map(|s| &s.resolved_method)
            .or_else(|| self.inner.factors().map(|f| &f.resolved_method))
            .map(ordering_to_str)
    }

    /// Minimum absolute pivot magnitude over the last factor, or `None`.
    #[getter]
    fn min_pivot_magnitude(&self) -> Option<f64> {
        self.inner.min_pivot_magnitude()
    }

    /// Maximum absolute pivot magnitude over the last factor, or `None`.
    #[getter]
    fn max_pivot_magnitude(&self) -> Option<f64> {
        self.inner.max_pivot_magnitude()
    }

    /// Count of MC64 → inf-norm scaling fallbacks (matching failures).
    #[getter]
    fn mc64_fallback_count(&self) -> usize {
        self.inner.mc64_fallback_count()
    }

    /// Count of MC64 scaling-degenerate fallbacks.
    #[getter]
    fn mc64_scaling_fallback_count(&self) -> usize {
        self.inner.mc64_scaling_fallback_count()
    }

    /// Count of MC64 retry attempts.
    #[getter]
    fn mc64_retry_attempt_count(&self) -> usize {
        self.inner.mc64_retry_attempt_count()
    }

    /// Count of MC64-cache hits across refactors.
    #[getter]
    fn mc64_cache_hit_count(&self) -> usize {
        self.inner.mc64_cache_hit_count()
    }

    /// Outcome of the scaling stage of the last factor as a
    /// :class:`ScalingInfo`, or `None` if no factor is stored.
    #[getter]
    fn scaling_info(&self) -> Option<ScalingInfo> {
        self.inner.scaling_info().map(Into::into)
    }

    /// Snapshot of the numeric factor (assembled `L`/`D`, permutation,
    /// scaling) as a :class:`Factors`, or `None` if no factor is
    /// stored.
    fn factors(&self) -> Option<Factors> {
        self.inner.factors().map(Factors::snapshot)
    }

    /// Snapshot of the symbolic factorization (ordering, elimination
    /// tree, supernodes, nnz prediction) as a :class:`SymbolicAnalysis`,
    /// or `None` if no analysis is cached.
    fn symbolic(&self) -> Option<SymbolicAnalysis> {
        self.inner.symbolic().map(SymbolicAnalysis::snapshot)
    }

    /// Summary statistics of the most recent factorization as a
    /// :class:`FactorStats`, or `None` if no factor is stored.
    fn last_factor_stats(&self) -> Option<FactorStats> {
        self.inner.last_factor_stats().map(Into::into)
    }

    /// Per-stage timing of the most recent numeric factorization as a
    /// :class:`ProfileReport`. Returns `None` unless the solver was
    /// built with `profiling=True` and a factor is stored.
    fn profile_report(&self) -> Option<ProfileReport> {
        self.inner.profile_report().map(|r| (&r).into())
    }

    /// Per-stage timing of the most recent symbolic analysis as a
    /// :class:`SymbolicProfileReport`. Returns `None` unless the solver
    /// was built with `profiling=True`; also `None` on a symbolic-cache
    /// hit (no fresh analysis ran).
    fn symbolic_profile_report(&self) -> Option<SymbolicProfileReport> {
        self.inner.symbolic_profile_report().map(|r| (&r).into())
    }

    /// Drop the stored numeric factor, forcing the next `solve` to
    /// require a fresh `factor`. The cached symbolic factorization is
    /// retained.
    fn invalidate_factors(&mut self) {
        self.inner.invalidate_factors();
        // The numeric factor is gone, but the sparsity-pattern guard
        // can remain — `refactor` still wants to compare patterns.
    }

    /// Drop the cached symbolic factorization, forcing the next
    /// `factor` to re-run symbolic analysis even on an unchanged
    /// pattern.
    fn invalidate_symbolic_cache(&mut self) {
        self.inner.invalidate_symbolic_cache();
        self.last_pattern = None;
    }

    fn __repr__(&self) -> String {
        format!(
            "Solver(parallel={}, scaling={}, pivot_threshold={:.3e}, sqd_mode={})",
            self.inner.parallel(),
            format!("{:?}", self.inner.scaling_strategy()),
            self.inner.pivot_threshold(),
            self.inner.sqd_mode()
        )
    }

    fn __enter__<'py>(slf: PyRefMut<'py, Self>) -> PyRefMut<'py, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        // Drop and rebuild a fresh inner solver so the cached factor
        // and rayon pool are released deterministically. Preserve the
        // configured sqd_mode so the context manager's exit doesn't
        // silently reset the contract.
        let mut np = NumericParams::default();
        np.sqd_mode = self.inner.sqd_mode();
        let _ = std::mem::replace(
            &mut self.inner,
            RustSolver::with_params(np, SupernodeParams::default()),
        );
        self.last_pattern = None;
        false
    }
}
