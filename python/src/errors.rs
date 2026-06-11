//! Exception hierarchy and the `FeralError` → `PyErr` mapping.
//!
//! The hierarchy is rooted at `FeralError(PyException)`, with
//! `FactorError` and `SolveError`/`PatternMismatch`/`FeralIOError` as
//! direct children and the more specific factor failures below
//! `FactorError`. Keeping every exception a subclass of `FeralError`
//! (and the factor-time ones a subclass of `FactorError`) lets callers
//! write coarse `except FactorError` / `except FeralError` handlers; new
//! leaf types are therefore purely additive.

use feral::error::FeralError as RustFeralError;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

create_exception!(_feral, FeralError, PyException);
create_exception!(_feral, FactorError, FeralError);
create_exception!(_feral, SingularError, FactorError);
create_exception!(_feral, WrongInertiaError, FactorError);
create_exception!(_feral, NumericFailure, FactorError);
create_exception!(_feral, SqdContractViolated, FactorError);
create_exception!(_feral, DelayBudgetExceeded, FactorError);
create_exception!(_feral, SolveError, FeralError);
create_exception!(_feral, PatternMismatch, FeralError);
create_exception!(_feral, FeralIOError, FeralError);
// Unsymmetric LU basis-engine failures (issue #81). `SingularBasisError`
// is a `FactorError` (a singular basis is a factorization failure);
// `NeedsRefactorError` is a `FeralError` (a control-flow signal that the
// product-form update budget is exhausted, not a numeric failure). Both
// remain subclasses of the coarse roots, so existing `except FactorError`
// / `except FeralError` handlers keep working — additive.
create_exception!(_feral, SingularBasisError, FactorError);
create_exception!(_feral, NeedsRefactorError, FeralError);

pub(crate) fn map_feral_err(e: RustFeralError) -> PyErr {
    match e {
        RustFeralError::NumericallyRankDeficient => {
            SingularError::new_err("matrix is numerically rank-deficient")
        }
        RustFeralError::InvalidInput(s) => PyValueError::new_err(s),
        RustFeralError::DimensionMismatch { expected, got } => SolveError::new_err(format!(
            "dimension mismatch: expected {expected}, got {got}"
        )),
        RustFeralError::IoError(s) => FeralIOError::new_err(s),
        RustFeralError::NoFactor => {
            SolveError::new_err("no factorization available; call Solver.factor() first")
        }
        RustFeralError::SqdContractViolated { column, pivot } => SqdContractViolated::new_err(
            format!("SQD contract violated at column {column}: pivot = {pivot:e}"),
        ),
        RustFeralError::DelayBudgetExceeded {
            supernode,
            required,
            capacity,
        } => DelayBudgetExceeded::new_err(format!(
            "delayed-pivot budget exceeded at supernode {supernode}: \
             required {required} delayed columns, capacity {capacity}"
        )),
        // Unsymmetric LU basis-engine variants (issue #81), now surfaced
        // through dedicated exception types by the LU Python API.
        RustFeralError::SingularBasis { column } => {
            SingularBasisError::new_err(format!("LU basis is singular at column {column}"))
        }
        RustFeralError::NeedsRefactor => NeedsRefactorError::new_err(
            "LU basis update requires a refactor (update or stability budget reached)",
        ),
    }
}

/// Register the exception types on the module. Called from `lib.rs`.
pub(crate) fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FeralError", py.get_type_bound::<FeralError>())?;
    m.add("FactorError", py.get_type_bound::<FactorError>())?;
    m.add("SingularError", py.get_type_bound::<SingularError>())?;
    m.add(
        "WrongInertiaError",
        py.get_type_bound::<WrongInertiaError>(),
    )?;
    m.add("NumericFailure", py.get_type_bound::<NumericFailure>())?;
    m.add(
        "SqdContractViolated",
        py.get_type_bound::<SqdContractViolated>(),
    )?;
    m.add(
        "DelayBudgetExceeded",
        py.get_type_bound::<DelayBudgetExceeded>(),
    )?;
    m.add("SolveError", py.get_type_bound::<SolveError>())?;
    m.add("PatternMismatch", py.get_type_bound::<PatternMismatch>())?;
    m.add("FeralIOError", py.get_type_bound::<FeralIOError>())?;
    m.add(
        "SingularBasisError",
        py.get_type_bound::<SingularBasisError>(),
    )?;
    m.add(
        "NeedsRefactorError",
        py.get_type_bound::<NeedsRefactorError>(),
    )?;
    Ok(())
}
