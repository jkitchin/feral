//! PyO3 bindings for the feral sparse symmetric indefinite direct solver.
//!
//! See `python/README.md` for the user-facing documentation and
//! `dev/plans/python-interface.md` for the scoping plan.
//!
//! The binding is split into internal modules — the compiled extension
//! module is still `feral._feral`, and every public class/exception is
//! registered here so the Python-visible surface is unchanged:
//! - `common`: numpy↔Vec helpers, status/quality codes, `Inertia`.
//! - `errors`: the exception hierarchy and `FeralError → PyErr` map.
//! - `matrix`: `CscMatrix` (symmetric lower-triangular CSC).
//! - `solver`: the LDLᵀ `Solver`.
//! - `factors`: `Factors` — assembled `L`/`D` snapshot.
//! - `symbolic`: `SymbolicAnalysis` + the standalone `analyze`.
//! - `introspect`: factor-stats / profile / scaling value types.
//! - `lu`: the unsymmetric LU basis engine (`LuMatrix`, `LuFactor`).

mod common;
mod errors;
mod factors;
mod introspect;
mod lu;
mod matrix;
mod solver;
mod symbolic;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use common::{
    Inertia, QUALITY_BASELINE, QUALITY_EXHAUSTED, QUALITY_PIVOT_RAISED, QUALITY_SCALING_ENABLED,
    STATUS_INTERRUPTED, STATUS_NUMERIC_FAILURE, STATUS_SINGULAR, STATUS_SUCCESS,
    STATUS_WRONG_INERTIA,
};
use factors::Factors;
use introspect::{
    BucketStats, FactorStats, ProfileReport, PrologueBreakdown, ScalingInfo, StagePct,
    SymbolicProfileReport,
};
use lu::{LuFactor, LuMatrix};
use matrix::CscMatrix;
use solver::Solver;
use symbolic::SymbolicAnalysis;

#[pymodule]
fn _feral(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Inertia>()?;
    m.add_class::<CscMatrix>()?;
    m.add_class::<Solver>()?;
    m.add_class::<LuMatrix>()?;
    m.add_class::<LuFactor>()?;
    m.add_class::<Factors>()?;
    m.add_class::<SymbolicAnalysis>()?;
    m.add_class::<FactorStats>()?;
    m.add_class::<ProfileReport>()?;
    m.add_class::<PrologueBreakdown>()?;
    m.add_class::<BucketStats>()?;
    m.add_class::<SymbolicProfileReport>()?;
    m.add_class::<StagePct>()?;
    m.add_class::<ScalingInfo>()?;

    // Standalone symbolic-analysis entry point.
    m.add_function(wrap_pyfunction!(symbolic::analyze, m)?)?;

    // Exceptions
    errors::register(py, m)?;

    // Status / quality codes (mirrored as IntEnum on the Python side)
    let status = PyDict::new_bound(py);
    status.set_item("SUCCESS", STATUS_SUCCESS)?;
    status.set_item("SINGULAR", STATUS_SINGULAR)?;
    status.set_item("WRONG_INERTIA", STATUS_WRONG_INERTIA)?;
    status.set_item("NUMERIC_FAILURE", STATUS_NUMERIC_FAILURE)?;
    status.set_item("INTERRUPTED", STATUS_INTERRUPTED)?;
    m.add("_STATUS_CODES", status)?;

    let quality = PyDict::new_bound(py);
    quality.set_item("BASELINE", QUALITY_BASELINE)?;
    quality.set_item("SCALING_ENABLED", QUALITY_SCALING_ENABLED)?;
    quality.set_item("PIVOT_RAISED", QUALITY_PIVOT_RAISED)?;
    quality.set_item("EXHAUSTED", QUALITY_EXHAUSTED)?;
    m.add("_QUALITY_CODES", quality)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
