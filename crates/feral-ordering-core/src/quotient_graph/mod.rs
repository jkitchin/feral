//! Shared quotient-graph machinery for AMD-family bottom-up
//! orderings.
//!
//! This module hosts the workspace, elimination loop, and assembly-
//! tree postorder used by `feral-amd` and (planned) `feral-amf`.
//! Both orderings share the quotient-graph data structures
//! (`PE / IW / LEN / NV / ELEN`), the standard / aggressive element
//! absorption logic, the mass-elimination fast path, the
//! supervariable hash bucket detection, and the inline garbage
//! collector. They differ only in the *selection metric* —
//! approximate degree (AMD) vs approximate fill (AMF) — which is
//! abstracted behind the [`Metric`] trait. Phase A.2 ships the trait
//! plus the AMD-specialised [`MinDegree`] impl; Phase B will add
//! [`Metric`]'s second impl ([`feral-amf`'s `MinFill`]) and decide
//! whether the inner-loop body itself becomes generic over `M`.
//!
//! Reference: Amestoy, Davis, Duff (1996) "An approximate minimum
//! degree ordering algorithm," SIAM J. Matrix Analysis 17:886-905;
//! Amestoy (1999) habilitation thesis (AMF metric).

#![allow(dead_code)]
// Quotient-graph internals (Workspace fields, StepFlops fields, etc.)
// are pub because the planned `feral-amf` crate will read them
// directly. They are deliberately not part of the locked
// ordering-crate contract; see CONTRACT_VERSION.
#![allow(missing_docs)]

mod algo;
mod metric;
mod workspace;

pub use algo::{
    create_element, finalize_permutation, finalize_step, run_elimination, select_pivot, StepFlops,
};
pub use metric::{Metric, MinDegree};
pub use workspace::{clear_flag, flip, Workspace, NONE};

use crate::{CscPattern, OrderingError};

/// Tunable parameters for the shared quotient-graph workspace.
///
/// Only the workspace-relevant parameters live here. Crate-specific
/// knobs (e.g. `aggressive` for the elimination loop) are passed
/// directly to the relevant entry point.
#[derive(Debug, Clone)]
pub struct WorkspaceOptions {
    /// Dense-row threshold multiplier (Davis 1996 §5). A variable
    /// with initial degree exceeding
    /// `max(16, min(n, dense_alpha * sqrt(n)))` is deferred to the
    /// end of the ordering. A negative value sets the threshold to
    /// `n - 2`, suppressing deferral for everything but true hubs of
    /// degree `n - 1`.
    pub dense_alpha: f64,
}

impl Default for WorkspaceOptions {
    fn default() -> Self {
        Self { dense_alpha: 10.0 }
    }
}

/// Diagnostic counters extracted from a completed [`Workspace`].
///
/// Surfaced by [`order`] alongside the permutation so callers can
/// build crate-specific stats structs without re-borrowing the
/// workspace internals.
#[derive(Debug, Clone, Copy, Default)]
pub struct OrderDiagnostics {
    pub ncmpa: u32,
    pub n_mass_elim: u32,
    pub n_supervar_merge: u32,
    pub ndense: i32,
    pub flops: StepFlops,
}

/// Run a metric-driven AMD-family ordering on a full-symmetric
/// pattern, returning the permutation plus diagnostic counters.
///
/// Equivalent to:
///
/// ```ignore
/// let mut ws = Workspace::new(pattern, opts)?;
/// let flops = M::run_elimination(&mut ws, aggressive)?;
/// let perm = finalize_permutation(&mut ws);
/// ```
///
/// `M` selects the metric (and, transitively, the elimination loop).
/// AMD uses [`MinDegree`]; the planned AMF crate will pass `MinFill`.
pub fn order<M: Metric>(
    pattern: &CscPattern<'_>,
    opts: &WorkspaceOptions,
    aggressive: bool,
) -> Result<(Vec<i32>, OrderDiagnostics), OrderingError> {
    let mut ws = Workspace::new(pattern, opts)?;
    let flops = M::run_elimination(&mut ws, aggressive)?;
    let diag = OrderDiagnostics {
        ncmpa: ws.ncmpa,
        n_mass_elim: ws.n_mass_elim,
        n_supervar_merge: ws.n_supervar_merge,
        ndense: ws.ndense,
        flops,
    };
    let perm = finalize_permutation(&mut ws);
    Ok((perm, diag))
}
