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
//! approximate degree (AMD) vs approximate fill (AMF) — which today
//! is hardcoded to AMD's degree-based pivot ordering and will be
//! abstracted behind a `Metric` trait when AMF lands (see
//! `dev/plans/amf-clean-room.md` Phase A.2).
//!
//! Reference: Amestoy, Davis, Duff (1996) "An approximate minimum
//! degree ordering algorithm," SIAM J. Matrix Analysis 17:886-905.

#![allow(dead_code)]
// Quotient-graph internals (Workspace fields, StepFlops fields, etc.)
// are pub because the planned `feral-amf` crate will read them
// directly. They are deliberately not part of the locked
// ordering-crate contract; see CONTRACT_VERSION.
#![allow(missing_docs)]

mod algo;
mod workspace;

pub use algo::{
    create_element, finalize_permutation, finalize_step, run_elimination, select_pivot, StepFlops,
};
pub use workspace::{clear_flag, flip, Workspace, NONE};

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
