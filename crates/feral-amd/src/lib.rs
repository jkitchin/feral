//! Approximate Minimum Degree (AMD) fill-reducing ordering.
//!
//! Standalone implementation of the in-place quotient-graph AMD
//! algorithm (Amestoy, Davis & Duff 1996, 2004). See the crate
//! README and `dev/plans/ordering-amd-upgrade.md` for scope and
//! references.
//!
//! This module currently contains the scaffolding types only. The
//! core algorithm lands in subsequent commits (see Slice A plan).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod algo;
mod error;
mod pattern;
mod stats;
mod workspace;

pub use error::AmdError;
pub use pattern::CscPattern;
pub use stats::AmdStats;

/// Tunable parameters for AMD ordering.
///
/// Defaults match faer / SuiteSparse: `aggressive = true`,
/// `dense_alpha = 10.0`.
#[derive(Debug, Clone)]
pub struct AmdOptions {
    /// Enable aggressive element absorption in the Pass-2 degree
    /// loop (faer `amd.rs:404-407`).
    pub aggressive: bool,
    /// Dense-row threshold multiplier. A variable with initial
    /// degree exceeding `max(16, min(n, dense_alpha * sqrt(n)))` is
    /// deferred to the end of the ordering. A negative value sets
    /// the threshold to `n - 2` (faer `amd.rs:173-177`), which in
    /// practice suppresses deferral for all but true hubs of degree
    /// `n - 1`.
    pub dense_alpha: f64,
}

impl Default for AmdOptions {
    fn default() -> Self {
        Self {
            aggressive: true,
            dense_alpha: 10.0,
        }
    }
}

/// Compute a fill-reducing AMD ordering.
///
/// Returns a permutation `perm` (new-to-old) such that factoring
/// `P·A·Pᵀ` with `P = perm` is expected to produce less fill than
/// the natural ordering.
///
/// The input must be the full symmetric pattern (both halves).
///
/// Not yet implemented; all commits past scaffolding will be added
/// in Slice A.
pub fn amd_order(_pattern: &CscPattern<'_>) -> Result<Vec<usize>, AmdError> {
    Err(AmdError::NotImplemented)
}

/// Compute an AMD ordering and return diagnostic counters.
///
/// See [`amd_order`].
pub fn amd_order_with_stats(_pattern: &CscPattern<'_>) -> Result<(Vec<usize>, AmdStats), AmdError> {
    Err(AmdError::NotImplemented)
}

/// Compute an AMD ordering with explicit options.
pub fn amd_order_opts(
    _pattern: &CscPattern<'_>,
    _opts: &AmdOptions,
) -> Result<(Vec<usize>, AmdStats), AmdError> {
    Err(AmdError::NotImplemented)
}
