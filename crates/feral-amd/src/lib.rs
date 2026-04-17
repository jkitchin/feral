//! Approximate Minimum Degree (AMD) fill-reducing ordering.
//!
//! Standalone implementation of the in-place quotient-graph AMD
//! algorithm (Amestoy, Davis & Duff 1996, 2004). See the crate
//! README and `dev/plans/ordering-amd-upgrade.md` for scope and
//! references.
//!
//! Slice B is complete: mass elimination (Commit 9) and
//! supervariable detection (Commit 10) are both live, so the
//! ordering matches SuiteSparse / faer on the full oracle
//! fixture suite.

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
/// `P·A·Pᵀ` with `P[k] = perm[k]` produces less fill than the
/// natural ordering. The input must be the full symmetric pattern
/// (both halves present).
pub fn amd_order(pattern: &CscPattern<'_>) -> Result<Vec<usize>, AmdError> {
    amd_order_opts(pattern, &AmdOptions::default()).map(|(perm, _)| perm)
}

/// Compute an AMD ordering and return diagnostic counters.
///
/// See [`amd_order`] and [`AmdStats`].
pub fn amd_order_with_stats(pattern: &CscPattern<'_>) -> Result<(Vec<usize>, AmdStats), AmdError> {
    amd_order_opts(pattern, &AmdOptions::default())
}

/// Compute an AMD ordering with explicit options.
///
/// Returns `(perm, stats)`. See [`amd_order`] and [`AmdStats`].
pub fn amd_order_opts(
    pattern: &CscPattern<'_>,
    opts: &AmdOptions,
) -> Result<(Vec<usize>, AmdStats), AmdError> {
    let mut ws = workspace::AmdWorkspace::new(pattern, opts)?;
    let ndense = ws.ndense;
    let flops = algo::run_elimination(&mut ws, opts.aggressive)?;
    let ncmpa = ws.ncmpa;
    let n_mass_elim = ws.n_mass_elim;
    let n_supervar_merge = ws.n_supervar_merge;
    let perm = algo::finalize_permutation(&mut ws);
    let stats = AmdStats {
        ncmpa,
        n_clear_flag: 0,
        n_mass_elim,
        n_supervar_merge,
        n_dense_deferred: ndense.max(0) as u32,
        ndiv: flops.ndiv.max(0.0) as u64,
        nms_lu: flops.nms_lu.max(0.0) as u64,
        nms_ldl: flops.nms_ldl.max(0.0) as u64,
    };
    Ok((perm, stats))
}
