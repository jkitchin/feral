//! KaHIP-style flow-based nested-dissection fill-reducing ordering.
//!
//! **Status: pre-implementation scaffold.** This crate currently
//! exposes only the contract-conforming signatures; [`kahip_order`]
//! always returns [`OrderingError::Internal`] with the message
//! `"KaHIP: phases K1-K6 not yet implemented"`. Consumers that need a
//! working ordering should use `feral-metis` or `feral-scotch` for
//! nested dissection, or `feral-amd` for minimum-degree.
//!
//! **Plan.** `dev/plans/ordering-kahip.md` tracks the six
//! implementation phases:
//!   - K1: Data reduction (degree-1 / degree-2 / twin / neighborhood-
//!     subset rules, fixed-point loop, expansion permutation stack).
//!   - K2: Push-relabel max-flow (with gap relabeling).
//!   - K3: Flow-based edge refinement (band extraction, super-source/
//!     sink construction, Most Balanced Min Cut).
//!   - K4: Flow-based node separator (vertex-capacitated max-flow).
//!   - K5: V-cycle / F-cycle controller (cut-edge-preserving
//!     re-coarsening for monotone quality improvement).
//!   - K6: Driver and Fast / Eco / Strong modes.
//!
//! **Reference papers** (published, public-domain algorithms — the
//! implementation must be clean-room from these sources, not from
//! KaHIP's C++ codebase):
//!   - Sanders & Schulz, "Engineering Multilevel Graph Partitioning
//!     Algorithms" (2011) — the kaffpa framework.
//!   - Ost, Schulz & Strash, "Engineering Data Reduction for Nested
//!     Dissection" (2021) — the K1 reduction rules.
//!
//! The public surface conforms to the FERAL ordering-crate contract
//! (`dev/plans/ordering-crate-contract.md`): `CscPattern`,
//! `OrderingStats`, `OrderingError`, and `CONTRACT_VERSION` are
//! re-exported from `feral-ordering-core`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub use feral_ordering_core::{CscPattern, OrderingError, OrderingStats, CONTRACT_VERSION};

/// Crate-specific diagnostic statistics.
///
/// Populated by [`kahip_order_full`] once the implementation lands;
/// zeroed while the crate is in its scaffold state.
#[derive(Debug, Default, Clone)]
pub struct KahipStats {
    /// Number of vertices after data-reduction preprocessing.
    /// `reduced_n == 0` indicates the reduction phase has not run
    /// (scaffold state).
    pub reduced_n: usize,
    /// Largest max-flow subproblem size, in vertices, encountered
    /// during flow-based refinement. `0` while scaffolded.
    pub max_flow_vertices: usize,
    /// Number of V-cycles (or F-cycles) completed. `0` while
    /// scaffolded.
    pub cycles: usize,
}

/// Quality / speed tradeoff modes for the KaHIP driver.
///
/// The exact tuning of each mode is fixed once phase K6 lands. Until
/// then the enum is reserved so that callers can encode intent
/// without the crate compiling the mapping.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum KahipMode {
    /// METIS-comparable wall-clock; single multilevel pass.
    #[default]
    Fast,
    /// 2-3× Fast; one V-cycle with flow refinement at the finest
    /// level.
    Eco,
    /// 5-10× Fast; F-cycle with flow refinement at every level.
    Strong,
}

/// Tunable parameters for KaHIP nested-dissection ordering.
///
/// Kept intentionally narrow while the crate is a scaffold —
/// defaults will match KaHIP's library defaults (seed=0, mode=Fast)
/// once phase K6 is implemented.
#[derive(Debug, Clone)]
pub struct KahipOptions {
    /// Deterministic RNG seed. Two runs with the same seed on the
    /// same input must produce the same permutation.
    pub seed: u64,
    /// Quality / speed tradeoff. See [`KahipMode`].
    pub mode: KahipMode,
}

impl Default for KahipOptions {
    fn default() -> Self {
        Self {
            seed: 1,
            mode: KahipMode::default(),
        }
    }
}

/// Compute a fill-reducing KaHIP nested-dissection ordering.
///
/// Thin wrapper over [`kahip_order_full`] that discards the
/// diagnostic stats. Returns a permutation `perm` (new-to-old).
///
/// **Currently returns [`OrderingError::Internal`] unconditionally.**
/// The implementation of phases K1-K6 is tracked in
/// `dev/plans/ordering-kahip.md`.
pub fn kahip_order(pattern: &CscPattern<'_>) -> Result<Vec<i32>, OrderingError> {
    kahip_order_full(pattern, &KahipOptions::default()).map(|(perm, _, _)| perm)
}

/// Contract-conforming ordering producer.
///
/// Signature matches the shape every FERAL ordering crate must expose
/// per `dev/plans/ordering-crate-contract.md`: input is a
/// full-symmetric [`CscPattern`] and options; output is a three-tuple
/// of `(perm, OrderingStats, crate-stats)`, with errors in
/// [`OrderingError`].
///
/// **Currently returns [`OrderingError::Internal`] unconditionally.**
/// See module-level docs for the phase-by-phase implementation plan.
pub fn kahip_order_full(
    pattern: &CscPattern<'_>,
    _opts: &KahipOptions,
) -> Result<(Vec<i32>, OrderingStats, KahipStats), OrderingError> {
    if pattern.col_ptr.len() != pattern.n + 1 {
        return Err(OrderingError::MalformedInput);
    }
    Err(OrderingError::Internal(
        "KaHIP: phases K1-K6 not yet implemented",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_rejects_malformed_input() {
        let col_ptr = [0i32, 0];
        let row_idx: [i32; 0] = [];
        let pattern = CscPattern::new(5, &col_ptr, &row_idx);
        assert!(pattern.is_none(), "malformed pattern must fail validation");
    }

    #[test]
    fn scaffold_returns_not_implemented_on_valid_input() {
        let col_ptr = [0i32, 1, 2, 3];
        let row_idx = [0i32, 1, 2];
        let pattern = CscPattern::new(3, &col_ptr, &row_idx).expect("valid pattern");
        let err = kahip_order(&pattern).expect_err("scaffold must refuse to order");
        assert!(matches!(err, OrderingError::Internal(_)));
    }

    #[test]
    fn scaffold_propagates_malformed_input_to_caller() {
        // Caller-side malformed check: col_ptr len mismatch.
        // We have to construct this manually since CscPattern::new
        // refuses it — so build the struct through a sibling-crate
        // pattern then corrupt through direct field access is not
        // possible. Instead, test the same invariant via public API.
        let col_ptr = [0i32, 2];
        let row_idx = [0i32, 1];
        let pattern = CscPattern::new(1, &col_ptr, &row_idx);
        assert!(
            pattern.is_none(),
            "n=1 but col_ptr suggests 1 column with 2 rows"
        );
    }
}
