//! Multilevel nested-dissection fill-reducing ordering.
//!
//! Clean-room Rust implementation of the algorithm described in
//! Karypis & Kumar, "A Fast and High Quality Multilevel Scheme for
//! Partitioning Irregular Graphs" (SIAM J. Sci. Comput., 1998), and
//! George, "Nested Dissection of a Regular Finite Element Mesh"
//! (SIAM J. Numer. Anal., 1973).
//!
//! The public surface conforms to the FERAL ordering-crate contract
//! (`dev/plans/ordering-crate-contract.md`): `CscPattern`,
//! `OrderingStats`, `OrderingError`, and `CONTRACT_VERSION` are
//! re-exported from `feral-ordering-core`.
//!
//! **Status: K1 scaffold.** The public signatures are locked to the
//! contract shape. The core algorithm (graph coarsening, initial
//! bisection, FM refinement, node-separator construction, recursive
//! nested dissection) is not yet implemented; `metis_order_full`
//! currently returns [`OrderingError::Internal`] with the message
//! `"not yet implemented"`. See `dev/plans/ordering-metis.md` for
//! the implementation milestones (M1 graph infra, M2 coarsening, ...
//! M7 ND driver, M8 integration).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

// Modules are exercised only by `metis_order_full` once all
// milestones land; until then, dead-code lint is suppressed at the
// module root for internal helpers.
#[allow(dead_code)]
mod coarsen;
#[allow(dead_code)]
mod graph;
#[allow(dead_code)]
mod initial_partition;
#[allow(dead_code)]
mod rng;

pub use feral_ordering_core::{CscPattern, OrderingError, OrderingStats, CONTRACT_VERSION};

/// Tunable parameters for METIS nested-dissection ordering.
///
/// Defaults mirror METIS 5.2.0's `METIS_NodeND` defaults as documented
/// in `dev/plans/ordering-metis.md` audit (MUMPS uses stock METIS
/// defaults for KKT problems: `METIS_OPTION_NUMBERING = 1`, all other
/// options at library default).
#[derive(Debug, Clone)]
pub struct MetisOptions {
    /// Deterministic RNG seed. Defaults to 1. Two runs with the same
    /// seed on the same input must produce the same permutation.
    pub seed: u64,
    /// Number of initial-bisection trials at the coarsest level
    /// (METIS 5.2.0 default: 7). Each trial alternates GGP and random
    /// BFS and is scored on its post-FM cut.
    pub niparts: u32,
    /// Stop coarsening when the graph has fewer than this many
    /// vertices (METIS 5.2.0 default: 120).
    pub coarsen_floor: u32,
    /// Switch from recursive ND to AMD on uncoarsened subproblems of
    /// at most this many vertices (METIS 5.2.0 default: 200).
    pub nd_to_amd_switch: u32,
    /// Reduction-ratio threshold below which SHEM falls back to
    /// 2-hop matching (METIS 5.2.0 default: 0.85).
    pub two_hop_ratio_threshold: f64,
    /// Maximum partition imbalance factor (`ufactor` in METIS terms,
    /// encoded as a fraction here). METIS 5.2.0 uses 200, which
    /// corresponds to 1.20 load balance tolerance; expressed as the
    /// fractional deviation 0.20.
    pub max_imbalance: f64,
    /// Number of FM passes at each uncoarsening level (METIS 5.2.0
    /// default: 10).
    pub fm_passes: u32,
}

impl Default for MetisOptions {
    fn default() -> Self {
        Self {
            seed: 1,
            niparts: 7,
            coarsen_floor: 120,
            nd_to_amd_switch: 200,
            two_hop_ratio_threshold: 0.85,
            max_imbalance: 0.20,
            fm_passes: 10,
        }
    }
}

/// Crate-specific diagnostic counters for METIS nested dissection.
///
/// Populated incrementally as the implementation milestones land.
/// Callers that only need the permutation should use
/// [`metis_order`]; callers that need the shared [`OrderingStats`]
/// (wall time, fill estimate) should use [`metis_order_full`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MetisStats {
    /// Number of coarsening levels built.
    pub n_levels: u32,
    /// Number of top-level connected components encountered.
    pub n_components: u32,
    /// Number of vertices assigned to a separator at any level.
    pub n_separator_vertices: u32,
    /// Number of FM passes executed across all levels.
    pub n_fm_passes: u32,
    /// Number of times SHEM fell through to the 2-hop matching path.
    pub n_two_hop_fallbacks: u32,
    /// Number of subgraphs handed off to the AMD leaf solver (when
    /// `nd_to_amd_switch` triggers).
    pub n_amd_leaf_calls: u32,
}

/// Compute a fill-reducing METIS nested-dissection ordering.
///
/// Thin wrapper over [`metis_order_full`] that discards the
/// diagnostic stats. Returns a permutation `perm` (new-to-old).
pub fn metis_order(pattern: &CscPattern<'_>) -> Result<Vec<i32>, OrderingError> {
    metis_order_full(pattern, &MetisOptions::default()).map(|(perm, _, _)| perm)
}

/// Contract-conforming ordering producer.
///
/// Signature matches the shape every FERAL ordering crate must expose
/// per `dev/plans/ordering-crate-contract.md`: input is a
/// full-symmetric [`CscPattern`] and options; output is a three-tuple
/// of `(perm, OrderingStats, crate-stats)`, with errors in
/// [`OrderingError`].
///
/// `OrderingStats.time_us` is the wall-clock time of this call.
/// `fill_estimate` and `flop_estimate` stay `None` — METIS does not
/// produce them at the ordering boundary; they belong to a downstream
/// symbolic analysis.
///
/// **Currently a stub.** Returns
/// `Err(OrderingError::Internal("not yet implemented"))` until the
/// M1-M7 milestones in `dev/plans/ordering-metis.md` land.
pub fn metis_order_full(
    pattern: &CscPattern<'_>,
    _opts: &MetisOptions,
) -> Result<(Vec<i32>, OrderingStats, MetisStats), OrderingError> {
    // Pattern must already be valid (CscPattern::new enforces that),
    // but we accept the borrowed view here and assume it. A length
    // sanity check keeps the stub honest against zero-width input.
    if pattern.col_ptr.len() != pattern.n + 1 {
        return Err(OrderingError::MalformedInput);
    }
    Err(OrderingError::Internal("not yet implemented"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trivial_pattern() -> (Vec<i32>, Vec<i32>) {
        // Diagonal n=3: col_ptr=[0,1,2,3], row_idx=[0,1,2]
        (vec![0, 1, 2, 3], vec![0, 1, 2])
    }

    #[test]
    fn options_defaults_match_metis_5_2_0() {
        let o = MetisOptions::default();
        assert_eq!(o.niparts, 7);
        assert_eq!(o.coarsen_floor, 120);
        assert_eq!(o.nd_to_amd_switch, 200);
        assert_eq!(o.seed, 1);
    }

    #[test]
    fn stats_default_is_zeros() {
        let s = MetisStats::default();
        assert_eq!(s.n_levels, 0);
        assert_eq!(s.n_components, 0);
        assert_eq!(s.n_separator_vertices, 0);
        assert_eq!(s.n_fm_passes, 0);
        assert_eq!(s.n_two_hop_fallbacks, 0);
        assert_eq!(s.n_amd_leaf_calls, 0);
    }

    #[test]
    fn stub_returns_internal_not_yet_implemented() {
        let (cp, ri) = trivial_pattern();
        let pat = CscPattern::new(3, &cp, &ri).unwrap();
        let err = metis_order_full(&pat, &MetisOptions::default()).unwrap_err();
        assert_eq!(err, OrderingError::Internal("not yet implemented"));
    }

    #[test]
    fn stub_convenience_wrapper_propagates_error() {
        let (cp, ri) = trivial_pattern();
        let pat = CscPattern::new(3, &cp, &ri).unwrap();
        let err = metis_order(&pat).unwrap_err();
        assert_eq!(err, OrderingError::Internal("not yet implemented"));
    }

    #[test]
    fn contract_version_matches_core() {
        assert_eq!(CONTRACT_VERSION, feral_ordering_core::CONTRACT_VERSION);
    }
}
