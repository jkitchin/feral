//! SCOTCH-style nested-dissection fill-reducing ordering.
//!
//! Clean-room pure-Rust implementation of the algorithms described in
//! Pellegrini, "SCOTCH: A software package for static mapping by dual
//! recursive bipartitioning of process and architecture graphs"
//! (HPCN Europe, 1996), §3. This crate provides an alternative to
//! [`feral_metis`](../feral_metis) with:
//!
//! - Graph compression (supervariable merging before partitioning).
//! - Direct vertex-separator FM (tighter separators than edge-cut
//!   conversion on structured meshes).
//! - Adaptive refinement (boundary / halo / band FM).
//!
//! The public surface conforms to the FERAL ordering-crate contract
//! (`dev/plans/ordering-crate-contract.md`): [`CscPattern`],
//! [`OrderingStats`], [`OrderingError`], and [`CONTRACT_VERSION`] are
//! re-exported from `feral-ordering-core`.
//!
//! **Status: S1 complete.** Only graph compression is wired up so far
//! (see [`compress`] below). The full `scotch_order` contract producer
//! and the nested-dissection driver land in milestones S2–S5, tracked
//! in `dev/plans/ordering-scotch.md`.
//!
//! ## Clean-room provenance
//!
//! No code is copied or paraphrased from SCOTCH's C source
//! (`libscotch/`, CeCILL-C). Algorithms are reconstructed from
//! Pellegrini 1996 §3 and the research notes under `dev/research/`.
//! Constants, data layouts, and hash choices are independently
//! justified and documented per-module.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[allow(dead_code)]
mod compress;
#[allow(dead_code)]
mod graph;
#[allow(dead_code)]
mod vertex_separator;

#[cfg(test)]
mod test_util;

pub use feral_ordering_core::{CscPattern, OrderingError, OrderingStats, CONTRACT_VERSION};

/// Tunable parameters for SCOTCH nested-dissection ordering.
///
/// Defaults mirror SCOTCH 7.0's vertex-separation ordering defaults as
/// documented in the audit of `dev/plans/ordering-scotch.md`:
/// `cmin = 100` (coarsening floor), `amd_switch = 120`,
/// `n_sep_trials = 5`, `fm_move_cap = 200`, `bal = 0.05`,
/// `compress_ratio = 0.7`, `seed = 0xDEAD_BEEF`.
///
/// **S1 status:** only [`compress`](Self::compress) and
/// [`compress_ratio`](Self::compress_ratio) are consumed by shipped
/// code. The remaining fields are defined at the S1 boundary so that
/// the public type does not drift between milestones.
#[derive(Debug, Clone)]
pub struct ScotchOptions {
    /// Apply graph compression before partitioning.
    pub compress: bool,
    /// Compress only if `n_compressed / n < compress_ratio` (i.e.,
    /// compress only when compression saves at least
    /// `(1 - compress_ratio) * 100` % of vertices). SCOTCH uses 0.75;
    /// we follow the plan's slightly more aggressive 0.7.
    pub compress_ratio: f64,
    /// Switch from recursive ND to AMD on subproblems with at most
    /// this many vertices (SCOTCH default: 120).
    pub amd_switch: u32,
    /// Stop coarsening when the graph has fewer than this many
    /// vertices (SCOTCH `cmin` = 100 for vertex-separation contexts).
    pub coarsen_floor: u32,
    /// Number of separator trials at each recursion level (SCOTCH
    /// default: 5).
    pub n_sep_trials: u32,
    /// FM refinement: per-pass move cap (SCOTCH default: 200).
    pub fm_move_cap: u32,
    /// FM refinement: per-call pass cap. SCOTCH's default is "passes
    /// until no improvement"; we clamp at 32 for bounded runtime.
    pub fm_pass_cap: u32,
    /// Imbalance tolerance (SCOTCH default: 0.05).
    pub max_imbalance: f64,
    /// Deterministic RNG seed for coarsening matching.
    pub seed: u64,
}

impl Default for ScotchOptions {
    fn default() -> Self {
        Self {
            compress: true,
            compress_ratio: 0.7,
            amd_switch: 120,
            coarsen_floor: 100,
            n_sep_trials: 5,
            fm_move_cap: 200,
            fm_pass_cap: 32,
            max_imbalance: 0.05,
            seed: 0xDEAD_BEEF,
        }
    }
}

/// Crate-specific diagnostic counters for SCOTCH nested dissection.
///
/// Zero-initialized at the start of each ordering call and filled in
/// as the pipeline runs. S1 only touches the compression counters.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ScotchStats {
    /// Number of vertices dropped by compression at the top level
    /// (`n_original - n_compressed`). Zero if compression was
    /// attempted but returned `None`, or if compression is disabled.
    pub n_compressed_out: u32,
    /// Number of coarsening levels built. Unused until S2.
    pub n_levels: u32,
    /// Number of top-level connected components encountered. Unused
    /// until S5.
    pub n_components: u32,
    /// Number of vertices assigned to a separator at any level.
    /// Unused until S5.
    pub n_separator_vertices: u32,
    /// Number of FM passes executed across all levels. Unused until
    /// S3/S4.
    pub n_fm_passes: u32,
    /// Number of times the recursion bottomed out in the AMD leaf
    /// fallback. Unused until S5.
    pub n_amd_leaf_calls: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_defaults_match_plan() {
        let o = ScotchOptions::default();
        assert!(o.compress);
        assert_eq!(o.compress_ratio, 0.7);
        assert_eq!(o.amd_switch, 120);
        assert_eq!(o.coarsen_floor, 100);
        assert_eq!(o.n_sep_trials, 5);
        assert_eq!(o.fm_move_cap, 200);
        assert_eq!(o.fm_pass_cap, 32);
        assert_eq!(o.max_imbalance, 0.05);
        assert_eq!(o.seed, 0xDEAD_BEEF);
    }

    #[test]
    fn stats_default_is_zeros() {
        let s = ScotchStats::default();
        assert_eq!(s.n_compressed_out, 0);
        assert_eq!(s.n_levels, 0);
        assert_eq!(s.n_components, 0);
        assert_eq!(s.n_separator_vertices, 0);
        assert_eq!(s.n_fm_passes, 0);
        assert_eq!(s.n_amd_leaf_calls, 0);
    }

    #[test]
    fn contract_version_matches_core() {
        assert_eq!(CONTRACT_VERSION, feral_ordering_core::CONTRACT_VERSION);
    }
}
