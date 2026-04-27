//! Selection-metric trait abstracting AMD vs AMF differences.
//!
//! The shared quotient-graph machinery (`Workspace`, `select_pivot`,
//! `create_element`, `finalize_step`, hash-based supervariable
//! detection, mass elimination, aggressive absorption) is identical
//! across AMD and AMF. The metrics differ in:
//!
//! 1. **Initial score** seeded from each row's adjacency length.
//!    AMD: identity. AMF: identity (both = `len`).
//! 2. **Bucket array length.** AMD: `n + 1`. AMF: `2 * n + 2` because
//!    the quantized RMF can exceed `n`.
//! 3. **Bucket index for a score.** AMD: identity. AMF: identity for
//!    `s ≤ n`, then coarse stride `PAS = max(n / 8, 1)` above.
//! 4. **Pivot selection within a bucket.** AMD: head only.
//!    AMF: linear scan when the bucket is in the coarse region.
//! 5. **Score on supervariable merge.** AMD: no-op (only `nv[i]`
//!    accumulates). AMF: `score[i] = max(score[i], score[j])`.
//! 6. **Score finalisation** at the end of Pass-2: AMD's loose-degree
//!    formula `min(deg_prev, scan2_deg) + degme - nvi` clamped at
//!    `nleft - nvi`; AMF's quantized RMF (Amestoy 1999 thesis).
//!
//! The trait below covers (1)–(5). Site (6) is metric-specific in the
//! Pass-2 inner loop and is reached via the `run_elimination`
//! dispatch — each metric impl provides its own concrete loop today
//! (AMD reuses the existing loop in `algo.rs`). Phase B will decide
//! whether to share Pass-2 via further generic-ification or keep two
//! concrete loops; the trait shape here is forward-compatible with
//! either choice.
//!
//! Reference: `dev/research/amf-clean-room.md` Section 6.

use super::algo::{run_elimination as run_elimination_amd, StepFlops};
use super::workspace::Workspace;
use crate::OrderingError;

/// Selection metric for an AMD-family bottom-up ordering.
///
/// All methods are zero-overhead `#[inline(always)]` no-ops or
/// identity functions in the AMD case; AMF will provide non-trivial
/// implementations in Phase B. The trait is currently consumed only
/// at dispatch points (`run_elimination`, the ordering driver in
/// [`crate::quotient_graph::order`]) — the inner-loop sites that read
/// the trait directly will land in Phase B alongside MinFill.
pub trait Metric {
    /// Bucket key produced by the selection metric. AMD uses `i32`
    /// (the running degree); AMF will also use `i32` (quantized RMF).
    type Score: Copy + Ord + Default;

    /// Length of the bucket head array `Workspace::head`. AMD: `n`
    /// (indexed up to `n - 1` by `select_pivot`'s `while deg < n`).
    /// AMF: `2 * n + 2`.
    fn n_buckets(n: usize) -> usize;

    /// Initial score for a freshly-loaded variable with adjacency
    /// length `len`. AMD and AMF both seed `len`.
    fn init_score(len: i32) -> Self::Score;

    /// Bucket index for the given score. AMD: identity. AMF: identity
    /// for `s ≤ n`, coarse-stride above.
    fn bucket(score: Self::Score, n: usize) -> usize;

    /// Whether `idx` falls in the "coarse" bucket region — i.e.
    /// `select_pivot` must linear-scan the bucket chain to pick the
    /// minimum-score entry, rather than just taking the head. AMD
    /// always returns `false`; AMF returns `idx > n`.
    fn coarse_bucket(idx: usize, n: usize) -> bool;

    /// Update `parent`'s score on supervariable merge of `child` into
    /// `parent`. AMD: no-op. AMF: `*parent = max(*parent, child)`.
    fn merge_supervariable(parent: &mut Self::Score, child: Self::Score);

    /// Run the metric's elimination loop on a freshly initialised
    /// `Workspace`. Returns the accumulated flop counters.
    ///
    /// MinDegree dispatches to the AMD-specific concrete loop in
    /// `algo.rs`. Phase B's MinFill will provide its own concrete
    /// loop or the AMD loop generic-ified over `M: Metric`.
    fn run_elimination(ws: &mut Workspace, aggressive: bool) -> Result<StepFlops, OrderingError>;
}

/// Minimum-degree metric — the AMD selection rule of Amestoy, Davis,
/// Duff (1996).
///
/// Score is the running degree. Bucket index is the score itself.
/// All buckets are "fine" (head-only pivot selection). Supervariable
/// merge does not update the score (AMD tracks degree only via
/// `nv[i]` and the per-iteration Pass-2 monotone cap).
#[derive(Debug, Clone, Copy, Default)]
pub struct MinDegree;

impl Metric for MinDegree {
    type Score = i32;

    #[inline(always)]
    fn n_buckets(n: usize) -> usize {
        n
    }

    #[inline(always)]
    fn init_score(len: i32) -> i32 {
        len
    }

    #[inline(always)]
    fn bucket(score: i32, _n: usize) -> usize {
        score as usize
    }

    #[inline(always)]
    fn coarse_bucket(_idx: usize, _n: usize) -> bool {
        false
    }

    #[inline(always)]
    fn merge_supervariable(_parent: &mut i32, _child: i32) {
        // AMD does not maintain a per-supervariable score; degree
        // bookkeeping flows entirely through `nv[i]` and the
        // re-insertion loop's loose-degree formula.
    }

    #[inline(always)]
    fn run_elimination(ws: &mut Workspace, aggressive: bool) -> Result<StepFlops, OrderingError> {
        run_elimination_amd(ws, aggressive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_degree_n_buckets_matches_workspace_alloc() {
        // Workspace::new allocates head of length `n`; MinDegree
        // must agree so the AMD code path indexes the right region.
        for n in [0usize, 1, 5, 100, 10_000] {
            assert_eq!(MinDegree::n_buckets(n), n);
        }
    }

    #[test]
    fn min_degree_init_score_is_identity() {
        for len in [0i32, 1, 17, 1024] {
            assert_eq!(MinDegree::init_score(len), len);
        }
    }

    #[test]
    fn min_degree_bucket_is_identity() {
        for s in [0i32, 1, 7, 100] {
            assert_eq!(MinDegree::bucket(s, 200), s as usize);
        }
    }

    #[test]
    fn min_degree_no_coarse_buckets() {
        for n in [10usize, 100, 10_000] {
            for idx in [0usize, 1, n / 2, n - 1] {
                assert!(!MinDegree::coarse_bucket(idx, n));
            }
        }
    }

    #[test]
    fn min_degree_merge_does_not_touch_parent() {
        let mut parent: i32 = 42;
        MinDegree::merge_supervariable(&mut parent, 7);
        assert_eq!(parent, 42, "AMD merge is a true no-op on the score");
    }
}
