//! Hungarian algorithm for the minimum-cost perfect bipartite
//! matching problem.
//!
//! The core kernel for MC64-style scaling. Given a sparse
//! non-negative cost matrix in CSC form, finds a perfect matching
//! (row-to-column) minimizing total cost, and returns both the
//! matching permutation and the optimal dual variables from the
//! LP dual. Those duals are what get exponentiated into the
//! row/column scalings in `mc64.rs`.
//!
//! Reference: citet:duff2001mc64 §4. Source model:
//! `ref/spral/src/scaling.f90::hungarian_match` (lines 938-1171),
//! itself a clean-room rewrite of HSL_MC80. The algorithm is the
//! standard shortest-augmenting-path variant — each augmenting
//! path is a Dijkstra-like search on the reduced-cost graph, and
//! the dual variables are updated to preserve complementary
//! slackness.
//!
//! PHASE 2.2.1 STATUS: this is a stub for Step 1 scaffolding.
//! It returns the identity matching and zero duals for any input.
//! Step 3 of the implementation plan will replace this with a
//! real Hungarian implementation.

/// A sparse non-negative cost graph for the Hungarian algorithm.
///
/// Stored in CSC format on a square pattern (rows = cols = n).
/// All costs must be finite and non-negative; the MC64 wrapper
/// ensures this via per-column normalization of log absolute values.
/// Explicit zero entries in the pattern are allowed and represent
/// "cost 0" edges (which are the column-maximum entries after
/// normalization).
#[derive(Debug, Clone)]
pub(crate) struct CostGraph {
    pub n: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub cost: Vec<f64>,
}

/// Result of a Hungarian matching run.
#[derive(Debug, Clone)]
pub(crate) struct Matching {
    /// `perm[j]` is the row matched to column `j`. `usize::MAX`
    /// sentinel for unmatched columns (only populated in the
    /// partial-matching case).
    pub perm: Vec<usize>,
    /// Dual variable `u[i]` for row `i` (length `n`).
    pub u: Vec<f64>,
    /// Dual variable `v[j]` for column `j` (length `n`).
    pub v: Vec<f64>,
    /// Number of columns successfully matched. `n_matched == n`
    /// means a full perfect matching was found; a smaller value
    /// indicates structural singularity on the cost graph.
    pub n_matched: usize,
}

/// Solve the minimum-cost perfect bipartite matching problem via
/// the shortest-augmenting-path Hungarian algorithm.
///
/// At termination the dual variables satisfy
/// `u[i] + v[j] <= cost[i][j]` for every edge, with equality on
/// matched edges (the LP complementary-slackness conditions).
///
/// **Phase 2.2.1 Step 1 stub.** Returns the identity matching and
/// zero duals. This is wrong for any non-trivial input, but has
/// the useful property that plugging it into the MC64 wrapper
/// produces identity scaling, so existing feral tests continue
/// to pass while the rest of the module is built up.
///
/// Step 3 of the implementation plan (`dev/plans/mc64-scaling.md`)
/// replaces this with a real implementation following
/// `ref/spral/src/scaling.f90::hungarian_match`.
pub(crate) fn hungarian_match(cost: &CostGraph) -> Matching {
    let n = cost.n;
    Matching {
        perm: (0..n).collect(),
        u: vec![0.0; n],
        v: vec![0.0; n],
        n_matched: n,
    }
}

#[cfg(test)]
mod tests {
    //! Hungarian kernel unit tests.
    //!
    //! These tests exercise the `hungarian_match` function directly
    //! on small cost graphs where the answer can be hand-derived.
    //! Pre-Step 3 (the stub), tests that assert on identity-like
    //! behavior pass; tests that assert on non-trivial matchings
    //! or non-zero duals fail. This is intentional — the test file
    //! is the red→green gate for Step 3.
    //!
    //! Hand-derivation method: any minimum-cost perfect matching on
    //! a bipartite graph satisfies the LP optimality conditions
    //!   `u[i] + v[j] ≤ cost[i][j]`   for all edges,
    //!   `u[i] + v[j] == cost[i][j]`  on matched edges,
    //! so the matching plus any feasible dual that makes the total
    //! `sum(u) + sum(v)` equal to `sum(cost[matched])` is optimal.

    use super::*;

    /// Build a `CostGraph` from dense (row, col, cost) triples.
    /// Only used in tests — converts a small list of entries into
    /// the CSC format the Hungarian kernel expects.
    fn build_cost_graph(n: usize, entries: &[(usize, usize, f64)]) -> CostGraph {
        let mut by_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for &(r, c, v) in entries {
            by_col[c].push((r, v));
        }
        let mut col_ptr = vec![0usize; n + 1];
        let mut row_idx = Vec::new();
        let mut cost = Vec::new();
        for j in 0..n {
            by_col[j].sort_by_key(|&(r, _)| r);
            for &(r, v) in &by_col[j] {
                row_idx.push(r);
                cost.push(v);
            }
            col_ptr[j + 1] = row_idx.len();
        }
        CostGraph {
            n,
            col_ptr,
            row_idx,
            cost,
        }
    }

    /// Verify the LP optimality conditions for a `Matching` on a
    /// `CostGraph`: for every edge `u[i] + v[j] ≤ cost[i][j]` with
    /// equality on matched edges.
    fn assert_matching_optimal(cost: &CostGraph, m: &Matching) {
        let n = cost.n;
        assert_eq!(m.u.len(), n);
        assert_eq!(m.v.len(), n);
        assert_eq!(m.perm.len(), n);

        let mut matched_row = vec![false; n];
        for j in 0..n {
            if m.perm[j] != usize::MAX {
                matched_row[m.perm[j]] = true;
            }
        }

        for j in 0..n {
            for k in cost.col_ptr[j]..cost.col_ptr[j + 1] {
                let i = cost.row_idx[k];
                let c = cost.cost[k];
                let reduced = m.u[i] + m.v[j];
                assert!(
                    reduced <= c + 1e-10,
                    "edge ({},{}) has cost {} but u+v={} (reduced > cost)",
                    i,
                    j,
                    c,
                    reduced
                );
                if m.perm[j] == i {
                    assert!(
                        (reduced - c).abs() < 1e-10,
                        "matched edge ({},{}) has cost {} but u+v={} (not tight)",
                        i,
                        j,
                        c,
                        reduced
                    );
                }
            }
        }
    }

    /// 3×3 identity pattern: matching is trivially identity with
    /// zero duals. The stub passes this because "identity matching
    /// with zero duals" is exactly what it returns.
    #[test]
    fn match_diagonal_3x3_identity() {
        let cost = build_cost_graph(3, &[(0, 0, 0.0), (1, 1, 0.0), (2, 2, 0.0)]);
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 3);
        assert_eq!(m.perm, vec![0, 1, 2]);
        assert_matching_optimal(&cost, &m);
    }

    /// 3×3 with a non-identity permutation pattern:
    ///   cost(0, 1) = 0
    ///   cost(1, 2) = 0
    ///   cost(2, 0) = 0
    /// The only perfect matching is 0↔1, 1↔2, 2↔0 (i.e., col 0 is
    /// matched with row 2, etc.). The stub returns identity, which
    /// is NOT a valid matching on this sparsity pattern, so this
    /// test MUST fail on the stub.
    ///
    /// Gated `#[ignore]` until the Step 3 real Hungarian kernel
    /// lands; at that point the ignore attribute comes off and the
    /// test becomes a hard CI gate.
    #[test]
    #[ignore = "Step 3 of dev/plans/mc64-scaling.md — fails on stub"]
    fn match_permutation_3x3() {
        let cost = build_cost_graph(3, &[(1, 0, 0.0), (2, 1, 0.0), (0, 2, 0.0)]);
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 3);
        // perm[j] is the row matched to column j
        assert_eq!(m.perm[0], 1, "col 0 should match row 1");
        assert_eq!(m.perm[1], 2, "col 1 should match row 2");
        assert_eq!(m.perm[2], 0, "col 2 should match row 0");
        assert_matching_optimal(&cost, &m);
    }

    /// 3×3 with a non-trivial cost matrix where the answer requires
    /// actual Hungarian logic. The costs are:
    ///    col 0: row 0 -> 3, row 1 -> 1
    ///    col 1: row 0 -> 2, row 2 -> 4
    ///    col 2: row 1 -> 5, row 2 -> 0
    /// Minimum total cost is 1 + 2 + 0 = 3 via matching
    /// 0↔1, 1↔0, 2↔2 (col 0 ↔ row 1, col 1 ↔ row 0, col 2 ↔ row 2).
    /// Alternative matching 0↔0, 1↔2, 2↔1 has cost 3 + 4 + 5 = 12.
    /// Only the first is optimal. The stub returns identity
    /// `perm = [0, 1, 2]`, which on this cost graph would be
    /// 0↔0, 1↔1, 2↔2 — but (1,1) has no entry in our graph, so
    /// the stub's matching is not even feasible.
    ///
    /// Gated `#[ignore]` until Step 3.
    #[test]
    #[ignore = "Step 3 of dev/plans/mc64-scaling.md — fails on stub"]
    fn match_hand_computed_3x3() {
        let cost = build_cost_graph(
            3,
            &[
                (0, 0, 3.0),
                (1, 0, 1.0),
                (0, 1, 2.0),
                (2, 1, 4.0),
                (1, 2, 5.0),
                (2, 2, 0.0),
            ],
        );
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 3);
        assert_eq!(m.perm[0], 1, "col 0 matches row 1 (cost 1)");
        assert_eq!(m.perm[1], 0, "col 1 matches row 0 (cost 2)");
        assert_eq!(m.perm[2], 2, "col 2 matches row 2 (cost 0)");
        assert_matching_optimal(&cost, &m);
        let total: f64 = (0..3).map(|j| m.u[m.perm[j]] + m.v[j]).sum();
        assert!(
            (total - 3.0).abs() < 1e-10,
            "total matching cost should be 3 (1+2+0), got {}",
            total
        );
    }
}
