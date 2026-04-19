//! Global scaling for sparse symmetric indefinite matrices.
//!
//! Implements MC64-style matching-based scaling following
//! Duff & Koster 2001 and Duff & Pralet 2005, using a pure-Rust
//! Hungarian algorithm. The resulting scaling vector `s` is applied
//! symmetrically: `A ↦ diag(s) · A · diag(s)` before factorization.
//!
//! Design: see `dev/research/mc64-scaling.md`.
//! Plan:   see `dev/plans/mc64-scaling.md`.
//!
//! This module is Phase 2.2.1 work — closing the residual gap that
//! Phase 2.1.2's sanity check exposed on n > 500 matrices.
//!
//! ## Quick reference
//!
//! The caller computes scaling via `compute_scaling(matrix, strategy)`,
//! which returns `(Vec<f64>, ScalingInfo)`. The vector is in user-order
//! indexing (same numbering as the input CSC's row/column indices).
//! It is the responsibility of later symbolic-factorization code to
//! permute the vector into pivot-order before handing off to the
//! numeric phase.
//!
//! Once the scaling vector is available, three things must happen:
//!
//!   1. During frontal assembly in `numeric::factorize`, each original
//!      matrix entry `a[i,j]` is multiplied by `s[i] * s[j]` as it is
//!      scattered into the frontal matrix.
//!   2. In `numeric::solve`, the right-hand side `b` is pre-scaled by
//!      `b[i] *= s[i]` at the permutation boundary before the forward
//!      sweep.
//!   3. In `numeric::solve`, the solution `x` is post-scaled by
//!      `x[i] *= s[i]` at the un-permutation boundary after the
//!      backward sweep. **Same vector on both ends**, not its
//!      inverse — see the research note for the derivation.

use crate::error::FeralError;
use crate::sparse::csc::CscMatrix;

#[allow(dead_code)] // Real uses arrive in Step 3 of the implementation plan.
mod hungarian;
mod infnorm;
mod mc64;

/// User-facing scaling strategy selector.
///
/// Default is `Auto` — adaptive shape-based routing that picks
/// `Mc64Symmetric` for matrices with the arrow-KKT signature
/// (`diag_only / n >= 0.30`) and `InfNorm` everywhere else. Flipped
/// from the prior `InfNorm` default on 2026-04-19 after the
/// per-matrix residual-set diff confirmed the trade: 8× tail
/// compression on factor/MUMPS (worst case 83× → 10×) and material
/// wins on the VESUVIO/CRESC IPM corpus, against a net −9 change
/// in the residual_pass count out of 154 588. Of the 21 regressions,
/// 14 are oracle-`numerically_intractable` and 1 is `excluded`
/// (boundary flicker on already-hard matrices); 5 of the remaining
/// 6 `definitive` regressions are tolerance-edge effects (residuals
/// 1e-10 → 1e-9 around the `n·ε·1e6` threshold). The lone material
/// residual regression is MSS1_0009 (6e-12 → 1e-6, inertia preserved).
/// Inertia hard rule is satisfied on every regression. See
/// `dev/research/lever-c-residual-diff-2026-04-19.md`.
///
/// `InfNorm` (Knight-Ruiz iterative ∞-norm equilibration) is still
/// available as an opt-in; it is the only choice that solves
/// MSS1_0009 to working precision today and is the right pick for
/// pipelines that cannot tolerate the MSS1-class residual loss
/// pending Policy 4 (post-scaling trial-residual diagnostic).
///
/// `Mc64Symmetric` is also opt-in; it is useful on matrices where
/// matching provides better conditioning than ∞-norm balancing
/// (e.g. SSINE_2529, VESUVIA_0000 in the parity panel) but pays the
/// MC64 symbolic overhead unconditionally.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ScalingStrategy {
    /// Knight-Ruiz ∞-norm iterative equilibration. Matches the
    /// scaling algorithm used by the dense BK path. Was the default
    /// from Phase 2.2.3 through the 2026-04-19 lever-C residual diff
    /// (now opt-in).
    InfNorm,
    /// MC64-style symmetric matching-based scaling. Matches the
    /// default behavior of MUMPS (SYM=2) and SSIDS
    /// (options%scaling=1). Useful on matrices where matching
    /// provides better conditioning than ∞-norm balancing.
    Mc64Symmetric,
    /// Identity scaling (no-op). Use for regression testing and for
    /// inputs where any scaling is inappropriate.
    Identity,
    /// User-supplied pre-computed scaling vector in user-order
    /// indexing. Length must equal the matrix dimension.
    External(Vec<f64>),
    /// Adaptive shape-based routing: `Mc64Symmetric` when the matrix
    /// has the arrow-KKT signature (many degree-1 "constraint slack"
    /// columns), else `InfNorm`. The routing rule is documented at
    /// [`pick_scaling_strategy`]; threshold is `diag_only / n >= 0.3`.
    /// Default since 2026-04-19. See
    /// `dev/research/lever-c-residual-diff-2026-04-19.md`.
    #[default]
    Auto,
}

/// Diagnostic information about how the scaling was computed.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingInfo {
    /// MC64 matching ran to completion on a non-singular matrix.
    Applied,
    /// MC64 matching found a partial solution; unmatched rows and
    /// columns fall back to identity scaling. `n_unmatched` is the
    /// number of variables that could not be matched. The returned
    /// scaling vector has `1.0` at the unmatched positions.
    PartialSingular { n_unmatched: usize },
    /// No matching-based scaling was applied (e.g., the caller
    /// requested `Identity` or `External`).
    NotApplied,
}

/// Compute the symmetric scaling vector for a sparse symmetric
/// matrix stored in CSC with only the lower triangle, following
/// `strategy`.
///
/// Returns a vector of length `n` in **user-order** indexing such
/// that applying `D = diag(scaling)` as the congruence transform
/// `D · A · D` produces a matrix whose largest-magnitude entries lie
/// on the diagonal. The off-diagonals are bounded by 1 in absolute
/// value when MC64 succeeds on a non-singular matrix.
///
/// Users of the result must permute the vector into pivot-order
/// indexing before the numeric phase looks it up.
pub fn compute_scaling(
    matrix: &CscMatrix,
    strategy: &ScalingStrategy,
) -> Result<(Vec<f64>, ScalingInfo), FeralError> {
    match strategy {
        ScalingStrategy::Identity => Ok((vec![1.0; matrix.n], ScalingInfo::NotApplied)),
        ScalingStrategy::External(s) => {
            if s.len() != matrix.n {
                return Err(FeralError::InvalidInput(format!(
                    "external scaling has length {} but matrix has n={}",
                    s.len(),
                    matrix.n,
                )));
            }
            Ok((s.clone(), ScalingInfo::NotApplied))
        }
        ScalingStrategy::InfNorm => Ok(infnorm::compute_infnorm(matrix)),
        ScalingStrategy::Mc64Symmetric => mc64::compute_symmetric(matrix),
        ScalingStrategy::Auto => compute_scaling(matrix, &pick_scaling_strategy(matrix)),
    }
}

/// Resolve `ScalingStrategy::Auto` to a concrete strategy based on
/// matrix shape.
///
/// Routes to `Mc64Symmetric` when the matrix has the arrow-KKT
/// signature — many degree-1 "constraint slack" columns whose only
/// stored row is the diagonal. Else routes to `InfNorm`.
///
/// Threshold: `diag_only / n >= 0.3`. Selected from the `vesuvio_diag`
/// shape distribution: VESUVIOU/VESUVIO/VESUVIA/MUONSINE/CRESC132 all
/// have ratios above 0.3 and benefit from MC64 (delays drop to zero,
/// 6×–229× factor speedup); HYDCAR20/METHANL8/SWOPF/HATFLDG (the
/// matrices that motivated the InfNorm default) have ratios below
/// 0.3. See `dev/research/lever-c-adaptive-scaling.md`.
///
/// One O(n) pass over the column pointers and one O(nnz) pass over
/// the row indices. No allocations.
pub fn pick_scaling_strategy(matrix: &CscMatrix) -> ScalingStrategy {
    let n = matrix.n;
    if n == 0 {
        return ScalingStrategy::InfNorm;
    }
    let mut diag_only = 0usize;
    for j in 0..n {
        let start = matrix.col_ptr[j];
        let end = matrix.col_ptr[j + 1];
        if end - start != 1 {
            continue;
        }
        if matrix.row_idx[start] == j {
            diag_only += 1;
        }
    }
    if diag_only as f64 / n as f64 >= 0.3 {
        ScalingStrategy::Mc64Symmetric
    } else {
        ScalingStrategy::InfNorm
    }
}

// Hungarian types are used by the `mc64` module once Step 3 lands.
// Not part of the public API.
#[allow(unused_imports)]
pub(crate) use hungarian::{hungarian_match, CostGraph, Matching};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    /// Build a CSC with `n` columns where the first `diag_only`
    /// columns are degree-1 (just the diagonal), and the remaining
    /// `n - diag_only` columns each store the diagonal plus one
    /// off-diagonal row at column 0. Lower-triangular only — no
    /// validity beyond the column-degree pattern is required for
    /// `pick_scaling_strategy`, which only inspects col_ptr and
    /// row_idx.
    fn shape_csc(n: usize, diag_only: usize) -> CscMatrix {
        assert!(diag_only <= n);
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        col_ptr.push(0);
        for j in 0..n {
            row_idx.push(j);
            values.push(1.0);
            if j >= diag_only && j != 0 {
                row_idx.push(j.max(1) - 1);
                values.push(0.1);
            }
            col_ptr.push(row_idx.len());
        }
        CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        }
    }

    #[test]
    fn pick_scaling_strategy_picks_mc64_for_arrow_kkt() {
        // 10 of 20 columns are diag-only → ratio = 0.5 ≥ 0.3.
        let csc = shape_csc(20, 10);
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::Mc64Symmetric);
    }

    #[test]
    fn pick_scaling_strategy_picks_infnorm_for_dense() {
        // 0 of 20 columns are diag-only → ratio = 0.0 < 0.3.
        let csc = shape_csc(20, 0);
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::InfNorm);
    }

    #[test]
    fn pick_scaling_strategy_threshold_boundary() {
        // 29 of 100 → 0.29 < 0.30 → InfNorm.
        let below = shape_csc(100, 29);
        assert_eq!(pick_scaling_strategy(&below), ScalingStrategy::InfNorm);
        // 30 of 100 → 0.30 ≥ 0.30 → MC64.
        let at = shape_csc(100, 30);
        assert_eq!(pick_scaling_strategy(&at), ScalingStrategy::Mc64Symmetric);
    }

    #[test]
    fn pick_scaling_strategy_empty_matrix_picks_infnorm() {
        let csc = CscMatrix {
            n: 0,
            col_ptr: vec![0],
            row_idx: vec![],
            values: vec![],
        };
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::InfNorm);
    }

    #[test]
    fn compute_scaling_auto_routes_to_mc64_on_arrow_kkt() {
        // Build a small symmetric arrow KKT: 4 diag-only "slack"
        // columns + 2 dense "linking" columns. Lower-triangular CSC.
        // Ratio diag_only / n = 4/6 = 0.67 → Auto resolves to MC64.
        let n = 6;
        let mut col_ptr = vec![0usize];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        // 4 diag-only columns.
        for j in 0..4 {
            row_idx.push(j);
            values.push(2.0);
            col_ptr.push(row_idx.len());
        }
        // 2 dense columns (diagonal + all earlier rows).
        for j in 4..n {
            row_idx.push(j);
            values.push(2.0);
            for i in (j + 1)..n {
                row_idx.push(i);
                values.push(0.1);
            }
            col_ptr.push(row_idx.len());
        }
        let csc = CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        };
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::Mc64Symmetric);
        // Auto and explicit Mc64Symmetric must produce the same vector.
        let (auto_s, _) =
            compute_scaling(&csc, &ScalingStrategy::Auto).expect("Auto routing should succeed");
        let (mc64_s, _) =
            compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric).expect("MC64 should succeed");
        assert_eq!(auto_s, mc64_s);
    }
}
