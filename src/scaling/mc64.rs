//! MC64 wrapper: input preprocessing, Hungarian call, symmetric
//! averaging, and output guards.
//!
//! Given a sparse symmetric matrix (lower triangle only in the
//! input CSC), this module produces a symmetric scaling vector
//! `s` such that `D · A · D` (with `D = diag(s)`) has
//! magnitude-bounded off-diagonals and unit-scale diagonals.
//!
//! Algorithm (mirrors `ref/spral/src/scaling.f90::hungarian_wrapper`):
//!
//!   1. Expand the lower-triangle CSC to a full symmetric pattern.
//!   2. Drop explicit zero entries (log of zero is -∞).
//!   3. Compute `c[k] = log |a[k]|` on the remaining entries.
//!   4. For each column j, compute `C[j] = max_k c[k]` and replace
//!      `c[k]` by `C[j] - c[k]`. The cost graph is now non-negative
//!      and has minimum 0 in each column.
//!   5. Run `hungarian_match` on the cost graph.
//!   6. Unwind the normalization:
//!      - `rscaling[i] = u[i]` (row dual unchanged)
//!      - `cscaling[j] = v[j] - C[j]` (col dual minus column max)
//!
//!      This matches `ref/spral/src/scaling.f90:681-682`.
//!   7. Symmetric average: `s[i] = exp((rscaling[i] + cscaling[i]) / 2)`.
//!      Matches `ref/spral/src/scaling.f90:169`.
//!   8. Safety guards: clamp dual variables whose exponential would
//!      overflow to finite values; rewrite any `s[i] == 0` to 1.
//!   9. On partial matching, set `s[i] = 1` for unmatched indices
//!      and return `ScalingInfo::PartialSingular`.
//!
//! **Phase 2.2.1 Step 1 status:** this is a stub that returns
//! identity scaling `[1.0; n]` with `ScalingInfo::NotApplied`.
//! Step 4 of `dev/plans/mc64-scaling.md` implements the real
//! wrapper once the Hungarian kernel in `hungarian.rs` is real.

use super::ScalingInfo;
use crate::error::FeralError;
use crate::sparse::csc::CscMatrix;

/// Compute the MC64 symmetric scaling for a sparse symmetric matrix.
///
/// The input `matrix` stores only the lower triangle (including the
/// diagonal). This function must internally expand to a full
/// symmetric pattern before running the matching.
///
/// **Phase 2.2.1 Step 1 stub.** Returns identity scaling and
/// `ScalingInfo::NotApplied` for any input. Step 4 of the
/// implementation plan replaces this with the real algorithm.
pub(crate) fn compute_symmetric(matrix: &CscMatrix) -> Result<(Vec<f64>, ScalingInfo), FeralError> {
    Ok((vec![1.0; matrix.n], ScalingInfo::NotApplied))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check: the stub returns identity scaling for any
    /// input. The real implementation tests live in
    /// `tests/mc64_scaling.rs` (created in Step 2 of the plan).
    #[test]
    fn stub_returns_identity_scaling() {
        let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
        let (s, info) = compute_symmetric(&csc).unwrap();
        assert_eq!(s, vec![1.0; 3]);
        assert_eq!(info, ScalingInfo::NotApplied);
    }
}
