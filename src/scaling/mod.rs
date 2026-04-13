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
/// Default is `InfNorm` — Knight-Ruiz iterative ∞-norm equilibration,
/// the same algorithm used by feral's dense BK path (see
/// `src/dense/equilibrate.rs`). This was moved to default after the
/// Phase 2.2.3 follow-up diagnostic showed MC64 was a silent no-op on
/// matrices like HYDCAR20, METHANL8, SWOPF, and HATFLDG — matrices
/// whose raw row norms span 4+ orders of magnitude but whose MC64
/// matching-based scaling came out near-identity. Knight-Ruiz
/// equilibration scales those matrices successfully and the sparse
/// path then matches the MUMPS oracle (see
/// `examples/dense_vs_sparse.rs` and
/// `examples/parity_config_sweep.rs` for the evidence).
///
/// `Mc64Symmetric` is still available as an opt-in; it is useful on
/// matrices where matching provides better conditioning than ∞-norm
/// balancing (e.g. SSINE_2529, VESUVIA_0000 in the parity panel).
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ScalingStrategy {
    /// Knight-Ruiz ∞-norm iterative equilibration. Default since the
    /// Phase 2.2.3 follow-up. Matches the scaling algorithm used by
    /// the dense BK path.
    #[default]
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
    }
}

// Hungarian types are used by the `mc64` module once Step 3 lands.
// Not part of the public API.
#[allow(unused_imports)]
pub(crate) use hungarian::{hungarian_match, CostGraph, Matching};
