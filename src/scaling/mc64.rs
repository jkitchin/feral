//! MC64 wrapper: input preprocessing, Hungarian call, symmetric
//! averaging, and output guards.
//!
//! Given a sparse symmetric matrix (lower triangle only in the
//! input CSC), this module produces a symmetric scaling vector
//! `s` such that `D · A · D` (with `D = diag(s)`) has
//! magnitude-bounded off-diagonals and unit-scale diagonals.
//!
//! Algorithm (mirrors `ref/spral/src/scaling.f90::hungarian_wrapper`,
//! lines 597-801, in its non-singular branch):
//!
//!   1. Expand the lower-triangle CSC to a full symmetric pattern,
//!      carrying the original values with the transpose entries.
//!   2. Drop explicit zero entries (log of zero is -∞).
//!   3. Compute `c[k] = log |a[k]|` on the remaining entries.
//!   4. For each column j, compute `cmax[j] = max_k c[k]` and replace
//!      each `c[k]` by `cmax[j] - c[k]`. The cost graph is now
//!      non-negative and has minimum 0 in each column.
//!   5. Run `hungarian_match` on the cost graph.
//!   6. Unwind the normalization (SPRAL scaling.f90:681-682):
//!      - `rscaling[i] = u[i]` (row dual unchanged)
//!      - `cscaling[j] = v[j] - cmax[j]` (column dual minus column max)
//!   7. Symmetric average (SPRAL scaling.f90:169):
//!      `s[i] = exp((rscaling[i] + cscaling[i]) / 2)`.
//!   8. Safety guards: clamp the exponent to avoid overflow,
//!      rewrite any `s[i] == 0` or non-finite result to `1.0`.
//!   9. On partial matching, set `s[i] = 1.0` for unmatched indices
//!      and return `ScalingInfo::PartialSingular { n_unmatched }`.
//!
//! The partial-singular path deviates from SPRAL, which runs a
//! second Hungarian pass on the full-rank submatrix and then
//! applies a Duff-Pralet correction (scaling.f90:688-800). The
//! research note `dev/research/mc64-scaling.md` §"Structurally
//! singular matrices" specifies identity fallback for unmatched
//! rows/columns as the correct behavior for feral, because KKT
//! matrices from IPOPT are occasionally structurally rank-deficient
//! and a hard failure would regress the current `ForceAccept`
//! pathway.

use super::hungarian::{hungarian_match, CostGraph, Matching};
use super::ScalingInfo;
use crate::error::FeralError;
use crate::sparse::csc::CscMatrix;

/// Upper bound on the argument to `exp` before overflow.
/// `ln(f64::MAX) ≈ 709.78`. We use 709.0 as a safe ceiling.
const LOG_HUGE: f64 = 709.0;

/// Compute the MC64 symmetric scaling for a sparse symmetric matrix.
///
/// The input `matrix` stores only the lower triangle (including the
/// diagonal). This function expands to a full symmetric pattern
/// internally before running the matching.
///
/// Returns a length-`n` scaling vector in **user-order** indexing
/// together with a `ScalingInfo` diagnostic. On a non-singular
/// matrix the returned info is `ScalingInfo::Applied`; if the
/// matching is partial the info is
/// `ScalingInfo::PartialSingular { n_unmatched }` and the
/// unmatched positions are filled with `1.0` as an identity
/// fallback.
pub(crate) fn compute_symmetric(matrix: &CscMatrix) -> Result<(Vec<f64>, ScalingInfo), FeralError> {
    let n = matrix.n;

    // Trivial 0-dimensional case: nothing to scale.
    if n == 0 {
        return Ok((Vec::new(), ScalingInfo::Applied));
    }

    // Step 1-4: build the non-negative cost graph on the full
    // symmetric pattern, together with `cmax[j]` per column.
    let (cost_graph, cmax) = build_cost_graph(matrix)?;

    // Step 5: run the Hungarian kernel.
    let Matching {
        perm,
        u,
        v,
        n_matched,
    } = hungarian_match(&cost_graph);

    // Step 6-7: unwind normalization and form the symmetric average.
    //
    //   rscaling[i] = u[i]
    //   cscaling[i] = v[i] - cmax[i]
    //   s[i]        = exp((rscaling[i] + cscaling[i]) / 2)
    //               = exp((u[i] + v[i] - cmax[i]) / 2)
    //
    // Matches SPRAL scaling.f90:681-682 followed by :169.
    let mut scaling = vec![1.0_f64; n];
    for i in 0..n {
        // If the column had no usable entries at all, cmax[i] is
        // `f64::NEG_INFINITY` (see `build_cost_graph`). Any such
        // index is "empty" — the Hungarian kernel cannot match
        // that column meaningfully — so we fall back to identity
        // scaling for it. This is the structurally empty-column
        // case from the research note.
        if !cmax[i].is_finite() {
            scaling[i] = 1.0;
            continue;
        }

        // For unmatched columns, fall back to identity scaling
        // rather than using the dual variables (which are
        // meaningless on the unmatched part of the graph).
        if perm[i] == usize::MAX {
            scaling[i] = 1.0;
            continue;
        }

        let mut arg = (u[i] + v[i] - cmax[i]) / 2.0;

        // Clamp to avoid overflow on `exp`. A dual variable can
        // grow to ±∞-ish magnitudes on pathological inputs; both
        // MUMPS (dana_aux.F:1797-1816) and SSIDS guard against
        // this. The clamp is symmetric so that a clamped row
        // exponentiates to a very large or very small but finite
        // value rather than `+∞` or `0`.
        if !arg.is_finite() {
            scaling[i] = 1.0;
            continue;
        }
        arg = arg.clamp(-LOG_HUGE, LOG_HUGE);

        let s = arg.exp();

        // Defensive rewrite: a zero or non-finite scaling would
        // annihilate a whole row/column and destroy symmetry.
        // Mirrors MUMPS dana_aux.F:1809-1811.
        if s == 0.0 || !s.is_finite() {
            scaling[i] = 1.0;
        } else {
            scaling[i] = s;
        }
    }

    let info = if n_matched == n {
        ScalingInfo::Applied
    } else {
        ScalingInfo::PartialSingular {
            n_unmatched: n - n_matched,
        }
    };

    Ok((scaling, info))
}

/// Build the Hungarian cost graph and per-column maximum (`cmax`).
///
/// Expands the lower-triangle CSC `matrix` to a full symmetric
/// pattern, drops explicit-zero entries, takes the log of the
/// absolute value of each remaining entry, and normalizes each
/// column by subtracting its maximum so that the resulting costs
/// are non-negative.
///
/// Returns `(CostGraph, cmax)` where `cmax[j]` is the pre-
/// normalization column maximum (i.e., `max_i log|a[i,j]|`) used
/// in step 6 of `compute_symmetric` to unwind the normalization.
/// Columns with no finite (non-zero) entries have
/// `cmax[j] = f64::NEG_INFINITY`, which the caller treats as a
/// "fall back to identity" signal.
///
/// Algorithmic mirror: `ref/spral/src/scaling.f90:636-657`.
fn build_cost_graph(matrix: &CscMatrix) -> Result<(CostGraph, Vec<f64>), FeralError> {
    let n = matrix.n;

    // Two-pass expansion: first count the non-zero entries per
    // expanded column, then fill in the rows and values.
    //
    // For each stored lower-triangle entry at (row=i, col=j):
    //   * if val != 0 and i == j: contributes one entry to column j.
    //   * if val != 0 and i > j:  contributes to both column j (row i)
    //                             and column i (row j).
    //
    // Zero entries are dropped at the counting step so `log 0`
    // never appears.
    let mut col_counts = vec![0usize; n];
    for j in 0..n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            let i = matrix.row_idx[k];
            let val = matrix.values[k];
            if val == 0.0 {
                continue;
            }
            col_counts[j] += 1;
            if i != j {
                col_counts[i] += 1;
            }
        }
    }

    // Prefix sum to column pointers.
    let mut col_ptr = vec![0usize; n + 1];
    for j in 0..n {
        col_ptr[j + 1] = col_ptr[j] + col_counts[j];
    }
    let nnz_full = col_ptr[n];

    let mut row_idx = vec![0usize; nnz_full];
    let mut cost = vec![0.0_f64; nnz_full];
    let mut offsets: Vec<usize> = col_ptr[..n].to_vec();

    // Second pass: place entries.
    for j in 0..n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            let i = matrix.row_idx[k];
            let val = matrix.values[k];
            if val == 0.0 {
                continue;
            }
            let logabs = val.abs().ln();
            // (i, j) stays in column j.
            let p = offsets[j];
            row_idx[p] = i;
            cost[p] = logabs;
            offsets[j] += 1;
            // (j, i) transpose entry, if off-diagonal.
            if i != j {
                let q = offsets[i];
                row_idx[q] = j;
                cost[q] = logabs;
                offsets[i] += 1;
            }
        }
    }

    // Sort each column's rows ascending (Hungarian kernel does not
    // strictly require this, but a predictable order makes the
    // greedy initialization deterministic and matches SPRAL's
    // behaviour after `half_to_full`).
    for j in 0..n {
        let start = col_ptr[j];
        let end = col_ptr[j + 1];
        let mut pairs: Vec<(usize, f64)> = (start..end).map(|k| (row_idx[k], cost[k])).collect();
        pairs.sort_by_key(|&(r, _)| r);
        for (k, (r, c)) in (start..end).zip(pairs.into_iter()) {
            row_idx[k] = r;
            cost[k] = c;
        }
    }

    // Column-max normalization: for each column, find the maximum
    // log-absolute value and subtract it from every entry in that
    // column. Entries of an all-zero column are absent, so the
    // `cmax` for an empty column is `f64::NEG_INFINITY` and its
    // range is already empty — nothing to normalize.
    let mut cmax = vec![f64::NEG_INFINITY; n];
    for j in 0..n {
        let start = col_ptr[j];
        let end = col_ptr[j + 1];
        if start == end {
            continue;
        }
        let mut m = cost[start];
        for &c in &cost[(start + 1)..end] {
            if c > m {
                m = c;
            }
        }
        cmax[j] = m;
        for c in &mut cost[start..end] {
            *c = m - *c;
        }
    }

    let graph = CostGraph {
        n,
        col_ptr,
        row_idx,
        cost,
    };
    Ok((graph, cmax))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagonal SPD: expansion is a no-op, cost is all zero after
    /// normalization, Hungarian returns identity matching with
    /// `u = v = 0`, unwinding gives `scaling[i] = exp(-log A_ii / 2)
    /// = 1/sqrt(A_ii)`, and the scaled diagonal is 1.
    #[test]
    fn diagonal_matrix_produces_inverse_sqrt_scaling() {
        let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
        let (s, info) = compute_symmetric(&csc).unwrap();
        assert_eq!(info, ScalingInfo::Applied);
        let expected = [
            1.0 / 2.0_f64.sqrt(),
            1.0 / 3.0_f64.sqrt(),
            1.0 / 5.0_f64.sqrt(),
        ];
        for i in 0..3 {
            assert!(
                (s[i] - expected[i]).abs() < 1e-12,
                "s[{}] = {}, expected {}",
                i,
                s[i],
                expected[i]
            );
        }
    }

    /// Empty 0×0 matrix returns an empty scaling vector.
    #[test]
    fn empty_matrix_returns_empty_scaling() {
        let csc = CscMatrix {
            n: 0,
            col_ptr: vec![0],
            row_idx: vec![],
            values: vec![],
        };
        let (s, info) = compute_symmetric(&csc).unwrap();
        assert!(s.is_empty());
        assert_eq!(info, ScalingInfo::Applied);
    }
}
