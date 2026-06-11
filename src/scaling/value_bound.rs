//! Value-bounded validity check for reusing a cached MC64 scaling
//! across warm `Solver::factor` replays (Track B2).
//!
//! An IPM driver calls `Solver::factor` repeatedly on a
//! bit-identical sparsity pattern with drifting values. The MC64
//! Hungarian matching that produces the scaling vector is dominated
//! by the *pattern*, not the values (see
//! `dev/research/mc64-value-bounded-cache-2026-05-17.md`), so the
//! scaling vector `D₀` computed on iter 0 is usually still good for
//! iter N. "Still good" means: applying `D₀` to the current matrix
//! `A_N` keeps it diagonally dominant enough that Bunch-Kaufman
//! pivoting selects pivots — and hence computes inertia — the same
//! way fresh scaling `D_N` would.
//!
//! This module provides the O(nnz) check that decides reuse-vs-
//! recompute. It does **not** rerun the Hungarian. Plan:
//! `dev/plans/mc64-value-bounded-cache.md`.
//!
//! ## Qualifying rows (Deviation 1 from the research note)
//!
//! The research note assumed a fully-populated diagonal. A KKT with
//! a structurally-zero `(2,2)` block has rows with a zero diagonal
//! and nonzero off-diagonals; their dominance ratio is `+∞`, which
//! would neuter the growth-factor and growth-count conditions. The
//! check therefore aggregates only over **qualifying rows** — rows
//! whose diagonal entry is structurally present and nonzero in the
//! cache-baseline matrix. `rocket_12800` (the B2 target) has a
//! fully-populated, all-nonzero diagonal, so every row qualifies and
//! there is no behaviour change on the target.

use crate::sparse::csc::CscMatrix;

/// Reject the cached scaling once the worst diagonal-dominance ratio
/// has grown past this multiple of the baseline `r0`.
const GROWTH_FACTOR: f64 = 2.0;
/// Reject once the count of off-dominant rows has grown past this
/// multiple of the baseline count.
const GROWTH_COUNT: f64 = 1.5;
/// Reject when a scaled diagonal has collapsed below this fraction
/// of the baseline mean scaled diagonal.
const EPS_DIAG: f64 = 1e-12;

/// Diagonal-dominance summary of `D · A · D`, aggregated over the
/// qualifying rows only. Produced by [`scaled_dominance_stats`].
#[derive(Debug, Clone, Copy)]
struct DominanceStats {
    /// `max_j (off_max_scaled[j] / diag_scaled[j])` over qualifying
    /// rows. A qualifying row with a zero scaled diagonal and a
    /// nonzero off contributes `+∞` (the diagonal collapsed).
    max_ratio: f64,
    /// Count of qualifying rows whose ratio exceeds 1 (the row's
    /// largest scaled magnitude is off-diagonal).
    n_off_dominant: usize,
    /// Minimum `|scaled diagonal|` over qualifying rows.
    min_diag: f64,
    /// Mean `|scaled diagonal|` over qualifying rows.
    mean_diag: f64,
}

/// Baseline diagonal-dominance fingerprint of the matrix that
/// produced a cached MC64 scaling. Stored alongside the cached
/// scaling vector at `Solver` scope; consumed by
/// [`mc64_value_bound_passes`] on every warm `factor()`.
#[derive(Debug, Clone)]
pub(crate) struct Mc64CacheValidity {
    /// Per-row mask: `true` where the diagonal was structurally
    /// present and nonzero in the baseline matrix. Length `n`.
    qualifying: Vec<bool>,
    /// `max(1.0, baseline max_ratio)` — the growth-factor reference.
    /// Clamped to `>= 1` so a well-conditioned baseline still
    /// permits the off-diagonals to reach the diagonal magnitude
    /// before the cache is rejected.
    r0: f64,
    /// Baseline count of off-dominant qualifying rows.
    n_off_dominant_0: usize,
    /// Baseline mean `|scaled diagonal|` — a fixed reference for the
    /// collapse check (does not drift with the current matrix).
    mean_diag_0: f64,
}

/// One O(nnz) sweep of `D · A · D` producing the [`DominanceStats`]
/// aggregated over `qualifying` rows.
///
/// `matrix` is a symmetric matrix in lower-triangle CSC; each stored
/// off-diagonal `a[i,j]` (`i > j`) updates the running off-max of
/// both row `i` and row `j`. `scaling` and `qualifying` must both
/// have length `matrix.n` — callers guarantee this.
fn scaled_dominance_stats(
    matrix: &CscMatrix,
    scaling: &[f64],
    qualifying: &[bool],
) -> DominanceStats {
    let n = matrix.n;
    let mut diag = vec![0.0_f64; n];
    let mut off_max = vec![0.0_f64; n];
    for j in 0..n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            let i = matrix.row_idx[k];
            let v = (matrix.values[k] * scaling[i] * scaling[j]).abs();
            if i == j {
                diag[j] = v;
            } else {
                if v > off_max[i] {
                    off_max[i] = v;
                }
                if v > off_max[j] {
                    off_max[j] = v;
                }
            }
        }
    }
    let mut max_ratio = 0.0_f64;
    let mut n_off_dominant = 0usize;
    let mut min_diag = f64::INFINITY;
    let mut sum_diag = 0.0_f64;
    let mut count = 0usize;
    for j in 0..n {
        if !qualifying[j] {
            continue;
        }
        let d = diag[j];
        let ratio = if d > 0.0 {
            off_max[j] / d
        } else if off_max[j] > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        if ratio > max_ratio {
            max_ratio = ratio;
        }
        if ratio > 1.0 {
            n_off_dominant += 1;
        }
        if d < min_diag {
            min_diag = d;
        }
        sum_diag += d;
        count += 1;
    }
    let (min_diag, mean_diag) = if count > 0 {
        (min_diag, sum_diag / count as f64)
    } else {
        (0.0, 0.0)
    };
    DominanceStats {
        max_ratio,
        n_off_dominant,
        min_diag,
        mean_diag,
    }
}

/// Per-row mask of structurally-present, nonzero diagonal entries.
fn qualifying_rows(matrix: &CscMatrix) -> Vec<bool> {
    let n = matrix.n;
    let mut q = vec![false; n];
    for (j, qj) in q.iter_mut().enumerate() {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            if matrix.row_idx[k] == j {
                *qj = matrix.values[k] != 0.0;
                break;
            }
        }
    }
    q
}

/// Compute the baseline validity fingerprint for `scaling` applied
/// to `matrix` (the matrix the MC64 Hungarian was just run on).
///
/// `scaling` must have length `matrix.n`. Callers pass the
/// freshly-applied `SparseFactors::scaling`, which satisfies this by
/// construction.
pub(crate) fn precompute_mc64_validity(matrix: &CscMatrix, scaling: &[f64]) -> Mc64CacheValidity {
    let qualifying = qualifying_rows(matrix);
    let stats = if scaling.len() == matrix.n {
        scaled_dominance_stats(matrix, scaling, &qualifying)
    } else {
        // Defensive: a length mismatch should be impossible here
        // (callers pass `SparseFactors::scaling`, length `n` by
        // construction), but never index out of bounds. This all-zero
        // fingerprint is a safe placeholder that is never actually
        // consulted to make a decision: it is produced only when
        // `scaling.len() != matrix.n`, and `mc64_value_bound_passes`
        // tests that *same* length mismatch first — its length gate
        // returns `false` before any condition is evaluated. The
        // fingerprint values do NOT by themselves force a reject: with
        // `mean_diag_0 = 0` the diagonal-collapse threshold
        // `EPS_DIAG * mean_diag_0` is `0`, so condition 3 becomes
        // vacuous (it passes for any non-negative scaled diagonal), and
        // `r0 = 1` does not force condition 1 to fail either. Rejection
        // on a length mismatch comes from the length gate, not from
        // these values.
        DominanceStats {
            max_ratio: 0.0,
            n_off_dominant: 0,
            min_diag: 0.0,
            mean_diag: 0.0,
        }
    };
    Mc64CacheValidity {
        qualifying,
        r0: stats.max_ratio.max(1.0),
        n_off_dominant_0: stats.n_off_dominant,
        mean_diag_0: stats.mean_diag,
    }
}

/// `true` when reusing the cached `scaling` on `matrix` is within
/// the value bound — the matching's diagonal-dominance guarantee
/// still holds and Bunch-Kaufman will see a qualitatively equivalent
/// matrix.
///
/// Rejects (returns `false`) if any of:
/// 1. the worst dominance ratio grew past `GROWTH_FACTOR · r0`;
/// 2. the off-dominant row count grew past `GROWTH_COUNT · count₀`;
/// 3. a qualifying scaled diagonal collapsed below
///    `EPS_DIAG · mean_diag₀`.
///
/// A length mismatch (`scaling` or the stored mask not `matrix.n`)
/// also returns `false` — recompute fresh, never index out of
/// bounds.
pub(crate) fn mc64_value_bound_passes(
    matrix: &CscMatrix,
    scaling: &[f64],
    validity: &Mc64CacheValidity,
) -> bool {
    let n = matrix.n;
    if scaling.len() != n || validity.qualifying.len() != n {
        return false;
    }
    let stats = scaled_dominance_stats(matrix, scaling, &validity.qualifying);
    let cond_ratio = stats.max_ratio <= GROWTH_FACTOR * validity.r0;
    let cond_count =
        (stats.n_off_dominant as f64) <= GROWTH_COUNT * (validity.n_off_dominant_0 as f64);
    let cond_diag = stats.min_diag >= EPS_DIAG * validity.mean_diag_0;
    cond_ratio && cond_count && cond_diag
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3×3 symmetric, lower-triangle CSC:
    /// ```text
    /// [ 4  2  0 ]
    /// [ 2  9  1 ]
    /// [ 0  1 16 ]
    /// ```
    fn matrix_3x3() -> CscMatrix {
        CscMatrix::from_triplets(
            3,
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[4.0, 2.0, 9.0, 1.0, 16.0],
        )
        .expect("valid CSC")
    }

    /// `scaled_dominance_stats` under identity scaling. Hand oracle:
    /// diag = [4, 9, 16]; off_max = [2, 2, 1]
    /// (entry (1,0)=2 feeds rows 0 and 1; entry (2,1)=1 feeds rows
    /// 1 and 2, but row 1 keeps its larger 2).
    /// ratios = [0.5, 0.2222…, 0.0625] → max 0.5, none off-dominant,
    /// min_diag 4, mean_diag 29/3.
    #[test]
    fn dominance_stats_identity_scaling_hand_oracle() {
        let m = matrix_3x3();
        let s = [1.0, 1.0, 1.0];
        let q = [true, true, true];
        let st = scaled_dominance_stats(&m, &s, &q);
        assert!(
            (st.max_ratio - 0.5).abs() < 1e-15,
            "max_ratio {}",
            st.max_ratio
        );
        assert_eq!(st.n_off_dominant, 0);
        assert!(
            (st.min_diag - 4.0).abs() < 1e-15,
            "min_diag {}",
            st.min_diag
        );
        assert!(
            (st.mean_diag - 29.0 / 3.0).abs() < 1e-13,
            "mean_diag {}",
            st.mean_diag
        );
    }

    /// `scaled_dominance_stats` under non-identity scaling
    /// s = [2, 1, 0.5]. Hand oracle:
    /// diag = [4·4, 9·1, 16·0.25] = [16, 9, 4];
    /// off (1,0): |2·1·2| = 4 → rows 0,1;
    /// off (2,1): |1·0.5·1| = 0.5 → rows 1,2 (row 1 keeps 4);
    /// off_max = [4, 4, 0.5]; ratios = [0.25, 0.4444…, 0.125].
    #[test]
    fn dominance_stats_scaled_hand_oracle() {
        let m = matrix_3x3();
        let s = [2.0, 1.0, 0.5];
        let q = [true, true, true];
        let st = scaled_dominance_stats(&m, &s, &q);
        assert!(
            (st.max_ratio - 4.0 / 9.0).abs() < 1e-15,
            "max_ratio {}",
            st.max_ratio
        );
        assert_eq!(st.n_off_dominant, 0);
        assert!(
            (st.min_diag - 4.0).abs() < 1e-15,
            "min_diag {}",
            st.min_diag
        );
        assert!(
            (st.mean_diag - 29.0 / 3.0).abs() < 1e-13,
            "mean_diag {}",
            st.mean_diag
        );
    }

    /// Deviation 1: a structurally-absent diagonal (column 2 empty)
    /// must NOT pull `max_ratio` to `+∞`. Matrix:
    /// ```text
    /// [ 4  2  0 ]
    /// [ 2  9  3 ]
    /// [ 0  3  · ]   (2,2) structurally absent
    /// ```
    /// `qualifying_rows` → [true, true, false]; stats over rows 0,1:
    /// off_max = [2, 3, 3]; ratios = [0.5, 1/3]; max 0.5, finite.
    #[test]
    fn zero_diagonal_row_excluded_from_stats() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 1, 2], &[0, 0, 1, 1], &[4.0, 2.0, 9.0, 3.0])
            .expect("valid CSC");
        let q = qualifying_rows(&m);
        assert_eq!(q, vec![true, true, false], "row 2 has no stored diagonal");
        let s = [1.0, 1.0, 1.0];
        let st = scaled_dominance_stats(&m, &s, &q);
        assert!(st.max_ratio.is_finite(), "max_ratio must be finite");
        assert!(
            (st.max_ratio - 0.5).abs() < 1e-15,
            "max_ratio {}",
            st.max_ratio
        );
        assert_eq!(st.n_off_dominant, 0, "row 2 (ratio +inf) is excluded");
    }

    /// An explicit-zero diagonal value also disqualifies the row.
    #[test]
    fn explicit_zero_diagonal_disqualifies_row() {
        let m = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[5.0, 2.0, 0.0])
            .expect("valid CSC");
        let q = qualifying_rows(&m);
        assert_eq!(q, vec![true, false], "row 1 diagonal is explicit zero");
    }

    /// Round trip: precompute on a matrix, then check the same
    /// matrix with the same scaling — the value bound must pass
    /// (zero drift).
    #[test]
    fn value_bound_passes_on_identical_matrix() {
        let m = matrix_3x3();
        let s = [1.0, 1.0, 1.0];
        let v = precompute_mc64_validity(&m, &s);
        assert!(
            mc64_value_bound_passes(&m, &s, &v),
            "identical matrix must pass the value bound"
        );
    }

    /// Condition 1 (growth factor) is the lone trigger.
    /// A_N = [[1, 5],[5, 1]] under identity scaling →
    /// max_ratio 5, n_off_dominant 2, min_diag 1, mean_diag 1.
    /// validity r0 = 2.0 → `GROWTH_FACTOR·r0 = 4.0`; 5 > 4 → reject.
    /// count budget `1.5·10 = 15 >= 2` and diag `1 >= 1e-12` both ok.
    #[test]
    fn value_bound_rejects_on_ratio_growth() {
        let m = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[1.0, 5.0, 1.0])
            .expect("valid CSC");
        let s = [1.0, 1.0];
        let v = Mc64CacheValidity {
            qualifying: vec![true, true],
            r0: 2.0,
            n_off_dominant_0: 10,
            mean_diag_0: 1.0,
        };
        assert!(
            !mc64_value_bound_passes(&m, &s, &v),
            "ratio 5 > GROWTH_FACTOR·r0 = 4 must reject"
        );
    }

    /// Condition 2 (growth count) is the lone trigger.
    /// A_N is a 3×3 with every off ratio 1.2 → max_ratio 1.2,
    /// n_off_dominant 3. validity r0 = 2.0 (ratio budget 4.0 ≥ 1.2,
    /// ok) but n_off_dominant_0 = 1 → count budget `1.5·1 = 1.5`;
    /// 3 > 1.5 → reject.
    #[test]
    fn value_bound_rejects_on_count_growth() {
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[1.0, 1.2, 1.0, 1.2, 1.0],
        )
        .expect("valid CSC");
        let s = [1.0, 1.0, 1.0];
        let v = Mc64CacheValidity {
            qualifying: vec![true, true, true],
            r0: 2.0,
            n_off_dominant_0: 1,
            mean_diag_0: 1.0,
        };
        assert!(
            !mc64_value_bound_passes(&m, &s, &v),
            "3 off-dominant rows > GROWTH_COUNT·1 = 1.5 must reject"
        );
    }

    /// Condition 3 (diagonal collapse) is the lone trigger.
    /// A_N is diagonal [[1e-20, ·],[·, 1.0]] — no off-diagonals, so
    /// max_ratio 0 and n_off_dominant 0 keep conditions 1 and 2
    /// satisfied. min_diag 1e-20 < `EPS_DIAG·mean_diag_0 = 1e-12`
    /// → reject.
    #[test]
    fn value_bound_rejects_on_diagonal_collapse() {
        let m = CscMatrix::from_triplets(2, &[0, 1], &[0, 1], &[1e-20, 1.0]).expect("valid CSC");
        let s = [1.0, 1.0];
        let v = Mc64CacheValidity {
            qualifying: vec![true, true],
            r0: 1.0,
            n_off_dominant_0: 0,
            mean_diag_0: 1.0,
        };
        assert!(
            !mc64_value_bound_passes(&m, &s, &v),
            "collapsed scaled diagonal 1e-20 < 1e-12 must reject"
        );
    }

    /// In-bounds drift passes: A_N = [[1, 5],[5, 1]] (max_ratio 5)
    /// against a generous validity (r0 = 3.0 → budget 6.0 ≥ 5).
    #[test]
    fn value_bound_passes_within_budget() {
        let m = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[1.0, 5.0, 1.0])
            .expect("valid CSC");
        let s = [1.0, 1.0];
        let v = Mc64CacheValidity {
            qualifying: vec![true, true],
            r0: 3.0,
            n_off_dominant_0: 10,
            mean_diag_0: 1.0,
        };
        assert!(
            mc64_value_bound_passes(&m, &s, &v),
            "ratio 5 <= GROWTH_FACTOR·r0 = 6 must pass"
        );
    }

    /// A length mismatch rejects rather than panicking.
    #[test]
    fn value_bound_rejects_on_length_mismatch() {
        let m = matrix_3x3();
        let v = precompute_mc64_validity(&m, &[1.0, 1.0, 1.0]);
        assert!(
            !mc64_value_bound_passes(&m, &[1.0, 1.0], &v),
            "scaling length 2 != n 3 must reject"
        );
    }
}
