//! Knight-Ruiz ∞-norm iterative equilibration for sparse symmetric
//! matrices.
//!
//! Given a symmetric CSC matrix `A`, compute a diagonal `d` such that
//! each row of `D·A·D` has infinity-norm ≈ 1, where `D = diag(d)`.
//! This is the same algorithm used by the dense path in
//! `src/dense/equilibrate.rs`, adapted to iterate over lower-triangular
//! CSC storage.
//!
//! Phase 2.2.3 follow-up: the dense BK factorization succeeds on
//! HYDCAR20 / METHANL8 / SWOPF / HATFLDG because it equilibrates the
//! matrix before BK. The sparse multifrontal path was missing this
//! step — MC64 matching happens to classify these matrices as already
//! balanced even when their row norms span 4+ orders of magnitude.
//! Porting the dense path's equilibration recovers these matrices for
//! the sparse path.
//!
//! Algorithm (Jacobi-style, converges in the same number of iterations
//! as the Gauss-Seidel variant used by `dense::equilibrate` while being
//! simpler to implement over CSC lower-triangle storage):
//!
//! 1. Initialize `d = 1`.
//! 2. Repeat up to `max_iter` times:
//!    a. For each row `i`, compute `max_i = max_j |d[i]·a[i,j]·d[j]|`.
//!    b. Update `d[i] /= sqrt(max_i)` for every row whose `max_i > 0`.
//!    c. Stop when `max_i |1 − max_i|` falls below `tol`.

use crate::dense::matrix::SymmetricMatrix;
use crate::scaling::ScalingInfo;
use crate::sparse::csc::CscMatrix;

/// Compute the Knight-Ruiz ∞-norm symmetric scaling vector for a
/// lower-triangular symmetric CSC matrix. Returns the diagonal `d`
/// such that `D·A·D` has unit-∞ rows, paired with `ScalingInfo::Applied`.
pub fn compute_infnorm(matrix: &CscMatrix) -> (Vec<f64>, ScalingInfo) {
    let n = matrix.n;
    if n == 0 {
        return (Vec::new(), ScalingInfo::Applied);
    }
    let mut d = vec![1.0f64; n];

    // 10 iterations is the same cap the dense path uses. Most matrices
    // converge in 2–4 iterations; a few pathological ones need all 10.
    let max_iter = 10;
    let tol = 1e-8;

    // Work buffer for the row ∞-norms.
    let mut row_max = vec![0.0f64; n];

    for _ in 0..max_iter {
        // Reset the row-max buffer.
        for r in row_max.iter_mut() {
            *r = 0.0;
        }

        // Accumulate row maxes by scanning the lower triangle once.
        // Each (i, j) entry with i >= j contributes to row i (via the
        // explicit storage) and to row j (by symmetry).
        //
        // The row-j accumulation is hoisted to a register-resident
        // `col_max` across the inner k-loop, then folded into
        // `row_max[j]` once at column end. The diagonal entry (i == j)
        // is folded into `col_max` only — its `row_max[i]` write would
        // be overwritten by the end-of-column store, so we skip the
        // memory traffic and rely on `col_max` to carry the diagonal's
        // contribution. Off-diagonal entries (i > j) update both
        // `row_max[i]` and `col_max`.
        //
        // Bit-identical to the prior formulation: max(·,·) is
        // associative on non-NaN inputs (every `v` is `|·|` of finite
        // products), so combining via a register accumulator then
        // folding into `row_max[j]` produces the same value as in-place
        // updates in any iteration order.
        for j in 0..n {
            let col_start = matrix.col_ptr[j];
            let col_end = matrix.col_ptr[j + 1];
            let dj = d[j];
            let mut col_max = row_max[j];

            for k in col_start..col_end {
                let i = matrix.row_idx[k];
                let v = (d[i] * matrix.values[k] * dj).abs();
                if i != j && v > row_max[i] {
                    row_max[i] = v;
                }
                if v > col_max {
                    col_max = v;
                }
            }
            row_max[j] = col_max;
        }

        // Update diagonal and check convergence.
        let mut max_dev = 0.0f64;
        for i in 0..n {
            let m = row_max[i];
            if m > 0.0 {
                d[i] /= m.sqrt();
                let dev = (m - 1.0).abs();
                if dev > max_dev {
                    max_dev = dev;
                }
            }
            // Rows with all-zero entries keep d[i] at the current value
            // (initially 1.0) — they are structurally zero and the
            // downstream numeric phase will reject them as singular
            // pivots.
        }

        if max_dev < tol {
            break;
        }
    }

    (d, ScalingInfo::Applied)
}

/// Knight-Ruiz ∞-norm scaling computed on a `SymmetricMatrix`
/// (column-major dense storage with the lower triangle authoritative).
///
/// Produces a bit-exact scaling vector with [`compute_infnorm`] for
/// any matrix whose sparse CSC and dense column-major lower-triangle
/// store the same set of nonzeros (i.e. zeros at "missing" positions
/// in the sparse storage). All such fast-path-gate matrices satisfy
/// this, since `CscMatrix::to_dense_into` writes only the stored
/// entries and leaves the rest as zero — and the inner KR step
/// `max(v, row_max[i])` is a no-op for `v == 0.0`.
///
/// The win over [`compute_infnorm`] on small-dense matrices comes
/// from removing the `row_idx[k]` indirection — the dense loop walks
/// contiguous columns with the compiler able to keep `d[j]` in a
/// register and stride `data[col + i]` linearly. On TRO3X3_0013
/// (n=69, density 0.73) this halves the scaling phase from ~34 µs
/// to ~17 µs — see
/// `dev/results/lever-d3/stage1-stage2-2026-04-19.md` §1 for the
/// pre-change breakdown.
///
/// Intended for the D.3/D.4 dense fast-path; see
/// [`crate::scaling::compute_scaling_dense_fast`].
pub fn compute_infnorm_dense(sym: &SymmetricMatrix) -> (Vec<f64>, ScalingInfo) {
    let n = sym.n;
    if n == 0 {
        return (Vec::new(), ScalingInfo::Applied);
    }
    let mut d = vec![1.0f64; n];

    // Same cap and tolerance as the sparse path.
    let max_iter = 10;
    let tol = 1e-8;

    let mut row_max = vec![0.0f64; n];

    for _ in 0..max_iter {
        for r in row_max.iter_mut() {
            *r = 0.0;
        }

        // Walk the lower triangle column-by-column. Diagonal handled
        // scalar; off-diagonal (i > j) handled via pulp-dispatched
        // lane-wise multiply / abs / max, with `col_max` accumulated
        // in a vector register and reduced once per column. Entries
        // above the lower-triangle gate are zero (set by
        // `to_dense_into`) and never read.
        //
        // Bit-exact with the prior scalar formulation: each lane's
        // multiplies match the scalar order `((d[i] * data) * dj)`,
        // `abs` is per-lane, and max of non-NaN finite values is
        // associative (so a tree reduction over the SIMD lanes
        // produces the same result as a left-fold).
        for j in 0..n {
            let col = j * n;
            let dj = d[j];

            // Diagonal entry: `v_diag = |dj * data[col+j] * dj|`.
            // In the prior code this updated `row_max[j]` via the
            // i-branch (and never the j-branch, gated by `i != j`).
            // Here we fold it into the local accumulator only — the
            // end-of-column store overwrites `row_max[j]` anyway.
            let v_diag = (dj * sym.data[col + j] * dj).abs();
            let mut col_max = row_max[j].max(v_diag);

            // Off-diagonal: lanes i in (j, n). Each lane updates
            // `row_max[i]` (lane-wise max) and contributes to
            // `col_max` (reduced after the sweep).
            if j + 1 < n {
                let d_off = &d[j + 1..n];
                let data_off = &sym.data[col + j + 1..col + n];
                let (_lhs, rm_rhs) = row_max.split_at_mut(j + 1);
                let off_max = scan_offdiag_simd(d_off, data_off, dj, rm_rhs);
                if off_max > col_max {
                    col_max = off_max;
                }
            }

            row_max[j] = col_max;
        }

        let mut max_dev = 0.0f64;
        for i in 0..n {
            let m = row_max[i];
            if m > 0.0 {
                d[i] /= m.sqrt();
                let dev = (m - 1.0).abs();
                if dev > max_dev {
                    max_dev = dev;
                }
            }
        }

        if max_dev < tol {
            break;
        }
    }

    (d, ScalingInfo::Applied)
}

/// SIMD inner kernel for [`compute_infnorm_dense`]. For each contiguous
/// lane, computes `v = |d_off · data_off · dj|`, lane-wise updates
/// `row_max_off ← max(row_max_off, v)`, and returns the max value over
/// all lanes (the column-max contribution from the off-diagonal sweep).
///
/// Dispatched through `pulp::Arch::new()` — picks AVX-512 / AVX2+FMA /
/// SSE2 / NEON / scalar fallback per host CPU. All three slices must
/// be equal length; an empty input returns `0.0`.
///
/// Bit-exact with a scalar loop over the same slice: each lane's
/// `mul → mul → abs` chain matches the scalar order, and the
/// `reduce_max` over non-NaN finite values is associative.
fn scan_offdiag_simd(d_off: &[f64], data_off: &[f64], dj: f64, row_max_off: &mut [f64]) -> f64 {
    assert_eq!(d_off.len(), data_off.len());
    assert_eq!(d_off.len(), row_max_off.len());
    if d_off.is_empty() {
        return 0.0;
    }

    struct K<'a> {
        dj: f64,
        d_off: &'a [f64],
        data_off: &'a [f64],
        row_max_off: &'a mut [f64],
    }

    impl pulp::WithSimd for K<'_> {
        type Output = f64;

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> f64 {
            let Self {
                dj,
                d_off,
                data_off,
                row_max_off,
            } = self;
            let dj_v = simd.splat_f64s(dj);
            let mut col_max_v = simd.splat_f64s(0.0);

            let (d_body, d_tail) = S::as_simd_f64s(d_off);
            let (da_body, da_tail) = S::as_simd_f64s(data_off);
            let (rm_body, rm_tail) = S::as_mut_simd_f64s(row_max_off);

            for ((dv, dav), rmv) in d_body.iter().zip(da_body).zip(rm_body.iter_mut()) {
                let prod = simd.mul_f64s(simd.mul_f64s(*dv, *dav), dj_v);
                let v = simd.abs_f64s(prod);
                *rmv = simd.max_f64s(*rmv, v);
                col_max_v = simd.max_f64s(col_max_v, v);
            }

            let mut col_max = simd.reduce_max_f64s(col_max_v);

            if !d_tail.is_empty() {
                // `partial_load` zero-pads beyond the tail length;
                // `partial_store` writes only the valid prefix. The
                // out-of-range lanes compute `|0·0·dj| = 0`, which is
                // the identity for max — safe to fold into `col_max`.
                let dv = simd.partial_load_f64s(d_tail);
                let dav = simd.partial_load_f64s(da_tail);
                let rmv = simd.partial_load_f64s(rm_tail);
                let prod = simd.mul_f64s(simd.mul_f64s(dv, dav), dj_v);
                let v = simd.abs_f64s(prod);
                let new_rm = simd.max_f64s(rmv, v);
                simd.partial_store_f64s(rm_tail, new_rm);
                let tail_max = simd.reduce_max_f64s(v);
                if tail_max > col_max {
                    col_max = tail_max;
                }
            }

            col_max
        }
    }

    pulp::Arch::new().dispatch(K {
        dj,
        d_off,
        data_off,
        row_max_off,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    /// Diagonal matrix diag(2, 3, 5). The oracle scaling is
    /// d = [1/sqrt(2), 1/sqrt(3), 1/sqrt(5)], so that
    /// D·A·D = diag(1, 1, 1).
    #[test]
    fn diag_3x3() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
        let (d, _info) = compute_infnorm(&m);
        let expected = [1.0 / 2f64.sqrt(), 1.0 / 3f64.sqrt(), 1.0 / 5f64.sqrt()];
        for i in 0..3 {
            assert!(
                (d[i] - expected[i]).abs() < 1e-12,
                "d[{}] = {} != {}",
                i,
                d[i],
                expected[i]
            );
        }
    }

    /// 2x2 matrix [[4, 2], [2, 9]]. Row max [i=0]: max(|4|, |2|) = 4;
    /// row max [i=1]: max(|2|, |9|) = 9. After one KR sweep:
    /// d = [1/2, 1/3]. Check D·A·D row norms converge to 1.
    #[test]
    fn sym_2x2() {
        let m = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[4.0, 2.0, 9.0]).unwrap();
        let (d, _) = compute_infnorm(&m);
        // D·A·D:
        //   [d0*d0*4, d0*d1*2]
        //   [d0*d1*2, d1*d1*9]
        let a00 = d[0] * d[0] * 4.0;
        let a01 = d[0] * d[1] * 2.0;
        let a11 = d[1] * d[1] * 9.0;
        let row0 = a00.abs().max(a01.abs());
        let row1 = a01.abs().max(a11.abs());
        assert!((row0 - 1.0).abs() < 1e-6, "row0 max = {}", row0);
        assert!((row1 - 1.0).abs() < 1e-6, "row1 max = {}", row1);
    }

    /// Dense KR must produce a bit-equal scaling vector to sparse KR
    /// on any matrix whose dense column-major lower triangle contains
    /// exactly the entries that the sparse CSC stores. The `arrow_6x6`
    /// pattern hits both off-diag and degree-1 rows.
    #[test]
    fn dense_matches_sparse_on_arrow_6x6() {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..6 {
            rows.push(j);
            cols.push(j);
            vals.push((j + 2) as f64);
        }
        for j in 0..5 {
            rows.push(5);
            cols.push(j);
            vals.push(1.0);
        }
        let m = CscMatrix::from_triplets(6, &rows, &cols, &vals).unwrap();
        let sym = m.to_dense();
        let (d_sparse, _) = compute_infnorm(&m);
        let (d_dense, _) = compute_infnorm_dense(&sym);
        assert_eq!(d_sparse.len(), d_dense.len());
        for i in 0..d_sparse.len() {
            assert_eq!(
                d_sparse[i].to_bits(),
                d_dense[i].to_bits(),
                "dense-vs-sparse KR parity broke at i={}: sparse={} dense={}",
                i,
                d_sparse[i],
                d_dense[i],
            );
        }
    }

    /// Bit-exact parity on a fully-dense small matrix — the dense
    /// fast-path's target regime.
    #[test]
    fn dense_matches_sparse_on_dense_5x5() {
        // Lower-triangular dense block of a 5×5 symmetric matrix.
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..5 {
            for i in j..5 {
                rows.push(i);
                cols.push(j);
                // Diagonally-dominant non-trivial pattern.
                vals.push(if i == j {
                    10.0 * (i as f64 + 1.0)
                } else {
                    1.0 + 0.1 * (i - j) as f64
                });
            }
        }
        let m = CscMatrix::from_triplets(5, &rows, &cols, &vals).unwrap();
        let sym = m.to_dense();
        let (d_sparse, _) = compute_infnorm(&m);
        let (d_dense, _) = compute_infnorm_dense(&sym);
        for i in 0..5 {
            assert_eq!(
                d_sparse[i].to_bits(),
                d_dense[i].to_bits(),
                "dense KR diverged at i={}: sparse={} dense={}",
                i,
                d_sparse[i],
                d_dense[i],
            );
        }
    }

    /// Arrow matrix: diagonal [2, 3, 4, 5, 6, 7] with (5, 0..=4) = 1.
    /// Row 5 has 5 off-diagonal entries plus the diagonal 7; its
    /// initial ∞-norm is max(1, 1, 1, 1, 1, 7) = 7. The first KR
    /// sweep should shrink d[5] by sqrt(7).
    #[test]
    fn arrow_6x6() {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..6 {
            rows.push(j);
            cols.push(j);
            vals.push((j + 2) as f64);
        }
        for j in 0..5 {
            rows.push(5);
            cols.push(j);
            vals.push(1.0);
        }
        let m = CscMatrix::from_triplets(6, &rows, &cols, &vals).unwrap();
        let (d, _) = compute_infnorm(&m);
        // After KR convergence, every row's max-magnitude entry in
        // D·A·D should be ≈ 1.
        for i in 0..6 {
            let mut row_max = 0.0f64;
            for j in 0..6 {
                // Look up a[i, j] from lower triangle
                let (ii, jj) = if i >= j { (i, j) } else { (j, i) };
                let mut v = 0.0;
                for k in m.col_ptr[jj]..m.col_ptr[jj + 1] {
                    if m.row_idx[k] == ii {
                        v = m.values[k];
                        break;
                    }
                }
                let scaled = (d[i] * v * d[j]).abs();
                if scaled > row_max {
                    row_max = scaled;
                }
            }
            assert!(
                (row_max - 1.0).abs() < 1e-6,
                "row {} max = {}, expected 1",
                i,
                row_max
            );
        }
    }
}
