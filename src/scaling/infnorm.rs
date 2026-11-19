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

    // Issue #237: on a rayon worker and above a size threshold, each
    // pass runs over column chunks in parallel. Bit-identical to the
    // serial pass (see `ParallelSweep`).
    let mut par = ParallelSweep::plan(matrix);

    for _ in 0..max_iter {
        match par.as_mut() {
            Some(p) => p.sweep(matrix, &d, &mut row_max),
            None => sweep_serial(matrix, &d, &mut row_max),
        }

        // Update diagonal and check convergence.
        let mut max_dev = 0.0f64;
        for i in 0..n {
            let m = row_max[i];
            if m > 0.0 {
                d[i] = crate::scaling::kr_guarded_update(d[i], m);
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

/// One serial pass: `row_max[i] = max_j |d[i]·a[i,j]·d[j]|` over the full
/// symmetric matrix, from its lower-triangle storage.
fn sweep_serial(matrix: &CscMatrix, d: &[f64], row_max: &mut [f64]) {
    let n = matrix.n;
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
}

/// Issue #237: the per-pass row-max scan split over contiguous column
/// ranges, one private buffer per range, merged per row.
///
/// Column range `[j0, j1)` of lower-triangle storage writes only rows
/// `i >= j0`, so its buffer covers rows `[j0, n)`. Each range runs the
/// serial scan's loop body unchanged against its own buffer; the merge
/// takes, per row, the maximum over the ranges that start at or before it.
///
/// Bit-identical to [`sweep_serial`]: every update in both is
/// `if v > acc { acc = v }` from `0.0`. For non-NaN `v >= 0` that is the
/// exact maximum of the contributions, whatever the order or grouping;
/// a NaN `v` never wins a `>` comparison, so it is ignored in every
/// order as well. The ranges partition the columns, so each row sees the
/// same multiset of contributions as in the serial pass.
struct ParallelSweep {
    /// Range boundaries: range `t` is columns `starts[t]..starts[t + 1]`.
    starts: Vec<usize>,
    /// `bufs[t][i - starts[t]]` is range `t`'s partial max for row `i`.
    bufs: Vec<Vec<f64>>,
}

impl ParallelSweep {
    /// Below this many stored entries a pass is a few milliseconds at
    /// most; stay serial.
    const MIN_NNZ: usize = 1 << 18;
    /// Lower bound on stored entries per range, so tiny ranges do not
    /// pay a buffer and a merge term for no work.
    const MIN_NNZ_PER_RANGE: usize = 1 << 16;
    /// Rows per merge task.
    const MERGE_CHUNK: usize = 1 << 13;

    /// `Some` only on a rayon worker (inside a `Solver`'s pool or the
    /// parallel driver) with more than one thread and enough entries.
    fn plan(matrix: &CscMatrix) -> Option<Self> {
        let n = matrix.n;
        let nnz = matrix.col_ptr.get(n).copied().unwrap_or(0);
        if nnz < Self::MIN_NNZ || rayon::current_thread_index().is_none() {
            return None;
        }
        let ranges = rayon::current_num_threads()
            .min(nnz / Self::MIN_NNZ_PER_RANGE)
            .max(1);
        if ranges < 2 {
            return None;
        }
        Some(Self::with_ranges(matrix, ranges))
    }

    /// Split the columns into `ranges` contiguous ranges of roughly equal
    /// stored-entry count. Some ranges may be empty (one heavy column).
    fn with_ranges(matrix: &CscMatrix, ranges: usize) -> Self {
        let n = matrix.n;
        let nnz = matrix.col_ptr.get(n).copied().unwrap_or(0);
        let mut starts = Vec::with_capacity(ranges + 1);
        starts.push(0usize);
        let mut j = 0usize;
        for t in 1..ranges {
            let target = nnz / ranges * t;
            while j < n && matrix.col_ptr[j] < target {
                j += 1;
            }
            starts.push(j);
        }
        starts.push(n);
        let bufs = (0..ranges).map(|t| vec![0.0f64; n - starts[t]]).collect();
        Self { starts, bufs }
    }

    fn sweep(&mut self, matrix: &CscMatrix, d: &[f64], row_max: &mut [f64]) {
        use rayon::prelude::*;
        let starts = &self.starts;
        self.bufs.par_iter_mut().enumerate().for_each(|(t, buf)| {
            let j0 = starts[t];
            let j1 = starts[t + 1];
            buf.fill(0.0);
            // Same loop body as `sweep_serial`, rows offset by `j0`.
            for j in j0..j1 {
                let col_start = matrix.col_ptr[j];
                let col_end = matrix.col_ptr[j + 1];
                let dj = d[j];
                let mut col_max = buf[j - j0];
                for k in col_start..col_end {
                    let i = matrix.row_idx[k];
                    let v = (d[i] * matrix.values[k] * dj).abs();
                    if i != j && v > buf[i - j0] {
                        buf[i - j0] = v;
                    }
                    if v > col_max {
                        col_max = v;
                    }
                }
                buf[j - j0] = col_max;
            }
        });
        let bufs = &self.bufs;
        row_max
            .par_chunks_mut(Self::MERGE_CHUNK)
            .enumerate()
            .for_each(|(c, out)| {
                let base = c * Self::MERGE_CHUNK;
                for (off, slot) in out.iter_mut().enumerate() {
                    let i = base + off;
                    let mut m = 0.0f64;
                    for (t, buf) in bufs.iter().enumerate() {
                        let j0 = starts[t];
                        if j0 > i {
                            break;
                        }
                        let v = buf[i - j0];
                        if v > m {
                            m = v;
                        }
                    }
                    *slot = m;
                }
            });
    }
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
                d[i] = crate::scaling::kr_guarded_update(d[i], m);
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

    /// Issue #119: the guarded Knight–Ruiz step applies `d/√m` only when it
    /// stays finite and positive, else holds `d`. Healthy inputs are unchanged;
    /// overflow (`d/√m → Inf`), underflow-to-zero (`m = Inf`), and `NaN` inputs
    /// all return the input `d` instead of poisoning it.
    #[test]
    fn kr_guarded_update_guards_extremes() {
        // Healthy: exact bare division.
        assert_eq!(crate::scaling::kr_guarded_update(1.0, 4.0), 0.5);
        assert_eq!(crate::scaling::kr_guarded_update(6.0, 9.0), 2.0);
        // Overflow: 1e300/√(1e-40) = 1e320 → Inf → held.
        assert_eq!(crate::scaling::kr_guarded_update(1e300, 1e-40), 1e300);
        // m = Inf ⇒ d/Inf = 0 (not > 0) → held.
        assert_eq!(crate::scaling::kr_guarded_update(1.0, f64::INFINITY), 1.0);
        // m = NaN ⇒ NaN (not finite) → held.
        assert_eq!(crate::scaling::kr_guarded_update(1.0, f64::NAN), 1.0);
    }

    /// Issue #119: a subnormal off-diagonal coupling with no diagonal in row 0
    /// drives the unguarded iteration to `d = [NaN, 0.0]` on a finite input
    /// (reproduced offline). The guarded iteration must return all-finite,
    /// strictly-positive factors (no silent NaN/0 poisoning the factorization),
    /// and the scaled entry must stay finite.
    #[test]
    fn kr_subnormal_coupling_stays_finite() {
        // Lower triangle: (1,0) = 1e-320 (subnormal), (1,1) = 1; no (0,0), so
        // row 0's only entry is the subnormal coupling.
        let m = CscMatrix::from_triplets(2, &[1, 1], &[0, 1], &[1e-320, 1.0]).unwrap();
        let (d, info) = compute_infnorm(&m);
        assert!(matches!(info, ScalingInfo::Applied));
        for (i, &di) in d.iter().enumerate() {
            assert!(
                di.is_finite() && di > 0.0,
                "d[{i}] = {di} must be finite and strictly positive"
            );
        }
        let scaled = (d[1] * 1e-320 * d[0]).abs();
        assert!(scaled.is_finite(), "scaled entry {scaled} must be finite");
    }

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

    /// Issue #237 fixture: banded lower triangle (bandwidth 12) plus an
    /// arrow (dense first column), an all-zero row/column, explicit
    /// zeros, and values spanning ~12 orders of magnitude so the
    /// scaling vector is far from 1. nnz > `ParallelSweep::MIN_NNZ`.
    fn issue237_fixture() -> CscMatrix {
        let n = 24_000usize;
        let zero_row = 777usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            if j == zero_row {
                continue;
            }
            for i in j..(j + 12).min(n) {
                if i == zero_row {
                    continue;
                }
                let e = ((i * 7 + j * 13) % 25) as i32 - 12;
                let v = if (i + j) % 97 == 0 {
                    0.0
                } else {
                    (1.0 + ((i ^ j) % 5) as f64) * 10f64.powi(e / 2)
                };
                let v = if i != j && (i + 2 * j) % 3 == 0 {
                    -v
                } else {
                    v
                };
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
            if j > 12 && j != zero_row && j % 3 == 0 {
                rows.push(j);
                cols.push(0);
                vals.push(1e3 / (1.0 + j as f64));
            }
        }
        CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
    }

    /// Issue #237: one chunked pass equals the serial pass bit for bit at
    /// several range counts (range boundaries move with the count),
    /// iterated through ten passes with the real `d` updates in between.
    /// Oracle: `sweep_serial`, the pre-existing pass.
    #[test]
    fn chunked_sweep_matches_serial_bitwise_issue_237() {
        let m = issue237_fixture();
        assert!(m.nnz() > ParallelSweep::MIN_NNZ);
        let n = m.n;
        for ranges in [1usize, 2, 3, 4, 7] {
            let mut par = ParallelSweep::with_ranges(&m, ranges);
            let mut d = vec![1.0f64; n];
            let mut serial = vec![0.0f64; n];
            let mut chunked = vec![f64::NAN; n];
            for pass in 0..10 {
                sweep_serial(&m, &d, &mut serial);
                par.sweep(&m, &d, &mut chunked);
                let sb: Vec<u64> = serial.iter().map(|v| v.to_bits()).collect();
                let cb: Vec<u64> = chunked.iter().map(|v| v.to_bits()).collect();
                assert_eq!(sb, cb, "ranges={ranges} pass={pass}");
                for i in 0..n {
                    if serial[i] > 0.0 {
                        d[i] = crate::scaling::kr_guarded_update(d[i], serial[i]);
                    }
                }
            }
            assert_eq!(serial[777], 0.0, "zero row stays zero");
        }
    }

    /// Issue #237: `compute_infnorm` takes the parallel pass on a rayon
    /// worker and the serial pass elsewhere; the scaling vectors are
    /// bit-identical, and not trivially all-ones.
    #[test]
    fn compute_infnorm_on_worker_matches_serial_issue_237() {
        let m = issue237_fixture();
        assert!(rayon::current_thread_index().is_none());
        assert!(ParallelSweep::plan(&m).is_none(), "serial off a worker");
        let (serial, _) = compute_infnorm(&m);
        assert!(serial.iter().any(|&x| (x - 1.0).abs() > 1e-3));
        for threads in [2usize, 3, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let par = pool.install(|| {
                assert!(ParallelSweep::plan(&m).is_some(), "parallel on a worker");
                compute_infnorm(&m).0
            });
            let sb: Vec<u64> = serial.iter().map(|v| v.to_bits()).collect();
            let pb: Vec<u64> = par.iter().map(|v| v.to_bits()).collect();
            assert_eq!(sb, pb, "threads={threads}");
        }
    }
}
