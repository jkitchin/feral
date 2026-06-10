//! Block-32 register-resident LDLᵀ kernel (Phase 2.4.3, issue #9).
//!
//! See `dev/plans/phase-2.4.3-block32-kernel.md` and
//! `dev/research/block32-register-resident-kernel.md`.
//!
//! **Status: Step 1 scaffolding.** `factor_block32` currently delegates
//! to `factor_frontal` so the dispatch and test harness can land
//! independently of the SIMD body. The real `update_1x1_block32` /
//! `update_2x2_block32` and the monomorphized BLOCK_SIZE=32 driver
//! arrive in Steps 2–4 of the plan.
//!
//! ## Bit-parity contract
//!
//! At every step of this plan the kernel must produce
//! `f64::to_bits()`-equal `(L, D, perm, subdiag, contrib)` to
//! `factor_frontal` on the same input. The scalar oracle is
//! `factor_frontal` (not `factor_frontal_blocked`) because the
//! unblocked scalar loop's per-element rounding chain
//! (`axpy_minus_unroll4_nofma` applied eagerly to the ground-truth
//! trailing state) is exactly what the eager block-32 update
//! reproduces.
//!
//! ## Rounding discipline
//!
//! Inherits the 2026-04-14 decision (`dev/decisions.md:464`): every
//! SIMD trailing-update lane uses separate `mul_f64s` + `sub_f64s`
//! instead of `mul_add_f64s`. No FMA anywhere in this module.

use crate::dense::factor::{BunchKaufmanParams, FrontalFactors};
use crate::dense::matrix::SymmetricMatrix;
use crate::dense::schur_kernel;
use crate::error::FeralError;

/// Hard-coded block size for this kernel.
///
/// The kernel is monomorphized at BS=32 because that is the dominant
/// front size on KKT chain matrices (see
/// `dev/research/ssids-small-frontal-speed.md` §0). Other block sizes
/// take the existing `factor_frontal_blocked` path.
pub const BLOCK_SIZE: usize = 32;

/// Factor a 32×32 fully-summed front in place using the register-resident
/// block-32 kernel, reusing the caller's pooled `scratch`.
///
/// This is the production entry the multifrontal dispatch
/// (`factor_frontal_blocked_in_place_with_scratch`) routes to for the
/// dominant 32×32 KKT front size. It factors directly into `matrix.data`
/// (treated as scratch — the lower-triangle content is undefined on
/// return) and reuses the caller's `FactorScratch`, paying none of the
/// public `factor_frontal` entry's overhead: a `validate()` re-scan, an
/// n×n working copy, and a throwaway `FactorScratch` (D7,
/// `dev/research/repo-review-2026-06-09.md`). Previously this dispatch
/// delegated to `factor_frontal` and re-paid all three on every 32×32
/// front, defeating the whole purpose of the enclosing W-3a in-place path
/// (issue #13).
///
/// The eager `do_1x1_update` / `do_2x2_update` body already routes to the
/// block-32 SIMD kernels (`update_1x1_block32` quad/dual/single tiling) at
/// `n == 32`, so this path gets the fast kernels while staying bit-exact
/// with `factor_frontal` (the documented oracle):
/// `factor_frontal_in_place_with_scratch` is bit-exact with
/// `factor_frontal` on finite input, and that is what this delegates to.
///
/// **Step 1 stub:** delegates to `factor_frontal_in_place_with_scratch`.
/// Steps 2–4 (issue #9) replace the body with the monomorphized BS=32
/// driver; the bit-parity tests in this module (which exercise this
/// production entry) remain the oracle.
pub(crate) fn factor_block32(
    matrix: &mut SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
    scratch: &mut crate::dense::factor::FactorScratch,
) -> Result<FrontalFactors, FeralError> {
    if matrix.n != BLOCK_SIZE {
        return Err(FeralError::InvalidInput(format!(
            "factor_block32: matrix size {} != BLOCK_SIZE {}",
            matrix.n, BLOCK_SIZE
        )));
    }
    crate::dense::factor::factor_frontal_in_place_with_scratch(
        matrix, ncol, may_delay, params, scratch,
    )
}

/// Rank-1 trailing update for a 1×1 pivot at column `p` of a 32×32
/// column-major dense block.
///
/// Bit-exact analogue of `factor::do_1x1_update(a, 32, p)`. Reads the
/// pivot value from `a[p*32 + p]`, scales the strict-below-diagonal
/// portion of column `p` by `1/d` in place, then applies
/// `a[j*32 + r] -= alpha * a[p*32 + r]` for every `(j, r)` with
/// `p < j < 32` and `j <= r < 32`, where `alpha = a[p*32 + j]_post · d`
/// (the same two-rounding chain `do_1x1_update` uses).
///
/// **Step 3 body (issue #9):** packs trailing destination columns in
/// tiles of four through [`schur_panel_minus_nofma_strided_quad`] with
/// `n_elim = 1`, then a [`schur_panel_minus_nofma_strided_dual`] for
/// the trailing pair (when applicable), then a final
/// [`axpy_minus_unroll4_nofma`] for the last column. Per the quad
/// kernel's bit-exactness contract, each destination column observes
/// the same per-element `dst = sub(dst, mul(alpha, src))` chain as four
/// sequential single-column rank-1 dispatches — i.e. byte-identical to
/// the Step 2 scalar body and to `do_1x1_update`. Verified by the
/// unit tests in this module.
pub(crate) fn update_1x1_block32(a: &mut [f64], p: usize, fma: bool) {
    debug_assert!(a.len() >= BLOCK_SIZE * BLOCK_SIZE);
    debug_assert!(p < BLOCK_SIZE);
    let n = BLOCK_SIZE;
    let d = a[p * n + p];
    if d.abs() == 0.0 {
        return;
    }
    let inv_d = 1.0 / d;
    for i in (p + 1)..n {
        a[p * n + i] *= inv_d;
    }

    // Tile trailing columns in groups of 4, then a final group of 2 if
    // the count is even, then a final group of 1 if the count is odd.
    // The quad/dual kernels were added in Phase 2.4.2/2.4.3 for issue
    // #9 (re-scoped) and are bit-exact per column with the single-
    // column `axpy_minus_unroll4_nofma` reference; see the docstrings
    // and parity sweep in `schur_kernel`.
    let mut j = p + 1;
    while j + 3 < n {
        let alpha0 = a[p * n + j] * d;
        let alpha1 = a[p * n + (j + 1)] * d;
        let alpha2 = a[p * n + (j + 2)] * d;
        let alpha3 = a[p * n + (j + 3)] * d;
        if alpha0 != 0.0 || alpha1 != 0.0 || alpha2 != 0.0 || alpha3 != 0.0 {
            // Carve four disjoint mutable dst slices. Column p (the
            // src) lives in `before`, since p < j.
            let (before, rest) = a.split_at_mut(j * n);
            let (col_j, rest1) = rest.split_at_mut(n);
            let (col_j1, rest2) = rest1.split_at_mut(n);
            let (col_j2, col_j3_and_after) = rest2.split_at_mut(n);
            let dst0 = &mut col_j[j..n];
            let dst1 = &mut col_j1[(j + 1)..n];
            let dst2 = &mut col_j2[(j + 2)..n];
            let dst3 = &mut col_j3_and_after[(j + 3)..n];
            if fma {
                schur_kernel::schur_panel_minus_fma_strided_quad(
                    dst0,
                    dst1,
                    dst2,
                    dst3,
                    before,
                    p,
                    1,
                    n,
                    j,
                    &[alpha0],
                    &[alpha1],
                    &[alpha2],
                    &[alpha3],
                );
            } else {
                schur_kernel::schur_panel_minus_nofma_strided_quad(
                    dst0,
                    dst1,
                    dst2,
                    dst3,
                    before,
                    p,
                    1,
                    n,
                    j,
                    &[alpha0],
                    &[alpha1],
                    &[alpha2],
                    &[alpha3],
                );
            }
        }
        j += 4;
    }

    if j + 1 < n {
        let alpha0 = a[p * n + j] * d;
        let alpha1 = a[p * n + (j + 1)] * d;
        if alpha0 != 0.0 || alpha1 != 0.0 {
            let (before, rest) = a.split_at_mut(j * n);
            let (col_j, after_j) = rest.split_at_mut(n);
            let dst0 = &mut col_j[j..n];
            let dst1 = &mut after_j[(j + 1)..n];
            if fma {
                schur_kernel::schur_panel_minus_fma_strided_dual(
                    dst0,
                    dst1,
                    before,
                    p,
                    1,
                    n,
                    j,
                    &[alpha0],
                    &[alpha1],
                );
            } else {
                schur_kernel::schur_panel_minus_nofma_strided_dual(
                    dst0,
                    dst1,
                    before,
                    p,
                    1,
                    n,
                    j,
                    &[alpha0],
                    &[alpha1],
                );
            }
        }
        j += 2;
    }

    if j < n {
        let alpha = a[p * n + j] * d;
        if alpha != 0.0 {
            let (before, rest) = a.split_at_mut(j * n);
            let src = &before[p * n + j..p * n + n];
            let dst = &mut rest[j..n];
            if fma {
                schur_kernel::axpy_minus_unroll4(dst, src, alpha);
            } else {
                schur_kernel::axpy_minus_unroll4_nofma(dst, src, alpha);
            }
        }
    }
}

/// Rank-2 trailing update for a 2×2 pivot at columns `p`, `p+1` of a
/// 32×32 column-major dense block.
///
/// Bit-exact analogue of `factor::do_2x2_update(a, 32, p, d11, d21,
/// d22)`. Computes the 2×2 inverse from `(d11, d21, d22)`, scales
/// the strict-below-diagonal portion of columns `p` and `p+1` in
/// place by that inverse, then applies the rank-2 update for every
/// trailing column `j` in `(p+2)..32`.
///
/// **Status:** scalar reference (per-column `axpy2_minus_unroll4_nofma`).
/// Step 4 of the plan (4-dst-column packing) is deferred — the quad
/// kernel's per-q sequential `sub(sub(dst, m0), m1)` rounding chain is
/// not bit-exact with `axpy2_minus_unroll4_nofma`'s fused
/// `sub(dst, add(m0, m1))` chain, so the rank-2 4-column kernel needs
/// a fresh pulp dispatch. Tracked separately as Step 4 follow-up; for
/// now this body remains the scalar path used by every 2×2 pivot.
pub(crate) fn update_2x2_block32(a: &mut [f64], p: usize, d11: f64, d21: f64, d22: f64, fma: bool) {
    debug_assert!(a.len() >= BLOCK_SIZE * BLOCK_SIZE);
    debug_assert!(p + 1 < BLOCK_SIZE);
    let n = BLOCK_SIZE;
    let det = d11 * d22 - d21 * d21;
    if det.abs() == 0.0 {
        return;
    }
    let inv_det = 1.0 / det;

    for i in (p + 2)..n {
        let a_ik = a[p * n + i];
        let a_ik1 = a[(p + 1) * n + i];
        a[p * n + i] = (d22 * a_ik - d21 * a_ik1) * inv_det;
        a[(p + 1) * n + i] = (d11 * a_ik1 - d21 * a_ik) * inv_det;
    }

    for j in (p + 2)..n {
        let l_j0 = a[p * n + j];
        let l_j1 = a[(p + 1) * n + j];
        let dl_j0 = d11 * l_j0 + d21 * l_j1;
        let dl_j1 = d21 * l_j0 + d22 * l_j1;
        let (before, rest) = a.split_at_mut(j * n);
        let src0 = &before[p * n + j..p * n + n];
        let src1 = &before[(p + 1) * n + j..(p + 1) * n + n];
        let dst = &mut rest[j..n];
        if fma {
            schur_kernel::axpy2_minus_unroll4(dst, src0, dl_j0, src1, dl_j1);
        } else {
            schur_kernel::axpy2_minus_unroll4_nofma(dst, src0, dl_j0, src1, dl_j1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::factor::{factor_frontal, FactorScratch};

    /// Build a 32×32 lower-triangular `SymmetricMatrix` from a
    /// row-major dense slice. Only the lower triangle is read; the
    /// upper is ignored by every consumer.
    fn from_lower(rows: &[[f64; BLOCK_SIZE]; BLOCK_SIZE]) -> SymmetricMatrix {
        let mut data = vec![0.0f64; BLOCK_SIZE * BLOCK_SIZE];
        for j in 0..BLOCK_SIZE {
            for i in j..BLOCK_SIZE {
                data[j * BLOCK_SIZE + i] = rows[i][j];
            }
        }
        SymmetricMatrix {
            n: BLOCK_SIZE,
            data,
        }
    }

    /// Construct a 32×32 indefinite symmetric matrix with a fixed
    /// seed. Diagonal entries are pseudo-random in `[-1, 1)`, off-
    /// diagonal in `[-0.5, 0.5)`. Diagonally non-dominant so the BK
    /// pivot rules genuinely fire 1×1 / swap-1×1 / 2×2 branches.
    fn seeded_indefinite_32x32() -> SymmetricMatrix {
        // Splitmix64 — deterministic, no external crate, identical
        // output across architectures.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || -> f64 {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            // Map to [0, 1) by taking the high 53 bits.
            ((z >> 11) as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
        };
        let mut rows = [[0.0f64; BLOCK_SIZE]; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            for j in 0..=i {
                if i == j {
                    rows[i][j] = 2.0 * next() - 1.0;
                } else {
                    rows[i][j] = next() - 0.5;
                }
            }
        }
        from_lower(&rows)
    }

    fn assert_factors_bit_equal(actual: &FrontalFactors, expected: &FrontalFactors) {
        assert_eq!(actual.nrow, expected.nrow, "nrow");
        assert_eq!(actual.ncol, expected.ncol, "ncol");
        assert_eq!(actual.nelim, expected.nelim, "nelim");
        assert_eq!(actual.n_delayed, expected.n_delayed, "n_delayed");
        assert_eq!(actual.inertia, expected.inertia, "inertia");
        assert_eq!(actual.perm, expected.perm, "perm");
        assert_eq!(actual.perm_inv, expected.perm_inv, "perm_inv");

        assert_eq!(actual.l.len(), expected.l.len(), "L length");
        for k in 0..actual.l.len() {
            assert_eq!(
                actual.l[k].to_bits(),
                expected.l[k].to_bits(),
                "L[{k}] mismatch: actual={} expected={}",
                actual.l[k],
                expected.l[k]
            );
        }

        assert_eq!(actual.d_diag.len(), expected.d_diag.len(), "d_diag length");
        for k in 0..actual.d_diag.len() {
            assert_eq!(
                actual.d_diag[k].to_bits(),
                expected.d_diag[k].to_bits(),
                "d_diag[{k}] mismatch"
            );
        }

        assert_eq!(
            actual.d_subdiag.len(),
            expected.d_subdiag.len(),
            "d_subdiag length"
        );
        for k in 0..actual.d_subdiag.len() {
            assert_eq!(
                actual.d_subdiag[k].to_bits(),
                expected.d_subdiag[k].to_bits(),
                "d_subdiag[{k}] mismatch"
            );
        }

        assert_eq!(
            actual.contrib.len(),
            expected.contrib.len(),
            "contrib length"
        );
        for k in 0..actual.contrib.len() {
            assert_eq!(
                actual.contrib[k].to_bits(),
                expected.contrib[k].to_bits(),
                "contrib[{k}] mismatch"
            );
        }
    }

    /// Construct two independent copies of the same lower triangle so
    /// that scalar and block-32 paths each get their own scratch.
    fn dup_lower(src: &SymmetricMatrix) -> (SymmetricMatrix, SymmetricMatrix) {
        let a = SymmetricMatrix {
            n: src.n,
            data: src.data.clone(),
        };
        let b = SymmetricMatrix {
            n: src.n,
            data: src.data.clone(),
        };
        (a, b)
    }

    /// Build a 32×32 dense lower-triangular block (column-major) seeded
    /// from splitmix64 — same generator as `seeded_indefinite_32x32`
    /// but returns the raw `[f64; 1024]` for direct primitive tests
    /// (no `SymmetricMatrix` wrapper).
    fn seeded_block_1024(seed: u64) -> Vec<f64> {
        let mut state: u64 = seed;
        let mut next = || -> f64 {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            ((z >> 11) as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
        };
        let mut data = vec![0.0f64; BLOCK_SIZE * BLOCK_SIZE];
        for j in 0..BLOCK_SIZE {
            for i in j..BLOCK_SIZE {
                let v = if i == j {
                    2.0 * next() - 1.0
                } else {
                    next() - 0.5
                };
                data[j * BLOCK_SIZE + i] = v;
            }
        }
        data
    }

    fn assert_blocks_bit_equal(actual: &[f64], expected: &[f64], context: &str) {
        assert_eq!(actual.len(), expected.len(), "{}: block length", context);
        for k in 0..actual.len() {
            assert_eq!(
                actual[k].to_bits(),
                expected[k].to_bits(),
                "{}: a[{k}] mismatch (actual={}, expected={})",
                context,
                actual[k],
                expected[k],
            );
        }
    }

    /// `update_1x1_block32(a, p)` produces a block byte-identical to
    /// `factor::do_1x1_update(a, 32, p)`. Under the Step 2a scalar body
    /// this is bit-equal by construction; the same assertion catches
    /// any divergence introduced by Step 3's SIMD body.
    #[test]
    fn update_1x1_block32_matches_do_1x1_update_at_p0() {
        let a0 = seeded_block_1024(0xA5A5_5A5A_DEAD_BEEF);
        let mut a_scalar = a0.clone();
        let mut a_block = a0;
        crate::dense::factor::do_1x1_update(&mut a_scalar, BLOCK_SIZE, 0, false);
        update_1x1_block32(&mut a_block, 0, false);
        assert_blocks_bit_equal(&a_block, &a_scalar, "update_1x1_block32 at p=0");
    }

    /// Same parity check at p=5 — a mid-block pivot. Stages the matrix
    /// by running `do_1x1_update` for pivots 0..5 first so the inputs
    /// to the primitive at p=5 are a realistic mid-factorization state.
    #[test]
    fn update_1x1_block32_matches_do_1x1_update_at_p5() {
        let a0 = seeded_block_1024(0x1234_5678_9ABC_DEF0);
        // Stage: run pivots 0..5 with the scalar primitive.
        let mut a_staged = a0;
        for p in 0..5 {
            crate::dense::factor::do_1x1_update(&mut a_staged, BLOCK_SIZE, p, false);
        }
        let mut a_scalar = a_staged.clone();
        let mut a_block = a_staged;
        crate::dense::factor::do_1x1_update(&mut a_scalar, BLOCK_SIZE, 5, false);
        update_1x1_block32(&mut a_block, 5, false);
        assert_blocks_bit_equal(&a_block, &a_scalar, "update_1x1_block32 at p=5");
    }

    /// `update_1x1_block32` near the trailing edge (p=30, only one
    /// remaining column) — exercises the small-trail path that Step 3
    /// must still handle correctly.
    #[test]
    fn update_1x1_block32_matches_do_1x1_update_at_p30() {
        let a0 = seeded_block_1024(0xF00D_FACE_C0FF_EE00);
        let mut a_staged = a0;
        for p in 0..30 {
            crate::dense::factor::do_1x1_update(&mut a_staged, BLOCK_SIZE, p, false);
        }
        let mut a_scalar = a_staged.clone();
        let mut a_block = a_staged;
        crate::dense::factor::do_1x1_update(&mut a_scalar, BLOCK_SIZE, 30, false);
        update_1x1_block32(&mut a_block, 30, false);
        assert_blocks_bit_equal(&a_block, &a_scalar, "update_1x1_block32 at p=30");
    }

    /// `update_1x1_block32` with d == 0 is a no-op (early return),
    /// matching `do_1x1_update`. Verifies the early-exit branch is
    /// byte-equivalent.
    #[test]
    fn update_1x1_block32_zero_pivot_is_noop() {
        let a0 = seeded_block_1024(0xBADD_F00D_DEAD_BEEF);
        let mut a_scalar = a0.clone();
        let mut a_block = a0;
        // Zero out the pivot diagonal at p=2.
        a_scalar[2 * BLOCK_SIZE + 2] = 0.0;
        a_block[2 * BLOCK_SIZE + 2] = 0.0;
        crate::dense::factor::do_1x1_update(&mut a_scalar, BLOCK_SIZE, 2, false);
        update_1x1_block32(&mut a_block, 2, false);
        assert_blocks_bit_equal(&a_block, &a_scalar, "update_1x1_block32 zero pivot");
    }

    /// `update_2x2_block32(a, p, d11, d21, d22)` byte-matches
    /// `factor::do_2x2_update(a, 32, p, d11, d21, d22)` at p=0 with
    /// arbitrary 2×2 inverse coefficients.
    #[test]
    fn update_2x2_block32_matches_do_2x2_update_at_p0() {
        let a0 = seeded_block_1024(0xCAFE_BABE_1234_5678);
        let mut a_scalar = a0.clone();
        let mut a_block = a0;
        let d11 = 2.5;
        let d21 = -0.75;
        let d22 = 1.125;
        crate::dense::factor::do_2x2_update(&mut a_scalar, BLOCK_SIZE, 0, d11, d21, d22, false);
        update_2x2_block32(&mut a_block, 0, d11, d21, d22, false);
        assert_blocks_bit_equal(&a_block, &a_scalar, "update_2x2_block32 at p=0");
    }

    /// `update_2x2_block32` mid-block (p=10) and near edge (p=28).
    #[test]
    fn update_2x2_block32_matches_do_2x2_update_at_p10_and_p28() {
        let a0 = seeded_block_1024(0x0123_4567_89AB_CDEF);
        let (d11, d21, d22) = (-1.5, 0.25, 0.875);

        let mut a_scalar = a0.clone();
        let mut a_block = a0.clone();
        crate::dense::factor::do_2x2_update(&mut a_scalar, BLOCK_SIZE, 10, d11, d21, d22, false);
        update_2x2_block32(&mut a_block, 10, d11, d21, d22, false);
        assert_blocks_bit_equal(&a_block, &a_scalar, "update_2x2_block32 at p=10");

        let mut a_scalar2 = a0.clone();
        let mut a_block2 = a0;
        crate::dense::factor::do_2x2_update(&mut a_scalar2, BLOCK_SIZE, 28, d11, d21, d22, false);
        update_2x2_block32(&mut a_block2, 28, d11, d21, d22, false);
        assert_blocks_bit_equal(&a_block2, &a_scalar2, "update_2x2_block32 at p=28");
    }

    /// Singular 2×2 (det == 0) is a no-op in both paths.
    #[test]
    fn update_2x2_block32_singular_is_noop() {
        let a0 = seeded_block_1024(0xDEAD_BEEF_FEED_FACE);
        let mut a_scalar = a0.clone();
        let mut a_block = a0;
        // d11*d22 - d21^2 == 0 → det = 0.
        let (d11, d21, d22) = (1.0, 1.0, 1.0);
        crate::dense::factor::do_2x2_update(&mut a_scalar, BLOCK_SIZE, 5, d11, d21, d22, false);
        update_2x2_block32(&mut a_block, 5, d11, d21, d22, false);
        assert_blocks_bit_equal(&a_block, &a_scalar, "update_2x2_block32 singular");
    }

    #[test]
    fn factor_block32_rejects_wrong_size() {
        let mut m = SymmetricMatrix::zeros(16);
        let params = BunchKaufmanParams::default();
        let mut scratch = FactorScratch::new();
        let res = factor_block32(&mut m, 16, false, &params, &mut scratch);
        assert!(res.is_err());
    }

    /// Smoke test: on a diagonal SPD matrix, the block-32 path and
    /// the scalar oracle agree bit-for-bit. Under the Step 1 stub
    /// this is tautological; the same assertion is the load-bearing
    /// regression test once the kernel body lands in Step 2+.
    #[test]
    fn factor_block32_diagonal_spd_matches_scalar() {
        let mut rows = [[0.0f64; BLOCK_SIZE]; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            rows[i][i] = (i as f64) + 1.0;
        }
        let src = from_lower(&rows);
        let (a, mut b) = dup_lower(&src);
        let params = BunchKaufmanParams::default();
        let mut scratch = FactorScratch::new();
        let scalar = factor_frontal(&a, BLOCK_SIZE, false, &params).expect("scalar");
        let block =
            factor_block32(&mut b, BLOCK_SIZE, false, &params, &mut scratch).expect("block32");
        assert_factors_bit_equal(&block, &scalar);
    }

    /// Bit-parity on a seeded random indefinite 32×32 matrix. This is
    /// the harness Step 2/3/4 commits will re-run against the real
    /// kernel body. Under the Step 1 stub it passes trivially.
    #[test]
    fn factor_block32_seeded_indefinite_matches_scalar() {
        let src = seeded_indefinite_32x32();
        let (a, mut b) = dup_lower(&src);
        let params = BunchKaufmanParams::default();
        let mut scratch = FactorScratch::new();
        let scalar = factor_frontal(&a, BLOCK_SIZE, false, &params).expect("scalar");
        let block =
            factor_block32(&mut b, BLOCK_SIZE, false, &params, &mut scratch).expect("block32");
        assert_factors_bit_equal(&block, &scalar);
    }
}
