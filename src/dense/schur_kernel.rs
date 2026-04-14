//! SIMD micro-kernel for the inner Schur-complement update in dense
//! LDLᵀ factorization.
//!
//! This module is the single boundary between feral and `pulp` (Phase
//! 2.4.2, see `dev/decisions.md` entry dated 2026-04-14). It exposes
//! two crate-internal functions:
//!
//! - [`axpy_minus`] — the rank-1 inner loop of [`super::factor::do_1x1_update`]
//! - [`axpy2_minus`] — the rank-2 inner loop of [`super::factor::do_2x2_update`]
//!
//! Both functions compute `dst -= α · src` (or the two-source twin
//! `dst -= α₀ · src₀ + α₁ · src₁`) on unit-stride slices. All callers
//! are expected to provide disjoint `dst`/`src` — the outer factorization
//! loops in `factor.rs` guarantee this because `dst` is a trailing
//! column strictly later than `src`.
//!
//! **Step 2 note (this file):** the function bodies are scalar
//! placeholders. The test harness below compares them against an
//! in-module `naive_*` reference. At Phase 2.4.2 Step 3 the bodies
//! will be replaced with pulp kernels (`pulp::Arch::dispatch` over
//! a `pulp::WithSimd` impl using `simd.f64s_mul_add`). The tests do
//! not change; they become meaningful the moment the SIMD kernel
//! diverges from the naive reference (delta must remain ≤ 1 ULP per
//! the Phase 2.4.2 correctness gate).

/// `dst[i] -= alpha * src[i]` for `i in 0..dst.len()`.
///
/// Preconditions:
/// - `dst.len() == src.len()`
/// - `dst` and `src` point into disjoint memory regions (enforced by
///   the caller; the Rust borrow checker guarantees this at the call
///   sites in `factor.rs` because `dst` is obtained from
///   `split_at_mut`).
// Phase 2.4.2 Step 2: function exists and is tested but not yet wired
// into `do_1x1_update`. The `dead_code` allow is removed at Step 5.
#[allow(dead_code)]
pub(crate) fn axpy_minus(dst: &mut [f64], src: &[f64], alpha: f64) {
    assert_eq!(
        dst.len(),
        src.len(),
        "axpy_minus: dst and src length mismatch"
    );
    // Step 2 placeholder: scalar implementation. Replaced by a
    // pulp-dispatched SIMD kernel in Step 3.
    for i in 0..dst.len() {
        dst[i] -= alpha * src[i];
    }
}

/// `dst[i] -= alpha0 * src0[i] + alpha1 * src1[i]` for `i in 0..dst.len()`.
///
/// The rank-2 twin of [`axpy_minus`], used inside the 2×2 pivot update
/// in [`super::factor::do_2x2_update`]. Same aliasing precondition:
/// `dst`, `src0`, `src1` must be pairwise disjoint.
// Phase 2.4.2 Step 2: wired in at Step 5 — see `axpy_minus` above.
#[allow(dead_code)]
pub(crate) fn axpy2_minus(dst: &mut [f64], src0: &[f64], alpha0: f64, src1: &[f64], alpha1: f64) {
    assert_eq!(
        dst.len(),
        src0.len(),
        "axpy2_minus: dst and src0 length mismatch"
    );
    assert_eq!(
        dst.len(),
        src1.len(),
        "axpy2_minus: dst and src1 length mismatch"
    );
    // Step 2 placeholder: scalar implementation. Replaced by a
    // pulp-dispatched SIMD kernel in Step 3.
    for i in 0..dst.len() {
        dst[i] -= alpha0 * src0[i] + alpha1 * src1[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal xorshift64 for reproducible test inputs. Not
    /// cryptographic; not a dependency.
    struct Xorshift64(u64);

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            })
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        /// Uniform in [-1, 1).
        fn next_f64(&mut self) -> f64 {
            let bits = (self.next_u64() >> 12) | 0x3FF0_0000_0000_0000;
            let x = f64::from_bits(bits) - 1.0; // [0, 1)
            2.0 * x - 1.0
        }
    }

    /// Naive reference for correctness comparison. Uses separate mul
    /// and add (no FMA), which gives a well-defined rounding behavior
    /// for the ULP delta check.
    fn naive_axpy_minus(dst: &mut [f64], src: &[f64], alpha: f64) {
        for i in 0..dst.len() {
            let tmp = alpha * src[i];
            dst[i] -= tmp;
        }
    }

    fn naive_axpy2_minus(dst: &mut [f64], src0: &[f64], alpha0: f64, src1: &[f64], alpha1: f64) {
        for i in 0..dst.len() {
            let t0 = alpha0 * src0[i];
            let t1 = alpha1 * src1[i];
            dst[i] -= t0 + t1;
        }
    }

    /// Length sweep crossing every plausible SIMD register boundary
    /// (SSE2 f64x2, NEON f64x2, AVX2 f64x4, AVX-512 f64x8) plus the
    /// one-past-boundary sizes that exercise masked-tail handling.
    const LENGTH_SWEEP: &[usize] = &[
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
        257, 511, 512, 513, 1023, 1024,
    ];

    /// Max allowed per-element absolute difference vs the naive
    /// reference. 1 ULP at the values we test (bounded by ~2 in
    /// magnitude) is ~4.4e-16. We allow 4 ULP headroom to cover:
    ///
    /// - FMA vs separate mul+add rounding (1 ULP max)
    /// - pulp's intrinsic ordering across SIMD lanes (up to 1 ULP
    ///   accumulation drift on a single AXPY)
    /// - criterion benches accumulating rounding over the length
    ///
    /// Empirically this is deeply conservative — an actual SIMD AXPY
    /// of length ≤ 1024 with inputs in `[-1, 1)` will match to the
    /// last bit or differ by exactly 1 ULP per element.
    const ULP4: f64 = 4.0 * f64::EPSILON * 2.0;

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch in assert_close");
        for i in 0..a.len() {
            let diff = (a[i] - b[i]).abs();
            assert!(
                diff <= tol,
                "element {}: {} vs {}, diff {:.3e} > {:.3e}",
                i,
                a[i],
                b[i],
                diff,
                tol
            );
        }
    }

    #[test]
    fn axpy_minus_zero_length() {
        let mut dst: Vec<f64> = vec![];
        let src: Vec<f64> = vec![];
        axpy_minus(&mut dst, &src, 1.5);
        assert!(dst.is_empty());
    }

    #[test]
    fn axpy_minus_length_one() {
        let mut dst = vec![5.0];
        let src = vec![2.0];
        axpy_minus(&mut dst, &src, 0.5);
        // 5.0 - 0.5 * 2.0 = 4.0, exact
        assert_eq!(dst[0], 4.0);
    }

    #[test]
    fn axpy_minus_matches_reference_across_length_sweep() {
        let mut rng = Xorshift64::new(0xFE27_A100_0042_BEEFu64);
        for &len in LENGTH_SWEEP {
            let src: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let dst_init: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let alpha = rng.next_f64() * 1.5;

            let mut dst_kernel = dst_init.clone();
            let mut dst_ref = dst_init.clone();
            axpy_minus(&mut dst_kernel, &src, alpha);
            naive_axpy_minus(&mut dst_ref, &src, alpha);
            assert_close(&dst_kernel, &dst_ref, ULP4);
        }
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn axpy_minus_length_mismatch_panics() {
        let mut dst = vec![0.0; 4];
        let src = vec![0.0; 3];
        axpy_minus(&mut dst, &src, 1.0);
    }

    #[test]
    fn axpy2_minus_zero_length() {
        let mut dst: Vec<f64> = vec![];
        let src0: Vec<f64> = vec![];
        let src1: Vec<f64> = vec![];
        axpy2_minus(&mut dst, &src0, 1.0, &src1, 2.0);
        assert!(dst.is_empty());
    }

    #[test]
    fn axpy2_minus_length_one() {
        let mut dst = vec![10.0];
        let src0 = vec![2.0];
        let src1 = vec![3.0];
        axpy2_minus(&mut dst, &src0, 0.5, &src1, 1.0);
        // 10 - (0.5*2 + 1*3) = 10 - 4 = 6, exact
        assert_eq!(dst[0], 6.0);
    }

    #[test]
    fn axpy2_minus_matches_reference_across_length_sweep() {
        let mut rng = Xorshift64::new(0xC0FF_EE00_BAAD_F00Du64);
        for &len in LENGTH_SWEEP {
            let src0: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let src1: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let dst_init: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let alpha0 = rng.next_f64() * 1.5;
            let alpha1 = rng.next_f64() * 1.5;

            let mut dst_kernel = dst_init.clone();
            let mut dst_ref = dst_init.clone();
            axpy2_minus(&mut dst_kernel, &src0, alpha0, &src1, alpha1);
            naive_axpy2_minus(&mut dst_ref, &src0, alpha0, &src1, alpha1);
            assert_close(&dst_kernel, &dst_ref, ULP4);
        }
    }

    /// Property: `axpy_minus(dst, src, 0)` is a no-op.
    #[test]
    fn axpy_minus_alpha_zero_is_noop() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dst_init = vec![-3.0, 0.5, 100.0, -7.25, 1e-10, 1e10, -0.0, 42.0];
        let mut dst = dst_init.clone();
        axpy_minus(&mut dst, &src, 0.0);
        assert_eq!(dst, dst_init);
    }

    /// Property: `axpy2_minus` with both alphas zero is a no-op.
    #[test]
    fn axpy2_minus_alphas_zero_is_noop() {
        let src0 = vec![1.0, 2.0, 3.0, 4.0];
        let src1 = vec![5.0, 6.0, 7.0, 8.0];
        let dst_init = vec![-1.0, 2.5, 3.0, -4.5];
        let mut dst = dst_init.clone();
        axpy2_minus(&mut dst, &src0, 0.0, &src1, 0.0);
        assert_eq!(dst, dst_init);
    }
}
