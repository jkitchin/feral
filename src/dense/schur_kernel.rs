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
//! **Implementation (Step 3):** each function builds a local
//! `pulp::WithSimd` impl that splats `-alpha` once, iterates the
//! full SIMD body with `simd.mul_add_f64s(neg_alpha, src, dst)`
//! (one fused multiply-add per lane), and finishes the trailing
//! scalar tail with `simd.partial_load_f64s` / `partial_store_f64s`
//! (masked loads on AVX-512, sequential on NEON/SSE). The kernel
//! is dispatched through `pulp::Arch::new().dispatch(...)` which
//! picks the best monomorphized variant based on runtime CPU
//! feature detection — AVX-512 / AVX2+FMA / SSE2 / NEON / scalar
//! fallback — at the cost of one dispatch branch per top-level call
//! (not per inner iteration).
//!
//! The scalar fallback path inside pulp guarantees that feral
//! continues to work on architectures without SIMD; no explicit
//! `#[cfg(target_arch)]` gates are needed in this module.

/// `dst[i] -= alpha * src[i]` for `i in 0..dst.len()`.
///
/// Preconditions:
/// - `dst.len() == src.len()`
/// - `dst` and `src` point into disjoint memory regions (enforced by
///   the caller; the Rust borrow checker guarantees this at the call
///   sites in `factor.rs` because `dst` is obtained from
///   `split_at_mut`).
// Phase 2.4.2 Step 3: pulp-dispatched SIMD kernel. The `dead_code`
// allow stays until Step 5 wires this into `do_1x1_update`.
#[allow(dead_code)]
pub(crate) fn axpy_minus(dst: &mut [f64], src: &[f64], alpha: f64) {
    assert_eq!(
        dst.len(),
        src.len(),
        "axpy_minus: dst and src length mismatch"
    );

    struct K<'a> {
        neg_alpha: f64,
        src: &'a [f64],
        dst: &'a mut [f64],
    }

    impl pulp::WithSimd for K<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            let Self {
                neg_alpha,
                src,
                dst,
            } = self;
            let neg_a = simd.splat_f64s(neg_alpha);

            let (src_body, src_tail) = S::as_simd_f64s(src);
            let (dst_body, dst_tail) = S::as_mut_simd_f64s(dst);

            for (d, s) in dst_body.iter_mut().zip(src_body) {
                // d <- (-alpha) * s + d  =  d - alpha * s
                *d = simd.mul_add_f64s(neg_a, *s, *d);
            }

            if !src_tail.is_empty() {
                let s = simd.partial_load_f64s(src_tail);
                let d = simd.partial_load_f64s(dst_tail);
                simd.partial_store_f64s(dst_tail, simd.mul_add_f64s(neg_a, s, d));
            }
        }
    }

    pulp::Arch::new().dispatch(K {
        neg_alpha: -alpha,
        src,
        dst,
    });
}

/// `dst[i] -= alpha0 * src0[i] + alpha1 * src1[i]` for `i in 0..dst.len()`.
///
/// The rank-2 twin of [`axpy_minus`], used inside the 2×2 pivot update
/// in [`super::factor::do_2x2_update`]. Same aliasing precondition:
/// `dst`, `src0`, `src1` must be pairwise disjoint.
// Phase 2.4.2 Step 3: pulp-dispatched SIMD kernel. Same structure as
// `axpy_minus` but with two source columns and two FMAs per lane
// (one for each `-alphaN * srcN` contribution, accumulating into the
// same destination lane). Wired in at Step 5.
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

    struct K<'a> {
        neg_alpha0: f64,
        neg_alpha1: f64,
        src0: &'a [f64],
        src1: &'a [f64],
        dst: &'a mut [f64],
    }

    impl pulp::WithSimd for K<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            let Self {
                neg_alpha0,
                neg_alpha1,
                src0,
                src1,
                dst,
            } = self;
            let na0 = simd.splat_f64s(neg_alpha0);
            let na1 = simd.splat_f64s(neg_alpha1);

            let (s0_body, s0_tail) = S::as_simd_f64s(src0);
            let (s1_body, s1_tail) = S::as_simd_f64s(src1);
            let (d_body, d_tail) = S::as_mut_simd_f64s(dst);

            for ((d, s0), s1) in d_body.iter_mut().zip(s0_body).zip(s1_body) {
                // d <- (-alpha0)*s0 + d
                // d <- (-alpha1)*s1 + d
                let tmp = simd.mul_add_f64s(na0, *s0, *d);
                *d = simd.mul_add_f64s(na1, *s1, tmp);
            }

            if !s0_tail.is_empty() {
                let s0v = simd.partial_load_f64s(s0_tail);
                let s1v = simd.partial_load_f64s(s1_tail);
                let dv = simd.partial_load_f64s(d_tail);
                let tmp = simd.mul_add_f64s(na0, s0v, dv);
                let r = simd.mul_add_f64s(na1, s1v, tmp);
                simd.partial_store_f64s(d_tail, r);
            }
        }
    }

    pulp::Arch::new().dispatch(K {
        neg_alpha0: -alpha0,
        neg_alpha1: -alpha1,
        src0,
        src1,
        dst,
    });
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
