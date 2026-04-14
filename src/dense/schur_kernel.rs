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
pub fn axpy_minus(dst: &mut [f64], src: &[f64], alpha: f64) {
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
pub fn axpy2_minus(dst: &mut [f64], src0: &[f64], alpha0: f64, src1: &[f64], alpha1: f64) {
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

// ---------------------------------------------------------------------
// Phase 2.4.2 Step 4 diagnostic: "direct-monomorphized" variants that
// bypass `pulp::Arch::new().dispatch()` and call `WithSimd::with_simd`
// directly on a pre-constructed Simd token. On aarch64 we use the
// baseline `aarch64::Neon` (NEON is ARMv8 mandatory, so
// `Neon::new_unchecked()` is safe). These exist only to test the
// hypothesis that the pulp dispatch + `#[target_feature]` trampoline
// is the source of the NEON bench regression. If they close the gap
// vs the `Arch::dispatch()` path, Step 5 will be rewritten to use the
// direct-token pattern instead of per-call dispatch.
//
// On non-aarch64 targets these variants fall back to `pulp::Scalar`
// (no SIMD), which is intentionally unhelpful — the point of the
// diagnostic is only the aarch64 NEON path, and we don't want a
// misleading x86 measurement from a non-representative Simd choice.

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub fn axpy_minus_direct(dst: &mut [f64], src: &[f64], alpha: f64) {
    assert_eq!(
        dst.len(),
        src.len(),
        "axpy_minus_direct: dst and src length mismatch"
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
                *d = simd.mul_add_f64s(neg_a, *s, *d);
            }

            if !src_tail.is_empty() {
                let s = simd.partial_load_f64s(src_tail);
                let d = simd.partial_load_f64s(dst_tail);
                simd.partial_store_f64s(dst_tail, simd.mul_add_f64s(neg_a, s, d));
            }
        }
    }

    // SAFETY: on aarch64/ARMv8, NEON is a baseline feature guaranteed
    // by the architecture, so `Neon::new_unchecked` is always sound.
    const NEON: pulp::aarch64::Neon = unsafe { pulp::aarch64::Neon::new_unchecked() };
    use pulp::WithSimd;
    K {
        neg_alpha: -alpha,
        src,
        dst,
    }
    .with_simd(NEON);
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub fn axpy2_minus_direct(dst: &mut [f64], src0: &[f64], alpha0: f64, src1: &[f64], alpha1: f64) {
    assert_eq!(
        dst.len(),
        src0.len(),
        "axpy2_minus_direct: dst and src0 length mismatch"
    );
    assert_eq!(
        dst.len(),
        src1.len(),
        "axpy2_minus_direct: dst and src1 length mismatch"
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

    // SAFETY: NEON is baseline on aarch64/ARMv8.
    const NEON: pulp::aarch64::Neon = unsafe { pulp::aarch64::Neon::new_unchecked() };
    use pulp::WithSimd;
    K {
        neg_alpha0: -alpha0,
        neg_alpha1: -alpha1,
        src0,
        src1,
        dst,
    }
    .with_simd(NEON);
}

// ---------------------------------------------------------------------
// Phase 2.4.2 Step 4b diagnostic: 4-way unrolled variants. Same
// direct-NEON dispatch as the `_direct` variants, but the SIMD body
// processes 4 lane-vectors per iteration with four independent
// accumulators. Targets the specific gap measured in Step 4: at
// L >= 256 the single-accumulator pulp kernel loses 30-40% to
// rustc's autovectorized scalar loop, which LLVM unrolls and feeds
// through multiple NEON FMA pipes. Explicit unrolling restores ILP
// the single-lane loop body was missing.

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub fn axpy_minus_unroll4(dst: &mut [f64], src: &[f64], alpha: f64) {
    assert_eq!(
        dst.len(),
        src.len(),
        "axpy_minus_unroll4: dst and src length mismatch"
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

            // 4-way unrolled main loop. Four independent FMA chains
            // let the M-series NEON FMA pipes issue in parallel.
            let mut d_chunks = dst_body.chunks_exact_mut(4);
            let mut s_chunks = src_body.chunks_exact(4);
            for (dc, sc) in (&mut d_chunks).zip(&mut s_chunks) {
                let r0 = simd.mul_add_f64s(neg_a, sc[0], dc[0]);
                let r1 = simd.mul_add_f64s(neg_a, sc[1], dc[1]);
                let r2 = simd.mul_add_f64s(neg_a, sc[2], dc[2]);
                let r3 = simd.mul_add_f64s(neg_a, sc[3], dc[3]);
                dc[0] = r0;
                dc[1] = r1;
                dc[2] = r2;
                dc[3] = r3;
            }

            // Cleanup: 0-3 leftover full-lane vectors.
            let d_rem = d_chunks.into_remainder();
            let s_rem = s_chunks.remainder();
            for (d, s) in d_rem.iter_mut().zip(s_rem) {
                *d = simd.mul_add_f64s(neg_a, *s, *d);
            }

            // Masked tail (< one full lane).
            if !src_tail.is_empty() {
                let s = simd.partial_load_f64s(src_tail);
                let d = simd.partial_load_f64s(dst_tail);
                simd.partial_store_f64s(dst_tail, simd.mul_add_f64s(neg_a, s, d));
            }
        }
    }

    // SAFETY: NEON is a baseline feature on aarch64/ARMv8.
    const NEON: pulp::aarch64::Neon = unsafe { pulp::aarch64::Neon::new_unchecked() };
    use pulp::WithSimd;
    K {
        neg_alpha: -alpha,
        src,
        dst,
    }
    .with_simd(NEON);
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub fn axpy2_minus_unroll4(dst: &mut [f64], src0: &[f64], alpha0: f64, src1: &[f64], alpha1: f64) {
    assert_eq!(
        dst.len(),
        src0.len(),
        "axpy2_minus_unroll4: dst and src0 length mismatch"
    );
    assert_eq!(
        dst.len(),
        src1.len(),
        "axpy2_minus_unroll4: dst and src1 length mismatch"
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

            let mut d_chunks = d_body.chunks_exact_mut(4);
            let mut s0_chunks = s0_body.chunks_exact(4);
            let mut s1_chunks = s1_body.chunks_exact(4);
            for ((dc, s0c), s1c) in (&mut d_chunks).zip(&mut s0_chunks).zip(&mut s1_chunks) {
                let t0 = simd.mul_add_f64s(na0, s0c[0], dc[0]);
                let t1 = simd.mul_add_f64s(na0, s0c[1], dc[1]);
                let t2 = simd.mul_add_f64s(na0, s0c[2], dc[2]);
                let t3 = simd.mul_add_f64s(na0, s0c[3], dc[3]);
                let r0 = simd.mul_add_f64s(na1, s1c[0], t0);
                let r1 = simd.mul_add_f64s(na1, s1c[1], t1);
                let r2 = simd.mul_add_f64s(na1, s1c[2], t2);
                let r3 = simd.mul_add_f64s(na1, s1c[3], t3);
                dc[0] = r0;
                dc[1] = r1;
                dc[2] = r2;
                dc[3] = r3;
            }

            let d_rem = d_chunks.into_remainder();
            let s0_rem = s0_chunks.remainder();
            let s1_rem = s1_chunks.remainder();
            for ((d, s0), s1) in d_rem.iter_mut().zip(s0_rem).zip(s1_rem) {
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

    const NEON: pulp::aarch64::Neon = unsafe { pulp::aarch64::Neon::new_unchecked() };
    use pulp::WithSimd;
    K {
        neg_alpha0: -alpha0,
        neg_alpha1: -alpha1,
        src0,
        src1,
        dst,
    }
    .with_simd(NEON);
}

// ---------------------------------------------------------------------
// Phase 2.4.3 non-FMA unroll4 variants. Same 4-way unrolled structure
// as `axpy*_minus_unroll4` but the inner body uses separate mul + sub
// instead of a fused multiply-add. This reproduces the scalar
// `dst[i] -= alpha * src[i]` rounding behavior bit-for-bit
// (two IEEE 754 roundings per element: one for `alpha*src[i]`, one
// for `dst[i] - that`) so that wiring these into `do_1x1_update` /
// `do_2x2_update` preserves the pivot classification boundary that
// the FMA unroll4 variants perturbed on 4 KKT matrices
// (ACOPP14_0001, ACOPP30_0004, FBRAIN3LS_0848, FBRAIN3LS_0851 — see
// `dev/tried-and-rejected.md` 2026-04-14 Phase 2.4.2 entry).
//
// The ILP gain from 4 independent accumulators is preserved;
// the per-op throughput cost is ~2x compared to FMA (two pipe slots
// per element instead of one). Whether the net speedup over the
// autovectorized scalar is large enough to be worth wiring in is
// the open Phase 2.4.3 question.

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub fn axpy_minus_unroll4_nofma(dst: &mut [f64], src: &[f64], alpha: f64) {
    assert_eq!(
        dst.len(),
        src.len(),
        "axpy_minus_unroll4_nofma: dst and src length mismatch"
    );

    struct K<'a> {
        alpha: f64,
        src: &'a [f64],
        dst: &'a mut [f64],
    }

    impl pulp::WithSimd for K<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            let Self { alpha, src, dst } = self;
            let a = simd.splat_f64s(alpha);

            let (src_body, src_tail) = S::as_simd_f64s(src);
            let (dst_body, dst_tail) = S::as_mut_simd_f64s(dst);

            let mut d_chunks = dst_body.chunks_exact_mut(4);
            let mut s_chunks = src_body.chunks_exact(4);
            for (dc, sc) in (&mut d_chunks).zip(&mut s_chunks) {
                let m0 = simd.mul_f64s(a, sc[0]);
                let m1 = simd.mul_f64s(a, sc[1]);
                let m2 = simd.mul_f64s(a, sc[2]);
                let m3 = simd.mul_f64s(a, sc[3]);
                let r0 = simd.sub_f64s(dc[0], m0);
                let r1 = simd.sub_f64s(dc[1], m1);
                let r2 = simd.sub_f64s(dc[2], m2);
                let r3 = simd.sub_f64s(dc[3], m3);
                dc[0] = r0;
                dc[1] = r1;
                dc[2] = r2;
                dc[3] = r3;
            }

            let d_rem = d_chunks.into_remainder();
            let s_rem = s_chunks.remainder();
            for (d, s) in d_rem.iter_mut().zip(s_rem) {
                *d = simd.sub_f64s(*d, simd.mul_f64s(a, *s));
            }

            if !src_tail.is_empty() {
                let s = simd.partial_load_f64s(src_tail);
                let d = simd.partial_load_f64s(dst_tail);
                simd.partial_store_f64s(dst_tail, simd.sub_f64s(d, simd.mul_f64s(a, s)));
            }
        }
    }

    const NEON: pulp::aarch64::Neon = unsafe { pulp::aarch64::Neon::new_unchecked() };
    use pulp::WithSimd;
    K { alpha, src, dst }.with_simd(NEON);
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub fn axpy2_minus_unroll4_nofma(
    dst: &mut [f64],
    src0: &[f64],
    alpha0: f64,
    src1: &[f64],
    alpha1: f64,
) {
    assert_eq!(
        dst.len(),
        src0.len(),
        "axpy2_minus_unroll4_nofma: dst and src0 length mismatch"
    );
    assert_eq!(
        dst.len(),
        src1.len(),
        "axpy2_minus_unroll4_nofma: dst and src1 length mismatch"
    );

    struct K<'a> {
        alpha0: f64,
        alpha1: f64,
        src0: &'a [f64],
        src1: &'a [f64],
        dst: &'a mut [f64],
    }

    impl pulp::WithSimd for K<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            let Self {
                alpha0,
                alpha1,
                src0,
                src1,
                dst,
            } = self;
            let a0 = simd.splat_f64s(alpha0);
            let a1 = simd.splat_f64s(alpha1);

            let (s0_body, s0_tail) = S::as_simd_f64s(src0);
            let (s1_body, s1_tail) = S::as_simd_f64s(src1);
            let (d_body, d_tail) = S::as_mut_simd_f64s(dst);

            let mut d_chunks = d_body.chunks_exact_mut(4);
            let mut s0_chunks = s0_body.chunks_exact(4);
            let mut s1_chunks = s1_body.chunks_exact(4);
            for ((dc, s0c), s1c) in (&mut d_chunks).zip(&mut s0_chunks).zip(&mut s1_chunks) {
                // Order of ops reproduces scalar `d -= s0*a0 + s1*a1`:
                //   t_i = round(round(a0*s0_i) + round(a1*s1_i))
                //   d_i = round(d_i - t_i)
                let m00 = simd.mul_f64s(a0, s0c[0]);
                let m01 = simd.mul_f64s(a0, s0c[1]);
                let m02 = simd.mul_f64s(a0, s0c[2]);
                let m03 = simd.mul_f64s(a0, s0c[3]);
                let m10 = simd.mul_f64s(a1, s1c[0]);
                let m11 = simd.mul_f64s(a1, s1c[1]);
                let m12 = simd.mul_f64s(a1, s1c[2]);
                let m13 = simd.mul_f64s(a1, s1c[3]);
                let t0 = simd.add_f64s(m00, m10);
                let t1 = simd.add_f64s(m01, m11);
                let t2 = simd.add_f64s(m02, m12);
                let t3 = simd.add_f64s(m03, m13);
                dc[0] = simd.sub_f64s(dc[0], t0);
                dc[1] = simd.sub_f64s(dc[1], t1);
                dc[2] = simd.sub_f64s(dc[2], t2);
                dc[3] = simd.sub_f64s(dc[3], t3);
            }

            let d_rem = d_chunks.into_remainder();
            let s0_rem = s0_chunks.remainder();
            let s1_rem = s1_chunks.remainder();
            for ((d, s0), s1) in d_rem.iter_mut().zip(s0_rem).zip(s1_rem) {
                let m0 = simd.mul_f64s(a0, *s0);
                let m1 = simd.mul_f64s(a1, *s1);
                *d = simd.sub_f64s(*d, simd.add_f64s(m0, m1));
            }

            if !s0_tail.is_empty() {
                let s0v = simd.partial_load_f64s(s0_tail);
                let s1v = simd.partial_load_f64s(s1_tail);
                let dv = simd.partial_load_f64s(d_tail);
                let m0 = simd.mul_f64s(a0, s0v);
                let m1 = simd.mul_f64s(a1, s1v);
                let r = simd.sub_f64s(dv, simd.add_f64s(m0, m1));
                simd.partial_store_f64s(d_tail, r);
            }
        }
    }

    const NEON: pulp::aarch64::Neon = unsafe { pulp::aarch64::Neon::new_unchecked() };
    use pulp::WithSimd;
    K {
        alpha0,
        alpha1,
        src0,
        src1,
        dst,
    }
    .with_simd(NEON);
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

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn axpy_minus_unroll4_matches_reference_across_length_sweep() {
        let mut rng = Xorshift64::new(0x4E27_A101_00FE_BEEFu64);
        for &len in LENGTH_SWEEP {
            let src: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let dst_init: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let alpha = rng.next_f64() * 1.5;

            let mut dst_kernel = dst_init.clone();
            let mut dst_ref = dst_init.clone();
            axpy_minus_unroll4(&mut dst_kernel, &src, alpha);
            naive_axpy_minus(&mut dst_ref, &src, alpha);
            assert_close(&dst_kernel, &dst_ref, ULP4);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn axpy2_minus_unroll4_matches_reference_across_length_sweep() {
        let mut rng = Xorshift64::new(0xC1FF_EE00_BAAD_F00Du64);
        for &len in LENGTH_SWEEP {
            let src0: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let src1: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let dst_init: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let alpha0 = rng.next_f64() * 1.5;
            let alpha1 = rng.next_f64() * 1.5;

            let mut dst_kernel = dst_init.clone();
            let mut dst_ref = dst_init.clone();
            axpy2_minus_unroll4(&mut dst_kernel, &src0, alpha0, &src1, alpha1);
            naive_axpy2_minus(&mut dst_ref, &src0, alpha0, &src1, alpha1);
            assert_close(&dst_kernel, &dst_ref, ULP4);
        }
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

    // Phase 2.4.3: bit-exactness tests for the non-FMA unroll4
    // variants. These are the whole point of the non-FMA variants —
    // they must reproduce the scalar loop's rounding behavior exactly
    // so that wiring them into `do_1x1_update` / `do_2x2_update` does
    // not perturb pivot classification on the 4 KKT matrices that
    // regressed under FMA unroll4. `assert_eq!` on f64 slices checks
    // every bit pattern, which is the correct assertion here.

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn axpy_minus_unroll4_nofma_is_bit_exact_vs_scalar() {
        let mut rng = Xorshift64::new(0xB17_EAC70_0042_F00Du64);
        for &len in LENGTH_SWEEP {
            let src: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let dst_init: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let alpha = rng.next_f64() * 1.5;

            let mut dst_kernel = dst_init.clone();
            let mut dst_ref = dst_init.clone();
            axpy_minus_unroll4_nofma(&mut dst_kernel, &src, alpha);
            naive_axpy_minus(&mut dst_ref, &src, alpha);
            assert_eq!(
                dst_kernel, dst_ref,
                "non-FMA unroll4 must be bit-exact vs scalar at len={}",
                len
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn axpy2_minus_unroll4_nofma_is_bit_exact_vs_scalar() {
        let mut rng = Xorshift64::new(0xB17_EAC70_BAAD_F00Du64);
        for &len in LENGTH_SWEEP {
            let src0: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let src1: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let dst_init: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
            let alpha0 = rng.next_f64() * 1.5;
            let alpha1 = rng.next_f64() * 1.5;

            let mut dst_kernel = dst_init.clone();
            let mut dst_ref = dst_init.clone();
            axpy2_minus_unroll4_nofma(&mut dst_kernel, &src0, alpha0, &src1, alpha1);
            naive_axpy2_minus(&mut dst_ref, &src0, alpha0, &src1, alpha1);
            assert_eq!(
                dst_kernel, dst_ref,
                "non-FMA unroll4 must be bit-exact vs scalar at len={}",
                len
            );
        }
    }
}
