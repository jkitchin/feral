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

// Direct-token dispatch helper for the FMA kernels. Mirrors
// `dispatch_nofma` (defined further below) — same architecture
// matrix, opposite numerical contract: these kernels DO use
// `simd.mul_add_f64s(...)` and therefore produce one rounding step
// per multiply-accumulate rather than two. Result is one ULP off the
// scalar `naive_axpy_minus` reference, by design.
//
// Opt-in only — see `dev/research/fma-kernel-opt-in.md` and the
// `NumericParams::fma` policy. Default execution still routes
// through `dispatch_nofma` so existing bit-exactness invariants
// hold for all callers that haven't opted in.
#[inline(always)]
fn dispatch_fma<K: pulp::WithSimd>(k: K) -> K::Output {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is a baseline feature on aarch64/ARMv8;
        // NEON's `vfmaq_f64` provides one-rounding multiply-add.
        const NEON: pulp::aarch64::Neon = unsafe { pulp::aarch64::Neon::new_unchecked() };
        k.with_simd(NEON)
    }
    #[cfg(target_arch = "x86_64")]
    {
        match pulp::x86::V3::try_new() {
            Some(v3) => k.with_simd(v3),
            None => pulp::Arch::new().dispatch(k),
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        pulp::Arch::new().dispatch(k)
    }
}

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

    dispatch_fma(K {
        neg_alpha: -alpha,
        src,
        dst,
    });
}

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

    dispatch_fma(K {
        neg_alpha0: -alpha0,
        neg_alpha1: -alpha1,
        src0,
        src1,
        dst,
    });
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

// Direct-token dispatch helper for the wired `_nofma` kernels.
//
// On aarch64 we always use the baseline NEON token (NEON is ARMv8
// mandatory, so `Neon::new_unchecked()` is sound).
//
// On x86_64 we runtime-detect AVX2+FMA via `pulp::x86::V3::try_new()`
// — pulp caches the CPUID result in a static AtomicU8, so per-call
// overhead is one relaxed atomic load + one branch (i.e. cheap
// enough to keep the per-update dispatch model). When V3 is not
// available (older x86_64 without AVX2/FMA) we fall back to
// `pulp::Arch::new().dispatch(...)`, which selects the best Simd
// flavor pulp can produce on the live machine (V2/SSE2/scalar).
//
// On other architectures we go straight to `pulp::Arch::new().dispatch`
// so the code path is universally available — only the wired-token
// performance story is arch-specific.
//
// Bit-exactness vs the scalar Rust loop is preserved on every
// architecture because the kernel bodies use explicit `mul + sub`
// (or `mul + add + sub` for the rank-2 form) and never `mul_add`,
// matching `naive_axpy_minus` / `naive_axpy2_minus`.
#[inline(always)]
fn dispatch_nofma<K: pulp::WithSimd>(k: K) -> K::Output {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is a baseline feature on aarch64/ARMv8.
        const NEON: pulp::aarch64::Neon = unsafe { pulp::aarch64::Neon::new_unchecked() };
        k.with_simd(NEON)
    }
    #[cfg(target_arch = "x86_64")]
    {
        match pulp::x86::V3::try_new() {
            Some(v3) => k.with_simd(v3),
            None => pulp::Arch::new().dispatch(k),
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        pulp::Arch::new().dispatch(k)
    }
}

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

    dispatch_nofma(K { alpha, src, dst });
}

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

    dispatch_nofma(K {
        alpha0,
        alpha1,
        src0,
        src1,
        dst,
    });
}

/// Rank-`n_elim` deferred-Schur trailing-column update with strided source.
///
/// Computes, **for one trailing column at a time**, the cumulative
/// rank-1 axpy:
///
/// ```text
///     for q in 0..n_elim {
///         dst[i] = dst[i] - alphas[q] * src[q*col_stride + i]   for i in 0..len
///     }
/// ```
///
/// `src` is laid out as `n_elim` columns of stride `col_stride`,
/// where each column contributes only the first `len` entries to
/// `dst` (`len == dst.len()`). The slack `col_stride - len` between
/// successive columns is unused by the kernel and may contain
/// arbitrary data — this matches the column-major front layout where
/// `col_stride = nrow` and `len = nrow - j` for trailing column `j`.
///
/// The contract with [`axpy_minus_unroll4_nofma`] is bit-exact:
/// per-element, the subtractions are issued in ascending `q` order
/// using the same explicit `mul + sub` rounding sequence (no FMA).
/// In particular, `alphas[q] == 0.0` is skipped — matching the
/// `do_1x1_update` early-return on zero alpha. Calling this function
/// with `n_elim` rank-1 contributions produces the same bit pattern as
/// `n_elim` sequential calls to `axpy_minus_unroll4_nofma`, without
/// the per-call pulp dispatch overhead.
///
/// Preconditions:
/// - `dst.len() == len`.
/// - `src.len() >= (n_elim - 1) * col_stride + len` when `n_elim > 0`.
/// - `dst` is disjoint from `src`.
/// - `alphas.len() == n_elim`.
///
/// W-2 from `dev/plans/dense-kernel-speedup.md`. Replaces the
/// `O(n_elim * trailing)` pulp dispatch pattern with `O(trailing)`
/// dispatches by accumulating all `n_elim` contributions in
/// register-resident accumulators per row-vector.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn schur_panel_minus_nofma_strided(
    dst: &mut [f64],
    src_block: &[f64],
    src_first_col: usize,
    n_elim: usize,
    col_stride: usize,
    src_row_offset: usize,
    len: usize,
    alphas: &[f64],
) {
    assert_eq!(
        dst.len(),
        len,
        "schur_panel_minus_nofma_strided: dst.len() must equal len"
    );
    assert_eq!(
        alphas.len(),
        n_elim,
        "schur_panel_minus_nofma_strided: alphas.len() must equal n_elim"
    );

    if n_elim == 0 || len == 0 {
        return;
    }

    // Compute the start of pivot column q's row-`src_row_offset`
    // entry inside `src_block`. Caller passes the entire `before`
    // slice (everything ahead of the trailing column being updated)
    // so we can address all pivot columns from a single base. The
    // last byte we touch is
    //   (src_first_col + n_elim - 1) * col_stride + src_row_offset + len - 1.
    let last_q = n_elim - 1;
    let max_idx = (src_first_col + last_q) * col_stride + src_row_offset + len;
    assert!(
        src_block.len() >= max_idx,
        "schur_panel_minus_nofma_strided: src_block too short ({} < {})",
        src_block.len(),
        max_idx
    );

    struct K<'a> {
        dst: &'a mut [f64],
        src_block: &'a [f64],
        src_first_col: usize,
        n_elim: usize,
        col_stride: usize,
        src_row_offset: usize,
        len: usize,
        alphas: &'a [f64],
    }

    impl pulp::WithSimd for K<'_> {
        type Output = ();

        // The `for q in 0..n_elim` loops below use `q` to index multiple
        // disjoint quantities (alphas[q], src_first_col + q, the per-q
        // src column offset). Rewriting as
        // `alphas.iter().enumerate().take(n_elim)` would obscure the
        // hot-loop intent and complicate the zero-alpha skip.
        #[allow(clippy::needless_range_loop)]
        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            let Self {
                dst,
                src_block,
                src_first_col,
                n_elim,
                col_stride,
                src_row_offset,
                len,
                alphas,
            } = self;

            let (dst_body, dst_tail) = S::as_mut_simd_f64s(dst);
            let body_len = dst_body.len();
            let tail_off = body_len * S::F64_LANES;

            // 4-way unrolled main body. Per chunk, we hold 4 dst
            // accumulators in SIMD registers, then iterate q=0..n_elim
            // and apply `dst -= mul(alpha_q, src_q)` to each accumulator
            // sequentially. This preserves the bit-exact rounding order
            // of `axpy_minus_unroll4_nofma` called n_elim times: for
            // each lane, the sequence of rounded subtractions is
            //   acc <- round(acc - round(alpha_q * src_q[i]))
            // for q in ascending order — identical to the rank-1 outer
            // loop in `apply_blocked_schur`.
            let chunks = body_len / 4;
            for chunk_idx in 0..chunks {
                let base = chunk_idx * 4;
                let mut a0 = dst_body[base];
                let mut a1 = dst_body[base + 1];
                let mut a2 = dst_body[base + 2];
                let mut a3 = dst_body[base + 3];
                for q in 0..n_elim {
                    let alpha_q = alphas[q];
                    if alpha_q == 0.0 {
                        continue;
                    }
                    let av = simd.splat_f64s(alpha_q);
                    let col_off = (src_first_col + q) * col_stride + src_row_offset;
                    let src_q = &src_block[col_off..col_off + len];
                    let (sb, _st) = S::as_simd_f64s(src_q);
                    let m0 = simd.mul_f64s(av, sb[base]);
                    let m1 = simd.mul_f64s(av, sb[base + 1]);
                    let m2 = simd.mul_f64s(av, sb[base + 2]);
                    let m3 = simd.mul_f64s(av, sb[base + 3]);
                    a0 = simd.sub_f64s(a0, m0);
                    a1 = simd.sub_f64s(a1, m1);
                    a2 = simd.sub_f64s(a2, m2);
                    a3 = simd.sub_f64s(a3, m3);
                }
                dst_body[base] = a0;
                dst_body[base + 1] = a1;
                dst_body[base + 2] = a2;
                dst_body[base + 3] = a3;
            }

            // Tail full-lane SIMD vectors (0..3 leftover after the 4-way unroll).
            let tail_chunks_start = chunks * 4;
            for body_idx in tail_chunks_start..body_len {
                let mut acc = dst_body[body_idx];
                for q in 0..n_elim {
                    let alpha_q = alphas[q];
                    if alpha_q == 0.0 {
                        continue;
                    }
                    let av = simd.splat_f64s(alpha_q);
                    let col_off = (src_first_col + q) * col_stride + src_row_offset;
                    let src_q = &src_block[col_off..col_off + len];
                    let (sb, _st) = S::as_simd_f64s(src_q);
                    let m = simd.mul_f64s(av, sb[body_idx]);
                    acc = simd.sub_f64s(acc, m);
                }
                dst_body[body_idx] = acc;
            }

            // Masked tail (< one full lane). Same per-element ordering.
            if !dst_tail.is_empty() {
                let mut acc = simd.partial_load_f64s(dst_tail);
                for q in 0..n_elim {
                    let alpha_q = alphas[q];
                    if alpha_q == 0.0 {
                        continue;
                    }
                    let av = simd.splat_f64s(alpha_q);
                    let col_off = (src_first_col + q) * col_stride + src_row_offset;
                    let src_q = &src_block[col_off..col_off + len];
                    let src_q_tail = &src_q[tail_off..];
                    let s = simd.partial_load_f64s(src_q_tail);
                    let m = simd.mul_f64s(av, s);
                    acc = simd.sub_f64s(acc, m);
                }
                simd.partial_store_f64s(dst_tail, acc);
            }
        }
    }

    dispatch_nofma(K {
        dst,
        src_block,
        src_first_col,
        n_elim,
        col_stride,
        src_row_offset,
        len,
        alphas,
    });
}

/// FMA-using sibling of [`schur_panel_minus_nofma_strided`].
///
/// Same signature, same accumulation order, same zero-alpha skip —
/// **the only numerical difference is one rounding step per
/// multiply-accumulate instead of two**. Per element:
///
/// ```text
///   acc <- mul_add(-alpha_q, src_q, acc)   // one IEEE 754 rounding
/// ```
///
/// vs the `_nofma` form's
///
/// ```text
///   acc <- acc - round(alpha_q * src_q)    // two IEEE 754 roundings
/// ```
///
/// The FMA form is one ULP per accumulate "more accurate" (single
/// rounding) but breaks bit-identity with `axpy_minus_unroll4_nofma`
/// and with the scalar `naive_axpy_minus` reference. Opt-in only —
/// see `dev/research/fma-kernel-opt-in.md` and `NumericParams::fma`.
///
/// Performance note: on Apple M-series NEON and on x86 V3 (AVX2+FMA),
/// the FMA path issues one pipe slot per element vs two for `mul + sub`,
/// for the documented ~2x throughput target on the dominant trailing-
/// update path. See issue #8.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn schur_panel_minus_fma_strided(
    dst: &mut [f64],
    src_block: &[f64],
    src_first_col: usize,
    n_elim: usize,
    col_stride: usize,
    src_row_offset: usize,
    len: usize,
    alphas: &[f64],
) {
    assert_eq!(
        dst.len(),
        len,
        "schur_panel_minus_fma_strided: dst.len() must equal len"
    );
    assert_eq!(
        alphas.len(),
        n_elim,
        "schur_panel_minus_fma_strided: alphas.len() must equal n_elim"
    );

    if n_elim == 0 || len == 0 {
        return;
    }

    let last_q = n_elim - 1;
    let max_idx = (src_first_col + last_q) * col_stride + src_row_offset + len;
    assert!(
        src_block.len() >= max_idx,
        "schur_panel_minus_fma_strided: src_block too short ({} < {})",
        src_block.len(),
        max_idx
    );

    struct K<'a> {
        dst: &'a mut [f64],
        src_block: &'a [f64],
        src_first_col: usize,
        n_elim: usize,
        col_stride: usize,
        src_row_offset: usize,
        len: usize,
        alphas: &'a [f64],
    }

    impl pulp::WithSimd for K<'_> {
        type Output = ();

        #[allow(clippy::needless_range_loop)]
        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            let Self {
                dst,
                src_block,
                src_first_col,
                n_elim,
                col_stride,
                src_row_offset,
                len,
                alphas,
            } = self;

            let (dst_body, dst_tail) = S::as_mut_simd_f64s(dst);
            let body_len = dst_body.len();
            let tail_off = body_len * S::F64_LANES;

            let chunks = body_len / 4;
            for chunk_idx in 0..chunks {
                let base = chunk_idx * 4;
                let mut a0 = dst_body[base];
                let mut a1 = dst_body[base + 1];
                let mut a2 = dst_body[base + 2];
                let mut a3 = dst_body[base + 3];
                for q in 0..n_elim {
                    let alpha_q = alphas[q];
                    if alpha_q == 0.0 {
                        continue;
                    }
                    let nav = simd.splat_f64s(-alpha_q);
                    let col_off = (src_first_col + q) * col_stride + src_row_offset;
                    let src_q = &src_block[col_off..col_off + len];
                    let (sb, _st) = S::as_simd_f64s(src_q);
                    a0 = simd.mul_add_f64s(nav, sb[base], a0);
                    a1 = simd.mul_add_f64s(nav, sb[base + 1], a1);
                    a2 = simd.mul_add_f64s(nav, sb[base + 2], a2);
                    a3 = simd.mul_add_f64s(nav, sb[base + 3], a3);
                }
                dst_body[base] = a0;
                dst_body[base + 1] = a1;
                dst_body[base + 2] = a2;
                dst_body[base + 3] = a3;
            }

            let tail_chunks_start = chunks * 4;
            for body_idx in tail_chunks_start..body_len {
                let mut acc = dst_body[body_idx];
                for q in 0..n_elim {
                    let alpha_q = alphas[q];
                    if alpha_q == 0.0 {
                        continue;
                    }
                    let nav = simd.splat_f64s(-alpha_q);
                    let col_off = (src_first_col + q) * col_stride + src_row_offset;
                    let src_q = &src_block[col_off..col_off + len];
                    let (sb, _st) = S::as_simd_f64s(src_q);
                    acc = simd.mul_add_f64s(nav, sb[body_idx], acc);
                }
                dst_body[body_idx] = acc;
            }

            if !dst_tail.is_empty() {
                let mut acc = simd.partial_load_f64s(dst_tail);
                for q in 0..n_elim {
                    let alpha_q = alphas[q];
                    if alpha_q == 0.0 {
                        continue;
                    }
                    let nav = simd.splat_f64s(-alpha_q);
                    let col_off = (src_first_col + q) * col_stride + src_row_offset;
                    let src_q = &src_block[col_off..col_off + len];
                    let src_q_tail = &src_q[tail_off..];
                    let s = simd.partial_load_f64s(src_q_tail);
                    acc = simd.mul_add_f64s(nav, s, acc);
                }
                simd.partial_store_f64s(dst_tail, acc);
            }
        }
    }

    dispatch_fma(K {
        dst,
        src_block,
        src_first_col,
        n_elim,
        col_stride,
        src_row_offset,
        len,
        alphas,
    });
}

/// FMA-using sibling of [`schur_panel_minus_nofma_strided_dual`].
///
/// Identical interface and per-column bit-exactness contract as the
/// non-FMA dual kernel, except each `acc <- acc - alpha*src` step is
/// fused into a single `mul_add(-alpha, src, acc)` (one rounding vs
/// two). Bit-exact against `n_elim` sequential calls of the FMA rank-1
/// reference (`axpy_minus_unroll4`) on each column independently —
/// **not** against the non-FMA dual kernel. Opt-in only — see
/// `dev/research/fma-kernel-opt-in.md` and `NumericParams::fma`.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn schur_panel_minus_fma_strided_dual(
    dst0: &mut [f64],
    dst1: &mut [f64],
    src_block: &[f64],
    src_first_col: usize,
    n_elim: usize,
    col_stride: usize,
    src_row_offset: usize,
    alphas0: &[f64],
    alphas1: &[f64],
) {
    let len0 = dst0.len();
    let len1 = dst1.len();
    assert_eq!(
        len1 + 1,
        len0,
        "schur_panel_minus_fma_strided_dual: dst1 must be exactly one shorter than dst0 \
         (len0={}, len1={})",
        len0,
        len1
    );
    assert_eq!(
        alphas0.len(),
        n_elim,
        "schur_panel_minus_fma_strided_dual: alphas0.len() must equal n_elim"
    );
    assert_eq!(
        alphas1.len(),
        n_elim,
        "schur_panel_minus_fma_strided_dual: alphas1.len() must equal n_elim"
    );

    if n_elim == 0 || len0 == 0 {
        return;
    }

    let last_q = n_elim - 1;
    let max_idx = (src_first_col + last_q) * col_stride + src_row_offset + len0;
    assert!(
        src_block.len() >= max_idx,
        "schur_panel_minus_fma_strided_dual: src_block too short ({} < {})",
        src_block.len(),
        max_idx
    );

    // Cap: dst0[0]. Scalar f64 mul_add matches single-lane SIMD mul_add.
    for (q, &alpha_q) in alphas0.iter().enumerate().take(n_elim) {
        if alpha_q == 0.0 {
            continue;
        }
        let col_off = (src_first_col + q) * col_stride + src_row_offset;
        let s = src_block[col_off];
        dst0[0] = (-alpha_q).mul_add(s, dst0[0]);
    }

    if len1 == 0 {
        return;
    }

    struct K<'a> {
        dst0_bulk: &'a mut [f64],
        dst1: &'a mut [f64],
        src_block: &'a [f64],
        src_first_col: usize,
        n_elim: usize,
        col_stride: usize,
        src_row_offset_bulk: usize,
        len: usize,
        alphas0: &'a [f64],
        alphas1: &'a [f64],
    }

    impl pulp::WithSimd for K<'_> {
        type Output = ();

        #[allow(clippy::needless_range_loop)]
        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            let Self {
                dst0_bulk,
                dst1,
                src_block,
                src_first_col,
                n_elim,
                col_stride,
                src_row_offset_bulk,
                len,
                alphas0,
                alphas1,
            } = self;

            let (d0_body, d0_tail) = S::as_mut_simd_f64s(dst0_bulk);
            let (d1_body, d1_tail) = S::as_mut_simd_f64s(dst1);
            let body_len = d0_body.len();
            debug_assert_eq!(body_len, d1_body.len());
            let tail_off = body_len * S::F64_LANES;

            let chunks = body_len / 4;
            for chunk_idx in 0..chunks {
                let base = chunk_idx * 4;
                let mut a00 = d0_body[base];
                let mut a01 = d0_body[base + 1];
                let mut a02 = d0_body[base + 2];
                let mut a03 = d0_body[base + 3];
                let mut a10 = d1_body[base];
                let mut a11 = d1_body[base + 1];
                let mut a12 = d1_body[base + 2];
                let mut a13 = d1_body[base + 3];
                for q in 0..n_elim {
                    let a0q = alphas0[q];
                    let a1q = alphas1[q];
                    if a0q == 0.0 && a1q == 0.0 {
                        continue;
                    }
                    let col_off = (src_first_col + q) * col_stride + src_row_offset_bulk;
                    let src_q = &src_block[col_off..col_off + len];
                    let (sb, _st) = S::as_simd_f64s(src_q);
                    let s0v = sb[base];
                    let s1v = sb[base + 1];
                    let s2v = sb[base + 2];
                    let s3v = sb[base + 3];
                    if a0q != 0.0 {
                        let nav0 = simd.splat_f64s(-a0q);
                        a00 = simd.mul_add_f64s(nav0, s0v, a00);
                        a01 = simd.mul_add_f64s(nav0, s1v, a01);
                        a02 = simd.mul_add_f64s(nav0, s2v, a02);
                        a03 = simd.mul_add_f64s(nav0, s3v, a03);
                    }
                    if a1q != 0.0 {
                        let nav1 = simd.splat_f64s(-a1q);
                        a10 = simd.mul_add_f64s(nav1, s0v, a10);
                        a11 = simd.mul_add_f64s(nav1, s1v, a11);
                        a12 = simd.mul_add_f64s(nav1, s2v, a12);
                        a13 = simd.mul_add_f64s(nav1, s3v, a13);
                    }
                }
                d0_body[base] = a00;
                d0_body[base + 1] = a01;
                d0_body[base + 2] = a02;
                d0_body[base + 3] = a03;
                d1_body[base] = a10;
                d1_body[base + 1] = a11;
                d1_body[base + 2] = a12;
                d1_body[base + 3] = a13;
            }

            let tail_chunks_start = chunks * 4;
            for body_idx in tail_chunks_start..body_len {
                let mut acc0 = d0_body[body_idx];
                let mut acc1 = d1_body[body_idx];
                for q in 0..n_elim {
                    let a0q = alphas0[q];
                    let a1q = alphas1[q];
                    if a0q == 0.0 && a1q == 0.0 {
                        continue;
                    }
                    let col_off = (src_first_col + q) * col_stride + src_row_offset_bulk;
                    let src_q = &src_block[col_off..col_off + len];
                    let (sb, _st) = S::as_simd_f64s(src_q);
                    let s = sb[body_idx];
                    if a0q != 0.0 {
                        let nav0 = simd.splat_f64s(-a0q);
                        acc0 = simd.mul_add_f64s(nav0, s, acc0);
                    }
                    if a1q != 0.0 {
                        let nav1 = simd.splat_f64s(-a1q);
                        acc1 = simd.mul_add_f64s(nav1, s, acc1);
                    }
                }
                d0_body[body_idx] = acc0;
                d1_body[body_idx] = acc1;
            }

            if !d0_tail.is_empty() {
                let mut acc0 = simd.partial_load_f64s(d0_tail);
                let mut acc1 = simd.partial_load_f64s(d1_tail);
                for q in 0..n_elim {
                    let a0q = alphas0[q];
                    let a1q = alphas1[q];
                    if a0q == 0.0 && a1q == 0.0 {
                        continue;
                    }
                    let col_off = (src_first_col + q) * col_stride + src_row_offset_bulk;
                    let src_q = &src_block[col_off..col_off + len];
                    let src_q_tail = &src_q[tail_off..];
                    let s = simd.partial_load_f64s(src_q_tail);
                    if a0q != 0.0 {
                        let nav0 = simd.splat_f64s(-a0q);
                        acc0 = simd.mul_add_f64s(nav0, s, acc0);
                    }
                    if a1q != 0.0 {
                        let nav1 = simd.splat_f64s(-a1q);
                        acc1 = simd.mul_add_f64s(nav1, s, acc1);
                    }
                }
                simd.partial_store_f64s(d0_tail, acc0);
                simd.partial_store_f64s(d1_tail, acc1);
            }
        }
    }

    let (dst0_cap, dst0_bulk) = dst0.split_at_mut(1);
    let _ = dst0_cap;
    dispatch_fma(K {
        dst0_bulk,
        dst1,
        src_block,
        src_first_col,
        n_elim,
        col_stride,
        src_row_offset_bulk: src_row_offset + 1,
        len: len1,
        alphas0,
        alphas1,
    });
}

/// Dual-column rank-`n_elim` deferred-Schur trailing-update.
///
/// Processes two adjacent trailing columns `j` and `j+1` in lockstep,
/// sharing each `src_q` load between both column accumulators.
/// Computes:
///
/// ```text
///     for q in 0..n_elim {
///         dst0[i] -= alphas0[q] * src[(src_first_col + q)*col_stride + src_row_offset + i]
///                                                                  for i in 0..len0
///         dst1[i] -= alphas1[q] * src[(src_first_col + q)*col_stride + src_row_offset + 1 + i]
///                                                                  for i in 0..len1
///     }
/// ```
///
/// `dst0` is column `j` (length `len0 = nrow - j`); `dst1` is column
/// `j+1` (length `len1 = len0 - 1`). Critically, `dst0[1..]` and `dst1`
/// reference the **same** src memory — the bulk of the work shares
/// a single `src_q` load per chunk per `q` step. The cap entry
/// `dst0[0]` is processed by a small scalar prologue.
///
/// Bit-exactness contract (Phase B-1): for each column independently,
/// the per-element rounding sequence
///   `acc <- round(acc - round(alpha_q * src_q[i]))`
/// is issued in `q` ascending order, identical to two sequential
/// calls of [`schur_panel_minus_nofma_strided`] on columns `j` and
/// `j+1`. Verified by the dedicated bit-exactness sweep below.
///
/// Preconditions:
/// - `dst0.len() == len0`, `dst1.len() == len0 - 1` (asserted).
/// - `alphas0.len() == alphas1.len() == n_elim`.
/// - `src_block.len() >= (src_first_col + n_elim - 1)*col_stride + src_row_offset + len0`.
/// - `dst0`, `dst1`, `src_block` pairwise disjoint (caller's burden;
///   guaranteed by `apply_blocked_schur_panel`'s `split_at_mut`).
///
/// Phase B-1 from `dev/plans/dense-kernel-blas3.md`. Halves the src
/// memory traffic for the trailing update on wide root supernodes
/// where `(nrow - j)` is large (the qcqp1500-1c 1061x1061 root path).
#[allow(dead_code, clippy::too_many_arguments)]
pub fn schur_panel_minus_nofma_strided_dual(
    dst0: &mut [f64],
    dst1: &mut [f64],
    src_block: &[f64],
    src_first_col: usize,
    n_elim: usize,
    col_stride: usize,
    src_row_offset: usize,
    alphas0: &[f64],
    alphas1: &[f64],
) {
    let len0 = dst0.len();
    let len1 = dst1.len();
    assert_eq!(
        len1 + 1,
        len0,
        "schur_panel_minus_nofma_strided_dual: dst1 must be exactly one shorter than dst0 \
         (len0={}, len1={})",
        len0,
        len1
    );
    assert_eq!(
        alphas0.len(),
        n_elim,
        "schur_panel_minus_nofma_strided_dual: alphas0.len() must equal n_elim"
    );
    assert_eq!(
        alphas1.len(),
        n_elim,
        "schur_panel_minus_nofma_strided_dual: alphas1.len() must equal n_elim"
    );

    if n_elim == 0 || len0 == 0 {
        return;
    }

    let last_q = n_elim - 1;
    let max_idx = (src_first_col + last_q) * col_stride + src_row_offset + len0;
    assert!(
        src_block.len() >= max_idx,
        "schur_panel_minus_nofma_strided_dual: src_block too short ({} < {})",
        src_block.len(),
        max_idx
    );

    // Cap: dst0[0] (the diagonal entry of column j). Process via a
    // scalar q-loop. Bit-exact with rank-1's per-element rounding,
    // since each lane of the SIMD body executes the same `d - mul(a, s)`
    // op and a single-element scalar is one such op.
    for (q, &alpha_q) in alphas0.iter().enumerate().take(n_elim) {
        if alpha_q == 0.0 {
            continue;
        }
        let col_off = (src_first_col + q) * col_stride + src_row_offset;
        let s = src_block[col_off];
        dst0[0] -= alpha_q * s;
    }

    if len1 == 0 {
        return;
    }

    // Bulk: dst0[1..] (length len1) and dst1 (length len1) share src
    // memory — both index into src_block at column offset
    //   (src_first_col + q) * col_stride + src_row_offset + 1
    // for the same row `i`.

    struct K<'a> {
        dst0_bulk: &'a mut [f64],
        dst1: &'a mut [f64],
        src_block: &'a [f64],
        src_first_col: usize,
        n_elim: usize,
        col_stride: usize,
        src_row_offset_bulk: usize,
        len: usize,
        alphas0: &'a [f64],
        alphas1: &'a [f64],
    }

    impl pulp::WithSimd for K<'_> {
        type Output = ();

        // The `for q in 0..n_elim` loops below use `q` to index
        // multiple disjoint quantities (alphas0[q], alphas1[q],
        // src_first_col + q). Rewriting as `.iter().enumerate()` would
        // obscure the hot-loop intent.
        #[allow(clippy::needless_range_loop)]
        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            let Self {
                dst0_bulk,
                dst1,
                src_block,
                src_first_col,
                n_elim,
                col_stride,
                src_row_offset_bulk,
                len,
                alphas0,
                alphas1,
            } = self;

            let (d0_body, d0_tail) = S::as_mut_simd_f64s(dst0_bulk);
            let (d1_body, d1_tail) = S::as_mut_simd_f64s(dst1);
            // dst0_bulk and dst1 have identical length (len); their
            // SIMD body/tail split is therefore identical.
            let body_len = d0_body.len();
            debug_assert_eq!(body_len, d1_body.len());
            let tail_off = body_len * S::F64_LANES;

            // 4-way unrolled main body. Per chunk, hold 4 dst0
            // accumulators and 4 dst1 accumulators in SIMD registers
            // (8 vector registers; AVX2/NEON have ≥16). Per q:
            //   - load src_q chunk ONCE (4 vector loads)
            //   - apply (alpha0_q, src) to dst0 accumulators
            //   - apply (alpha1_q, src) to dst1 accumulators
            // Bit-exact per column with the rank-1 reference.
            let chunks = body_len / 4;
            for chunk_idx in 0..chunks {
                let base = chunk_idx * 4;
                let mut a00 = d0_body[base];
                let mut a01 = d0_body[base + 1];
                let mut a02 = d0_body[base + 2];
                let mut a03 = d0_body[base + 3];
                let mut a10 = d1_body[base];
                let mut a11 = d1_body[base + 1];
                let mut a12 = d1_body[base + 2];
                let mut a13 = d1_body[base + 3];
                for q in 0..n_elim {
                    let a0q = alphas0[q];
                    let a1q = alphas1[q];
                    if a0q == 0.0 && a1q == 0.0 {
                        continue;
                    }
                    let col_off = (src_first_col + q) * col_stride + src_row_offset_bulk;
                    let src_q = &src_block[col_off..col_off + len];
                    let (sb, _st) = S::as_simd_f64s(src_q);
                    let s0v = sb[base];
                    let s1v = sb[base + 1];
                    let s2v = sb[base + 2];
                    let s3v = sb[base + 3];
                    if a0q != 0.0 {
                        let av0 = simd.splat_f64s(a0q);
                        a00 = simd.sub_f64s(a00, simd.mul_f64s(av0, s0v));
                        a01 = simd.sub_f64s(a01, simd.mul_f64s(av0, s1v));
                        a02 = simd.sub_f64s(a02, simd.mul_f64s(av0, s2v));
                        a03 = simd.sub_f64s(a03, simd.mul_f64s(av0, s3v));
                    }
                    if a1q != 0.0 {
                        let av1 = simd.splat_f64s(a1q);
                        a10 = simd.sub_f64s(a10, simd.mul_f64s(av1, s0v));
                        a11 = simd.sub_f64s(a11, simd.mul_f64s(av1, s1v));
                        a12 = simd.sub_f64s(a12, simd.mul_f64s(av1, s2v));
                        a13 = simd.sub_f64s(a13, simd.mul_f64s(av1, s3v));
                    }
                }
                d0_body[base] = a00;
                d0_body[base + 1] = a01;
                d0_body[base + 2] = a02;
                d0_body[base + 3] = a03;
                d1_body[base] = a10;
                d1_body[base + 1] = a11;
                d1_body[base + 2] = a12;
                d1_body[base + 3] = a13;
            }

            // Tail full-lane SIMD vectors (0..3 leftover).
            let tail_chunks_start = chunks * 4;
            for body_idx in tail_chunks_start..body_len {
                let mut acc0 = d0_body[body_idx];
                let mut acc1 = d1_body[body_idx];
                for q in 0..n_elim {
                    let a0q = alphas0[q];
                    let a1q = alphas1[q];
                    if a0q == 0.0 && a1q == 0.0 {
                        continue;
                    }
                    let col_off = (src_first_col + q) * col_stride + src_row_offset_bulk;
                    let src_q = &src_block[col_off..col_off + len];
                    let (sb, _st) = S::as_simd_f64s(src_q);
                    let s = sb[body_idx];
                    if a0q != 0.0 {
                        let av0 = simd.splat_f64s(a0q);
                        acc0 = simd.sub_f64s(acc0, simd.mul_f64s(av0, s));
                    }
                    if a1q != 0.0 {
                        let av1 = simd.splat_f64s(a1q);
                        acc1 = simd.sub_f64s(acc1, simd.mul_f64s(av1, s));
                    }
                }
                d0_body[body_idx] = acc0;
                d1_body[body_idx] = acc1;
            }

            // Masked tail (< one full lane). Same per-element ordering.
            if !d0_tail.is_empty() {
                let mut acc0 = simd.partial_load_f64s(d0_tail);
                let mut acc1 = simd.partial_load_f64s(d1_tail);
                for q in 0..n_elim {
                    let a0q = alphas0[q];
                    let a1q = alphas1[q];
                    if a0q == 0.0 && a1q == 0.0 {
                        continue;
                    }
                    let col_off = (src_first_col + q) * col_stride + src_row_offset_bulk;
                    let src_q = &src_block[col_off..col_off + len];
                    let src_q_tail = &src_q[tail_off..];
                    let s = simd.partial_load_f64s(src_q_tail);
                    if a0q != 0.0 {
                        let av0 = simd.splat_f64s(a0q);
                        acc0 = simd.sub_f64s(acc0, simd.mul_f64s(av0, s));
                    }
                    if a1q != 0.0 {
                        let av1 = simd.splat_f64s(a1q);
                        acc1 = simd.sub_f64s(acc1, simd.mul_f64s(av1, s));
                    }
                }
                simd.partial_store_f64s(d0_tail, acc0);
                simd.partial_store_f64s(d1_tail, acc1);
            }
        }
    }

    let (dst0_cap, dst0_bulk) = dst0.split_at_mut(1);
    let _ = dst0_cap;
    dispatch_nofma(K {
        dst0_bulk,
        dst1,
        src_block,
        src_first_col,
        n_elim,
        col_stride,
        src_row_offset_bulk: src_row_offset + 1,
        len: len1,
        alphas0,
        alphas1,
    });
}

/// Quad-column rank-`n_elim` deferred-Schur trailing-update.
///
/// Processes four adjacent trailing columns `j`, `j+1`, `j+2`, `j+3`
/// in lockstep, sharing each `src_q` load between all four column
/// accumulators. Computes, for each column c in 0..4 independently:
///
/// ```text
///     for q in 0..n_elim {
///         dst_c[i] -= alphas_c[q] * src[(src_first_col + q)*col_stride
///                                       + src_row_offset + c + i]
///                                                            for i in 0..(len0 - c)
///     }
/// ```
///
/// `dst0` has length `len0 = nrow - j`; `dst1`, `dst2`, `dst3` have
/// lengths `len0 - 1`, `len0 - 2`, `len0 - 3` respectively (each
/// trailing column is one row shorter than the previous, matching the
/// lower-triangular L panel layout). The cap rows 0, 1, 2 are handled
/// by a scalar prologue (3 elements; tiny relative to the bulk). The
/// bulk processes rows where all four columns have a corresponding
/// entry, length `len3 = len0 - 3`.
///
/// Bit-exactness contract: for each column independently the
/// per-element rounding sequence
///   `acc <- round(acc - round(alpha_q * src_q[i]))`
/// is issued in `q` ascending order, identical to four sequential
/// calls of [`schur_panel_minus_nofma_strided`] on columns `j..j+4`.
/// The bulk SIMD body's unroll factor is arch-gated: 4 on aarch64 (16
/// accumulators, 4 row-vecs × 4 cols, fit the 32-register NEON budget),
/// 2 elsewhere (8 accumulators fit AVX2's 16-ymm budget without
/// spilling). The unroll factor only regroups body vectors into chunks;
/// it never reorders a per-element accumulation, so it is bit-neutral.
///
/// Preconditions:
/// - `dst0.len() == len0`, `dst1.len() == len0 - 1`,
///   `dst2.len() == len0 - 2`, `dst3.len() == len0 - 3` (asserted).
/// - `alphas{0,1,2,3}.len() == n_elim`.
/// - `src_block.len() >= (src_first_col + n_elim - 1)*col_stride
///                       + src_row_offset + len0`.
/// - All four `dst` slices and `src_block` pairwise disjoint
///   (caller's burden; guaranteed by `apply_blocked_schur_panel`'s
///   nested `split_at_mut`).
///
/// Phase 2.4.3 (issue #9, re-scoped). Halves source memory traffic
/// vs the dual kernel; doubles arithmetic per src-vector load.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn schur_panel_minus_nofma_strided_quad(
    dst0: &mut [f64],
    dst1: &mut [f64],
    dst2: &mut [f64],
    dst3: &mut [f64],
    src_block: &[f64],
    src_first_col: usize,
    n_elim: usize,
    col_stride: usize,
    src_row_offset: usize,
    alphas0: &[f64],
    alphas1: &[f64],
    alphas2: &[f64],
    alphas3: &[f64],
) {
    let len0 = dst0.len();
    let len1 = dst1.len();
    let len2 = dst2.len();
    let len3 = dst3.len();
    assert_eq!(
        len1 + 1,
        len0,
        "schur_panel_minus_nofma_strided_quad: dst1 must be exactly one shorter than dst0 \
         (len0={}, len1={})",
        len0,
        len1
    );
    assert_eq!(
        len2 + 2,
        len0,
        "schur_panel_minus_nofma_strided_quad: dst2 must be exactly two shorter than dst0 \
         (len0={}, len2={})",
        len0,
        len2
    );
    assert_eq!(
        len3 + 3,
        len0,
        "schur_panel_minus_nofma_strided_quad: dst3 must be exactly three shorter than dst0 \
         (len0={}, len3={})",
        len0,
        len3
    );
    assert_eq!(
        alphas0.len(),
        n_elim,
        "schur_panel_minus_nofma_strided_quad: alphas0.len() must equal n_elim"
    );
    assert_eq!(
        alphas1.len(),
        n_elim,
        "schur_panel_minus_nofma_strided_quad: alphas1.len() must equal n_elim"
    );
    assert_eq!(
        alphas2.len(),
        n_elim,
        "schur_panel_minus_nofma_strided_quad: alphas2.len() must equal n_elim"
    );
    assert_eq!(
        alphas3.len(),
        n_elim,
        "schur_panel_minus_nofma_strided_quad: alphas3.len() must equal n_elim"
    );

    if n_elim == 0 || len0 == 0 {
        return;
    }

    let last_q = n_elim - 1;
    let max_idx = (src_first_col + last_q) * col_stride + src_row_offset + len0;
    assert!(
        src_block.len() >= max_idx,
        "schur_panel_minus_nofma_strided_quad: src_block too short ({} < {})",
        src_block.len(),
        max_idx
    );

    // Cap row 0 (dst0[0] only). Process via scalar q-loop; bit-exact
    // with rank-1's per-element rounding since SIMD `mul`/`sub` of a
    // single-lane vector is identical IEEE-754 to scalar `*`/`-`.
    for (q, &alpha_q) in alphas0.iter().enumerate().take(n_elim) {
        if alpha_q == 0.0 {
            continue;
        }
        let col_off = (src_first_col + q) * col_stride + src_row_offset;
        let s = src_block[col_off];
        dst0[0] -= alpha_q * s;
    }

    if len0 == 1 {
        return;
    }

    // Cap row 1 (dst0[1] and dst1[0]). Both share src row offset
    // src_row_offset + 1.
    for q in 0..n_elim {
        let col_off = (src_first_col + q) * col_stride + src_row_offset + 1;
        let s = src_block[col_off];
        let a0 = alphas0[q];
        let a1 = alphas1[q];
        if a0 != 0.0 {
            dst0[1] -= a0 * s;
        }
        if a1 != 0.0 {
            dst1[0] -= a1 * s;
        }
    }

    if len0 == 2 {
        return;
    }

    // Cap row 2 (dst0[2], dst1[1], dst2[0]). All three share src row
    // offset src_row_offset + 2.
    for q in 0..n_elim {
        let col_off = (src_first_col + q) * col_stride + src_row_offset + 2;
        let s = src_block[col_off];
        let a0 = alphas0[q];
        let a1 = alphas1[q];
        let a2 = alphas2[q];
        if a0 != 0.0 {
            dst0[2] -= a0 * s;
        }
        if a1 != 0.0 {
            dst1[1] -= a1 * s;
        }
        if a2 != 0.0 {
            dst2[0] -= a2 * s;
        }
    }

    if len0 == 3 {
        return;
    }

    // Bulk: rows j+3..nrow. All four columns have a corresponding
    // entry. dst0_bulk[i], dst1_bulk[i], dst2_bulk[i], dst3_bulk[i]
    // all share src row offset src_row_offset + 3 + i.

    struct K<'a> {
        dst0_bulk: &'a mut [f64],
        dst1_bulk: &'a mut [f64],
        dst2_bulk: &'a mut [f64],
        dst3_bulk: &'a mut [f64],
        src_block: &'a [f64],
        src_first_col: usize,
        n_elim: usize,
        col_stride: usize,
        src_row_offset_bulk: usize,
        len: usize,
        alphas0: &'a [f64],
        alphas1: &'a [f64],
        alphas2: &'a [f64],
        alphas3: &'a [f64],
    }

    impl pulp::WithSimd for K<'_> {
        type Output = ();

        // The `for q in 0..n_elim` loops below use `q` to index
        // multiple disjoint quantities (alphas{0..3}[q],
        // src_first_col + q). Rewriting as `.iter().enumerate()` would
        // obscure the hot-loop intent.
        #[allow(clippy::needless_range_loop)]
        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            let Self {
                dst0_bulk,
                dst1_bulk,
                dst2_bulk,
                dst3_bulk,
                src_block,
                src_first_col,
                n_elim,
                col_stride,
                src_row_offset_bulk,
                len,
                alphas0,
                alphas1,
                alphas2,
                alphas3,
            } = self;

            let (d0_body, d0_tail) = S::as_mut_simd_f64s(dst0_bulk);
            let (d1_body, d1_tail) = S::as_mut_simd_f64s(dst1_bulk);
            let (d2_body, d2_tail) = S::as_mut_simd_f64s(dst2_bulk);
            let (d3_body, d3_tail) = S::as_mut_simd_f64s(dst3_bulk);
            // All four bulks have identical length; SIMD body/tail
            // splits coincide.
            let body_len = d0_body.len();
            debug_assert_eq!(body_len, d1_body.len());
            debug_assert_eq!(body_len, d2_body.len());
            debug_assert_eq!(body_len, d3_body.len());
            let tail_off = body_len * S::F64_LANES;

            // Register-blocked main body. `UNROLL` SIMD row-vectors per
            // dst column are kept live across the whole rank-q loop:
            // 4 on aarch64 (16 accumulators + 4 src + 1 alpha splat ≈
            // 21 live regs, fits 32-register NEON), 2 on x86 (8 acc +
            // 2 src + 1 splat ≈ 11 live, fits AVX2's 16 ymm). `UNROLL`
            // is `const`, so the `for k in 0..UNROLL` loops fully
            // unroll and SROA scalarizes the small `[S::f64s; UNROLL]`
            // arrays back into registers.
            //
            // Per q: load each src_q chunk vector ONCE, then apply
            // (alpha0_q, s) → dst0 accumulators, (alpha1_q, s) → dst1,
            // (alpha2_q, s) → dst2, (alpha3_q, s) → dst3. Bit-exact per
            // column with four sequential single-column rank-1
            // dispatches: each body vector's accumulator runs the same
            // q-ascending `acc <- sub(acc, mul(alpha_q, src))` chain
            // regardless of which chunk it lands in.
            const UNROLL: usize = if cfg!(target_arch = "aarch64") { 4 } else { 2 };
            let chunks = body_len / UNROLL;
            for chunk_idx in 0..chunks {
                let base = chunk_idx * UNROLL;
                let mut a0: [S::f64s; UNROLL] = core::array::from_fn(|k| d0_body[base + k]);
                let mut a1: [S::f64s; UNROLL] = core::array::from_fn(|k| d1_body[base + k]);
                let mut a2: [S::f64s; UNROLL] = core::array::from_fn(|k| d2_body[base + k]);
                let mut a3: [S::f64s; UNROLL] = core::array::from_fn(|k| d3_body[base + k]);
                for q in 0..n_elim {
                    let a0q = alphas0[q];
                    let a1q = alphas1[q];
                    let a2q = alphas2[q];
                    let a3q = alphas3[q];
                    if a0q == 0.0 && a1q == 0.0 && a2q == 0.0 && a3q == 0.0 {
                        continue;
                    }
                    let col_off = (src_first_col + q) * col_stride + src_row_offset_bulk;
                    let src_q = &src_block[col_off..col_off + len];
                    let (sb, _st) = S::as_simd_f64s(src_q);
                    let s: [S::f64s; UNROLL] = core::array::from_fn(|k| sb[base + k]);
                    if a0q != 0.0 {
                        let av0 = simd.splat_f64s(a0q);
                        for k in 0..UNROLL {
                            a0[k] = simd.sub_f64s(a0[k], simd.mul_f64s(av0, s[k]));
                        }
                    }
                    if a1q != 0.0 {
                        let av1 = simd.splat_f64s(a1q);
                        for k in 0..UNROLL {
                            a1[k] = simd.sub_f64s(a1[k], simd.mul_f64s(av1, s[k]));
                        }
                    }
                    if a2q != 0.0 {
                        let av2 = simd.splat_f64s(a2q);
                        for k in 0..UNROLL {
                            a2[k] = simd.sub_f64s(a2[k], simd.mul_f64s(av2, s[k]));
                        }
                    }
                    if a3q != 0.0 {
                        let av3 = simd.splat_f64s(a3q);
                        for k in 0..UNROLL {
                            a3[k] = simd.sub_f64s(a3[k], simd.mul_f64s(av3, s[k]));
                        }
                    }
                }
                d0_body[base..base + UNROLL].copy_from_slice(&a0);
                d1_body[base..base + UNROLL].copy_from_slice(&a1);
                d2_body[base..base + UNROLL].copy_from_slice(&a2);
                d3_body[base..base + UNROLL].copy_from_slice(&a3);
            }

            // Tail full-lane SIMD vectors (0..UNROLL-1 leftover after
            // the register-blocked main body), one vector at a time.
            let tail_chunks_start = chunks * UNROLL;
            for body_idx in tail_chunks_start..body_len {
                let mut acc0 = d0_body[body_idx];
                let mut acc1 = d1_body[body_idx];
                let mut acc2 = d2_body[body_idx];
                let mut acc3 = d3_body[body_idx];
                for q in 0..n_elim {
                    let a0q = alphas0[q];
                    let a1q = alphas1[q];
                    let a2q = alphas2[q];
                    let a3q = alphas3[q];
                    if a0q == 0.0 && a1q == 0.0 && a2q == 0.0 && a3q == 0.0 {
                        continue;
                    }
                    let col_off = (src_first_col + q) * col_stride + src_row_offset_bulk;
                    let src_q = &src_block[col_off..col_off + len];
                    let (sb, _st) = S::as_simd_f64s(src_q);
                    let s = sb[body_idx];
                    if a0q != 0.0 {
                        let av0 = simd.splat_f64s(a0q);
                        acc0 = simd.sub_f64s(acc0, simd.mul_f64s(av0, s));
                    }
                    if a1q != 0.0 {
                        let av1 = simd.splat_f64s(a1q);
                        acc1 = simd.sub_f64s(acc1, simd.mul_f64s(av1, s));
                    }
                    if a2q != 0.0 {
                        let av2 = simd.splat_f64s(a2q);
                        acc2 = simd.sub_f64s(acc2, simd.mul_f64s(av2, s));
                    }
                    if a3q != 0.0 {
                        let av3 = simd.splat_f64s(a3q);
                        acc3 = simd.sub_f64s(acc3, simd.mul_f64s(av3, s));
                    }
                }
                d0_body[body_idx] = acc0;
                d1_body[body_idx] = acc1;
                d2_body[body_idx] = acc2;
                d3_body[body_idx] = acc3;
            }

            // Masked tail (< one full lane). Same per-element ordering.
            if !d0_tail.is_empty() {
                let mut acc0 = simd.partial_load_f64s(d0_tail);
                let mut acc1 = simd.partial_load_f64s(d1_tail);
                let mut acc2 = simd.partial_load_f64s(d2_tail);
                let mut acc3 = simd.partial_load_f64s(d3_tail);
                for q in 0..n_elim {
                    let a0q = alphas0[q];
                    let a1q = alphas1[q];
                    let a2q = alphas2[q];
                    let a3q = alphas3[q];
                    if a0q == 0.0 && a1q == 0.0 && a2q == 0.0 && a3q == 0.0 {
                        continue;
                    }
                    let col_off = (src_first_col + q) * col_stride + src_row_offset_bulk;
                    let src_q = &src_block[col_off..col_off + len];
                    let src_q_tail = &src_q[tail_off..];
                    let s = simd.partial_load_f64s(src_q_tail);
                    if a0q != 0.0 {
                        let av0 = simd.splat_f64s(a0q);
                        acc0 = simd.sub_f64s(acc0, simd.mul_f64s(av0, s));
                    }
                    if a1q != 0.0 {
                        let av1 = simd.splat_f64s(a1q);
                        acc1 = simd.sub_f64s(acc1, simd.mul_f64s(av1, s));
                    }
                    if a2q != 0.0 {
                        let av2 = simd.splat_f64s(a2q);
                        acc2 = simd.sub_f64s(acc2, simd.mul_f64s(av2, s));
                    }
                    if a3q != 0.0 {
                        let av3 = simd.splat_f64s(a3q);
                        acc3 = simd.sub_f64s(acc3, simd.mul_f64s(av3, s));
                    }
                }
                simd.partial_store_f64s(d0_tail, acc0);
                simd.partial_store_f64s(d1_tail, acc1);
                simd.partial_store_f64s(d2_tail, acc2);
                simd.partial_store_f64s(d3_tail, acc3);
            }
        }
    }

    // Split each dst into (cap, bulk). dst0 cap is 3 elements,
    // dst1 cap is 2, dst2 cap is 1, dst3 has no cap.
    let (_d0_cap, d0_bulk) = dst0.split_at_mut(3);
    let (_d1_cap, d1_bulk) = dst1.split_at_mut(2);
    let (_d2_cap, d2_bulk) = dst2.split_at_mut(1);
    dispatch_nofma(K {
        dst0_bulk: d0_bulk,
        dst1_bulk: d1_bulk,
        dst2_bulk: d2_bulk,
        dst3_bulk: dst3,
        src_block,
        src_first_col,
        n_elim,
        col_stride,
        src_row_offset_bulk: src_row_offset + 3,
        len: len3,
        alphas0,
        alphas1,
        alphas2,
        alphas3,
    });
}

/// FMA-using sibling of [`schur_panel_minus_nofma_strided_quad`].
///
/// Identical interface and per-column bit-exactness contract as the
/// non-FMA quad kernel, except each `acc <- acc - alpha*src` step is
/// fused into a single `mul_add(-alpha, src, acc)` (one rounding vs
/// two). Bit-exact against `n_elim` sequential calls of the FMA rank-1
/// reference (`axpy_minus_unroll4`) on each of the four columns
/// independently — **not** against the non-FMA quad kernel. Opt-in
/// only — see `dev/research/fma-kernel-opt-in.md` and
/// `NumericParams::fma`.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn schur_panel_minus_fma_strided_quad(
    dst0: &mut [f64],
    dst1: &mut [f64],
    dst2: &mut [f64],
    dst3: &mut [f64],
    src_block: &[f64],
    src_first_col: usize,
    n_elim: usize,
    col_stride: usize,
    src_row_offset: usize,
    alphas0: &[f64],
    alphas1: &[f64],
    alphas2: &[f64],
    alphas3: &[f64],
) {
    let len0 = dst0.len();
    let len1 = dst1.len();
    let len2 = dst2.len();
    let len3 = dst3.len();
    assert_eq!(
        len1 + 1,
        len0,
        "schur_panel_minus_fma_strided_quad: dst1 must be exactly one shorter than dst0 \
         (len0={}, len1={})",
        len0,
        len1
    );
    assert_eq!(
        len2 + 2,
        len0,
        "schur_panel_minus_fma_strided_quad: dst2 must be exactly two shorter than dst0 \
         (len0={}, len2={})",
        len0,
        len2
    );
    assert_eq!(
        len3 + 3,
        len0,
        "schur_panel_minus_fma_strided_quad: dst3 must be exactly three shorter than dst0 \
         (len0={}, len3={})",
        len0,
        len3
    );
    assert_eq!(
        alphas0.len(),
        n_elim,
        "schur_panel_minus_fma_strided_quad: alphas0.len() must equal n_elim"
    );
    assert_eq!(
        alphas1.len(),
        n_elim,
        "schur_panel_minus_fma_strided_quad: alphas1.len() must equal n_elim"
    );
    assert_eq!(
        alphas2.len(),
        n_elim,
        "schur_panel_minus_fma_strided_quad: alphas2.len() must equal n_elim"
    );
    assert_eq!(
        alphas3.len(),
        n_elim,
        "schur_panel_minus_fma_strided_quad: alphas3.len() must equal n_elim"
    );

    if n_elim == 0 || len0 == 0 {
        return;
    }

    let last_q = n_elim - 1;
    let max_idx = (src_first_col + last_q) * col_stride + src_row_offset + len0;
    assert!(
        src_block.len() >= max_idx,
        "schur_panel_minus_fma_strided_quad: src_block too short ({} < {})",
        src_block.len(),
        max_idx
    );

    // Cap row 0 (dst0[0] only). Scalar f64 mul_add.
    for (q, &alpha_q) in alphas0.iter().enumerate().take(n_elim) {
        if alpha_q == 0.0 {
            continue;
        }
        let col_off = (src_first_col + q) * col_stride + src_row_offset;
        let s = src_block[col_off];
        dst0[0] = (-alpha_q).mul_add(s, dst0[0]);
    }

    if len0 == 1 {
        return;
    }

    // Cap row 1 (dst0[1] and dst1[0]).
    for q in 0..n_elim {
        let col_off = (src_first_col + q) * col_stride + src_row_offset + 1;
        let s = src_block[col_off];
        let a0 = alphas0[q];
        let a1 = alphas1[q];
        if a0 != 0.0 {
            dst0[1] = (-a0).mul_add(s, dst0[1]);
        }
        if a1 != 0.0 {
            dst1[0] = (-a1).mul_add(s, dst1[0]);
        }
    }

    if len0 == 2 {
        return;
    }

    // Cap row 2 (dst0[2], dst1[1], dst2[0]).
    for q in 0..n_elim {
        let col_off = (src_first_col + q) * col_stride + src_row_offset + 2;
        let s = src_block[col_off];
        let a0 = alphas0[q];
        let a1 = alphas1[q];
        let a2 = alphas2[q];
        if a0 != 0.0 {
            dst0[2] = (-a0).mul_add(s, dst0[2]);
        }
        if a1 != 0.0 {
            dst1[1] = (-a1).mul_add(s, dst1[1]);
        }
        if a2 != 0.0 {
            dst2[0] = (-a2).mul_add(s, dst2[0]);
        }
    }

    if len0 == 3 {
        return;
    }

    struct K<'a> {
        dst0_bulk: &'a mut [f64],
        dst1_bulk: &'a mut [f64],
        dst2_bulk: &'a mut [f64],
        dst3_bulk: &'a mut [f64],
        src_block: &'a [f64],
        src_first_col: usize,
        n_elim: usize,
        col_stride: usize,
        src_row_offset_bulk: usize,
        len: usize,
        alphas0: &'a [f64],
        alphas1: &'a [f64],
        alphas2: &'a [f64],
        alphas3: &'a [f64],
    }

    impl pulp::WithSimd for K<'_> {
        type Output = ();

        #[allow(clippy::needless_range_loop)]
        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            let Self {
                dst0_bulk,
                dst1_bulk,
                dst2_bulk,
                dst3_bulk,
                src_block,
                src_first_col,
                n_elim,
                col_stride,
                src_row_offset_bulk,
                len,
                alphas0,
                alphas1,
                alphas2,
                alphas3,
            } = self;

            let (d0_body, d0_tail) = S::as_mut_simd_f64s(dst0_bulk);
            let (d1_body, d1_tail) = S::as_mut_simd_f64s(dst1_bulk);
            let (d2_body, d2_tail) = S::as_mut_simd_f64s(dst2_bulk);
            let (d3_body, d3_tail) = S::as_mut_simd_f64s(dst3_bulk);
            let body_len = d0_body.len();
            debug_assert_eq!(body_len, d1_body.len());
            debug_assert_eq!(body_len, d2_body.len());
            debug_assert_eq!(body_len, d3_body.len());
            let tail_off = body_len * S::F64_LANES;

            let chunks = body_len / 2;
            for chunk_idx in 0..chunks {
                let base = chunk_idx * 2;
                let mut a00 = d0_body[base];
                let mut a01 = d0_body[base + 1];
                let mut a10 = d1_body[base];
                let mut a11 = d1_body[base + 1];
                let mut a20 = d2_body[base];
                let mut a21 = d2_body[base + 1];
                let mut a30 = d3_body[base];
                let mut a31 = d3_body[base + 1];
                for q in 0..n_elim {
                    let a0q = alphas0[q];
                    let a1q = alphas1[q];
                    let a2q = alphas2[q];
                    let a3q = alphas3[q];
                    if a0q == 0.0 && a1q == 0.0 && a2q == 0.0 && a3q == 0.0 {
                        continue;
                    }
                    let col_off = (src_first_col + q) * col_stride + src_row_offset_bulk;
                    let src_q = &src_block[col_off..col_off + len];
                    let (sb, _st) = S::as_simd_f64s(src_q);
                    let s0 = sb[base];
                    let s1 = sb[base + 1];
                    if a0q != 0.0 {
                        let nav0 = simd.splat_f64s(-a0q);
                        a00 = simd.mul_add_f64s(nav0, s0, a00);
                        a01 = simd.mul_add_f64s(nav0, s1, a01);
                    }
                    if a1q != 0.0 {
                        let nav1 = simd.splat_f64s(-a1q);
                        a10 = simd.mul_add_f64s(nav1, s0, a10);
                        a11 = simd.mul_add_f64s(nav1, s1, a11);
                    }
                    if a2q != 0.0 {
                        let nav2 = simd.splat_f64s(-a2q);
                        a20 = simd.mul_add_f64s(nav2, s0, a20);
                        a21 = simd.mul_add_f64s(nav2, s1, a21);
                    }
                    if a3q != 0.0 {
                        let nav3 = simd.splat_f64s(-a3q);
                        a30 = simd.mul_add_f64s(nav3, s0, a30);
                        a31 = simd.mul_add_f64s(nav3, s1, a31);
                    }
                }
                d0_body[base] = a00;
                d0_body[base + 1] = a01;
                d1_body[base] = a10;
                d1_body[base + 1] = a11;
                d2_body[base] = a20;
                d2_body[base + 1] = a21;
                d3_body[base] = a30;
                d3_body[base + 1] = a31;
            }

            let tail_chunks_start = chunks * 2;
            for body_idx in tail_chunks_start..body_len {
                let mut acc0 = d0_body[body_idx];
                let mut acc1 = d1_body[body_idx];
                let mut acc2 = d2_body[body_idx];
                let mut acc3 = d3_body[body_idx];
                for q in 0..n_elim {
                    let a0q = alphas0[q];
                    let a1q = alphas1[q];
                    let a2q = alphas2[q];
                    let a3q = alphas3[q];
                    if a0q == 0.0 && a1q == 0.0 && a2q == 0.0 && a3q == 0.0 {
                        continue;
                    }
                    let col_off = (src_first_col + q) * col_stride + src_row_offset_bulk;
                    let src_q = &src_block[col_off..col_off + len];
                    let (sb, _st) = S::as_simd_f64s(src_q);
                    let s = sb[body_idx];
                    if a0q != 0.0 {
                        let nav0 = simd.splat_f64s(-a0q);
                        acc0 = simd.mul_add_f64s(nav0, s, acc0);
                    }
                    if a1q != 0.0 {
                        let nav1 = simd.splat_f64s(-a1q);
                        acc1 = simd.mul_add_f64s(nav1, s, acc1);
                    }
                    if a2q != 0.0 {
                        let nav2 = simd.splat_f64s(-a2q);
                        acc2 = simd.mul_add_f64s(nav2, s, acc2);
                    }
                    if a3q != 0.0 {
                        let nav3 = simd.splat_f64s(-a3q);
                        acc3 = simd.mul_add_f64s(nav3, s, acc3);
                    }
                }
                d0_body[body_idx] = acc0;
                d1_body[body_idx] = acc1;
                d2_body[body_idx] = acc2;
                d3_body[body_idx] = acc3;
            }

            if !d0_tail.is_empty() {
                let mut acc0 = simd.partial_load_f64s(d0_tail);
                let mut acc1 = simd.partial_load_f64s(d1_tail);
                let mut acc2 = simd.partial_load_f64s(d2_tail);
                let mut acc3 = simd.partial_load_f64s(d3_tail);
                for q in 0..n_elim {
                    let a0q = alphas0[q];
                    let a1q = alphas1[q];
                    let a2q = alphas2[q];
                    let a3q = alphas3[q];
                    if a0q == 0.0 && a1q == 0.0 && a2q == 0.0 && a3q == 0.0 {
                        continue;
                    }
                    let col_off = (src_first_col + q) * col_stride + src_row_offset_bulk;
                    let src_q = &src_block[col_off..col_off + len];
                    let src_q_tail = &src_q[tail_off..];
                    let s = simd.partial_load_f64s(src_q_tail);
                    if a0q != 0.0 {
                        let nav0 = simd.splat_f64s(-a0q);
                        acc0 = simd.mul_add_f64s(nav0, s, acc0);
                    }
                    if a1q != 0.0 {
                        let nav1 = simd.splat_f64s(-a1q);
                        acc1 = simd.mul_add_f64s(nav1, s, acc1);
                    }
                    if a2q != 0.0 {
                        let nav2 = simd.splat_f64s(-a2q);
                        acc2 = simd.mul_add_f64s(nav2, s, acc2);
                    }
                    if a3q != 0.0 {
                        let nav3 = simd.splat_f64s(-a3q);
                        acc3 = simd.mul_add_f64s(nav3, s, acc3);
                    }
                }
                simd.partial_store_f64s(d0_tail, acc0);
                simd.partial_store_f64s(d1_tail, acc1);
                simd.partial_store_f64s(d2_tail, acc2);
                simd.partial_store_f64s(d3_tail, acc3);
            }
        }
    }

    let (_d0_cap, d0_bulk) = dst0.split_at_mut(3);
    let (_d1_cap, d1_bulk) = dst1.split_at_mut(2);
    let (_d2_cap, d2_bulk) = dst2.split_at_mut(1);
    dispatch_fma(K {
        dst0_bulk: d0_bulk,
        dst1_bulk: d1_bulk,
        dst2_bulk: d2_bulk,
        dst3_bulk: dst3,
        src_block,
        src_first_col,
        n_elim,
        col_stride,
        src_row_offset_bulk: src_row_offset + 3,
        len: len3,
        alphas0,
        alphas1,
        alphas2,
        alphas3,
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

    #[test]
    fn axpy_minus_unroll4_nofma_is_bit_exact_vs_scalar() {
        let mut rng = Xorshift64::new(0xB17E_AC70_0042_F00D_u64);
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

    /// W-2: rank-`n_elim` accumulator must reproduce the bit pattern
    /// produced by `n_elim` sequential `axpy_minus_unroll4_nofma`
    /// calls in q-outer order. Since both kernels use explicit
    /// `mul + sub` (no FMA), the per-element rounding sequence is
    /// identical:
    ///
    ///   acc <- round(acc - round(alpha_q * src_q[i]))   for q in 0..n_elim
    ///
    /// We compare against an explicit reference loop rather than a
    /// rebuilt naive_axpy_minus chain to keep the assertion tight.
    #[test]
    fn schur_panel_minus_nofma_strided_is_bit_exact_vs_rank1_reference() {
        let mut rng = Xorshift64::new(0x517E_3D5C_4242_5042);
        // n_elim values cover the W-2 fast-path range (1×1 panels);
        // dst lengths cover SIMD boundaries.
        let n_elim_sweep = [1usize, 2, 4, 7, 8, 16, 31, 32];
        let len_sweep: &[usize] = &[1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 256, 257];
        for &n_elim in &n_elim_sweep {
            for &len in len_sweep {
                // col_stride > len: simulate the multifrontal layout
                // where each pivot column has trailing slack.
                let col_stride = len + 7 + n_elim;
                // src_block holds `n_elim` columns of `col_stride` each;
                // we treat row 0 of each column as the data the kernel
                // touches (src_row_offset = 0 for simplicity).
                let total = n_elim * col_stride;
                let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
                let dst_init: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
                let alphas: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();

                // Reference: run n_elim rank-1 axpys in q-ascending
                // order. Each call uses the bit-exact rank-1 kernel.
                let mut dst_ref = dst_init.clone();
                for q in 0..n_elim {
                    let alpha = alphas[q];
                    if alpha == 0.0 {
                        continue;
                    }
                    let col_off = q * col_stride;
                    let src_q = &src_block[col_off..col_off + len];
                    axpy_minus_unroll4_nofma(&mut dst_ref, src_q, alpha);
                }

                // Test kernel: single rank-`n_elim` dispatch.
                let mut dst_kernel = dst_init.clone();
                schur_panel_minus_nofma_strided(
                    &mut dst_kernel,
                    &src_block,
                    /* src_first_col */ 0,
                    n_elim,
                    col_stride,
                    /* src_row_offset */ 0,
                    len,
                    &alphas,
                );

                assert_eq!(
                    dst_kernel, dst_ref,
                    "rank-{} accumulator must be bit-exact vs n_elim*rank-1 \
                     at len={}, col_stride={}",
                    n_elim, len, col_stride
                );
            }
        }
    }

    /// Issue #8: FMA variant must be bit-exact with `n_elim`
    /// sequential `axpy_minus_unroll4` (FMA rank-1) calls in
    /// q-ascending order. Both kernels collapse `dst -= alpha_q *
    /// src_q` into a single `mul_add(-alpha_q, src_q, dst)`, so the
    /// per-element rounding sequence is:
    ///
    ///   acc <- round(acc - alpha_q * src_q[i])   for q in 0..n_elim
    ///
    /// (One IEEE 754 rounding per accumulate. The `_nofma` reference
    /// does two — that's why this test pairs FMA against FMA, not
    /// FMA against `axpy_minus_unroll4_nofma`.)
    #[test]
    fn schur_panel_minus_fma_strided_is_bit_exact_vs_rank1_fma_reference() {
        let mut rng = Xorshift64::new(0xF0AA_5C5C_BEEF_F00Du64);
        let n_elim_sweep = [1usize, 2, 4, 7, 8, 16, 31, 32];
        let len_sweep: &[usize] = &[1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 256, 257];
        for &n_elim in &n_elim_sweep {
            for &len in len_sweep {
                let col_stride = len + 7 + n_elim;
                let total = n_elim * col_stride;
                let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
                let dst_init: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
                let alphas: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();

                let mut dst_ref = dst_init.clone();
                for q in 0..n_elim {
                    let alpha = alphas[q];
                    if alpha == 0.0 {
                        continue;
                    }
                    let col_off = q * col_stride;
                    let src_q = &src_block[col_off..col_off + len];
                    axpy_minus_unroll4(&mut dst_ref, src_q, alpha);
                }

                let mut dst_kernel = dst_init.clone();
                schur_panel_minus_fma_strided(
                    &mut dst_kernel,
                    &src_block,
                    /* src_first_col */ 0,
                    n_elim,
                    col_stride,
                    /* src_row_offset */ 0,
                    len,
                    &alphas,
                );

                assert_eq!(
                    dst_kernel, dst_ref,
                    "FMA rank-{} accumulator must be bit-exact vs n_elim*rank-1-FMA \
                     at len={}, col_stride={}",
                    n_elim, len, col_stride
                );
            }
        }
    }

    /// Issue #8: FMA variant must respect the same zero-alpha skip as
    /// the `_nofma` reference.
    #[test]
    fn schur_panel_minus_fma_strided_skips_zero_alphas() {
        let mut rng = Xorshift64::new(0xABCD_5050_AAAA_BBBBu64);
        let n_elim = 4;
        let len = 17;
        let col_stride = len + 5;
        let total = n_elim * col_stride;
        let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
        let dst_init: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
        let alphas = vec![0.5, 0.0, -0.25, 0.0];

        let mut dst_ref = dst_init.clone();
        for q in 0..n_elim {
            let alpha = alphas[q];
            if alpha == 0.0 {
                continue;
            }
            let col_off = q * col_stride;
            let src_q = &src_block[col_off..col_off + len];
            axpy_minus_unroll4(&mut dst_ref, src_q, alpha);
        }

        let mut dst_kernel = dst_init.clone();
        schur_panel_minus_fma_strided(
            &mut dst_kernel,
            &src_block,
            0,
            n_elim,
            col_stride,
            0,
            len,
            &alphas,
        );
        assert_eq!(dst_kernel, dst_ref);
    }

    /// W-2: alpha_q == 0 must be a no-op for that q (matches the
    /// rank-1 reference's `if alpha == 0.0 { continue }` guard).
    #[test]
    fn schur_panel_minus_nofma_strided_skips_zero_alphas() {
        let mut rng = Xorshift64::new(0xABCD_5050_0042u64);
        let n_elim = 4;
        let len = 17;
        let col_stride = len + 5;
        let total = n_elim * col_stride;
        let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
        let dst_init: Vec<f64> = (0..len).map(|_| rng.next_f64()).collect();
        // Mix zero and non-zero alphas; q=1 and q=3 are zero.
        let alphas = vec![0.5, 0.0, -0.25, 0.0];

        let mut dst_ref = dst_init.clone();
        for q in 0..n_elim {
            let alpha = alphas[q];
            if alpha == 0.0 {
                continue;
            }
            let col_off = q * col_stride;
            let src_q = &src_block[col_off..col_off + len];
            axpy_minus_unroll4_nofma(&mut dst_ref, src_q, alpha);
        }

        let mut dst_kernel = dst_init.clone();
        schur_panel_minus_nofma_strided(
            &mut dst_kernel,
            &src_block,
            0,
            n_elim,
            col_stride,
            0,
            len,
            &alphas,
        );
        assert_eq!(dst_kernel, dst_ref);
    }

    /// Phase B-1: dual-column kernel must be bit-exact with two
    /// sequential calls to the rank-`n_elim` strided kernel — one for
    /// column j (length len0) and one for column j+1 (length len0-1).
    #[test]
    fn schur_panel_minus_nofma_strided_dual_is_bit_exact_vs_two_singles() {
        let mut rng = Xorshift64::new(0xD5A1_C0F1_DEAD_BEEFu64);
        let n_elim_sweep = [1usize, 2, 4, 7, 8, 16, 31, 32];
        // len0 = nrow - j; len1 = len0 - 1. Sweep boundary-spanning sizes.
        let len0_sweep: &[usize] = &[
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257,
        ];
        for &n_elim in &n_elim_sweep {
            for &len0 in len0_sweep {
                // src layout: pivot cols at indices [0..n_elim), each
                // of column-stride col_stride. Rows touched: [src_row_offset, src_row_offset + len0).
                let src_row_offset = 3usize;
                let col_stride = src_row_offset + len0 + 5 + n_elim;
                let total = n_elim * col_stride;
                let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();

                let dst0_init: Vec<f64> = (0..len0).map(|_| rng.next_f64()).collect();
                let len1 = len0 - 1;
                let dst1_init: Vec<f64> = (0..len1).map(|_| rng.next_f64()).collect();
                let alphas0: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();
                let alphas1: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();

                // Reference: two separate strided dispatches.
                let mut dst0_ref = dst0_init.clone();
                let mut dst1_ref = dst1_init.clone();
                schur_panel_minus_nofma_strided(
                    &mut dst0_ref,
                    &src_block,
                    0,
                    n_elim,
                    col_stride,
                    src_row_offset,
                    len0,
                    &alphas0,
                );
                if len1 > 0 {
                    schur_panel_minus_nofma_strided(
                        &mut dst1_ref,
                        &src_block,
                        0,
                        n_elim,
                        col_stride,
                        src_row_offset + 1,
                        len1,
                        &alphas1,
                    );
                }

                // Test kernel: single dual dispatch.
                let mut dst0_kernel = dst0_init.clone();
                let mut dst1_kernel = dst1_init.clone();
                schur_panel_minus_nofma_strided_dual(
                    &mut dst0_kernel,
                    &mut dst1_kernel,
                    &src_block,
                    0,
                    n_elim,
                    col_stride,
                    src_row_offset,
                    &alphas0,
                    &alphas1,
                );

                assert_eq!(
                    dst0_kernel, dst0_ref,
                    "dst0 mismatch at n_elim={}, len0={}",
                    n_elim, len0
                );
                assert_eq!(
                    dst1_kernel, dst1_ref,
                    "dst1 mismatch at n_elim={}, len0={}",
                    n_elim, len0
                );
            }
        }
    }

    /// Phase B-1: zero-alpha skips honored independently per column.
    #[test]
    fn schur_panel_minus_nofma_strided_dual_skips_zero_alphas_independently() {
        let mut rng = Xorshift64::new(0xFEED_CAFE_0042_BABEu64);
        let n_elim = 5;
        let len0 = 33;
        let len1 = len0 - 1;
        let src_row_offset = 2usize;
        let col_stride = src_row_offset + len0 + 4 + n_elim;
        let total = n_elim * col_stride;
        let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
        let dst0_init: Vec<f64> = (0..len0).map(|_| rng.next_f64()).collect();
        let dst1_init: Vec<f64> = (0..len1).map(|_| rng.next_f64()).collect();
        // alphas0 zero at q=1; alphas1 zero at q=0,4. q=2 zero in both.
        let alphas0 = vec![0.5, 0.0, 0.0, -0.25, 0.75];
        let alphas1 = vec![0.0, 0.4, 0.0, 0.6, 0.0];

        let mut dst0_ref = dst0_init.clone();
        let mut dst1_ref = dst1_init.clone();
        schur_panel_minus_nofma_strided(
            &mut dst0_ref,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset,
            len0,
            &alphas0,
        );
        schur_panel_minus_nofma_strided(
            &mut dst1_ref,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset + 1,
            len1,
            &alphas1,
        );

        let mut dst0_kernel = dst0_init.clone();
        let mut dst1_kernel = dst1_init.clone();
        schur_panel_minus_nofma_strided_dual(
            &mut dst0_kernel,
            &mut dst1_kernel,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset,
            &alphas0,
            &alphas1,
        );
        assert_eq!(dst0_kernel, dst0_ref);
        assert_eq!(dst1_kernel, dst1_ref);
    }

    /// Phase 2.4.3 (issue #9): quad-column kernel must be bit-exact
    /// with four sequential calls to the single-column strided kernel,
    /// one per dst column.
    #[test]
    fn schur_panel_minus_nofma_strided_quad_is_bit_exact_vs_four_singles() {
        let mut rng = Xorshift64::new(0x9A77_E11E_2026_0512_u64);
        let n_elim_sweep = [1usize, 2, 4, 7, 8, 16, 31, 32];
        // len0 must be >= 4 for the quad path to engage non-trivially.
        // We still test small len0 (1..=3) to exercise the cap-only
        // shortcuts; the dispatch caller skips quad in those cases.
        let len0_sweep: &[usize] = &[
            3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 31, 32, 33, 63, 64, 65, 127, 128, 257,
        ];
        for &n_elim in &n_elim_sweep {
            for &len0 in len0_sweep {
                let src_row_offset = 3usize;
                let col_stride = src_row_offset + len0 + 5 + n_elim;
                let total = n_elim * col_stride;
                let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();

                let len1 = len0 - 1;
                let len2 = len0 - 2;
                let len3 = len0 - 3;
                let dst0_init: Vec<f64> = (0..len0).map(|_| rng.next_f64()).collect();
                let dst1_init: Vec<f64> = (0..len1).map(|_| rng.next_f64()).collect();
                let dst2_init: Vec<f64> = (0..len2).map(|_| rng.next_f64()).collect();
                let dst3_init: Vec<f64> = (0..len3).map(|_| rng.next_f64()).collect();
                let alphas0: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();
                let alphas1: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();
                let alphas2: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();
                let alphas3: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();

                let mut dst0_ref = dst0_init.clone();
                let mut dst1_ref = dst1_init.clone();
                let mut dst2_ref = dst2_init.clone();
                let mut dst3_ref = dst3_init.clone();
                schur_panel_minus_nofma_strided(
                    &mut dst0_ref,
                    &src_block,
                    0,
                    n_elim,
                    col_stride,
                    src_row_offset,
                    len0,
                    &alphas0,
                );
                if len1 > 0 {
                    schur_panel_minus_nofma_strided(
                        &mut dst1_ref,
                        &src_block,
                        0,
                        n_elim,
                        col_stride,
                        src_row_offset + 1,
                        len1,
                        &alphas1,
                    );
                }
                if len2 > 0 {
                    schur_panel_minus_nofma_strided(
                        &mut dst2_ref,
                        &src_block,
                        0,
                        n_elim,
                        col_stride,
                        src_row_offset + 2,
                        len2,
                        &alphas2,
                    );
                }
                if len3 > 0 {
                    schur_panel_minus_nofma_strided(
                        &mut dst3_ref,
                        &src_block,
                        0,
                        n_elim,
                        col_stride,
                        src_row_offset + 3,
                        len3,
                        &alphas3,
                    );
                }

                let mut dst0_kernel = dst0_init.clone();
                let mut dst1_kernel = dst1_init.clone();
                let mut dst2_kernel = dst2_init.clone();
                let mut dst3_kernel = dst3_init.clone();
                schur_panel_minus_nofma_strided_quad(
                    &mut dst0_kernel,
                    &mut dst1_kernel,
                    &mut dst2_kernel,
                    &mut dst3_kernel,
                    &src_block,
                    0,
                    n_elim,
                    col_stride,
                    src_row_offset,
                    &alphas0,
                    &alphas1,
                    &alphas2,
                    &alphas3,
                );

                assert_eq!(
                    dst0_kernel, dst0_ref,
                    "dst0 mismatch at n_elim={}, len0={}",
                    n_elim, len0
                );
                assert_eq!(
                    dst1_kernel, dst1_ref,
                    "dst1 mismatch at n_elim={}, len0={}",
                    n_elim, len0
                );
                assert_eq!(
                    dst2_kernel, dst2_ref,
                    "dst2 mismatch at n_elim={}, len0={}",
                    n_elim, len0
                );
                assert_eq!(
                    dst3_kernel, dst3_ref,
                    "dst3 mismatch at n_elim={}, len0={}",
                    n_elim, len0
                );
            }
        }
    }

    /// Phase 2.4.3 (issue #9): zero-alpha skips honored independently
    /// per column. Each column's reference uses its own alpha pattern;
    /// the all-zero-q skip in the kernel must not affect any column
    /// whose own alpha is non-zero at that q.
    #[test]
    fn schur_panel_minus_nofma_strided_quad_skips_zero_alphas_independently() {
        let mut rng = Xorshift64::new(0xCAFE_F00D_2026_0512_u64);
        let n_elim = 6;
        let len0 = 67;
        let len1 = len0 - 1;
        let len2 = len0 - 2;
        let len3 = len0 - 3;
        let src_row_offset = 2usize;
        let col_stride = src_row_offset + len0 + 4 + n_elim;
        let total = n_elim * col_stride;
        let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
        let dst0_init: Vec<f64> = (0..len0).map(|_| rng.next_f64()).collect();
        let dst1_init: Vec<f64> = (0..len1).map(|_| rng.next_f64()).collect();
        let dst2_init: Vec<f64> = (0..len2).map(|_| rng.next_f64()).collect();
        let dst3_init: Vec<f64> = (0..len3).map(|_| rng.next_f64()).collect();
        // Independent zero patterns: q=0 zero in alphas0; q=1 zero in
        // alphas1,2; q=2 zero in all four (the all-zero-q skip path);
        // q=3 zero in alphas3; q=4 zero in alphas2; q=5 non-zero in all.
        let alphas0 = vec![0.0, 0.5, 0.0, -0.25, 0.75, 1.1];
        let alphas1 = vec![0.4, 0.0, 0.0, 0.6, 0.3, -0.9];
        let alphas2 = vec![-0.5, 0.0, 0.0, 0.2, 0.0, 0.7];
        let alphas3 = vec![0.8, -0.3, 0.0, 0.0, 0.55, 0.4];

        let mut dst0_ref = dst0_init.clone();
        let mut dst1_ref = dst1_init.clone();
        let mut dst2_ref = dst2_init.clone();
        let mut dst3_ref = dst3_init.clone();
        schur_panel_minus_nofma_strided(
            &mut dst0_ref,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset,
            len0,
            &alphas0,
        );
        schur_panel_minus_nofma_strided(
            &mut dst1_ref,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset + 1,
            len1,
            &alphas1,
        );
        schur_panel_minus_nofma_strided(
            &mut dst2_ref,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset + 2,
            len2,
            &alphas2,
        );
        schur_panel_minus_nofma_strided(
            &mut dst3_ref,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset + 3,
            len3,
            &alphas3,
        );

        let mut dst0_kernel = dst0_init.clone();
        let mut dst1_kernel = dst1_init.clone();
        let mut dst2_kernel = dst2_init.clone();
        let mut dst3_kernel = dst3_init.clone();
        schur_panel_minus_nofma_strided_quad(
            &mut dst0_kernel,
            &mut dst1_kernel,
            &mut dst2_kernel,
            &mut dst3_kernel,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset,
            &alphas0,
            &alphas1,
            &alphas2,
            &alphas3,
        );
        assert_eq!(dst0_kernel, dst0_ref);
        assert_eq!(dst1_kernel, dst1_ref);
        assert_eq!(dst2_kernel, dst2_ref);
        assert_eq!(dst3_kernel, dst3_ref);
    }

    /// Phase 1 (issue #8): FMA dual-column kernel must be bit-exact
    /// with two sequential calls to the FMA single-column kernel,
    /// one per dst column. Bit-exactness is against the FMA reference,
    /// not the non-FMA dual kernel.
    #[test]
    fn schur_panel_minus_fma_strided_dual_is_bit_exact_vs_two_fma_singles() {
        let mut rng = Xorshift64::new(0x1234_5678_FACE_FEED_u64);
        let n_elim_sweep = [1usize, 2, 4, 7, 8, 16, 31, 32];
        let len0_sweep: &[usize] = &[
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257,
        ];
        for &n_elim in &n_elim_sweep {
            for &len0 in len0_sweep {
                let src_row_offset = 3usize;
                let col_stride = src_row_offset + len0 + 5 + n_elim;
                let total = n_elim * col_stride;
                let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();

                let dst0_init: Vec<f64> = (0..len0).map(|_| rng.next_f64()).collect();
                let len1 = len0 - 1;
                let dst1_init: Vec<f64> = (0..len1).map(|_| rng.next_f64()).collect();
                let alphas0: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();
                let alphas1: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();

                let mut dst0_ref = dst0_init.clone();
                let mut dst1_ref = dst1_init.clone();
                schur_panel_minus_fma_strided(
                    &mut dst0_ref,
                    &src_block,
                    0,
                    n_elim,
                    col_stride,
                    src_row_offset,
                    len0,
                    &alphas0,
                );
                if len1 > 0 {
                    schur_panel_minus_fma_strided(
                        &mut dst1_ref,
                        &src_block,
                        0,
                        n_elim,
                        col_stride,
                        src_row_offset + 1,
                        len1,
                        &alphas1,
                    );
                }

                let mut dst0_kernel = dst0_init.clone();
                let mut dst1_kernel = dst1_init.clone();
                schur_panel_minus_fma_strided_dual(
                    &mut dst0_kernel,
                    &mut dst1_kernel,
                    &src_block,
                    0,
                    n_elim,
                    col_stride,
                    src_row_offset,
                    &alphas0,
                    &alphas1,
                );

                assert_eq!(
                    dst0_kernel, dst0_ref,
                    "FMA dst0 mismatch at n_elim={}, len0={}",
                    n_elim, len0
                );
                assert_eq!(
                    dst1_kernel, dst1_ref,
                    "FMA dst1 mismatch at n_elim={}, len0={}",
                    n_elim, len0
                );
            }
        }
    }

    #[test]
    fn schur_panel_minus_fma_strided_dual_skips_zero_alphas_independently() {
        let mut rng = Xorshift64::new(0xABCD_1234_F00D_BABE_u64);
        let n_elim = 5;
        let len0 = 33;
        let len1 = len0 - 1;
        let src_row_offset = 2usize;
        let col_stride = src_row_offset + len0 + 4 + n_elim;
        let total = n_elim * col_stride;
        let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
        let dst0_init: Vec<f64> = (0..len0).map(|_| rng.next_f64()).collect();
        let dst1_init: Vec<f64> = (0..len1).map(|_| rng.next_f64()).collect();
        let alphas0 = vec![0.5, 0.0, 0.0, -0.25, 0.75];
        let alphas1 = vec![0.0, 0.4, 0.0, 0.6, 0.0];

        let mut dst0_ref = dst0_init.clone();
        let mut dst1_ref = dst1_init.clone();
        schur_panel_minus_fma_strided(
            &mut dst0_ref,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset,
            len0,
            &alphas0,
        );
        schur_panel_minus_fma_strided(
            &mut dst1_ref,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset + 1,
            len1,
            &alphas1,
        );

        let mut dst0_kernel = dst0_init.clone();
        let mut dst1_kernel = dst1_init.clone();
        schur_panel_minus_fma_strided_dual(
            &mut dst0_kernel,
            &mut dst1_kernel,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset,
            &alphas0,
            &alphas1,
        );
        assert_eq!(dst0_kernel, dst0_ref);
        assert_eq!(dst1_kernel, dst1_ref);
    }

    /// Phase 1 (issue #8): FMA quad-column kernel must be bit-exact
    /// with four sequential calls to the FMA single-column kernel.
    #[test]
    fn schur_panel_minus_fma_strided_quad_is_bit_exact_vs_four_fma_singles() {
        let mut rng = Xorshift64::new(0xDEAD_BEEF_2026_0513_u64);
        let n_elim_sweep = [1usize, 2, 4, 7, 8, 16, 31, 32];
        let len0_sweep: &[usize] = &[
            3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 31, 32, 33, 63, 64, 65, 127, 128, 257,
        ];
        for &n_elim in &n_elim_sweep {
            for &len0 in len0_sweep {
                let src_row_offset = 3usize;
                let col_stride = src_row_offset + len0 + 5 + n_elim;
                let total = n_elim * col_stride;
                let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();

                let len1 = len0 - 1;
                let len2 = len0 - 2;
                let len3 = len0 - 3;
                let dst0_init: Vec<f64> = (0..len0).map(|_| rng.next_f64()).collect();
                let dst1_init: Vec<f64> = (0..len1).map(|_| rng.next_f64()).collect();
                let dst2_init: Vec<f64> = (0..len2).map(|_| rng.next_f64()).collect();
                let dst3_init: Vec<f64> = (0..len3).map(|_| rng.next_f64()).collect();
                let alphas0: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();
                let alphas1: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();
                let alphas2: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();
                let alphas3: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();

                let mut dst0_ref = dst0_init.clone();
                let mut dst1_ref = dst1_init.clone();
                let mut dst2_ref = dst2_init.clone();
                let mut dst3_ref = dst3_init.clone();
                schur_panel_minus_fma_strided(
                    &mut dst0_ref,
                    &src_block,
                    0,
                    n_elim,
                    col_stride,
                    src_row_offset,
                    len0,
                    &alphas0,
                );
                if len1 > 0 {
                    schur_panel_minus_fma_strided(
                        &mut dst1_ref,
                        &src_block,
                        0,
                        n_elim,
                        col_stride,
                        src_row_offset + 1,
                        len1,
                        &alphas1,
                    );
                }
                if len2 > 0 {
                    schur_panel_minus_fma_strided(
                        &mut dst2_ref,
                        &src_block,
                        0,
                        n_elim,
                        col_stride,
                        src_row_offset + 2,
                        len2,
                        &alphas2,
                    );
                }
                if len3 > 0 {
                    schur_panel_minus_fma_strided(
                        &mut dst3_ref,
                        &src_block,
                        0,
                        n_elim,
                        col_stride,
                        src_row_offset + 3,
                        len3,
                        &alphas3,
                    );
                }

                let mut dst0_kernel = dst0_init.clone();
                let mut dst1_kernel = dst1_init.clone();
                let mut dst2_kernel = dst2_init.clone();
                let mut dst3_kernel = dst3_init.clone();
                schur_panel_minus_fma_strided_quad(
                    &mut dst0_kernel,
                    &mut dst1_kernel,
                    &mut dst2_kernel,
                    &mut dst3_kernel,
                    &src_block,
                    0,
                    n_elim,
                    col_stride,
                    src_row_offset,
                    &alphas0,
                    &alphas1,
                    &alphas2,
                    &alphas3,
                );

                assert_eq!(
                    dst0_kernel, dst0_ref,
                    "FMA dst0 mismatch at n_elim={}, len0={}",
                    n_elim, len0
                );
                assert_eq!(
                    dst1_kernel, dst1_ref,
                    "FMA dst1 mismatch at n_elim={}, len0={}",
                    n_elim, len0
                );
                assert_eq!(
                    dst2_kernel, dst2_ref,
                    "FMA dst2 mismatch at n_elim={}, len0={}",
                    n_elim, len0
                );
                assert_eq!(
                    dst3_kernel, dst3_ref,
                    "FMA dst3 mismatch at n_elim={}, len0={}",
                    n_elim, len0
                );
            }
        }
    }

    #[test]
    fn schur_panel_minus_fma_strided_quad_skips_zero_alphas_independently() {
        let mut rng = Xorshift64::new(0xF00D_FEED_2026_0513_u64);
        let n_elim = 6;
        let len0 = 67;
        let len1 = len0 - 1;
        let len2 = len0 - 2;
        let len3 = len0 - 3;
        let src_row_offset = 2usize;
        let col_stride = src_row_offset + len0 + 4 + n_elim;
        let total = n_elim * col_stride;
        let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
        let dst0_init: Vec<f64> = (0..len0).map(|_| rng.next_f64()).collect();
        let dst1_init: Vec<f64> = (0..len1).map(|_| rng.next_f64()).collect();
        let dst2_init: Vec<f64> = (0..len2).map(|_| rng.next_f64()).collect();
        let dst3_init: Vec<f64> = (0..len3).map(|_| rng.next_f64()).collect();
        let alphas0 = vec![0.0, 0.5, 0.0, -0.25, 0.75, 1.1];
        let alphas1 = vec![0.4, 0.0, 0.0, 0.6, 0.3, -0.9];
        let alphas2 = vec![-0.5, 0.0, 0.0, 0.2, 0.0, 0.7];
        let alphas3 = vec![0.8, -0.3, 0.0, 0.0, 0.55, 0.4];

        let mut dst0_ref = dst0_init.clone();
        let mut dst1_ref = dst1_init.clone();
        let mut dst2_ref = dst2_init.clone();
        let mut dst3_ref = dst3_init.clone();
        schur_panel_minus_fma_strided(
            &mut dst0_ref,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset,
            len0,
            &alphas0,
        );
        schur_panel_minus_fma_strided(
            &mut dst1_ref,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset + 1,
            len1,
            &alphas1,
        );
        schur_panel_minus_fma_strided(
            &mut dst2_ref,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset + 2,
            len2,
            &alphas2,
        );
        schur_panel_minus_fma_strided(
            &mut dst3_ref,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset + 3,
            len3,
            &alphas3,
        );

        let mut dst0_kernel = dst0_init.clone();
        let mut dst1_kernel = dst1_init.clone();
        let mut dst2_kernel = dst2_init.clone();
        let mut dst3_kernel = dst3_init.clone();
        schur_panel_minus_fma_strided_quad(
            &mut dst0_kernel,
            &mut dst1_kernel,
            &mut dst2_kernel,
            &mut dst3_kernel,
            &src_block,
            0,
            n_elim,
            col_stride,
            src_row_offset,
            &alphas0,
            &alphas1,
            &alphas2,
            &alphas3,
        );
        assert_eq!(dst0_kernel, dst0_ref);
        assert_eq!(dst1_kernel, dst1_ref);
        assert_eq!(dst2_kernel, dst2_ref);
        assert_eq!(dst3_kernel, dst3_ref);
    }

    /// Phase 1 (issue #8): cross-policy tolerance — FMA and non-FMA
    /// variants of the same kernel must agree within
    /// `n_elim * EPS * max(|fma|, |nofma|, 1.0)`. The FMA path uses one
    /// rounding per multiply-accumulate; the non-FMA path uses two. Per
    /// q step they differ by ≤ 1 ULP of the smaller magnitude; over
    /// `n_elim` steps the worst-case drift is `n_elim` ULPs.
    ///
    /// Covers all five panel kernels:
    ///   strided (single), strided_dual, strided_quad,
    ///   axpy_minus_unroll4 (rank-1 reference), axpy2_minus_unroll4 (rank-2).
    #[test]
    fn fma_vs_nofma_panel_kernels_within_n_elim_ulps() {
        let mut rng = Xorshift64::new(0xC0DE_BABE_A11A_FEED_u64);
        let n_elim_sweep = [1usize, 4, 8, 16, 32];
        let len0_sweep: &[usize] = &[1, 4, 8, 17, 32, 65, 257];

        let close = |fma: &[f64], nofma: &[f64], n_elim: usize, label: &str| {
            assert_eq!(fma.len(), nofma.len(), "length mismatch in {}", label);
            let tol_factor = (n_elim as f64) * f64::EPSILON;
            for i in 0..fma.len() {
                let diff = (fma[i] - nofma[i]).abs();
                let scale = fma[i].abs().max(nofma[i].abs()).max(1.0);
                let tol = tol_factor * scale * 4.0;
                assert!(
                    diff <= tol,
                    "{}: idx {}: fma={} nofma={} diff={:.3e} > tol={:.3e} (n_elim={})",
                    label,
                    i,
                    fma[i],
                    nofma[i],
                    diff,
                    tol,
                    n_elim
                );
            }
        };

        for &n_elim in &n_elim_sweep {
            for &len0 in len0_sweep {
                let src_row_offset = 3usize;
                let col_stride = src_row_offset + len0 + 5 + n_elim;
                let total = n_elim * col_stride;
                let src_block: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
                let dst0_init: Vec<f64> = (0..len0).map(|_| rng.next_f64()).collect();
                let alphas0: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();

                // Single-column strided.
                let mut dst_fma = dst0_init.clone();
                let mut dst_nofma = dst0_init.clone();
                schur_panel_minus_fma_strided(
                    &mut dst_fma,
                    &src_block,
                    0,
                    n_elim,
                    col_stride,
                    src_row_offset,
                    len0,
                    &alphas0,
                );
                schur_panel_minus_nofma_strided(
                    &mut dst_nofma,
                    &src_block,
                    0,
                    n_elim,
                    col_stride,
                    src_row_offset,
                    len0,
                    &alphas0,
                );
                close(&dst_fma, &dst_nofma, n_elim, "strided");

                // Dual.
                if len0 >= 1 {
                    let len1 = len0 - 1;
                    let dst1_init: Vec<f64> = (0..len1).map(|_| rng.next_f64()).collect();
                    let alphas1: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();

                    let mut d0_fma = dst0_init.clone();
                    let mut d1_fma = dst1_init.clone();
                    let mut d0_nofma = dst0_init.clone();
                    let mut d1_nofma = dst1_init.clone();
                    schur_panel_minus_fma_strided_dual(
                        &mut d0_fma,
                        &mut d1_fma,
                        &src_block,
                        0,
                        n_elim,
                        col_stride,
                        src_row_offset,
                        &alphas0,
                        &alphas1,
                    );
                    schur_panel_minus_nofma_strided_dual(
                        &mut d0_nofma,
                        &mut d1_nofma,
                        &src_block,
                        0,
                        n_elim,
                        col_stride,
                        src_row_offset,
                        &alphas0,
                        &alphas1,
                    );
                    close(&d0_fma, &d0_nofma, n_elim, "dual.dst0");
                    close(&d1_fma, &d1_nofma, n_elim, "dual.dst1");
                }

                // Quad. len0 >= 3 required for the bulk to engage; cap
                // cases (1..=3) still exercise the prologue scalar path.
                if len0 >= 1 {
                    let len1 = len0 - 1;
                    let len2 = len0.saturating_sub(2);
                    let len3 = len0.saturating_sub(3);
                    let dst1_init: Vec<f64> = (0..len1).map(|_| rng.next_f64()).collect();
                    let dst2_init: Vec<f64> = (0..len2).map(|_| rng.next_f64()).collect();
                    let dst3_init: Vec<f64> = (0..len3).map(|_| rng.next_f64()).collect();
                    let alphas1: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();
                    let alphas2: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();
                    let alphas3: Vec<f64> = (0..n_elim).map(|_| rng.next_f64() * 1.5).collect();

                    if len0 >= 1 && len1 + 1 == len0 && len2 + 2 == len0 && len3 + 3 == len0 {
                        let mut d0_fma = dst0_init.clone();
                        let mut d1_fma = dst1_init.clone();
                        let mut d2_fma = dst2_init.clone();
                        let mut d3_fma = dst3_init.clone();
                        let mut d0_nofma = dst0_init.clone();
                        let mut d1_nofma = dst1_init.clone();
                        let mut d2_nofma = dst2_init.clone();
                        let mut d3_nofma = dst3_init.clone();
                        schur_panel_minus_fma_strided_quad(
                            &mut d0_fma,
                            &mut d1_fma,
                            &mut d2_fma,
                            &mut d3_fma,
                            &src_block,
                            0,
                            n_elim,
                            col_stride,
                            src_row_offset,
                            &alphas0,
                            &alphas1,
                            &alphas2,
                            &alphas3,
                        );
                        schur_panel_minus_nofma_strided_quad(
                            &mut d0_nofma,
                            &mut d1_nofma,
                            &mut d2_nofma,
                            &mut d3_nofma,
                            &src_block,
                            0,
                            n_elim,
                            col_stride,
                            src_row_offset,
                            &alphas0,
                            &alphas1,
                            &alphas2,
                            &alphas3,
                        );
                        close(&d0_fma, &d0_nofma, n_elim, "quad.dst0");
                        close(&d1_fma, &d1_nofma, n_elim, "quad.dst1");
                        close(&d2_fma, &d2_nofma, n_elim, "quad.dst2");
                        close(&d3_fma, &d3_nofma, n_elim, "quad.dst3");
                    }
                }

                // axpy_minus_unroll4 (FMA) vs axpy_minus_unroll4_nofma.
                // Single rank-1, not n_elim — drift bound is 1 ULP.
                let src: Vec<f64> = (0..len0).map(|_| rng.next_f64()).collect();
                let alpha = rng.next_f64() * 1.5;
                let mut d_fma = dst0_init.clone();
                let mut d_nofma = dst0_init.clone();
                axpy_minus_unroll4(&mut d_fma, &src, alpha);
                axpy_minus_unroll4_nofma(&mut d_nofma, &src, alpha);
                close(&d_fma, &d_nofma, 1, "axpy_minus_unroll4");

                // axpy2_minus_unroll4 (FMA) vs axpy2_minus_unroll4_nofma.
                let src_b: Vec<f64> = (0..len0).map(|_| rng.next_f64()).collect();
                let alpha_b = rng.next_f64() * 1.5;
                let mut d2_fma = dst0_init.clone();
                let mut d2_nofma = dst0_init.clone();
                axpy2_minus_unroll4(&mut d2_fma, &src, alpha, &src_b, alpha_b);
                axpy2_minus_unroll4_nofma(&mut d2_nofma, &src, alpha, &src_b, alpha_b);
                // Two rank-1's so drift bound is 2 ULP.
                close(&d2_fma, &d2_nofma, 2, "axpy2_minus_unroll4");
            }
        }
    }

    #[test]
    fn axpy2_minus_unroll4_nofma_is_bit_exact_vs_scalar() {
        let mut rng = Xorshift64::new(0xB17E_AC70_BAAD_F00D_u64);
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
