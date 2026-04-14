# Phase 2.4.2 — SIMD Micro-Kernel for the Schur Update

**Date:** 2026-04-14
**Related:** `dev/plans/phase-2-planning.md` §2.4.2, `dev/tried-and-rejected.md`
(Phase 2.4.1a null result), `dev/research/dense-ldlt.md` §10.2
**Baseline to beat:** dense factor p90 vs MUMPS 2.27 (target ≤ 2.0);
sparse factor p90 vs MUMPS 3.18 (target ≤ 3.0).

## Why a SIMD kernel is the right next step

The Phase 2.4.1a attempt established empirically and confirmed via
faer-expert source trace that **pure scalar loop reordering does
nothing** for the dense LDLᵀ factorization. The rank-1/rank-2 Schur
update arithmetic is the same whether you batch it into a triangular
matmul or apply it per pivot, so without a vectorized kernel the block
structure is pure overhead. Bench numbers: sparse p90 3.18 → 3.53
(+11%) when the deferral was added with scalar loops; 80/80 tests
pass, zero inertia regressions, speed regression only. See
`dev/tried-and-rejected.md` §"Phase 2.4.1a contribution-block
deferral".

faer-expert verified that faer's entire blocked Bunch-Kaufman
advantage lives in `linalg::matmul::triangular::matmul` at
`bunch_kaufman/factor.rs:684`, which lowers to a pulp-dispatched
register-blocked SIMD GEMM (`matmul_simd` → `Ukr<MR, NR, T>`, MR ×
NR register tiles, masked tail loads). The panel routine
`lblt_blocked_step` is plain scalar Rust. The speedup is 100%
vectorization, 0% loop reordering.

Conclusion: the only remaining lever for the Phase 2 exit criterion is
a vectorized inner kernel. This research note scopes that kernel and
picks the tooling.

## The hot loop

From `src/dense/factor.rs` (post-revert state, commit `6ca8832`),
`do_1x1_update` at line ~1063:

```rust
fn do_1x1_update(a: &mut [f64], n: usize, k: usize) {
    let d = a[k * n + k];
    if d.abs() == 0.0 { return; }
    let inv_d = 1.0 / d;
    for i in (k + 1)..n {
        a[k * n + i] *= inv_d;
    }
    for j in (k + 1)..n {
        let l_jk = a[k * n + j];
        for i in j..n {
            a[j * n + i] -= l_jk * d * a[k * n + i];
        }
    }
}
```

The doubly-nested rank-1 update is the hot loop. Per pivot `k` it
does `O((n-k)²/2)` FMAs. Summed over all `n` pivots of a front of
dimension `n` that's the dominant `n³/6` term of LDLᵀ.

**Inner-loop structure (fixed `j`, inner `i`):**

```rust
let alpha = -l_jk * d;                      // loop-invariant
let src  = &a[k*n + j .. k*n + n];          // unit stride, length n - j
let dst  = &mut a[j*n + j .. j*n + n];      // unit stride, length n - j
for i in 0..(n - j) {
    dst[i] += alpha * src[i];
}
```

This is a textbook AXPY (`y += α · x`): unit-stride loads on both
operands, FMA-shaped (one multiply, one add), loop-invariant scalar
multiplier, no aliasing (`src` is column `k`, `dst` is column `j`,
and the outer loop ensures `j > k`). The length varies from
`n - k - 1` at the first inner iteration down to `1` at the last.

This is as ideal a SIMD target as factorization code ever provides.
AVX2 gives 4 doubles per FMA instruction (× 2 FMA ports on Zen/Skylake
= 8 doubles per cycle); AVX-512 gives 8 per FMA. A well-tuned kernel
should get within 70–90% of peak on this loop for lengths above ~64.

**The rank-2 twin** in `do_2x2_update` has the same inner structure
but with two source columns and two α's:

```rust
let alpha0 = -(d11 * l_j0 + d21 * l_j1);    // column-k contribution
let alpha1 = -(d21 * l_j0 + d22 * l_j1);    // column-(k+1) contribution
for i in 0..(n - j) {
    dst[i] += alpha0 * src0[i] + alpha1 * src1[i];
}
```

Still FMA-shaped, still unit-stride, two fused multiply-adds per
iteration. Same kernel, one extra source column.

## SIMD tooling: hand-rolled intrinsics vs pulp

The constraint stack:

- **Stable Rust:** rules out `std::simd` (still nightly behind
  `portable_simd`).
- **Zero non-Rust deps:** per CLAUDE.md "Pure Rust, stable toolchain;
  zero non-Rust dependencies in the core solver". `pulp` is pure Rust,
  so this constraint does not exclude it.
- **Clean-room from papers:** philosophical preference for minimal
  deps, but we've already accepted `serde` and `serde_json`.

### Option A: hand-rolled `core::arch` intrinsics

Write two functions per operation (one AVX2/FMA, one scalar fallback),
gated behind `#[cfg(target_arch = "x86_64")]` and
`#[target_feature(enable = "avx2,fma")]`, dispatched via
`is_x86_feature_detected!("avx2")`. For aarch64, add a third
function with `core::arch::aarch64::*` NEON intrinsics behind
`#[cfg(target_arch = "aarch64")]`.

- **Pros:** zero new deps, total control, mapping from source to
  instructions is explicit.
- **Cons:** `unsafe` blocks with safety comments in `src/`, two (or
  three) separate kernels to keep synchronized, we own the
  target-feature attribute soup, and the kernel won't benefit from
  AVX-512 without a fourth variant. 4–6 hours budgeted but realistic
  time is 8–12 once you add the tests and the correctness harness.

### Option B: pulp

`pulp 0.22.2` (matches faer's version). Write the kernel once as an
`impl pulp::WithSimd`, use the `simd.f64s_mul_add` vocabulary, and
`pulp::Arch::new().dispatch(kernel)` does CPU feature detection and
routes to the best monomorphized variant at runtime (SSE2 / AVX2 /
AVX-512 on x86_64, NEON on aarch64, wasm SIMD on wasm, scalar
fallback everywhere else).

- **Pros:** one kernel, cross-arch for free, no `unsafe` blocks in
  feral code, battle-tested as faer's SIMD backbone, scales to AVX-512
  without a code rewrite, masked tail handling is built in, ~10x less
  code than option A.
- **Cons:** one new runtime dependency, one more crate we trust and
  don't fully audit. pulp itself contains `unsafe` blocks (it has to
  — it wraps `core::arch` intrinsics) but they're inside the crate,
  not in our source.

### Decision

**Use pulp.** Rationale:

1. The Phase 2.4.2 effort budget (4–6 hours) is realistic only with
   pulp. Writing separate AVX2 and NEON kernels ourselves would eat
   that budget just on boilerplate and leave no time for the
   microbench and integration work.
2. The `unsafe` boundary stays out of `src/`, which is consistent
   with the CLAUDE.md rule and makes future refactors safer.
3. faer has already validated pulp at scale on exactly this workload
   — if we're trying to close the gap with faer's blocked BK, using
   the same SIMD backbone is the shortest path.
4. The "pure Rust / zero non-Rust deps" rule is about BLAS, LAPACK,
   and Fortran. pulp is MIT/Apache-2.0 pure Rust and does not
   contradict the intent of that rule.

**Noted replacement trigger (future work):** if we ever need to ship
feral as a zero-external-dep crate (e.g. for embedded or hardened
environments), replace pulp with hand-rolled `core::arch::x86_64` +
`core::arch::aarch64` kernels at that time. The interface boundary
(a single `schur_axpy_minus` function plus a rank-2 variant) is small
enough that the swap is mechanical. Recorded in `dev/decisions.md`.

## Expected speedup

Scalar inner loop on a 2014-era Haswell core:

- `dst[i] += alpha * src[i]` → 1 FMA + 2 loads + 1 store per iter.
- Effective throughput: ~1 double/cycle (memory-bound on L1).

AVX2 + FMA:

- `_mm256_fmadd_pd(a, x, y)` processes 4 doubles per instruction.
- With two FMA ports (Haswell and later): theoretical peak
  8 doubles/cycle.
- Realistic on an AXPY with L1-resident data: 4–6 doubles/cycle.
- **Expected speedup over scalar: 4–6×** on the inner loop itself.
- On the full factorization, Amdahl's law caps the total speedup
  at whatever fraction of time is spent inside `do_1x1_update`.
  For a medium front (n ~500) that's ~80% of wall time, so the
  full-factor speedup should land around 3–4×.

AVX-512 (Ice Lake and later):

- 8 doubles per FMA, so theoretical 16 doubles/cycle.
- Realistic 8–12 doubles/cycle on L1-resident AXPY.
- **Expected speedup over scalar: 8–12×** on the inner loop.
- Full-factor speedup ~5–8×.

pulp will pick whichever is available at runtime. The dispatch is a
single branch at the top of the kernel; monomorphization means there's
no per-call cost inside the inner loop.

**Amdahl check against the Phase 2 exit criterion:**

- Current dense p90 vs MUMPS: 2.27 → target 2.0. That's a 12%
  reduction needed.
- Current sparse p90 vs MUMPS: 3.18 → target 3.0. That's a 6%
  reduction needed.

Both targets are well within a 3–4× speedup on the inner loop,
assuming the Amdahl fraction is above 50% on the relevant matrix
sizes. Even a conservative 2× on the inner loop clears both bars.

## Microbenchmark design

A `benches/schur_kernel.rs` criterion benchmark that:

1. Isolates the inner AXPY. For lengths `L ∈ {8, 16, 32, 64, 128,
   256, 512, 1024}`, times one scalar and one SIMD call on pre-warmed
   buffers. Expected: SIMD curve crosses below scalar around L=8,
   asymptotes at 4–8× for L ≥ 128.
2. Isolates the rank-2 twin (two sources, two α's). Same length sweep.
3. Times a full `do_1x1_update` on a synthetic lower-triangular
   trailing matrix of dimension `n ∈ {64, 128, 256, 512, 1024}`.
   Compares scalar vs SIMD end-to-end on the whole cascading update.

The microbench is the ground truth for "did we actually write a SIMD
kernel that's faster than scalar". The full KKT bench is the ground
truth for "did it move the p90 needle".

## Correctness testing

The existing `src/dense/factor.rs` test module covers correctness on
hand-computed Bunch-Kaufman examples. The SIMD kernel must produce
*bitwise-identical* or *within 1 ULP* results compared to the scalar
fallback on:

1. **Random symmetric positive definite matrices** (sizes 8, 16, 17,
   64, 65, 128, 129, 256, 257, 512). The `17`, `65`, `129`, `257`
   sizes exercise the masked-tail path in pulp — one element past
   a SIMD register boundary.
2. **Random symmetric indefinite matrices** — same size sweep.
3. **The full Bunch-Kaufman test suite** (existing tests) re-run
   unchanged.
4. **The full KKT corpus bench** — inertia and residuals must match
   the Phase 2.1.8 baseline byte-for-byte.

Floating-point reproducibility caveat: FMA produces a different
bit-for-bit result than separate mul+add because it does one rounding
instead of two. If pulp's SIMD path uses FMA and the scalar fallback
does not, results will differ by 1 ULP. This is acceptable as long as
inertia matches exactly and the refined solve residual is not worse.

## Risks

1. **FMA changes inertia on edge cases.** A 1-ULP difference in the
   Schur update can flip a pivot from "just above `zero_tol`" to
   "just below", changing the inertia. Mitigation: run the full KKT
   bench and diff inertia against the baseline; if any matrix flips,
   investigate whether the scalar path was on the wrong side of the
   threshold and adjust `zero_tol` or accept the flip as an
   improvement (NOT via loosening the test tolerance — per CLAUDE.md
   hard rule).
2. **pulp version churn.** Pin to `0.22.2` exactly, matching faer.
   Review the changelog before bumping.
3. **Short-vector overhead.** For lengths < 16 the SIMD path may be
   *slower* than scalar due to dispatch and tail masking cost.
   Mitigation: add a length threshold check at the top of the SIMD
   entry point, fall through to the scalar loop for small inputs.
   Measure this crossover in the microbench.
4. **pulp's `default-features = false, features = ["x86-v3"]`**
   matches faer's config. If the target machine lacks AVX2, pulp
   falls back to SSE2; we need to verify this still compiles.
   (We're on Apple M-series anyway, so the aarch64 NEON path is the
   actual runtime target on the dev machine. pulp's aarch64 support
   is production-quality in faer.)

## Exit criterion

Phase 2.4.2 is done when all of:

1. Microbench shows ≥ 2× SIMD-over-scalar speedup on the inner AXPY
   at L = 256 (conservative floor — we hope for 4×).
2. All existing dense factor tests pass with the SIMD path active
   (bit-for-bit or within 1 ULP, inertia exact).
3. Full KKT bench passes with zero inertia regressions vs the Phase
   2.1.8 baseline (dense 152911/154481, sparse 153009/154588).
4. Dense factor p90 vs MUMPS ≤ 2.0.
5. No individual matrix in the top-100 worst dense list has its
   ratio increase by more than 10%.
6. The scalar fallback is retained and exercisable (pulp handles
   this automatically via CPU feature detection — if the runtime
   has no SIMD, the scalar variant runs).

Items 1–3 are hard gates. Items 4–5 are the performance target.
If 4 misses but the microbench (1) and correctness (2, 3) hold, the
kernel ships and we look for the next bottleneck (Phase 2.4.3 or
2.5.x).
