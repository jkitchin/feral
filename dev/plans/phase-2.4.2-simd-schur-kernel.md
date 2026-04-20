# Phase 2.4.2 — SIMD Micro-Kernel for the Schur Update

**Status:** Pre-implementation plan (updated 2026-04-20 after Phase 2.4.1b close)
**Date:** 2026-04-14 (updated 2026-04-20)
**Research note:** `dev/research/phase-2.4.2-simd-schur-kernel.md`
**Related:** `dev/plans/phase-2-planning.md` §2.4.2,
`dev/plans/phase-2.4.1-blocked-ldlt.md`
**Baseline:** `dev/sessions/2026-04-20-07.md`
**Targets:** close the mid-tail sparse regressions observed after Phase
2.4.1b wire-up (CRESC100 7.62→10.10, KIRBY2 +0.4–0.6, VESUVIO
+0.15–0.89) while preserving arrow-KKT wins (MUONSINE 10.86→9.14,
VESUVIA 8.43→7.21).

## Addendum 2026-04-20 — Updated kernel surface

Phase 2.4.1b added two new hot-path kernels:

- `schur_kernel::axpy_minus_unroll4_nofma` — 4-way unrolled scalar
  AXPY used by both `do_1x1_update` (scalar kernel) and the new
  `peek_ahead_column` / `apply_blocked_schur` (blocked kernel) in
  `src/dense/factor.rs`.
- `apply_blocked_schur` — deferred rank-k update after a panel. Hot
  inner loop calls `axpy_minus_unroll4_nofma` in a column-outer /
  pivot-inner traversal.

Both call paths must switch together. If SIMD lands only for the
blocked `apply_blocked_schur` and not for scalar `do_1x1_update`, the
bit-parity tests in `tests/blocked_ldlt.rs` break. The cleanest move is
to replace `axpy_minus_unroll4_nofma` itself with the pulp-dispatched
version — all callers (both blocked and scalar) get SIMD simultaneously
and parity is preserved by construction.

This means Step 5 below ("wire into factor.rs") becomes: update the
single `axpy_minus_unroll4_nofma` definition in `schur_kernel.rs`.
No call-site changes required at factor.rs.

## Goal

Replace the scalar inner AXPY in `do_1x1_update` and the rank-2
twin in `do_2x2_update` with a pulp-dispatched SIMD kernel.
Preserve exact inertia, correctness tolerance, and the existing
public API. The scalar loops in feral are retained as the fallback
path via pulp's CPU feature detection (automatic).

## Design

One pulp kernel per operation, colocated in a new module
`src/dense/schur_kernel.rs`:

```rust
// SAFETY: no unsafe in feral source — pulp owns it.
pub(crate) fn axpy_minus(dst: &mut [f64], src: &[f64], alpha: f64) {
    struct K<'a> { dst: &'a mut [f64], src: &'a [f64], alpha: f64 }
    impl pulp::WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: pulp::Simd>(self, simd: S) -> () {
            // dst[i] -= alpha * src[i]  via simd.f64s_mul_add
            // masked tail handled by pulp's prefix/suffix helpers
        }
    }
    pulp::Arch::new().dispatch(K { dst, src, alpha });
}

pub(crate) fn axpy2_minus(
    dst: &mut [f64],
    src0: &[f64], alpha0: f64,
    src1: &[f64], alpha1: f64,
) { /* rank-2 twin */ }
```

Call sites in `src/dense/factor.rs`:

```rust
// do_1x1_update inner loop:
for j in (k + 1)..n {
    let l_jk = a[k * n + j];
    let alpha = l_jk * d;   // positive coefficient; kernel subtracts
    let src_start = k * n + j;
    let dst_start = j * n + j;
    let len = n - j;
    // ... split borrows for dst and src ...
    schur_kernel::axpy_minus(dst, src, alpha);
}
```

The split-borrow dance is the only awkward bit. `a[k*n+j..k*n+n]`
(source, column k) and `a[j*n+j..j*n+n]` (destination, column j) do
not overlap because `j > k`. Use `split_at_mut` to get two disjoint
mutable slices without `unsafe`.

## Step-by-step implementation

One commit per step. Tests before code.

### Step 1. Add pulp to Cargo.toml.

```toml
[dependencies]
pulp = { version = "0.22.2", default-features = false, features = ["x86-v3"] }
```

Matches faer. `default-features = false` avoids pulling in optional
deps; `x86-v3` enables AVX2/FMA codegen paths at compile time. On
Apple Silicon the dev machine uses the aarch64 NEON path, which is
available without any feature flag.

Record the dep in `dev/decisions.md` under a new entry:
"2026-04-14 — Accepted pulp 0.22.2 as SIMD backbone". Include the
replacement trigger condition and interface boundary so a future
session can swap it out.

Commit message: "Phase 2.4.2 Step 1: add pulp 0.22.2 dep for
SIMD kernel".

### Step 2. Write the correctness harness.

New file `src/dense/schur_kernel.rs` with a `#[cfg(test)]` module
that will eventually test both `axpy_minus` and `axpy2_minus`. For
step 2 the module contains only the scalar *reference* versions:

```rust
fn scalar_axpy_minus(dst: &mut [f64], src: &[f64], alpha: f64) {
    for i in 0..dst.len() {
        dst[i] -= alpha * src[i];
    }
}
```

And four tests:

1. `axpy_minus_zero_length` — empty slices, no-op.
2. `axpy_minus_length_one` — single element, exact.
3. `axpy_minus_random_lengths` — lengths `[1, 2, 3, 4, 5, 7, 8, 9,
   15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
   512, 1024]`, random uniform inputs, compare against a naive
   reference. This test at this step runs against `scalar_axpy_minus`
   itself (tautology — sanity check the harness), and in step 3 it
   will run against the real `axpy_minus` and the scalar reference
   side-by-side.
4. `axpy2_minus_*` triple for the rank-2 twin.

Commit message: "Phase 2.4.2 Step 2: schur_kernel test harness,
scalar reference".

### Step 3. Write the pulp-backed kernels.

Replace the placeholder functions with actual pulp kernels. Use
`simd.f64s_mul_add` for the per-lane FMA, `simd.f64s_splat` for the
scalar broadcast, and pulp's prefix/suffix slice methods for the
masked tail (faer uses `simd.mask_between` or similar).

Run the step 2 tests — they now compare the SIMD kernel against the
scalar reference. Acceptable delta per CLAUDE.md hard rule: bit-for-bit
OR within 1 ULP (FMA-vs-mul-add rounding difference is permitted).
Anything larger is a bug.

Commit message: "Phase 2.4.2 Step 3: pulp-dispatched SIMD axpy and
axpy2 kernels".

### Step 4. Microbenchmark.

New file `benches/schur_kernel.rs`. Criterion groups for:

- `axpy_minus_scalar_vs_simd` at lengths 8, 16, 32, 64, 128, 256,
  512, 1024.
- `axpy2_minus_scalar_vs_simd` at the same lengths.
- `do_1x1_update_scalar_vs_simd` at triangular sizes `n ∈ {64,
  128, 256, 512, 1024}`. This is the end-to-end test of the hot
  loop, not just one inner AXPY.

Runs via `cargo bench --bench schur_kernel`. The speedup numbers
for L = 256 are the Phase 2.4.2 microbench gate (≥ 2× on
`axpy_minus`; ≥ 1.5× on `do_1x1_update` end-to-end).

Commit message: "Phase 2.4.2 Step 4: schur_kernel microbenchmark".

### Step 5. Wire into factor.rs.

Replace the inner loops of `do_1x1_update` and `do_2x2_update`
with calls to `axpy_minus` / `axpy2_minus`. Use `split_at_mut` to
get the disjoint column slices without `unsafe`.

For very small inner lengths (`len < THRESHOLD`, where `THRESHOLD`
is calibrated in step 4 — expect somewhere around 8–16) fall
through to a scalar inlined loop. Rationale: pulp's dispatch +
tail mask setup has a fixed cost that only amortizes past a few
SIMD iterations.

Run all dense factor tests. They must pass byte-for-byte or within
1 ULP. Run one targeted new test that exercises a front large enough
to hit the SIMD path on both `do_1x1_update` and `do_2x2_update`
(n = 256 indefinite random).

Commit message: "Phase 2.4.2 Step 5: wire SIMD kernel into
do_1x1_update and do_2x2_update".

### Step 6. Full KKT bench + validation report.

Run `cargo run --bin bench --release`. Compare against Phase 2.1.8
baseline (`dev/sessions/phase-2-baseline.md`). Write
`dev/validation/phase-2.4.2-simd.md` with:

- Before/after dense + sparse factor p50/p90/p99/max vs MUMPS.
- Inertia match counts (must equal baseline byte-for-byte).
- Top-10 worst factor-ratio matrices before and after.
- Microbench numbers from step 4.
- Any inertia flips (expected: zero; if nonzero, investigate).

Commit message: "Phase 2.4.2 validation report: SIMD kernel
benchmark results".

### Step 7. (Optional, deferred) AVX-512 tuning.

pulp handles this automatically if `x86-v3` is enabled and the host
CPU has AVX-512. No code changes needed. Verify by running the
microbench on a Zen 4 or Ice Lake machine if one is available.

## Test plan (correctness)

Hard requirements — all must pass before step 5 is committed:

1. **Unit tests on `axpy_minus` and `axpy2_minus`** at the length
   sweep from step 2, delta ≤ 1 ULP against scalar reference.
2. **Existing dense factor tests** unchanged, all passing.
3. **One targeted integration test** that runs `factor_frontal` on
   a random 256×256 symmetric indefinite matrix and compares the
   resulting (L, D, inertia) against the scalar path saved in a
   separate `scalar_frontal_ref.rs` test helper. Delta ≤ 1 ULP.
4. **Full KKT bench** inertia and residual numbers match Phase
   2.1.8 baseline exactly (dense 152911/154481 match,
   sparse 153009/154588 match, residual caps ≤ 1.87e-4).

If any inertia count differs after step 5, stop and investigate.
Do NOT loosen tolerances to paper over differences.

## Risks

1. **FMA rounding flips an edge-case pivot.** The SIMD path may
   use FMA where the scalar path used separate mul+add, giving a
   1-ULP difference that tips a near-zero pivot across `zero_tol`.
   Detection: full KKT bench inertia diff. Mitigation: if it
   happens on matrices currently at the failure boundary, it is
   likely an *improvement*; investigate case-by-case and record.
2. **Short-vector overhead slows down small fronts.** The sparse
   bench is dominated by tiny fronts; if the SIMD path has a
   fixed ~50ns dispatch cost, it could slow them down. Mitigation:
   the length-threshold fallback in step 5, calibrated against the
   step 4 microbench.
3. **pulp 0.22.2 vs pulp 0.23+.** Pin exact version. Review
   changelog before any bump.
4. **aarch64 NEON yields less speedup than AVX2.** NEON is 128-bit
   (2 doubles) vs AVX2 256-bit (4 doubles), so the dev machine
   (Apple Silicon) will see roughly half the speedup of a
   Zen/Skylake box. This is fine as long as the microbench shows
   ≥ 2× on NEON and the full bench hits p90 ≤ 2.0; if it misses,
   the fix is on a different machine, not in feral code.

## Exit criterion

Phase 2.4.2 is done when:

1. Steps 1–6 all committed.
2. Microbench shows ≥ 2× speedup on `axpy_minus` at L = 256.
3. All dense factor tests pass (≤ 1 ULP delta vs scalar).
4. Full KKT bench inertia + residual exactly matches baseline.
5. Validation report committed at `dev/validation/phase-2.4.2-simd.md`.

Performance soft targets (if hit, Phase 2.8 exit criterion
satisfied for the dense path):

6. Dense factor p90 vs MUMPS ≤ 2.0 (currently 2.27).
7. Sparse factor p90 vs MUMPS ≤ 3.0 (currently 3.18).

If soft targets miss, the kernel still ships (it's strictly
faster) and Phase 2.5.x becomes the next lever.
