# Phase 2.4.3 — Block-32 Register-Resident LDLᵀ Kernel

**Status:** Pre-implementation plan
**Date:** 2026-05-12
**Issue:** #9
**Research note:** `dev/research/block32-register-resident-kernel.md`
**Inherits from:** 2026-04-14 decision (mul + sub, never FMA)

## Goal

Close the ~3× per-nnz throughput gap vs SSIDS on `CHAINWOO_0000` by
adding a monomorphized 32×32 LDLᵀ kernel that (a) matches feral's
existing Bunch-Kaufman pivot semantics bit-for-bit and (b) packs four
trailing destination columns per source-vector load in a register-
resident inner loop.

Acceptance, restated from issue #9:

1. Bit-parity unit test: every `f64` in `(L, D, perm, subdiag, contrib)`
   is `to_bits()`-equal between scalar and block-32 paths on
   hand-crafted and randomized 32×32 indefinite matrices. NEON, AVX2,
   AVX-512.
2. Corpus parity: `parallel_corpus_parity` shows 0-delta on inertia
   and residual vs. the pre-kernel baseline.
3. Throughput: `≤ 44 ns/nnz_L` on `CHAINWOO_0000` (1.5× SSIDS's 29 ns/nnz).
   Measured by `src/bin/diag_chainwoo_profile.rs` (the issue named this
   `feral-kernel-profile-chainwoo`; we keep the existing name).

## Non-goals

- Pivot-search SIMD (out of scope per issue; folded into APP sub-issue).
- BLOCK_SIZE ≠ 32 (e.g., 64). Other sizes can land later.
- New pivot strategy (rook, APP). Strategy stays standard BK; only
  the kernel changes.
- Multi-front batching / `SmallLeafNumericSubtree` analogue (separate
  follow-up; gated on Gap A being closed first).

## Architecture

Three new things, in dependency order:

1. **`update_1x1_block32`** in `src/dense/block_ldlt32.rs`.
   - One pulp `WithSimd` dispatch wraps the whole trailing update.
   - Inside: 4-way unroll over destination columns `c`, with one
     source-vector load per `(c, r)` tile shared across the 4 dest
     columns. Per-lane op order: `prod = mul_f64s(work[c], src); avec = sub_f64s(avec, prod)`.
   - Falls back to per-column axpy for `nrow - p` < 4 trailing columns.
2. **`update_2x2_block32`** in the same file.
   - Same shape as above, but two source columns. Per-lane:
     `t0 = mul(work0[c], s0); a = sub(a, t0); t1 = mul(work1[c], s1); a = sub(a, t1)`.
3. **`block_ldlt32` driver** in the same file.
   - BLOCK_SIZE = 32 const generic.
   - BK pivot rules ported line-for-line from `lblt_panel_frontal`:
     gamma0 = column max-loc, alpha threshold, swap-1×1 / no-swap 2×2
     / swap-2×2 / rejection / delayed-pivot branches.
   - Eager trailing update inside the driver (calls
     `update_1x1_block32` / `update_2x2_block32` per pivot). No
     peek-ahead, no deferred-Schur — that simplifies the
     scalar-fallback handoff state.
   - Returns `(n_elim, PanelStatus)` identical to `lblt_panel_frontal`.

Dispatch site: `factor_frontal_blocked_in_place`
(`src/dense/factor.rs:1071`). New early branch:

    if nrow == 32 && ncol == 32 && nrow * 8 % vlen == 0 {
        return block_ldlt32(...);
    }

For everything else (small fronts, large fronts, partial fronts) we
keep the existing `lblt_panel_frontal` path. The block-32 entry is a
fast-path opt-in; the scalar oracle is preserved.

## Step-by-step

### Step 1 — TDD scaffolding (this session)

a. Create `src/dense/block_ldlt32.rs` with module skeleton:
   - Stub `update_1x1_block32` that delegates to `axpy_minus_unroll4_nofma`
     per column — correct but slow. (Proof of bit-parity at the unit-test
     level before introducing the 4-wide SIMD body.)
   - Stub `update_2x2_block32` that delegates to `axpy2_minus_unroll4_nofma`
     per column.
   - Stub `block_ldlt32` that returns `PanelStatus::ScalarFallback` with
     `n_elim = 0` (so the dispatch fallback is exercised by the
     production corpus once wired).
b. Add `mod block_ldlt32;` in `src/dense/mod.rs`.
c. Write the bit-parity tests at the `update_*` level. These must pass
   with the stub already (since stub delegates to the bit-exact axpy).
d. Write the bit-parity tests at the full-block level. These will
   only pass once Step 2 lands; mark `#[ignore]` until then with a
   comment pointing to the plan.
e. Commit. Tests pass; production behavior unchanged.

### Step 2 — Block-level driver (separate session)

a. Port `lblt_panel_frontal` body into `block_ldlt32` with BLOCK_SIZE
   baked in. Same scalar BK rules; same swap/reject branches.
b. Remove `#[ignore]` from block-level parity tests. They must pass
   on hand-crafted 32×32 matrices first, then on randomized.
c. Wire dispatch in `factor_frontal_blocked_in_place`. Run
   `parallel_corpus_parity` — must show 0-delta vs. pre-kernel head.
d. Commit.

### Step 3 — SIMD body for `update_1x1_block32` (separate session)

a. Replace stub `update_1x1_block32` with one-pulp-dispatch body that
   packs 4 destination columns per source load. Inner-loop op order
   bit-exact-matches the per-column `mul(work[c], src); sub(a, prod)`
   chain.
b. Run the block-level parity tests again; must still pass byte-for-byte.
c. Run `parallel_corpus_parity` again; must still be 0-delta.
d. Run `diag_chainwoo_profile`; record ns/nnz before and after.
e. Commit.

### Step 4 — SIMD body for `update_2x2_block32` (separate session)

a. Same pattern as Step 3, two source loads instead of one.
b. Parity tests; corpus parity; chainwoo profile; commit.

### Step 5 — Cross-arch CI gate (separate session, possibly separate
issue)

a. Add a GitHub Actions matrix job that runs the parity tests under
   `RUSTFLAGS='-C target-feature=+avx2'` and (separately) `+avx512f`
   on an x86 runner. NEON parity is already covered on the dev box;
   the CI gate proves the no-FMA contract holds across ISAs.

## Files touched (Step 1 only)

- `src/dense/block_ldlt32.rs` — new file.
- `src/dense/mod.rs` — add `mod block_ldlt32;`.
- `dev/research/block32-register-resident-kernel.md` — already
  committed.
- `dev/plans/phase-2.4.3-block32-kernel.md` — this file.

## Bit-parity oracle

The scalar reference for the parity tests is `factor_frontal` (not
`factor_frontal_blocked`). Rationale: `factor_frontal` runs the
unblocked BK loop with no peek-ahead, no deferred Schur — its rounding
chain at every column is `axpy_minus_unroll4_nofma` applied to the
ground-truth trailing state. That is exactly the rounding chain the
block-32 kernel reproduces with the eager update inside the driver.

Test harness:

```rust
fn assert_block32_bit_parity(a_lower: &[f64; 32 * 33 / 2]) {
    let mut a_scalar = expand_to_full(a_lower);
    let mut a_block  = a_scalar.clone();

    let r_scalar = factor_frontal(&mut a_scalar, /* ncol */ 32, false, &params);
    let r_block  = block_ldlt32(&mut a_block,    /* ncol */ 32, false, &params);

    for i in 0..r_scalar.l.len() {
        assert_eq!(r_scalar.l[i].to_bits(), r_block.l[i].to_bits(),
                   "L[{i}] mismatch");
    }
    for i in 0..r_scalar.d_diag.len() {
        assert_eq!(r_scalar.d_diag[i].to_bits(), r_block.d_diag[i].to_bits());
    }
    for i in 0..r_scalar.d_subdiag.len() {
        assert_eq!(r_scalar.d_subdiag[i].to_bits(),
                   r_block.d_subdiag[i].to_bits());
    }
    assert_eq!(r_scalar.perm, r_block.perm);
    assert_eq!(r_scalar.inertia, r_block.inertia);
    // contrib block — for ncol == nrow it is empty; skip.
}
```

## Risks recap

The ones from the research note that matter for this plan:

1. **Lane-width parity** — handled by the same per-lane equivalence
   argument that already covers `axpy_minus_unroll4_nofma`. Test on
   NEON locally; CI matrix is Step 5.
2. **Pivot-semantic drift** — the block-32 driver must port feral's BK
   rules verbatim, including the no-swap 2×2 inline fast path and the
   panel-internal rejection criteria. Risk mitigated by porting line-
   by-line from `lblt_panel_frontal` (not rewriting from scratch).
3. **Eager-update vs deferred-update rounding** — by inheriting
   `axpy_minus_unroll4_nofma`'s per-lane op order in the SIMD body,
   the eager block-32 update reproduces scalar's eager-update rounding
   chain byte-for-byte. No tolerance.
