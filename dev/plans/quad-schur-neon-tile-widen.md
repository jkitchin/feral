# Plan — widen the quad Schur kernel's register tile on aarch64

Issue #44. Context: `dev/journal/2026-05-22-02.org`,
`dev/research/per-factor-cost-cluster-2026-05-21.md`,
`dev/research/faer-dense-speed-reference.md`.

## Problem

`schur_panel_minus_nofma_strided_quad` (`src/dense/schur_kernel.rs`) is
the dominant cost of NARX_CFy factorization (94.6% of the iter-0 numeric
loop, ~8 GFLOP/s achieved). Its bulk SIMD body uses **unroll factor 2**:
per chunk it holds 8 live accumulator registers (2 SIMD row-vectors × 4
trailing columns). The unroll factor was deliberately sized for AVX2's
16-ymm budget (8 acc + 2 src + 1 alpha splat ≈ 11 live, no spill).

aarch64 (Apple Silicon, the box this runs on) has **32** NEON registers.
The kernel is leaving register file unused: more independent accumulator
chains hide the dependent-`sub` latency better, raising IPC.

## Change

Make the bulk unroll factor a compile-time constant gated on the target
arch:

```rust
const UNROLL: usize = if cfg!(target_arch = "aarch64") { 4 } else { 2 };
```

- aarch64: unroll-4 → 16 accumulators (4 row-vecs × 4 cols) + 4 src
  vecs + 1 alpha splat ≈ 21 live regs, fits 32-NEON budget.
- x86 / everything else: unroll-2, unchanged.

Rewrite the bulk loop to carry the per-column accumulators in
`[S::f64s; UNROLL]` arrays instead of named scalars `a00..a31`. The
`for k in 0..UNROLL` inner loops fully unroll (UNROLL is `const`); LLVM
SROA scalarizes the small `Copy` arrays back into SSA values → registers.
For x86 (`UNROLL == 2`) this produces codegen equivalent to the current
explicit `a00,a01,...` bindings.

The existing unroll-1 tail loop (`tail_chunks_start..body_len`) and the
masked partial-load tail are unchanged; they now mop up 0..=3 leftover
body vectors instead of 0..=1.

## Bit-exactness

Trivially preserved. The unroll factor only changes how body vectors are
**grouped** into chunks; each body vector's accumulator still runs the
identical `q`-ascending `acc <- sub(acc, mul(alpha_q, src_q))` chain,
independent of every other vector and of the grouping. No per-element
accumulation is reordered.

Gate: the existing
`schur_panel_minus_nofma_strided_quad_is_bit_exact_vs_four_singles`
test — external oracle is four sequential single-column
`schur_panel_minus_nofma_strided` calls. Its `len0` sweep includes
non-multiples of 4 (5,6,7,9,10,15,17,18,19,31,33,63,65,127,257) so the
unroll-4 tail (`body_len % 4 ∈ {1,2,3}`) is exercised.

## Scope / risk

- No API change, no dispatch change in `apply_blocked_schur_panel`.
- Only the `nofma` quad kernel. (FMA quad is a regression on aarch64 —
  `probe_fma_kernel`, issue #35 — and is not on the hot path.)
- Expected gain: modest, ~1.2–1.5× on the quad kernel in isolation. Does
  not close the 4.4×-per-iter gap alone; this is the safe first step.

## Verification

1. `cargo test schur_panel_minus_nofma_strided_quad` — bit-exact + the
   zero-alpha-skip test.
2. `cargo run --release --bin probe_fma_kernel` — before/after nofma
   GFLOP/s on square_1928 / wide_2829 / narrow_512x32.
3. `cargo run --release --bin probe_narx_factor` and
   `diag_narx_kernel_gflops` — end-to-end NARX_CFy factor time.
4. Full `cargo test`, `cargo clippy --all-targets -- -D warnings`.
