# Issue #57 fix #2 — BLAS-3 panel kernels for the multi-RHS sparse solve

**Date:** 2026-05-30
**Issue:** #57 fix #2 — "BLAS-3 `dtrsm`/`dgemm` on each supernode's
frontal panel"
**Builds on:** `dev/research/issue-57-multirhs-row-major.md` (fix #1,
the row-major working-buffer layout, commit 80348f9) and
`dev/research/multi-rhs.md` (F1.0 API + D3 kernel-dispatch decision).
**Goal:** Reach the 5–10× per-RHS regime (issue #57 "Measured impact")
for large `nrhs` by replacing the rank-1 cascade inside each supernode
with a register-blocked dense panel solve (TRSM + GEMM), without
loosening any tolerance or changing the public column-major contract.

## Why fix #1 is not enough

Fix #1 made the inner `c`-loop contiguous, so it auto-vectorizes. But
the kernel is still a **sequence of rank-1 updates**: in the forward
trailing update each `w[i, :]` (length `nrhs`) is read-modified-written
once per eliminated column `j`, i.e. `nelim` times. Arithmetic
intensity is ~1 FMA per 2 memory ops — memory-bound. The bench
(`bench_multirhs`) confirms only ~1.0–1.3× per-RHS amortization, far
from the 5–10× a dense GETRS reaches on the same panels.

The win requires keeping the `c`-accumulator in registers across the
`j`-reduction so each output `w[i, c]` is written **once**, and reusing
the `w[j, :]` panel rows across several output rows. That is a
register-blocked GEMM microkernel — the textbook supernodal multifrontal
multi-RHS optimization the issue calls out.

## Layout (unchanged from fix #1)

- `ff.l`: column-major, leading dim `nrow`. `L[i,j] = ff.l[j*nrow + i]`,
  `i ∈ 0..nrow`, `j ∈ 0..nelim`. Unit lower-trapezoidal (the forward
  loop never touches the diagonal → implicit unit diagonal). The
  `D` factor is applied separately in phase 2.
- `w`: row-major `nrow × nrhs`, `w[i*nrhs + c]`. Rows `0..nelim` are the
  eliminated block (`w_top`); rows `nelim..nrow` are the trailing /
  separator block (`w_bot`).
- The column-major `y`/`rhs`/`x` contract and the gather/scatter
  transpose loops are untouched (fix #1's invariant). Only the in-panel
  solve kernels change.

## Decomposition

Write the panel as the 2×1 block
```
L = [ L_11 ]   L_11 = nelim × nelim, unit lower triangular
    [ L_21 ]   L_21 = (nrow - nelim) × nelim
```

**Forward (phase 1), `w := L^{-1} w`:**
1. TRSM: forward-substitute `L_11` against `w_top` (sequential in `j`,
   updates only panel rows `i ∈ (j+1)..nelim`). Small (`nelim²`); keep
   the existing per-`c` vectorized rank-1 loop.
2. GEMM: `w_bot -= L_21 @ w_top`. Big block; register-blocked microkernel.

**Back (phase 3), `w := L^{-T} w`:**
1. GEMM: `w_top -= L_21^T @ w_bot` (the trailing contribution to every
   panel column `j`; no intra-panel dependency since `w_bot` is already
   finalized by descendant nodes). Register-blocked microkernel.
2. TRSM: back-substitute `L_11^T` against `w_top` (`j` from `nelim-1`
   down to `0`, updates only panel rows). Keep the existing per-`c`
   `acc` loop.

The D-block solve (phase 2) is unchanged — it stays per-column (the
1×1/2×2 logic is value-dependent and cheap).

## Microkernel (MR × NR register tile)

`C[mb × nrhs] -= A[mb × nelim] @ B[nelim × nrhs]`, with `A` column-major
(leading dim `nrow`, row offset `arow_off`), `B`/`C` row-major
(leading dim `nrhs`). `MR = 4`, `NR = 8`:

```
for i0 in (0..mb).step(MR):
  for c0 in (0..nrhs).step(NR):
    acc[t][s] = C[i0+t, c0+s]            // load MR×NR tile (init = current C)
    for j in 0..nelim:
      let b[s] = B[j, c0+s]              // NR loads, reused across MR rows
      for t in 0..MR:
        let a = A[i0+t, j]               // = ff.l[j*nrow + arow_off + i0+t]
        for s in 0..NR: acc[t][s] -= a * b[s]
    C[i0+t, c0+s] = acc[t][s]            // store once
  // c-tail: NR=1 ; i-tail (mb % MR): MR=1
```

`acc` is a fixed `[[f64; NR]; MR]` stack array → LLVM keeps it in
registers and vectorizes the `s`-loop (NR = 8 = one or two f64 SIMD
vectors on x86-v3 / aarch64). Each `C[i,c]` is written once; each
`B[j,c]` is loaded once per MR rows; each `A[i,j]` once per NR cols.

## Bit-exactness analysis

The single-RHS path `solve_sparse_core_into` (external, pre-existing) is
the oracle. Per output element, the cascade computes a **left fold**
`w[i] = b[i] - L[i,0]w[0] - L[i,1]w[1] - …` over increasing `j` (forward)
/ increasing `i` (back).

- **Forward GEMM is bit-identical.** The microkernel inits the
  accumulator to `C[i,c]` (= `b[i,c]`) and subtracts `A[i,j]·B[j,c]` in
  increasing `j` — the same left fold as the cascade. The TRSM panel is
  the existing loop (unchanged order). So the whole forward solve is
  bit-for-bit identical to looping single-RHS.
- **Back solve is equal to ~1e-15, not bit-identical.** The cascade
  accumulates, per `j`, the panel rows (`i ∈ j+1..nelim`) *then* the
  trailing rows (`i ∈ nelim..nrow`). The split does GEMM (trailing)
  first, then TRSM (panel) — a reordered sum. The result differs only by
  floating-point reassociation (~`κ·eps`), well inside the `1e-12`
  parity tolerance on well-conditioned panels. **No tolerance is
  loosened**; the existing `1e-12` parity assertions are reused.

## Threshold dispatch (de-risk the IPM hot path)

Keep the fix-#1 row-major kernels for `nrhs < BLAS3_NRHS_THRESHOLD`
(= 32) and route `nrhs ≥ 32` through the BLAS-3 kernels. Rationale:

- The IPM predictor/corrector hot path uses small `nrhs` (1–few) and
  routes through `solve_sparse_core_into` anyway; the small-`nrhs` many
  path stays **bit-identical** (existing `nrhs ≤ 17` parity tests
  unchanged, `max|many−single| = 0`).
- D3 (`multi-rhs.md`) put the BLAS-3 crossover at `k ≈ 16`; 32 is a
  conservative crossover that guarantees the microkernel's fixed setup
  (tile loop overhead) is amortized. The pounce `jax.jacrev` workload
  is `nrhs = n` (hundreds–thousands), far above 32.
- Two paths means the new kernels are isolated; a regression cannot
  touch the single-RHS or small-`nrhs` paths.

## Oracle & acceptance (satisfies the CLAUDE.md "external oracle" rule)

Oracle = the pre-existing independent single-RHS `solve_sparse`.
`solve_sparse_many` (which now dispatches to BLAS-3 at `nrhs ≥ 32`) must
equal `k` independent `solve_sparse` calls per entry.

1. New parity tests at `nrhs ∈ {32, 37, 64}` on the 2-D Laplacian
   (`n = 144`, multiple supernodes, `nrow/nelim` and `nrhs` chosen to hit
   the MR/NR tails) assert `max|many − single| < 1e-12` (tolerance NOT
   loosened — same constant as fix #1).
2. Existing `nrhs ≤ 17` parity tests stay green and bit-identical
   (below threshold).
3. `cargo test --lib` ≥ 317 pass; clippy `-D warnings` clean; fmt clean;
   no `unwrap`/`expect`/`unsafe` in `src/`.
4. `bench_multirhs` per-RHS batched/looped ratio at `nrhs ∈ {64, 256}`:
   report measured numbers and state plainly whether the 5–10× target
   (issue #57) or the F1.2 ≤ 0.75× per-column target is met.

## Risk

- **Indexing slips** in the tiled GEMM (the one real risk). Mitigated by
  the MR/NR-tail test cases and the tight external oracle.
- **No `unsafe`.** Stack `[[f64; NR]; MR]` accumulators and slice
  indexing only; the borrow split between `w_top` (read) and `w_bot`
  (write) is a `split_at_mut` at `nelim*nrhs`.
- **Back-sub non-bit-identical.** Documented above; bounded by `1e-12`
  parity tests. Forward stays bit-identical.

## Results (2026-05-30)

Implemented as designed (MR=4, NR=8, threshold 32). Parity: all
multi-RHS tests green, `max|many − single| ≤ 1.6e-15` at `nrhs ∈
{31, 32, 37, 64}` — far inside the 1e-12 gate, tolerance untouched.

**First measurement was disappointing and exposed two bigger wins than
the GEMM itself:**

1. The naive BLAS-3 (GEMM only) gave ~3× on n=484/2025 but *regressed*
   n=1024 to 1.0–1.2× (slower than looping). Root cause was **not** the
   GEMM: the per-supernode gather/scatter read the column-major `y`
   with stride `n`, and at n=1024 (power of two) consecutive RHS columns
   aliased into the same cache sets — conflict-miss storms. That stride-
   `n` transpose ran 3× per supernode, capping every size.

2. **Fix: flip the internal `y` to row-major** so gather/scatter become
   contiguous memcpys (the only stride-`n` access moves to the one-time
   entry/exit permute). This killed the n=1024 regression (1.2× →
   0.33×) and ~halved wide-solve time everywhere. It benefits the
   rank-1 path too and is bit-identical (same values, different order).

3. **Fuse forward-sub + D-solve into one postorder pass** (a node's
   eliminated rows are final after its own forward-sub; ancestors only
   touch its separator rows), removing one of the three gather/scatter
   rounds. Modest additional gain.

Final per-RHS batched/looped ratios (`bench_multirhs`, idle machine,
2-D Laplacians, `nrhs ∈ {64, 256}`):

| n    | ratio        | speedup |
|------|--------------|---------|
| 484  | 0.18–0.24    | ~4–5×   |
| 1024 | 0.32–0.34    | ~3×     |
| 2025 | 0.17–0.23    | ~5–6×   |

The issue's 5–10× target (set against a *dense* GETRS reference) is
reached at n=2025 and approached elsewhere; the residual gap is the
irreducible sparse overhead a dense GETRS does not pay — tree
traversal, the (now contiguous) gather/scatter, the D-solve, and the
entry/exit permute. n=1024 stays lowest because its power-of-two front
dimensions still strain the strided column-major `L` access inside the
GEMM; packing `L` into a contiguous panel (BLIS-style) is the next
lever if more is needed, deferred until a workload demands it.
