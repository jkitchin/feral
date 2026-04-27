# Plan: Dense BK Kernel Speedup (close the 5x per-nnz gap to MUMPS)

Date: 2026-04-27.
Owner: dense BK kernel.
Status: PROPOSAL — pending user approval before any production code edit.
Inputs:
- `dev/research/feral-kernel-profile-chainwoo.md` (measured profile, 2026-04-27)
- `dev/research/mumps-small-frontal-speed.md` (Fortran reference, 2026-04-27)
- `dev/research/ssids-small-frontal-speed.md` (C++ reference, 2026-04-27)
- `dev/research/faer-dense-speed-reference.md` (Rust reference, 2026-04-27)

## 1. Problem statement (one paragraph)

On CHAINWOO_0000 (chain-like KKT, n=4000, max sym front 32) feral takes 24 ms
to factor; MUMPS takes 0.73 ms (33x faster) and SSIDS 3.6 ms (6.7x faster).
Per-nnz_L the gap is roughly 5x (feral 89 ns/nnz, SSIDS 29 ns/nnz, MUMPS 14
ns/nnz). The fill axis (5.4x more nnz_L than MUMPS) is being addressed by
ordering work; this plan addresses the kernel axis only.

The measured profile pinpoints the cost: 99.95% of factor time is in 64
supernodes (the ones with `ncol=32`); a single root supernode (snode 1933,
1984x32) is **62% of total factor time**. Inside, the dominant work is a
BLAS-1 rank-1 trailing update issued 32 times per pivot panel, each touching
a 30 MB trailing block end-to-end (`src/dense/factor.rs:1841 do_1x1_update`).
The blocked-panel kernel that *would* batch these into a deferred update
exists but never engages (gated by `ncol > block_size` with default
`block_size = 64`, so 32-col fronts always fall through to scalar).

## 2. Three convergent recommendations

The three reference solvers diverge on architecture but converge on
*technique*. All three confirm the same kernel-level ideas:

| technique                                | MUMPS                       | SSIDS                            | faer                          |
|------------------------------------------|-----------------------------|----------------------------------|-------------------------------|
| Deferred panel update (one flush)        | DMUMPS_FAC_SQ_LDLT (BLAS-3) | block_ldlt + apply_pivot_app     | lblt_blocked_step (gemm)      |
| Coarse SIMD dispatch (1 dispatch / O(n^2)) | autovec scalar Fortran       | hand-rolled SimdVec, 1 per body  | pulp::WithSimd, 1 per rank-k  |
| In-place factor (no per-front Vec churn) | S(LA) arena via POSELT      | BuddyAlloc + SmallLeafSubtree    | temp_mat_uninit on MemStack   |
| Argmax fused with prior update           | MAXFROMM in FAC_MQ_LDLT     | find_maxloc tournament           | update_and_offdiag_argmax     |
| Tiny fronts skip blocking entirely       | NBKJIB=NASS for NASS<24     | block_ldlt covers 32 fully       | n<=64 -> lblt_unblocked       |

Notable divergences:
- MUMPS at n=32 is *purely* scalar Fortran; the speed comes from S(LA) arena +
  Inextpiv + MAXFROMM fusion + autovectorization-friendly inner loops. This
  rules out "you need a SIMD kernel" as a sufficient condition.
- SSIDS uses a fully SIMD-vectorized monomorphized 32x32 in-register kernel
  with 4-way trailing-column unroll. SSIDS is faster than MUMPS per dispatch
  call but slower overall (because it pays threshold-pivot bookkeeping).
- faer matches SSIDS's per-call ratio at n=32 (one pulp dispatch covers the
  entire rank-1 update of 31 trailing columns) but uses BLAS-3 once n>=128.

The lesson: **kernel formulation matters more than SIMD width**. Even
unvectorized Fortran with the right loop structure beats vectorized Rust
with the wrong loop structure. The dispatch-per-column pattern at
`apply_blocked_schur` (factor.rs:1280) is the worst of both worlds.

## 3. Ranked work items

### W-1. Engage the existing blocked panel for `ncol <= block_size`  (~30 LoC + tests)

**Where**: `src/dense/factor.rs:949`.

**What**: The early-out
```rust
if bs < 2 || ncol <= bs {
    return factor_frontal(matrix, ncol, may_delay, params);
}
```
sends every CHAINWOO 32x32 supernode (and *every* 32-col front in the corpus)
to the scalar `factor_frontal`. Lower the threshold so the deferred-Schur
panel runs whenever `ncol >= 8`, with `bs = min(ncol, 64)`. The panel kernel
itself (`lblt_panel_frontal`) already handles the small-`bs` case; we just
need to widen the gate.

**Risk**: bit-exactness with scalar reference is enforced by
`tests/blocked_ldlt.rs`. Lowering the threshold expands the matrix surface
the test must cover — likely needs new fixtures at 8 <= ncol <= 64.

**Expected impact**: 1.5-1.8x on CHAINWOO factor time per the profile note's
QW-1 estimate. Multiplies with W-2.

**Validation**: re-run `diag_chainwoo_profile` and compare snode 1933
factor_one_supernode time before/after.

### W-2. Replace BLAS-1 trailing update with rank-`bs` accumulator  (~150 LoC + tests)

**Where**: `src/dense/factor.rs:1280-1305` (`apply_blocked_schur`).

**What**: Currently:
```rust
for (q, &d_q) in d_panel.iter().enumerate().take(n_elim) {
    ...
    for j in j_start..nrow {
        schur_kernel::axpy_minus_unroll4_nofma(dst, src, alpha);  // dispatch
    }
}
```
**O(n_elim * trailing) pulp dispatches per panel flush.** Replace with one
`pulp::WithSimd` impl whose body iterates `for j in j_start..nrow` outermost,
loads 4 column-batches of `dst`, accumulates *all* `n_elim` contributions in
register accumulators, then stores. This is the technique that all three
reference solvers use: faer via `triangular::matmul`, SSIDS via inlined
4-col-unroll inside `block_ldlt::update_1x1`, MUMPS via BLAS-3 DSYRK.

For our scope (CHAINWOO: ncol=32, trailing up to 1952), the kernel needs a
clear function signature like:
```rust
fn schur_panel_minus_simd(
    dst: &mut [f64],   // trailing block, column-major
    dst_lda: usize,
    src: &[f64],       // L panel, column-major (n_elim cols)
    src_lda: usize,
    d_panel: &[f64],   // n_elim diagonal entries (1x1) or pairs (2x2)
    n_elim: usize,
    nrow: usize,
    j_start: usize,
);
```

**Risk**: bit-exactness contract must hold. Current `axpy_minus_unroll4_nofma`
uses explicit `mul / add / sub` ordering for reproducibility (no FMA). The
new accumulator must do the same: accumulate into `f64` registers without
FMA, store back column-by-column. faer's `*_unroll4_nofma` already
demonstrates the pattern.

**Risk**: 2x2 pivots inside the panel currently update via separate code
paths inside `do_2x2_pivot`. The accumulator must accept either a `&[f64]`
diagonal or a `&[(f64, f64, f64)]` 2x2-block array. Could ship 1x1-only
first and keep 2x2 on the slow path; the profile shows 2x2 pivots are a
small fraction on CHAINWOO.

**Expected impact**: 2-3x on CHAINWOO total factor time (profile note QW-2).
Combined with W-1: 3-5x total.

**Validation**: `tests/blocked_ldlt.rs` bit-exactness suite + new fixtures
covering `n_elim in {1,2,4,8,16,32}` and `trailing in {0..2*nrow}`. Then
re-run `diag_chainwoo_profile`.

### W-3. Pool per-frontal `Vec`s in `factor_frontal_blocked`  (~80 LoC)

**Where**: `src/dense/factor.rs:903`-`:954`.

**What**: Every call to `factor_frontal_blocked` allocates fresh `Vec`s for
`a` (nrow x nrow), `subdiag`, `perm`, `perm_inv`, `l`, `d_diag`, `contrib`.
The `contrib` vec for the 1984 root front is 30 MB — pure malloc traffic
on the hot path. Add a `DenseKernelWorkspace` borrowed from
`FactorWorkspace`, reuse buffers across supernodes, gate the
`SymmetricMatrix::validate()` call (`src/dense/matrix.rs:69`) behind
`debug_assertions` since the data was already value-checked at CSC parse.

**Risk**: low. Mechanical refactor. The buffers must be cleared / resized
per call but `Vec::resize` reuses the existing allocation when capacity is
sufficient.

**Expected impact**: 5-10% on CHAINWOO total time per the profile note's
QW-3 estimate. Bigger on IPM-style hot loops where the same matrix shape
repeats hundreds of times.

**Validation**: bench corpus end-to-end, check no allocator-pressure
regression on tiny-front matrices, confirm CHAINWOO total factor drops.

### W-4. (deferred) Fused argmax-with-update for panel pivot search

Out of scope for the first pass. The profile shows pivot search is ~10% of
inner loop time on the wide-front CHAINWOO root; W-1+W-2+W-3 should already
close most of the gap. Revisit if pivot search becomes the new bottleneck
post-W-2. The clean-room reference is faer's
`update_and_offdiag_argmax` (factor.rs:461).

## 4. Implementation order

1. **Land W-3 first** (lowest risk, smallest LoC, no algorithmic change). It
   sets up the workspace plumbing W-1 and W-2 will reuse and gives an
   immediate measurable win on the bench.
2. **Then W-1** (engage the existing panel). Risk: bit-exactness fixtures
   for new `(bs, ncol)` combinations.
3. **Then W-2** (rewrite the trailing accumulator). Highest risk + highest
   payoff. Land 1x1-only first; 2x2 panel updates can stay on the fallback
   path until bit-exact 2x2 accumulator lands.

Each work item is a separate commit. After each, re-run:
- `cargo test --lib` (162 tests)
- `tests/blocked_ldlt.rs` bit-exact parity
- `cargo run --release --bin diag_chainwoo_profile` (single-matrix sanity)
- `cargo run --release --bin bench` (corpus regression check)

## 5. Out-of-scope (intentionally)

- **Adopting faer as a dependency**: ruled out by the clean-room constraint
  and dependency-minimalism intent. We adopt techniques only.
  See `dev/research/faer-dense-speed-reference.md` Section 6.1.
- **Symbolic frontal-width fix** (the actual_nrow=1984 problem): tracked
  separately with the ORBIT2 dense-column quotient work
  (`dev/research/orbit2-cluster-regression.md`).
- **Small-leaf-subtree batching** (SSIDS technique): substantial driver
  rewrite. Defer until W-1+W-2+W-3 is measured.
- **Threading**: MUMPS gates OMP at NCB1 > 300 — for CHAINWOO max actual
  trailing = 1952 this would engage, but feral has no rayon dep in the
  numeric kernel and the pulp kernels are already vectorized. Defer until
  single-thread is competitive.

## 6. Acceptance criteria

End-to-end target on CHAINWOO_0000 after W-1 + W-2 + W-3:
- `feral factor_us < 8 ms` (3x speedup from current 24 ms).
- Stretch: `< 4 ms` (6x), bringing per-nnz cost from 89 ns to ~14 ns and
  closing the gap to MUMPS at its current fill ratio.
- All existing bit-exact tests pass.
- No regression > 5% on any matrix in the bench corpus.

After ordering work also lands, the per-nnz path should converge with
SSIDS (~29 ns/nnz). Beating MUMPS on absolute time will require both axes
(fill + kernel) at parity.
