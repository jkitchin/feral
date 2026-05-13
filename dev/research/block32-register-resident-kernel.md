# Block-32 Register-Resident LDLᵀ Kernel — Research Note

**Date:** 2026-05-12
**Issue:** #9 — Phase 2.4.3: 32×32 in-register SIMD kernel with bit-exact rounding (mul + sub, not FMA)
**Status:** Pre-implementation research (mandatory per CLAUDE.md before any code)

## 1. Problem statement

Feral's per-nnz throughput on small KKT fronts is ~3× worse than SSIDS and
~6× worse than MUMPS on `CHAINWOO_0000` (89 ns/nnz vs 29 ns/nnz vs
14 ns/nnz; see `dev/research/ssids-small-frontal-speed.md` §0). The
research note attributes this to three concrete gaps:

- **Gap A:** No fixed-block monomorphized kernel for the 32×32 front size
  that dominates KKT chains. Feral runs through the generic
  `factor_frontal_blocked` → `lblt_panel_frontal` driver, which carries
  runtime `bs`, `peek_ahead_column` replay, `PivotOutcome` matches, and
  `is_aligned` branching per pivot.
- **Gap C:** Schur update is per-pivot, not per-block. `apply_blocked_schur`
  (`src/dense/factor.rs:1280`) issues one `axpy_minus_unroll4_nofma`
  call per `(pivot, trailing-column)` pair, routing each through
  `dispatch_nofma` (`schur_kernel.rs:520`). For a 32×32 front this is
  ~960 pulp dispatch calls vs SSIDS's ~240 inlined FMA bodies that pack
  four trailing columns per source-vector load.
- Pivot-search SIMD: out of scope per issue (folded into the APP
  sub-issue).

This note studies the SSIDS reference, identifies what feral can adopt,
and pins down the constraints the new kernel must respect.

## 2. SSIDS `block_ldlt` reference (`ref/spral/src/ssids/cpu/kernels/block_ldlt.hxx`)

### 2.1 Driver shape (`block_ldlt`, lines 289–414)

Templated on `<T, BLOCK_SIZE>` so the compiler monomorphizes the entire
factorization. The driver loop:

    for(int p = from; p < BLOCK_SIZE; ) {
       find_maxloc<T, BLOCK_SIZE>(p, a, lda, bestv, t, m);
       /* small/singular fall-through */
       /* compute pivsiz {0,1,2} from a11,a21,a22 + test_2x2 */
       if(pivsiz == 1) { swap_cols; divide-through copy; update_1x1; }
       else            { swap_cols×2; 2×2 inverse; update_2x2; }
       p += pivsiz;
    }

Two structural points:

1. **Two-dimensional max-loc.** SSIDS searches the *entire* uneliminated
   block (`t`, `m` are both indices into `[p, BLOCK_SIZE)`). The pivot
   pair `(a[m,m], a[t,t], a[m,t])` is decided from that global max-loc.
2. **Divide-through-and-copy.** Before the trailing update, the pivot
   column is divided by `d11` (or 2×2 inverse) in place and the *original*
   values are copied into a stride-`BLOCK_SIZE` workspace `work`. The
   update reads from `work`, not from the now-divided column, which lets
   the update `avec[r] -= ld[c] * lvec[r]` use unmodified `lvec`.

### 2.2 `update_1x1` (lines 233–282)

The hot kernel. Two loops in sequence:

    // (a) unaligned head: c walks 1 column at a time
    for c in p+1 .. roundup_unroll(p+1):
        ldvec = -ld[c]
        for r in vlen_floor(c) .. BLOCK_SIZE step vlen:
            avec ← load_aligned(&a[c*lda+r])
            lvec ← load_aligned(&a[p*lda+r])    // SOURCE column
            avec = fmadd(avec, lvec, ldvec)     // avec += lvec * ldvec
            store_aligned(&a[c*lda+r], avec)

    // (b) unrolled body: c walks 4 columns at a time, one source load
    for c in roundup_unroll .. BLOCK_SIZE step 4:
        ldvec0..3 = -ld[c+0..3]
        for r in vlen_floor(c) .. BLOCK_SIZE step vlen:
            lvec ← load_aligned(&a[p*lda+r])    // ONE source load
            avec0..3 ← load_aligned(&a[(c+0..3)*lda+r])
            avec0..3 = fmadd(avec0..3, lvec, ldvec0..3)   // 4 FMAs
            store_aligned

The "register-resident" property: within `update_1x1` the trailing
columns `a[c..c+3, r..r+vlen]` are loaded once per (c, r) tile, four
destination columns are updated in registers, and only then stored. The
source column `a[p, r..r+vlen]` is loaded once per `(c, r)` tile.

### 2.3 `update_2x2` (lines 220–229)

Plain `#pragma omp simd` 2×2 update. Less aggressive because 2×2 pivots
are rare on chain-like matrices. Same divide-through-and-copy structure
into `work[0..BLOCK_SIZE-1]` and `work[BLOCK_SIZE..2*BLOCK_SIZE-1]`.

### 2.4 `find_maxloc` (lines 77–206)

Hand-rolled SIMD max-reduction over the upper-triangle of the
uneliminated block. Explicitly out of scope here — folded into the APP
sub-issue.

## 3. Semantic mismatch: SSIDS BK ≠ feral BK

SSIDS `block_ldlt` is the inner kernel for SSIDS's APP-aggressive
strategy, where the column-relative threshold test is *deferred* and a
post-hoc per-block check decides which pivots to accept. The pivot rule
used inside `block_ldlt` is therefore the simple two-dimensional max-loc
+ `test_2x2` sketched above.

Feral's existing dense kernels (`scalar_pivot_step`,
`lblt_panel_frontal`) implement standard Bunch-Kaufman with column-
relative gamma0 search:

- `gamma0 = max_{i > col} |a[col, i]|`
- `akk = |a[col, col]|`
- If `akk >= alpha * gamma0` → 1×1 pivot at `(col, col)`
- Else → try swap-1×1 at row `r` (argmax of column `col`),
  LAPACK-extension 1×1, swap-2×2; reject if none clear the threshold.

**The new kernel must match feral's BK rule, not SSIDS's.**
Otherwise the kernel produces a different factorization and the
bit-parity contract with `scalar_pivot_step` (the contract that
`parallel_corpus_parity` enforces) is broken.

Concretely:

- The maxloc is over the current pivot column, not over the entire
  uneliminated block.
- The 2×2 acceptance path follows `lblt_panel_frontal`'s "no-swap 2×2"
  / "swap-2×2" branches (`src/dense/factor.rs:1442–1500`).
- Rejection at column `col` returns control to the scalar driver
  with `nelim = col - k` and `status = ScalarFallback`. Feral's
  `parallel_corpus_parity` harness depends on this contract.

So the analogue to `block_ldlt` is *not* a direct translation. It is a
monomorphized BLOCK_SIZE=32 version of feral's existing panel logic,
with `update_1x1` and `update_2x2` swapped in for the
`apply_blocked_schur` axpy loop.

## 4. Bit-exact rounding constraint (2026-04-14 decision)

The decision entry at `dev/decisions.md:464–547` is binding:

> The production `do_1x1_update` / `do_2x2_update` hot-path wiring uses
> `axpy_minus_unroll4_nofma` / `axpy2_minus_unroll4_nofma`, the
> 4-way-unrolled pulp kernels whose inner body issues separate
> `simd.mul_f64s` + `simd.sub_f64s` instead of a fused
> `simd.mul_add_f64s`.

Cause: FMA performs one rounding; scalar `a -= alpha * s` performs two
(`round(alpha·s)` then `round(a − that)`). Bit-parity with scalar
requires reproducing the two-rounding chain in SIMD. The 4 inertia
regressions on ACOPP14_0001, ACOPP30_0004, FBRAIN3LS_0848,
FBRAIN3LS_0851 and the 26 residual regressions are the cost of getting
this wrong.

**The block-32 kernel inherits this discipline.** The SSIDS reference
uses `fmadd`; the feral analogue must use `mul_f64s` + `sub_f64s` (or
`mul_f64s` + `sub_f64s` interleaved across the four destination
columns, preserving the per-column rounding chain).

The bit-parity test already in `schur_kernel.rs:1420` (and the rank-2
variant at `:1683`) covers the axpy primitives. The new kernel needs an
analogous test at the block-factorization level: factor the same 32×32
matrix scalar vs block-32-kernel, assert
`f64::to_bits()`-equality on every entry of L, D, perm, subdiag, and
the contribution block.

## 5. Register residency in pulp (vs SSIDS C++)

SSIDS gets register residency by inlining `update_1x1` into the driver
loop, with template monomorphization keeping `BLOCK_SIZE` constant.
LLVM/GCC then schedule loads/stores so the inner tile keeps `avec0..3`
and `ldvec0..3` in vector registers across the `r` loop, and the
register file is large enough on AVX-512 (32 zmm) to hold the 4
destination columns + 1 source + 4 scaled negations + spare.

In Rust + pulp the corresponding pattern is:

1. **Single `WithSimd` dispatch per factorization step.** A `pulp` call
   with `Arch::new().dispatch(closure)` resolves the ISA token once;
   inside the closure, SIMD ops use that token directly with no
   per-call dispatch. The 4-way unroll inside the closure keeps `avec0..3`
   in registers under LLVM's register allocator.
2. **Constants via `const generic`.** A `block_ldlt32<const BS: usize>`
   would let the compiler unroll the `r` loop fully for `BS = 32`,
   matching SSIDS's `template<int BLOCK_SIZE>`. Concretely `BS / vlen` is
   2 on AVX-512 (vlen=8), 4 on AVX2 (vlen=4), 4 on NEON (vlen=2 ×
   stride). All small constants.
3. **Stride 32 means no tail handling.** With BLOCK_SIZE=32 and any of
   {2, 4, 8}-lane f64 SIMD, `BLOCK_SIZE % vlen == 0` always. The
   "unaligned head" loop SSIDS needs (lines 250–258) is unnecessary
   here — the kernel either runs the unrolled body or falls back to
   scalar; there is no in-between.
4. **One pulp dispatch per pivot.** Doing the whole factorization
   inside one `WithSimd::with_simd` is *possible* but mixes pivot logic
   and SIMD-ops in a single body. Cleaner: one dispatch per
   `update_1x1` / `update_2x2` call, matching how
   `axpy_minus_unroll4_nofma` already wraps a closure. Per-call
   dispatch overhead is then ~30 cycles (the dispatch is just a
   pre-selected function-pointer call), which is amortized over the
   ~30 trailing-column update. Versus the current ~30 dispatches per
   pivot from the per-axpy loop — a 30× reduction in dispatch overhead
   alone.

Concrete API sketch (subject to revision in the plan note):

```rust
// In src/dense/block_ldlt32.rs.

/// Trailing update for one 1×1 pivot at column `p`, BLOCK_SIZE = 32.
/// `a` is the column-major 32×32 block (lda = 32, aligned to vlen).
/// `work` holds the original pivot column `a[p, p+1..32]` for the
/// duration of this call. After this call,
/// `a[c, r] = a[c, r] - work[c] * a[p, r]` for all c in p+1..32,
/// r in c..32. Lane-equivalent to the scalar two-rounding chain.
fn update_1x1_block32(p: usize, a: &mut [f64], work: &[f64]);

/// Symmetric for 2×2 pivots.
fn update_2x2_block32(p: usize, a: &mut [f64], work0: &[f64], work1: &[f64]);

/// Full BLOCK_SIZE=32 BK factorization. Returns `n_elim` and
/// (Full | ScalarFallback | Delayed). Bit-identical L/D/perm/subdiag
/// to `lblt_panel_frontal` on the same input.
fn block_ldlt32(
    a: &mut [f64], lda: usize, ncol: usize,
    may_delay: bool, params: &BunchKaufmanParams,
    pos: &mut usize, neg: &mut usize, zero: &mut usize,
    needs_refinement: &mut bool,
    d_panel: &mut [f64; 32], subdiag: &mut [f64; 32], perm: &mut [usize; 32],
) -> Result<(usize, PanelStatus), FeralError>;
```

`update_1x1_block32` and `update_2x2_block32` are the SIMD-resident
hot paths. `block_ldlt32` is the driver — it owns the BK pivot logic,
swap_cols, divide-through, peek-ahead, and rejection branches, mirroring
`lblt_panel_frontal` line-for-line but with `BLOCK_SIZE = 32` baked in.

## 6. What "in-register" actually buys (quantification)

From `dev/research/ssids-small-frontal-speed.md` §6:

- Current feral 32×32 front: ~960 pulp dispatch calls + many small
  branches in the panel driver, plus `Vec::resize(actual_nrow²)` per
  front.
- SSIDS analogue: ~240 inlined update bodies, one source load per 4
  destination columns, no per-call dispatch.

Per-call savings:
- Source loads: 4× reduction (1 per 4 dest cols vs 1 per axpy).
- Dispatch overhead: ~30× reduction (1 per kernel call vs per axpy).
- Branch elimination: pivot peek-ahead and `is_aligned` removed.

Expected end-to-end on CHAINWOO_0000: 89 ns/nnz → target ≤44 ns/nnz
(1.5× of SSIDS's 29 ns/nnz). Factor wall-clock 25000 μs → ≤12500 μs.
Dense p90 vs MUMPS on the corpus: 1.86 → target ~1.5 (the chain
matrices currently dominate the tail).

These are upper-bound estimates. The actual gain depends on whether
the panel driver still calls `peek_ahead_column` per pivot (it should
not, in the block-32 path), and whether LLVM keeps `avec0..3` in
registers across the `r` loop (it should, with `const BS = 32`).

## 7. Risks / open questions

1. **Bit-parity under different lane widths.** The unroll4 axpy already
   passes bit-parity across SSE2/NEON/AVX2/AVX-512 because lane width
   does not affect the per-element computation (each lane is an
   independent scalar `d - mul(α, s)`). The same property holds for
   `update_1x1_block32`: 4 destination columns × `r/vlen` tiles ×
   `(mul, sub)` per lane = same per-lane operation order as scalar.
   This must be tested across at least NEON and AVX2.
2. **Swap-cols cost.** SSIDS's `swap_cols` (block_ldlt.hxx:swap_cols)
   swaps two columns of `a` plus the corresponding `work` and `perm`
   entries. In feral the same primitive exists in `scalar_pivot_step`.
   The block-32 kernel should reuse it; no new code needed here.
3. **Divide-through-and-copy.** SSIDS modifies the pivot column in
   place (storing `a[p, r] / d11`) and copies the *pre-divide* values
   into `work` for the update. Feral's scalar path currently divides
   into the L column directly (no `work` copy) — both produce the
   same final L. The block-32 kernel can either follow SSIDS (use
   `work`) or compute `alpha = work[c] = a[p, c]` first and then
   `a[p, c] /= d11`, keeping the original API. Either way the rounding
   chain inside `update_1x1` is `mul(work[c], a[p, r])` followed by
   `sub(a[c, r], …)` — bit-identical to scalar.
4. **Mid-panel scalar fallback.** When pivot rejection forces a scalar
   step, the block-32 kernel must hand off `(a, perm, pos/neg/zero,
   subdiag, n_elim)` in a state byte-identical to what the panel
   driver produces today. The contract: `n_elim` columns are committed
   in `a` and `perm`; pending updates from those pivots are
   *not deferred* (i.e., the SSIDS divide-and-update is applied
   eagerly, not peek-ahead'd). This makes the handoff state simpler
   than `lblt_panel_frontal`'s deferred-Schur state. Tradeoff: at
   handoff to scalar, the trailing columns past `n_elim` already see
   the rank-1 updates from pivots `< n_elim`, which matches scalar's
   eager semantics and removes the `j_start = k + n_elim + 1`
   peek-ahead adjustment the panel driver currently does.

## 8. Conclusion

The block-32 kernel is **not** a port of SSIDS's `block_ldlt` and is
**not** a "just replace the FMA call" cleanup. It is:

- A monomorphized `BLOCK_SIZE = 32` analogue of feral's existing
  panel driver, with feral's BK pivot rules baked in.
- An `update_1x1_block32` and `update_2x2_block32` pair that pack
  four trailing destination columns per source load, using
  `mul_f64s` + `sub_f64s` (never FMA) to preserve scalar bit-parity.
- One pulp dispatch per kernel call (not per axpy), eliminating the
  ~30× per-pivot dispatch overhead in the current path.
- Eager (not deferred) trailing update inside the block-32 driver,
  matching scalar semantics and simplifying the scalar-fallback
  handoff.

Acceptance: bit-parity unit tests at the block-factorization level,
zero deltas on `parallel_corpus_parity`, and ≤44 ns/nnz on
`CHAINWOO_0000` (1.5× of SSIDS, ~2× faster than feral today).
