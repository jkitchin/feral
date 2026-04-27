# faer's dense Bunch-Kaufman: speed reference for FERAL's small-front kernel

Date: 2026-04-27
Author: research agent (faer expert)
Faer version examined: 0.24.0 at `/Users/jkitchin/Dropbox/projects/ripopt/ref/faer-rs/`
FERAL kernel: `/Users/jkitchin/Dropbox/projects/feral/src/dense/factor.rs` (~2342 lines)

Purpose: extract what faer's dense LBL^T factorization does that makes it fast on
32x32 to ~200x200 matrices, and identify clean-room techniques FERAL can adopt.
Read-only research; no FERAL source modified.

## 1. faer's Bunch-Kaufman implementation

All paths live in
`/Users/jkitchin/Dropbox/projects/ripopt/ref/faer-rs/faer/src/linalg/cholesky/bunch_kaufman/factor.rs`
(1234 lines, single file, all variants).

### 1.1 Top-level dispatch — `cholesky_in_place` (factor.rs:1161)

- Reads `LbltParams` (factor.rs:27): `pivoting`, `block_size = 64`,
  `par_threshold = 128*128 = 16384` (factor.rs:903-912 = `Auto::auto()`).
- Sets `bs = 0` if `bs < 2 || n <= bs` (factor.rs:1198-1201). Below this
  threshold the unblocked path runs directly.
- For `Full` pivoting calls `lblt_full_piv` (factor.rs:264). Otherwise calls
  `lblt_blocked` (factor.rs:700) which loops over panels and either calls
  `lblt_blocked_step` (BLAS-3) or `lblt_unblocked` (per-element) per panel.

### 1.2 The two factorization paths

**Unblocked path — `lblt_unblocked` (factor.rs:784)** Used when remaining size
<= block_size. For each pivot:
- `offdiag_argmax` (factor.rs:448) to get column-max gamma_i, picks 1x1 vs 2x2
  via the BK alpha test `A[i0,i0] >= alpha * gamma_i`.
- 1x1 dispatch to `rank1_update` (factor.rs:1044) — SIMD-aware.
- 2x2 dispatch to `rank2_update` (factor.rs:913) — SIMD-aware.

**Blocked path — `lblt_blocked_step` (factor.rs:491)** This is the panel
algorithm: pick `block_size - 1` pivots inside a panel, deferring the rank-1/2
trailing updates. Only the panel itself (and the W workspace columns) is
touched per pivot. After the panel, **one** triangular `gemm` flushes the
deferred updates against the trailing matrix in BLAS-3 (factor.rs:684-694):

    linalg::matmul::triangular::matmul(
        Ar.rb_mut(), StrictTriangularLower, Accum::Add,
        W, Rectangular, Al.adjoint(), Rectangular, -one, par,
    );

Per-pivot work inside the panel uses `update_and_offdiag_argmax`
(factor.rs:461) which fuses (a) the pending deferred rank-`k` update to a
single column via `linalg::matmul::matmul` (gemv), and (b) the argmax for the
next pivot search. This means each pivot pays only **O(n * c)** flops for
update+argmax, where c is the count of pivots already done in this panel,
instead of touching the full trailing matrix.

### 1.3 SIMD inner kernels (factor.rs:913-1125)

Both `rank1_update` and `rank2_update` follow the same dispatch structure
(factor.rs:922-934):

    if const { T::SIMD_CAPABILITIES.is_simd() } {
        if let (Some(A), Some(L0), Some(L1)) = (
            A.try_as_col_major_mut(),
            L0.try_as_col_major_mut(),
            L1.try_as_col_major_mut(),
        ) { rank2_update_simd(A, L0, L1, ...) }
        else { rank2_update_fallback(...) }
    } else { rank2_update_fallback(...) }

The SIMD path (factor.rs:936-1020 for rank-2, 1062-1109 for rank-1) wraps the
work in a `pulp::WithSimd` impl. Inside `with_simd`:

- `SimdCtx::new(T::simd_ctx(simd), subrange_len)` (factor.rs:985, 1092)
  — builds an indexed iterator over a column with a head/body/tail
  decomposition that encodes alignment (see Section 3).
- `simd.splat(&w0_conj_neg)`, `simd.read`, `simd.write`, `simd.mul_add`
  (factor.rs:990-998) — the unrolled FMA chain runs at one `mul_add` per
  source per lane. The macro `simd_iter!(for i in [simd.indices()] { ... })`
  (lib.rs:1285) expands to a head-fixup + body-batch + tail-fixup loop with
  `batch_size` const-generic unrolling 1..8.

The fallback (factor.rs:1021-1043) is the same algorithm in plain `for` loops
over `for j in 0..n { for i in j..n { ... } }`.

### 1.4 Pivot kernel structure summary

For the inner pivot loop in the **blocked panel** (factor.rs:517-680):

1. Compute `(r, gamma_i) = update_and_offdiag_argmax(...)` — fused gemv +
   argmax of the pivot's own column. `par` is passed through; this gemv can
   itself parallelize.
2. If diagonal test passes (factor.rs:546): npiv = 1.
3. Otherwise: do another `update_and_offdiag_argmax` for the candidate
   off-diagonal row (factor.rs:577-584), check 2x2 conditions, possibly
   rook-search loop (factor.rs:551-575).
4. Self-adjoint swap of pivot rows (factor.rs:606-616) and corresponding swaps
   in `Al`, `Wl`, `Wr`.
5. **Update only the W workspace** (factor.rs:620-670). The trailing A is not
   touched per pivot.

The deferred trailing update (factor.rs:684-694) is the only place touching
the bulk of the matrix; it runs once per panel, not once per pivot.

### 1.5 Blocking thresholds

- `block_size = 64` (default, factor.rs:907)
- `par_threshold = 128 * 128 = 16384` (factor.rs:908) — below this the
  parallel split inside `rank_1_update_and_argmax` falls back to sequential.
- Unblocked path for `n <= 64` or `block_size < 2` (factor.rs:1199-1201,
  722-733): for `n = 32`, faer does **not** use the blocked path. It runs
  the unblocked path with the SIMD-vectorized rank-1/rank-2 updates.
- For `n in (64, 200]`: blocked path with one full panel (size 64) plus one
  unblocked tail.

## 2. Small-matrix dispatch: what runs at n = 32

For `n = 32` and default params:

1. `cholesky_in_place` (factor.rs:1161) is called, sets `bs = 0` since
   `n <= bs`.
2. `lblt_blocked` (factor.rs:700) runs with `block_size = 0`, so the inner
   `if block_size < 2 || n - k <= block_size` branch (factor.rs:722) is
   true and dispatches to `lblt_unblocked` immediately.
3. `lblt_unblocked` (factor.rs:784) loops pivots 0..n, each calling
   `rank1_update` or `rank2_update`.
4. Each `rank1_update` call (factor.rs:1044) runs `dispatch!(...)` (lib.rs:201)
   which calls `pulp::Arch::default().dispatch(...)`. Inside, the SIMD body
   runs `simd_iter!` over column slices, doing 1 `mul_add` per element.

So **for n = 32, faer's hot path is: scalar pivot search + SIMD-vectorized
column-axpy with `pulp::Arch` dispatch every rank-1 call**. Not pure scalar
fallback.

## 3. Memory layout & SIMD alignment

### 3.1 Owned matrix alignment — `align_for` (matown.rs:8)

    pub(crate) fn align_for(size: usize, align: usize, needs_drop: bool) -> usize {
        if needs_drop || !size.is_power_of_two() { align }
        else { Ord::max(align, 64) }
    }

For `f64` (size = 8, power of two, no Drop): faer over-aligns owned matrix
allocations to **64 bytes** — enough for AVX-512 zmm registers. Allocation
goes through `Layout::from_size_align(size, align)` (matown.rs:92).

The row capacity is rounded up to a multiple of `align/size = 8` doubles
(matown.rs:78-82). So even a `Mat::<f64>::zeros(32, 32)` has stride 32 (exact
multiple of 8) and the column base pointers are 64-byte aligned.

### 3.2 Strided views — `MatRef`/`MatMut`

`MatRef` (matref.rs) carries arbitrary `row_stride` and `col_stride`. The
SIMD kernels test `try_as_col_major_mut()` (factor.rs:923-927) before
dispatching to the SIMD path; if the stride is not 1 the fallback path is
used. The `ContiguousFwd` type marker (factor.rs:937) is a compile-time
witness that `row_stride == 1`, gating the SIMD entry point.

### 3.3 Alignment-aware SIMD iteration

`SimdCtx::new` (utils/simd.rs:158) computes a head/body/tail decomposition
based on the actual pointer alignment, then `simd_iter!` (lib.rs:1285) walks
those three regions. Loads/stores in the body are aligned (single-instruction
on AVX-512, two on AVX2). Head and tail use masked loads where supported.

This means **faer doesn't require alignment** — it adapts at runtime — but
the over-allocated 64-byte alignment ensures the body region is the full
column for typical `Mat<f64>` allocations.

### 3.4 Stack matrices — `stack_mat!` macro (lib.rs:222)

For temporary panel workspaces, faer can use `stack_mat!` which puts a
`#[repr(align(64))] struct __Col<T, const A: usize>([T; A])` on the stack —
no heap allocation, full SIMD alignment. The Bunch-Kaufman path uses
`temp_mat_uninit` (factor.rs:719) backed by `MemStack` (passed in via
`stack: &mut MemStack`) which is a bump allocator; the `MemStack::new_aligned`
constructor preserves alignment.

## 4. The pulp dispatch mechanism

### 4.1 Dispatch entry points

`pulp::Arch::default()` returns a singleton `Arch` describing the live CPU.
It is constructed via CPUID (cached in a static `AtomicU8`, so per-call
overhead is one relaxed atomic load + one branch). The `dispatch` method
(faer-traits/src/lib.rs:1156-1163):

    pub trait SimdArch: Copy + Default + Send + Sync {
        fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R;
    }
    impl SimdArch for pulp::Arch {
        fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R { self.dispatch(f) }
    }

`pulp::Arch::dispatch` does an indirect call into one of N monomorphizations
of `f.with_simd::<S>(self)`, where S is `pulp::x86::V4` (AVX-512), `V3`
(AVX2+FMA), `V2` (SSE2), `aarch64::Neon`, or `Scalar`. The selected variant
is decided once via the cached CPUID feature bits.

### 4.2 Per-call cost

For the body of a SIMD kernel, the overhead is:
- 1 atomic load for the cached arch token
- 1 indirect call (function pointer through a vtable)
- The closure captures (`A, L0, L1, d, ...`) are passed by value through the
  monomorphized `with_simd` impl — no allocation.

For tiny matrices this is **a few hundred cycles of overhead per dispatch
call**. faer mitigates this by:

(a) NOT dispatching per inner iteration — the dispatch happens once per
    rank-1 / rank-2 call, which itself processes O(n^2) elements (at n=32,
    ~500 elements). So overhead is amortized across ~500 fmas.

(b) For very inner kernels (the per-column axpy inside `rank1_update_simd`),
    the loop body is inlined inside `with_simd` and there is **no further
    dispatch** — just a `for j in 0..n` loop wrapped in one `Arch::dispatch`.

### 4.3 Does the overhead matter for tiny n?

Yes — for n = 32 with a single rank-1 update touching 31 columns, the
inner per-column axpy loop has ~500 fmas total. At one dispatch per
`rank1_update` call (not per column), the dispatch cost is ~1% of the
useful work. **But**: faer does the dispatch once per pivot, so for n = 32
that's 32 dispatches per factorization. FERAL's `axpy_minus_unroll4_nofma`
dispatches **once per column per pivot** (see `apply_blocked_schur`
factor.rs:1302), i.e. ~500 dispatches per n = 32 factorization. This is the
key insight for FERAL's perf gap.

## 5. Comparison to FERAL's current dense kernel

FERAL's `factor()` (factor.rs:177) and `factor_frontal_blocked`
(factor.rs:897) are the two production entry points. Inner kernels:
`do_1x1_pivot` (factor.rs:2004), `do_2x2_pivot` (factor.rs:2096),
`apply_blocked_schur` (factor.rs:1280), and `schur_kernel::*_unroll4_nofma`
(schur_kernel.rs:541, 600).

### 5.1 Critical inefficiency #1: dispatch granularity (per-column instead of per-pivot)

**FERAL `apply_blocked_schur` (factor.rs:1280-1305):**
```rust
for (q, &d_q) in d_panel.iter().enumerate().take(n_elim) {
    ...
    for j in j_start..nrow {
        ...
        schur_kernel::axpy_minus_unroll4_nofma(dst, src, alpha);
    }
}
```
This calls `axpy_minus_unroll4_nofma` once **per (q, j)** pair — a single
column slice of length `nrow - j`. Each call goes through `dispatch_nofma`
(schur_kernel.rs:520) which does a CPUID branch + V3::try_new + indirect
call.

**faer `lblt_blocked_step` (factor.rs:684-694):** flushes the entire
panel-deferred update to the trailing matrix in **one** call to
`linalg::matmul::triangular::matmul`, which itself dispatches once. The inner
SIMD loop is a register-blocked GEMM kernel that processes the n_elim x
(nrow - k - n_elim) trailing block as one fused operation.

Impact estimate: at n_elim = 16, nrow = 64, FERAL pays 16 * 48 = 768
dispatches; faer pays 1.

### 5.2 Critical inefficiency #2: rank-1 inner kernel touches one column at a time

**FERAL `do_1x1_pivot` (factor.rs:2081-2087):**
```rust
for j in (k + 2)..n {
    let l_jk = a[k * n + j];
    let l_jk_d = l_jk * d;
    for i in j..n {
        a[j * n + i] -= a[k * n + i] * l_jk_d;
    }
}
```
This is a scalar `for i in j..n` loop. It is **not SIMDized**: the rank-1
update from FERAL's unblocked path runs in plain scalar Rust until the
schur_kernel was wired in. The SIMD path (`do_1x1_update`) is used only by
the blocked path.

**faer `rank1_update_simd` (factor.rs:1062-1109):** wraps the entire
`for j in 0..n { for i in j..n { axpy } }` double loop inside one
`pulp::WithSimd` impl. The SIMD body uses `simd_iter!` with const-generic
unroll 4 (see lib.rs:1310-1318). One dispatch covers the whole rank-1 update.

Impact: FERAL's scalar `do_1x1_pivot` rank-1 update on n = 32 leaves ~500
fmas un-vectorized. With AVX2 (4 doubles/lane), that's a 4x speedup left on
the table for the unblocked path.

### 5.3 Behavioral inefficiency #3: peek-ahead replay duplicates work

**FERAL `peek_ahead_column` (factor.rs:1245-1269):** before each pivot search
inside a panel, FERAL replays all c prior rank-1 updates against the single
candidate column. This is O(c * (nrow - col)) per pivot search.

**faer `update_and_offdiag_argmax` (factor.rs:461-489):** does the same
fused update + argmax via a **single gemv** call:
```rust
linalg::matmul::matmul(dst.rb_mut(), Accum::Add, Al.rb(), Wl.row(i0).adjoint(), -one, par);
```
This is a single BLAS-2 call processing c source columns simultaneously
against one destination column. With register blocking + FMA pipelining,
this fuses the c separate axpys into one gemv at near-peak FMA throughput,
where FERAL's c-call loop pays c separate dispatch + loop-setup costs.

Impact: at panel pivot c = 32, FERAL pays 32 dispatches + 32 loop setups;
faer pays 1 gemv that the matmul kernel internally tiles to feed FMA units
continuously.

## 6. Recommended adoption path: techniques, not dependency

### 6.1 Should FERAL just call faer?

**No.** Three reasons:

(a) **License / clean-room constraint.** FERAL's project spec explicitly says
    "Clean-room implementation from published papers and BSD-licensed
    references." faer is MIT, which is permissive enough to use as a
    dependency, but the constraint is stricter: the goal is a clean-room
    derivation. Adopting faer as a dependency does not violate the MIT
    license but does violate the project's clean-room intent.

(b) **Architectural intent.** CLAUDE.md lists "Zero non-Rust dependencies in
    the core solver (no BLAS, LAPACK, Fortran)" — faer would satisfy this
    (pure Rust), but pulling in faer also pulls in `pulp`, `dyn-stack`,
    `equator`, `reborrow`, `spindle`, `bytemuck`, and a transitive Rayon
    dependency. That's a major surface-area expansion.

(c) **Different pivoting requirements.** FERAL's BK has KKT-specific
    behavior: zero-tolerance handling, force-accept inertia rules,
    Duff-Reid 2x2 growth bound, MUMPS-compatible delayed pivoting (SSIDS
    threshold test), and exact-inertia guarantees. Adapting faer's
    `cholesky_in_place` to expose those is non-trivial.

The clean-room *technique* adoption is what's wanted.

### 6.2 Top three clean-room techniques to adopt, ranked

**Rank 1 (highest impact): collapse `apply_blocked_schur` to a single GEMM-like call.**

  - FERAL location: `apply_blocked_schur` at
    `/Users/jkitchin/Dropbox/projects/feral/src/dense/factor.rs:1280`.
  - faer reference: `lblt_blocked_step` flush at
    `/Users/jkitchin/Dropbox/projects/ripopt/ref/faer-rs/faer/src/linalg/cholesky/bunch_kaufman/factor.rs:684-694`.
  - Technique: instead of nested `for q in 0..n_elim { for j in j_start..nrow { axpy } }`,
    write a single `pulp::WithSimd` kernel whose body iterates
    `for j in j_start..nrow` outermost and accumulates *all* `n_elim`
    contributions into 4 register accumulators per j-batch before storing
    back. This is a triangular GEMM-like inner kernel — same algorithm,
    one dispatch, register-blocked.
  - Expected impact: for typical small fronts (n_elim = 16, trailing = 32-64),
    eliminates ~500 dispatches per Schur flush. Should close most of the 5x
    gap to MUMPS for fronts in the 32-200 range.

**Rank 2: SIMDize `do_1x1_pivot` and `do_2x2_pivot` rank-update inner loops.**

  - FERAL location: `do_1x1_pivot` rank-1 update at
    `/Users/jkitchin/Dropbox/projects/feral/src/dense/factor.rs:2081-2087`;
    `do_2x2_pivot` rank-2 update at `factor.rs:2170-2182`.
  - faer reference: `rank1_update_simd` and `rank2_update_simd` in
    `bunch_kaufman/factor.rs:1062-1109` and `:936-1020`. The SIMD body is
    one `pulp::WithSimd` impl wrapping the full `for j { for i }` double
    loop.
  - Technique: wrap the entire rank-1 / rank-2 update inside a single
    `pulp::WithSimd` impl. The fused argmax (next_gamma0 / next_r tracking
    on column k+1) can stay as a scalar fix-up before / after the SIMD body
    to preserve bit-exactness. Use `S::as_simd_f64s` + `chunks_exact_mut(4)`
    unrolling exactly like `axpy_minus_unroll4_nofma` already does, but
    inline the `for j` outer.
  - Expected impact: for the unblocked path used at n <= block_size (which
    is dominant for 32x32 fronts in FERAL's multifrontal driver), this
    converts 100% scalar updates to 4-wide SIMD — ~4x speedup on AVX2,
    ~8x on AVX-512 / NEON-pair.

**Rank 3: replace `peek_ahead_column` with a fused gemv + argmax kernel.**

  - FERAL location: `peek_ahead_column` at
    `/Users/jkitchin/Dropbox/projects/feral/src/dense/factor.rs:1245-1269`,
    called inside `lblt_panel_frontal` at `factor.rs:1151`.
  - faer reference: `update_and_offdiag_argmax` at
    `bunch_kaufman/factor.rs:461-489`. Uses one `linalg::matmul::matmul`
    call to fuse c rank-1 contributions into one gemv, plus an in-line
    `l1_argmax` (factor.rs:431).
  - Technique: write a single SIMD kernel `gemv_minus_with_argmax(dst,
    src_cols, scales, dst_argmax)` that takes c source columns and c
    scales, does `dst -= sum_q scales[q] * src_cols[q]` in 4-way unrolled
    SIMD, and tracks the absolute-value argmax of `dst` simultaneously.
    This replaces the q-outer loop entirely.
  - Expected impact: removes c dispatches per panel pivot (where c grows
    0..bs). For bs = 16, this saves ~120 dispatches per panel of 16 pivots
    plus enables FMA pipelining across the c sources. Estimated 1.5-2x on
    the panel pivot search itself.

### 6.3 Architectural notes for adoption

Each of the three kernels is a **leaf SIMD function** that lives in
`crate::dense::schur_kernel`. The proposed change is purely:
  - extend `schur_kernel` with three new pulp-dispatched kernels;
  - rewire `apply_blocked_schur`, `do_1x1_pivot`, `do_2x2_pivot`, and
    `peek_ahead_column` call sites in `factor.rs` to call them.

No changes to factorization algorithm, pivot strategy, inertia accounting,
or the bit-exactness contracts; the rounding can still be controlled by
explicit `mul + add + sub` ordering (as `axpy*_unroll4_nofma` already does)
to maintain scalar-bit-exactness if required.

The `pulp` dependency is already present (FERAL uses it via
`schur_kernel`), so this is purely a kernel-rewrite, not a dep-graph change.

## 7. Final summary (<=400 words)

faer's dense Bunch-Kaufman LBL^T factorization
(`faer/src/linalg/cholesky/bunch_kaufman/factor.rs`) achieves its speed
through three layered techniques:

**Deferred panel updates with one BLAS-3 flush.** `lblt_blocked_step`
(factor.rs:491) processes block_size pivots inside a panel, accumulating
deferred rank-1/2 contributions into a workspace W. After the panel,
**one** `triangular::matmul` call (factor.rs:684-694) flushes the entire
update to the trailing matrix, achieving GEMM-bound throughput on the bulk
of the work.

**Coarse-grained pulp dispatch.** `pulp::Arch::default().dispatch(...)` is
called **once per rank-1 or rank-2 update** (factor.rs:1044, 913), wrapping
the full `for j { for i }` double loop. Inside `with_simd`, the body is
fully monomorphized, register-blocked, and inlined — no further dispatch
boundaries. CPUID is cached in a static atomic, so dispatch overhead is
~ns.

**Stride-witnessed SIMD with 64-byte alignment.** Owned matrices (matown.rs:8)
are over-aligned to 64 bytes for AVX-512. The `ContiguousFwd` type marker
(factor.rs:937) gates SIMD entry on `row_stride == 1`. `SimdCtx`
(utils/simd.rs) decomposes columns into head/body/tail for aligned
loads/stores; `simd_iter!` (lib.rs:1285) does const-generic unrolling 1..8.

For n = 32, faer skips the blocked path entirely (factor.rs:1199-1201) and
runs `lblt_unblocked` with the SIMD `rank1_update_simd` /
`rank2_update_simd` kernels — one dispatch per pivot covers all (n - k)
columns.

**Should FERAL adopt faer as a dependency?** **No.** The project's
clean-room and dependency-minimalism constraints rule this out, and the
kernels needed are small and self-contained.

**Three highest-impact clean-room techniques for FERAL** (in priority order):

1. Rewrite `apply_blocked_schur` (`src/dense/factor.rs:1280`) as one
   pulp-dispatched triangular-GEMM kernel — eliminates O(n_elim * trailing)
   dispatches per Schur flush. Mirrors faer's `lblt_blocked_step` flush
   (factor.rs:684).

2. Wrap `do_1x1_pivot` rank-1 update (`src/dense/factor.rs:2081`) and
   `do_2x2_pivot` rank-2 update (`src/dense/factor.rs:2170`) in single
   pulp `WithSimd` impls covering the whole `for j { for i }` double loop.
   Mirrors faer's `rank1_update_simd` / `rank2_update_simd`
   (factor.rs:936, 1062).

3. Fuse `peek_ahead_column` (`src/dense/factor.rs:1245`) into a single
   gemv-with-argmax SIMD kernel. Mirrors faer's `update_and_offdiag_argmax`
   (factor.rs:461).

All three are leaf SIMD kernels; no algorithmic, pivot-strategy, or
inertia-accounting changes are required, and bit-exactness with the scalar
loop can be preserved by explicit `mul/add/sub` ordering as the existing
`*_unroll4_nofma` kernels already demonstrate.

