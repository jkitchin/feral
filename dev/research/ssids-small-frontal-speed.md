# How SSIDS Achieves Small-Frontal Speed (CHAINWOO-style KKT)

Reference run: CHAINWOO_0000 (n=4000, max front 32x32 per symbolic). MUMPS 726us (14 ns/nnz), SSIDS 3564us (29 ns/nnz), feral 25000us (89 ns/nnz). Goal: identify what SSIDS does that lets a non-vendor-tuned C++ kernel achieve 3x feral's per-nnz throughput on tiny fronts.

All SPRAL paths refer to `/Users/jkitchin/Dropbox/projects/ripopt/ref/spral/`. All FERAL paths refer to `/Users/jkitchin/Dropbox/projects/feral/`.

## Top-line architectural fact

SSIDS does not have a single LDL^T kernel. It has THREE, dispatched by front geometry and pivot success:

1. `block_ldlt<T, INNER_BLOCK_SIZE=32>` — fully SIMD-vectorized in-register 32x32 LDL^T (`src/ssids/cpu/kernels/block_ldlt.hxx:289`). Used when ncol >= 32 AND the column block pointer is aligned. NO threshold pivoting: this is the unpivoted/aggressive APP path that operates on entire 32-column inner blocks in one shot.
2. `ldlt_app_factor` — recursive blocked LDL^T with appended pivoting (`src/ssids/cpu/kernels/ldlt_app.cxx`). For block_size != 32 it recurses with INNER_BLOCK_SIZE=32 inner blocks; for block_size == 32 it dispatches to either `block_ldlt` (aligned full block) or `ldlt_tpp_factor` (residual). This is the default `pivot_method = app_aggressive`.
3. `ldlt_tpp_factor` — scalar threshold pivoting (`src/ssids/cpu/kernels/ldlt_tpp.cxx:166`). Self-described in line 165 as "Intended for finishing off small matrices, not for performance." It uses BLAS-level GEMM for the trailing rank-1/rank-2 update (lines 217, 232, 251), but the pivot search itself is scalar.

The dispatch lives in `LDLT::factor()` at `ldlt_app.cxx:996-1020`:
- `if (ncol() < INNER_BLOCK_SIZE || !is_aligned(aval_))` -> `ldlt_tpp_factor`
- else -> `block_ldlt<T, INNER_BLOCK_SIZE>` (the SIMD path)

So for a 32x32 fully-summed block CHAINWOO front, SSIDS goes through `block_ldlt`, which is the fastest path.

## 1. The 32x32 in-register kernel (`block_ldlt.hxx`)

This is the heart of SSIDS for small fronts. Three pieces, all templated on BLOCK_SIZE=32:

### find_maxloc (block_ldlt.hxx:77-206)
Vectorized full-block max-abs scan. Maintains per-lane `bestv`, `bestr`, `bestc` SimdVec accumulators with two-way unroll. Single pass over the lower triangle. On AVX2 this is ~64 vectorized loads to scan a full 32x32 lower triangle.

Key optimization: a "best-in-lane" tournament rather than a serial scan. The inner body at lines 156-174 unrolls by 2x vector length and uses `blend()` (mask-merge) instead of branches.

### update_1x1 (block_ldlt.hxx:233-282)
The rank-1 trailing update: `a[c,r] -= ld[c] * a[p,r]`. Done in-place on the 32x32 block, NOT via BLAS. Manually unrolled 4 columns at a time (lines 259-280) with FMA. Each inner iteration loads ONE source vector and FOUR destination vectors and issues four FMAs. This is the right structure for tiny in-register blocks: maximize FMA throughput, amortize load cost across multiple destination columns.

Note line 251: `SimdVec ldvec( -ld[c] )` — they negate up front so `fmadd(avec, lvec, ldvec)` performs the subtract.

### update_2x2 (block_ldlt.hxx:222-229)
Rank-2 update with `#pragma omp simd`. Less aggressive than the 1x1 path (no manual unroll) because 2x2 pivots are rare on chain-like matrices.

### Driver (block_ldlt.hxx:289-414)
`block_ldlt()` runs the entire 32x32 factorization in one function. NO BLAS calls. NO function-call overhead between pivot search and update. The pivots, swaps, divisions, and Schur updates all happen on data already in cache (often in registers across iterations).

This is the answer to question (2): for 32x32 SSIDS does NOT do "panel + trailing GEMM". It does an all-in-one in-register LDL^T with hand-rolled SIMD per primitive.

## 2. Frontal matrix layout (NumericNode + align_lda)

`src/ssids/cpu/cpu_iface.hxx:38` defines `align_lda<T>()`. Frontal columns are stored column-major with leading dimension rounded up to 32 bytes (AVX) / 64 bytes (AVX-512). For a 32x32 front in double precision the lda is 32 (already aligned), so `is_aligned(aval_)` is true and the SIMD `block_ldlt` path triggers.

Allocation: `NumericNode::lcol` is a single contiguous block of `(ldl + 2) * ncol` doubles, including 2 extra "rows" past the L data to hold D (1x1 and 2x2 pivot info), see `SmallLeafNumericSubtree.hxx:247-249`. Allocated from a `BuddyAllocator` page (`BuddyAllocator.hxx`). For small_leaf subtrees a single contiguous `lcol_` buffer of `nfactor_` doubles holds factors for ALL nodes in the subtree (`SmallLeafNumericSubtree.hxx:39, 47`), pre-zeroed via `memset`.

Contribution blocks are dense column-major (lda = nrow - ncol), allocated from a separate `PoolAllocator` (`BuddyAllocator<T>`).

## 3. Per-front overhead — small leaf subtrees

This is the second crucial fact. For chains of small fronts, SSIDS does NOT run the per-front driver loop once per front. The analyse phase (`SymbolicSubtree.hxx:57-84`) walks each leaf upward, accumulating flop count. While `flops[current] < small_subtree_threshold` (default `4*10^6` flops, `datatypes.f90:243`), the entire chain of nodes is grouped into one `SmallLeafSymbolicSubtree`.

At factor time (`NumericSubtree.hxx:97-155`) each small-leaf subtree is one OpenMP task. Inside `SmallLeafNumericSubtree::SmallLeafNumericSubtree` (indef variant at `SmallLeafNumericSubtree.hxx:192-220`):
- One pre-allocated `lcol_` for ALL nodes (no per-node malloc).
- Workspace buffers (the 32-row `ld` scratch, the `map` lookup) are taken from the thread's `Workspace` once and reused across the whole subtree.
- A flat for-loop over `ni in sa..=en` calls `assemble_pre`, `factor_node`, `assemble_post` for each node.
- The contribution blocks ARE still allocated per-node from the pool, but the pool allocator is a buddy allocator that recycles ranges, so it is essentially malloc-free in steady state.

For CHAINWOO the entire 4000-column chain factorizes inside a small handful of small-leaf subtree tasks, each running the inner loop with cache-resident state. The per-front driver overhead seen by FERAL (allocate `frontal_buf`, build `row_indices`, populate `row_map`, extend-add, factor, deposit contrib, restore `row_map`) is collapsed into one allocation for the whole subtree.

Driver locations:
- Generic per-node: `NumericSubtree.hxx:158-237` (`assemble_pre`, `factor_node`, `assemble_post`).
- Small-leaf chain: `SmallLeafNumericSubtree.hxx:196-219` for indef, `:54-69` for posdef. Note in posdef the subtree uses one shared `lcol_` and a single `memset`; per-node `assemble` only handles children's contribution blocks.

## 4. Threshold partial pivoting cost on chain-like matrices

For `pivot_method = app_aggressive` (default) SSIDS factors WITHOUT pivoting first via `block_ldlt`, postponing the threshold check to a per-block "a-posteriori" test (`apply_pivot_app` in `ldlt_app.cxx:1042`, `check_threshold` and `Column::test_fail` etc. in same file). On chains the delay rate is essentially zero, so:

- `block_ldlt` runs to completion on each 32-col block.
- The a-posteriori test passes for all 32 columns.
- No re-factorization of any column is ever needed.
- Failed pivots, when they occur, are appended to the parent (`ndelay_out -> ndelay_in`) — handled by `factor_node_indef` at `factor.hxx:75-113`.

The TPP fallback (`ldlt_tpp_factor`) only runs when (a) the user forces `pivot_method = tpp`, (b) we're at the root (m==n) and need to finish off, or (c) `failed_pivot_method == tpp` and APP failed some columns. For chains it is essentially never hit.

This is the real win: on numerically benign matrices SSIDS pays ZERO overhead for pivot search beyond a per-block max scan that runs at SIMD throughput. The cost model of TPP — every pivot search is O(m) scalar work — never enters the picture.

## 5. Two-axis cost model: kernel vs fill

Observations from the bench data:
- MUMPS:  14 ns/nnz, 51964 nnz
- SSIDS:  29 ns/nnz, 123447 nnz
- feral:  89 ns/nnz, 281526 nnz

SSIDS has 2.4x more fill than MUMPS but only 2x slower per nnz. Both ratios suggest separate issues: MUMPS uses a more aggressive amalgamation/ordering producing larger dense fronts that BLAS-3 handles efficiently. SSIDS keeps fronts small (`nemin_default = 32` per `datatypes.f90:21`) but compensates with a hand-rolled 32x32 SIMD kernel.

FERAL has 2.3x more fill than SSIDS AND 3x slower per nnz — both axes are bad. The fill gap likely comes from less-aggressive supernode amalgamation; the kernel gap comes from the points below.

## 6. Specific FERAL gaps

### Gap A: No fixed-block in-register kernel for the dominant front size
FERAL's `factor_frontal_blocked` (`src/dense/factor.rs:897`) runs a generic blocked path with runtime `bs = params.block_size` (typically 32 or 64). The panel itself (`lblt_panel_frontal` at `factor.rs:1131`) is a flexible scalar+pivot+axpy loop. There is no equivalent of `block_ldlt<T, BLOCK_SIZE=32>` — a monomorphized, branchless, inline LDL^T for one fixed block size that fits in registers.

Compare:
- SSIDS dispatch: `if(ncol() < 32 || !is_aligned) ldlt_tpp_factor() else block_ldlt<T,32>()` (`ldlt_app.cxx:996-1020`). For a 32-col aligned front, ZERO branches in the inner loop.
- FERAL dispatch: panel logic with `PanelStatus::ScalarFallback`, peek-ahead replay of pending updates per column (`peek_ahead_column` at `factor.rs:1245`), `try_reject_1x1_frontal` per pivot, etc. The hot path takes many small branches.

### Gap B: Per-front allocation overhead, no small-leaf batching
`factor_one_supernode` (`src/numeric/factorize.rs:1020`) allocates a fresh `frontal_buf` (line 1114), builds `row_indices` (line 1089), populates and clears `row_map` (lines 1107-1109, 1165-1167), and calls `factor_frontal_blocked` PER FRONT. CHAINWOO has thousands of fronts.

SSIDS amortizes this over a whole subtree: one buddy-allocator allocation for the joint `lcol_` of all nodes, `Workspace::get_ptr` calls are bump-pointer slices into a thread-local arena, `map` is re-used across nodes (`SmallLeafNumericSubtree.hxx:201`).

There is a `factor_one_small_leaf` in FERAL (`factorize.rs:1199`) that pre-computes `row_indices`, but it still allocates `frontal_buf` per leaf (it lives under `ws.frontal_values` so it is a `Vec::resize(actual_nrow * actual_nrow, 0.0)` — a fill for every front).

### Gap C: Schur update is per-pivot, not per-block
FERAL's `apply_blocked_schur` (`factor.rs:1280-1305`) loops over pivots `q` outermost, then over trailing columns `j`, calling `axpy_minus_unroll4_nofma` per pair. That is `n_elim * (nrow - n_elim)` separate calls into `dispatch_nofma` (line 520) — each call goes through pulp's WithSimd dispatch, even though the dispatch token is selected once.

SSIDS's `update_1x1` (`block_ldlt.hxx:233-282`) inlines the FMA loop directly with no dispatch per call, four trailing columns at a time per source vector load. This is ~4x fewer source loads and ZERO dispatch overhead. For a 32x32 front with 32 pivots and ~30 trailing columns, FERAL makes ~960 dispatch calls vs SSIDS's ~240 inlined FMA bodies (and SSIDS's bodies pack 4 columns per source load).

Note `axpy2_minus_unroll4_nofma` exists in `schur_kernel.rs:600` (rank-2 dual-source) but no rank-K-greater kernel (e.g. `axpyN_minus` that takes 4 columns at once). That is the missing primitive.

## 7. The dominant kernel-level gap

The single biggest factor: **SSIDS has a monomorphized 32x32 in-register LDL^T kernel with manual 4-way unroll over trailing destination columns, with no per-pivot dispatch overhead, no peek-ahead bookkeeping, no `is_aligned` branching, no PivotOutcome match arm**. The whole 32x32 factorization is a tight inlined loop.

FERAL's blocked path achieves correctness via a panel that mimics scalar semantics column by column with a peek-ahead replay. That gives bit-exactness with the scalar reference but pays ~5x in per-pivot overhead vs an unbranched in-register kernel. Combined with per-front malloc/Vec::resize and per-axpy dispatch, the 3x per-nnz gap is plausible.
