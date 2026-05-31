# Performance review — determinism-constrained optimization headroom (2026-05-31)

Branch: `claude/linalg-perf-review-Bib1d`. This is a survey/analysis note, not
an implementation. It maps the hot paths, states what is already optimal, and
prioritizes the remaining gains *under the hard constraint that they stay pure
Rust and bit-exact* (inertia must be exactly correct — that rules out the two
classic levers, FMA and order-dependent parallel reductions, unless they can be
made bit-identical to the sequential reference).

## 0. Where the time goes today

`bench` (session 2026-05-30-01): factor-time **geomean already < 1.0 vs MUMPS**
(sparse geomean ~0.36 — feral is faster than MUMPS on the average matrix). The
problem is entirely in the **tail**:

- p90 dense 1.34 / 1.74, sparse 1.50 / 1.50 (all PASS, but above 1.0).
- Worst cases: `KIRBY2_0007` n=458 ratio **8.76**, `CRESC132_0000` n=5314 **6.63**,
  `MUONSINE_0000` n=1537 **5.63**.
- Parallel speedup ceiling: cont-201 measured **1.44× at T=8** against a 4.83×
  critical-path bound (tried-and-rejected 2026-05-12). Tree parallelism starves
  near the root.

Two distinct tail populations, with different bottlenecks:

1. **Small matrices (numeric sub-ms): symbolic-dominated.** KIRBY2 numeric is
   tiny; its bad ratio is analysis time (ordering + the LdltCompress
   double-Hungarian; compression adds ~17× symbolic on KIRBY2 per
   tried-and-rejected 2026-04-23). The 154k-matrix geomean/p90 lives here.
2. **Large matrices (CRESC132, pinene, MUONSINE roots): numeric-dominated,
   serial.** A wide near-dense root supernode factors on one thread; this is
   the parallel ceiling and the worst absolute times (issue #8 pinene_3200:
   118k delayed pivots → ~14k-col root → 87s).

## 1. What is already optimal (do not touch)

- **Dense Schur micro-kernel** (`src/dense/schur_kernel.rs`): pulp-dispatched,
  register-blocked, NR = 4/2/1 trailing columns sharing the L-panel load across
  accumulators, **4 independent non-FMA accumulators**. This already harvests
  the ILP win deterministically — separate `mul`+`sub` reproduces the scalar
  loop's two-rounding semantics bit-for-bit (decisions.md 2026-04-14). On the
  Apple-Silicon dev target nofma vs FMA is 1.87→1.86 (noise): the pipes are
  already saturated.
- **Tree-parallel multifrontal driver**
  (`factorize_multifrontal_supernodal_parallel`): default-on, gated by
  `N_PAR_MIN=32` supernodes / ≥2-child branching / `PAR_MIN_FLOPS=1e7`. The
  ~1–2% inertia race was root-caused (double-spawn from `pending.load()==0`
  mid-seed) and fixed 2026-04-21; `tests/parallel_parity.rs` is green and
  unignored, `diag_par_repeat` = 0 mismatches over 38 878 runs. Determinism is
  preserved by iterating children in fixed `snode.children` order (no parallel
  FP reduction).
- **Multi-RHS solve** (`src/numeric/solve.rs`): row-major working buffer (issue
  #57 — fixed the stride-`n` cache-aliasing bottleneck), BLAS-3 panel kernel
  (MR=4, NR=8) above `BLAS3_NRHS_THRESHOLD=32`, fused D-solve, active-set
  compaction in batched refinement.
- **Symbolic**: column counts are Gilbert-Ng-Peyton O(nnz+n·α) (Phase 2.5.1,
  down from O(n²)); the in-tree AMD is retired in favour of the external
  `feral-amd` crate (18–88× faster on large).

## 2. Prioritized opportunities (bit-exact unless noted)

### Tier 1 — highest leverage, determinism is *free*

**1.1 Intra-front (node-level) parallelism for large/root supernodes.**
This is THE ceiling. The parallel driver only does *inter-front* (tree)
parallelism; the dense factor of a single front
(`factor_frontal_blocked_in_place_with_scratch`) runs on one thread. Near the
root, tree parallelism has nothing left to schedule, so the 1.44×@T=8 result is
dominated by the serial root.

The trailing Schur update `apply_blocked_schur_panel`
(`src/dense/factor.rs:2957`) is *embarrassingly parallel across trailing-column
blocks* and **bit-exact regardless of thread count**: each output element
`a[i,j]` is reduced over the same pivot order `q ∈ 0..n_elim` on a single
thread; splitting the `while j+3 < nrow` column loop with `par_chunks_mut`
introduces **no cross-thread reduction**, so the FP result is identical for any
T. The panel factorization (Bunch-Kaufman pivot search) stays serial — correct,
since it is inherently sequential and a small FLOP fraction.

This is exactly MUMPS/SSIDS "type-2 / node-level" parallelism and directly
attacks CRESC132 / pinene / MUONSINE and the parallel-speedup ceiling.
- Implementation: a per-front flop threshold (reuse `est_flops` machinery) so
  only wide fronts go parallel; nest inside the existing `rayon::scope` or use a
  separate dispatch when a front exceeds the threshold while the tree is starved.
- Risk: low on correctness (no reassociation), medium on engineering
  (rayon-inside-rayon scoping, threshold calibration). **Determinism: free.**

**1.2 Cache (L2/L3) blocking + L-panel packing in the dense Schur update for
large fronts.** Today `src_block` is read column-major at stride `nrow` and
re-streamed per trailing-column group; `block_size=64` gives L1 reuse on the
panel but there is no n-dimension (trailing) tiling and no packing. For the wide
dense roots this is memory-traffic-bound. A BLIS-style packed micro-kernel with
3-level blocking cuts traffic. **Bit-exact**: packing is a copy, not a
reassociation; the per-element accumulation order is unchanged. Pairs naturally
with 1.1 (pack once, parallelize the packed kernel).

### Tier 2 — solid, mostly bit-exact

**2.1 Put the solve/refinement on the pulp path + parallelize across RHS.** The
solve GEMM (`gemm_panel_minus`, solve.rs:800) relies on LLVM auto-vectorization,
not the pulp kernels used in factor; refinement `norm2` and the SpMV are scalar.
For IPM hosts (many solves) and many-RHS workloads: (a) route forward/back GEMM
through the deterministic nofma pulp kernels, (b) parallelize across *independent
RHS columns* (embarrassingly parallel, bit-exact). Lower leverage for single-RHS,
real for many-RHS / heavy refinement.

**2.2 Symbolic-phase speedups for the small-matrix bulk.** The p90 vs MUMPS on
small matrices is symbolic-dominated. Two concrete, already-flagged items
(tried-and-rejected 2026-04-23): plumb the cached MC64 matching into the
`ldlt_compress` path (kills the double-Hungarian), and auto-dispatch compression
only on predicted-tail matrices (large-n + MC64 compRat ≤ 0.7), mirroring
`ScalingStrategy::Auto`. This moves the metric the 154k corpus actually lives in.

### Tier 3 — platform-dependent / low priority

**3.1 FMA with a boundary-safe fallback (the open "option 2").** ~0% on ARM (4
nofma accumulators already saturate the pipes); possibly material on x86
AVX-512 (untested). Only viable with the detect-pivot-within-k·eps-of-zero_tol
and fall-back-to-scalar mitigation, and only after measuring a real x86 gap.
High complexity, conditional payoff — keep opt-in.

**3.2 Wider micro-kernel NR (4→6/8).** Diminishing returns on the small fronts
that dominate, more remainder code. Measure only after 1.1/1.2 land.

### GPU — recommend against, for this workload

- The corpus is dominated by small fronts (n<500); GPU kernel-launch latency
  (~µs) dwarfs per-front compute. Only the few huge dense roots could benefit —
  but those are gated by the inherently *serial* Bunch-Kaufman pivot search,
  exactly what a GPU cannot accelerate.
- **Determinism**: GPU FP (atomic reductions, non-fixed reduction trees,
  default FMA) breaks the bit-exact inertia guarantee the library is built on.
  Reproducing bit-exact LDLᵀ inertia on GPU is not practically achievable.
- **Pure-Rust constraint**: wgpu/cubecl/rust-gpu drag in shader compilers and
  drivers — a large non-Rust-toolchain surface against "zero non-Rust deps."
- Verdict: out of scope. The same effort spent on Tier-1 CPU intra-front
  parallelism yields a deterministic win on the matrices that actually hurt.

## 3. Recommended next step

Tier-1 1.1 (intra-front parallel Schur update) — highest leverage, zero
determinism risk, hits both the worst-case ratios and the parallel ceiling.
Per protocol: research note → a par-vs-seq bit-parity harness on the wide-root
subset (CRESC132 / pinene_3200 / MUONSINE / nql180) → implement behind a
per-front flop threshold → benchmark. Confirm `parallel_corpus_parity` stays at
0 mismatches (it must, by construction) and that `bench` p90/worst improve on the
large-n bucket without regressing the small-front geomean (guard the threshold).
