# Issue #249: real memory high-water vs `WorkEstimate::peak_bytes`

Session 2026-11-24-03. Follow-up from #246 / PR #247.

## Question

`WorkEstimate::peak_bytes` was `factor_bytes + peak_contrib_bytes`: the
slacked column-count factor plus the simulated postorder peak of square
contribution blocks. On the Primal2_1000 LP KKT the maintainer measured
VmHWM at 1.4–1.8× that (sequential) and 2.6× (parallel), and the
parallel driver ran out of memory under a 13.5 GB `RLIMIT_AS` on AMD.
`max_factor_bytes` compares against `peak_bytes`, so it bounded the
prediction, not the process. Break the gap down, then fix the estimate
or the allocation.

## Method

`examples/issue249_memory_breakdown.rs`: a counting global allocator
(live and peak heap bytes), a 250 ms window-peak sampler for the
timeline, and after `Solver::factor` a census of every retained
structure by `capacity()`. Peaks are reported above the input matrix,
which is the caller's. `predict` prints the model from the analysis
alone.

Matrix: Primal2_1000 from plato.asu.edu, converted with
`dev/scripts/mps_to_kkt.py` (n = 3 858 760, 9 356 900 stored entries,
input CSC 0.181 GB). 4 cores, 15 GB. The earlier VmHWM figures (6.37 GB
sequential AMF) are not reproduced here (4.93 GB); the same ratios to
the old prediction hold for the heap counter.

## Breakdown (AMF, before any change)

| term | bytes | in old `peak_bytes`? |
|---|---|---|
| L values, `Σ nrow·ncol·8` | 2.487 GB | as 1.996 GB (`1.2 × Σ col_counts`) |
| per-node index data | 0.778 GB | no |
| analysis kept by the `Solver` | 0.772 GB | no |
| workspace and `Solver` caches | ~0.4 GB retained | no |
| contribution blocks (sequential, packed) | ≤ 0.32 GB above final | 1.534 GB (square) |

- **Amalgamation fill.** The column counts are taken before
  amalgamation: `Σ col_counts` = 2.079e8, the stored blocks
  `Σ nrow·ncol` = 3.109e8 (1.50×), of which 0.20 GB is the strict upper
  triangle of each diagonal block. The 1.2 slack does not cover it.
- **Per-node index data.** BK `perm` and `perm_inv` are `nrow` long
  (0.298 GB), row indices 0.149 GB, the `NodeFactors` record 352 B ×
  720 331 supernodes = 0.254 GB, `D` 0.078 GB.
- **Analysis.** `static_row_flat` 0.268 GB, `permuted_pattern` 0.150,
  supernode row lists 0.146, permutations 0.062, the rest small.
- **Workspace.** The permuted copy of the matrix (kept by the permute
  cache), the cache's copy of the input pattern, inverse permutation and
  value route, the `Solver`'s fingerprint pattern, `row_map`, scaling
  vectors, the frontal pool.

Heap timeline, parallel driver, 4 threads: the live heap climbs to its
peak at 28.5 s and then falls ~0.12 GB every 4–5 s until the end (67 s).
The off-chain subtree tasks finish early and their blocks wait for the
long top chain. The sequential driver climbs monotonically; its peak
comes near the end, 0.32 GB above the final retained set.

| driver (AMF) | heap peak − input | retained after | VmHWM |
|---|---|---|---|
| sequential | 4.870 GB | 4.549 GB | 4.93 GB |
| parallel, 1 thread | 5.425 GB | 4.387 GB | 6.07 GB |
| parallel, 2 threads | 6.953 GB | 4.387 GB | 6.69 GB |
| parallel, 4 threads | 7.675 GB | 4.387 GB | 7.38 GB |

The parallel excess over sequential (2.8 GB at 4 threads) is the frontier
of waiting task blocks (held **square** by the parallel driver), one
workspace per worker (`row_map`, `build_seen`, an `n_snodes` slot
vector, a frontal pool), and the shared slot vectors. The largest
antichain of square blocks over the whole supernode tree is 6.03 GB: a
schedule-free bound, but 2× the observed frontier.

AMD, sequential: 9.364 GB measured vs old prediction 8.303 GB.

## Three allocation changes

1. **The parallel driver holds packed blocks.** It kept square blocks
   only because the packing change (decisions 2026-11-12) was scoped to
   the sequential drivers; `factor_one_supernode` is generic over the
   layout, and the factors are unchanged (parity tests). Primal2 AMF, 4
   threads: heap peak 7.63 → 6.12 GB (−20%), wall 69 → 62 s.
2. **The contribution pool recycles only a close fit** (revised after
   the PR #251 review; see the end of this note). Each consumed
   child's buffer goes to a single-slot pool, and `extract_contrib`
   reused it with `clear()` + `reserve(len)`. A larger buffer kept its
   capacity, pinning the excess until the block's parent assembled it;
   a smaller one was reallocated (moved, so nothing saved) and doubled.
   This made AMD sequential 0.5% above a model that ignored capacity,
   and in the parallel driver it has no schedule-free bound below twice
   the largest block. Now a buffer is reused only when
   `len ≤ capacity ≤ len + max(len/8, 512)`; otherwise the block is
   allocated exactly. Values are unchanged; every block's capacity is at
   most `recycle_cap_limit(len)`.
3. **An unfit pooled buffer is freed before the new block is allocated.**
   As the `match` scrutinee's temporary it used to be dropped after
   `Vec::with_capacity`, so both were live: the AMD sequential timeline
   showed a ~0.35 GB spike of ~0.25 s at every top-chain front. Freeing
   it first: AMD sequential 8.906 → 8.555 GB.

The model also counts the dense kernel's pooled panel scratch
(`3 · nrow · 64` pack entries for the largest front so far, ~22 MB on
AMD).

## The model (`src/numeric/memory_model.rs`)

Sequential (`peak_bytes`) = `analysis_bytes` (exact, by capacity) +
`workspace_bytes` (input-proportional formula) + `numeric_peak_bytes`, a
walk of the postorder that mirrors the driver: the frontal pool grows by
`grow_pooled_front`'s rule, children's blocks pass through the pool at
assembly, each block's capacity follows the recycle rule, and each node
adds `nrow·ncol` values plus its index data. Largest total wins.

Parallel (`peak_bytes_parallel(threads)`) = analysis + workspace + the
whole stored factor + shared vectors + **frontier** (maximum-weight
antichain of task-root blocks over the task tree, at each block's
largest possible capacity) + `parallel_running_bytes[threads]`: for
`threads + 1` running tasks (the workers and the caller), each worker's
fixed workspace plus the `threads + 1` largest per-task frontal pools
and in-task block peaks. A worker's pool only grows to the largest front
of the tasks it ran, so distinct workers are bounded by distinct tasks.

## Results (heap peak above input)

Final code (GB; measured is the heap peak above the input):

| case | old `peak_bytes` | new prediction | measured | new / measured | wall |
|---|---|---|---|---|---|
| Primal2 AMF sequential | 3.530 | 4.924 | 4.862 | 1.01 | 98 s |
| Primal2 AMD sequential | 8.303 | ~8.58 | 8.555 | 1.003 | 978 s |
| Primal2 AMF parallel, 4 t | 3.530 | 10.15 | 5.894 | 1.72 | 60 s |
| Primal2 AMD parallel, 4 t | 8.303 | 34.87 | 11.74 | 2.97 | 429 s |

Before the allocation changes, the same runs measured 4.870 / 9.364 /
7.63 / >12.27 GB, with AMF parallel at 69 s. The AMD sequential
prediction is the `Solver`-path figure without the scratch term
(8.557) plus that term's 0.023 GB, from `predict` mode. The `Solver`'s
analysis is 0.12 GB smaller than the default-params one `predict` builds.

| small cases (`tests/issue249_memory_model.rs`) | old | seq | par |
|---|---|---|---|
| grid120 AMD | 0.34× | 1.09 | 1.21 |
| grid120 METIS | 0.32× | 1.08 | 1.20 |
| KKT 70 AMF (15 delays) | 0.30× | 1.06 | 1.25 |

The sequential figure is tight. The parallel one is a bound and is loose
where many tasks have large fronts: on AMD each of five tasks carries a
frontal pool of up to 3.4 GB. Refusing on it would reject jobs that fit,
so a parallel `Solver` whose budget admits `peak_bytes` but not
`peak_bytes_parallel` runs that factorization on the sequential driver
(bit-exact) and reports it via `ran_sequential_for_budget()`.

## Limits

- Delayed pivots move columns into larger parent fronts and are not
  modelled. The KKT test case has 15 delays and stays bounded, but a
  delay-heavy factorization can exceed the prediction.
- The parallel bound assumes at most `threads + 1` tasks in progress; a
  task stolen inside a wide front's intra-front loop (issue #102's
  fallback workspace) can add one more.
- `workspace_bytes` assumes `Solver` use and a lower-triangle input with
  a full diagonal (it recovers the input's stored entries from the
  permuted pattern).
- VmHWM also counts allocator overhead: with 4 threads glibc's arenas
  added ~0.7–0.9 GB over the heap counter on Primal2 AMF (VmHWM 6.8–7.0
  GB vs 5.9 + 0.18 input).
- The AMD sequential margin is thin (1.003×). Anything the walk misses
  that is not in a pooled or stored buffer would show up there first.

## Follow-ups (not done)

- The parallel driver's real footprint, not just its bound: release a
  worker's frontal pool after a task whose fronts the next tasks will not
  need, or schedule big-front tasks so fewer pools grow.
- BK `perm` / `perm_inv` are `nrow` long but only the first `nelim`
  entries move (0.30 GB on Primal2).
- `NodeFactors` is 352 B per supernode (0.25 GB on Primal2).

## Revision after the PR #251 review

The maintainer measured the close-fit rule (slack 512 entries) at 1.164×
geomean refactor time on the 61 parity KKTs, up to 1.70× (default
`Solver`, 4 cores), and 1.019× with the old reuse restored. The rule is now:
reuse when `len ≤ cap ≤ len + max(len/8, RECYCLE_SMALL)`, grow a smaller
buffer in place when `len ≤ RECYCLE_SMALL`, else free and allocate exactly,
with `RECYCLE_SMALL = 8192` entries (64 KiB). Below 64 KiB this is main's
behaviour, except that a pooled buffer more than 64 KiB larger than the
block is freed. `recycled_block_cap` holds the rule for both the kernel
and the model, and a unit test pins the kernel's capacities to it.

Measured here (not the reviewer's machine):
- allocations per warm refactor, summed over the 61: 18 109 for main, the
  PR and the fix alike, on the sequential and the default driver. The
  close-fit rule never turned a reuse into a malloc on this set.
- sequential driver, 8 alternating rounds: PR/main 0.982 geomean, fix/main
  1.001; no allocation difference, and per-matrix ratios move 12–19%
  between runs (ACOPP30_0001 fix/main 1.19 in one run, 1.00 in the next).
- default `Solver`: on the smallest KKTs a refactor costs ~30 µs, of which
  the factorization is ~1.5 µs (sequential timing); the rest is thread-pool
  dispatch, whose jitter moved single-matrix ratios 0.80–1.09.
- slack choice: `1 << 17` raised Primal2 AMF's parallel bound 10.27 →
  16.97 GB; `1 << 14` 10.84; `1 << 13` 10.51 (chosen). On the small test
  matrices the parallel bound loosens from 1.2× to 1.7–2.8× (64 KiB charged
  per live block). Primal2 memory unchanged (AMF seq 4.862, par 5.894 GB).
