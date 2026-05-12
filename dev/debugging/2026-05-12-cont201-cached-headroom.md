# cont-201 cached-symbolic parallel headroom (T=4)

**Date:** 2026-05-12
**Branch:** main
**Driver:** `factorize_multifrontal_parallel_with_workspace`
**Test:** `solver_parallel_lock_breakdown` (#[ignore], `src/numeric/solver.rs`)

## TL;DR

Lock contention is **not** the limiting factor for the rayon parallel
driver. On the four-matrix sample (T=4) the global contribution-block
and node-factors mutexes account for **<3.5% of aggregate body time**
in every case. cont-201's previously-reported 30%-of-ceiling speedup
in single-shot timing was mostly the **sequential symbolic factorize**
(157 ms of a 214 ms wall). Once `Solver`'s pattern-fingerprint cache
kicks in (the production / pounce-IPM regime), cont-201 wall drops
214 → 56 ms, body_frac jumps 0.15× → 0.55×, and the remaining
headroom is roughly **1.5× inside the rayon::scope itself** —
parallel utilization 68.5%, not lock waits.

## Method

Added `AtomicLockStats` telemetry to `NumericParams` (`src/numeric/factorize.rs`):

- six lock-wait/hold + body-time + task-count counters
- eight per-phase wall-time counters wrapping the sequential
  prologue/epilogue inside the parallel driver (scaling, permute,
  symmetric_pattern, tree_setup, thread_ws, leaves, the rayon::scope,
  collect)

The diagnostic test reads the same four matrices used in the prior
threadcount sweep, runs the **cold** factor (pays symbolic cost),
zeroes the 14 atomics, runs a **cached** factor on the same `Solver`,
and snapshots stats from the second call only.

## Data (T=4, M3 Max)

| matrix    | cold ms | cached ms | scope ms | body_agg ms | body/T ms | body_frac | non_loop ms | contrib wait+hold | nf wait+hold |
| --------- | ------: | --------: | -------: | ----------: | --------: | --------: | ----------: | ----------------: | -----------: |
| bcsstk38  |    27.4 |    15.0   |     8.0  |       14.3  |      3.6  |    0.24×  |        6.6  | 0.221 + 0.025     | 0.019 + 0.012 |
| bratu3d   |   689.4 |   618.2   |   614.2  |      956.1  |    239.0  |    0.39×  |        3.1  | 1.479 + 0.385     | 0.151 + 0.132 |
| c-big     | 268078  | 262134    | 262011   |    273526   |   68381   |    0.26×  |       35.0  | 48.811 + 6.442    | 7.761 + 2.549 |
| cont-201  |   213.7 |    56.2   |    45.1  |      123.5  |     30.9  |    0.55×  |        9.0  | 3.785 + 0.430     | 0.402 + 0.217 |

Definitions:

- `body_ms_agg` — Σ over tasks of (factor_node body time per task), summed across all 4 workers
- `body/T` = `body_ms_agg / 4` — average per-worker useful work
- `body_frac` = `body/T ÷ cached_wall` — fraction of wall a single worker is doing useful work
- `non_loop` = sum of all eight phase timers minus the `scope` timer
- All wait/hold times include both lock acquisition wait and time spent inside the critical section

## Findings

1. **Lock contention is rounding error.** Worst case (c-big) shows
   48.8 + 7.8 = 56.6 ms of total mutex wait across all workers, against
   273526 ms of aggregate body — 0.02%. cont-201 sits at 3.4%.
   Re-engineering the contribution-block store into a lock-free
   structure would buy ≤4% on cont-201 and ≤0.02% on the others.

2. **Symbolic factorize dominated cont-201's "cold" gap.** Single-shot
   wall 214 ms − cached wall 56 ms = 158 ms attributable to
   `symbolic_factorize`. That call is single-threaded and is **cached
   per pattern by `Solver`**, so the pounce/IPM hot path pays it
   exactly once per pattern fingerprint, not once per factor.

3. **Cached cont-201 has real headroom, but inside the rayon::scope,
   not at the lock sites.** Cached wall 56.2 ms breaks down as:
   - non_loop driver overhead 9.0 ms (16% of wall)
     — scaling 3.95 + permute 3.75 + sympat 0.81 + tree/ws/leaves/collect 0.49
   - rayon::scope 45.1 ms (80% of wall)
     — but body_per_T = 30.9 ms, so loop utilization = 30.9/45.1 = **68.5%**

   Critical-path analysis from the prior session gave critical path
   23.5 ms and total work 113.5 ms; the T=4 ideal-loop time is
   max(23.5, 113.5/4) = 28.4 ms. Current loop runs at 45.1 ms — a
   **1.59× gap inside the parallel section** that is not lock waits.
   Best plausible cached wall at T=4 (loop ideal + measured non_loop):
   28.4 + 9.0 = 37.4 ms vs measured 56.2 ms — **1.50× headroom**.

4. **Small-matrix floor is the non_loop driver overhead.** On
   bcsstk38 the non_loop block is 6.6 ms / 15.0 ms = 44% of the cached
   wall. scaling alone is 3.78 ms = 25% of wall. compute_scaling is
   re-run every factor; investigating whether
   `compute_scaling_with_cache` actually hits its cache on the second
   call is the next concrete probe.

5. **bratu3d and c-big are body-time-bound, not driver-bound.**
   non_loop is 0.5% and 0.01% of wall respectively. These matrices
   benefit directly from any per-task speedup; they will not benefit
   from non_loop optimizations.

## What this rules out

- **Mutex/global-store redesign for cont-201.** Even a lock-free
  contribution store eliminates at most 4.2 ms of 56.2 ms (best
  case, T→∞), and would not change the within-scope utilization gap.
  Not worth implementing speculatively.

- **Re-litigating the `N_PAR_MIN=32` gate as the cause of
  small-problem regressions.** bcsstk38 cached wall is 15 ms with the
  parallel driver active; that's already faster than the cold 27 ms
  number. The non_loop overhead (6.6 ms) is the same on the
  sequential driver — it is **not** parallel-driver overhead.

## Next probes (ranked by expected payoff)

A. **cont-201 within-scope utilization.** body_per_T=30.9 ms vs scope
   45.1 ms — the 14.2 ms gap is some mix of work-stealing tail,
   dependency-chain idle (etree depth 28), and the contribution-block
   serialization stall that lock-wait time underestimates because the
   `Mutex<FactorWorkspace>` releases before the rayon task returns.
   Adding a per-task `task_busy_ns` band (wait-for-deps,
   numeric-kernel, post-store) would localize this.

B. **`compute_scaling_with_cache` cache-hit verification.** Add a
   one-shot eprintln on cache miss path, or a hit/miss counter on the
   cache itself. If the cache is missing on the second `factor()`
   call we can fix the cache key. Expected saving on cont-201 cached:
   ~3.95 ms (= 7% of wall, 1.07× on cont-201; larger on bcsstk38).

C. **etree-depth schedule.** cont-201 has etree depth 28 and 11121
   supernodes. The current ready-queue is one global atomic counter
   on `n_pending`; depth-28 chains spawn one task at a time near the
   root. A topological-level scheduler (one rayon::scope per level)
   would not add concurrency that isn't there, but would expose
   whether the within-scope gap is concurrency or scheduling.

## What ships from this investigation

- `AtomicLockStats` + 14 atomic counters added to `NumericParams`
  (opt-in, default `None`; no perf impact when disabled).
- Phase timers wired through
  `factorize_multifrontal_supernodal_parallel` and the per-task body.
- `solver_parallel_lock_breakdown` test extended with cold+cached
  pair and per-phase breakdown.

This is a permanent diagnostic surface, gated behind the
`parallel_telemetry` field. The test stays `#[ignore]`'d to keep
`cargo test` quick.

---

## Iteration 2 (2026-05-12, same session): within-scope localization

Added two more atomic counters to localize the 14.2 ms cached-mode
gap between `body_per_T` and `scope` on cont-201:

- `task_wall_ns` — bracket of the entire `scope.spawn(...)` closure
  body (includes locks + factor_body + per-task control flow). Lets
  us compute `rayon_idle = scope · T − task_wall_agg`, the time the
  rayon pool spent with workers waiting for an eligible task (i.e.
  parallelism unavailable due to etree dependencies, not engineering
  loss).
- `ws_lock_wait_ns` — wait time on the per-worker
  `Mutex<FactorWorkspace>`. Expected near zero (one slot per
  worker); confirms the per-worker workspace design.

### Data (T=4, cached-symbolic, ms)

| matrix    | scope | scope·T capacity | task_wall_agg | rayon_idle | idle % | in_task_locks | ctrl_flow | ws_wait |
| --------- | ----: | ---------------: | ------------: | ---------: | -----: | ------------: | --------: | ------: |
| bcsstk38  |   7.6 |             30.6 |          14.7 |       15.9 |    52% |          0.20 |      0.39 |   0.013 |
| bratu3d   |   648 |             2591 |          1017 |       1574 |    61% |          2.26 |      2.45 |   0.110 |
| c-big     | 279432 |          1117728 |        292151 |     825577 |    74% |         76.13 |     68.32 |   2.506 |
| cont-201  |  48.6 |            194.5 |         145.3 |       49.2 |    25% |          6.73 |      5.98 |   0.231 |

Definitions:

- `rayon_idle = scope · T − task_wall_agg` — aggregate worker-idle
  time inside the rayon::scope. Bound on parallelism that the
  current etree-ordered ready-queue cannot fill.
- `in_task_locks_agg = contrib_wait + contrib_hold + nf_wait + nf_hold + ws_wait`
- `ctrl_flow_agg = task_wall_agg − body_agg − in_task_locks` — per-task
  closure overhead (n_tasks bump, fast-exit check, snode lookup,
  pending decrement, recursive spawn call).

### Findings

**1. cont-201's residual headroom is etree-topology bound.** Of the
14.2 ms cached-mode gap between body_per_T (30.9 ms) and scope
(45.1 ms in iter 1 / 48.6 ms in iter 2), the breakdown at T=4 is:

- rayon_idle: 49.2 / 4 = **12.3 ms/worker** (dep-chain wait)
- in_task_locks: 6.73 / 4 = 1.68 ms/worker
- ctrl_flow: 5.98 / 4 = 1.50 ms/worker

The dominant share (~78%) is rayon idle, i.e. workers genuinely have
no eligible task. cont-201's etree has depth 28 and 11121 supernodes,
so near the root the dependency chain has insufficient breadth to
keep 4 workers busy. **A topological-level scheduler will not fix
this** — rayon's work-stealing already drains the ready-queue
greedily. The missing parallelism isn't there.

To recover this 12 ms/worker, the only axis available is
**within-supernode parallelism** (panel-BK or threaded dense kernels
inside `factor_one_supernode`), which is what MUMPS' threaded BLAS
+ SPRAL's panel scheduler give them.

**2. c-big is essentially sequential at T=4.** Of the four matrices,
c-big shows the parallel driver buying a 1.04× speedup at T=4
(body_agg 292s vs wall 280s). 74% of worker capacity is rayon idle.
This is consistent with c-big's structure funneling almost all the
work through a thin critical path of large supernodes near the root.
Critical-path analysis (not yet run on c-big) would confirm.

Same conclusion as (1) applies: the parallelism deficit cannot be
recovered by changing the inter-task scheduler. It needs
within-supernode parallelism. Until then, **c-big is not a profitable
target for assembly-tree parallelism**.

**3. Per-worker workspace mutex is confirmed uncontended.** Worst
case is c-big at 2.5 ms across 117898 tasks over 4 workers (= 0.02
ms/worker). Validates the `thread_ws[thread_idx]` design.

**4. Small matrices (bcsstk38) waste worker capacity on idle.**
52% rayon idle on bcsstk38 reflects the etree running out of breadth
quickly. Combined with iter 1's 6.6 ms non_loop floor, this argues
for `N_PAR_MIN`-style gating to fall back to sequential below a
size threshold — already in place (parallel driver self-gates).

### Conclusions

The investigation that started as "cont-201 30% of ceiling" has
walked through:

- **iter 0** (prior session): critical-path analysis → 1.44× of
  4.83× theoretical at T=8 → "must be lock contention or task
  spawn overhead"
- **iter 1** (this session): mutex telemetry + phase breakdown →
  lock contention is rounding error; "cold gap" was symbolic
  factorize; **cached cont-201 wall 56 ms is the production
  number**, with 1.5× residual headroom in the scope.
- **iter 2** (this session): within-scope localization → the 1.5×
  is rayon idle, not locks/control-flow. Etree topology bound.
  Within-supernode parallelism is the only remaining axis.

This closes the cont-201 assembly-tree-parallelism investigation.
Reopen only if a within-supernode parallelism prototype is built.

---

## Iteration 3 (2026-05-12, same session): scaling-cache verification

Iteration 1 noted that cont-201's cached-mode wall spends 3.95 ms in
the `scaling` phase and flagged this as a candidate cache miss
("investigating whether `compute_scaling_with_cache` actually hits
its cache on the second call is the next concrete probe").

Resolution: probe via a tiny diagnostic test
(`solver_scaling_phase_split`, `#[ignore]`) that loads each corpus
matrix and times three components separately — `pick_scaling_strategy`
(O(n) col_ptr scan), `compute_scaling_with_cache(..., cache=None)`,
and the `scaling_pivot_order` gather. Cache-less timing tells us the
upper bound of `compute_scaling` work when the symbolic-stage MC64
cache is absent.

### Data

| matrix    |      n |     nnz | picked        | pick_ms | scale_ms (no cache) | reorder_ms |
| --------- | -----: | ------: | ------------- | ------: | ------------------: | ---------: |
| bcsstk38  |   8032 | 181 746 | InfNorm       |   0.004 |               4.048 |      0.006 |
| bratu3d   | 27 792 |  88 627 | InfNorm       |   0.014 |               0.814 |      0.012 |
| c-big     | 345 241 | 1 343 126 | Mc64Symmetric |   0.216 |            2 302.744 |      0.148 |
| cont-201  | 80 595 | 239 596 | InfNorm       |   0.030 |               4.146 |      0.032 |

### Findings

**1. MC64 cache works as designed.** c-big picks `Mc64Symmetric`.
With `cache = None` the full Hungarian matching runs and takes
**2.3 seconds**. The parallel-driver test (`AtomicLockStats`)
measured `phase_scaling_ns` = 2.43 ms on cached c-big — a **1000×
speedup**, exactly the `scaling_from_cache` O(n) fast path. The
cache hits and delivers the expected win.

**2. cont-201's 3.95 ms is fundamental InfNorm work, not a missed
cache.** cont-201 picks `InfNorm` (Auto's default for non-arrow-KKT
matrices). InfNorm runs up to 10 Knight-Ruiz iterations and depends
on matrix **values**, not pattern. There is no cache to hit across
IPM iterations because the values change every Newton step. The
3.95 ms is roughly 10 iterations × 240k nnz × ~1.6 ns/op,
consistent with the measurement.

**3. bcsstk38's 3.78 ms scaling slice has the same explanation** —
InfNorm on 182k nnz.

### Conclusion

Probe #2 from prior session (verify `compute_scaling_with_cache`
cache hits) is **resolved with no action needed**. The cache is
operating correctly; the per-factor scaling cost is unavoidable
value-dependent work for the matrices that pick InfNorm.

Engineering opportunities ranked:

- **InfNorm SIMD vectorization** — the Knight-Ruiz inner loop is
  abs + max + sqrt + reciprocal, ~1.6 ns/op scalar. SIMD would
  shave maybe 3× off cont-201's 3.95 ms, recovering 2-3 ms of a 56
  ms wall (5%). Low priority unless the IPM hot path becomes
  scaling-bound on other matrices.
- **InfNorm iteration-count instrumentation** — count actual
  Knight-Ruiz iterations per matrix. If cont-201 hits the 10-iter
  cap, that's pathological and worth investigating; if it converges
  in 2-3 iterations, the cost is already minimal. Not worth doing
  speculatively.

This completes the cont-201 cached-symbolic investigation.
