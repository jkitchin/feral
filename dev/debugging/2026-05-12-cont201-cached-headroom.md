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
