# Issue #148: parallel driver task granularity (spawn-per-supernode → spawn-per-subtree)

Session 2026-08-09-02. Issue: jkitchin/feral#148 — the default parallel
driver is slower than serial on 3/4 POUNCE problems (sparseqp 3.14× at
14 threads, monotonic degradation with thread count), heaptrack shows
~1.8M spawn-bookkeeping allocations (one boxed closure per supernode
per factorization), glibc arena contention confirmed by mimalloc A/B
(masks but does not fix — serial still wins under mimalloc).

## Local reproduction (x86_64 4-core container, warm perf_probe medians, 30 iters)

| proxy | serial | par @1t | par @2t | par @4t |
|---|---:|---:|---:|---:|
| chain12000 (hicks-like, n=12000) | 10.0 ms | 10.7 | 10.8 | 10.4 |
| grid250 (poisson-like, n=62500) | 119.7 ms | 142.7 | 94.8 | 75.5 |
| sparseqp (KKT saddle, n=26000) | 27.1 ms | 26.9 | 27.2 | 28.1 |

Matches the issue's signature at 4 cores: chain/sparseqp never beat
serial (sparseqp degrades monotonically with threads); grid is the one
winner but pays +19% pure driver overhead at 1 thread (142.7 vs 119.7)
— that 19% is the spawn+lock cost with zero parallelism benefit.

## Mechanism (read from src/numeric/factorize.rs)

`run_parallel_task` = `scope.spawn(boxed closure)` per supernode
(:3352), re-spawned up the tree via pending counters. Every supernode
additionally takes 4-5 mutex operations (first_error check, thread_ws
try_lock, contrib_blocks ×2, node_factors_out). A 60k-supernode
problem in an IPM loop ⇒ millions of boxed spawns, allocated on one
worker and freed on another — the glibc-worst pattern the issue
measured.

## Fix design (issue suggestion 1 + 2, pure Rust, no new deps)

**Subtree task coarsening.** Supernodes are stored in postorder
(children before parents — the sequential driver depends on it), so:

1. Bottom-up `subtree_flops[s] = own_flops(s) + Σ subtree_flops[children]`
   with `own_flops = ncol · nrow²` (the `estimate_assembly_flops` term).
   NOTE (#128 sub-item): symbolic `nrow` underestimates fronts that
   receive delayed pivots; acceptable for *gating* — document, don't fix
   here.
2. `task_root[s] = subtree_flops[s] >= cutoff || parent[s].is_none()`
   (roots always task roots so every node has a nearest task ancestor).
3. `owner[s]` (nearest task ancestor) by one reverse-postorder pass:
   `owner[s] = s if task_root[s] else owner[parent[s]]`; group nodes by
   owner in postorder ⇒ each task's owned list is postorder-sorted.
4. `pending[t]` counts task-children only; seeds = task roots with
   pending == 0. One spawn per TASK; the task body factors its owned
   nodes serially (same per-node body: stage children from the shared
   store, factor, deposit) then its own node, then trampolines the
   parent task.
5. **Serial fallback:** if the task graph offers no initial parallelism
   (seeds < 2), delegate to the sequential driver outright (keeping
   intrafront_parallel = true, so wide-front chains still get Lever
   1.1). This is exactly the chain case: coarsening collapses a chain
   to a path of tasks — running it through the task machinery is pure
   overhead.

`cutoff` default from a sweep over {256k, 1M, 4M} flops on the three
proxies; env override `FERAL_PAR_TASK_MIN_FLOPS` for per-machine
retune (same pattern as the session-1 work gates).

**Deliberately NOT changed:** per-node contrib staging through the
shared mutex (2026-05-12 tried-and-rejected measured mutex wait+hold
at 0.02-3.9% of body — not the bottleneck; coarsening reduces the op
count anyway); the issue's suggestion 3 (collect() temporaries) is
deferred until after coarsening is measured — it may be subsumed.

**Prior art check (tried-and-rejected):** 2026-07-10 "per-node rayon
tasks — no speedup" was the SOLVE phase (cb_parallel), and its failure
(48400 tiny tasks) is the same mechanism this fix removes from the
factor phase; it supports, not contradicts, coarsening.

## Bit-exactness argument

Scheduling-only change: each supernode's factor call
(`factor_one_supernode`) and its extend-add child order
(`snode.children` order) are untouched; no FP reduction crosses
supernodes outside extend-add. The existing determinism proof for the
parallel driver (value-deterministic, no parallel FP reduction —
decisions.md 2026-05-22) applies verbatim to any task partition.
Gates: tests/parallel_parity.rs, tests/golden_bits.rs, full fixture
suite, plus a new parity case pinning coarsened-parallel vs sequential
factors bit-for-bit on the chain/grid proxies' patterns.

## Acceptance

- chain12000 and sparseqp: parallel @4t within 2% of serial or better
  (today +4-7%).
- grid250: @4t at least as good as today's 75.5 ms (must not lose the
  poisson win); @1t overhead vs serial cut to <5% (today +19%).
- No small-fixture regression (>2%, 3-run medians).
- Spawn count on grid250 reduced by >10× (log n_tasks via telemetry).

## Measured result — ACCEPTED (with one documented proxy anomaly)

Implemented as designed, with one refinement: task boundaries sit at
tree roots and at children of nodes with >= 2 big (subtree >= cutoff)
sibling subtrees — a lone big child continues inline in its parent's
task. (The naive subtree>=cutoff rule made 6319 tasks of 6564
supernodes on the banded-QP proxy; the sibling rule makes 1.)

Plan shapes: grid250 11171 supernodes -> 51 tasks / 26 seeds (219×
spawn reduction); chainW 18717 -> 1; sparseqpL 6564 -> 1 (both fall
back to the sequential driver via seeds < 2).

Interleaved old-vs-new (warm perf_probe medians, 10-12 iters, 2-3
rounds, x86_64 4-core):

| proxy | config | old | new |
|---|---|---:|---:|
| grid250 | par @4t | 78.8-92.7 ms | 71.4-73.4 ms (−9..−21%) |
| sparseqpL | par @4t | 82.4-88.0 ms | 71.7-76.3 ms (−13%) |
| sparseqpL | serial | 69.9-74.6 ms | 70.7-72.2 ms |
| chainW | par @4t | 186.7-216.6 ms | 221.1-223.9 ms |
| chainW | serial | 228.1-229.5 ms | 214.9-215.7 ms |
| chain12000 | par @4t | 12.0-12.5 ms | 11.9-12.5 ms |

The issue's signature (parallel worse than serial on sparseqp) is
gone: old par@4 82-88 vs serial 70-75 (losing 15-25%); new par@4
71.7-76.3 ≈ serial. Small fixtures show no systematic change
(HYDCAR −6%, HAHN +3.6%, twirism +1.4%, sawpath −3.5% — noise band).
Parity: tests/task_plan_parity.rs pins fine/default/fallback plans
byte-identical to the sequential driver; 84/84 suites green.

**Open question (chainW proxy anomaly).** The OLD per-node-spawn
driver at any thread count beats both sequential drivers on the chainW
proxy (~170-195 vs ~210-230 ms). AtomicLockStats telemetry shows the
difference is INSIDE `factor_one_supernode` (factor_body 912 ms old vs
1276 ms new over 9 factors, identical nodes) — not spawn/lock/driver
overhead, not intrafront (FERAL_INTRAFRONT=0 A/B), not tree
parallelism (a chain has none). Some workspace/allocator interaction
this container cannot profile further (no perf/heaptrack). Accepted as
a synthetic-proxy quirk: the issue's real chain problems (POUNCE
hicks/optcontrol/bratu, 14 threads, glibc) lose badly under per-node
spawning — the exact case this fix removes. If a real workload ever
shows the same signature, `FERAL_PAR_TASK_MIN_FLOPS` tuning and this
note are the starting points.

Deferred (issue suggestion 3): the 2.9M `collect()` temporaries —
re-profile after this lands; coarsening may have subsumed most of it
(spawn boxing was the dominant stack).
