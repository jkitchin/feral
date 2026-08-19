# Issue #175: the tree-parallel solve gate has no per-call-overhead term

Session 2026-08-19-03. Issue: jkitchin/feral#175 — the tree-parallel CB
solve (#131 Gap A) is a net 15% loss on the wide-sparse Mittelmann KKT
`NARX_CFy` under POUNCE (Apple M-series, 14 cores), and
`CbTaskPlan::worthwhile` accepts it because every term of that gate is a
property of the tree's *shape*.

## What the reporter measured

100 IPM iterations, feral 0.15.1, release, 14 cores:

| config | seconds | invol. ctx switches |
|---|---:|---:|
| baseline (all parallel) | 49.41 | 4,523,354 |
| tree-parallel solve serialized (`FERAL_CB_THRESH`) | 42.06 | 1,483,565 |
| ⤷ + factorization task graph serialized | 39.84 | 71,769 |
| `FERAL_PARALLEL=0` | 39.72 | 35,532 |

The solve is 7.35 s of 49.41 (15%) and ~3.0M of the context switches.
Full solves: 12.7% faster per iteration with both gates raised, sys time
628 s → 3.9 s.

Two things follow that the issue does not state explicitly:

1. `FERAL_CB_THRESH` does **not** choose the solve *core*. Since #177
   that is `cb_core_profitable`, a host-independent predicate that reads
   neither the worker count nor the environment. The env var only
   coarsens the task plan, and a huge value collapses it to one task, so
   `worthwhile` turns false and the **same** CB core runs serially.
   The whole 7.35 s is therefore scheduling overhead: *fixing it cannot
   move a single bit of any solve.*
2. Rows 3 and 4 are within noise, so the CB core running serially is
   already as fast as switching parallelism off entirely on this
   problem. Nothing here argues for changing which core runs.

## Mechanism

`cb_run_parallel` (src/numeric/solve.rs) pays, per **front**:

- one `contribs` mutex acquisition per child drained plus one to store
  the front's own contribution block — a single `Mutex<Vec<Option<..>>>`
  shared by every worker, taken inside the per-front loop;
- a `scratch` mutex acquisition per task, and an `AtomicUsize` pending
  decrement per task root;

against `total = Σ nrow·(nelim+1)` of available work. `NARX_CFy` has
45,736 supernodes and a Lagrangian Hessian of only 19,851 nonzeros: tiny
fronts, so the per-front lock traffic is O(45k) acquisitions per solve
from 14 threads — the 3M involuntary context switches. An IPM host runs
several refined solves per factorization, so this fixed per-front cost
is paid tens of thousands of times per solve run.

The gate's three terms — `fwd_seeds ≥ 2`, `total ≥ MIN_TOTAL_COST`,
`max_local < 0.7·total` — have no term for it. `MIN_TOTAL_COST` is a
floor on *total* work; overhead scales with the *number of fronts*, so a
tree can clear the floor and still spend most of the solve
synchronizing. Wide-and-thin is exactly the shape that looks most
parallel-friendly to the current gate and loses.

## Local calibration (x86_64 4-core container)

Harness: `issue175_cb_gate_calibration`, an `#[ignore]`d test in
src/numeric/solve.rs. It times the *pooled* CB core
(`CbSolveWorkspace::solve_into`) serial vs tree-parallel — the exact
choice `worthwhile` makes — best of 30, arms interleaved, under rayon
pools of 1/2/4/8 workers.

Fixture family `narx_proxy(blocks, steps, width)`: `blocks` independent
dynamic-system KKT blocks (near-empty Hessian, one multiplier per step,
one light coupling row joining them). `width` sets the frontal size and
therefore the work each front does per unit of synchronization. Poisson
grids are the known parallel winners.

par/ser, two independent runs (<1 = tree-parallel wins):

| fixture | total/n_nodes | run 1 (w=2,4,8) | run 2 (w=2,4,8) | geo. mean |
|---|---:|---|---|---:|
| narx_w1 | 25 | 1.07 1.10 1.42 | 0.91 0.98 1.04 | 1.08 |
| narx_w2 † | 28 | 0.97 1.08 1.04 | 0.94 1.14 0.98 | 1.02 |
| narx_w4 | 53 | 1.01 1.11 0.93 | 0.89 1.02 0.95 | 0.98 |
| narx_w6 | 74 | 1.15 0.95 0.88 | 0.74 0.88 0.87 | 0.91 |
| narx_w8 | 103 | 0.74 0.94 0.70 | 0.81 0.84 0.83 | 0.81 |
| poisson_96 † | 202 | 0.70 0.76 0.70 | 0.74 0.80 0.69 | 0.73 |
| poisson_160 | 235 | 0.74 0.78 0.73 | 0.80 0.75 0.68 | 0.75 |
| narx_w3 | 305 | 0.87 0.64 0.64 | 0.63 0.54 0.51 | 0.63 |

† **Not reachable in production.** The harness force-sets
`ws.plan.worthwhile` so it can time both arms on any fixture, but two of
the eight fixtures are rejected by an earlier term than the one being
calibrated: `narx_w2` (total 958,763) and `poisson_96` (total 351,544)
are both under `MIN_TOTAL_COST` = 1e6, so `shape_ok` is already false and
no per-front floor can change their fate. Measured directly on this
branch:

    fixture       nodes      total   per_front  shape_ok  worthwhile
    poisson_96     1736     351544         202     false       false
    poisson_160    4633    1090303         235      true        true
    narx_w1       47228    1208092          25      true       false
    narx_w2       33138     958763          28     false       false
    narx_w3        6024    1840981         305      true        true
    narx_w4       21595    1155171          53      true       false
    narx_w6       15384    1150719          74      true        true
    narx_w8       11880    1229411         103      true        true

The break-even is therefore set from the six reachable points — 25, 53,
74, 103, 235, 305 — which are still monotone and still bracket it
between 53 and 74. Dropping the two unreachable rows changes nothing
about the constant; they are reported because they were measured, not
because they carry evidence.

Monotone in work per front, with break-even between 53 and 74 cost
units. `narx_w1` (n = 96,001, 47,228 supernodes, total 1.21M, 22 seeds,
25 units/front) is the local analogue of `NARX_CFy` (45,736 supernodes,
11 seeds): it passes every term of the current gate and never pays.

The local losses (up to 1.42×) are milder than the reported 15%
end-to-end because this container has 4 cores; the dominant cost is
contention on one shared mutex, which worsens with worker count.

## Fix

Add the missing term to the **scheduling** gate only:

    total >= MIN_COST_PER_FRONT * n_nodes,  MIN_COST_PER_FRONT = 64

i.e. a front must average at least 64 `nrow·(nelim+1)` units — about an
8×8 front — before the per-front synchronization is worth paying. 64
sits at the measured break-even: reachable fixtures at or below 53 are a
wash or a loss (geo. mean 0.98–1.08), reachable fixtures at or above 74
pay (0.63–0.91).
`NARX_CFy`, at ~30 units/front, is rejected; every measured winner keeps
a ≥1.1× margin over the floor.

Scope and blast radius:

- The term goes in `CbTaskPlan::worthwhile`, which chooses
  `cb_run_parallel` vs `cb_run_serial` — two byte-identical cores. No
  factor's arithmetic changes.
- `cb_core_profitable` (CB core vs shared-vector core, #177) keeps the
  unchanged three-term shape rule, so *which* core runs is unchanged on
  every matrix. The two predicates are now explicitly split into a
  shared shape half and a scheduling-only overhead half, and
  `cb_core_profitable_matches_the_plan_gate` pins the shape half.
- No new environment knob. `FERAL_CB_THRESH` already serializes the
  plan for per-machine experiments, and #175's sibling issue is about
  env parsing being silent.

## Rejected: the per-seed floor the issue suggests

`total / fwd_seeds.len()` below a floor does not separate the data.
`poisson_160` at 8 workers is 1.09M over 50 seeds = 21.8k per seed and
wins 0.73; `NARX_CFy` is ~1.4M over 11 seeds ≈ 127k per seed and loses.
A per-seed floor high enough to reject `NARX_CFy` would reject the
winners first. Work per *front* is the discriminator, which is also what
the mechanism predicts: the lock traffic is per front, not per task.

## Not addressed here

- The factorization task graph is a further 2.2 s (4.5%) on the same
  problem. That gate (`PAR_TASK_MIN_FLOPS`, issue #148) already has a
  per-task work floor — the right *shape* of term — and only its
  calibration is in question; out of scope for this issue.
- The per-front `contribs` mutex could be removed outright (each slot is
  written by exactly one task and drained by exactly one parent, ordered
  by the pending counters), which would shrink the overhead this gate
  now avoids. That is an optimization, not a gate fix, and needs its own
  `unsafe` safety argument.
