# Plan — a solve-phase performance gate that measures the path hosts use

Issue: #189 (items 1–3). Blocks: #131.
Status: planning. Step 1 scoped from measurement; no bench code written yet.

## Why this is first

#131 asked for a parallel-solve entry point on `Solver`. Two probes
(session 2026-08-20-01) answer that and turn up a harness defect that is
larger than the thing #131 proposed to fix.

`crates/feral-diagnostics/src/bin/probe_solve_reconcile.rs` times three
configurations interleaved within each repetition — `sv` = the serial
shared-vector core the bench uses, `au` = `SolveCore::Auto` serial,
`ap` = `Auto` parallel (the host path) — 11 reps, medians, one process.
Ratios > 1 mean the second configuration is faster.

| matrix | n | core `sv/au` | sched `au/ap` | bench→shipped `sv/ap` |
|---|---:|---:|---:|---:|
| bcsstk38 | 8,032 | 1.02 | 0.90 | 0.91 |
| r05_kkt | 14,842 | 1.65 | 1.15 | 1.90 |
| bratu3d | 27,792 | 1.31 | 0.99 | 1.29 |
| qap15_kkt | 50,880 | 1.37 | 1.05 | 1.45 |
| dirichlet120_kkt | 54,363 | 1.12 | 1.00 | 1.12 |
| cont-201 | 80,595 | 1.45 | 0.98 | 1.42 |
| cont5_late_kkt | 180,900 | 1.88 | 1.01 | 1.90 |
| **geomean, `steps=1`** | | **1.37** | **1.01** | **1.38** |
| **geomean, default depth** | | **1.30** | **0.98** | **1.27** |

Two readings:

- **#131's parallel schedule is a wash** — geomean 1.01/0.98, nothing
  outside 0.85–1.15. That confirms the issue's conclusion.
- **The bench under-reports the shipped solve by geomean 1.27–1.38×,**
  up to 1.90×, because it times a core no host runs. This is a
  correctness defect in the harness, not a coverage gap, and it is why
  the gate comes first: extending the current measurement to more
  matrices would extend the wrong measurement.

### Retracted: the variance argument

An earlier version of this plan argued from same-matrix spreads of up to
2.20× between runs and concluded "we cannot presently tell a 2× solve
regression from scheduler noise." That was an artifact of measuring
configurations in **separate blocks and separate runs**. Interleaved, the
same comparisons are stable to ~1.0–1.15×; on `r05_kkt` the blocked
method was off by 2.4× (reported 0.67, actual 1.65). The solver is
reproducible; the old harness was not. Design consequence in Step 1.

## Code inspection — what the bench measures today

`src/bin/bench.rs`, KKT corpus loop at `:1670-1700`:

```rust
let t1 = Instant::now();
let x = match solve_refined(&matrix, &factors, &rhs) { ... };
let solve_us = t1.elapsed().as_micros();
```

Four properties of that line, all of which the gate has to deal with:

1. **`solve_refined` hardcodes the serial shared-vector core.** It reaches
   `solve_sparse_refined_opts`, which passes `SolveCore::SharedVector`
   (`src/numeric/solve.rs:2077`). The free functions never install a rayon
   pool — `Solver::refine_into` (`src/numeric/solver.rs:1719`) is the only
   place that does. So the bench cannot observe either the CB core or the
   parallel schedule, which is exactly what #131 is about.
2. **Single RHS.** `solve_sparse_many` is never benchmarked, so the lever
   that measures geomean 2.29× is invisible to the harness.
3. **Single shot at large n.** `should_resample` (`:1042`) triggers on
   `mumps_timing.factor_us < 200`, i.e. only on *tiny* matrices. The
   denoise was built for the "tens-of-µs scale" noise diagnosed in session
   2026-04-20-01 (`dev/plans/bench-denoise.md`). Large matrices — the ones
   #189 wants to gate, and the ones I measured 2× spread on — get exactly
   one sample.
4. `solve_us` for resampled matrices is the **median** of 5, while
   `factor_us` is the **min** (`:1080`). Reasonable for the small-matrix
   case it was written for; needs restating for a gate.

Property 3 is the one that inverts the intent: replication is applied
where the matrices are cheap and omitted where they are noisy.

The bucket definitions (`:660-666`) are then:

```rust
if t.max_front < 200 && t.n <= 1_000   { small.push(r); }
if t.max_front < 500 && t.n <= 10_000  { medium.push(r); }
```

with `r` a factor ratio in both cases. `MatrixTiming.solve_us` and
`OracleTiming.solve_us` both already exist, and solve ratios are already
computed for the report at `:493` and `:500` — so plumbing is not the
work. Deciding what to measure is.

## Sequence

The steps are ordered so each one produces evidence the next one needs.
Nothing here sets a numeric target until a measurement exists to set it
from — #189 item 1 says "from a first measurement rather than guessed",
and that constraint is the whole point.

### Step 1 — time the path hosts take, interleaved

The defect above is the first fix, and it is small: the bench must time
`Solver::solve_refined` with a pool installed rather than the free
`solve_refined`, so the reading describes `SolveCore::Auto` — the core
every host gets.

- Switch the KKT-corpus solve timing at `:1670-1700` onto `Solver`.
  Expect the reported solve column to *improve* by geomean ~1.3× on
  large matrices; that is not a speedup, it is the harness starting to
  measure the shipped path, and the CHANGELOG has to say so or the next
  reader will misread the jump as a regression fixed.
- Where two configurations are compared, interleave them within each
  repetition rather than timing one then the other. This — not a larger
  N — is what made the probe reproducible. Blocked best-of-5 carried a
  2.4× error that 11 interleaved reps removed entirely.
- Decouple the resample predicate from matrix size. `should_resample`
  (`:1042`) triggers on `mumps_timing.factor_us < 200`, so replication
  lands on cheap matrices and is omitted on the ones a gate would watch.
  Replicate based on "this row feeds a gate" instead.
- Report **spread** alongside the statistic. Interleaved medians on the
  large matrices hold to ~1.05×, so this is a cheap assertion to make and
  a cheap one to check, not a hedge.
- Exit criterion: a repeated bench run over `tests/data/large/` reports a
  solve-ratio spread we can state, and the solve column reflects `Auto`.

### Step 2 — measure the paths hosts actually take

Add solve timings for the configurations that exist in the wild, since
gating only the serial single-RHS shared-vector path gates something
pounce never calls:

- `Solver::solve_many` at `nrhs` = 1, 13, 32 (the batching + BLAS-3
  path). Interleaved against the looped single-RHS equivalent this
  measures geomean **2.47×**, min 2.20×
  (`probe_batch_interleaved.rs`) — the largest reproducible solve-path
  lever there is, and entirely invisible to the harness today.

`Solver::solve_refined` with a pool moves to Step 1, since it is the fix
for the defect rather than an addition.

Report only. No verdicts yet.

### Step 3 — the large-n bucket, with targets read off Steps 1–2

`n > 10^4`, gating `factor/MUMPS` and `solve/MUMPS`. Targets set from the
distribution the first two steps produce, with the spread from Step 1
folded into the margin.

### Step 4 — the fixture question (#189 item 3, split-out candidate)

`tests/data/large/` is gitignored, so `large_matrix_smoke` SKIPs on CI
and any baseline recorded there is enforced locally only. Two honest
options — enforce at release from a recorded baseline, or make a small
number of matrices fetchable in CI. This is a real decision and should
not be smuggled in with the rest.

## What this plan does not do

- It does not add a `Solver::solve_auto` entry point. The parallel
  schedule measures 1.01/0.98 geomean; there is nothing for that entry
  point to deliver.
- It does not re-fit `cb_core_profitable`. An earlier version of this
  plan held that item open on the grounds that the predicate approves
  `ContribBlock` on a matrix running at 0.67. That 0.67 was the blocked-
  measurement artifact; interleaved, `r05_kkt` runs at 1.65. The gate is
  in fact correct on all seven matrices — `ContribBlock` on exactly the
  six where the core wins serially (1.12–1.88), `SharedVector` on
  `bcsstk38` where the two cores are a wash (1.02). There is no evidence
  it needs re-fitting, and since it picks a *core* any change to it is
  #177-visible arithmetic. Closed, not deferred.
- It does not touch `BLAS3_NRHS_THRESHOLD` (#189 item 4). That lever
  measures 1.2–2.3× but is unreachable until a host batches, and it is a
  last-bits change needing a CHANGELOG entry. It is real, it is just not
  blocking anything.

## Oracle discipline

Per CLAUDE.md: implementation and test oracle must not both be written in
one session without an external source. The oracle here is MUMPS/SSIDS
sidecar timings already in `external_benchmarks/`, which is external. The
probe's numbers are a cross-check on plausibility, not the oracle.
