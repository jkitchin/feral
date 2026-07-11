# issue #102 — verifying the PR #92 cont5_2_4_l / dirichlet120 stall — 2026-07-01

Verification pass on the issue-#102 claim: PR #92 (`d4502f1`) regresses
`cont5_2_4_l` and `dirichlet120` (POUNCE mittelmann) from converged to a 300 s
timeout spent at **~0 % CPU** inside `factor_frontal_blocked_in_place_with_scratch`.
The issue attributes the regression to #92's **ordering** change (the
`OrderingPreprocess::Auto` fill-verified race) by elimination — the other three
post-0.11.3 commits only add diagnostic getters.

## What can and cannot be reproduced in this container

- **The matrices are not here.** No POUNCE, no `../ripopt`, no `benchmarks/mittelmann/nl/`,
  no dumped KKTs. `cont5_2_4_l` (n=90 600) and `dirichlet120` (n=53 881) cannot be
  factored here without the dumps.
- **The `.sample.txt` profiles the issue cites are not on this branch** either
  (`dev/research/issue-91-cont5_2_4_l-*.sample.txt` do not exist in the repo).
- **Wrong platform / core count.** The stall symptom is a macOS `__psynch_mutexwait`
  on a 14-core M4 Pro. This container is 4-core Linux; `__psynch_mutexwait` is a
  macOS pthread primitive and the parallel contention profile is core-count
  sensitive. A 0 %-CPU parallel stall is not expected to reproduce identically here.

So this is a **code-level verification** of the causal claim, not an end-to-end
repro. The end-to-end repro must run on the reporter's machine with the dumps.

## Finding: #92 changed TWO things, not one — and the issue omits the second

`git show --stat d4502f1` — PR #92 touches `src/dense/factor.rs` as well as
`src/symbolic/mod.rs`. The `factor.rs` change is **Lever B**:

```
-pub const INTRAFRONT_MIN_AREA: usize = 256 * 256;   // 65536  (v0.11.3 / pre-#92)
+pub const INTRAFRONT_MIN_AREA: usize = 256 * 128;   // 32768  (#92 / main)
```

`INTRAFRONT_MIN_AREA` is the trailing-update area `(nrow−j_start)·n_elim` below
which a dense front's Schur update **stays serial**; at or above it the update is
dispatched to the intra-front parallel `par_chunks_mut` path
(`apply_blocked_schur_panel`, `src/dense/factor.rs:3339`). #92 **halved** the
floor, so every front with trailing area in `[32768, 65536)` that v0.11.3
factored serially now goes onto the parallel path — the exact path the issue's
profile shows stalling.

The issue's bisection ("only #92's ordering change is the cause") is therefore
**under-specified**: it correctly rules out #93/#94/#95, but #92 itself bundles
an ordering change *and* a parallel-scheduling change. Both co-vary between
v0.11.3 (healthy) and `main` (stalled):

| | v0.11.3 (252 % CPU, 8.1 s) | `main` (0 % CPU, timeout) |
|---|---|---|
| `OrderingPreprocess::Auto` | trusts `pick_ordering_preprocess` (LdltCompress when predicate fires) | fill-verified race (may fall back to None) |
| `INTRAFRONT_MIN_AREA` | 65536 | **32768** |

## Why the 0 %-CPU symptom points at scheduling, not front size

If #92's *ordering* alone selected a larger/denser front, the factorization would
be **slower at high CPU** (more serial arithmetic, or a well-parallelized bigger
front) — not **0 % CPU**. A 0 %-CPU stall with workers parked in
`__psynch_mutexwait` and the main thread on `LockLatch::wait_and_reset` is a
**synchronization pathology**, and the only thing #92 changed on the
synchronization path is `INTRAFRONT_MIN_AREA`.

Mechanism (verified in code):

- Tree-level parallelism dispatches supernodes via `rayon::scope` + `scope.spawn`
  (`src/numeric/factorize.rs:2092`, `2171`). `factor_one_supernode` runs on a
  rayon worker.
- Inside it, `factor_frontal_blocked_in_place_with_scratch` → `apply_blocked_schur_panel`
  calls `par_chunks_mut` (`factor.rs:3349`) — **nested** rayon parallelism.
- Lowering the floor turns medium fronts that were serial leaves into nested
  parallel sections. On a saturated pool the outer spawns occupy workers while
  each inner `par_chunks_mut` needs help that cannot arrive; the main thread
  (calling from outside the pool via `in_worker_cold`) parks on the completion
  latch. The macOS `__psynch_mutexwait` samples are rayon's injector/job-queue
  lock (or the allocator) under this nested-contention shape — feral's own
  parallel path holds **no explicit mutex** (`apply_blocked_schur_panel` is a
  lock-free `split_at_mut` + `par_chunks_mut`, no allocation).

This is consistent with `dev/research/issue-91-parallel-dense-front-2026-06-30.md`:
the intra-front path already had a ~40 % serial floor and "scales only ~2×." A
front shape that pushes fork-join overhead past the work it parallelizes can tip
"~2×" into "net-negative / stall," and the lower floor admits exactly such fronts.

## The one-line experiment that isolates the two hypotheses

`INTRAFRONT_MIN_AREA` has a **byte-exact env override** (`factor.rs:704`):

```
FERAL_INTRAFRONT_MIN_AREA=65536   # restore the v0.11.3 floor, ordering untouched
```

On the reporter's M4 with the dumps, re-run `main`:

1. `FERAL_INTRAFRONT_MIN_AREA=65536 pounce cont5_2_4_l.nl` — if it **solves**, the
   regression is Lever B (scheduling), *not* the ordering, and the issue's
   diagnosis needs correcting.
2. If it still stalls, the ordering change is implicated and the next step is the
   ordering-comparison harness (factor the dumped KKT under None vs LdltCompress,
   compare front dims + intrafront dispatch counts).

This override is pure scheduling (per-column reduction is thread-independent), so
(1) changes nothing numerically — a clean bisection of the two bundled changes.

## Suggested feral-level reproducer (needs the dumps)

feral already loads symmetric MatrixMarket KKTs (`src/io/mtx.rs::read_mtx`). A
turnkey harness would take a dumped `cont5_2_4_l.mtx` and factor it four ways —
{None, LdltCompress} × {INTRAFRONT 32768, 65536} — reporting per-front
`(nrow, n_elim, area, serial|parallel)` and wall time, plus the resolved ordering.
That directly answers "which of the two #92 changes stalls, and on which front."
Not built here (no dump to validate against); flagged as the concrete next step.

## Status

- Verified (code): #92 bundles an ordering change **and** an `INTRAFRONT_MIN_AREA`
  halving; the issue's causal claim considers only the former.
- Verified (mechanism): the 0 %-CPU symptom is a synchronization stall on the
  nested-parallel dense-front path, which is what the halved floor feeds.
- Not verified (no matrices / wrong platform): the end-to-end stall itself, and
  which of the two changes actually trips it. Isolable on the reporter's machine
  with `FERAL_INTRAFRONT_MIN_AREA=65536`.
