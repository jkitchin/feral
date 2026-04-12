# Phase 2 — Optimized, Scaled, and Correct at Production Size

## Ultraplan (2026-04-12)

## Why this plan exists

Phase 1 is complete. Feral matches canonical Fortran MUMPS on 99.97% of
the 153,151 matrix KKT corpus, with zero failures on the Definitive
subset of a 3-oracle consensus framework
(`dev/sessions/2026-04-12-01.md`). The Phase 1 retrospective
(`dev/phase1-retrospective.org`) is the authoritative narrative of what
was accomplished, what was learned, and what is *not* yet proven.

The most important unresolved question going into Phase 2 is
**whether feral's sparse multifrontal path scales**. The entire Phase 1
validation corpus has dimension `n ≤ 500` because the benchmark harness
enforces `if mtx.n > 500 { continue; }` — a Phase 1a hold-over that was
never removed. Consequently, the sparse solver that is the main
deliverable of Phase 1b has never been run on a matrix where the dense
path was not also applicable. We do not know whether the symbolic
pipeline's documented `O(n²)` behavior in `column_counts` is a problem
at larger sizes, whether per-supernode vec-allocations in the sparse
solve dominate runtime at scale, or whether feral is within `10×` of
canonical MUMPS or within `1000×`. These are empirical questions with
unknown answers, and they are Phase 2 work.

Phase 2 also inherits a handful of concrete correctness gaps identified
and deliberately deferred in Phase 1b:

- The **ACOPP30 residual gap** — 12 orders of magnitude worse than
  MUMPS on a matrix where the *factorization* agrees. Almost certainly
  missing global MC64-style matching-based scaling
  (`dev/phase1-retrospective.org` §"The ACOPP30 residual gap").
- The **deferred 2×2 inertia fix** — `count_2x2_inertia` uses `a00`
  instead of `trace` in the near-singular branch. The
  mathematically-correct fix regressed 16 matrices against rmumps and
  was reverted (`dev/tried-and-rejected.md`), but was never
  re-evaluated against canonical MUMPS.
- The **88 sparse-only failure matrices** — the bench's
  dense ∩ sparse cross-comparison reports 88 matrices that fail the
  sparse path but pass the dense path. None have been triaged; the
  intended ERRINBAR_0824 triage turned out not to represent this set.

And an inherited methodological lesson: *validation infrastructure is
load-bearing*. Phase 1b's biggest mistake was not building the
multi-oracle consensus framework until the end. Phase 2 should not
repeat that mistake. The corpus expansion and performance-measurement
harness are Phase 2 Step 1, before any optimization, because you
cannot optimize what you cannot measure and you cannot fix what you
cannot reproduce.

The spec (FERAL-PROJECT-SPEC.md §1735) lists Phase 2 items without
ordering. This plan imposes the ordering implied by the Phase 1
lessons: measurement before optimization, correctness before
performance, small before large.

## What Phase 2 is (and what it isn't)

**Phase 2 is:**

- Adding a moderate-scale benchmark corpus (n = 10³ to 10⁴) drawn from
  `ripopt/benchmarks/{large_scale, grid, gas, water}`.
- Building a performance measurement harness that compares feral to
  canonical MUMPS and SSIDS on factor time, solve time, memory, and
  residual quality.
- Fixing the correctness gaps deferred from Phase 1b:
  - Global MC64 scaling (closes the ACOPP30 residual gap)
  - The 2×2 inertia trace fix, re-evaluated against canonical MUMPS
  - Whatever the 88 sparse-only failures turn out to be
- Implementing the pivoting work deliberately deferred from Phase 1:
  - Threshold partial pivoting (TPP) with `u = 0.01`
  - Delayed pivoting (SSIDS-style)
  - A posteriori pivoting (APP) blocked kernel
- Building performance primitives:
  - Blocked dense LDLᵀ with `block_size = 64`
  - SIMD micro-kernel for the Schur complement update
  - Shared-memory parallelism on the assembly tree (Rayon)
  - `ContribPool` transition from LIFO stack to buddy allocator
- Adding METIS ordering alongside AMD (priority for KKT structure)
- A closed-loop validation step where feral is run inside an
  IPOPT-style outer iteration before committing to Phase 3 POUNCE
  integration.

**Phase 2 is NOT:**

- Replacing MUMPS in ripopt (that's Phase 3)
- Distributed MPI factorization (Phase 4)
- GPU offload (Phase 4)
- Scaling to `n > 100K` (Phase 4)
- A rewrite of any Phase 1 correctness work. Phase 1's structural
  fixes (postorder pipeline, threshold consistency, best-iterate
  refinement, `zero_tol = ε`) stay exactly as they are.

**Exit criterion (from FERAL-PROJECT-SPEC.md):** *Within 2× of MUMPS
on the small-frontal KKT set; within 3× on the medium set.* This is
the bar Phase 2 must clear. Everything below is structured to
produce a defensible answer to "how close is feral to MUMPS and
SSIDS?" and to move that answer toward the exit criterion.

## The ordering principle: measurement, then correctness, then performance

Phase 1b's methodological lesson reframed as a Phase 2 ordering
constraint:

1. **Measurement infrastructure first.** The corpus expansion, the
   perf harness, and the baseline comparison numbers come before
   any optimization. If feral is currently 10× slower than MUMPS we
   need to know that before we spend effort on SIMD kernels that
   might give us 2×.

2. **Correctness fixes second.** The deferred Phase 1b gaps are
   closed before performance work starts, because they affect what
   "correct" means for the measurement. You cannot fairly compare a
   solver that gets wrong residuals against one that does not.

3. **Performance optimization third.** With the measurement harness
   and correctness fixes in place, performance work becomes
   scientific: every change is measured against a stable baseline,
   regressions are visible immediately, and the exit criterion
   ("within 2× of MUMPS") becomes testable.

4. **Closed-loop validation fourth.** Static KKT matrices are not
   the same as running inside an IPM. A Phase 2.7 "run feral inside
   an IPOPT-style loop on a small problem set" step catches
   behavior that static testing cannot — cumulative inertia errors,
   refinement convergence across outer iterations, and the subtle
   correctness requirements of the `increase_quality()` interface
   the spec mentions.

## Phase 2.1 — Corpus expansion and measurement infrastructure

**Duration:** 6–10 hours across 2–3 sessions.

### 2.1.1 — Lift the `n > 500` bench filter (30 minutes)

The filter `if mtx.n > 500 { continue; }` in `src/bin/bench.rs:283`
must go. Options:

- **A.** Delete the filter outright. All matrices run through both
  the dense and sparse paths.
- **B.** Split the filter by path. Dense BK becomes impractically slow
  above `n = 2000` or so; sparse should run without a ceiling. Use
  two separate filters: dense `n ≤ 2000`, sparse unlimited.
- **C.** Add a CLI flag `--size-limit` with a default of 2000 (dense)
  and no default (sparse), lettable via env var.

**Recommended: B.** The dense path has a known `O(n³)` cost and
benchmarking it on a `n = 10⁴` problem is 10⁶× slower than on a
`n = 500` problem — worse than useless. The sparse path should see
everything. Document the split in the bench output header.

### 2.1.2 — Sanity check on a single large problem (1–2 hours)

Before building the full collect_kkt extension, verify feral does not
trivially die on `n > 500`. Pick `BratuProblem::new(1000)` from
`ripopt/benchmarks/large_scale/problems.rs`, run it through
`collect_kkt` with a one-off override (hard-code the test problem in a
new `collect_kkt_sanity.rs` binary rather than extending the existing
infrastructure), dump the first iteration's KKT matrix, and run
feral's sparse path on it.

Three outcomes:

| Outcome | Diagnosis | Next step |
|---|---|---|
| Feral factors, inertia matches MUMPS, residual passes | Pipeline scales | Proceed to 2.1.3 |
| Feral factors slowly, inertia matches, residual passes | Known perf gap | Continue; perf is Phase 2.5 |
| Feral fails to factor, or produces garbage | Latent scaling bug | Stop, profile, triage — do not proceed to 2.1.3 until fixed |

This is the minimum-risk calibration run. It takes at most 2 hours and
it tells us whether Phase 2 can even begin on its current schedule or
whether we need to spend a session on a scaling bug first.

### 2.1.3 — Write `collect_kkt_large` (2 hours)

Extend `../ripopt/benchmarks/cutest/collect_kkt.rs` (or create a sibling
`../ripopt/benchmarks/large_scale/collect_kkt_large.rs`) that loops over:

- `ChainedRosenbrock { n }` for `n ∈ {100, 1_000, 10_000}`
- `BratuProblem::new(n)` for `n ∈ {100, 1_000, 10_000}`
- `OptimalControl::new(t)` for `t ∈ {50, 500, 5_000}` (KKT dim ≈ 3t)
- `PoissonControl::new(k)` for `k ∈ {10, 35, 100}` (KKT dim ≈ 3k²)
- `SparseQP { n }` for `n ∈ {1_000, 10_000}`

Each solve runs with `kkt_dump_dir = Some(...)` and writes per-iteration
`.mtx` + `.json` to `data/matrices/kkt/<PROBLEM>_<SIZE>/`. The total
number of new matrices depends on iteration counts — estimate 500 to
2000.

**Cross-repo note.** This binary lives in `ripopt`, not `feral`. It
pulls in `ripopt::NlpProblem` and the `kkt_dump_dir` option from
`ripopt::SolverOptions`. The build, test, and CI run in ripopt's tree.
Feral consumes the output data only.

### 2.1.4 — Run the canonical oracles on the new matrices (30 minutes)

```sh
python3 external_benchmarks/mumps_oracle/run_mumps.py data/matrices/kkt --skip-existing
python3 external_benchmarks/ssids_oracle/run_ssids.py data/matrices/kkt --skip-existing
```

These should both "just work" on the new matrices — no code changes
needed. Watch for ICNTL(14) workspace failures from MUMPS on the larger
problems; if any surface, bump the default workspace multiplier in
`mumps_bench.F`.

### 2.1.5 — Grid benchmark addition (optional, 1–2 hours)

The grid suite in `ripopt/benchmarks/grid/problems.rs` currently has
four cases (IEEE 3/5/14/30 bus). These are small but the ACOPF
structure is numerically difficult (nonconvex, ill-conditioned, bordered
KKT). Add them via the same `collect_kkt_large` binary. Optionally add
the larger PGLib-OPF cases (IEEE 118, 300, etc.) if the user wants
them and ripopt has the MATPOWER-case loading infrastructure.

### 2.1.6 — Gas and water suites (deferred; 3–4 hours if pursued)

The gas and water suites are AMPL `.nl` files. ripopt would need an
ASL (AMPL Solver Library) reader to consume them — unclear if it has
one. If yes, extend `collect_kkt_large` to handle them. If no, skip.
This is explicitly optional for Phase 2; the `large_scale` and `grid`
suites are sufficient for the Phase 2 exit criterion.

### 2.1.7 — Performance measurement harness (2–3 hours)

Extend `src/bin/bench.rs` to:

- Aggregate per-matrix factor and solve timings
- Compute geometric mean factor time per path
- Read the canonical oracles' timings from their sidecar JSONs
- Print a comparison table showing feral / MUMPS / SSIDS ratios
- Group by problem family, so the geometric mean is not dominated by
  many small problems
- Compute a per-matrix "feral factor time / MUMPS factor time" ratio
  and report its distribution (geometric mean, p50, p90, p99, max)
- Identify worst-case matrices by slowdown so they can be triaged
  individually

This harness is the tooling that will tell us whether the exit
criterion ("within 2× of MUMPS on the small-frontal set") is met. Build
it before any optimization work starts.

### 2.1.8 — Baseline report

After 2.1.1 through 2.1.7, run the full pipeline on the expanded corpus
and produce a Phase 2 baseline report. This is what every subsequent
change is measured against. Commit it as
`dev/sessions/phase-2-baseline.md`.

---

## Phase 2.2 — Deferred correctness fixes

**Duration:** 8–16 hours.

### 2.2.1 — Global MC64 scaling (the ACOPP30 fix)

**Estimated effort:** 4–8 hours.

ACOPP30_0000 has feral and canonical MUMPS agreeing on factorization
inertia `(71, 137, 1)` but feral's residual is 3.15e-2 versus MUMPS's
5.0e-14 — 12 orders of magnitude worse. The cause is almost certainly
that MUMPS applies MC64 matching-based scaling across the whole matrix
before factorization, while feral's Knight-Ruiz equilibration is
applied per-frontal inside the dense kernel and cannot propagate the
scaling across frontal boundaries.

**Implementation:**

1. Research note `dev/research/mc64-scaling.md` covering Duff &
   Koster's matching algorithm (cite: `duff2020aptp` in the existing
   references.bib, or add a specific MC64 citation).
2. New module `src/scaling/` with a pure-Rust MC64 implementation. The
   algorithm is a weighted bipartite matching; the literature has
   several O(n²) and O(n²·√n) variants. Start with the simplest
   correct version.
3. Plug the scaling into `symbolic_factorize` so the permuted pattern
   is scaled before supernode detection, and the scaling vector is
   stored in `SymbolicFactorization` for later solve-side unscaling.
4. Update `factorize_multifrontal` and `solve_sparse` to apply and
   undo the scaling around the numeric phase.
5. Regression test on ACOPP30_0000 — residual must drop below 1e-10.
6. Run the full consensus to measure the population impact.

**Risk:** MC64 is a matching algorithm on a bipartite graph; a naive
implementation can be slow (O(n³) worst case). If the naive version
is unacceptably slow at `n = 10⁴`, a simpler alternative is the
Ruiz iterative ∞-norm scaling we already have — applied globally
instead of per-frontal. This is strictly weaker than MC64 but
significantly cheaper.

### 2.2.2 — The deferred 2×2 inertia trace fix

**Estimated effort:** 1 hour.

During the ACOPP30 triage in Phase 1b we found that
`src/dense/factor.rs::count_2x2_inertia` uses `a00` instead of the
trace `a00 + a11` to decide the sign of the non-zero eigenvalue in
the near-singular branch. The comment says "the other has sign of
trace"; the code is wrong.

We drafted the fix, observed a 16-matrix dense regression against
rmumps, and reverted with a `KNOWN BUG` comment pointing to
`dev/tried-and-rejected.md`. What we did NOT do at the time was
check whether the 16-matrix regression showed up against *canonical
MUMPS* — the Fortran oracle was not yet built.

**Implementation:**

1. Re-apply the trace-based fix exactly as documented in
   `dev/tried-and-rejected.md`.
2. Run the full consensus on the corpus.
3. Compare the before/after Definitive failure counts against
   canonical MUMPS, not rmumps.
4. If canonical MUMPS agrees with the fix: land it, remove the
   `KNOWN BUG` comment, remove the `dev/tried-and-rejected.md`
   entry via a new supplement entry marking it resolved.
5. If canonical MUMPS also disagrees: investigate. The fix is
   mathematically correct; if two canonical solvers prefer the
   buggy version, that tells us something interesting about
   boundary-pivot conventions.

### 2.2.3 — Triage the 88 sparse-only failures

**Estimated effort:** 3–5 hours.

The Phase 1b cross-comparison reports 88 matrices that fail the
sparse path but pass the dense path. None have been individually
triaged. They could be real sparse-pipeline bugs (same nature as
the postorder issue) or they could be borderline matrices where
rounding accumulates differently in multifrontal vs. monolithic
factorization.

**Implementation:**

1. Extend the bench to dump the names of all 88 matrices grouped
   by problem family.
2. Pick the one with the largest residual difference between the
   dense and sparse paths and build a triage example like
   `examples/triage_polak6.rs`.
3. Identify the root cause.
4. If it is a bug, fix it.
5. If it is a borderline rounding issue, document it as one of the
   known limits and move on.

This is a targeted version of the Phase 1b triage discipline. The
first matrix that gets investigated will either reveal a single bug
that closes many of the 88, or reveal a class of matrices that
should be classified as Borderline under the consensus framework.

### 2.2.4 — Re-run consensus, publish delta

After 2.2.1, 2.2.2, and 2.2.3, re-run the full consensus on the
corpus and publish a delta against the baseline report from
2.1.8. Expected outcome: the 26 Definitive feral failures from the
rmumps-deprecation consensus run (mostly ACOPP30 and DEVGLA2)
collapse toward zero once global MC64 scaling lands.

---

## Phase 2.3 — Pivoting improvements

**Duration:** 15–30 hours.

The Phase 1 stand-in for real pivoting was
`ZeroPivotAction::ForceAccept`. The spec explicitly defers the real
work to Phase 2 because the compromise is operationally OK for
correctness but not for the residual quality of ill-conditioned
matrices where `ForceAccept` ends up producing a wrong `A⁻¹`.

### 2.3.1 — Threshold partial pivoting (TPP)

**Estimated effort:** 5–8 hours.

Add a `PivotStrategy::ThresholdPartialPivoting { u: f64 }` option
to `BunchKaufmanParams` with `u = 0.01` matching SSIDS and MUMPS
defaults. The TPP kernel accepts a pivot iff it exceeds `u` times
the maximum off-diagonal in its column. Unlike pure BK, TPP may
*reject* a pivot and return a per-column failure code that delayed
pivoting can handle.

**Implementation:**

1. Research note `dev/research/threshold-partial-pivoting.md` covering
   the Bunch & Parlett 1971 citep:bunch1971direct origins and the
   modern formulation in Hogg & Scott 2013 citep:hogg2013pivoting.
2. New module or extended `BunchKaufmanParams` with the threshold
   option.
3. Modify `factor_frontal` to support pivot rejection — return a
   partial factorization with an indication of which columns failed.
4. This is a public-API change; the CLAUDE.md trigger for spec
   review applies.

### 2.3.2 — Delayed pivoting (SSIDS-style)

**Estimated effort:** 8–15 hours.

When a frontal rejects a column under TPP, the column is not
discarded — it is *delayed*, passed to the parent frontal where the
additional fill from the sibling contribution may make the pivot
acceptable. This is the mechanism that makes TPP work for KKT
matrices.

**Implementation:**

1. Research note covering the SSIDS delayed-pivoting pipeline.
2. Modifications to `numeric::factorize`:
   - `factor_frontal` returns a list of "delayed" columns that did
     not eliminate
   - `factorize_multifrontal` collects delayed columns from child
     supernodes and includes them in the parent's fully-summed
     set
   - The symbolic phase must pessimistically allocate enough
     space for the delayed columns (the SSIDS approach uses a
     fill factor based on observed delay rates, or an explicit
     `delay_factor` parameter)
3. The `SparseFactors` struct needs to track which columns came
   from delays so the solve can handle them correctly.
4. The sparse solve gets more complex — delayed columns are
   eliminated in a different supernode than their original, so
   the gather/scatter needs to follow the delayed-column mapping.
5. Full test suite update; several existing tests assume no delay.

**Risk:** This is the largest single change in Phase 2. It touches
the symbolic phase, the numeric phase, the solve phase, and the
factorization data structures. The Phase 1b postorder fix was a
single-function change; delayed pivoting is a pipeline change.
Budget for 2–3 sessions of focused work with the expectation that
some tests will need to be rewritten.

### 2.3.3 — A posteriori pivoting (APP)

**Estimated effort:** 4–7 hours.

APP is the trick SSIDS uses to get the speed of blocked factorization
without losing pivoting safety. The block is factored *without*
pivoting, then the threshold test is applied after the fact: if the
block's pivots all exceed `u × max_offdiag`, accept; if not, roll
back and try again with explicit pivoting.

**Implementation depends on blocked dense LDLᵀ from Phase 2.4.** APP
is not implementable until there is a blocked kernel to roll back.
This phase item is a stub that gets filled in after 2.4.

### 2.3.4 — Test against baseline

After 2.3.1, 2.3.2, and 2.3.3, re-run the full consensus. Expected
outcome: matrices that were previously classified as Numerically
Intractable drop into Borderline or Definitive as delayed pivoting
recovers them. Target: the 487 Numerically Intractable matrices in
the current consensus should shrink by at least 50%.

---

## Phase 2.4 — Dense kernel performance

**Duration:** 10–20 hours.

### 2.4.1 — Blocked dense LDLᵀ

**Estimated effort:** 6–10 hours.

The current dense BK kernel is scalar and unblocked. For frontal
matrices above ~200 rows, a blocked kernel with `block_size = 64` is
essential for cache efficiency. Faer's approach (cited in
research notes) is the model.

### 2.4.2 — SIMD micro-kernel for Schur complement

**Estimated effort:** 4–6 hours.

The inner loop of the rank-1 / rank-2 update that produces the
Schur complement is the hottest loop in the factorization. A SIMD
micro-kernel (likely via `std::simd` once stable, or manually via
`core::arch::x86_64` / `core::arch::aarch64` on stable until then)
should give 4–8× on this loop.

### 2.4.3 — Fused update + argmax

**Estimated effort:** 2–4 hours.

Faer's fusion trick computes the next column's argmax while doing
the current update, halving the memory traffic. This was already
implemented in Phase 1a but may need revisiting for the blocked
kernel.

---

## Phase 2.5 — Sparse pipeline performance

**Duration:** 10–20 hours.

### 2.5.1 — Column counts via Liu's row subtree algorithm

**Estimated effort:** 4–6 hours.

Phase 1b's column counts implementation is `O(n²)` worst case (as
explicitly noted in the Phase 1b plan). Liu's row subtree algorithm
citep:hogg2013pivoting gives `O(nnz(A) + n × α)` where α is the
inverse Ackermann function — effectively linear. This is probably
the highest-leverage Phase 2.5 item because it affects every call to
`symbolic_factorize` and the current implementation is the documented
scaling weak point.

### 2.5.2 — Parallelism on the assembly tree

**Estimated effort:** 4–8 hours.

Use Rayon to parallelize the independent subtrees of the assembly
tree. Sibling supernodes can be factored in parallel; only the
join at the parent supernode is sequential. `ContribPool`
transitions from a LIFO stack to a buddy allocator so contribution
blocks from independently-running siblings do not collide.

### 2.5.3 — Better memory allocation

**Estimated effort:** 2–4 hours.

Remove the per-supernode vec allocations in `solve_sparse`. Preallocate
scratch buffers sized to `max(supernode.nrow)` once per solve, reuse
across supernodes.

### 2.5.4 — Fill prediction

**Estimated effort:** 2–4 hours.

Improve the `factor_slack` heuristic (currently `1.2×` the predicted
NNZ). Use the SSIDS approach of tracking the actual delay rate and
adjusting dynamically.

---

## Phase 2.6 — Ordering

**Duration:** 4–8 hours.

### 2.6.1 — METIS integration

**Estimated effort:** 2–4 hours.

AMD is known to be weak for bordered KKT matrices like ACOPF. METIS
nested dissection handles them much better. Add a `OrderingKind`
enum with `Amd` and `Metis` variants, defaulting to METIS for
`SymbolicFactorization` called with KKT-shaped inputs (detect by
structural heuristic or explicit flag).

METIS is a non-Rust library, so this cannot ship in the core crate.
Two options:

- **A.** Use METIS via FFI, gate behind a `metis` feature flag.
  Breaks the "zero non-Rust dependencies in the core solver" rule.
- **B.** Port the relevant parts of METIS to Rust. Significant
  effort, but compatible with the architectural constraint.
- **C.** Use an existing Rust port (if one exists).

**Recommendation:** start with (C); fall back to (B). (A) is not
acceptable per the architecture constraint.

### 2.6.2 — LDLᵀ-aware ordering preprocessing

**Estimated effort:** 2–4 hours.

MUMPS's `ICNTL(12)` implements a "compressed graph" preprocessing
that collapses the constraint block of a bordered KKT matrix into
a single super-variable for the purposes of ordering, then
expands back out after ordering. This often produces much better
orderings for saddle-point problems than running AMD or METIS on
the uncompressed matrix. Implement the compression as an optional
step in `symbolic_factorize`.

---

## Phase 2.7 — Closed-loop validation

**Duration:** 4–8 hours.

### 2.7.1 — Run feral inside an IPOPT-style outer iteration

**Estimated effort:** 4–8 hours.

Phase 1's correctness validation is against *static* KKT matrices.
The cumulative effect of feral's inertia decisions inside an IPM
is not tested. Build a minimum-viable IPOPT-style outer loop that:

1. Starts from an initial point on a small test problem
2. Calls feral to factor the KKT
3. Uses the inertia to decide on regularization updates
4. Solves for the search direction
5. Does a simple line search
6. Iterates to convergence

Run it on the HS set and compare outer-iteration counts and final
solutions against what rmumps-in-ripopt produces. This is a
Phase-2.5 task in the sense that it prepares for Phase 3 (POUNCE
integration) without committing to Phase 3's full scope.

The goal is not to build a competitive IPM — rmumps already does
that. The goal is to confirm that feral's inertia and solution
quality are good enough to drive an outer loop *at all*, before
we commit to ripopt integration in Phase 3.

### 2.7.2 — IPOPT-loop regression test suite

**Estimated effort:** 2 hours.

Capture the closed-loop test results as a regression suite that
runs on every commit affecting the solver, not just on explicit
Phase 2 test runs. This prevents later Phase 2 changes from
silently breaking the IPM correctness path.

---

## Phase 2.8 — Exit criteria and Phase 3 handoff

**Duration:** 4 hours.

### 2.8.1 — Measure against the spec exit criterion

The spec (FERAL-PROJECT-SPEC.md §1747) defines Phase 2 exit as:
*Within 2× of MUMPS on small-frontal KKT set; within 3× on medium
set.* "Small-frontal" and "medium" need concrete definitions:

- **Small-frontal:** max frontal dimension < 200, problem-scale
  `n ≤ 10³`
- **Medium:** max frontal dimension < 500, problem-scale
  `n ≤ 10⁴`

Run the perf harness and publish the pass/fail verdict against each
of these bars. If either fails, identify the bottleneck and
determine whether another optimization pass is needed or whether
the gap is intrinsic.

### 2.8.2 — Write the Phase 2 exit session file

Mirror of `dev/sessions/2026-04-12-01.md`. Include:

- Goal
- Accomplished (all of 2.1 through 2.7)
- Benchmark results (the exit-criterion numbers)
- Decisions made (architectural changes and why)
- Abandoned approaches
- Next session / Phase 3 should...

### 2.8.3 — Update FERAL-PROJECT-SPEC.md

§1735 (the Phase 2 section) gets an appended note similar to what
§1712 got for Phase 1: record the exit date, the numbers, and
link to the session file. Do not modify the existing text.

### 2.8.4 — Write the Phase 2 retrospective

Mirror of `dev/phase1-retrospective.org`. Scientific writing style
in org-mode with org-ref citations against the expanded
`references.bib`. Covers what Phase 2 accomplished, what was
learned (especially around perf optimization and delayed
pivoting — both are more subtle than the correctness work in
Phase 1), and an honest assessment of success against the exit
criterion.

---

## Risk register

### R1: The sanity check in 2.1.2 fails

If `BratuProblem::new(1000)` reveals a latent scaling bug, 2.1.3
onward is blocked. Depending on the bug, the fix could be 30
minutes (missing `Clone` on a large vec) or several days
(quadratic `column_counts` is the root cause and Liu's algorithm
needs to land before anything else). **Mitigation:** the sanity
check is the first thing we do; if it blocks Phase 2 that is
exactly the information we need and we reschedule the rest around
whatever needs to be fixed.

### R2: MC64 is too slow

A naive MC64 at `n = 10⁴` may take seconds per matrix. If this
turns into minutes per matrix, feral's time budget is blown on
scaling alone and we cannot measure the rest of the solver
meaningfully. **Mitigation:** start with a simpler global
infinity-norm Ruiz scaling as a placeholder. This is known to be
weaker than MC64 but can be swapped in quickly. Upgrade to a
proper MC64 implementation in a follow-up if the weaker scaling
is insufficient.

### R3: Delayed pivoting breaks existing tests

The postorder fix needed no test updates because it was a
symbolic-only change. Delayed pivoting is a cross-cutting pipeline
change and will likely require test updates — tests that assume
specific column ordering, specific inertia counts in edge cases,
or specific residual bounds. **Mitigation:** budget one extra
session for test updates; resist the temptation to loosen test
tolerances; document each test change in the commit body.

### R4: METIS cannot be implemented in pure Rust in reasonable time

METIS is a large and mature C library. A pure-Rust port is
probably beyond a single Phase 2 cycle. **Mitigation:** check
for an existing Rust METIS port first. If none exists at adequate
quality, implement a simpler nested-dissection variant
(recursive bipartition using the existing AMD-style heuristics)
that is strictly worse than METIS but better than AMD on
bordered KKT. Label it as "poor man's METIS" and track improving
it as Phase 2+ work.

### R5: The performance gap is too large

Phase 1 has not measured feral against MUMPS on any matrix. If
the first perf measurement reveals a 100× gap, the "within 2-3×
of MUMPS" exit criterion is several rounds of optimization
away. **Mitigation:** publish the baseline as soon as 2.1.8 is
complete. If the gap is too large for a single Phase 2, propose
to the user either (a) splitting Phase 2 into 2a (correctness,
scaling, pivoting) and 2b (performance), or (b) relaxing the
exit criterion in a new `dev/decisions.md` entry. Do not
silently push the bar; be explicit about any spec deviation.

### R6: The ACOPP30 residual gap has more than one cause

We hypothesized it is global scaling. It may also be affected by
iterative refinement differences, equilibration strategies, or a
subtle interaction with the BK pivot threshold. **Mitigation:**
the triage example `examples/triage_acopp30.rs` already exists.
After MC64 lands, re-run the triage. If the residual is still 12
orders worse, investigate further before declaring 2.2.1
complete.

### R7: Closed-loop validation reveals inertia-driven divergence

Static inertia correctness does not imply closed-loop
convergence. Feral may pass every Phase 1 and Phase 2.1–2.6
check and still fail to converge an IPM on a simple problem
because its inertia decisions nudge the outer iteration into a
region where rmumps would not have gone. **Mitigation:** 2.7 is
deliberately placed before 2.8 exit. If closed-loop fails, we
have to go back and investigate; the exit criterion is not
"pass the static benchmark" but "pass the closed-loop sanity
check" as well.

### R8: Cross-repo scope creep

The `collect_kkt_large` binary lives in ripopt, not feral. Changes
to ripopt during Phase 2 are expected to be minimal (just the new
binary) but could spread if we discover ripopt bugs while building
the benchmark. **Mitigation:** any ripopt change beyond the new
binary requires an explicit user check-in. Do not silently fix
ripopt issues in feral's Phase 2.

## Open questions for the user

These must be answered before Phase 2 begins in earnest:

1. **Cross-repo changes.** The `collect_kkt_large` binary and any
   fixes to it are in ripopt. Am I allowed to edit ripopt directly
   during Phase 2? If yes, scope: just the new binary, or broader
   fix-as-you-go?

2. **METIS.** Options A/B/C in §2.6.1. Which path? The architectural
   constraint rules out FFI in the core crate, but a dev-only
   feature-flag variant might be acceptable for benchmarking. Or
   pure-Rust-only regardless of cost?

3. **Exit criterion flexibility.** If the first perf measurement
   reveals a > 10× gap to MUMPS, do we (a) push through and try to
   close it in Phase 2, or (b) relax the criterion to "within 5× on
   small" as a new `dev/decisions.md` entry, or (c) split Phase 2
   into 2a (correctness + scaling) and 2b (performance)?

4. **Delayed pivoting scope.** The Phase 1 decision was to defer
   delayed pivoting to Phase 2 in exchange for the ForceAccept
   stand-in. Delayed pivoting is listed as a Phase 2 task in the
   spec. However, it is the largest single piece of work in this
   plan (8–15 hours). Is it a must-have for Phase 2 or can it be
   pushed to a Phase 2.5 that ships after the performance work?
   My recommendation is "must-have" because several of the
   Numerically Intractable matrices need it, but it is worth
   confirming.

5. **Corpus expansion scope.** Phase 2.1 lists `large_scale`,
   `grid`, and (optionally) `gas` and `water`. Which are in scope?
   Gas/water require AMPL `.nl` readers that ripopt may not have.

6. **Closed-loop validation depth.** Phase 2.7 proposes a minimal
   IPOPT-style loop. Is this sufficient, or should it be a full
   ipopt-replacement test (run feral inside actual ripopt, not a
   toy IPM)? The latter is closer to Phase 3 territory but it is
   also the most faithful test of feral-as-solver-for-ripopt.

## Phase ordering summary

```
2.1  Measurement infrastructure       6–10h    ← START HERE
     ├── 2.1.1 lift n>500 filter       0.5h
     ├── 2.1.2 sanity check            1–2h     ← GATE for 2.1.3+
     ├── 2.1.3 collect_kkt_large       2h
     ├── 2.1.4 run oracles             0.5h
     ├── 2.1.5 grid addition           1–2h    (optional)
     ├── 2.1.6 gas/water               3–4h    (deferred, optional)
     ├── 2.1.7 perf harness            2–3h
     └── 2.1.8 baseline report         0.5h
                                        ↓
2.2  Deferred correctness fixes       8–16h
     ├── 2.2.1 MC64 scaling            4–8h
     ├── 2.2.2 2x2 trace fix           1h
     ├── 2.2.3 88 sparse-only triage   3–5h
     └── 2.2.4 consensus delta         0.5h
                                        ↓
2.3  Pivoting improvements            15–30h
     ├── 2.3.1 TPP                     5–8h
     ├── 2.3.2 Delayed pivoting        8–15h   ← LARGEST item
     ├── 2.3.3 APP (after 2.4)         stub
     └── 2.3.4 consensus delta         0.5h
                                        ↓
2.4  Dense kernel performance         10–20h
     ├── 2.4.1 Blocked LDLᵀ            6–10h
     ├── 2.4.2 SIMD Schur              4–6h
     └── 2.4.3 Fused update+argmax     2–4h
                                        ↓
2.5  Sparse pipeline performance      10–20h
     ├── 2.5.1 Liu column counts       4–6h   ← probably highest leverage
     ├── 2.5.2 Rayon parallelism       4–8h
     ├── 2.5.3 Scratch allocation      2–4h
     └── 2.5.4 Fill prediction         2–4h
                                        ↓
2.6  Ordering                         4–8h
     ├── 2.6.1 METIS                   2–4h
     └── 2.6.2 LDLᵀ-aware preprocess   2–4h
                                        ↓
2.7  Closed-loop validation           4–8h
     ├── 2.7.1 Outer-iteration test    4–8h
     └── 2.7.2 Regression suite        2h
                                        ↓
2.8  Exit                              4h
     ├── 2.8.1 Measure vs criterion    1h
     ├── 2.8.2 Session file            1h
     ├── 2.8.3 Spec update             0.5h
     └── 2.8.4 Retrospective           2h
```

**Best-case cumulative:** 61 hours (about 2 focused weeks)
**Realistic cumulative:** 100 hours (about 3–4 focused weeks)
**Worst-case with blockers:** 150+ hours (6 weeks)

## What I need from you to start Phase 2

1. Answers to the six open questions above.
2. An explicit "start Phase 2" decision, recorded in
   `dev/decisions.md`, so there is a clear cutover from Phase 1
   closure to Phase 2 work.
3. Confirmation that the `dev/plans/phase-2-planning.md` ordering
   is the right ordering, or a counter-proposal.
4. If the ordering is right: permission to start Phase 2.1.1
   (lift the `n > 500` filter) and 2.1.2 (sanity check) as the
   first concrete tasks.

Phase 2.1.1 is a one-line code change. Phase 2.1.2 is a
self-contained 1–2 hour investigation with a clear
decision point at the end. Together they answer the most
important question Phase 2 opens with: *does feral's sparse path
scale at all?* — and the answer to that question determines
whether the rest of this plan is the right plan.
