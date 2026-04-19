## Sparse-path tail performance survey — post-Policy 4

**Date:** 2026-04-19
**Authorized by:** session continuation, "lets do 2 directly"
(after Policy 4 closure).
**Scope:** identify the next high-leverage perf lever on the
sparse path now that the corpus geomean factor/MUMPS = 0.48
and the lever-C arrow-KKT regression has been fixed.
**Output:** a recommended lever, not an implementation plan.

### 1. Where we stand

After Policy 4 (commit `af9315d`), bench-policy4 corpus
metrics on the sparse path:

| metric           | value                          |
|------------------|--------------------------------|
| residual_pass    | 154 233 / 154 588 (99.8%)      |
| inertia_match    | 153 009 / 154 588 (99.0%)      |
| factor/MUMPS     | geomean 0.48, p50 0.36, p90 1.96, p99 3.94, **max 101.58** |
| Phase 2.8.1 small-frontal p90 | 1.98 ≤ 2.0 PASS   |
| Phase 2.8.1 medium p90        | 1.98 ≤ 3.0 PASS   |

Geomean and p50 are well below 1; the pain is the long
right tail.

### 2. Top-10 worst factor-ratios (sparse path)

From `dev/results/lever-c/bench-policy4.txt`:

```
LAKES_1199          n=168    5587us / 55us    102×
TRO3X3_0013         n=69     5417us / 68us     80×
CHWIRUT1_0009       n=645    6373us / 234us    27×
HAHN1LS_0429        n=7       121us / 9us      13×
CHWIRUT2LS_0184     n=3       113us / 10us     11×
FBRAIN3LS_0003      n=6       125us / 12us     10×
KIRBY2_0007         n=458    1270us / 122us    10×
PALMER5B_0049       n=9       108us / 11us     10×
MUONSINE_0000       n=1537   3611us / 369us    10×  (already at floor)
KIRBY2_0006         n=458    1227us / 130us     9×
```

Three distinct outlier classes:

- **Class A — tiny matrices (n ≤ 9):** HAHN1LS_0429,
  CHWIRUT2LS_0184, FBRAIN3LS_0003, PALMER5B_0049. ~100 µs
  flat against MUMPS's ~10 µs. Per-call setup dominates.
  4 of top 10.
- **Class B — small-medium (n=69–645):** LAKES_1199,
  TRO3X3_0013, CHWIRUT1_0009, KIRBY2_0006/7. 1–6 ms feral
  vs 55–234 µs MUMPS.
- **Class C — large arrow-KKT:** MUONSINE_0000. Already at
  the perf floor for this class post lever-C; not worth
  re-attacking.

### 3. A measurement discrepancy worth noting

Direct cold-call timing (one-off
`symbolic_factorize + factorize_multifrontal` from a separate
binary) gives, e.g., LAKES_1199 = 528 µs (cold), 276 µs (5th
iter warm). The bench reports 5587 µs — **~10× higher**.
TRO3X3 and HAHN1LS show similar 7–12× discrepancies; KIRBY2
goes the OTHER way (bench 1270 µs vs cold 2535 µs).

Hypotheses:
1. **Heap fragmentation / allocator slow path** after 154 588
   matrices' worth of accumulated `Vec<f64>` / `Vec<usize>` /
   `Box<[T]>` allocations. The free list is full but
   poorly clustered for new requests.
2. **Cache pressure** — bench keeps the dump CSV writer, the
   per-matrix sidecar, and the MUMPS/SSIDS oracle data alive
   throughout the loop. Each new matrix factorization fights
   for L2/L3 against the loop's bookkeeping state.
3. **Inverse for KIRBY2** — KIRBY2 sees BENCH faster than
   cold-call. Possibly because the bench loop hits KIRBY2
   after thousands of similar-sized factorizations, so the
   allocator's freelist happens to match KIRBY2's request
   pattern; the cold call hits a fresh allocator with no
   matching blocks.

This is *not* an artifact-free signal — bench is what users
will see in real IPM workloads (each KKT factorization is
fresh, allocator state accumulates over 100s of iterations).
The discrepancy itself argues for an arena/scratchpad lever.

### 4. Per-family geomeans where we lose to MUMPS

From the bench top-25 families by count, sparse path:

| family   | count | geomean | p50  | max  |
|----------|------:|--------:|-----:|-----:|
| HS118    | 3000  | 1.05    | 1.07 | 4.56 |
| AVION2   | 2682  | 1.61    | 1.62 | 3.30 |
| BATCH    | 2054  | 1.85    | 1.88 | 2.96 |
| MCONCON  | 3000  | 0.71    | 0.73 | 7.06 |

About 7700 matrices (5% of corpus) where geomean > 1. Pulling
AVION2 and BATCH down to 0.5 would shift the corpus geomean
visibly, with broader impact than chasing the 10-matrix top
tail.

### 5. Candidate levers (ranked)

#### Lever D.1 — Allocator-aware scratch reuse

**Claim:** The bench-vs-cold discrepancy points at allocation
churn. A `FactorWorkspace` (or `SymbolicScratch` +
`NumericScratch`) struct that callers pre-allocate and reuse
across factorizations would amortize the per-call alloc cost.

**Impact estimate:** if bench's 10× overhead on Class A and
Class B outliers is allocation, this lever takes:
- LAKES_1199: 5587 µs → ~500 µs (ratio 102 → 9)
- TRO3X3_0013: 5417 µs → ~500 µs (ratio 80 → 7)
- HAHN1LS_0429: 121 µs → ~20 µs (ratio 13 → 2)

p99 likely drops from 3.94 → ~3.0; max from 102 → ~10.

**Cost:** medium. Requires API change (a workspace handle
threaded through `factorize_multifrontal`) and refactoring
of allocation sites. The `Solver` struct already exists for
caching the symbolic phase across IPM iterations — this
extends that pattern to the numeric scratch.

**Risk:** the discrepancy might not be allocation; could be
cache or thermal. Need to instrument before committing.

#### Lever D.2 — Per-family analysis: AVION2 + BATCH

**Claim:** AVION2 (geomean 1.61, 2682 matrices) and BATCH
(1.85, 2054 matrices) lose to MUMPS on average. Profile a
representative AVION2_0000 to find the bottleneck:
ordering? supernode merging? frontal assembly?

**Impact estimate:** 7700 matrices with current geomean ~1.7.
If we get to 0.7 (matching the median family), corpus geomean
shifts from 0.48 → ~0.43. Visible win.

**Cost:** unknown — depends on what the profile shows.
Could be a one-line tweak (e.g. nemin) or a larger
restructuring.

**Risk:** might find the bottleneck is fundamental (e.g.,
matrix structure that resists supernode amalgamation).

#### Lever D.3 — Dense fast-path for small matrices

**Claim:** TRO3X3_0013 (n=69, 37% dense) is in the sparse
top-10 because the multifrontal scaffolding is overkill.
A heuristic that routes "small-and-mostly-dense" matrices
directly to the dense BK kernel would eliminate this class.

**Impact estimate:** narrow. Affects perhaps 50-200 matrices
in the corpus (those small enough to be dense-eligible AND
dense enough to win). TRO3X3_0013 ratio: 80 → ~3.

**Cost:** small. The dense path already exists; adding a
gate on n × density is straightforward.

**Risk:** crossover threshold tuning.

#### Lever D.4 — Tiny-matrix fast-path

**Claim:** Class A (n ≤ 10) sees ~100 µs flat overhead. A
specialized small-n path could avoid the multifrontal
machinery entirely (no supernodes, no frontal allocation —
just inline the factor).

**Impact estimate:** ~4–8 outliers in top-10 fixed. Many
small matrices in corpus would see 5–10× speedup, but they
were already passing perf targets, so the corpus geomean
moves only slightly.

**Cost:** medium. Requires a new code path duplicating logic
that already exists in the dense kernel.

**Risk:** maintenance burden for a corner case.

### 5b. D.2 investigation result — nemin doesn't move it

Profiled AVION2_{0000, 0500, 1500} and BATCH_{0000, 0500, 1500}
at `nemin ∈ {1, 5, 32}` via `profile_sparse` with a
`FERAL_NEMIN` override:

| matrix       | n   | nnz | fac µs (nemin=32) | nemin=5 | nemin=1 |
|--------------|----:|----:|------------------:|--------:|--------:|
| AVION2_0000  |  94 | 167 |              35.2 |    33.2 |    48.4 |
| AVION2_0500  |  64 | 193 |              22.5 |    26.0 |    42.9 |
| BATCH_0000   | 121 | 299 |              80.0 |    82.0 |    92.3 |
| BATCH_0500   | 121 | 305 |              72.3 |    83.0 |   103.1 |

Default `nemin=32` is already at or near the optimum.
Smaller nemin only hurts (more, smaller fronts → more
scaffolding overhead). Per-stage breakdown:

- **AVION2_0000**: sym ≈ 13 µs, fac ≈ 35 µs, MUMPS ≈ 19 µs.
  Total feral ≈ 48 µs vs MUMPS 19 µs ≈ 2.5×. Bench geomean
  shows 1.61 because some matrices in the family are smaller
  (AVION2_0500 is n=64).
- **BATCH_0000**: sym ≈ 27 µs, fac ≈ 80 µs, MUMPS ≈ 13 µs.
  Total feral ≈ 107 µs vs MUMPS 13 µs ≈ 8×. The bench
  geomean of 1.85 is misleadingly low — this single sample
  is an 8× hit. Where: numeric factorization (80 µs for
  ~300 nnz / n=121) is the dominant cost.

The per-family loss is real and structural — it's the cost
of the multifrontal pipeline's per-supernode overhead at
small n. Tuning supernode amalgamation doesn't help because
the matrices are too small for amalgamation to be the lever.

### 6. Updated recommendation

**Lever D.1 (workspace reuse) is the right next step.**

D.2's nemin investigation rules out the most obvious symbolic-
phase tuning. The remaining gap is per-call allocation +
small-fixed-cost overhead in the multifrontal pipeline, which
is exactly what an arena-based scratch struct addresses:

- `factorize_multifrontal` allocates per-supernode frontal
  matrices (`Vec<f64>`), assembly index maps (`Vec<usize>`),
  and pivot sequences (`Vec<i32>`). Across 154 588 matrices
  this is millions of allocations; on small matrices the
  alloc cost dominates the floating-point cost.
- The `Solver` struct already caches `SymbolicFactorization`
  across IPM iterations. Extending it to cache the numeric
  scratch is a natural follow-on.

Concrete next steps (NOT yet authorized):

1. Instrument `factorize_multifrontal` to count allocations
   per call. Run on AVION2_0000 and BATCH_0000 to confirm
   alloc count.
2. If alloc count is high, draft a `FactorWorkspace` API.
3. Land workspace plumbing on a feature-flag, A/B in bench.

D.3 (dense fast-path for TRO3X3-class) and D.4 (tiny-matrix
fast-path) remain on the menu as narrow follow-ups.

What this lever cannot fix: the absolute MUMPS-floor on
matrices like BATCH_0000 where MUMPS does the same work in
13 µs. Some of the 80 µs gap may be MUMPS's specialized
small-matrix kernel; we'd need to compare against the MUMPS
sources to confirm.

Why D.2 over D.1:
- **Bigger blast radius.** 7700 matrices vs. ~10 outliers.
  Geomean is the metric that compounds.
- **Cheaper to investigate.** A profile run on AVION2_0000
  will tell us within an hour what the next steps are. D.1
  requires architectural design work upfront.
- **D.1 is conditional on a hypothesis** (allocation churn)
  we haven't yet confirmed. The bench-vs-cold discrepancy
  is suggestive, not conclusive (KIRBY2 goes the other way).

Why D.2 over D.3 / D.4:
- D.3/D.4 each affect narrow slices of the tail. They're
  good follow-ups once the broad lever is exhausted, but
  starting with them risks tuning a corner that doesn't
  matter.

### 7. Test-before-implement checklist

If D.2 is approved, the first concrete step is:

1. Run `profile_sparse` on `AVION2_0000` and `BATCH_0000`
   to get sym/fac/solve breakdown.
2. Compare with MUMPS's per-phase analysis (the JSON
   sidecar has factor_us; we can also run rmumps directly
   for a finer split).
3. Identify which phase (symbolic, numeric, or
   solve) accounts for the loss.
4. Survey 5-10 matrices per family to confirm the pattern
   isn't matrix-specific.

The research note for the implementation lever lands after
step 4 — this note doesn't authorize implementation, only
the investigation.

### 8. Files this session (Policy 4 + this note)

- `src/scaling/mod.rs` — Policy 4 fallback (committed
  `af9315d`).
- `src/bin/policy4_diag.rs` — diagnostic binary (committed
  `af9315d`).
- `dev/research/policy-4-scaling-fallback.md` — Policy 4
  research note (committed `af9315d`).
- `dev/plans/policy-4-scaling-fallback.md` — Policy 4 plan
  (committed `af9315d`).
- `dev/results/lever-c/bench-policy4.txt`,
  `dev/results/lever-c/dump-policy4.csv` — Policy 4 corpus
  evidence (committed `af9315d`).
- `dev/research/sparse-tail-perf-2026-04-19.md` — this
  note. Awaits next-lever decision.
