# issue #91 — `OrderingPreprocess::Auto` mis-fires LdltCompress on conic KKTs — 2026-06-30

## Symptom

feral factorizes the qap15 conic KKT (n=50880, nnz=168105, a quadratic-
assignment LP-relaxation KKT from pounce-convex) in **~15 s** vs faer's
**~0.22 s**. Reported as a "catastrophic slowdown" attributable to ordering
fill and dense-kernel throughput.

## Diagnosis (measured on the real matrix)

Reproduced the exact matrix via pounce-convex + `POUNCE_DBG_KKT_DUMP`
(matches issue dims byte-for-byte). Symbolic fill (`col_counts` sum,
amalgamation-independent), via `examples/bench_qap15.rs`:

| ordering | preprocessing | simplicial nnz_L |
|---|---|---|
| AMD | `Auto` (default) | **45,376,046** |
| AMD | `None`           |  **7,155,773** |
| AMF | `None`           |    7,644,543 |
| MetisND | `None`        |   19,052,431 |
| faer AMD (issue) | —    |   13,370,955 |

Numeric factor: default (Auto) **15.4 s / nnz_L 40.9M**; `preprocess=None`
**0.77 s / nnz_L 9.25M** (20×); +FMA **0.67 s**. Inertia
`(+22275,−28605,0)` in every case — correct, no cascade, no delayed-pivot
inflation.

**feral's AMD ordering is excellent (7.16M, beats faer's 13.4M).** The
entire blowup is `OrderingPreprocess::Auto`. Its predicate
`pick_ordering_preprocess` (src/symbolic/mod.rs:485) fires `LdltCompress`
when ≥30 % of columns have ≤2 nonzeros. A regularized quasi-definite IPM
KKT is saturated with degree-≤2 columns (the diagonal regularization rows:
the F block here is 22275 columns of ~1e-10 diagonal), so the structural
proxy fires **exactly** where MC64 symmetric-matching compression destroys
the elimination order: 7.16M → 45.4M (6.3×).

Refuted along the way: delayed pivots (nnz_L identical across pivot
strategies), amalgamation padding (nemin 1 vs 16 → identical simplicial
fill), AMD quality (None beats faer), SQD need (orthogonal; and the
existing SQD mode trips `SqdContractViolated` at col 0 on the 1e-10 pivot).

## Fix — verify, don't predict

The structural predicate is a one-way proxy; it cannot know whether
compression actually reduces fill. Make `OrderingPreprocess::Auto` a
**fill-verified race**, mirroring the existing `OrderingMethod::AutoRace`:

- When the predicate declines (`None`), keep `None` — it is the baseline;
  no compression cost incurred, no regression possible.
- When the predicate recommends `LdltCompress`, run the symbolic pipeline
  **both** ways (None and LdltCompress) and **keep LdltCompress unless it
  inflates `factor_nnz_estimate` past a catastrophe ceiling** (2× the None
  baseline, `PREPROCESS_FILL_INFLATION_LIMIT`). A modest fill increase
  keeps LdltCompress; only a runaway inflation falls back to None.

The asymmetric, generous threshold is deliberate, and an early "do no
fill harm" version (keep LdltCompress only on ties/improvements) was
**rejected** — it regressed inertia on near-singular corpus KKTs:

| matrix | None est | LdltCompress est | ratio | inertia under LdltCompress |
|---|---|---|---|---|
| qap15    | 8.59M | 54.45M | 6.34× | catastrophic blowup |
| twirism1 | 26683 |  30782 | 1.15× | (432,313,0) — oracle-correct |
| sawpath  |  7548 |   7557 | 1.00× | (789,670,116) — oracle-correct |

twirism1 is the key counterexample: LdltCompress costs +15 % fill but its
MC64-matched 2×2 pivots produce the **oracle-correct** inertia, while the
slightly-leaner `None` ordering misclassifies two near-zero pivots
(434,311,0). Pure fill-racing would wrongly discard that benefit and
break the `issue65_mc64_fallback` inertia gate. The 2× ceiling sits well
above the normal ~1.1–1.2× compression overhead (keeps twirism1/sawpath)
and well below qap15's 6.3× (catches the misfire). Validated by the full
corpus suite (no inertia regression) plus the issue-#80 arrow-KKT
profiler test (a fill tie → LdltCompress retained). The extra symbolic
pipeline only runs when LdltCompress was going to be applied anyway, and
its MC64 matching (the dominant cost) runs regardless.

Profiler handling mirrors `symbolic_factorize_race`: each candidate gets a
fresh profiler; the winner's is copied into the caller's shared one.

## Residual

After the fix feral is ~0.67–0.77 s vs faer ~0.22 s (~3×). The remainder is
dense-kernel throughput on large supernodes (FMA-on-large → BLAS-3 GEMM,
`dev/plans/dense-kernel-blas3.md`) — the real, durable kernel work, now the
only remaining lever rather than buried under a 20× heuristic bug.

## Regression fixture

`tests/data/large/qap15_kkt.mtx` (the reproduced matrix). Test asserts
`Auto` simplicial fill ≤ `None` fill and that `Auto` resolves below the
old 45.4M on this pattern.
