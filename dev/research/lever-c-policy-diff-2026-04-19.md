# Lever-C policy diff — verdict cross-reference

**Date:** 2026-04-19
**Session:** 2026-04-18-09 (continued)
**Status:** measurement update; supersedes the recommendation in
`lever-c-corpus-bench-2026-04-19.md` §4.

## Why this note exists

The earlier corpus-bench note recommended **do NOT flip the scaling
default**, based on raw pass-count deltas (Policy 3 lost net 9
residual passes vs baseline) and one matrix (POLAK6_0021) where MC64
inertia disagreed with the sidecar `expected_inertia`.

That recommendation conflated three failure populations:

1. matrices regressing because their residual was already at the
   pass/fail threshold and any perturbation pushes them to the wrong
   side ("threshold jiggle"),
2. matrices the corpus's own oracles disagree on (`verdict: excluded`
   or `inertia_agreement: none`), where the sidecar
   `expected_inertia` is one oracle's opinion, not a ground truth,
3. matrices where all oracles agree and feral really did get worse.

Only category (3) is a real regression. Policies should be judged
against (3) alone. To separate them, a per-matrix CSV dump from the
bench harness was added (`FERAL_BENCH_DUMP=path.csv`) and a
`dump_diff` binary cross-references each regression against
`<matrix>.verdict.json`'s `inertia_agreement` and `verdict` fields.

## Method

```sh
FERAL_SCALING=infnorm  FERAL_BENCH_DUMP=dev/results/lever-c/dump-baseline.csv  cargo run --release --bin bench
FERAL_SCALING=mc64     FERAL_BENCH_DUMP=dev/results/lever-c/dump-mc64.csv      cargo run --release --bin bench
FERAL_SCALING=adaptive FERAL_BENCH_DUMP=dev/results/lever-c/dump-adaptive.csv  cargo run --release --bin bench
cargo run --release --bin dump_diff
```

Each CSV has 154 588 rows (one per corpus matrix) with columns
`name,n,factor_us,solve_us,exp_p,exp_n,exp_z,act_p,act_n,act_z,inertia_ok,rel_res,residual_ok`.

`dump_diff` enumerates the union of {residual regressions, inertia
regressions} under each policy and parses each matrix's
`.verdict.json` for `inertia_agreement` ∈ {strong, weak, none} and
`verdict` ∈ {definitive, numerically_intractable, borderline,
excluded}.

## Counts

|                              | baseline | Policy 2 (mc64) | Policy 3 (adaptive) |
|------------------------------|---------:|----------------:|--------------------:|
| residual passes (of 154 588) |  154 241 |         154 225 |             154 232 |
| inertia passes               |  153 008 |         153 007 |             153 009 |
| residual regressions vs base |        — |              77 |                  21 |
| residual recoveries vs base  |        — |              61 |                  12 |
| **net residual delta**       |        — |        **−16**  |             **−9**  |
| inertia regressions vs base  |        — |               3 |                   1 |
| inertia recoveries vs base   |        — |               2 |                   2 |
| **net inertia delta**        |        — |         **−1**  |             **+1**  |

## Policy 2 (blanket MC64): 80 unique regressed matrices

Cross-reference summary:

| `inertia_agreement` | count | `verdict.verdict`         | count |
|---------------------|------:|---------------------------|------:|
| strong              |    77 | numerically_intractable   |    46 |
| weak                |     2 | definitive                |    32 |
| none                |     1 | excluded                  |     1 |
|                     |       | borderline                |     1 |

Of the 32 `definitive` regressions:
- 31 are tiny CUTEst residuals (n ∈ {3, 5, 6, 7, 8}) where baseline
  `rel_res ∈ [3.10e-10, 1.53e-9]` and policy `rel_res ∈ [8.62e-10,
  4.42e-9]` — i.e., the regression is the residual crossing the
  pass threshold (which lands at ~1e-9 for these tiny n via
  `n * f64::EPSILON * 1e6`), not a real loss of digits.
- 1 is **MSS1_0009** (n=163), the genuine regression: baseline
  `6.31e-12` → policy `9.96e-7`, five orders of magnitude.

The single `excluded` regression is **POLAK6_0021** — already
diagnosed in `polak6-triage-2026-04-19.md` as oracle-disagreement
(four oracles give four different inertias on a matrix with raw
|diag| range 1e46).

## Policy 3 (adaptive `diag_only/n ≥ 0.30`): 22 unique regressed matrices

| `inertia_agreement` | count | `verdict.verdict`         | count |
|---------------------|------:|---------------------------|------:|
| strong              |    20 | numerically_intractable   |    14 |
| weak                |     1 | definitive                |     6 |
| none                |     1 | excluded                  |     1 |
|                     |       | borderline                |     1 |

Of the 6 `definitive` regressions:
- 5 are tiny (n ∈ {3, 4, 8}) threshold jiggle (HATFLDFL_03xx,
  HATFLDFL_04xx, ALLINITA_0758, SNAKE_0101).
- **1 is MSS1_0009** — same matrix as in Policy 2.

The `excluded` is POLAK6_0021 (same as Policy 2).
The `borderline weak` is ACOPP14_0001 (n=106), an inertia-only
regression with both residuals at machine epsilon.

## Reframing

Excluding (a) threshold jiggle on n ≤ 8 matrices and (b) oracle-
disagreement matrices, the genuine regressions are:

| Policy | genuine regressions |
|--------|---------------------|
| 2      | MSS1_0009 (residual) + ALLINITA/HATFLDFL borderlines + 31 small-n threshold jigglers among `definitive` |
| 3      | **MSS1_0009 only** (in the `definitive` + `strong` quadrant, non-jiggle) |

Policy 3's net inertia delta is **+1** (better than baseline). Its
net residual delta of −9 is dominated by tiny CUTEst matrices where
the residual sits at the threshold under both policies and merely
jiggles across it.

## Decision

**Keep InfNorm as default; defer MSS1_0009 triage to a follow-up.**

MSS1_0009 is one matrix where MC64 produces a 9.96e-7 residual
instead of 6.31e-12. In isolation the five-orders-of-magnitude jump
looks alarming, but the bench harness does NOT run iterative
refinement. In a real outer-iteration consumer (Ipopt step,
`solve_sparse_refined`) a 1e-7 LDLᵀ residual is well within the
range refinement recovers to ~1e-14. The pathology is almost
certainly the same family as POLAK6_0021 (matched-but-bad MC64
scaling), but on a matrix the corpus oracles all agree on, so it
shows up as a "real" regression in the diff.

Two reasons to defer rather than chase now:

1. The lever-C arc's actual goal is the VESUVIO/CRESC speedup; the
   scaling-default flip is a *means* to it. Without an integration
   consumer (Ipopt or similar) we cannot tell whether a 1e-7 bench
   residual translates to a real downstream problem or is fully
   absorbed by refinement.
2. POLAK6 already established that single-shot scaling-quality
   heuristics cannot distinguish "matched but bad" from "matched
   and fine" without trial factorization. MSS1_0009 is likely the
   same shape of finding — useful telemetry, but not actionable
   without Policy-4-style trial-and-fallback machinery, which is
   premature for the same reason.

Recorded as a backlog item in `tasks.org` ("MSS1_0009 MC64-pathology
triage — defer until after integration"). The lever-C scaling
default stays at `InfNorm` for now; `Auto` ships as an opt-in
strategy via `FERAL_SCALING=adaptive` and the `ScalingStrategy::Auto`
public variant for downstream callers who want it.

## Files

- `src/bin/bench.rs` — added `FERAL_BENCH_DUMP` env var
- `src/bin/dump_diff.rs` — new diff binary
- `dev/results/lever-c/dump-{baseline,mc64,adaptive}.csv` — raw data
- `dev/research/lever-c-policy-diff-2026-04-19.md` — this note

No production code change.
