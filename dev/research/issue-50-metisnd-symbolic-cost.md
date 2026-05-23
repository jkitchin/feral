# Research Note: MetisND symbolic cost on power-grid KKT (issue #50)

**Status:** Pre-implementation
**Date:** 2026-05-23
**Author:** agent session 2026-05-23-NN
**Related issues:** https://github.com/jkitchin/feral/issues/50
**Related code:** `src/symbolic/mod.rs:291` (`pick_default_method`),
`src/symbolic/mod.rs:513` (`symbolic_factorize_with_method`),
`src/symbolic/profiler.rs` (per-stage timer infrastructure).
**Calibration precedent:** `dev/research/issue-10-ordering-supernode-shape.md`,
`dev/journal/2026-04-27-*.org` (chain-pattern catch).

## Overview

Reporter shows `Solver::factor` taking **113 s** on a 2 813 976-dim
symmetric-indefinite KKT from a GAMS `powerflow22` IPM solve.
Refactor of the same pattern (cached symbolic) is **1.05 s**, so the
cost is essentially all in the symbolic phase. AMD-symbolic on the
same matrix is **55 s**. MUMPS factors the same matrix in ~**3 s**
total (symbolic + numeric).

Stored `nnz/n ≈ 2.4`, so this matrix lands in the second `MetisND`
branch of `pick_default_method` (`n >= 2000 && avg_deg < 4.0`,
`src/symbolic/mod.rs:296`). That branch is the "chain-pattern catch"
calibrated on CHAINWOO/HYDROELL/DIXMAANH at n ≈ 3000–4033
(`dev/research/issue-10-ordering-supernode-shape.md`,
`dev/journal/2026-04-27-*.org`), three orders of magnitude smaller
than the issue's matrix.

The question this note opens: at n = 2.8 M with avg_deg 2.4, is
`MetisND` still the right default, or does the chain-catch rule
need a size cap?

## Reproduction

Inputs (provided locally by the reporter, NOT in repo):

```
gams/nlpbench/feral_repro/powerflow22/
  kkt_solve_init.bin     # very first multi_solve (RHS=0)
  kkt_solve_iter2.bin    # 3rd multi_solve (real RHS, barrier-tight diag)
```

Format (`u64 dim`, `u64 nnz`, `u64 nrhs`, `i64[nnz]` 1-based row
indices, `i64[nnz]` 1-based col indices, `f64[nnz]` vals,
`f64[dim*nrhs]` rhs). Both `.bin` files share sparsity pattern;
they differ only in diagonal values.

Diagnostic binary: `src/bin/diag_issue50_symbolic.rs`.

Usage (one-shot, after the reporter shares the `.bin` file):

```
cargo run --release --bin diag_issue50_symbolic -- path/to/kkt_solve_iter2.bin
```

Prints `total_us`, per-stage `us` and `pct_of_total`, plus
`factor_nnz_estimate` for `OrderingMethod::Auto`, `MetisND`, and
`Amd`. The Auto run resolves to the same concrete method as
`Solver::factor`'s default path (`symbolic_factorize` →
`pick_default_method`); `MetisND` and `Amd` give the explicit
comparison.

Expected reporter numbers (from the issue):

| ordering              | first factor | cached refactor |
|-----------------------|-------------:|----------------:|
| `Auto` (→ `MetisND`)  |     113.28 s |          1.05 s |
| `Amd`                 |      55.03 s |          0.97 s |

## Hypotheses

H1. **The MetisND call itself dominates.** Multilevel
nested-dissection on a 2.8M-node graph with avg_deg 2.4 may scale
poorly because the coarsening passes contract very little (sparse
graphs have few mergeable edges per level), forcing many coarsening
iterations before the partition is small enough to bisect. If H1
holds, `ordering` stage carries most of the 112 s.

H2. **The symbolic post-pass (etree / colcount / supernode detect)
scales poorly at n ≈ 3 M.** Phase 2.5.1's column-count switch was
to Gilbert-Ng-Peyton at O(nnz(A) + n·α(n)), which should be fine
at n = 3 M. But the rebuild path in `find_supernodes`, or the
`Renumber` strategy's etree rebuild, could carry a hidden constant.
If H2 holds, one of `etree_initial`, `col_counts`,
`find_supernodes`, or `renumber` carries most of the cost.

H3. **MUMPS uses METIS too, just with a faster inner config.**
MUMPS calls METIS through `ana_orderings.F` with its own option
vector. If MUMPS-via-METIS is 3 s and feral-via-MetisND is 113 s on
the same matrix, the gap could be in `feral-metis`'s option vector
(coarsening match policy, refinement passes per level) rather than
in feral's post-pass.

The diagnostic binary distinguishes H1 from H2 directly. H3
requires a side-by-side comparison with MUMPS instrumentation on
the same input.

## Decision frame

Three possible outcomes, in order of complexity:

D1. **Tune `feral-metis` options.** If H3 holds, expose the option
vector and pick defaults closer to MUMPS's. Lowest-risk fix; needs
no change to `pick_default_method`. Cost: a few hours plus a
re-validation pass on the IPM corpus.

D2. **Add a size cap to the chain-pattern catch.** If H1 holds and
AMD scales noticeably better at large n, change line `src/symbolic/
mod.rs:296` from
`(n >= 2000 && avg_deg < 4.0)` to
`(n >= 2000 && n <= N_CAP && avg_deg < 4.0)` for some calibrated
`N_CAP`. Needs the full IPM corpus re-bench to ensure CHAINWOO et
al. still hit the catch; needs a representative very-large-sparse
input to fix `N_CAP`. AMD-symbolic at 55 s is still ~18× MUMPS, so
D2 alone does not close the gap.

D3. **Fix or replace a slow post-pass.** If H2 holds, the fix is
algorithmic. Highest payoff (could close to MUMPS) and highest
risk (touches the hot symbolic path).

The order to attempt: instrument, classify the bottleneck, then
pick D1/D2/D3.

## Risks

- **Numeric quality on real IPM iters.** The reporter only timed
  the first factor and one cached refactor under each ordering.
  The IPM-loop cost is `n_iter × (first_factor_share +
  refactor_share)`; AMD-numeric may be slower per refactor even if
  AMD-symbolic is 2× faster, because the cascade structure of the
  power-grid KKT could be worse under AMD ordering (see CHAINWOO
  precedent, `dev/journal/2026-04-27-*.org`). The validation has
  to include a multi-iter refactor trajectory under both orderings,
  not just one factor.

- **Calibration drift.** `pick_default_method`'s second branch is
  load-bearing for CHAINWOO/HYDROELL/DIXMAANH/VESUVIO. Any
  threshold change has to be re-verified against the IPM corpus
  (`tests/ipm_corpus_*`), not just on powerflow22.

- **`feral-metis` API stability.** D1 may require exposing
  partition-coarsening knobs that are currently internal to
  `crates/feral-metis`. Doing so widens the public API of an
  external-crate wrapper.

## Open questions

- Q1. Does the reporter's 113 s break down as mostly-ordering or
  mostly-post-pass? Resolved by running
  `diag_issue50_symbolic` on `kkt_solve_iter2.bin`.

- Q2. What does MUMPS's own ordering choice look like on this
  matrix? `mumps -ICNTL(33)=1` (analysis-only) plus
  `-ICNTL(7)=auto` will print the chosen ordering and the analysis
  time. Probably out-of-scope for feral but cheap to ask the
  reporter for.

- Q3. Is there a representative open-data power-grid KKT we can
  pull into `tests/ipm_corpus_*` so this matrix becomes a
  permanent regression case? `powerflow22` is GAMS-licensed, but
  Texas-2000 / PJM-23k power-grid graphs are in the SuiteSparse
  collection and have similar structure.

- Q4. **Does the reporter's `Auto` actually resolve to `MetisND`?**
  The reporter quotes `pick_default_method` (which
  `symbolic_factorize` calls) and reads the rule
  `n >= 5000 && nnz/n < 6 → MetisND`. But `Solver::new()` defaults
  `ordering = OrderingMethod::Auto`, and the `Auto` path inside
  `symbolic_factorize_with_method` runs `choose_adaptive` first
  (`src/symbolic/mod.rs:142`), which adds a prior branch:
  `n > 100_000 && full_avg_deg < 5.0 → ScotchND`. With
  full_avg_deg ≈ 3.8 (`full ≈ 2·stored − n = 10.6 M`,
  `10.6 M / 2.8 M ≈ 3.8`), the matrix should hit `ScotchND`
  *before* reaching the chain-pattern catch. Three possibilities:
  (a) the resolved method really is `ScotchND` and the reporter
  read the wrong source line — in which case the issue is about
  SCOTCH cost, not METIS; (b) `choose_adaptive` is short-circuited
  in 0.5.0 (older codepath) and only HEAD has it; (c) reporter
  ran with an explicit `MetisND` config and quoted Auto. The
  diagnostic binary prints `resolved_method` from
  `SymbolicFactorization` and will close this directly.

## Status of this note

Skeleton only. The numbers under "Hypotheses" are predicates; none
are evidenced yet. The diagnostic binary below produces the
evidence. Append a "Findings" section after the first profiled run.

## Findings

Run on 2026-05-23 against `kkt_solve_iter2.bin` (located locally at
`pounce/gams/nlpbench/feral_repro/powerflow22/`). Single-threaded
Apple M-series. Loaded matrix: `n = 2 813 976`, stored
`nnz = 6 622 463`, stored `avg_deg = 2.35` (matches the issue).

| ordering    | requested | resolved   | total wall | `ordering` stage | post-pass total | `factor_nnz_estimate` |
|-------------|-----------|------------|-----------:|-----------------:|----------------:|----------------------:|
| `Auto`      | Auto      | **ScotchND** |  114.80 s |        113.84 s (99.2%) |          ~1.00 s |          15 783 342 |
| `MetisND`   | MetisND   | MetisND    |  118.36 s  |        117.44 s (99.2%) |          ~0.92 s |          20 469 771 |
| `Amd`       | Amd       | Amd        |   55.87 s  |         54.97 s (98.4%) |          ~0.90 s |          10 427 637 |

Per-stage detail saved in `/tmp/diag_issue50_out.txt`; the full
per-stage breakdown for all three runs is in
`dev/journal/2026-05-23-01.org` (see §15:00 entry).

### F1 — Q4 settled: `Auto` resolves to `ScotchND`, not `MetisND`

The reporter quoted `pick_default_method` and inferred `Auto →
MetisND`. The actual `Solver::factor` path runs `choose_adaptive`
first (`src/symbolic/mod.rs:142`), and `n > 100_000 && full_avg_deg
< 5.0 → ScotchND` fires on this matrix
(full_avg_deg = `(2·6.622M − 2.814M)/2.814M = 3.71`). ScotchND on
this input is ~3 s *faster* than explicit MetisND (113.8 s vs
117.4 s), so re-routing to MetisND would have made things worse,
not better. The issue's title and prose need a corrigendum.

### F2 — H1 confirmed, H2 and H3 eliminated

99.2% of total symbolic wall is the external `ordering` call.
Every post-pass stage (`etree_initial`, `col_counts`,
`find_supernodes`, etc.) measured under feral's control comes to
~1.0 s total across **all** three orderings. The post-pass at
n = 2.8 M is not slow — it is two orders of magnitude smaller than
the ordering call. H2 (slow feral post-pass) is dead; the fix has
to land in either the external ordering crate or the dispatcher.

H3 (MUMPS uses a faster METIS config) becomes the most likely
remaining cause for the 113 s / 3 s gap vs MUMPS, since both
SCOTCH and METIS through feral are in the same ~115 s ballpark on
this matrix. The 35× factor between feral-METIS and MUMPS-METIS
is unlikely to be partition quality — it has to be loop count or
coarsening config inside the external library. Verifying needs
MUMPS-side instrumentation (out of scope for this issue).

### F3 — AMD also produces 1.5–2× less fill than ND on this matrix

`factor_nnz_estimate`:

- Amd       10.4 M
- ScotchND  15.8 M
- MetisND   20.5 M

The chain-pattern catch's premise — that nested dissection finds
a better separator on low-degree KKT graphs — does not hold on
this matrix. AMD wins on *both* axes here (2× faster symbolic AND
40% less fill). Whether AMD also wins on the multi-iter refactor
trajectory (which is what the IPM loop actually pays for) is the
remaining unknown, but AMD's fill advantage suggests the numeric
phase under AMD will also be at least competitive.

This is the opposite of what `pick_default_method` is calibrated
for. At n ≈ 3000 (CHAINWOO/HYDROELL), the chain catch flipped a
runaway delay cascade (AMD's 2.10M nnz_L → MetisND's 282k nnz_L).
At n ≈ 2.8 M (powerflow22), the chain catch flips the *correct*
ordering (AMD's 10.4 M → MetisND's 20.5 M) in the wrong direction.

### F4 — Decision: D2 (size cap on the chain catch)

D1 (tune feral-metis option vector) does not help: both
feral-metis and feral-scotch are in the same 113–117 s ballpark,
so neither has a faster knob hiding inside it on this matrix.

D3 (fix feral post-pass) is dead — F2.

D2 (size cap on the chain-pattern catch) is the targeted fix.
Proposed change to `pick_default_method` (`src/symbolic/mod.rs:296`):

```rust
// before
if (n >= 5000 && avg_deg < 6.0) || (n >= 2000 && avg_deg < 4.0) {
    return OrderingMethod::MetisND;
}

// proposed
const CHAIN_CATCH_N_CAP: usize = 500_000;
if ((n >= 5000 && avg_deg < 6.0) || (n >= 2000 && avg_deg < 4.0))
    && n <= CHAIN_CATCH_N_CAP
{
    return OrderingMethod::MetisND;
}
```

A parallel change is needed in `choose_adaptive`
(`src/symbolic/mod.rs:142`): the `n > 100_000 && full_avg_deg <
5.0 → ScotchND` branch is what actually catches this matrix
through the `Solver::Auto` path. Either remove that branch
entirely (only justification on file was the 41-matrix shape
bakeoff, c.f. `dev/research/ordering-bakeoff-2026-04-18.md`) or
add a matching size guardrail.

`CHAIN_CATCH_N_CAP = 500 000` is provisional. The known
chain-catch beneficiaries (CHAINWOO_0000 n=4033,
HYDROELL_0000 n≈4k, DIXMAANH n≈3k, VESUVIO n≈10k,
CRESC132_0000 n≈3k) all sit two orders of magnitude below this,
so the cap leaves them in MetisND. The largest cap-relevant
candidates in the existing corpus are reportedly the IPM-dump
matrices around n=100k–200k; none reaches 500k. A wider sweep
needs `cargo run --bin probe_kkt_replay` over `tests/ipm_corpus_*`
with both old and new defaults before committing.

### F5 — Still open

- F5a. Does AMD's ordering actually carry through the IPM-loop
  cost on this matrix? Need to run the reporter's reproducer with
  `FERAL_ORDERING=amd` end-to-end (4 IPM iters) and compare wall
  time including all refactors. If AMD-numeric is comparable to
  ScotchND-numeric per iter (likely from F3's fill advantage), the
  total IPM wall under AMD should drop from 112 s to ~60–70 s
  total — closing about half the gap to MUMPS.

- F5b. The 35× MUMPS-METIS vs feral-METIS gap on the *ordering*
  call itself remains unexplained. Outside scope for the
  dispatcher fix; tracked as a separate follow-up (potential
  feral-metis configuration audit).

- F5c. Does `CHAIN_CATCH_N_CAP = 500_000` regress any large-n
  matrix in the corpus that currently benefits from the catch?
  Validation pending — see F6.

### F6 — Corpus inventory (symbolic): wrong metric, retracted

`src/bin/diag_issue50_inventory.rs` walked all 968 problem dirs
in `data/matrices/kkt{,-expansion}`, classifying each into
`pick_default_method` / `choose_adaptive` branches and recording
`factor_nnz_estimate` per ordering (AMD, MetisND, ScotchND) for
matrices up to `MAX_N = 200 000`. Output:
`/tmp/issue50_inventory.csv`.

Branch distribution: 250 matrices land in the chain-catch path
(206 in the `n≥5000 && avg_deg<6` branch, 52 in the
`n≥2000 && avg_deg<4` branch). Through `choose_adaptive`, only
33 of those actually reach MetisND — the rest re-route to
KahipND via the `n<10_000 && full_avg_deg<15` branch.

**This tally was the wrong metric.** Re-reading
`dev/journal/2026-04-27-02.org` (the original chain-catch
calibration) confirms the catch was selected to prevent a
*numeric-time* delay cascade in BK pivoting, not to improve
symbolic fill estimates. CHAINWOO_0000 has AMD sym 68 k vs
MetisND 70 k (3% gap) but AMD num_nnz_l 2.10 M vs MetisND
282 k (7.5× gap). Symbolic prediction is blind to the cascade
the catch defends against, so any `factor_nnz_estimate`-based
ranking can't speak to whether the catch is doing useful work.

Cap calibration moves to F7 (numeric inventory).

### F7 — Corpus inventory (numeric): cascade is gone

Probe: `src/bin/diag_issue50_numeric_inventory.rs` walks the
968 problem dirs in `data/matrices/kkt{,-expansion}`, filters to
matrices that fire `pick_default_method`'s chain-catch branches
(`n ≥ 5000 && avg_deg < 6` or `n ≥ 2000 && avg_deg < 4`), runs
*numeric* factorization on each under AMD / MetisND / ScotchND
with `Solver::new()` defaults, and records `factor_nnz()`
(post-pivoting `num_nnz_l`) per ordering.

Output: `dev/research/issue-50-numeric-inventory.csv`
(258 matrices; 7 matrices skipped for `MAX_N=200_000` or read
errors; 250 chain-catch rows with all three orderings ok).

**Result: the BK pivoting cascade no longer fires anywhere in
the chain-catch range.**

| n bucket           | matrices | with ratio ≥ 1.5× | max ratio | median |
|--------------------|---------:|------------------:|----------:|-------:|
| [    0,   5_000)   |       52 |                 0 |      1.36 |   0.98 |
| [5_000,  10_000)   |      165 |                 0 |      1.39 |   0.95 |
| [10_000, 25_000)   |       21 |                 0 |      1.22 |   0.92 |
| [25_000, 50_000)   |       10 |                 0 |      1.33 |   1.00 |
| [100_000, 200_000) |        2 |                 0 |      1.00 |   1.00 |

(Ratio = AMD `num_nnz_l` / MetisND `num_nnz_l`. Threshold 1.5×
is the cascade-firing criterion the original chain-catch
calibration used as load-bearing.)

The maximum AMD/MetisND ratio across all 250 matrices is 1.39
(HIER163A / HIER163D, n=9472). The median is below 1.0 in every
bucket — AMD usually produces *fewer* L nnz than MetisND on the
chain-catch-class. Cross-check on the original calibration
target:

```
CHAINWOO_0000 (n=4000):
  current code: AMD num_nnz_l = 22 984    MetisND = 25 579   (AMD wins)
  2026-04-27:   AMD num_nnz_l = 2 101 584 MetisND = 281 526  (7.5× cascade!)
```

A 90× drop in AMD's L fill on the same matrix. The cascade is
gone.

The cause is in commits to `src/dense/factor.rs` / `src/numeric/
factorize.rs` post-2026-04-27, primarily:

- `42434a5` fix(dense): fine-grained delayed pivoting kills the
  BK cascade amplifier (#46) — replaced break-on-first-delay
  with swap-to-boundary delayed pivoting in the two BK driver
  loops.
- `070840b` fix(ldlt): break the zero-(2,2)-block KKT
  delayed-pivot cascade (#46) — two-tier 2×2 partner selection
  in `scalar_pivot_step`.

The chain catch in `pick_default_method` was calibrated against
the *amplified* cascade. With the amplifier removed, the catch
is no longer load-bearing for any chain-catch matrix in the
corpus — and remains harmful on `powerflow22` (n=2.8 M).

### F8 — Revised Fix A: delete the chain catch

The Fix A from F4 (size-cap the catch) is superseded. The catch
itself is obsolete under current numeric defaults. Proposed
change to `pick_default_method` (`src/symbolic/mod.rs:296`):

```rust
// remove:
if (n >= 5000 && avg_deg < 6.0) || (n >= 2000 && avg_deg < 4.0) {
    return OrderingMethod::MetisND;
}
```

The matrices that previously fired the catch now reach the
`n <= 10_000 → Amf` rule or the `n > 10_000 → MetisND`
fallback, both of which produce equal-or-smaller `num_nnz_l`
under the F7 corpus measurement.

`choose_adaptive`'s `n > 100_000 && full_avg_deg < 5.0 →
ScotchND` (`src/symbolic/mod.rs:142`) is the branch that
actually fires for `Solver::Auto` on `powerflow22`. F3 showed
ScotchND loses on that matrix (113.8 s symbolic, 15.8 M nnz_L)
vs AMD (55 s, 10.4 M nnz_L). This branch should be removed at
the same time, with the same justification (cascade defang
makes nested dissection's fill advantage smaller and its
analysis cost prohibitive at very large n).


### F9 — Fix A validation: Auto routing on chain-catch corpus

Probe: `src/bin/diag_issue50_auto_validate.rs`. Walks the 968
IPM corpus reps, filters to chain-catch-class
(`pdm_branch in {chain_metis_5k_6, chain_metis_2k_4}` under the
*pre-Fix-A* classification), and runs `OrderingMethod::Auto`
end-to-end. Output: `dev/research/issue-50-auto-validate.csv`,
258 rows.

Results:

- Status: 250/250 matrices factor `ok`. The 8 non-ok rows are
  intentional `n > MAX_N=200_000` skips (BDRY2, PDE1, PDE2,
  QUADCOPTER, YATP1*, YATP2*), not failures of Fix A.

- Auto resolved-method distribution among the 250 ok rows:
    - 217 → KahipND (`choose_adaptive`'s small-and-sparse
      branch `n < 10_000 && avg_deg < 15.0`; this branch was
      untouched by Fix A and produced KahipND both before and
      after).
    - 29 → MetisND (`n > 10_000` chain-catch matrices, route
      via `pick_default_method`'s `n > 10_000 → MetisND`
      default; behavior identical to pre-Fix-A's chain catch).
    - 4 → AMF (the actual Fix A behavioral change for chain
      catches): BROYDNBD_0000, BRYBNDNE_0000, NONDIANE_0000,
      TRIDIA_0000. All four are n = 10_000 and previously hit
      `chain_metis_5k_6 → MetisND`. Under Fix A they fall
      through to `n <= 10_000 → Amf`.

Joined against `dev/research/issue-50-numeric-inventory.csv`
the 4 AMF picks are equal-or-better than every per-ordering
reference:

| matrix       | Auto AMF nnz | AMD nnz | MetisND nnz | ScotchND nnz |
| ------------ | -----------: | ------: | ----------: | -----------: |
| BROYDNBD     |     144_880  | 144_880 |     202_520 |      203_504 |
| BRYBNDNE     |     144_880  | 144_880 |     202_520 |      203_504 |
| NONDIANE     |      24_998  |  24_998 |      24_998 |       24_998 |
| TRIDIA       |      94_970  | 102_981 |     101_150 |      102_981 |

AMF matches AMD on the first three (and beats AMD on TRIDIA),
strict-improving over MetisND/ScotchND on the bordered cases.

### F10 — Large-and-sparse branch swap: corpus scope

Probe: `src/bin/diag_issue50_large_sparse_scan.rs`. Enumerates
the corpus and reports matrices that hit `choose_adaptive`'s
`n > 100_000 && full_avg_deg < 5.0` branch — the branch whose
target Fix A flipped from `ScotchND` to `Amd`.

Result (`dev/research/issue-50-large-sparse-scan.csv`):

| dir   | n        | full_avg_deg |
| ----- | -------: | -----------: |
| PDE2  | 451_195  |       4.5878 |

One corpus match. PDE2 is also outside the numeric inventory's
MAX_N=200_000 cap so we have no AMD/ScotchND num_nnz_l
comparison for it; the branch swap is justified by the
out-of-corpus `powerflow22` (n = 2.8 M, avg_deg ≈ 3.7) where
F3 measured AMD at 55 s / 10.4 M nnz_L vs ScotchND 113.8 s /
15.8 M nnz_L.

### F11 — Pre-existing KahipND-vs-best-ordering signal

The validate-vs-inventory join (`/tmp/analyze_auto_validate.py`)
flagged 76 matrices with `Auto num_nnz_l > 1.10 × min(AMD,
MetisND, ScotchND)` and 178 with `Auto factor_us > 1.50 × min`.
Almost all are `resolved=KahipND` on small chain-catch
matrices (DIXMAANF-P, FLOSP2HL/TL, BROYDN7D, LCH, OBSTCL*,
NONDQUAR, BDQRTIC, …).

KahipND was the Auto choice on these matrices both *before*
and *after* Fix A — `choose_adaptive`'s small-and-sparse
branch (`n < 10_000 && avg_deg < 15.0 → KahipND`) was not
touched. So these are **pre-existing** dispatcher quality
issues, not Fix A regressions. They deserve a separate
investigation (route small chain-catch matrices to AMD rather
than KahipND?), tracked outside issue #50.

### F12 — F11 follow-up: delete the small-and-sparse branch

F11 flagged the small-and-sparse KahipND branch as a
pre-existing dispatcher quality issue. To decide between
(a) swap to AMD, (b) split predicate, (c) delete branch,
ran `cargo run --release --bin diag_small_sparse_inventory`
over the IPM corpus (extended to include AMF after the first
pass). The probe filters to `n<10_000 && full_avg_deg<15.0`
— the exact predicate the deleted branch matched — and
factors each matrix four ways: AMD, AMF, MetisND, KahipND.

Output: `dev/research/small-sparse-inventory.csv`,
838 matrices with all four orderings ok.

Per-matrix strict wins (by num_nnz_l):

  AMD:     58  ( 6.9%)
  AMF:    169  (20.2%)
  MetisND: 21  ( 2.5%)
  KahipND: 16  ( 1.9%)
  ties:   574  (68.5%)

Pairwise AMF vs KahipND:

  AMF strictly better:    243
  KahipND strictly better: 41
  ties:                   554

Aggregate num_nnz_l (lower is better; normalized to AMD):

  AMD:     1.000x
  AMF:     0.870x  ← winner
  MetisND: 1.005x
  KahipND: 0.984x

Aggregate factor_us (lower is better; normalized to AMD):

  AMD:     1.000x
  AMF:     0.832x  ← winner
  MetisND: 1.135x
  KahipND: 0.990x

The decision is unambiguous: **delete the small-and-sparse
branch entirely** so this population falls through to
`pick_default_method`'s existing `n ≤ 10_000 → Amf` rule
(MUMPS ana_set_ordering.F SYM=2 N≤10000 default).

Where KahipND still wins (the 41 cases): concentrated on
high-avg-deg patterns (STEENBRD, HADAMARD, TABLE8) — all
sub-22k nnz_L absolute, so the regression budget is tiny.
KahipND remains reachable via
`OrderingMethod::KahipND` for callers who profile and pick
it explicitly.

Code change: `src/symbolic/mod.rs::choose_adaptive` now keeps
only the very-large-and-sparse branch (n>100_000, avg_deg<5
→ Amd) on top of `pick_default_method`. Test
`choose_adaptive_rules` updated: the small-and-sparse case
now asserts `Amf` instead of `KahipND`.
