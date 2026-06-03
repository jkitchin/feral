# Research Note: Arrow/bordered-KKT ordering routing (issue #64)

**Status:** Pre-implementation
**Date:** 2026-06-03
**Author:** agent session 2026-06-03-01
**Related issues:** https://github.com/jkitchin/feral/issues/64
**Related code:** `src/symbolic/mod.rs:317` (`pick_default_method`),
`src/symbolic/mod.rs:170` (`choose_adaptive`),
`src/symbolic/mod.rs:395` (`symbolic_factorize`).
**Prior art / precedent:** `dev/research/issue-50-metisnd-symbolic-cost.md`
(the *opposite* routing direction — deleting low-avg-degree → MetisND
escape hatches), `pick_ordering_preprocess` (an existing O(nnz)
degree-distribution predicate in the same file).

## Overview

`pick_default_method` routes the default ordering by `n` alone:
`n <= 10_000 → Amf`, else `MetisND`. It receives `_stored_nnz` and
discards it. On **arrow / bordered-KKT** patterns — a sparse body plus a
handful of very-high-degree "border" columns — nested dissection cannot
isolate the dense border, so the LDLᵀ factor blows up. AMF/AMD's
minimum-degree / min-fill heuristics naturally defer the dense border
columns to the end of the elimination, where they cost one dense
trailing block instead of smeared separator fill.

Found via POUNCE on the LP `r05.nl` (jkitchin/pounce#95): the initial
IPM KKT (n=14842) factors in ~16.5 s under `auto` (→ MetisND) vs 0.84 s
forcing AMF. Ipopt-MA57 (AMD-family) solves it in ~0.9 s. The cost is
entirely in the ordering choice, not the IPM.

This is the **opposite** failure from issue #50, which *deleted* two
escape hatches that routed *low-avg-degree* patterns *toward* MetisND.
Here `nnz/n ≈ 8` (full avg_deg ≈ 15) and the problem is a few dense
borders, not a uniformly thin matrix — so neither the old #50 catches
nor the current size rule help.

## Reproduction (regenerated locally, fixture NOT in repo)

`tests/data/large/` is gitignored (fetched, not committed). The r05 KKT
is a generated IPM matrix, not a SuiteSparse download, so it is
regenerated on demand (see `dev/scripts/regen_r05_kkt.sh`):

```
pounce <bench-data>/lp/nl/r05.nl --dump kkt:0 --dump-dir /tmp/d max_iter=1
jsonl_to_mtx.py /tmp/d/iter_000/kkt_solve_001.jsonl r05   # → r05_kkt.mtx
```

Produces `n=14842 nnz=118968` (matches the issue exactly).

## Structural signature

Full symmetric pattern of r05's iter-0 KKT (measured,
`src/bin/probe_issue64_arrow.rs`):

```
n=14842 full_nnz=223094 avg_deg=15.03 max_deg=502
171 columns of degree 502  (1.15% of n) carrying 38.5% of full_nnz
all other columns degree <= ~20
```

The 171 degree-502 columns are r05's dense inequality rows (85 500 of
103 955 Jacobian nonzeros live in 171 of 5171 constraints). In the
augmented KKT each becomes a dense border column.

Measured fill (default `SupernodeParams`, `col_counts.sum()`):

| ordering | nnz_L (this machine) | issue's numbers | ratio vs AMF |
|---|---|---|---|
| Amf | 506 210 | 526 940 | 1.00× |
| Amd | 607 519 | 568 879 | 1.20× |
| MetisND | 4 358 715 | 3 593 887 | **8.61×** |

Absolute counts drift from the issue (ordering-impl / METIS-seed
variation; pounce 0.3.1-dev vs the 0.9.0-era dump) but the **ranking is
robust**: MetisND is ~7–9× worse. AMF wins on this matrix, consistent
with the issue's end-to-end POUNCE table (amf 0.84 s vs auto 16–21 s).

## The predicate

Route to AMF (instead of MetisND) when, on the full symmetric pattern,
a *small set* of columns carries a *large share* of the nonzeros — the
arrow fingerprint. All thresholds are O(n)/O(nnz) and allocation-free.

```
avg_deg   = full_nnz / n
heavy_thr = max(HEAVY_DEG_FLOOR, HEAVY_AVG_MULT * avg_deg)   = max(64, 8*avg_deg)
heavy     = { columns with degree > heavy_thr }
ARROW iff  heavy.count >= 1
       AND heavy.count  <  ARROW_COUNT_FRAC * n        (0.05*n — a *small* set)
       AND heavy.nnz    >= ARROW_NNZ_SHARE * full_nnz  (0.20    — a *large* share)
```

Rationale for each constant:

- `HEAVY_DEG_FLOOR = 64`, `HEAVY_AVG_MULT = 8`: a "heavy" column is one
  whose degree dwarfs the body. The `max(64, …)` floor stops the
  multiplier from flagging columns in genuinely dense small matrices
  (e.g. bcsstk38, avg_deg 44 → thr 355). 8× is the issue's suggested
  `8*avg_deg`; on r05 that is 121, well below the border degree 502.
- `ARROW_COUNT_FRAC = 0.05`: the border must be a *handful* of columns.
  r05 = 1.15%. If a large fraction of columns are high-degree the matrix
  is just dense, and nested dissection is fine.
- `ARROW_NNZ_SHARE = 0.20`: the border must *concentrate* the nonzeros.
  This is the discriminating guard. r05 = 38.5% → fires; bcsstk38 = 0.3%
  → rejected even though it has 2 columns above `heavy_thr`.

### False-positive analysis (must stay on MetisND)

Measured on the committed-corpus large fixtures and the test-pinned KKTs:

| matrix | n | avg_deg | heavy_count | heavy_nnz share | predicate | current route |
|---|---|---|---|---|---|---|
| r05_kkt | 14842 | 15.0 | 171 (1.15%) | 38.5% | **ARROW→Amf** | MetisND (bug) |
| bratu3d | 27792 | 6.25 | 0 | 0% | no | MetisND |
| cont-201 | 80595 | 5.44 | 0 | 0% | no | MetisND |
| bcsstk38 | 8032 | 44.3 | 2 (0.03%) | 0.3% | no | Amf (n≤10k) |
| PoissonControl K=58 | 10092 | ~2.67 | 0 (uniform) | 0% | no | MetisND |
| powerflow22 | 2.8M | ~3.7 | 0 (uniform) | 0% | no | Amd (#50 branch) |

- bratu3d / cont-201 / PoissonControl / powerflow22 are uniformly
  low-degree (no column exceeds `heavy_thr`) → never flagged.
- bcsstk38 has 2 very-high-degree columns but they carry 0.3% of nnz →
  the share guard rejects it. (It is n≤10k → AMF regardless.)
- The `issue_3_auto_on_kkt_routes_via_pick_default_method` test
  (PoissonControl K=58 → MetisND) is preserved: uniform avg_deg 2.67,
  no heavy columns.

Within feral's domain (IPM KKT + SuiteSparse sparse-direct matrices)
the "small heavy set concentrating the nnz" signature is precisely the
arrow case where deferring the border lets min-degree win. Power-law
graphs where ND still wins despite a dense hub are not part of this
corpus; if one appears, the regression is bounded by the share guard
and surfaces as a fill increase caught by the corpus bench.

## Placement

The detector only ever needs to flip the *would-be-MetisND* decision
(`n > 10_000`) to AMF; it never touches the `n <= 10_000 → AMF` path nor
the `n>100_000 && avg_deg<5 → AMD` (#50) branch. Both entry points must
benefit:

1. `symbolic_factorize` → `pick_default_method(n, stored_nnz)` directly.
2. `symbolic_factorize_with_method(Auto)` → `choose_adaptive(pattern)`.

`pick_default_method` has only `(n, stored_nnz)` and cannot walk the
pattern. `choose_adaptive` already has the full symmetric pattern.
Cleanest unification: route `symbolic_factorize` through `Auto` so all
adaptive logic (the new arrow detector + the existing #50 large-sparse
branch) lives in `choose_adaptive`, the single source of truth. This
also fixes a latent inconsistency: today `symbolic_factorize` lacks the
`n>100k && avg_deg<5 → AMD` branch that `Auto` has, so the two can
already disagree on very-large sparse matrices despite the docstrings
claiming they agree.

`choose_adaptive` builds nothing new — it receives `full_pattern`
(`matrix.symmetric_pattern()`) and the detector is a single extra O(n)
pass over `col_ptr`.

## Routing target: AMF vs AMD

AMF wins on r05 (506k vs AMD 607k here; 527k vs 569k in the issue) and
is already the `n<=10_000` default, so arrow → **Amf** keeps the
dispatcher coherent (small-or-arrow → AMF; large-uniform → MetisND;
very-large-thin → AMD). AMD is the close runner-up; both are ~7–9×
better than MetisND, so the choice between them is second-order against
the bug being fixed.

## Test plan (oracle is external — the issue's measured fill)

- Unit: `is_arrow_bordered` fires on a synthetic arrow, rejects a
  uniform-sparse pattern and a uniformly-dense pattern. (Pattern-shape
  oracle, hand-constructed.)
- Unit: `choose_adaptive` / `pick_default_method`-path returns Amf on a
  synthetic arrow with n>10_000, and still MetisND on uniform n>10_000.
- Regression (skip-if-absent, gitignored fixture):
  `symbolic_factorize(r05_kkt)` yields `nnz_L < 1.0e6` and
  `resolved_method != MetisND`. Oracle: the issue's measured AMF/AMD ≈
  0.53–0.61M vs MetisND 3.6–4.4M; `1e6` separates them with wide margin.
- All existing `choose_adaptive_rules`, `pick_default_method_rules`,
  `issue_3_*`, `symbolic_factorize_default_uses_amf_for_small_matrices`
  must stay green.
