# FERAL Context (auto-generated)

Generated: 2026-05-23T20:07:42Z

## Latest Session
File: dev/sessions/2026-05-23-01.md
```
# Session 2026-05-23-01

## Goal
Continue issue #50 (slow Auto symbolic factorization on `powerflow22`)
through corpus replay validation, then pursue the F11 side finding
(KahipND vs best-direct-ordering regressions on small chain-catch
matrices in `choose_adaptive`'s small-and-sparse branch) as a follow-up
to #50.

## Accomplished

- **Issue #50 corpus replay validation** (`c442a0c`). Wrote
  `diag_issue50_auto_validate` and `diag_issue50_large_sparse_scan`
  probes. The Auto-validate probe runs `OrderingMethod::Auto` on every
  chain-catch-class representative in the IPM corpus and records the
  resolved method. The large-sparse probe scans for matrices that hit
  `choose_adaptive`'s `n>100_000 && full_avg_deg<5` branch.

  Findings (recorded as §F9–F11 in
  `dev/research/issue-50-metisnd-symbolic-cost.md`):
  - 258 chain-catch corpus rows under post-Fix-A `Auto`: 0 failures,
    0 num_nnz_l regressions vs the AMD/MetisND/ScotchND reference for
    matrices that actually reroute. The four `n=10000` chain matrices
    that change from MetisND to AMF gain on 3 / tie on 1.
  - Large-sparse branch corpus scope: only **PDE2** in the IPM corpus.
    `powerflow22` (the issue-#50 motivating matrix) is out-of-corpus
    and was validated separately.
  - F11 side finding: 76 of the 258 `Auto` rows show
    `num_nnz_l > 1.10 × min(AMD,MetisND,ScotchND)` and 178 show
    `factor_us > 1.50 × min`. Almost all are `resolved=KahipND` on
    small chain-catch matrices (DIXMAANF-P, FLOSP2HL/TL, BROYDN7D,
    LCH, OBSTCL*, NONDQUAR, BDQRTIC, …). KahipND was the Auto choice
    *both* before and after Fix A — these are **pre-existing**
    dispatcher quality issues from `choose_adaptive`'s small-and-sparse
    KahipND branch, not Fix A regressions.

- **F11 follow-up: retire small-and-sparse KahipND branch**
  (`3f8f6f6`). Wrote `diag_small_sparse_inventory` — a 4-way
  ordering probe (AMD/AMF/MetisND/KahipND) over the IPM corpus
  filtered by `choose_adaptive`'s small-and-sparse predicate
  (`n<10_000 && full_avg_deg<15.0`). 838 matrices with all four
  orderings ok. Analyzer at `/tmp/analyze_issue51_v2.py`; durable
  CSV at `dev/research/small-sparse-inventory.csv`. Decision
  evidence (also §F12 of the research note):

  | metric | AMD | AMF | MetisND | KahipND |
  |---|---:|---:|---:|---:|
  | strict per-matrix wins | 58 (6.9%) | **169 (20.2%)** | 21 (2.5%) | 16 (1.9%) |
  | sum num_nnz_l ÷ AMD | 1.000× | **0.870×** | 1.005× | 0.984× |
  | sum factor_us ÷ AMD | 1.000× | **0.832×** | 1.135× | 0.990× |
```

## Git Status
```
3f8f6f6 fix(symbolic): retire small-and-sparse KahipND branch
c442a0c fix(symbolic): retire obsolete chain catch and ScotchND large-and-sparse branch (#50)
407180e build: add release-checklist.sh to keep release versions in sync
dfb5029 release: bump Python package feral-solver to 0.5.0
33389bf release: v0.5.0
```

## Test Status
```
