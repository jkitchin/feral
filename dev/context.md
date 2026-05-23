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
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-b9fe9995ece84f0e)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_gate_boundary_n_16 ... ok
test test_determinism_tiny ... ok
test test_solve_parity_tiny_real_matrix ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 1 test
test src/symbolic/profiler.rs - symbolic::profiler::SymbolicProfiler (line 27) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-23-01.md)


All exit-partition buckets PASS. First bench run showed a small
upward p90 drift across all four buckets and two new top-10 outliers
(DISCS_0007 50.94×, DISCS_0006 26.66×). Because the `Auto` dispatcher
edits this session don't touch n=234 routing (still AMF on the
small-and-sparse fall-through both before and after), this looked
like noise, so the bench was rerun. The rerun confirms noise:

| bucket | last session | run #1 | rerun | target | verdict |
|---|---:|---:|---:|---:|---:|
| Dense small-frontal (<200) | 1.32 | 1.36 | **1.35** | ≤ 2.0 | PASS |
| Dense medium (<500)        | 1.71 | 1.78 | **1.78** | ≤ 3.0 | PASS |
| Sparse small-frontal (<200)| 1.53 | 1.62 | **1.54** | ≤ 2.0 | PASS |
| Sparse medium (<500)       | 1.53 | 1.62 | **1.54** | ≤ 3.0 | PASS |

DISCS_0007 / DISCS_0006 dropped out of the rerun's top-10 entirely;
KIRBY2_0007 snapped from 9.49× back to 7.92× (vs last session 8.03×),
and CRESC132_0000 from 8.13× to 6.77× (vs 7.13×). The first run was
co-tenant with a concurrent clippy build, which plausibly explains
the bump on the small-n end. Sparse p90s returned to last session's
values; Dense medium held at 1.78 but is comfortably under the 3.0
target. No regression, no follow-up needed.

Rerun output (the run preserved in this checkpoint):

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
PALMER7A                 3000       0.19       0.20       0.33
MGH10LS                  3000       0.10       0.11       0.25
HS90                     3000       0.20       0.20       0.33
ALLINITA                 3000       0.39       0.36       0.83
HS13                     3000       0.16       0.20       0.25
HATFLDBNE                3000       0.39       0.40       0.83
HATFLDH                  3000       0.42       0.45       0.55
CONCON                   3000       0.84       0.88       1.84
HS92                     3000       0.35       0.36       0.56
SSINE                    3000       0.27       0.27       0.36
HS89                     3000       0.19       0.20       0.30
DJTL                     3000       0.09       0.10       0.12
HS91                     3000       0.18       0.20       0.40
HS118                    3000       0.92       1.00       1.20
SSI                      3000       0.17       0.20       0.33
ALLINITC                 3000       0.19       0.20       0.27
MCONCON                  3000       0.88       0.93       1.84
PALMER5A                 3000       0.29       0.30       0.40
BIGGSC4                  3000       0.43       0.45       0.60
AVION2                   2682       1.55       1.58       2.27
CERI651ALS               2331       0.27       0.27       0.36
PFIT4                    2286       0.17       0.18       0.30
CERI651C                 2233       0.28       0.30       0.33
CERI651CLS               2227       0.26       0.27       0.33
BATCH                    2054       1.28       1.33       1.79

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
KIRBY2_0007                    458          943          119       7.92
KIRBY2_0006                    458          933          127       7.35
CRESC132_0000                 5314        83026        12266       6.77
KIRBY2_0008                    458          805          122       6.60
KIRBY2_0009                    458          716          128       5.59
MUONSINE_0000                 1537         2047          376       5.44
KIRBY2_0011                    458          624          120       5.20
KIRBY2_0010                    458          674          133       5.07
KIRBY2_0012                    458          485          118       4.11
GROUPING_0045                  225          461          113       4.08

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.35     <= 2.0     PASS
medium (<500)            152145     1.78     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.54     <= 2.0     PASS
medium (<500)            153560     1.54     <= 3.0     PASS

Run-#1 numbers preserved for the record (since the rerun confirmed
they were artifact, not signal):

Top 10 worst factor-ratio vs MUMPS (run #1, co-tenant with clippy):
DISCS_0007                     234        27713          544      50.94
DISCS_0006                     234        14636          549      26.66
KIRBY2_0007                    458         1129          119       9.49
KIRBY2_0008                    458          996          122       8.16
CRESC132_0000                 5314        99775        12266       8.13
KIRBY2_0006                    458          970          127       7.64
KIRBY2_0009                    458          928          128       7.25
KIRBY2_0010                    458          941          133       7.08
SWOPF_0026                     175          472           75       6.29
MUONSINE_0000                 1537         2232          376       5.94

--- Dense Phase 2.8.1 exit partition (run #1) ---
small-frontal (<200)     147982     1.36     <= 2.0     PASS
medium (<500)            152145     1.78     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (run #1) ---
small-frontal (<200)     153455     1.62     <= 2.0     PASS
medium (<500)            153560     1.62     <= 3.0     PASS

```

## Recent Decisions
  are concentrated on high-avg-deg patterns (STEENBRD, HADAMARD,
  TABLE8) and remain reachable via `OrderingMethod::KahipND`.
  See `dev/research/issue-50-metisnd-symbolic-cost.md` §F12.

**Consequences.**

- `Auto` is now a thin wrapper around `pick_default_method` plus a
  single guard for very-large-and-sparse matrices. The dispatcher
  no longer reaches for `KahipND` or `ScotchND` implicitly; callers
  who want those orderings must request them explicitly via
  `with_ordering`. This matches the explicit guidance in
  `OrderingMethod::Auto`'s doc comment: `Auto` is opt-in for known
  IPM workloads, and the default `symbolic_factorize` still uses
  `Amd`.

- The 4-matrix `n=10000` chain reroute (Fix A side effect) and the
  PDE2 + powerflow22 reroute are the entire observed behavior
  delta on the IPM corpus — every other Auto pick is unchanged.

- No correctness change: every reroute produced `Success`
  inertia matching the pre-fix path.

**References.**
- Commits `c442a0c` (#50 Fix A), `3f8f6f6` (F11 follow-up: retire
  small-and-sparse KahipND branch).
- `dev/research/issue-50-metisnd-symbolic-cost.md` §F7–§F12.
- `dev/sessions/2026-05-22-01.md` (Fix A research) and
  `dev/sessions/2026-05-23-01.md` (corpus validation + small-and-
  sparse retire).
- `CHANGELOG.md` Unreleased entries.

## Recent Tried-and-Rejected
**Removing the contrib-block zero-fill — not free, not pursued.**
The 16:00 journal claim "the `resize(cdim*cdim, 0.0)` is 100%
removable, provably safe" was wrong: it checked only `extend_add` (a
lower-triangle-only reader). Grepping every reader of `.contrib` found
**three consumers that bit-compare the full contrib Vec including the
upper triangle**: the `block_ldlt32` unit test (`to_bits()` per
element), `parallel_corpus_parity.rs:70`, and `diag_par_firstdiff.rs`.
The zero-fill is what makes the upper triangle deterministically
`0.0`; deleting it naively regresses the test and breaks parity.
Removing only its cost requires `unsafe Vec::set_len` — safe Rust
cannot length a `Vec` without N initializing writes, and `src/` has no
`unsafe` in the core numeric data path. The genuinely-wasted portion
is ~2% (the lower-triangle zeros the copy overwrites anyway); the
other half is load-bearing. Decision (jrk): not worth the first
core-path `unsafe` for ~2% on an already-correct solver. Issue #44
closed.

**Lesson.** "Provably dead" requires grepping *all* consumers of the
buffer, not the one obvious algorithmic reader. Diagnostic and test
binaries that bit-compare whole buffers make "never read" false.

## Source Files
```
src/bin/alloc_probe.rs
src/bin/bench_axpy_small.rs
src/bin/bench_fma_phase3.rs
src/bin/bench_issue8.rs
src/bin/bench_one_matrix.rs
src/bin/bench_orderings.rs
src/bin/bench_solver_corpus.rs
src/bin/bench_solver_reuse.rs
src/bin/bench_sqd.rs
src/bin/bench.rs
src/bin/blas3_prototype.rs
src/bin/calibrate_par_min_flops.rs
src/bin/d3_probe.rs
src/bin/d4_probe.rs
src/bin/diag_acopp30_residual.rs
src/bin/diag_acopr.rs
src/bin/diag_acopr14.rs
src/bin/diag_amalgamation.rs
src/bin/diag_amd_substages.rs
src/bin/diag_amf_vs_amd.rs
src/bin/diag_cascade_default_evidence.rs
src/bin/diag_cascade_ratio_distribution.rs
src/bin/diag_chainwoo_profile.rs
src/bin/diag_chainwoo.rs
src/bin/diag_clnlbeam_maxfromm.rs
src/bin/diag_clnlbeam_slb.rs
src/bin/diag_compress_costbenefit.rs
src/bin/diag_compress_profile.rs
src/bin/diag_compression_bench.rs
src/bin/diag_cond_parity.rs
src/bin/diag_dense_tail.rs
src/bin/diag_etree_shape.rs
src/bin/diag_factor_nnz_accounting.rs
src/bin/diag_fbrain3ls_pivtol_sweep.rs
src/bin/diag_fill_parity.rs
src/bin/diag_fill_tail.rs
src/bin/diag_inertia_mismatch.rs
src/bin/diag_issue50_auto_validate.rs
src/bin/diag_issue50_inventory.rs
src/bin/diag_issue50_large_sparse_scan.rs
src/bin/diag_issue50_numeric_inventory.rs
src/bin/diag_issue50_symbolic.rs
src/bin/diag_leaf_profile.rs
src/bin/diag_max_ncol.rs
src/bin/diag_mc64_cycles.rs
src/bin/diag_mittelmann.rs
src/bin/diag_narx_kernel_gflops.rs
src/bin/diag_near_singular_sweep.rs
src/bin/diag_nemin_amalgamation_panel.rs
src/bin/diag_orbit2_quotient.rs
src/bin/diag_ordering_panel.rs
src/bin/diag_ordering_race.rs
src/bin/diag_par_firstdiff.rs
src/bin/diag_par_frontal_hash.rs
src/bin/diag_par_repeat.rs
src/bin/diag_parent_unique.rs
src/bin/diag_phase_b_nemin_sweep.rs
src/bin/diag_pinene_0009_profile.rs
src/bin/diag_pinene_amd.rs
src/bin/diag_pinene_pivot_cliff.rs
src/bin/diag_pinene_static_pivot.rs
src/bin/diag_poisson_kkt.rs
src/bin/diag_qcqp_knobs.rs
src/bin/diag_qcqp_profile.rs
src/bin/diag_robot1600_eigs.rs
src/bin/diag_schur_parity.rs
src/bin/diag_small_leaf_gate.rs
src/bin/diag_small_leaf.rs
src/bin/diag_small_sparse_inventory.rs
src/bin/diag_sparse_memory.rs
src/bin/diag_split_tail.rs
src/bin/diag_strategy_compare.rs
src/bin/diag_supernode_cost.rs
src/bin/diag_swopf_w22x2.rs
src/bin/diag_symbolic_stages.rs
src/bin/dump_diff.rs
src/bin/feral_replay.rs
src/bin/feral_time.rs
src/bin/hs85_diag.rs
src/bin/parallel_corpus_parity.rs
src/bin/polak6_diag.rs
src/bin/policy4_diag.rs
src/bin/probe_acopp30_64.rs
src/bin/probe_cache_sequence.rs
src/bin/probe_cascade_perturb.rs
src/bin/probe_clnlbeam_refine.rs
src/bin/probe_clnlbeam_shape.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_dtoc2_mc64.rs
src/bin/probe_explicit_zeros.rs
src/bin/probe_f01.rs
src/bin/probe_fbrain.rs
src/bin/probe_fma_kernel.rs
src/bin/probe_hang_loop.rs
src/bin/probe_ir_trajectory.rs
src/bin/probe_issue_19.rs
src/bin/probe_issue45_ordering.rs

(truncated from      469 lines to 350 line budget)
