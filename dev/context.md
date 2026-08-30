# FERAL Context (auto-generated)

Generated: 2026-08-30T04:00:52Z

## Latest Session
File: dev/sessions/2026-08-30-02.md
```
# Session 2026-08-30-02

## Benchmark note (read first)

No new corpus regression to report, but two benchmark facts belong at the
top because both cut against a claim made earlier in the session:

1. **A single-shot Phase 2.8.1 run on #191 FAILED, and that result was
   wrong.** The failing number was produced while a busy-wait loop I had
   started (`until grep -q ...; do :; done`, no sleep) was pinning a core
   against a live benchmark. Load hit 15.71. MUMPS timings are read from
   on-disk oracle sidecars, so contention inflates only feral's numerator
   -- the ratio is not contention-symmetric and a loaded host reads as a
   regression. Do not trust any single-shot p90 taken under load.

2. **The interleaved re-run cleared it, 8/8 PASS.** Round 1 (equal load):
   `main` dense p90 1.60 / 2.00, sparse 1.58 / 1.58; branch 1.62 / 2.00,
   1.63 / 1.63. Round 2 (branch deliberately at load 15.71 vs `main` at
   9.67): `main` 1.62 / 2.00 / 1.64 / 1.64; branch 1.62 / 2.00 / 1.61 /
   1.61 -- the branch measured *better* under materially worse conditions.
   A delta that flips sign across rounds is noise, not signal. Both exit
   partitions PASS on `main` (small-frontal target <= 2.0, medium <= 3.0).

Lesson recorded in the journal: any background waiter in this repo must
`sleep`, never busy-poll, while a benchmark is live.

## Goal

Review PR #191 (`fix/componentwise-refine-default`) and #193, apply the
findings, and land both; answer the downstream comment on #191 asking
whether `RefineOutcome` carries the achieved omega; then review and land
#195 (issue #194, cooperative cancellation). All three are 0.18.0 material
and pounce asked that they land together so downstream does one pin bump
and one sweep.

## Accomplished

### PR #191 -- reviewed, fixed, merged (f1dc5ee)

Eight review findings, all verified against the source before acting, all
fixed. Two were material:

1. **`backward_error` returned `0.0` on a non-finite iterate**
   (`src/numeric/solve.rs`). `NaN` loses every `>` comparison, so a row
   whose `term` was `NaN` was silently skipped; an overflowed iterate makes
   *every* row `NaN` (`x -> inf`, so `|A||x|` and `|r|` both reach `inf`
   and the ratio is `NaN`), and the function returned `omega = 0.0` -- a
   diverged solve certifying itself as exactly backward stable. Under
   `StopCriterion::BackwardError(t)`, which #191 newly exposes, `reached()`
   is then true on the first test: the loop breaks immediately and returns
```

## Git Status
```
5d567e1 Merge remote-tracking branch 'origin/main' into claude/issue-194-p6gri8
f112b93 docs: record the #195 review round; halve the interrupt tests' debug cost
91ace05 docs: session checkpoint 2026-08-30-02 (#191, #193, #195 reviewed and landed)
d0b3000 Merge pull request #195 from jkitchin/claude/issue-194-p6gri8
7b42414 fix(solve): an interrupt during the MC64 retry must not be swallowed
```

## Test Status
```
test symbolic::tests::schur_symbolic_tail_invariant_reversed_user_order ... ok
test symbolic::tests::schur_symbolic_tail_invariant_user_order ... ok
test symbolic::tests::symbolic_factorize_amf_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_auto_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
test symbolic::tests::symbolic_factorize_external_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 464 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 3.54s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-30-02.md)


Run on `main` at d0b3000, i.e. with #191, #193 and #195 all merged. All
four Phase 2.8.1 exit partitions PASS: dense 1.63 / 2.00, sparse 1.55 /
1.55, against targets of 2.0 (small-frontal) and 3.0 (medium). Those are
consistent with the interleaved A/B numbers above (1.60-1.64 dense, 2.00
medium), so the session's landings cost nothing measurable.

**Caveat, stated because the top of this checkpoint argues for it:** this
run was *not* taken on an idle host. A `python` process belonging to a
different project's session held ~98% of a core for the whole walk (load
6.84 at start). It was not mine to kill. Since the p90s are ratios against
on-disk MUMPS oracle sidecars, contention inflates only feral's numerator,
so these numbers are if anything pessimistic -- they PASS despite the
load, which is the direction that makes the verdict safe to trust. A clean
baseline still needs to be taken on a quiet host before 0.18.0 ships.

This run also closes the gap left open by session 2026-08-30-01, whose
checkpoint honestly recorded that it could not compare against the
previous session because the 154k-matrix oracle corpus was not present in
its container and both exit-partition tables read `N/A`. The corpus is
present here, and #195's code is in the tree measured above.

AVION2                   2682       1.80       1.92       2.30
CERI651ALS               2331       0.09       0.09       0.11
PFIT4                    2286       0.09       0.09       0.10
CERI651C                 2233       0.09       0.10       0.11
CERI651CLS               2227       0.09       0.09       0.11
BATCH                    2054       3.84       4.04       4.58

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
GAUSS2_0034                    758        22470          246      91.34
GAUSS2_0016                    758        22459          246      91.30
CRESC100_0000                  806        21589          238      90.71
GAUSS2_0017                    758        22665          251      90.30
GAUSS2_0035                    758        22565          250      90.26
GAUSS2_0008                    758        21933          245      89.52
GAUSS2_0029                    758        22512          252      89.33
GAUSS2_0025                    758        22687          254      89.32
GAUSS2_0032                    758        22394          257      87.14
GAUSS2_0024                    758        22353          257      86.98

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.44       0.30       1.55       3.30       9.39
solve/MUMPS        153560       0.08       0.08       0.20       1.00       4.18
factor/SSIDS       154500       0.04       0.03       0.32       0.96       1.93
solve/SSIDS        154500       1.05       1.00       3.50      12.00      53.25
nnzL/MUMPS         153560       0.61       0.58       0.75       4.50      23.11
nnzL/SSIDS         154500       0.88       1.00       1.00       4.50       5.00

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
HATFLDH                  3000       0.43       0.45       0.60
CONCON                   3000       0.83       0.88       1.69
HS89                     3000       0.20       0.20       0.30
HATFLDBNE                3000       0.41       0.40       1.50
PALMER5A                 3000       0.29       0.30       0.40
BIGGSC4                  3000       0.43       0.45       0.60
HS13                     3000       0.17       0.20       0.33
DJTL                     3000       0.09       0.10       0.20
HS92                     3000       0.35       0.40       0.50
MCONCON                  3000       0.88       0.93       1.69
SSI                      3000       0.21       0.22       0.25
ALLINITA                 3000       0.40       0.40       0.92
ALLINITC                 3000       0.19       0.20       0.27
SSINE                    3000       0.27       0.27       0.33
HS91                     3000       0.27       0.30       0.44
MGH10LS                  3000       0.21       0.22       0.25
PALMER7A                 3000       0.28       0.30       0.44
HS90                     3000       0.20       0.20       0.33
HS118                    3000       0.92       0.95       1.20
AVION2                   2682       1.47       1.50       2.09
CERI651ALS               2331       0.28       0.30       0.40
PFIT4                    2286       0.25       0.27       0.33
CERI651C                 2233       0.28       0.30       0.33
CERI651CLS               2227       0.27       0.27       0.33
BATCH                    2054       1.34       1.40       1.92

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
KIRBY2_0007                    458         1117          119       9.39
KIRBY2_0006                    458         1080          127       8.50
KIRBY2_0008                    458          921          122       7.55
KIRBY2_0009                    458          818          128       6.39
KIRBY2_0010                    458          775          133       5.83
KIRBY2_0011                    458          692          120       5.77
GROUPING_0097                  225          659          119       5.54
GROUPING_0285                  225          621          118       5.26
CRESC132_0000                 5314        62242        12266       5.07
GROUPING_0217                  225          561          112       5.01

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.63     <= 2.0     PASS
medium (<500)            152145     2.00     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.55     <= 2.0     PASS
medium (<500)            153560     1.55     <= 3.0     PASS

```

## Recent Decisions

**Rejected alternative.** Computing ω unconditionally so the field is always
a number. That charges every caller an `abs_symv` per step for a quantity
their chosen criterion does not use, which is exactly the cost `EpsSqrtN`
exists to avoid.

## 2026-08-29 — a cancellation is not evidence about MC64

**Context.** `Solver::factor` may run two factorizations: when the first
reports `inertia.zero > 0` under non-MC64 `Auto` scaling, the issue-#65
rescue re-factors with `Mc64Symmetric` and adopts iff the zero count strictly
drops. Non-adoption arms `mc64_retry_not_adopted`, a latch keyed on the
pattern fingerprint and cleared only on a pattern change, so subsequent
same-pattern `factor()`s skip the retry.

**Decision.** `mc64_retry_not_adopted` may only be armed by a retry that ran
to completion and genuinely failed to reduce the zero count. Any `Err` out of
the retry — `FeralError::Interrupted` today, and by the same argument any
future error variant — propagates and leaves the latch disarmed.

**Why this is a correctness rule, not a tuning choice.** The latch is a cache
of a *measurement*: "MC64 does not help on this pattern." Arming it without
having taken the measurement gates off the inertia rescue on false pretences.
Because the latch survives every refactorization of the same pattern, and an
interior-point host holds one pattern for a whole solve, a single unrelated
cancellation would suppress the rescue for every remaining iterate and report
unrescued inertia — the one hard constraint in `CLAUDE.md`. The
`mc64_retry_not_adopted` doc already flags this interaction; it reasons about
the latch being armed by genuine evidence, and that premise must be enforced
at every arming site.

## Recent Tried-and-Rejected
**Replaced with.** Two fix-independent observables:
`mc64_retry_attempt_count() == 1` proves factorization #1 returned `Ok(..)`
(the issue-#65 gate keys on `Ok`, so the flag was set after it completed),
and `delay < call_elapsed` proves it was set before the call returned.
Together they pin the flag inside the retry without referencing the status
being asserted.

## 2026-08-29 — busy-wait shell pollers while a benchmark is live

**Tried.** `until grep -q ...; do :; done` to wait on a background job.

**Symptom.** Pinned a core, pushed load to 15.71 alongside `cargo test --all`
and a live A/B benchmark, and contaminated the branch arm of round 2. Phase
2.8.1 p90s are ratios against **on-disk MUMPS oracle sidecars**, so
contention inflates only feral's numerator — the metric is not
contention-symmetric and a loaded host reads as a regression. A single-shot
run under that load FAILED; the interleaved re-run was 8/8 PASS.

**Rule.** Any background waiter in this repo must `sleep`, never busy-poll.
Do not trust a single-shot p90 taken under load; interleave A/B.

## Source Files
```
src/bin/bench.rs
src/bin/perf_probe.rs
src/bin/probe_ft_eta.rs
src/bin/probe_lu_phases.rs
src/bin/probe_panel_frag.rs
src/capi.rs
src/dense/block_ldlt32.rs
src/dense/equilibrate.rs
src/dense/factor.rs
src/dense/matrix.rs
src/dense/mod.rs
src/dense/rook.rs
src/dense/schur_kernel.rs
src/dense/solve.rs
src/env.rs
src/error.rs
src/inertia.rs
src/io/mod.rs
src/io/mtx.rs
src/io/sidecar.rs
src/lib.rs
src/lu/condition.rs
src/lu/dense_factor.rs
src/lu/dense_matrix.rs
src/lu/dense_solve.rs
src/lu/dense_update.rs
src/lu/markowitz.rs
src/lu/mod.rs
src/lu/scaling.rs
src/lu/sparse_factor.rs
src/lu/sparse_hyper.rs
src/lu/sparse_matrix.rs
src/lu/sparse_solve.rs
src/lu/sparse_symbolic.rs
src/lu/sparse_triangular.rs
src/lu/sparse_update.rs
src/numeric/condition.rs
src/numeric/factorize.rs
src/numeric/mod.rs
src/numeric/solve.rs
src/numeric/solver.rs
src/ordering/amd.rs
src/ordering/elimination_tree.rs
src/ordering/mod.rs
src/ordering/postorder.rs
src/ordering/schur.rs
src/scaling/hungarian.rs
src/scaling/infnorm.rs
src/scaling/mc64.rs
src/scaling/mod.rs
src/scaling/value_bound.rs
src/sparse/csc.rs
src/sparse/mod.rs
src/symbolic/column_counts.rs
src/symbolic/ldlt_compress.rs
src/symbolic/mod.rs
src/symbolic/profiler.rs
src/symbolic/small_leaf.rs
src/symbolic/supernode.rs
```

## Test Files
```
tests/amf_corpus_oracle.rs
tests/auto_strategy.rs
tests/blocked_ldlt.rs
tests/build_row_indices_trailing_invariant.rs
tests/cb_core_choice_ignores_env.rs
tests/cb_solve_parity.rs
tests/column_renumbering.rs
tests/column_renumbering_parity.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/env_knob_parsing.rs
tests/env_knob_scan.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
tests/factors_ld_export.rs
tests/fine_grained_delay.rs
tests/fma_opt_in_roundtrip.rs
tests/golden_bits.rs
tests/growth_flag.rs
tests/issue102_intrafront_deadlock.rs
tests/issue102_ordering_escalation.rs
tests/issue107_external_ordering.rs
tests/issue112_bg_update.rs
tests/issue127_pipeline_split.rs
tests/issue128_supernode_nrow.rs
tests/issue177_parallel_entry_point_core.rs

(truncated from 417 lines to 350 line budget)
