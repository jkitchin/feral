# FERAL Context (auto-generated)

Generated: 2026-08-20T13:59:26Z

## Latest Session
File: dev/sessions/2026-08-20-01.md
```
# Session 2026-08-20-01

## Unfavorable comparison, reported first (per CLAUDE.md)

**A bench run in this session FAILED the Phase 2.8.1 sparse exit gate**:
sparse small-frontal factor p90 **2.03 vs target ≤ 2.0**, against a
baseline of 1.58 from session 2026-08-19-05. It was caused by a change
made in this session and is now fixed; two confirming runs report 1.54
and 1.61, both PASS. The full arc is in **Regression** below, including
the hypothesis I published in the journal that the control run refuted.

**Two conclusions I had posted publicly in the previous session were
wrong and have been retracted**, both from the same methodological
error. See **Corrections** below.

## Goal

Continue the prioritized list from issue #189 / #131:

1. (pounce-side, filed at pounce#698 — no feral work)
2. **#189 items 1–3, the large-n solve gate** ← this session
3. #189 item 4, `BLAS3_NRHS_THRESHOLD` — blocked on (1)
4. #131 — rescope, do not implement `solve_auto`

Before starting (2), settle a contradiction: two probes from the previous
session disagreed about whether the solve measurement was reproducible at
all. They were separate probes on separate runs, so nothing could be
concluded from the pair.

## Corrections — blocked measurement produced two wrong published claims

The previous session's probes timed configuration A to completion, then
configuration B. That lets machine drift between the two blocks appear as
an effect of the configuration. Re-measuring with A and B **interleaved
inside each repetition** removed it. On the strength of the blocked
numbers I had posted:

- **"The measurement is not reproducible"**, with a spread table showing
  1.44×–2.20× movement on the same matrix. **Retracted.** The
  irreproducibility was my method, not the solver.
- **"`cb_core_profitable` looks mis-calibrated"**, specifically that it
  approves `ContribBlock` on `r05_kkt` where the approved configuration
  runs at 0.67×. **Retracted.** Interleaved, `r05_kkt` runs at **1.65×
  faster** — a 2.4× error on the same machine with the same binary. The
  gate is correct on **7 of 7** matrices.

Retractions posted:
- feral#131 — https://github.com/jkitchin/feral/issues/131#issuecomment-5356330763
- feral#189 — https://github.com/jkitchin/feral/issues/189#issuecomment-5356336437

```

## Git Status
```
c09da70 bench: time the solve core hosts actually run (#189 item 1)
2ba437d probe: separate core, schedule, and depth in the solve measurement (#131, #189)
0ac4fb1 probe: measure the solve-phase levers claimed in #131 and #189
9b9e882 Merge pull request #188 from jkitchin/ci/codecov-coverage
1292984 ci: measure coverage with cargo-llvm-cov and report it to Codecov
```

## Test Status
```
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 444 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 1.71s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-20-01.md)


Three full corpus runs plus the 08-19-05 baseline:

| bucket | baseline 08-19-05 | control (unmodified) | run 2 | run 3 |
|---|---:|---:|---:|---:|
| sparse small-frontal p90 (≤2.0) | 1.58 PASS | 1.58 PASS | 1.54 PASS | 1.61 PASS |
| sparse medium p90 (≤3.0) | 1.58 PASS | 1.58 PASS | 1.54 PASS | 1.61 PASS |
| dense small-frontal p90 (≤2.0) | 1.58 PASS | 1.58 PASS | 1.59 PASS | 1.64 PASS |
| dense medium p90 (≤3.0) | 2.00 PASS | 2.00 PASS | 1.96 PASS | 1.98 PASS |

The two post-change runs **bracket** the baseline (1.54, 1.61 vs 1.58).
The honest reading is that the change is indistinguishable from baseline
on the factor partition — which is correct, since it touches only the
solve. **The 1.54 is not claimed as an improvement.** Run-to-run spread
on this machine is ~±2.5%; that is now recorded in the plan as the floor
below which a bench claim is noise.

Run 3, full sparse aggregate:

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.44       0.30       1.61       3.27       9.71
solve/MUMPS        153560       0.08       0.08       0.16       0.88       3.12
factor/SSIDS       154500       0.04       0.03       0.32       0.96       2.26
solve/SSIDS        154500       0.96       1.00       2.83      10.25      43.00
nnzL/MUMPS         153560       0.61       0.58       0.75       4.50      23.11
nnzL/SSIDS         154500       0.88       1.00       1.00       4.50       5.00

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.64     <= 2.0     PASS
medium (<500)            152145     1.98     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.61     <= 2.0     PASS
medium (<500)            153560     1.61     <= 3.0     PASS

Note on the `factor/MUMPS` **max** column: it reads 9.71 here, 68.79 on
the FAIL run, and the worst-offender list changes identity between runs
(CRESC100/MUONSINE ↔ ACOPR14/KIRBY2). That tail is resample-loop noise on
sub-millisecond matrices. **The max is not a stable statistic in this
harness and should not be quoted as a result.**

```

## Recent Decisions
The mechanism is that `resample_or_fallback` (`src/bin/bench.rs:1060`)
runs factor **and** solve interleaved inside one closure,
`RESAMPLE_COLD_REPS = 5` times, and reduces `factor_us` by **min** across
those replicates. Whatever the solve does to cache and allocator state is
the state the next replicate's factor is measured in. And
`should_resample` (`:1042`) fires on `mumps_timing.factor_us < 200` — the
same small matrices the `small-frontal (<200)` bucket gates.

The change had allocated its solve buffer inside the closure: an n-double
alloc-and-zero per replicate at n < 1000. Hoisting it to one allocation
per matrix, reused across replicates, restored the partition to 1.54 PASS
(dense 1.59 / 1.96 PASS) and dropped the worst sparse factor ratio from
68.79 to 9.29.

**The coupling itself is left in place, deliberately.** It is
pre-existing: any solve-side change can perturb the small-matrix factor
reading through it. Fixing it means giving factor and solve separate
timing passes, which changes every small-matrix number in the corpus and
needs its own commit with its own before/after — not a drive-by inside a
different change. Recorded as a Step 1 item in
`dev/plans/large-n-solve-gate.md`.

*Spread, added after a third confirming run:* the sparse small-frontal
p90 reads 1.54 and 1.61 on the two post-fix runs against a 1.58 baseline
— the change is indistinguishable from baseline on the factor partition,
which is the correct outcome for a solve-only change. The 1.54 above is a
single draw, not an improvement. Run-to-run spread on this machine is
~±2.5%; the `factor/MUMPS` **max** column is not stable at all (9.71 here,
68.79 on the failing run, with the offender list changing identity) and
should not be quoted as a result.

## Recent Tried-and-Rejected
total factor time** on a matrix whose paying bucket is 91% of that time. Moving
three quarters of the panel share into BLAS-3 buys 1.3%: the two kernels cost
nearly the same per flop at this front shape, so the 53.5% panel share is not
recoverable time.

It also does not generalize. `bs = 48` and `bs = 64` are identical for any front
with `ncol ≤ 48`, and that is every other matrix sampled — `ncol` p90 is 1-19
across clnlbeam, dtoc2, marine_1600, rocket, steering, gasoil_3200, pinene_3200,
robot_1600, svanberg, nql180, qcqp1500-1c, cont5_2_4_l; only dtoc1nd is at 63. On
the two with any wide fronts at all the paired sweep finds nothing: nql180 0.990
(5/12 wins, tied with the default), qcqp1500-1c 0.994 (3/12). A 1.3% win on one
corpus matrix and a no-op elsewhere is below the bar for changing a global default.

**Kept from this attempt:** `block_size` is bit-neutral on all three matrices
swept — identical inertia, zero delayed pivots, identical residual, and an
identical hash over every `L`/`D` bit in storage order across
`bs ∈ {8,16,24,32,48,62,64}` (`dtoc1nd_0010` 9cb93f568423e6c0, `nql180_0000`
4f588093d6bac8c7, `qcqp1500-1c_0000` cfec17df1a4f8d38). So future retuning of it
is a performance-only change. Not yet established on a matrix that actually
delays a pivot — all three report `d0`.

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
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
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
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/issue102_intrafront_deadlock.rs
tests/issue102_ordering_escalation.rs
tests/issue107_external_ordering.rs
tests/issue112_bg_update.rs
tests/issue127_pipeline_split.rs
tests/issue128_supernode_nrow.rs
tests/issue177_parallel_entry_point_core.rs
tests/issue178_refine_cap.rs
tests/issue178_solve_into.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/issue91_preprocess_misfire.rs
tests/issue99_fma_front_gate.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/lu_adversarial_inputs.rs
tests/lu_default_ordering.rs
tests/lu_dense_bump.rs
tests/lu_dense_update_bg.rs
tests/lu_dense.rs
tests/lu_ft_widebump.rs
tests/lu_hyper_sparse.rs
tests/lu_markowitz.rs
tests/lu_real_bases.rs
tests/lu_scaling.rs
tests/lu_sparse_rhs.rs
tests/lu_sparse.rs
tests/lu_update_alloc_probe.rs
tests/lu_update_casctanks.rs
tests/maxfromm_parity.rs
tests/mc64_end_to_end.rs
tests/mc64_scaling.rs
tests/multi_rhs.rs
tests/n2_static_pivot_scaling.rs
tests/n3_parallel_profiler.rs
tests/n4_mc64_retry_latch.rs
tests/parallel_parity.rs
tests/parity.rs
tests/pivot_rejection.rs
tests/pounce_interface.rs
tests/pounce710_refine_cap_nrhs2.rs
tests/profiler_smoke.rs
tests/property_tests.rs
tests/refined_solve_core_stability.rs
tests/rook_rescue_kkt.rs
tests/rook_rescue.rs
tests/small_leaf_parity.rs
tests/solver_with_ordering.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/sqd_fast_path.rs
tests/static_assembly_maps.rs
tests/stress_tests.rs

(truncated from      355 lines to 350 line budget)
