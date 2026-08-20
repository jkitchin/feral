# FERAL Context (auto-generated)

Generated: 2026-08-20T17:40:42Z

## Latest Session
File: dev/sessions/2026-08-20-02.md
```
# Session 2026-08-20-02

## Unfavorable comparison, reported first (per CLAUDE.md)

The session-end bench is at or slightly above the top of the previously
observed band on the factor partitions:

| bucket | baseline 08-19-05 | 08-20-01 runs | this run |
|---|---:|---:|---:|
| sparse small-frontal p90 | 1.58 | 1.54 / 1.61 | **1.65** |
| sparse medium p90 | 1.58 | 1.54 / 1.61 | **1.65** |
| dense small-frontal p90 | 1.58 | 1.59 / 1.64 | **1.66** |
| dense medium p90 | 2.00 | 1.96 / 1.98 | **2.04** |

All four still PASS their targets, and `solve/MUMPS` geomean and p50 held at
0.08 exactly. But this is the worst of the four runs recorded across the two
sessions, and it is +4.4% on the sparse buckets against the 08-19-05 baseline.

**This session's change cannot mechanically account for it.** The bench times
factorization and a *single-RHS* solve; the change is confined to
`solve_sparse_core_many_into` and the panel GEMM reached only from
`fwd_blas3`/`back_blas3`, which the multi-RHS path alone calls. The
single-RHS core (`solve_sparse_core_into`) and the whole factorization are
untouched.

The likely cause is machine state — the box had been under sustained compile
and benchmark load for hours before this run. That is a hypothesis, not a
result: I did not run a control on stashed code to test it, as was done on
08-20-01 to refute a similar attribution.

> **RESOLVED, same day.** The cold re-run (built cold, 180 s settle, machine
> quiet) put all four buckets at the *bottom* of the band: sparse
> small-frontal **1.54**, sparse medium **1.54**, dense small-frontal
> **1.59**, dense medium **1.97**. Two independent machine-state signatures
> confirm the hot run rather than the change: `factor/MUMPS` max fell
> **71.94 -> 9.10** and the worst-offender list changed identity entirely
> (GAUSS2/CRESC100 cold vs HAIFAM/HAHN1 hot); `factor/SSIDS` p99 fell
> 1.11 -> 0.95. `a0b4d64` did not regress the factor path.

## Goal

Answer issue #189 item 4: is `BLAS3_NRHS_THRESHOLD = 32` costing pounce
anything? Under the standing bar for this work — rigorous, thoroughly
correct, and a real performance gain, or it does not ship.

## Accomplished

### The threshold is fine. There was a defect underneath it.

`BLAS3_NRHS_THRESHOLD` is unchanged at 32. What the investigation found
```

## Git Status
```
2637964 docs: state which pounce call sites the padded stride actually reaches
ffb5862 docs: session checkpoint 2026-08-20-02 (multi-RHS stride alignment)
a0b4d64 perf(solve): align the multi-RHS row stride to a cache line
c34475e docs: session checkpoint 2026-08-20-01 (measurement corrections, #189 Step 1)
c09da70 bench: time the solve core hosts actually run (#189 item 1)
```

## Test Status
```
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test numeric::solve::tests::cb_coarsening_threshold_is_arithmetically_inert ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 445 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 1.65s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-20-02.md)


ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.46       0.33       1.65       4.13      71.94
solve/MUMPS        153560       0.08       0.08       0.16       0.88      25.59
factor/SSIDS       154500       0.04       0.03       0.34       1.11      23.34
solve/SSIDS        154500       0.96       1.00       2.83      10.25     583.65
nnzL/MUMPS         153560       0.61       0.58       0.75       4.50      23.11
nnzL/SSIDS         154500       0.88       1.00       1.00       4.50       5.00

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.66     <= 2.0     PASS
medium (<500)            152145     2.04     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.65     <= 2.0     PASS
medium (<500)            153560     1.65     <= 3.0     PASS

```

## Recent Decisions
Appended rather than editing the entry above, per the append-only rule. The
preceding entry states the padded-stride decision correctly but says nothing
about how much of a real host it reaches; a reader could take "1.36x on the
multi-RHS solve" for "1.36x on pounce". It is much narrower than that.

Prompted by pounce issue #698, comment 5359027510, which named a multi-RHS
call site the 2026-08-20-02 checkpoint had not enumerated.

pounce has three multi-RHS-capable call sites. The padded stride reaches one:

| pounce call site | nrhs | reached? |
|---|---|---|
| `std_aug_system_solver.rs:497,625` (IPM) | 1, hardcoded | no |
| `pounce-feral/src/lib.rs:1004,1006` (batched backsolve) | 6 | no — below the threshold |
| `pounce-feral/src/schur.rs:303,304` | `n_s` | yes, when `n_s >= 32` |

The batched backsolve's width is `limited_memory_max_history`, which defaults
to 6 (`alg_builder.rs:1037`, `:1508`). Below `BLAS3_NRHS_THRESHOLD = 32`, so
it takes the rank-1 kernel, which the change does not touch. That dispatch is
already correct — measured rank1/blas3 at `nrhs = 6` is 0.68-0.86 across all
seven probe matrices, i.e. rank-1 wins — so there is no unclaimed gain there
and no argument for lowering the threshold to capture it.

The Schur path is the one that benefits, and it can be wide:
`schur_aug_system_solver.rs:36` sets `DEFAULT_MAX_SCHUR_FRAC = 0.5`, so `n_s`
reaches half the KKT dimension before the backend refuses the partition.

**Consequence for release decisions:** the 1.36x figure is a property of the
multi-RHS solve at `nrhs >= 32` and not a multiple of 8. It is not a pounce
end-to-end number and must not be quoted as one.

## Recent Tried-and-Rejected
**Why, as far as the measurement shows.** Pre-mask, cost was a pure function
of `padded_ldw(nrhs)`, confirmed three ways: `t(36) ~ t(40)` on all seven
matrices (7059 vs 7111 us on bcsstk38; 104460 vs 104470 on dirichlet120),
`t(47) ~ t(48)` on all seven, and `t(33)/t(32) = 1.225` against `40/33 =
1.212` predicted. Post-mask, cost tracks neither model — 1.490 at `nrhs = 33`
is worse than the 1.250 pad model *and* far worse than the 1.031 work model.

Mechanism is inference, not measurement: iterating a non-multiple-of-8 column
count appears to cost more than the arithmetic it saves, presumably because
the fixed-width `NR = 8` tile is what lets the loops compile to whole
vector operations, and a variable-length `live` tail reintroduces exactly the
per-row irregularity the padding was added to remove. **The pad columns are
not waste — they are what keeps every loop a whole number of 8-wide
operations.** 8 extra multiply-adds on aligned lanes beat 7 skipped ones
behind a mask.

**Consequence.** `a0b4d64`'s padded stride stands as the final form. The
`40/33` residual is not recoverable this way and should not be described as
"available headroom" in future notes. Reverted in full; no code from this
attempt was kept.

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
tests/symbolic_profiler.rs
tests/task_plan_parity.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
