# FERAL Context (auto-generated)

Generated: 2026-08-20T16:15:53Z

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
08-20-01 to refute a similar attribution. **Next session should re-run the
bench cold before reading anything into these numbers.**

## Goal

Answer issue #189 item 4: is `BLAS3_NRHS_THRESHOLD = 32` costing pounce
anything? Under the standing bar for this work — rigorous, thoroughly
correct, and a real performance gain, or it does not ship.

## Accomplished

### The threshold is fine. There was a defect underneath it.

`BLAS3_NRHS_THRESHOLD` is unchanged at 32. What the investigation found
instead: the row-major multi-RHS work buffers used a leading dimension of
exactly `nrhs`, so any `nrhs` that is not a multiple of 8 makes every row of
every supernode panel straddle a cache line, compounding across the gather,
the kernels and the scatter.

Evidence, all within a single process so no cross-process drift can reach it.
`t(31)/t(32)` — `nrhs = 31` is 3% *less* work, so a healthy kernel gives
~0.97:
```

## Git Status
```
a0b4d64 perf(solve): align the multi-RHS row stride to a cache line
c34475e docs: session checkpoint 2026-08-20-01 (measurement corrections, #189 Step 1)
c09da70 bench: time the solve core hosts actually run (#189 item 1)
2ba437d probe: separate core, schedule, and depth in the solve measurement (#131, #189)
0ac4fb1 probe: measure the solve-phase levers claimed in #131 and #189
```

## Test Status
```
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test numeric::solve::tests::cb_coarsening_threshold_is_arithmetically_inert ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 445 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 3.44s

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
**Decision:** the panel path allocates and indexes at
`padded_ldw(nrhs) = nrhs.div_ceil(NR) * NR`, i.e. a multiple of 64 bytes.
The rank-1 path keeps the raw `nrhs` stride — it is row-major but not
tiled, and it carries the bit-identity band contract that
`tests/multi_rhs.rs:227,256` assert and pounce's `schur.rs:303` consumes.

The padding is paid for in flops: the kernels solve `padded_ldw(nrhs)`
columns, of which up to 7 are zero padding. That is 21% extra arithmetic at
`nrhs = 33`, 7% at 100, under 1% at 1000 — and it is still a net 1.36×
geomean win in the shipped regime (`nrhs >= 32`, not a multiple of 8),
because the alignment is worth more than the wasted lanes. After the fix
`t(33)/t(32) ≈ 1.21`, which is exactly `40/33`: the residual is entirely
the padding columns, with no misalignment left.

**Alternative not taken:** pad the *allocation* for alignment but iterate
only the `nrhs` live columns, masking the final tile. `gemm_tile` already
takes a `live` argument, so the kernel side is ready. It recovers roughly
the remaining 18% at `nrhs = 33`. It is not part of this decision because
it requires distinguishing stride from live width through five kernels, and
that belongs in its own commit with its own before/after.

**Bit-neutral**, verified three independent ways: an 800-shape `to_bits()`
test against a scalar left-fold reference (`gemm_tail_tests`); the probe's
`max |rank1 - blas3|` column unchanged at all 105 measured (matrix, `nrhs`)
points; and all 13 `tests/multi_rhs.rs` green including both
`assert_eq!(max_diff, 0.0)` band contracts. No tolerance was touched.

`BLAS3_NRHS_THRESHOLD` stays at 32 — the crossover constant was the
suspect, but the defect was underneath it, and the threshold is load-bearing
for the bit-identity contract.

## Recent Tried-and-Rejected
**Why.** A full sweep is ~25 minutes, so two "adjacent" runs are 25 minutes
apart. That is blocked measurement with an interleaved label on it — the same
error behind the two claim retractions earlier the same day, one level up the
stack. Cross-process A/B of a few-millisecond kernel does not work on this
machine at this cadence.

The remaining runs were killed (`pkill -f probe_blas3_crossover`, 0 left) and
no number from the paired dataset was reported as a result.

**Replaced by** an in-process design: the `kernel-probe` feature (off by
default, compiled out entirely) exposes
`set_blas3_nrhs_threshold`, so one process can time rank-1 and BLAS-3 at the
*same* `nrhs`, alternating within each repetition — microseconds apart instead
of 25 minutes. See `dev/research/blas3-threshold-refit.md`.

**Kept from this attempt:** the single-build, single-process runs are valid,
because looped-vs-batched was measured within one process. Two findings survive
and are recorded in the research note: the 31→32 dispatch discontinuity (3% more
work, batched time *drops* 1.36–1.77×), and that at `nrhs ∈ {2, 4}` batching is
**21–27% slower** than looping single-RHS solves.

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
