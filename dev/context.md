# FERAL Context (auto-generated)

Generated: 2026-08-30T00:31:02Z

## Latest Session
File: dev/sessions/2026-08-19-05.md
```
# Session 2026-08-19-05

## Goal

Close out issue #153 item 2, review the six PRs merged since v0.16.0 for
mistakes before shipping, and cut release 0.17.0 (which unblocks
pounce#710's second blocker).

## Benchmark Results

No regression. Both exit partitions PASS, and the small-frontal p90
improved slightly against the run earlier in this session (1.60/1.61 ->
1.58); the change is within run-to-run noise on this container and is not
claimed as an improvement.

```
--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.58     <= 2.0     PASS
medium (<500)            152145     2.00     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.58     <= 2.0     PASS
medium (<500)            153560     1.58     <= 3.0     PASS

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===
ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.45       0.32       1.58       3.45      10.58
solve/MUMPS        153560       0.08       0.08       0.15       0.79       2.71
factor/SSIDS       154500       0.04       0.03       0.33       1.02       2.30
solve/SSIDS        154500       0.95       1.00       2.50       9.00      35.25
nnzL/MUMPS         153560       0.61       0.58       0.75       4.50      23.11
nnzL/SSIDS         154500       0.88       1.00       1.00       4.50       5.00
```

## Accomplished

### Issue #153 item 2 — both hypotheses falsified (PR #184, merged)

Rescoped #153 to item 2 and measured rather than assumed. Both standing
hypotheses for the dtoc1nd dense-front gap — a packed-SIMD work gate, and
`block_size` — were falsified by measurement. Research note written; two
diagnostic probes landed.

### Pre-release review of six PRs (PR #185, merged)

Reviewed #174, #179, #180, #181, #182, #183 with agents, then verified
every reported finding by hand against git before acting on it. One
behavioural regression, one guard-test gap, and a set of unsupported
```

## Git Status
```
9b9e882 Merge pull request #188 from jkitchin/ci/codecov-coverage
1292984 ci: measure coverage with cargo-llvm-cov and report it to Codecov
ad0d96d Merge pull request #187 from jkitchin/docs/session-2026-08-19-05
b224ed1 docs: session checkpoint 2026-08-19-05 (pre-release review, 0.17.0 tagged)
2fbf9b7 Merge pull request #186 from jkitchin/release/0.17.0
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

test result: ok. 442 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.84s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-19-05.md)


No regression. Both exit partitions PASS, and the small-frontal p90
improved slightly against the run earlier in this session (1.60/1.61 ->
1.58); the change is within run-to-run noise on this container and is not
claimed as an improvement.

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.58     <= 2.0     PASS
medium (<500)            152145     2.00     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.58     <= 2.0     PASS
medium (<500)            153560     1.58     <= 3.0     PASS

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===
ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.45       0.32       1.58       3.45      10.58
solve/MUMPS        153560       0.08       0.08       0.15       0.79       2.71
factor/SSIDS       154500       0.04       0.03       0.33       1.02       2.30
solve/SSIDS        154500       0.95       1.00       2.50       9.00      35.25
nnzL/MUMPS         153560       0.61       0.58       0.75       4.50      23.11
nnzL/SSIDS         154500       0.88       1.00       1.00       4.50       5.00

```

## Recent Decisions
guard. Under `cargo llvm-cov` the build is unoptimized *and* carries
per-region counters, and the factor misses the budget:

    test dirichlet120_parallel_factor_does_not_deadlock ... FAILED
    panicked at tests/issue102_intrafront_deadlock.rs:72:13:
    issue #102 regression: ... did not finish in 120 s
    test result: FAILED. 0 passed; 1 failed; finished in 123.25s
    real 3m4.912s   user 11m14.718s

The same test passes under `cargo test` and under `cargo test --release`
in the same tree. So this is instrumentation overhead, and had the
workflow shipped without the skip the coverage job would have been red
from its first run in a way that reads like issue #102 came back.

The alternative — raising the 120 s budget so it survives instrumentation
— is loosening a tolerance, which needs human approval, and it would
weaken the guard everywhere to accommodate one job. Skipping in the
coverage job only leaves the assertion at 120 s in ci.yml's `check` job,
which runs on every push and PR. The guard keeps its teeth; coverage
just does not measure that path. An audit found no other test that needs
the same treatment: `large_matrix_smoke` and `rook_rescue_kkt` time
themselves but assert nothing about the elapsed time, so
`issue102_intrafront_deadlock` is the tree's only wall-clock assertion.

**How to read the number.** Fixture-gated tests SKIP-and-pass when their
gitignored fixtures are absent, so on CI every path they guard reads as
uncovered while being covered locally. The coverage job prints the skip
list to the run summary — the same step ci.yml has — so a low number in
those modules can be checked against it before being treated as a real
gap.

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
tests/issue178_refine_cap.rs
tests/issue178_solve_into.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/issue91_preprocess_misfire.rs
tests/issue99_fma_front_gate.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/lu_adversarial_inputs.rs
tests/lu_default_ordering.rs
tests/lu_dense.rs
tests/lu_dense_bump.rs
tests/lu_dense_update_bg.rs
tests/lu_ft_widebump.rs
tests/lu_hyper_sparse.rs
tests/lu_markowitz.rs
tests/lu_real_bases.rs
tests/lu_scaling.rs
tests/lu_sparse.rs
tests/lu_sparse_rhs.rs
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
tests/pounce710_refine_cap_nrhs2.rs
tests/pounce_interface.rs
tests/profiler_smoke.rs
tests/property_tests.rs
tests/refined_solve_core_stability.rs
tests/rook_rescue.rs
tests/rook_rescue_kkt.rs
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
