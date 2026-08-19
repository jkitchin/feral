# FERAL Context (auto-generated)

Generated: 2026-08-19T14:05:36Z

## Latest Session
File: dev/sessions/2026-08-19-01.md
```
# Session 2026-08-19-01

## BENCHMARK NUMBERS ARE NOT COMPARABLE TO LAST SESSION

Reported first, per the hard rule in CLAUDE.md.

`cargo run --bin bench --release` in this container found only the 8
synthetic matrices; the external corpus and the MUMPS/SPRAL oracle
timings are not present. Both Phase 2.8.1 exit partitions therefore
report `N/A` rather than a p90:

    --- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
    bucket                    count      p90     target  verdict
    small-frontal (<200)          0        -     <= 2.0      N/A
    medium (<500)                 0        -     <= 3.0      N/A

**No comparison against 2026-08-15-02's 1.61 / 2.00 / 1.67 / 1.67 is
possible from this run.** I am not claiming the numbers held; I am
saying they were not measured. A session with the corpus mounted should
re-run the partition before reading anything into it.

What the run does confirm: correctness is intact on what it could see —
inertia 2/2 vs MUMPS, residual 2/2, worst residual 1.26e-16.

This is also the wrong benchmark for this change, which touches the
*solve* path and not the factor path. The relevant measurements are in
"Benchmark Results" below.

## Goal

Fix issue #177 — "parallel solve is not bit-identical to serial on
henon120, breaking #131's stated contract".

## Accomplished

### The report is real, but not the bug it looks like

The reporter compared two runs that differed only in `FERAL_CB_THRESH`,
held the factorization sequential to rule out #16, and found the two
parallel runs bit-identical to each other but not to the "serial" one.
They concluded there was a fixed ordering difference in the parallel
path — "findable deterministically".

There is no such ordering difference. feral has **two numerically
distinct solve cores**:

- `solve_sparse_core_into` — folds each front's separator update into a
  global vector in flat postorder;
- the contribution-block core (#131 Gap A) — assembles each front's RHS
  from its children's contribution blocks, summed in ascending child
```

## Git Status
```
b75da82 test(solve): pin the refined solve's arithmetic against the host (#177)
3cafe57 fix(solve): choose the solve core from the factor, not the host (#177)
6fb9d26 Merge pull request #174 from jkitchin/feat/scaling-router-invariance
d00666a docs: session checkpoint 2026-08-15-02 (KIRBY2 localized, #153 closed)
3029905 docs(compress): localize the KIRBY2 factor-ratio outlier to LdltCompress
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
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 428 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.90s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-19-01.md)


Refined solve, best of 7, 4 workers, microseconds. This is the
measurement the change is about; the `bench` binary's factor-ratio
partitions are unrelated to it and were unavailable anyway (see the
top of this file).

    matrix        n      shared-vector  auto-serial     auto-par(4)
    chain_400     400         22.3      22.9 (1.03x)   22.9 (1.03x)
    chain_2000    2000       112.9     114.8 (1.02x)  114.6 (1.01x)
    chain_20000   20000     1276.3    1276.8 (1.00x) 1272.0 (1.00x)
    poisson_40    1600       642.2     657.1 (1.02x)  691.8 (1.08x)
    poisson_96    9216      4549.3    4612.6 (1.01x) 4631.5 (1.02x)
    poisson_160   25600    27761.6   30632.9 (1.10x) 20763.6 (0.75x)

Factors the predicate rejects are at parity (1.00-1.03x). On the one
factor it routes to the CB core, a host with no workers pays ~1.10x for
the determinism and a 4-worker host gains 25%.

The `bench` binary run for this session:

8 matrices benchmarked
2 KKT matrices total
KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)

--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

--- Dense perf vs oracles: no matrices have oracle timings ---
--- Sparse perf vs oracles: no matrices have oracle timings ---

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A

```

## Recent Decisions

Rejected alternatives, and why:

- *Make the CB core bit-identical to the shared-vector core.* Would
  require the CB forward to fold contributions in postorder-of-source-
  front, but it folds a grandchild's block into its child's block before
  that block reaches the parent. Matching the flat postorder means
  abandoning the subtree grouping, i.e. the parallelism itself.

- *Always use the CB core when parallelism is requested.* Correct and
  simple, but measured 1.08-1.86x slower than the shared-vector core on
  every factor the gate rejects (path-like chains, small grids), where
  the CB core wins nothing — its only measured win is 0.72x on
  poisson_160 at 4 workers. It also fails to close the issue, since
  `use_parallel` is itself defaulted from the host's core count.

- *Retire the CB core, or make it opt-in only.* Restores determinism at
  zero cost on rejected trees, but forfeits issue #131 Gap A's actual
  win (25% on the bushy factors where tree-parallel solve pays).

Accepted cost: on a factor the predicate routes to the CB core, a host
that cannot spawn workers now pays ~1.10x on the refined solve
(poisson_160: 27.8 ms shared-vector, 30.6 ms CB-serial), where before it
would have silently taken the shared-vector core and a different answer.
Factors the predicate rejects are unchanged, at 1.00-1.03x.

Evidence: issue #177; `dev/journal/2026-08-19-01.org`;
`tests/refined_solve_core_stability.rs` (fails at 6fb9d26 with 24295 of
25600 entries differing between the pooled and pool-less arms);
`tests/cb_core_choice_ignores_env.rs`.

## Recent Tried-and-Rejected

**Rejected on measurement.** The predicate runs on every refined solve,
including the ones it rejects, and `CbTaskPlan::build` allocates three
`Vec<Vec<usize>>` of length `n_nodes` (`build_children`, `owned`,
`tr_children`). Cost of the verdict alone, versus the shared-vector
baseline it was supposed to preserve:

    chain_400     1.29x
    chain_2000    1.27x
    chain_20000   1.24x

That is the same 1.24-1.29x the design existed to avoid — the predicate
cost as much as the core it was declining. Replaced by a flat
`O(n_nodes)` computation (four `Vec`s of scalars, subtree costs folded
into parents using the postorder guarantee, no child lists), which
brings the rejected trees back to 1.00-1.03x.

The cost of that replacement is a second implementation of one gate.
`cb_core_profitable_matches_the_plan_gate` pins the two together across
six fixtures landing on both sides of the gate.

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
