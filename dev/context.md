# FERAL Context (auto-generated)

Generated: 2026-08-10T00:14:28Z

## Latest Session
File: dev/sessions/2026-08-09-09.md
```
# Session 2026-08-09-09

## Goal

Size issue #125 step 2 before building it. That measurement said not to
build it, and pointed at the MC64 scaling cache instead — so the second
half of the session went there: find why the value-bound gate never
hits on warm iterates, and fix it.

## Accomplished

### Issue #125 step 2 — measured, recommended against

Static frontal row layout for fronts that receive delayed columns. Step 1
landed in PR #139; step 2 would extend it past `n_delayed_in == 0`.

- **Reach:** 7 of 41 corpus matrices have any front with `n_delayed_in > 0`.
  Material dynamic rows on 3: steering_12800 61.53%, robot_1600 28.86%,
  qcqp1500-1nc 25.04%. clnlbeam, dtoc1nd, dtoc2, marine_1600 and
  rocket_12800 are all at **0.00%**.
- **Cost ceiling** (`BUILDROW_NS / factor`): **0.21%–3.13%**. It is the
  smallest column in the phase profile.
- The fast-path `.to_vec()` cannot become a borrow — `row_indices` is
  moved into `NodeFactors` at `factorize.rs:2739`, so the copy is a
  genuine ownership requirement.

~1–2% on 3 of 41 matrices, paid for by touching the delayed-pivot path.
I explicitly retract my earlier characterization of #125 as "obvious
work" — that was true of step 1, not step 2.

### Factorization phase profile

Driver wall, best of 3 warm sequential reps. **Methodology correction
made mid-session:** the first version measured against the
`Solver::factor` wall, which folds in work the phase counters do not
cover, and showed 18.5%–50.9% "outside" assembly+dense. Rewritten to
drive `factorize_multifrontal_supernodal_with_workspace` directly; the
books then close (prologue + epilogue + loop ≈ 100%). Same
timed-region-mismatch error class that caused two retractions earlier
in the day.

| matrix | drv_us | prol% | scaling% | schur% | cbextr% | dense_o% | buildrow% |
|---|---|---|---|---|---|---|---|
| clnlbeam | 26297 | 18.1% | 15.2% | 11.2% | 6.6% | 22.1% | 2.2% |
| dtoc1nd | 12540 | 25.0% | 19.5% | 8.2% | 12.0% | 9.6% | 0.8% |
| dtoc2 | 76483 | 12.8% | 8.0% | 5.7% | 14.4% | 13.8% | 2.0% |
| marine_1600 | 31271 | 22.4% | 18.9% | 15.7% | 5.7% | 19.3% | 1.2% |
| rocket_12800 | 20461 | 34.7% | 29.3% | 11.3% | 1.8% | 24.5% | 0.9% |
| steering_12800 | 37880 | 17.5% | 14.6% | 15.3% | 5.7% | 19.8% | 3.7% |
| robot_1600 | 8662 | 19.0% | 15.8% | 15.0% | 6.4% | 22.1% | 3.0% |
```

## Git Status
```
bd1bc26 fix(scaling): make MC64 value-bound condition 3 a drift measure
8f8aa8c feat(diag): trajectory-level scaling profile and a scaling-reuse safety check
16b423c feat(diag): size issue #125 step 2 before building it -- and find a bigger target
7d9812d docs(research): profile MC64 before optimizing it — dense-column diagnosis fails
197c7be feat(bench): pre-scale offline so both solvers factorize identical numbers
```

## Test Status
```
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 416 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.50s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-09-09.md)


`cargo run --bin bench --release`, post-fix. Both exit partitions PASS.

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===
ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.43       0.30       1.54       3.30       8.70
solve/MUMPS        153560       0.07       0.08       0.14       0.68       3.04
factor/SSIDS       154500       0.04       0.03       0.32       0.96       2.03
solve/SSIDS        154500       0.93       1.00       2.44       8.35      36.50
nnzL/MUMPS         153560       0.61       0.58       0.75       4.50      23.11
nnzL/SSIDS         154500       0.88       1.00       1.00       4.50       5.00

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.58     <= 2.0     PASS
medium (<500)            152145     2.00     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.54     <= 2.0     PASS
medium (<500)            153560     1.54     <= 3.0     PASS

Against the earlier run in this session (pre-fix):

| partition | before | after |
|---|---|---|
| dense small-frontal p90 | 1.62 | 1.58 |
| dense medium p90 | 2.04 | 2.00 |
| sparse small-frontal p90 | 1.57 | 1.54 |
| sparse medium p90 | 1.57 | 1.54 |

**Do not read this as a win from the fix.** The bench factors each
matrix once with a fresh solver, so the value-bound gate is never
consulted and the change should be a no-op here — which is what "no
regression" means in this table. The p90 movement is run-to-run noise;
the worst-ratio top 10 turned over completely between the two runs
(SWOPF/QPNBLEND before, KIRBY2 after) on matrices of n = 157–458 where
absolute times are sub-millisecond.

The bench corpus is also a different population from the seven
kkt-mittelmann families the fix was measured on. It is a regression
control, not evidence for the change.


```

## Recent Decisions
**Decision:** `Mc64CacheValidity` gains `min_diag_0`, and condition 3 becomes a
disjunction of the existing absolute floor and a new drift bound
`min_diag >= DIAG_SHRINK * min_diag_0`, with `DIAG_SHRINK = 1.0 / GROWTH_FACTOR
= 0.5`.

**Why a disjunction rather than a replacement.** It is a strict widening of the
accept set, so no matrix that the gate accepts today can start being rejected.
And it is zero-drift-safe by construction: re-checking the baseline matrix
gives `min_diag == min_diag_0` exactly, so the drift clause holds for any
`DIAG_SHRINK <= 1`.

**Why 0.5.** Symmetric with `GROWTH_FACTOR`: the minimum diagonal may shrink by
the same factor the worst dominance ratio may grow. The constant is **not**
load-bearing — every value swept from 0.5 to 1e-6 produces the same 25 accepts
out of 53 corpus gate evaluations. 0.5 is the tightest defensible choice, not a
tuned one.

**Evidence.** 53 gate evaluations across the 7 corpus families that route to
MC64. Condition 3 was the sole blocker on 3: two `robot_1600` checks at drift
0.988 and 1.000 (false positives) and one `arki0003` check at drift 2.1e-08 (a
genuine eight-order collapse, still rejected after the change). Pre/post-fix
binaries give a complete hit-pattern diff of two flips, both `robot_1600`, with
inertia byte-identical on every iterate of every family.

**Scope.** This does not touch conditions 1 or 2, and does not revisit the
2026-05-21 rejection of Track B2 (`tried-and-rejected.md:2087`), which turned on
condition 1 being confounded by the IPM barrier trajectory. That finding still
holds: `pinene_3200` rejects 8/8 on condition 1 after this change.

Research note: `dev/research/mc64-value-bound-diag-drift-2026-08-09.md`.

## Recent Tried-and-Rejected
Swept `RAYON_NUM_THREADS` = 1, 2, 4, 8, 10 against the default on six
real chain KKTs, 15 paired runs each, on an M4 Pro (10P + 4E — *more*
efficiency cores than the 4P+4E M2 the hypothesis came from, so the
predicted effect should be larger). Every ratio vs the default landed
within 6% of 1.0; the only significant one was `marine_1600` at t1
(0.961, 0/15, p = 0.0001), in the wrong direction to support the
hypothesis. `steering_12800` at one thread: 1.003 (9/15, p = 0.6072).
`dtoc1nd`, the matrix that actually regressed, at t4: 1.005 (7/15,
p = 1.0000).

The knob was verified live, not assumed: feral reads the global rayon
pool (`rayon::current_num_threads()`, `src/numeric/factorize.rs:3262`),
and the same variable moved the proxy matrices by up to 65%.

Consequence worth carrying forward: single-threaded main matches
all-threads main on every one of these matrices, so **#150's 1.20x to
2.05x gains on the large chains are not parallelism gains**. Do not
build on the assumption that they are.

Full data: `dev/research/chain-kkt-corpus-2026-08-09.md`, Result 3.

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
src/lu/mod.rs
src/lu/scaling.rs
src/lu/sparse_factor.rs
src/lu/sparse_matrix.rs
src/lu/sparse_solve.rs
src/lu/sparse_symbolic.rs
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
tests/cb_solve_parity.rs
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
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
tests/lu_dense_update_bg.rs
tests/lu_dense.rs
tests/lu_ft_widebump.rs
tests/lu_scaling.rs
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
tests/profiler_smoke.rs
tests/property_tests.rs
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
