# FERAL Context (auto-generated)

Generated: 2026-10-10T01:18:02Z

## Latest Session
File: dev/sessions/2026-10-09-01.md
```
# Session 2026-10-09-01

## Benchmark note (read first)

`cargo run --bin bench --release` ran on the full local KKT corpus
(156 929 matrices, 2339 parse-skipped) for both this branch and
`origin/main` (32c936b, detached worktree), same machine, back to back.
Correctness lines are identical: dense inertia 154429/154482, residual
154149/154482; sparse inertia vs MUMPS 154531/154590, residual
154258/154590; 48 both-path inertia mismatches; 78 sparse-only, 33
dense-only failures. Factor-ratio p90 vs MUMPS:

| bucket | main | this branch |
|---|---|---|
| dense small (<200) | 1.61 | 1.61 |
| dense medium (<500) | 2.00 | 2.00 |
| sparse small (<200) | 1.58 | 1.59 |
| sparse medium (<500) | 1.58 | 1.59 |

The sparse p90 is 0.01 worse on this branch. Two repeat runs of this branch
moved the dense p90 by 0.03–0.04 (1.58/1.96 vs 1.61/2.00), so 0.01 is
within run-to-run noise; it is reported, not explained away.

## Goal

Review PR #214 (issue #213: cascade-break floor) and fix what the review
found: the `n_cascade_break_nodes` overclaim, the missing case-(b) counter,
the semver label, the unverified records, and the undocumented
`zero_tol`-only gates.

## Accomplished

- **Review evidence.** Full `cargo test --release` with the corpus passes.
  kkt-mittelmann (613/618 readable) main vs PR: inertia identical, 0 oracle
  mismatches, no unrefined residual worse by more than 3×. The floor fired on
  marine_1600_0011/0013/0016 and qcqp1500-1nc_0000 (n_tiny 0 → 2263).
- **Reporter's matrix, rebuilt (integral, default settings).** max|L|
  2.98e16 → 6.63e12; min|d| 5.9e-16 → 1.03e-9; entries |L|>1e8 3982 →
  2275; unrefined resid 4.69 → 0.234; refined 3.5e-13 both; inertia
  (75796, 75596, 0) both, which is the expected count, so the issue's +2 did not
  reproduce. n_tiny 24, ncb 2, new tfa counter 16. Derivative form: no
  CB, unchanged (resid 1.9e-5, max|L| 9.17e7, tfa 0).
- **nemin sweep (unrefined resid main → PR):** 4: 179 → 16.1; 8: 3.38 →
  3.77; 16: 4.69 → 0.234; 32: 3.6e-5 both; 64: 8.6e-4 both.
- **`n_threshold_fail_accepts`** added at the 1×1 reject sites (scalar and
  blocked), on `FrontalFactors`, `SparseFactors`, `FactorStats`, Python.
- **Docs:** `n_cascade_break_nodes`/`cascade_break` now say the bound *may*
  fail (marine_1600_0009: ncb 1, n_tiny 0, resid 4e-16). `PerturbToEps`
  explains why the `count_1x1_inertia` and trailing-pivot paths keep the
  `zero_tol` gate (col_max = 0).
```

## Git Status
```
a90e63d docs(changelog): mark #213 struct fields breaking, record verified numbers
33b8847 feat(numeric): count threshold-failing pivots kept unchanged; fix CB docs (#213)
1959ecc docs(dev): session 2026-10-08-01 checkpoint, journal, decision, changelog for #213
949aac7 feat(stats): report cascade-break supernodes in FactorStats (#213)
6e683df fix(cascade-break): floor threshold-failing pivots below abs_floor, not only below zero_tol (#213)
```

## Test Status
```
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 456 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 1.76s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-10-09-01.md)

KKT summary: 156929 matrices (154482 dense-eligible n <= 1000, 108 skipped n > 1000, 2339 parse-skipped)
  Inertia match: 154429/154482 (100.0%)
  Residual pass: 154149/154482 (99.8%)
  Worst residual: 2.46e-1 (POLAK6_0021)
Sparse solver: 154590/156929 total
  Inertia match vs MUMPS: 154531/154590 (100.0%)
  Residual pass: 154258/154590 (99.8%)
  Worst residual: 2.94e-4 (ERRINBAR_0824)
--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
small-frontal (<200)     147982     1.61     <= 2.0     PASS
medium (<500)            152145     2.00     <= 3.0     PASS
--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
small-frontal (<200)     153455     1.59     <= 2.0     PASS
medium (<500)            153560     1.59     <= 3.0     PASS

```

## Recent Decisions
**Decision.** Count 1×1 pivots that fail `|d| <= max(u·col_max,
null_pivot_tol)` but are kept live and unchanged because the front cannot
delay (case a' under `ForceAccept`, case b), in
`FrontalFactors::n_threshold_fail_accepts`, summed to
`FactorStats::n_threshold_fail_accepts`. Zeroed or floored pivots stay in
`n_tiny`; the two counters do not overlap. `n_cascade_break_nodes` is
documented as "the `|L| <= 1/u` bound *may* not cover the factor", not
"does not". The zero-off-diagonal path (`count_1x1_inertia`) and the
trailing pivot of dense `factor` keep the `|d| <= zero_tol` gate, and the
reason is now in the `PerturbToEps` doc: `col_max = 0`, so these pivots
cannot fail the threshold and no L column is divided by them. The new
public fields on exhaustive structs make the next release 0.19.0.

**Why.** The 2026-10-08 entry left case-(b) pivots in `[abs_floor,
u·col_max]` accepted but invisible. The CB flag comes from the delay
budget, so it fires on clean fronts too: marine_1600_0009 has 1 CB node,
`n_tiny` 0, and a backward error of 4e-16. A caller needs the per-pivot
count to tell the two apart. On the rebuilt issue #213 matrix the counter
reads 16 after the fix, so these pivots are present there.

**Verification of the 2026-10-08 change** (that entry said it was not run
on the reporter's matrix): see `dev/sessions/2026-10-09-01.md`. In short,
max|L| went from 2.98e16 to 6.63e12 and the unrefined backward error from
4.69 to 0.234. Inertia was correct before and after. kkt-mittelmann
inertia is unchanged.

**Evidence.** `cascade_break_pivots_show_in_counters`: legacy CB gives
(tfa, n_tiny) = (8, 0) and bounded CB gives (0, 8).
`threshold_fail_accepts_counts_only_unchanged_accepts` has 5 kernel cases.
`cargo test --release` with the corpus: 956 passed, 0 failed.

## Recent Tried-and-Rejected

**Why they fail, as far as it was chased.** `dtoc_wide`'s `max_front` is 31.
At that width the numeric phase is all per-front overhead and no arithmetic,
so the comparison measures supernode count rather than flops, and MetisND's
fewer-but-wider fronts win for reasons that have nothing to do with dtoc2.
Widening the chain from `(nx,nu) = (4,2)` to `(8,4)` moved `max_front` only
24 -> 31; the real dtoc2's avg_deg of 17.5 is not reachable by this
construction at the right `n`.

**This was already on the record.**
`external_benchmarks/chain_proxy/README.md` documents geometry-matched
proxies producing the wrong answer and says to prefer the real corpus. That
warning was read during this session and the proxies were built anyway.

**What survives.** Nothing about routing. The non-proxy measurements in the
same session stand on their own and are recorded in
`dev/research/issue-203-auto-routing-2026-09-17.md`: on the issue #203
patterns MetisND is 1.62x faster sequentially and 3.5x in parallel, and
`nnz_L`, `flop_proxy` and `max_front` were each checked as routing guards
and each mispredicts.

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
tests/issue203_ordering_race.rs
tests/issue204_work_estimate.rs
tests/issue205_ordering_info.rs
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
