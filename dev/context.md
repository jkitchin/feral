# FERAL Context (auto-generated)

Generated: 2026-09-06T00:55:23Z

## Latest Session
File: dev/sessions/2026-09-05-01.md
```
# Session 2026-09-05-01

## Benchmark note (read first)

**The corpus benchmark was not re-run, and the exit-partition numbers
below are cited, not measured this session.** `src/` on this branch is
byte-identical to `main` (`git diff main...HEAD -- src/` is empty), which
is itself the 0.17.0 baseline; the only Rust added is a diagnostic binary
under `crates/feral-diagnostics`. The full partition needs the 169,591-
matrix / 5.4 GB KKT corpus with MUMPS sidecars — a multi-hour run to
confirm that unchanged solver code produces unchanged numbers.

`cargo run --bin bench --release` was run and is recorded verbatim below;
`data/benchmark-config.toml` is absent in this clone, so it took its
synthetic fallback (8 matrices) and emitted no MUMPS ratios.

Cited from `dev/sessions/2026-08-19-05.md` (0.17.0 release):

```
--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.58     <= 2.0     PASS
medium (<500)            153560     1.58     <= 3.0     PASS
```

The next session that touches solver code must run the corpus and report
against these.

## Goal

Re-examine issue #200 after a new comment (2026-09-05, from the
pounce/pglib side) retracted the issue's own headline using a 351k-row
AC-OPF KKT — a matrix an order of magnitude larger than the one the issue
was filed from — and stated conclusions incompatible with the comment
this branch published on 2026-09-04.

## Accomplished

**Their finding reproduces on a different matrix class, and two of my own
2026-09-04 claims do not survive re-measurement.**

- **Confirmed their §2 in the source.** `src/dense/factor.rs:2168` sets
  `PANEL_MIN_NCOL = 8` and routes `ncol < 8` to
  `factor_frontal_in_place_with_scratch`, whose impl (lines 1724–1990)
  carries `LEXTRACT_NS`/`CONTRIBEXTRACT_NS`/`CONTRIBZEROFILL_NS` but no
  `PANELFACTOR_NS`/`SCHUR_NS`/`SCALARTAIL_NS`. Small-front arithmetic is
  uncounted and lands in `DENSEFACTOR`'s unnamed remainder. This is a
  second mechanism feeding the UNATTRIBUTED bucket, additive to the probe
  tax I reported, and it is the larger half.

```

## Git Status
```
90eebf1 perf(numeric): hoist the child->parent index map out of extend_add's inner loop
c6f106c docs: reject InfNorm scaling-vector caching, a third re-proposal of closed work
e9a438c fix(diag): time the driver's real kernel entry point, not factor_frontal
775b8a8 diag(#153): split the MA57 deficit into plumbing, prologue and kernel
2dcb842 docs: session checkpoint 2026-09-05-01 (#200 retracted and closed as a duplicate of #153)
```

## Test Status
```
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test numeric::solve::tests::cb_coarsening_threshold_is_arithmetically_inert ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 444 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 1.73s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-09-05-01.md)


FERAL benchmark harness
  ordering: default (symbolic_factorize heuristic)
  scaling: default (SupernodeParams::default)
Loading matrices from data/benchmark-config.toml ... not found

name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10          100           42     (10, 0, 0)
spd_50             50           24            3     (50, 0, 0)
spd_100           100           82            5    (100, 0, 0)
spd_200           200          404           18    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           21            1    (30, 10, 0)
kkt_50_15          65           49            2    (50, 15, 0)
kkt_100_30        130          207            7   (100, 30, 0)

8 matrices benchmarked

```

## Recent Decisions
| robot_1600 | 1044 | 2861 | 2.7× | 1.62× |
| steering_12800 | 917 | 2934 | 3.2× | 1.39× |
| ex4_2_160 | 5943 | 14329 | 2.4× | 2.05× |
| arki0009 | 6173 | 9500 | 1.5× | 1.73× |

**Decision.** feral runs at **1.5–3.2× fewer flops per second than MA57
at every front size**, on 0.44–1.20× the work (the fill and flop-volume
ratios from 2026-09-04 are symbolic, not timing-derived, and stand).
That is an arithmetic/memory-throughput deficit spread across the whole
front-size distribution, not a per-supernode constant. Issue #200 is
therefore the same defect as #153 ("dtoc1nd: 3.8× MA57, concentrated in
148 fronts of ncol≈62") and should be worked there.

**Corollaries, binding on future work.**
- A small-front fast path is capped at 34% of wall on the most
  favourable matrix in the corpus and 4% on the least. It cannot reach
  parity on any of them. Do not open one as a #200 fix.
- The corollaries of the 2026-09-04 decision that do not depend on the
  fitted intercept still hold: report fill and flops beside any
  wallclock ratio; ordering, fill and scaling are exonerated as the
  cause; `nemin` is not a lever (three rejections).
- `PHASE_TIMING_ENABLED` is unfit for attribution on small-front trees
  for two independent reasons, both now confirmed: its ~10
  `Instant::now()` pairs per supernode cost 0.20–0.41 µs/front
  (`diag_200_probe_tax`), *and* `src/dense/factor.rs:2168` routes
  `ncol < 8` fronts to `factor_frontal_in_place_with_scratch`, which
  carries no `PANELFACTOR`/`SCHUR`/`SCALARTAIL` counters — so their
  arithmetic is uncounted and lands in `DENSEFACTOR`'s unnamed
  remainder. The second mechanism was identified on the pounce side and
  verified here in the source. Use `ProfileReport` instead.

## Recent Tried-and-Rejected
**What was claimed, and retracted.** An A/B run appeared to show that
rewriting `extend_add`'s inner loops from indexed access to the
slice-and-zip form clippy's `needless_range_loop` asks for cost 10% on
optmass: 1.080x indexed vs 0.977x zipped. On that basis an
`#[allow(clippy::needless_range_loop)]` was added to the function with the
numbers cited as justification.

**Why it was wrong.** The two figures came from *different* interleaved
campaigns, built at different times. A rerun measured the indexed form at
0.974x on optmass — statistically the same as the zipped form's 0.977x.
optmass's "before" minimum ranged 18504-19603 us (6%) across campaigns
built minutes apart, which swamps the effect being attributed. The
`allow` and its justification were removed and the idiomatic zipped form
kept.

**Rule this reinforces.** A ratio between two numbers measured in
different campaigns is not a measurement. Only per-round paired ratios
from a single interleaved campaign were used for the threshold and
loop-form decisions that survived. (Same class of error as the warm-repeat
cache protocol retracted earlier the same day.)

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
