# FERAL Context (auto-generated)

Generated: 2026-08-21T03:54:19Z

## Latest Session
File: dev/sessions/2026-08-20-03.md
```
# Session 2026-08-20-03

## Unfavorable result, reported first (per CLAUDE.md)

**Issue #190's stated premise does not reproduce on this corpus, and the
feature built for it is not a performance gain.**

#190 argues the hardwired `ε·√n` convergence target is unreachable on large
ill-conditioned systems, so every refined solve runs the full 10-step budget
and pays for iterations that buy nothing. Measured (`probe_refine_stop_criterion`,
best-of-5 wall, seven matrices, well-scaled RHS):

| matrix | n | default steps | stop | ω |
|---|---:|---:|---|---:|
| r05_kkt | 14,842 | 0 | Converged | 4.546e-13 |
| qap15_kkt | 50,880 | 2 | Converged | 2.690e-16 |
| dirichlet120_kkt | 54,363 | 0 | Converged | 1.679e-14 |
| cont-201 | 80,595 | 0 | Converged | 8.720e-14 |
| cont5_late_kkt | 180,900 | 1 | Converged | 1.711e-16 |
| bratu3d | 27,792 | 0 | Converged | 2.467e-15 |
| bcsstk38 | 8,032 | 0 | Converged | 1.324e-15 |

Zero to two steps, never at the cap. There is no wasted-iteration saving to
claim, and the feature must not be described as one.

Stated in the other direction so the limit is on the record: the `n = 118,276`
system #190 cites is a pounce *runtime* KKT and is not in the local corpus
(largest local matrix is `c-big`, `n = 345,241`, not in this set). The premise
is **untested at its own scale**, not refuted at it.

The measured *cost* of the new criteria is ~2x on the refinement wall
(0.39x–0.61x "vs def"), roughly one extra solve, factor excluded. The only
speedups measured are the caller deliberately buying less accuracy: qap15 easy
at `BackwardError(1e-10)` is 1.45x by stopping at ω 8.755e-12 instead of
2.690e-16; qap15 hard at `RelativeResidual(1e-12)` is 1.49x and *worse*
(ω 8.038e-10 vs the default's 1.229e-10).

Against the standing bar for this thread — "rigorous, thoroughly correct, and
result in performance gains, or we will not include it in a release" — #190
clears the first two and **fails the third**. Ship-or-not is a human call; see
"Next Session Should".

## Goal

Act on issue #190: the `ε·√n` refinement target is a hardwired constant with
no principled value; let the caller say what "converged" means.

## Accomplished

### `StopCriterion`: the caller says what converged means (`f547bc5`)
```

## Git Status
```
65f488c docs(refine): correct #190's premise against the corpus measurement
f547bc5 feat(refine): let the caller say what "converged" means (issue #190)
849caea fix(dense): flag L-growth against the pivot threshold's own promise
7de1a93 docs: cold bench resolves the drift; masked-tile follow-up rejected as slower
2637964 docs: state which pounce call sites the padded stride actually reaches
```

## Test Status
```
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 456 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 1.66s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-20-03.md)


=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.44       0.30       1.56       3.25       9.20
solve/MUMPS        153560       0.08       0.08       0.15       0.73       2.60
factor/SSIDS       154500       0.04       0.03       0.32       0.94       1.96
solve/SSIDS        154500       0.94       1.00       2.50       8.67      29.00
nnzL/MUMPS         153560       0.61       0.58       0.75       4.50      23.11
nnzL/SSIDS         154500       0.88       1.00       1.00       4.50       5.00

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.58     <= 2.0     PASS
medium (<500)            152145     2.00     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.56     <= 2.0     PASS
medium (<500)            153560     1.56     <= 3.0     PASS

Quiet machine, nothing else running. This is the best of the recent runs and
sits at the bottom of the observed band — better than 08-20-02's hot run on
every bucket and level with its cold re-run:

| bucket | 08-19-05 baseline | 08-20-02 hot | 08-20-02 cold | this run |
|---|---:|---:|---:|---:|
| sparse small-frontal p90 | 1.58 | 1.65 | 1.54 | **1.56** |
| sparse medium p90 | 1.58 | 1.65 | 1.54 | **1.56** |
| dense small-frontal p90 | 1.58 | 1.66 | 1.59 | **1.58** |
| dense medium p90 | 2.00 | 2.04 | 1.97 | **2.00** |

All four PASS. `solve/MUMPS` geomean 0.08 and p50 0.08 unchanged. The
machine-state signature the previous session identified is visible again in
the right direction: `factor/MUMPS` max is **9.20** here against 71.94 on the
hot run, and `factor/SSIDS` p99 is 0.94 against 1.11.

Neither of this session's commits should touch the bench path — the bench
times factorization and a single-RHS solve, `849caea` changes only which
matrices set an advisory `needs_refinement` flag (read by nobody in the bench),
and `f547bc5` adds a criterion whose default is bit-for-bit the old behavior.
The numbers are consistent with that.

```

## Recent Decisions
`n = 118,276` system #190 cites is a pounce runtime KKT and is not in the
local corpus (largest local is `c-big`, `n = 345,241`, not in this set), so
the premise is untested at its own scale rather than refuted at it.

**What the measurement did establish is a correctness gap.** With a RHS
whose entries span `1e-6..1e6`, the default declares `Converged` at normwise
`rel` of `1e-14..1e-17` while the componentwise backward error is up to
eleven orders worse: r05_kkt `9.5e-5`, bratu3d `9.0e-6`, cont-201 `3.9e-7`,
dirichlet120_kkt `4.3e-10`, bcsstk38 `1.1e-10`, qap15_kkt `1.2e-10`. One
`BackwardError` step lands all of them at `~3e-16`. The default cannot see
this by construction: `||r||_2/||b||_2` is dominated by the rows where
`|b_i| ~ 1e6`.

**Cost.** ~2x on the refinement wall time (0.39x–0.61x "vs def"), factor
excluded — roughly one extra solve. The only measured speedups are the
caller deliberately buying less accuracy: qap15 easy at
`BackwardError(1e-10)` is 1.45x by stopping at omega `8.8e-12` instead of
`2.7e-16`; qap15 hard at `RelativeResidual(1e-12)` is 1.49x and *worse*
(omega `8.0e-10` vs the default's `1.2e-10`).

**Decision.** The feature stays, documented as an accuracy/observability
knob with the measured cost stated, not as a speedup. `CHANGELOG.md`,
`README.md` and the `solve_sparse_refined` doc comment were corrected —
all three had asserted the unreachable-target premise as fact.

**Documented caveat, new.** An unreachable *componentwise* target has the
identical failure mode the old constant had. qap15_kkt with a badly-scaled
RHS and `BackwardError(1e-14)` runs 8 steps, exits `Stagnated` at omega
`4.2e-11`, and costs 0.32x for nothing; `BackwardError(1e-10)` reaches
`4.4e-11` in 3 steps. The knob does not remove the need to read `stop`.

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
tests/issue190_refine_target.rs
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

(truncated from      354 lines to 350 line budget)
