# FERAL Context (auto-generated)

Generated: 2026-08-21T15:45:10Z

## Latest Session
File: dev/sessions/2026-08-21-01.md
```
# Session 2026-08-21-01

## Unfavorable result, reported first (per CLAUDE.md)

**The solve tail regressed 33–56%, this session's change caused it, and a
controlled A/B proves that rather than leaving it as a hypothesis.**

`src/bin/bench.rs:1985` times the sparse solve with `RefineOptions::default()`,
so the new stopping criterion sits directly in the measured path. I ran the
bench with the shipped default, then reverted `Default for RefineOptions` to
`EpsSqrtN`, rebuilt, and re-ran on the same quiet machine:

| ratio | control (`EpsSqrtN`) | shipped default | delta |
|---|---:|---:|---:|
| factor/MUMPS p90 | 1.54 | 1.56 | +1.3% (noise) |
| solve/MUMPS geomean | 0.08 | 0.08 | none |
| solve/MUMPS p50 | 0.08 | 0.08 | none |
| solve/MUMPS p90 | 0.15 | **0.20** | **+33%** |
| solve/MUMPS p99 | 0.71 | **1.08** | **+52%** |
| solve/SSIDS geomean | 0.94 | **1.06** | **+13%** |
| solve/SSIDS p90 | 2.50 | **3.60** | **+44%** |
| solve/SSIDS p99 | 8.33 | **13.00** | **+56%** |

**The shape is what the design predicts.** Geomean and p50 are identical to
three decimals — the well-scaled majority never needed the extra conjunct
and does not pay for it. The tail pays, because that is where the matrices
live whose componentwise error was above `√ε` and which now actually refine.
On the tracked parity corpus that is 13 of 63 (21%), the right order to move
a p90.

Factor is unchanged within noise, as it must be — the change is confined to
the refinement loop's stopping test. That also validates the control: machine
state would have moved factor too, and did not. **Unlike 2026-08-20-02,
nothing here is attributed to machine state.**

**Against the standing bar** ("rigorous, thoroughly correct, and result in
performance gains, or we will not include it in a release"): this is a
measured performance *cost*, not a gain. It buys MA57/MUMPS componentwise
parity on systems where feral returned ω up to 9.5e-5 and reported
`Converged`. Whether that trade ships is the human's call. The alternative is
to keep `EpsSqrtN` as the default and make the conjunction opt-in — which
costs pounce the fix it asked for.

## Goal

Human instruction: *"we need to fix the deficiency gap with ma57/mumps if
there is one because this is causing an issue on a class of problems in
pounce. surprisingly, it works well for the vast majority of problems."*

## Accomplished
```

## Git Status
```
f5bc004 docs(dense): drop unverifiable pounce provenance from the growth-flag docs
396bfa3 fix(solve): default refinement now certifies componentwise accuracy
001a7db diag: four probes that locate the MA57/MUMPS deficiency gap
963884c docs: session checkpoint 2026-08-20-03 (#190 measured; premise refuted)
65f488c docs(refine): correct #190's premise against the corpus measurement
```

## Test Status
```
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test numeric::solve::tests::cb_coarsening_threshold_is_arithmetically_inert ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
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

test result: ok. 456 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 1.68s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-21-01.md)


Inertia gate: **154531/154590 (100.0%) match vs MUMPS** — holds.

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.44       0.30       1.56       3.45      11.05
solve/MUMPS        153560       0.08       0.08       0.20       1.08       4.56
factor/SSIDS       154500       0.04       0.03       0.32       1.02       2.49
solve/SSIDS        154500       1.06       1.00       3.60      13.00      52.33
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

All four partitions PASS and sit at the bottom of the observed band against
the 08-19-05 baseline of 1.58/1.58/1.58/2.00. `factor/MUMPS` max fell
71.94 → 11.05, the machine-quietness signature. The solve regression is
covered at the top of this file.

Suite: **956 passed, 0 failed, 113 suites.** `cargo clippy -- -D warnings` and
`cargo clippy -p feral-diagnostics --all-targets -- -D warnings` both clean.

```

## Recent Decisions
tolerance or residual gate. `EpsSqrtN` remains available and bit-for-bit
unchanged for callers that want the historical behavior.

**Cost, measured — and it is not free.** On the seven large matrices:
well-scaled RHS identical to the old default (same steps, same iterates);
badly-scaled RHS one extra step (5–14 ms) on the three worst, which then sit
level with MUMPS at ~3e-16.

That understates the corpus-wide cost. A controlled A/B over the 154,588
benchmark matrices — same machine, only `Default for RefineOptions` changed —
shows the median untouched and the tail paying: solve/MUMPS p90 0.15 → 0.20
(+33%), p99 0.71 → 1.08 (+52%); solve/SSIDS geomean 0.94 → 1.06 (+13%), p90
2.50 → 3.60 (+44%), p99 8.33 → 13.00 (+56%). Factor is unchanged within noise,
which confirms the effect is the refinement loop rather than machine state.

So this decision trades tail latency for componentwise correctness. It is
recorded as such, not as a free improvement. Callers needing the old latency
profile can pass `StopCriterion::EpsSqrtN` explicitly and accept the gap.

**Rejected alternatives** (both in `dev/tried-and-rejected.md` with their
failing cases): `BackwardError(√ε)` alone, which fails `tests/parity.rs` on
ROSZMAN1_0241 because ω ≤ √ε can hold while the normwise residual is looser
than the old rule delivered; and `BackwardError(f64::EPSILON)`, LAPACK
`dgerfs`'s target, which stagnates or exhausts the step budget on five of
seven large matrices under the badly-scaled RHS.

**Regression coverage.** `tests/issue190_componentwise_default.rs`. The
defect reproduces on the tracked parity corpus — `EpsSqrtN` leaves ω above
√ε on 13 of 63 matrices, worst DEGENLPB_0046 at 359×√ε — and the new default
drives all 63 to ω ≤ √ε.

## Recent Tried-and-Rejected
`StopCriterion::EpsSqrtNAndBackwardError(√ε)`, which is strictly harder to
satisfy than the old default and therefore cannot regress any caller.
ROSZMAN1_0241 and the rest of `tests/parity.rs` pass unchanged under it.

## 2026-08-21 — `BackwardError(f64::EPSILON)` as the componentwise target

**Tried.** Using `f64::EPSILON` — LAPACK `dgerfs`'s componentwise target —
rather than `√ε` for `DEFAULT_BACKWARD_ERROR_TARGET`. Measured with
`OMEGA_EPS=1 HARD_RHS=1 probe_vs_mumps_residual`.

**Failed on cost.** Under the badly-scaled RHS it stagnates or exhausts the
step budget on **five of the seven** large matrices, taking up to 10
correction steps, while `√ε` converges on the same systems in at most one.
That is precisely the wasted-budget pathology issue #190 complained about,
reintroduced from the other direction.

**Disposition.** Rejected. `√ε` is what MUMPS itself targets
(`ref/mumps/src/dini_defaults.F:1094`), so matching it is both the cheaper
and the better-justified choice. Recorded in the doc comment on
`DEFAULT_BACKWARD_ERROR_TARGET` so it is not retried.

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
tests/issue190_componentwise_default.rs
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
tests/task_plan_parity.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
