# FERAL Context (auto-generated)

Generated: 2026-06-03T11:28:06Z

## Latest Session
File: dev/sessions/2026-06-03-01.md
```
# Session 2026-06-03-01

## Goal

Fix issue #64: `Auto`/default ordering picks MetisND on arrow/bordered
KKTs (a thin body plus a few very-high-degree border columns), blowing
the LDLᵀ factor up ~7-9× vs AMF/AMD. Found via POUNCE on the LP `r05`
(n=14842): ~16 s auto (→ MetisND) vs 0.84 s forcing AMF.

User decisions up front: (1) detect with a cheap O(n) degree predicate
(not AutoRace); (2) regenerate the r05 fixture on demand into the
gitignored `tests/data/large/`, regression test skips when absent (no
blob in git).

## Accomplished

Arrow/bordered detection implemented and shipped; r05 regression test
goes green; full suite + clippy + fmt clean.

- **`is_arrow_bordered(pattern)`** (`src/symbolic/mod.rs`): O(n)
  degree pass over the full symmetric pattern. A "heavy" column has
  degree > max(64, 8·avg_deg); fire iff `1 ≤ heavy_count < 0.05·n` (small
  set) AND `heavy_nnz ≥ 0.20·full_nnz` (large nnz share).
- **Routing**: `choose_adaptive` overrides the would-be-MetisND decision
  (`n > 10_000`) to AMF when the predicate fires. The `n ≤ 10_000 → AMF`
  and `n > 100_000 && avg_deg < 5 → AMD` (#50) paths are untouched.
- **Unification**: `symbolic_factorize` now resolves through
  `OrderingMethod::Auto` instead of calling `pick_default_method`
  directly, so the no-arg default and an explicit `Auto` caller resolve
  to the same concrete ordering on every matrix. This also fixed a
  latent inconsistency (only `choose_adaptive` had the #50 large-sparse
  branch). `issue_3_auto_on_kkt_routes_via_pick_default_method` still
  passes (PoissonControl K=58 → MetisND, uniform, no arrow).
- **Calibration on real data** (`src/bin/probe_issue64_arrow.rs`):

  | matrix | n | avg_deg | heavy_count | heavy_nnz% | predicate |
  |---|---|---|---|---|---|
  | r05_kkt | 14842 | 15.0 | 171 (1.15%) | 38.5% | **ARROW→Amf** |
  | bratu3d | 27792 | 6.25 | 0 | 0% | no |
  | cont-201 | 80595 | 5.44 | 0 | 0% | no |
  | bcsstk38 | 8032 | 44.3 | 2 (0.03%) | 0.3% | no (share guard) |

  r05 nnz_L: Amf=506210 Amd=607519 MetisND=4358715 (MetisND/Amf=8.61×).
  Absolute counts drift from the issue's numbers (ordering-impl / METIS
  seed) but the ranking and the `<1e6` test threshold are robust.
- **Tests**: 4 unit tests for the predicate (fires on synthetic arrow;
  rejects uniform-sparse, many-hubs (count guard), low-share border
  (share guard)); `choose_adaptive_routes_arrow_to_amf`; regression
  `tests/issue64_arrow_ordering.rs` (skip-if-absent, asserts
  `nnz_L < 1.0e6` and `resolved_method != MetisND`). Verified red→green.
```

## Git Status
```
72c05ee Merge pull request #66 from jkitchin/claude/issue-64-arrow-ordering
b93446a docs(session): checkpoint 2026-06-03-01 — issue #64 arrow ordering catch
4882313 fix(ordering): arrow/bordered-KKT catch — route MetisND→AMF on dense-border patterns (#64)
8877a48 docs(lever-1.2): row-band blocking measured — bit-exact but a 0.74-0.95x regression (#62)
55f6a70 perf(dense): intra-front parallel Schur (perf-review Lever 1.1) + lever-sweep docs (#61)
```

## Test Status
```
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_resolves_to_amd_when_bisection_degenerates ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok

test result: ok. 322 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 0.43s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-03-01.md)


name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10           40            9     (10, 0, 0)
spd_50             50           23            3     (50, 0, 0)
spd_100           100           84            5    (100, 0, 0)
spd_200           200          413           16    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           25            1    (30, 10, 0)
kkt_50_15          65           49            2    (50, 15, 0)
kkt_100_30        130          205            7   (100, 30, 0)

The 8-matrix synthetic bench is all n ≤ 200 (sub-threshold → AMF), so it
exercises none of the arrow path; numbers are within prior-session noise
(spd_200 413µs vs 362–389 in 2026-05-31-03) and inertia is exact. The
arrow catch only consults `is_arrow_bordered` on the would-be-MetisND
branch (n > 10_000), so these matrices are bit-identical to before. The
load-robust evidence for the fix is the r05 regression test and the
real-data calibration table above, not this bench.

```

## Recent Decisions
degree > max(64, 8·avg_deg); fire iff 1 ≤ heavy_count < 0.05·n (a small
set) AND heavy_nnz ≥ 0.20·full_nnz (a large nnz share). The share guard
is the discriminator — it fires on r05 (38.5%) and rejects bcsstk38
(0.3% share despite two degree-614 columns); the count guard rejects
"many hub" patterns. Uniformly-thin matrices (PoissonControl,
powerflow22, bratu3d, cont-201) have no heavy column and are untouched.

Routing target is AMF (the existing n≤10_000 default and the measured
winner on r05), keeping the dispatcher coherent: small-or-arrow → AMF,
large-uniform → MetisND, very-large-thin → AMD.

Placement: the catch lives in `choose_adaptive`, and `symbolic_factorize`
now resolves through `OrderingMethod::Auto` instead of calling
`pick_default_method` directly. This unifies the two entry points — the
no-arg default and an explicit `Auto` caller now resolve to the same
concrete ordering on every matrix. Previously they could disagree on
very-large-and-sparse patterns (only `choose_adaptive` had the #50
`n>100_000 && avg_deg<5 → Amd` branch), a latent inconsistency the
docstrings claimed did not exist.

This is the *opposite* routing direction from issue #50, which deleted
escape hatches that pushed low-avg-degree patterns *toward* MetisND. Here
the body is not uniformly thin (full avg_deg ≈ 15); the problem is a few
dense borders. A purely synthetic arrow did not faithfully reproduce the
fill ranking (issue #64 reporter note), so the regression fixture is the
real regenerated r05 KKT, gitignored and skip-if-absent.

Evidence: dev/research/issue-64-arrow-bordered-ordering.md,
dev/journal/2026-06-03-01.org, src/symbolic/mod.rs is_arrow_bordered +
choose_adaptive, tests/issue64_arrow_ordering.rs, dev/scripts/regen_r05_kkt.sh.

## Recent Tried-and-Rejected
**c-block outer / m-tile inner** would keep a small `B`-block
L1-resident and cut the dominant re-streaming by the factor `NR/MR = 2`.

Tried the swap. **Measured: no improvement.** n=1024 stayed at ratio
~1.0–1.2 (still a regression), and n=484/2025 were within noise of the
m-outer order. The loop order was not the bottleneck at these sizes.

Reverted to the simpler m-outer kernel (the comment claiming the swap
fixed n=1024 would have been false). The actual n=1024 cause was the
**stride-`n` gather/scatter** reading the column-major `y` — power-of-
two `n` aliased RHS columns into the same cache sets. Flipping the
internal `y` buffer to row-major (contiguous memcpy gather/scatter)
fixed it (ratio 1.2 → 0.33) and ~halved wide-solve time everywhere.
See `dev/research/issue-57-blas3-panel.md` Results and the
`dev/decisions.md` 2026-05-30 entry.

**Lesson.** Diagnose the bottleneck before micro-optimizing the kernel:
the transpose in the gather/scatter dominated, not the GEMM's operand
re-streaming. A loop-order change to the GEMM was wasted motion until
the layout (row-major `y`) was fixed.

## Source Files
```
src/bin/alloc_probe.rs
src/bin/bench_axpy_small.rs
src/bin/bench_dense_multirhs.rs
src/bin/bench_fma_phase3.rs
src/bin/bench_intrafront.rs
src/bin/bench_issue8.rs
src/bin/bench_multirhs.rs
src/bin/bench_one_matrix.rs
src/bin/bench_orderings.rs
src/bin/bench_solver_corpus.rs
src/bin/bench_solver_reuse.rs
src/bin/bench_sqd.rs
src/bin/bench.rs
src/bin/blas3_prototype.rs
src/bin/calibrate_par_min_flops.rs
src/bin/d3_probe.rs
src/bin/d4_probe.rs
src/bin/diag_acopp30_residual.rs
src/bin/diag_acopr.rs
src/bin/diag_acopr14.rs
src/bin/diag_amalgamation.rs
src/bin/diag_amd_substages.rs
src/bin/diag_amf_vs_amd.rs
src/bin/diag_cascade_default_evidence.rs
src/bin/diag_cascade_ratio_distribution.rs
src/bin/diag_chainwoo_profile.rs
src/bin/diag_chainwoo.rs
src/bin/diag_clnlbeam_maxfromm.rs
src/bin/diag_clnlbeam_slb.rs
src/bin/diag_compress_costbenefit.rs
src/bin/diag_compress_profile.rs
src/bin/diag_compression_bench.rs
src/bin/diag_cond_parity.rs
src/bin/diag_dense_tail.rs
src/bin/diag_etree_shape.rs
src/bin/diag_factor_nnz_accounting.rs
src/bin/diag_fbrain3ls_pivtol_sweep.rs
src/bin/diag_fill_parity.rs
src/bin/diag_fill_tail.rs
src/bin/diag_inertia_mismatch.rs
src/bin/diag_issue50_auto_validate.rs
src/bin/diag_issue50_inventory.rs
src/bin/diag_issue50_large_sparse_scan.rs
src/bin/diag_issue50_numeric_inventory.rs
src/bin/diag_issue50_symbolic.rs
src/bin/diag_leaf_profile.rs
src/bin/diag_max_ncol.rs
src/bin/diag_mc64_cycles.rs
src/bin/diag_mittelmann.rs
src/bin/diag_narx_kernel_gflops.rs
src/bin/diag_near_singular_sweep.rs
src/bin/diag_nemin_amalgamation_panel.rs
src/bin/diag_orbit2_quotient.rs
src/bin/diag_ordering_panel.rs
src/bin/diag_ordering_race.rs
src/bin/diag_par_firstdiff.rs
src/bin/diag_par_frontal_hash.rs
src/bin/diag_par_repeat.rs
src/bin/diag_parent_unique.rs
src/bin/diag_phase_b_nemin_sweep.rs
src/bin/diag_pinene_0009_profile.rs
src/bin/diag_pinene_amd.rs
src/bin/diag_pinene_pivot_cliff.rs
src/bin/diag_pinene_static_pivot.rs
src/bin/diag_poisson_kkt.rs
src/bin/diag_qcqp_knobs.rs
src/bin/diag_qcqp_profile.rs
src/bin/diag_robot1600_eigs.rs
src/bin/diag_schur_parity.rs
src/bin/diag_small_leaf_gate.rs
src/bin/diag_small_leaf.rs
src/bin/diag_small_sparse_inventory.rs
src/bin/diag_sparse_memory.rs
src/bin/diag_split_tail.rs
src/bin/diag_strategy_compare.rs
src/bin/diag_supernode_cost.rs
src/bin/diag_swopf_w22x2.rs
src/bin/diag_symbolic_stages.rs
src/bin/dump_diff.rs
src/bin/feral_replay.rs
src/bin/feral_time.rs
src/bin/hs85_diag.rs
src/bin/parallel_corpus_parity.rs
src/bin/phase0_cb_on_revalidation.rs
src/bin/polak6_diag.rs
src/bin/policy4_diag.rs
src/bin/probe_acopp30_64.rs
src/bin/probe_cache_sequence.rs
src/bin/probe_cascade_perturb.rs
src/bin/probe_clnlbeam_refine.rs
src/bin/probe_clnlbeam_shape.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_dtoc2_mc64.rs
src/bin/probe_explicit_zeros.rs
src/bin/probe_f01.rs
src/bin/probe_fbrain.rs
src/bin/probe_fma_kernel.rs
src/bin/probe_front_concentration.rs
src/bin/probe_hang_loop.rs
src/bin/probe_intrafront_schur.rs
src/bin/probe_ir_trajectory.rs
src/bin/probe_issue_19.rs
src/bin/probe_issue45_ordering.rs
src/bin/probe_issue45.rs
src/bin/probe_issue46_preprocess.rs
src/bin/probe_issue46_supernode.rs
src/bin/probe_issue46.rs
src/bin/probe_issue49.rs
src/bin/probe_issue54_alpha_shift.rs
src/bin/probe_issue54_cascade.rs
src/bin/probe_issue54_ma57_alpha.rs
src/bin/probe_issue54.rs
src/bin/probe_issue64_arrow.rs
src/bin/probe_kkt_replay.rs
src/bin/probe_marine_shape.rs
src/bin/probe_marine_time.rs
src/bin/probe_mc64_spread.rs
src/bin/probe_mc64_synth.rs
src/bin/probe_narx_factor.rs
src/bin/probe_narx_phases.rs
src/bin/probe_panel_attribution.rs
src/bin/probe_pinene_issue38_fix.rs
src/bin/probe_rkt_shape.rs
src/bin/probe_robot_profile.rs
src/bin/probe_rocket_profile.rs
src/bin/probe_rocket_residuals.rs
src/bin/probe_rocket_slow.rs
src/bin/probe_scaling_policy4.rs
src/bin/probe_static_pivot_inertia.rs
src/bin/probe_supernode_widths.rs
src/bin/probe_thomson_hessian.rs
src/bin/probe_value_determinism.rs
src/bin/probe_warm_cascade.rs
src/bin/probe_wide_supernode.rs
src/bin/produce_dense_schur.rs
src/bin/profile_hot.rs
src/bin/profile_sparse.rs
src/bin/profile_supernode_distribution.rs
src/bin/solve_microbench.rs
src/bin/vesuvio_diag.rs
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

(truncated from      406 lines to 350 line budget)
