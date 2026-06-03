# FERAL Context (auto-generated)

Generated: 2026-06-03T12:58:54Z

## Latest Session
File: dev/sessions/2026-06-03-03.md
```
# Session 2026-06-03-03

## Goal

Pick up **issue #65**: default `Auto` scaling reports a wrong inertia (spurious
zero pivots) on ill-conditioned symmetric-indefinite KKTs; `Mc64Symmetric` fixes
it exactly. Downstream the consuming IPM (pounce) reads the spurious zeros as
`Singular` and falsely declares infeasibility (sawpath/discs).

## Accomplished

Shipped an **inertia-guided MC64 scaling fallback** in `Solver::factor`, with a
154k-matrix corpus validation showing **zero inertia regressions**.

- **Root-cause / design.** Reproduced sawpath iter-0 (n=1575): Auto/InfNorm →
  `(789,670,116)` ✗ (min|piv|=0, 116 spurious zeros); MC64 → `(789,786,0)` ✓
  (min|piv|=0.03; matches MA27/numpy). Proved a **structural router fix cannot
  work**: `twirism1` iter-0 has the *same* router signature (`diag_only=0,
  max_col_nnz>32`) but the *opposite* need — InfNorm correct `(432,313,0)`, MC64
  *wrong* `(433,311,1)`. pounce passes `check_inertia=None`, so feral's
  expected-inertia path never fires in production. The deciding signal must be
  numerical and feral-visible: **force-accepted zero pivots**.
- **Fix.** When the user configured `Auto`, the resolved scaling was not MC64,
  and the factor reports `inertia.zero > 0`, re-run with `Mc64Symmetric` and
  adopt iff it strictly reduces the zero count. Pin `auto_picked_strategy=MC64`
  on adoption (sticky — refactors skip the retry). New
  `Solver::mc64_scaling_fallback_count()`. Correctness-safe: MC64 can't change
  rank, so genuinely-singular matrices keep their original factor.
- **Behavior:** sawpath Auto → `(789,786,0)` ✓ (fallback fires); twirism1 iter-0
  → `(432,313,0)` ✓ (no fire, MC64's wrong inertia never adopted); explicit
  InfNorm respected (gate is Auto-only).
- **Tests:** `tests/issue65_mc64_fallback.rs` (3 tests, skip-if-absent; external
  oracle = MA27/numpy from the issue). `dev/scripts/regen_issue65_kkts.sh`.
- **Corpus validation** (`probe_issue65_corpus`, full KKT consensus corpus):
  **153,725 definitive+strong checked, 99.96% match, 64 mismatches all
  `fallback=0` (pre-existing ACOPP30/BATCH sign disagreements), fallback fired
  AND mismatch = 0**. The fallback never fires on the existing corpus — it is a
  surgical fix that activates only on the sawpath-class pathology and introduces
  zero regressions.
- Full suite green; clippy + fmt clean (see end).

### Out of scope / follow-up

`twirism1`'s **late-iteration** failure is a wrong *negative* count *without*
zeros (feral returns `Success`); a zero-trigger cannot see it, and it needs the
expected inertia, which pounce passes as `None`. Covering it requires pounce
passing expected inertia to `factor()` (then feral retries MC64 on
`WrongInertia`) or a self-contained "suspicious neg count" heuristic. Recorded
as follow-up; the sawpath/discs class is fixed.

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
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_resolves_to_amd_when_bisection_degenerates ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok

test result: ok. 322 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 0.42s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-03-03.md)


name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10           36           10     (10, 0, 0)
spd_50             50           49            3     (50, 0, 0)
spd_100           100          107            5    (100, 0, 0)
spd_200           200          439           16    (200, 0, 0)
kkt_10_3           13            3            0     (10, 3, 0)
kkt_30_10          40           21            1    (30, 10, 0)
kkt_50_15          65           48            2    (50, 15, 0)
kkt_100_30        130          327           15   (100, 30, 0)
In line with prior sessions; inertia exact on all 8.
No numeric kernel changed; the fallback only adds a conditional second factor
on the rare `Auto` + `zero>0` path (never fires on the 154k-matrix corpus).
Inertia exact on the 8 synthetic bench matrices.

```

## Recent Decisions
`Mc64Symmetric` and adopt iff it strictly reduces the zero count. Pin
`auto_picked_strategy = Mc64Symmetric` on adoption so refactors on the same
pattern skip the retry. New counter `mc64_scaling_fallback_count()`.

Why numerical, not structural: sawpath (needs MC64) and twirism1 iter-0 (needs
InfNorm — MC64 gives it the WRONG inertia (433,311,1)) have the IDENTICAL router
signature (diag_only=0, max_col_nnz>32). A structural router cannot separate
them; the deciding factor is whether the factorization hits the
working-precision floor. pounce-feral passes `check_inertia=None`, so feral's
own expected-inertia path never fires in production — the trigger must be a
signal feral sees unaided, i.e. force-accepted zero pivots.

Correctness safety: MC64 is a diagonal/permutation rescaling and cannot change
rank. On a genuinely singular matrix the retry also force-accepts zeros, the
strict-improvement gate fails, and the original factor is kept (cost: one wasted
factorization). So the fallback only moves feral TOWARD the MUMPS/SPRAL
consensus on effectively-full-rank-but-ill-conditioned matrices, never away from
a true singular classification. Corpus-validated (KKT consensus oracle): zero
fallback-caused inertia mismatches; fires rarely.

Scope: covers the spurious-zero / `Singular`-misclassification class (sawpath/
discs at iter 0). twirism1's LATE-iteration failure is a wrong NEGATIVE count
WITHOUT zeros (feral returns Success), which a zero-trigger cannot see and which
needs the expected inertia (pounce passes None today) — recorded as a follow-up,
not covered here.

Evidence: dev/research/issue-65-mc64-scaling-fallback.md,
dev/journal/2026-06-03-03.org, src/numeric/solver.rs (factor() fallback +
mc64_scaling_fallback_count), tests/issue65_mc64_fallback.rs,
src/bin/probe_issue65_{scaling,corpus}.rs, dev/scripts/regen_issue65_kkts.sh.

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
src/bin/probe_issue65_corpus.rs
src/bin/probe_issue65_scaling.rs
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
src/symbolic/small_leaf.rs
src/symbolic/supernode.rs

(truncated from      405 lines to 350 line budget)
