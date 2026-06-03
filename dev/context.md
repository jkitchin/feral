# FERAL Context (auto-generated)

Generated: 2026-06-03T16:09:38Z

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
05f97fa fix(ordering): thin-large default prefers AMF up to n<=100k (#67)
5bcfecc Merge pull request #68 from jkitchin/claude/issue-63-diagnosis
892d7ef Merge remote-tracking branch 'origin/main' into claude/issue-63-diagnosis
0673f1b Merge pull request #69 from jkitchin/claude/issue-65-mc64-scaling
cfd8f68 docs(session): checkpoint 2026-06-03-03 — MC64 scaling fallback (#65)
```

## Test Status
```
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_resolves_to_amd_when_bisection_degenerates ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
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
trade-off never materialized at this scale: MetisND is both larger and
slower. Above the band, pinene_3200 (n=127995) still favors AMF (time_r
1.18, fill_r 1.20) but RDW2D51U (n=195075) did not complete a single pass
in ~10 min — the n>100k regime is qualitatively more expensive and
under-sampled.

Decision: bounded reroute. In `choose_adaptive`, when the size rule would
pick MetisND and `n <= AMF_BAND_MAX` (100_000), override to AMF. Rejected
alternatives: (a) an average-degree predicate — the same axis #50 warned is
dangerous, and the band needs no degree key because *every* band matrix the
corpus contains landed on the AMF side; (b) `AutoRace(Amf, MetisND)` —
measured 50–255% overhead (`probe_issue67_race`, median ~118%) because
MetisND's nested-dissection symbolic ordering is 2–5× more expensive than
AMF's, paying the losing candidate's cost on every solve for zero benefit.
The threshold is `n`, not the measured-sample identity: the mechanism
(AMF's lower fill + cheaper symbolic on thin patterns at this scale) is
size-bounded, so the rule generalizes to unseen band matrices rather than
memorizing these 36.

Scope guard: only the would-be-MetisND decision in (10_000, 100_000] is
touched. The `n > 100_000 && avg_deg < 5 → Amd` (#50 powerflow-class) and
`n > 100_000 && avg_deg >= 5 → MetisND` (genuinely-large 3-D) paths are
unchanged — pinned by the `choose_adaptive_rules` test (n=150_000 →
MetisND). pinene's above-band win is left on the table deliberately as the
safety margin.

Evidence: dev/research/issue-67-thin-large-ordering.md,
dev/journal/2026-06-03-04.org, src/symbolic/mod.rs choose_adaptive +
AMF_BAND_MAX, tests/issue67_thin_ordering.rs, src/bin/probe_issue67_thin.rs,
src/bin/probe_issue67_race.rs.

## Recent Tried-and-Rejected
   destroyed), inertia scrambled. Strictly worse. Force-accept-and-report-zeros
   is the useful behavior: it signals singularity so pounce escalates δ_w.

3. Any principled "better inertia" change. The ordering that wins (metis)
   reports a MORE pessimistic, LESS correct inertia (neg 255 ≠ 252 expected) on
   the singular matrix; that makes pounce regularize earlier and escape a frozen
   2.30e-8 fixed point. There is no known-correct inertia change that fixes
   scrs8 — "correct" inertia (amf) is what under-regularizes into the stall.

4. Ordering-class heuristic (route this KKT class to metis/scotch). Not pursued:
   the issue itself calls it "papering over the symptom," and it risks the
   cascade-break don't-regress set (robot_1600, NARX_CFy, marine_1600,
   rocket_12800, pinene_3200).

Conclusion: the durable fix is the δ_w / inertia-acceptance interaction
(pounce-side or joint), not FERAL factorization accuracy. Full analysis:
dev/research/issue-63-nearsingular-ordering-diagnosis.md;
dev/journal/2026-06-03-02.org; probe src/bin/probe_issue63_nearsingular.rs.
Future sessions: do NOT re-attempt a FERAL-only fix for scrs8 without first
re-checking these four dead ends.

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
src/bin/probe_issue63_nearsingular.rs
src/bin/probe_issue64_arrow.rs
src/bin/probe_issue65_corpus.rs
src/bin/probe_issue65_scaling.rs
src/bin/probe_issue67_race.rs
src/bin/probe_issue67_thin.rs
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

(truncated from      409 lines to 350 line budget)
