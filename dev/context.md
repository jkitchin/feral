# FERAL Context (auto-generated)

Generated: 2026-05-21T22:14:27Z

## Latest Session
File: dev/sessions/2026-05-21-03.md
```
# Session 2026-05-21-03

## Goal

Track A3 — validate Fix 1 (fine-grained delayed pivoting, `42434a5`)
end-to-end via `probe_kkt_replay` on `pinene_3200`, `robot_1600`,
`marine_1600`; confirm the pinene iter 6–9 factor-time explosion is
gone and per-iter inertia stays exact.

A3 surfaced a **correctness regression**; per the human's "fix forward
this session" instruction the goal expanded to: keep Fix 1, diagnose
and fix the residual so pinene is both fast *and* inertia-exact.

## Accomplished

### A3 validation — found a regression

Fix 1 broke the pinene delayed-pivot cascade (456 s → 4.7 s) but
returned `WrongInertia` on the borderline near-singular iterates 8/9
(δ_c ≈ 1e-11): a spurious `inertia.zero`. Pre-Fix-1 warm replay was
all-exact (456 s, worktree at `ef5fb7e`) — so this is a Fix 1
regression. It violated the hard rule "inertia must be exactly correct
on non-singular matrices."

### Root cause — pre-existing 2×2-inertia cancellation bug

Fix 1 did not *cause* the regression; it *exposed* one. The pre-Fix-1
break-on-first cascade dumped ~116k–133k columns to a dense root front
whose full BK pivoting gave Sylvester-exact inertia — that cascade was
silently buying correctness. The latent bug: `count_2x2_inertia` /
`count_2x2_inertia_val` classified signs from `λ = 0.5·(tr ∓ s)`;
although `s` is cancellation-free, the *final* subtraction `0.5·(tr∓s)`
cancels — a genuine non-singular 2×2 whose small eigenvalue is below
`ULP(0.5·tr)` IEEE-rounds to *exactly 0.0*, counted as a `zero`.

### Fix 2 — cancellation-free 2×2 inertia classification

Lifecycle: research (journal §17:40/§18:05) → plan
(`dev/plans/kkt-cascade-fix2-2x2-inertia-cancellation.md`) →
tests-first → implement → verify → benchmark.

- Added `det_sym2x2` — Kahan fused difference-of-products
  (`w=fl(d21²)`, `e=fma(d21,d21,-w)`, `det=fma(d11,d22,-w)+e`),
  relative error ≤ 2·u for any inputs.
- Added `classify_2x2_inertia` — classifies from `sign(det)` +
  `sign(tr)`: `det<0`→(1,1,0); `det>0`→(2,0,0)/(0,2,0); `det==0`
  exactly→(1,0,1)/(0,1,1)/(0,0,2).
- `count_2x2_inertia_val` delegates to it; `count_2x2_inertia`'s three
  branches reclassified through it (force-accept bands fold a genuine
  zero into `neg`, preserving #42 Option A; non-singular branch reports
```

## Git Status
```
12585d9 docs(journal): record #47 root cause and #44 assessment
2eab12f test(issue-44): add NARX_CFy per-factor cost probe
6185415 test(issue-47): add explicit-zeros warm-refactor probe
787315f docs(session): checkpoint 2026-05-21-03 — Track A3 + Fix 2 (2×2 inertia)
80c05f5 fix(inertia): classify 2×2 blocks from cancellation-free sign(det) (#48)
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-bf35e57908f5d612)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_gate_boundary_n_16 ... ok
test test_determinism_tiny ... ok
test test_solve_parity_tiny_real_matrix ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 1 test
test src/symbolic/profiler.rs - symbolic::profiler::SymbolicProfiler (line 27) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-21-03.md)


`cargo run --bin bench --release`:

  Inertia match: 154432/154481 (100.0%)
  Residual pass: 154207/154481 (99.8%)
  Inertia match vs MUMPS: 154536/154588 (100.0%)
  Residual pass: 154256/154588 (99.8%)

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
small-frontal (<200)     147982     1.32     <= 2.0     PASS
medium (<500)            152145     1.74     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
small-frontal (<200)     153455     1.52     <= 2.0     PASS
medium (<500)            153560     1.52     <= 3.0     PASS

No regression vs the previous session. Inertia match 100.0% on both
paths; all four exit-partition buckets PASS.

```

## Recent Decisions
the delayed-pivot cascade that had been masking it.

**Why not `s` vs `|tr|`.** A comparison `s ⋚ |tr|` was considered and
**rejected**: for a block one of whose diagonal entries lies far below
the other's ULP, *both* `tr = d11+d22` and the discriminant
`(d11-d22)²` annihilate the small entry — the same cancellation — so
`s == |tr|` and it still mis-reports `det == 0`. The Kahan determinant
does not, because the product `d11·d22` never adds `d11` into `d22`.
(Worked example: journal `2026-05-21-03.org` §18:05.)

**Issue #42 Option A is preserved.** The two force-accept branches of
`count_2x2_inertia` (`zero_tol_2x2` / `null_pivot_tol_2x2` bands) still
never report a `zero`: a genuine zero eigenvalue from
`classify_2x2_inertia` is folded into `neg`, matching the pre-existing
`λ>0 → pos, else → neg` convention. The non-singular `else` branch
reports `zero` honestly — but `det_sym2x2` is accurate there, so it is
structurally 0 for any genuinely non-singular block.

**Evidence.** Tests-first: 4 new in-file tests (oracle = diagonal-2×2
inertia by hand calculation) failed on the pre-fix code, all 19
`sym2_inertia_tests` pass after. `probe_kkt_replay` default config:
`pinene_3200` all 10 iterates `(64000,63995,0)` exact (was iters 8/9
`WrongInertia`); `marine_1600` all 18 exact (was iter 17
`WrongInertia` — the defect filed as #48); `robot_1600` unchanged.
Bench inertia match 100.0%, all four exit-partition buckets PASS.

**References.**
- `dev/plans/kkt-cascade-fix2-2x2-inertia-cancellation.md` — the plan.
- `dev/journal/2026-05-21-03.org` §17:40/§18:05/§18:45/§19:00.
- `dev/sessions/2026-05-21-03.md` — session checkpoint.

## Recent Tried-and-Rejected

- Even with a perfect gate, B2 targets <2 % of the cost. pinene_3200's
  10 iters total 493.9 s; iters 6-9 are 64.8/77.8/135.7/208.2 s (the
  cost-cluster blowup, 98 %). The MC64 Hungarian is ≤6 s total.

- The named target rocket_12800 cannot even exhibit a hit: its 2-iter
  dump changes pattern between iters (332793→435190 nnz).

**What was kept.** The cache wiring (`Solver::with_mc64_cache`),
`src/scaling/value_bound.rs`, and — separately — the `External`
scaling correctness fix B2 surfaced (see `decisions.md` 2026-05-21).
All correct and tested; the *approach* of a cheap value-proxy gate
for cross-iteration MC64 reuse is what is rejected.

**Lesson.** Validate the cost model before building the optimization.
B2 assumed "MC64 Hungarian reruns every IPM iter and dominates" — true
for rocket_12800's iter-0 profile, false for pinene's actual 10-iter
trajectory where the delayed-pivot blowup dwarfs everything. A
per-factor profile of the *named target's full iteration sequence*,
not a single iteration, should precede the plan.

## Source Files
```
src/bin/alloc_probe.rs
src/bin/bench_axpy_small.rs
src/bin/bench_fma_phase3.rs
src/bin/bench_issue8.rs
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
src/bin/diag_leaf_profile.rs
src/bin/diag_max_ncol.rs
src/bin/diag_mc64_cycles.rs
src/bin/diag_mittelmann.rs
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
src/bin/polak6_diag.rs
src/bin/policy4_diag.rs
src/bin/probe_acopp30_64.rs
src/bin/probe_cascade_perturb.rs
src/bin/probe_clnlbeam_refine.rs
src/bin/probe_clnlbeam_shape.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_dtoc2_mc64.rs
src/bin/probe_explicit_zeros.rs
src/bin/probe_f01.rs
src/bin/probe_fbrain.rs
src/bin/probe_fma_kernel.rs
src/bin/probe_ir_trajectory.rs
src/bin/probe_issue_19.rs
src/bin/probe_issue45_ordering.rs
src/bin/probe_issue45.rs
src/bin/probe_issue46_preprocess.rs
src/bin/probe_issue46_supernode.rs
src/bin/probe_issue46.rs
src/bin/probe_kkt_replay.rs
src/bin/probe_marine_shape.rs
src/bin/probe_marine_time.rs
src/bin/probe_mc64_spread.rs
src/bin/probe_mc64_synth.rs
src/bin/probe_narx_factor.rs
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
```

## Test Files
```
tests/amf_corpus_oracle.rs
tests/auto_strategy.rs
tests/blocked_ldlt.rs
tests/build_row_indices_trailing_invariant.rs
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
tests/fine_grained_delay.rs
tests/fma_opt_in_roundtrip.rs
tests/growth_flag.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs

(truncated from      377 lines to 350 line budget)
