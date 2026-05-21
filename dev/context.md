# FERAL Context (auto-generated)

Generated: 2026-05-21T19:19:46Z

## Latest Session
File: dev/sessions/2026-05-21-01.md
```
# Session 2026-05-21-01

## Goal

Implement Track B2 of the per-factor cost cluster plan — eliminate the
per-call MC64 Hungarian on `rocket_12800` via a value-bounded MC64
scaling cache. Mid-session the goal pivoted (with human approval) to
**Track A**: investigate the `pinene_3200` iter 6-9 factor-time
explosion, which is 98 % of that problem's wall time.

## Accomplished

### B2 — value-bounded MC64 scaling cache (landed, then pivoted off)

- Wrote `dev/plans/mc64-value-bounded-cache.md`; implemented
  `src/scaling/value_bound.rs` (value-bound pure functions, 10 tests)
  and the Solver-scope `Mc64ScalingCache` in `src/numeric/solver.rs`
  (`with_mc64_cache` builder, cache-hit injection via
  `ScalingStrategy::External`, 5 integration tests).
- The cache is **correct and fully tested** but has **no measured
  corpus payoff**: the gate metric (diagonal dominance of `D·A·D`) is
  confounded by the IPM δ-regularization trajectory, and the MC64
  Hungarian it eliminates is < 2 % of factor cost. Ships as latent
  infrastructure. See `decisions.md` / `tried-and-rejected.md`.

### Fixed — `External` scaling 10× solve bug (pre-existing, latent)

- B2's integration test `mc64_cache_hit_bit_matches_cache_off` caught
  it: `ScalingStrategy::External` paired a real scaling vector with
  `ScalingInfo::NotApplied`. The factor applies `D·A·D`
  unconditionally; the solve keys un-scaling off
  `scaling_info != NotApplied` — so an `External` solve returned
  `D⁻¹A⁻¹D⁻¹b` (exactly 10× on `tridiag(6,10,1)`).
- Fix: the `External` arm now returns `ScalingInfo::Applied`.
  `NotApplied` is now exclusively `Identity` (genuine all-ones).
  Verified bit-identical across repeated calls; 302 lib tests + full
  suite green. Committed `c990def`.

### Track A — pinene_3200 iter 6-9 blowup characterized (A1)

- Established this is **issue #8**, and that the warm-state hypothesis
  was already disproved (2026-05-17): the cascade is structural to the
  iter-N matrix's *numeric content*, standalone ≈ warm.
- Ran `diag_pinene_pivot_cliff` (per-supernode 2×2 / delayed-pivot
  stats) on iterates 0008 and 0009 (n=127995, nnz=733k). **Direct
  evidence — the root front is a fully dense block, ~14 % of n:**

  | metric            |   0008 |   0009 |
  |-------------------|-------:|-------:|
  | root front nelim  |  15446 |  17538 |
```

## Git Status
```
ef5fb7e docs(session): checkpoint 2026-05-21-01 — A2 amplifier diagnosis
70f2e44 diag(dense): localize pinene KKT cascade — amplifier × two triggers
d3d93d2 docs(trackA): localize pinene cascade to the 2x2 stability gate
76174bd docs(session): checkpoint 2026-05-21-01 — B2 landed, pivot to Track A
c990def feat(scaling): value-bounded MC64 scaling cache (B2) + fix External 10× bug
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-8af795e9f581145d)

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
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-21-01.md)


PASS on all four exit-partition buckets — no regression from B2.

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.32     <= 2.0     PASS
medium (<500)            152145     1.74     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.57     <= 2.0     PASS
medium (<500)            153560     1.57     <= 3.0     PASS

```

## Recent Decisions
(`n_delayed = 0`, 1.25× factor) proved the forfeited tail columns are
pivotable — the break threw away real, doable work. Diagnosis:
`dev/research/kkt-cascade-amplifier-2026-05-21.md`.

**Why this is correctness-safe.** Swap-to-boundary is *real* delayed
pivoting, not force-accept or perturbation — the stuck column is
promoted to the parent front intact and re-attempted there with more
context. Inertia stays exact by construction. A `PivotOutcome::Delayed`
return leaves the front clean (columns `[k, nrow)` consistently
updated through pivot `k-1`), so the symmetric swap of two
un-eliminated columns introduces no inconsistency. The multifrontal
driver already maps the contribution block through `ff.perm`
(`factorize.rs` builds contrib row indices as
`row_indices[ff.perm[nelim + cj]]`), so the order of delayed columns
within the block does not matter. The change is bit-identical on any
matrix with no delayed pivots, and `may_delay == false` (root
supernode) never returns `Delayed` so root behaviour is unchanged.

**Evidence.** `pinene_3200_0009` (n=127995): `n_delayed`
133648 → 11309, factor-nonzeros ~165.7M → 3.6M (blowup 69× → 1.51×),
factor time ~183 s → 78 ms, inertia `(64000, 63995, 0)` exact and
unchanged. New tests `tests/fine_grained_delay.rs` (oracle: Bunch &
Kaufman 1977 pivot admissibility). Full suite + clippy green; bench
all four exit-partition buckets PASS.

**References.**
- `dev/research/kkt-cascade-amplifier-2026-05-21.md` — the diagnosis.
- `dev/plans/kkt-cascade-fix1-fine-grained-delay.md` — the plan.
- `dev/journal/2026-05-21-02.org` — implementation/test/benchmark log.
- `dev/sessions/2026-05-21-02.md` — session checkpoint.

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
tests/issue_46_saddle_kkt_cascade.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/maxfromm_parity.rs
tests/mc64_end_to_end.rs
tests/mc64_scaling.rs

(truncated from      369 lines to 350 line budget)
