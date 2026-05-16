# FERAL Context (auto-generated)

Generated: 2026-05-16T19:44:28Z

## Latest Session
File: dev/sessions/2026-05-16-13.md
```
# Session 2026-05-16-13

## Goal

Address GitHub issue #13 ("Per-supernode fixed-overhead reduction"):
attack the 600-1900 ns/supernode setup cost on the long-tail KKT
corpus (CRESC100/ACOPR30/KIRBY2 cluster). Three candidate targets,
in order of expected leverage:

1. Workspace pooling in `factor_frontal_blocked_in_place` (eliminate
   per-front `vec![0.0; nrow*nrow]`, pool L/D/contrib).
2. Bypass `SymmetricMatrix::validate()` when the caller has already
   validated the input.
3. `extend_add` direct slice writes (already done in earlier Phase B).

## Accomplished

### Combined Target (1) + (2): in-place + scratch-pooled scalar fallback

Commit: `ae980ab` -- `perf(dense): in-place scratch-pooled scalar
fallback (#13)`

The hot path through `factor_frontal_blocked_in_place_with_scratch`
delegates to the scalar `factor_frontal` whenever `ncol < PANEL_MIN_NCOL
= 8` or `bs < 2`. On the long-tail KKT corpus (CRESC100, ACOPR30,
KIRBY2) the median supernode size is 2-4, so **every** supernode hit
this fallback. The public `factor_frontal` paid two redundant fixed
costs per call:

- `matrix.validate()` -- O(n^2/2) NaN/Inf scan of the lower triangle
- `vec![0.0; nrow*nrow]` + lower-triangle copy of `matrix.data`

Both are pointless on the multifrontal driver's call path: the
assembled frontal is built from value-checked CSC values into the
driver's own `matrix`, the driver already holds the data in
`matrix.data`, and it is prepared to discard `matrix.data` content
after the in-place factor.

Extracted `factor_frontal_in_place_with_scratch` (private, takes
`&mut SymmetricMatrix` + `&mut FactorScratch`), refactored
`factor_frontal_with_profile` to validate + copy + delegate, and
routed the panel-fallback site through the new variant. The new impl
reuses `scratch.subdiag` and `scratch.contrib_pool` from the existing
Phase A/C pool. The finite-input invariant is enforced by a
`debug_assert!`. Public `factor_frontal` API untouched (preserves
test/bench/external caller behaviour).

### Evidence (`cargo run --bin diag_supernode_cost --release`)

| matrix         | ns/sup before | ns/sup after | delta  |
```

## Git Status
```
ae980ab perf(dense): in-place scratch-pooled scalar fallback (#13)
7f586e6 feat(stress): M3 corpus expansion -- 104 SuiteSparse matrices (#26)
00fbcb5 chore(context): refresh dev/context.md after session 2026-05-16-08
f76cb2d chore(session): 2026-05-16-08 -- SQD fast-path phases (c)-(h) (#34)
499e5de bench(#34): bench_sqd synthetic SQD-vs-BK harness (phase g)
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-583925e390be7782)

running 5 tests
test test_gate_tiny_sparse_in ... ok
test test_gate_just_outside_n_tiny ... ok
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
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-16-13.md)


=== Dense perf vs canonical oracles (154481 matrices) ===
ratio          count  geomean   p50   p90   p99   max
factor/MUMPS  153472    0.22   0.10  1.95 22.95 47.93
factor/SSIDS  154393    0.02   0.01  0.40  5.49 14.82

=== Sparse perf vs canonical oracles (154588 matrices) ===
ratio          count  geomean   p50   p90   p99   max
factor/MUMPS  153560    0.41   0.30  1.70  2.52 28.21
factor/SSIDS  154500    0.04   0.03  0.33  0.77  3.07

--- Dense Phase 2.8.1 exit partition ---
small-frontal (<200)  147982   1.37  <= 2.0  PASS
medium (<500)         152145   1.78  <= 3.0  PASS

--- Sparse Phase 2.8.1 exit partition ---
small-frontal (<200)  153455   1.69  <= 2.0  PASS
medium (<500)         153560   1.70  <= 3.0  PASS

KKT summary: 156927 matrices (154481 dense-eligible n <= 1000)
  Inertia match: 154428/154481 (100.0%)
  Residual pass: 154207/154481 (99.8%)

```

## Recent Decisions

**Why.** Phase (g) measurement (`src/bin/bench_sqd.rs`, 2026-05-16
M4 Pro) shows geomean speedup 1.025–1.05× across 6 synthetic SQD
shapes (tiny-dense through large-banded; n = 16..1000), with
~5% noise band that flips the sign on individual shapes. The
shared rank-1 trailing-update kernel (`do_1x1_update`) dominates
per-pivot wall-clock — skipping the BK 1×1-vs-2×2 search saves
only a modest constant per column.

**What ships:** the *contract*. Vanderbei (1995) Theorem 2.1
guarantees a diagonal D for any SQD input, *independent* of any
pivot search succeeding. For matrices near the BK pivot
threshold (IPM KKT systems as μ shrinks, ill-conditioned saddle
systems from constrained QP), the SQD path can complete cleanly
where BK is forced into 2×2 pivots, rook rescues, or
delayed-pivot cascades. Trips on the two contract guards
(`|d| > zero_tol`; `max|l_{ik}| <= 1/sqrt(EPS)`) surface
`FeralError::SqdContractViolated { column, pivot }` immediately
— never silent BK fallback.

**Default unchanged:** `NumericParams::sqd_mode = false`. Callers
who can assert the contract opt in via
`Solver::new().with_sqd_mode(true)`.

References:
- `dev/research/sqd-fast-path.md`
- `dev/sessions/2026-05-16-08.md` (phase-by-phase ship log)
- Commits: 58e7421 (c), 05730a4 (d), b44b9d9 (e), 4adef8c (f),
  499e5de (g).
- GH: #34

## Recent Tried-and-Rejected
**Symptom.** All 16 Schenk_AFE matrices have n ∈ [504855, 1508065].
The M3 issue specifies "n ≤ 100k" for the GHS_indef tier;
extrapolating that size cap to the AFE group leaves zero candidates.
Of the 16, 10 are SPD (`af_*_k101` family + half of `af_shell*`),
which the issue explicitly excludes ("skip the SPD ones"). The 6
indefinite shells (`af_shell1/2/5/6/9/10`) range from 504k to 1.5M
rows; sample timing on an existing 50k×500k matrix is ~40 ms factor,
so a 500k×17M shell would land in the 5–20 s range each. Six of
them would be ~1–2 minutes of suite time — fine for budget — but
they are coarse mesh slices of the same finite-element problem
(automotive shell, sequential time steps) and add little diversity
beyond a single representative.

**Why it was rejected (for this issue).** Bringing in 6 near-duplicate
mesh slices burns row-count headroom that's better spent on
structurally diverse matrices in the smaller-n tier. A future ticket
that wants to stress the dense-supernode path on million-row
matrices should add 1–2 representative `af_shell*` rows with a fresh
research note on whether they exhibit fill patterns distinct from
what `sparsine` / `copter2` already cover.

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
src/bin/hs85_diag.rs
src/bin/parallel_corpus_parity.rs
src/bin/polak6_diag.rs
src/bin/policy4_diag.rs
src/bin/probe_acopp30_64.rs
src/bin/probe_cascade_perturb.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_ir_trajectory.rs
src/bin/probe_issue_19.rs
src/bin/probe_panel_attribution.rs
src/bin/probe_scaling_policy4.rs
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
tests/fma_opt_in_roundtrip.rs
tests/growth_flag.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/maxfromm_parity.rs
tests/mc64_end_to_end.rs
tests/mc64_scaling.rs
tests/multi_rhs.rs
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
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
