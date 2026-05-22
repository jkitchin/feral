# FERAL Context (auto-generated)

Generated: 2026-05-22T13:00:25Z

## Latest Session
File: dev/sessions/2026-05-21-04.md
```
# Session 2026-05-21-04

## Goal

Investigation session (no `src/` changes). Two questions from the human:

1. **#47** — do explicit-zero KKT entries still cost a ~2× slowdown on
   POUNCE CHO `parmest` after Fix 1 (`42434a5` fine-grained delayed
   pivoting) + Fix 2 (`80c05f5` cancellation-free 2×2 inertia)?
2. **#44** — is the NARX_CFy per-factor cost issue (ipopt-feral times
   out at 600 s) still relevant after the same two fixes?

## Accomplished

### #47 — reproduces, root cause pinned

A standalone iter-0 factor shows **no** penalty (stripped 439 ms vs
kept 456 ms), consistent with the issue's own caveat that iter-0 is not
the slow case. So the cost is in the warm-refactor path.

End-to-end POUNCE CHO `parmest` on current feral HEAD (measured by
*temporarily* repointing `pounce-feral` at the local checkout and
env-gating its zero-strip — **both POUNCE edits reverted afterwards,
binary rebuilt against `b3e4d3e`; POUNCE is untouched**):

| variant             | wall   | IPM iters | feral factor calls |
|---------------------|--------|-----------|--------------------|
| explicit zeros stripped | 10.6 s | 41    | 50                 |
| explicit zeros kept     | 22.6 s | 35    | 44                 |

#47 **still reproduces** — ~2.1× wall — with the #46 cascade fix in
place. It is not the #46 cascade.

**Root cause (pinned):** explicit zeros defeat the MC64 value-bounded
scaling cache (Track B2). New probe `probe_explicit_zeros` factors the
iter-0 KKT 4× on one warm `Solver`:

```
stripped:            cold 434ms -> warm 15/14/15ms    symbolic_calls=1  mc64_cache_hits=1,2,3
explicit zeros kept: cold 468ms -> warm 359/359/360ms  symbolic_calls=1  mc64_cache_hits=0,0,0
```

- Cold factor fine either way (~450 ms) — not a cascade, not fill.
- Symbolic analysis **is** reused either way (`symbolic_calls` stays 1)
  — the pattern fingerprint / symbolic cache is not the problem.
- The MC64 cache **never hits** with explicit zeros
  (`mc64_cache_hits` stays 0) — the Hungarian match reruns every
  factor, ~345 ms of the ~360 ms warm refactor. Stripped, the cache
  hits and the warm refactor collapses to ~15 ms (24× gap).

```

## Git Status
```
86fb953 fix(scaling): gate B2 cache on MC64-actually-ran, not ScalingInfo::Applied (#49)
bb3b712 docs(session): checkpoint 2026-05-21-04 — fix #47 value-aware scaling routing
129f268 docs(plan): record issue-47 value-aware routing plan
e49694b fix(scaling): make pick_scaling_strategy value-aware (#47)
ed147dd test(issue-47): add scaling-strategy routing diagnostic to probe_explicit_zeros
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-7246f50f8450e0fc)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_determinism_tiny ... ok
test test_gate_boundary_n_16 ... ok
test test_solve_parity_tiny_real_matrix ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 1 test
test src/symbolic/profiler.rs - symbolic::profiler::SymbolicProfiler (line 27) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-21-04.md)


`cargo run --bin bench --release`:

  Inertia match: 154432/154481 (100.0%)
  Residual pass: 154207/154481 (99.8%)
  Inertia match vs MUMPS: 154536/154588 (100.0%)
  Residual pass: 154256/154588 (99.8%)

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
small-frontal (<200)     147982     1.34     <= 2.0     PASS
medium (<500)            152145     1.74     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
small-frontal (<200)     153455     1.52     <= 2.0     PASS
medium (<500)            153560     1.53     <= 3.0     PASS

Identical to session 2026-05-21-03 (sparse medium 1.53 vs 1.52 is
run-to-run jitter). Expected — this session added only `src/bin`
probes, no library change.


name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10           51            0     (10, 0, 0)
spd_50             50           20            2     (50, 0, 0)
spd_100           100           77            5    (100, 0, 0)
spd_200           200          389           16    (200, 0, 0)
kkt_10_3           13            2            0     (10, 3, 0)
kkt_30_10          40           20            1    (30, 10, 0)
kkt_50_15          65           49            2    (50, 15, 0)
kkt_100_30        130          203            7   (100, 30, 0)

8 matrices benchmarked

All inertia exact, factor times in line with prior sessions. The bench
harness uses in-repo synthetic matrices, not the POUNCE corpus, so #47
(a warm-refactor caching effect on a real KKT) does not surface here —
expected; the #47 evidence is the `probe_explicit_zeros` table above.

```

## Recent Decisions
is not split into per-source variants — that is a wider type change
touching every scaling consumer; gating on the strategy that was
actually requested/routed is exact and local. The cost is one
`pick_scaling_strategy` call on the `Auto` path, already O(nnz) and far
under factor cost.

**Not the #49 cost regression.** This fix is correctness-only.
Standalone feral factor cost on ex4_2 is flat regardless of cache state
(_320 ~242–291 ms miss vs ~241–250 ms hit). #49's cost symptom (a
POUNCE run 10.8 s → 600 s timeout) was separately diagnosed as a rare
(~1/200-per-factorization) **execution-time race in the parallel
multifrontal driver** — value-deterministic output, ~1000× wall-time
blowup on one factorization — not the cache and not #47. That hang is
left scoped as its own task.

**Evidence.** New test
`mc64_cache_does_not_engage_on_infnorm_route_issue_49` (factors
`tridiag(6,10,1)` 3× on one `Solver`; asserts route is `InfNorm`,
`mc64_cache_hit_count()==0`, `symbolic_call_count()==1`) fails pre-fix,
passes after. `probe_cache_sequence` over all 10 dumped ex4_2_320 IPM
iterates: `mc64_hits` 3/10 → 0/10. All 23 solver tests and the full
`cargo test` suite green; `cargo fmt`/`clippy` clean. Committed
`86fb953`.

**References.**
- `dev/journal/2026-05-22-01.org` §07:49, §09:30, §10:10.
- `src/numeric/solver.rs` — the `mc64_ran` gate.
- `src/bin/probe_issue49.rs`, `probe_cache_sequence.rs`,
  `probe_hang_loop.rs` — diagnostic probes.
- `dev/sessions/2026-05-22-01.md` — session checkpoint.

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
src/bin/probe_hang_loop.rs
src/bin/probe_ir_trajectory.rs
src/bin/probe_issue_19.rs
src/bin/probe_issue45_ordering.rs
src/bin/probe_issue45.rs
src/bin/probe_issue46_preprocess.rs
src/bin/probe_issue46_supernode.rs
src/bin/probe_issue46.rs
src/bin/probe_issue49.rs
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

(truncated from      400 lines to 350 line budget)
