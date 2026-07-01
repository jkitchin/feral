# FERAL Context (auto-generated)

Generated: 2026-07-01T11:51:27Z

## Latest Session
File: dev/sessions/2026-07-01-03.md
```
# Session 2026-07-01-03

## Goal

Issue #99: "loop over all levers until faer-level performance" on the dense-front
factorization throughput gap (~3.5× to faer on the qap15 conic KKT, dominated by
a 2955×2955 indefinite root).

## Reality check (reported up-front, per protocol)

The issue's premises are **not reproducible on this branch**, and I established
this before writing code:

- **PR #92 is open/unmerged.** It (issue #91) contains the `OrderingPreprocess::Auto`
  fill-verification fix that made qap15 tractable *and* the qap15 fixture
  generator + `bench_qap15` harness. This branch is cut from current `main` and
  lacks all of it — `INTRAFRONT_MIN_AREA` is still `256*256`, not #92's `256*128`.
- **The qap15 fixture, its generator, `examples/{profile,bench}_qap15.rs`, and the
  two research notes the issue cites for "the full diagnosis" exist on no remote
  branch** (unpushed/lost work). The end-to-end qap15 number that defines the
  target cannot be reproduced here.
- **This container has 4 cores; the issue's numbers are 10-core.** The
  parallel-scaling levers (1 assembly, 2 schur scaling) cannot be validated
  against the issue's targets here.
- The issue itself states **no byte-exact lever closes 3.5×**; faer-class needs a
  policy decision (default-on FMA / static-SQD) the owner deliberately deferred.

I attempted to ask the owner the fixture-strategy and policy questions via
`AskUserQuestion`; the harness failed to deliver it (permission-stream error).
Told to continue, I proceeded autonomously with the **defensible subset**:
additive, default-off, byte-exact-preserving, measurable-here work only — no
unilateral default flips, no unvalidatable parallel retunes.

## Accomplished

**Issue #99 Lever 3 (per-core kernel throughput) — delivered as an opt-in knob.**

- New `examples/bench_dense_front.rs`: self-contained synthetic indefinite front
  (no external fixture), factored through the real `factor_frontal_blocked` path,
  timing nofma/FMA × serial/intrafront with an inertia-equality gate. Fills the
  measurement-harness hole the issue's (missing) `bench_qap15` left.
- New `BunchKaufmanParams::fma_min_front_area: Option<usize>` (default `None`) +
  `effective_front_fma(params, nrow, ncol)` helper. Gated at the single dense
  front-factor entry `factor_frontal_blocked_in_place_with_scratch` by shadowing
  `params` with an fma-flipped clone **only when the gate fires** (unarmed path
  pays nothing). Both multifrontal drivers funnel through this entry, so one
  insertion covers all.
- New `Solver::with_fma_large_fronts(min_area)` — writes straight to
  `numeric_params.bk` (no `NumericParams` field / funnel needed; low churn).
- `None` default ⇒ strict no-op: the production cross-arch bit-exact contract is
```

## Git Status
```
efb205b issue #99: packed BLAS-3 dense trailing update — byte-exact ~8-10x on dense fronts
45777c0 issue #99: checkpoint update — PR #92 merge + byte-exact intra-front win
ad70b5b issue #99: shape-aware intra-front gate (Lever 1, byte-exact 2.68x) + FMA row gate
c1d00a2 Merge PR #92 (issue #91) into issue-99 branch: qap15 ordering fix + harness + Lever B
1286297 issue #99: session 2026-07-01-03 checkpoint (FMA size gate)
```

## Test Status
```
test symbolic::tests::schur_symbolic_supernodes_cover_n ... ok
test symbolic::tests::schur_symbolic_tail_invariant_reversed_user_order ... ok
test symbolic::tests::schur_symbolic_tail_invariant_user_order ... ok
test symbolic::tests::symbolic_factorize_amf_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_auto_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 392 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.33s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-07-01-03.md)


cargo run --bin bench --release  →  Sparse failure analysis: no failures
Dense ∩ Sparse failure overlap: 0 / 0 / 0
(no oracle timings loaded in this environment; perf partitions N/A)
Default path unchanged (gate defaults None) → residual gate identical to the
2026-07-01-02 baseline (2/2, worst residual 1.26e-16).

examples/bench_dense_front 2955 5 (the new issue-99 harness):
  nofma serial     25586.15 ms  0.34 GFLOP/s  1.00×
  nofma intrafront  8631.85 ms  1.00 GFLOP/s  2.96×
  fma   serial     15422.96 ms  0.56 GFLOP/s  1.66×
  fma   intrafront  5142.82 ms  1.67 GFLOP/s  4.98×
  inertia (+1478,−1477,0) identical across all four variants ✓

```

## Recent Decisions
register-tiled kernel then autovectorizes to 9–10.5 GFLOP/s (22–26× isolated).

**Supersedes** the 2026-06-30 `tried-and-rejected` "DST-bandwidth-bound"
conclusion for this hardware: that came from packing the source into the *same
strided kernel* (which kept the strided `q`-access and only shrank the stride).
A proper packed micro-kernel with a contiguous `q`-loop is a different design and
wins decisively. Recorded there, not overwriting the prior entry.

**Byte-exactness.** Each `A[i,j]` (i≥j) is reduced over ascending `q` with the
identical `mul → sub` (nofma) / `mul_add` (fma) as the reference — packing changes
only memory layout, not arithmetic order. A zero alpha gives `acc − round(0·L) =
acc` for finite `L`, matching the strided zero-skip. Validated: full byte-exact
factor-parity suite green with packed default + `packed_matches_scalar_reference_
bit_for_bit` unit test.

**Scope.** Reached only for all-1×1-pivot panels (the `apply_blocked_schur` W-2
fast path). 2×2-pivot panels fall to the un-packed `axpy2` fallback, so strongly
indefinite fronts — and the `fma` path, whose rounding drifts more pivots to 2×2 —
benefit less. Definite / quasi-definite (SQD, regularized KKT) / SPD fronts get
the full win. Packing the 2×2 fallback (Phase B-2) is the next step.

**Evidence.** dense-1500 schur 3165→309 ms (10.2×); dense-2955 nofma serial
25586→3202 ms (8.0×, 0.34→2.69 GFLOP/s); synth qap15 stand-in end-to-end
3379→2029 ms; full byte-exact stack (intra-front + packed) 9247→2029 ms (4.56×),
inertia `(+30000,−2000,0)` unchanged. Suite 736 passed / 0 failed; clippy clean.

**Caveat.** Absolute GFLOP/s (packed ~10, vs a fully-tuned BLAS-3 core ~30–50) is
partly this 4-core container; the tile (8×4) and the lack of L2 cache-blocking /
FMA-in-packed leave headroom. Re-tune on target hardware. The `fma` packed path
exists (bit-matches the fma reference) but is not the default.

## Recent Tried-and-Rejected
+23% option but is a reproducibility-policy change (kept opt-in), not a
bit-exact win.

## 2026-07-01 — UPDATE: packed micro-kernel succeeds where B-1a source-pack failed (issue #99)

The 2026-06-30 "B-1a panel packing" entry above rejected source-panel packing as a
net slowdown and concluded the root front is DST-bandwidth-bound. That conclusion
was **specific to the variant tried** — packing the source into a tighter stride
but *feeding the same strided kernels*, which keep the per-`q` `as_simd` + strided
access. It does **not** generalize to a proper packed micro-kernel.

A different design — pack the panel into `q`-contiguous MR=8×NR=4 micro-panels and
run a register-tiled kernel with a **contiguous inner `q`-loop**
(`apply_schur_panel_range_packed`) — is **22–26× faster in isolation and
byte-exact** (`examples/bench_schur_micro`), and gives 8–10× on real dense fronts.
So the bottleneck was strided-`q` cache latency, not DST bandwidth, on this
hardware. Not a rejection — a correction of scope. See
`dev/research/issue-99-dense-front-fma-gate.md` UPDATE 3 and `dev/decisions.md`
2026-07-01 (packed BLAS-3). The B-1a *source-into-strided-kernel* variant remains
rejected; the packed micro-kernel is the shipped design.

## Source Files
```
src/bin/bench.rs
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
src/lu/condition.rs
src/lu/dense_factor.rs
src/lu/dense_matrix.rs
src/lu/dense_solve.rs
src/lu/dense_update.rs
src/lu/mod.rs
src/lu/scaling.rs
src/lu/sparse_factor.rs
src/lu/sparse_matrix.rs
src/lu/sparse_solve.rs
src/lu/sparse_symbolic.rs
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
tests/column_renumbering.rs
tests/column_renumbering_parity.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
tests/factors_ld_export.rs
tests/fine_grained_delay.rs
tests/fma_opt_in_roundtrip.rs
tests/growth_flag.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/issue91_preprocess_misfire.rs
tests/issue99_fma_front_gate.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/lu_dense.rs
tests/lu_ft_widebump.rs
tests/lu_scaling.rs
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
tests/profiler_smoke.rs
tests/property_tests.rs
tests/rook_rescue.rs
tests/rook_rescue_kkt.rs
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
