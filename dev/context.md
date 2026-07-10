# FERAL Context (auto-generated)

Generated: 2026-07-10T17:13:00Z

## Latest Session
File: dev/sessions/2026-07-10-06.md
```
# Session 2026-07-10-06

## Goal

Implement issue #131 (parallelism gaps) after the session-05 measure-first pass
found it — unlike #130/#132/#133 — has genuine headroom. User scope: solve gap
(Gap A) → full contribution-block rewrite; assembly gap → #125 static maps
first, then Gap B.

## Accomplished

- **#125 — analysis-time static assembly maps** (commit `efa1000`).
  Precompute each supernode's `[own cols | sorted trailing]` frontal row layout
  at symbolic time (`compute_static_row_indices`, one postorder pass, CSR-flat
  on `SymbolicFactorization`); the numeric factor reads it on the no-delay fast
  path (`n_delayed_in == 0`) instead of recomputing with `build_row_indices`.
  Bit-identical (delayed fronts fall back). Tests (`tests/static_assembly_maps.rs`,
  8): A/B factor byte-identical static-on vs -off incl. KKT-with-delays; an
  independent BTreeSet oracle == `static_rows(i)`. **Bench: grid220 (n=48400)
  per-supernode loop 164.8→151.6 ms (~8%), warm factor 176.0→168.1 ms (~4.5%);
  arrow wash.**

- **#131 Gap B — measured not justified, not built** (commit `93cf6ed`).
  Assembly is 8.3% of factor on grid220 / 1.5% on dense1400, and independent
  fronts' assembly already overlaps in the parallel driver; only the serial
  root-front O(nrow²) remains, behind the already-parallel O(nrow³) factor.
  <1–3% ceiling → skipped with evidence (user agreed).
  `dev/research/issue-131-gapb-assembly-measure-2026-07-10.md`.

- **#131 Gap A (1/n) — carry the assembly tree into `SparseFactors`** (commit
  `c12a5d3`). `node_parents: Vec<Option<usize>>` mirrored from symbolic in all
  constructors. Behavior-neutral foundation for the tree-parallel solve.

- **#131 Gap A (2/n) — contribution-block tree-parallel single-RHS solve**
  (commit `d83ca3e`). Opt-in `solve_sparse_cb(factors, rhs, parallel)`; default
  `solve_sparse` untouched (no rebaseline). Forward = contribution-block
  (children summed in fixed ascending order → serial-CB == parallel-CB
  byte-identical); backward = unchanged shared-vector arithmetic, root-down.
  Global `y` written only at disjoint eliminated rows (concurrent via a
  Send+Sync raw-pointer wrapper + disjointness safety comment). Subtree-cost
  **coarsening** (`CbTaskPlan`) + a `worthwhile` gate (≥2 task roots, no
  Amdahl-dominant front, total ≥ 1e6 flops) — per-node tasks were far too fine.
  Tests (`tests/cb_solve_parity.rs`, 6, stable under RAYON_NUM_THREADS=8):
  serial==parallel byte-identical incl. an n=9216 concurrent-path fixture,
  KKT-with-delays, dense fast path; determinism; valid solve.
  **Bench: grid220 default solve 13.5 ms → cb_parallel 6.6 ms at 4 threads
  (~2.0×); cb_serial 13.0 ms (no regression); arrow → serial fallback,
  near-neutral.**

- **#131 Gap A (3/n) — pool the CB workspace + wire into `Solver`** (commit
```

## Git Status
```
ab4b2fc #131 Gap A (3/n): pool the CB workspace and wire it into Solver
3d91ffb session 2026-07-10-06 checkpoint: #125 + #131 Gap A/B
d83ca3e #131 Gap A (2/n): contribution-block tree-parallel single-RHS solve
c12a5d3 #131 Gap A (1/n): carry the assembly tree into SparseFactors
93cf6ed research: #131 Gap B (parallel assembly) measured not justified
```

## Test Status
```
test symbolic::tests::schur_symbolic_single_schur_index ... ok
test symbolic::tests::schur_symbolic_tail_invariant_reversed_user_order ... ok
test symbolic::tests::schur_symbolic_tail_invariant_user_order ... ok
test symbolic::tests::symbolic_factorize_amf_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_auto_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_external_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 405 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.98s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-07-10-06.md)


cargo run --bin bench --release: no matrices in this container (corpus absent,
all buckets count 0). Perf evidence is from src/bin/perf_probe.rs (warm factor +
solve, RAYON_NUM_THREADS-aware) and src/bin/probe_panel_frag.rs (phase timing):

#125   grid220 loop 164.8→151.6 ms (~8%), warm factor 176.0→168.1 ms (~4.5%)
Gap B  grid220 assembly 8.3% of factor; dense1400 1.5%  (skipped)
Gap A  grid220 solve 13.5 ms (default) → 6.6 ms (cb_parallel, 4t) = ~2.0×
       arrow (path) → worthwhile gate → serial fallback, near-neutral

```

## Recent Decisions

The tree-parallel single-RHS solve (`solve_sparse_cb`) is a **separate,
opt-in** path, not a replacement of `solve_sparse`. Rationale: a bit-exact
tree-parallel forward substitution must use a contribution-block reduction
(sum tree, fixed child order) rather than the default core's shared-global-
vector left-fold. Those two accumulation orders are not float-bit-identical, so
converting the default path in place would shift every ~1e-15 residual baseline
(and the single-vs-many-core bit-parity test). Keeping the CB solve as its own
path leaves the default `solve_sparse` — and every existing test/baseline —
untouched, and the #131 "serial == parallel byte-identical" contract is
satisfied within the CB path itself (serial-CB == parallel-CB by construction:
the child-reduction order is fixed regardless of thread scheduling). The
backward substitution keeps the default arithmetic unchanged (separator rows
are read-only, eliminated rows disjoint), so only forward is contribution-block.
Coarsening (subtree-cost task roots) and a `worthwhile` gate are required for a
net win — per-node rayon tasks are far too fine for the tiny per-front solve
work. Evidence: `dev/research/issue-131-parallelism-design-2026-07-10.md`,
`tests/cb_solve_parity.rs`, ~2.0× on grid220 (n=48400) at 4 threads.

## 2026-07-10 — #131 Gap B (parallel assembly): measured not justified, not built

Per-front assembly is 8.3% of the factor on grid220 / 1.5% on dense1400, and in
the parallel driver independent fronts' assembly already overlaps across
threads (each front's assembly is part of its own tree task). The only assembly
left on the critical path is the root/near-root fronts' O(nrow²), behind the
root's O(nrow³) dense factor that intra-front parallelism (Lever 1.1) already
targets — so column-partitioned parallel assembly would chase <1–3% of the
factor. #125 already captured the tractable, bit-exact assembly win
(`build_row_indices`). Not built. Evidence:
`dev/research/issue-131-gapb-assembly-measure-2026-07-10.md`.

## Recent Tried-and-Rejected
sweep replays across four hand constructions (journal 2026-07-10-01,
research note §UPDATE).

Also rejected en route: classic **Kahan** compensation for the sweep
accumulator (its `y = v − c` pre-subtraction re-absorbs the compensation
into the next 2²⁰-scale addend — computed `0.0` again; verified
numerically); the **Neumaier** two-sum variant works and shipped. And three
regression-matrix constructions whose base or replacement was numerically
singular for every path (±1 cascade to 2³⁴: `σ_min(B') = 1.5e-16`; diag-4
cascade: rescue-true `4.5e-13 <` ztol; spike-poison m=6: fresh LU burns the
4e6 spike entry and deflates its tail pivot to 0) — any single-shot
absorption reproducer necessarily has `σ_min(B') ⪅ δ·∏retained`, so the
"fresh factor succeeds" oracle is unsatisfiable without a multi-update
imbalance history.

**Shipped instead.** Always-on Neumaier-compensated scatter (recovers the
true pivot bit-for-bit on the regression basis) + `update_pivot_search` as an
always-on opt-in trajectory variant (bounded multipliers across chains),
default false. See `dev/research/issue-112-bg-update.md` §UPDATE and
`dev/decisions.md` 2026-07-10.

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
tests/cb_solve_parity.rs
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
tests/issue102_intrafront_deadlock.rs
tests/issue102_ordering_escalation.rs
tests/issue107_external_ordering.rs
tests/issue112_bg_update.rs
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
tests/lu_adversarial_inputs.rs
tests/lu_dense.rs
tests/lu_dense_update_bg.rs
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
tests/static_assembly_maps.rs
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
