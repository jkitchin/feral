# FERAL Context (auto-generated)

Generated: 2026-06-11T07:23:58Z

## Latest Session
File: dev/sessions/2026-06-10-01.md
```
# Session 2026-06-10-01

## Goal

Continue the `/loop` over `dev/research/repo-review-2026-06-09.md` §4
"Lower severity" ordering-crate findings (O-series), one finding per
iteration: each fix starts with a reproducing test; anything that cannot
be reproduced is routed to `dev/tried-and-rejected.md` citing the finding
ID. User override "finish this tonight" → complete the remaining O-series
findings back-to-back. Per the loop rule, do not start the next issue once
the series is exhausted.

## Accomplished

The O-series of the repo review is now **complete (O1–O21 all addressed)**.
This session closed out O16–O21:

- **O16** (`ece870c`) — kahip flow refinement balanced by raw vertex count
  on a *weighted* coarse graph. RED→GREEN: a weighted bisection test that
  the count-based balance accepted and the weight-aware balance rejects.
  User-visible → CHANGELOG.
- **O17** (`0c519e9`) — kahip `apply_degree2` rescans the seed list from 0
  each fixpoint round (O(n²)). The drafted monotone-cursor fix was
  *disproved* by code reading: the simplicial collapse adds no compensating
  edge, so a degree-3 endpoint can drop to degree 2 at an index below the
  cursor, reordering the op stack and changing the reconstructed
  permutation. Routed to tried-and-rejected (O17) + a documenting comment;
  no behaviour change.
- **O18** (`54c805a`) — `KahipStats` drift: `n_components` never surfaced,
  `cycles` doc described it as a per-coarsening-level counter. RED→GREEN
  on `n_components` (disconnected-components test asserts `== 2`); `cycles`
  doc corrected to "one V-cycle per bisection". User-visible → CHANGELOG.
- **O19** (`6bd1c6e`) — kahip push-relabel stranded-vertex branch sets
  height without decrementing `height_count` (would corrupt the gap
  histogram) but is unreachable: an active vertex always has a residual
  reverse edge. Added a `debug_assert_ne!` guard + comment; routed the
  dead branch to tried-and-rejected (O19).
- **O20** (`545ec1f`) — thrice-copied ND driver scaffolding across
  metis/scotch/kahip `node_nd.rs`, already drifted. The cross-crate
  consolidation is a non-reproducible refactor → deferred to
  tried-and-rejected (O20). The testable slice was harmonized: metis
  inverted `iperm` inline with `vec![0; n]` and only a range check (could
  silently emit a non-bijection); extracted `invert_iperm` mirroring
  scotch/kahip (`vec![-1; n]` + duplicate-position check). RED→GREEN
  (`invert_iperm_rejects_duplicate_positions`). Behaviour-neutral on
  reachable inputs.
- **O21** (this session's final commit) — AMF `finalize_step_amf` overloads
  `wf[e] = 0` as both the lazy-cache "uncached this iteration" sentinel and
  a possible genuine surface contribution of 0 (`amf_wf_surface` is 0 when
  `dext == 2*deg-1`), so a live element with true surface 0 is recomputed
```

## Git Status
```
bd1640a docs(ordering-core): document the AMF wf=0 cache-sentinel overload (O21)
545ec1f refactor(metis): extract invert_iperm with a duplicate-position check (O20)
6bd1c6e docs(kahip): assert the push-relabel stranded branch is unreachable (O19)
54c805a fix(kahip): surface n_components in KahipStats, fix cycles doc drift (O18)
0c519e9 docs(kahip): document the O(n^2) degree-2 seed scan and its cursor trap (O17)
```

## Test Status
```
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
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

test result: ok. 359 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.48s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-10-01.md)


`cargo run --bin bench --release` (unchanged by this session — all O-series
work was doc/hardening, not perf-affecting). All Phase 2.8.1 exit-partition
gates PASS:

=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===
ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.41       0.30       1.43       2.23       7.46
solve/MUMPS        153560       0.08       0.08       0.16       0.89       2.69
factor/SSIDS       154500       0.04       0.03       0.28       0.67       2.46
solve/SSIDS        154500       0.96       1.00       2.86      10.67      39.00
nnzL/MUMPS         153560       0.61       0.58       0.75       4.50      23.11
nnzL/SSIDS         154500       0.88       1.00       1.00       4.50       5.00

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.34     <= 2.0     PASS
medium (<500)            152145     1.76     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.43     <= 2.0     PASS
medium (<500)            153560     1.43     <= 3.0     PASS

Worst factor-ratio outliers vs MUMPS unchanged (KIRBY2_0007 7.46×,
CRESC132_0000 5.97×) — pre-existing, untouched by this session.

```

## Recent Decisions
Decision and rationale:

- **In-place bump elimination with partial pivoting.** The spike `ρ = G⁻¹L⁻¹P·aₙₑw` is set
  into `U`'s column `r`; the bump `[r, h]` (`h` = max spike support) is re-triangularized by
  sparse Gaussian elimination. **Partial pivoting is mandatory** and is the resolution of the
  zero-pivot landmine documented when this was deferred: the naive column-shift Bartels–Golub
  makes the Hessenberg diagonal pivots the old superdiagonal `U[k,k+1]`, which are frequently
  zero in a sparse `U`; partial pivoting instead uses a nonzero sub-diagonal spike entry as
  the pivot via a row interchange.

- **Swaps go into the eta, not the base `L`.** The unit-lower base `L` is never permuted
  (permuting the fully-formed `L` would break its triangularity). The bump elimination's
  elementary operations — `FtOp::Swap` (partial-pivot interchange) and `FtOp::Axpy`
  (`row -= mult·row`) — are recorded as a `FtEta` and replayed on the solve vector between the
  `L`-solve and `U`-solve in `ftran` (transposed, reversed, between `Uᵀ` and `Lᵀ` in `btran`).
  `U` is updated in place. Maintained invariant: `P A Q = L G U`, `G = E₁⁻¹…Eₜ⁻¹`.

- **`U` stored as mutable per-row vectors** (`Vec<Vec<(col,val)>>`, diagonal first) rather than
  flat CSR, so the in-place row operations / swaps / merges are tractable.

- **Consequence.** Warm-solve cost is bump-local (`O(Σ bump)`), independent of `n` for
  localized spikes (the realistic LP regime) — demonstrated flat across n=1000..8000. The
  inherent worst case is a dense spike (e.g. tridiagonal, where `L⁻¹` is dense), where the
  bump spans the tail and the cost degrades toward the old product-form; this is fundamental
  to any update scheme and bounded by the `max_updates` refactor budget.

- **Stability/budget.** Growth monitor over elimination multipliers → `NeedsRefactor` on
  `max_growth`; no acceptable bump pivot → `SingularBasis`; update count → `NeedsRefactor` on
  `max_updates`. Work is done on a clone of `U`, committed only on success, so failures leave
  `self` unchanged.

## Recent Tried-and-Rejected
The broader finding framing — that the ~600 LoC of duplicated AMD/AMF inner-loop
code is itself a drift hazard — is a structural-refactor observation (the same
class as O20's cross-crate consolidation), not a defect with a reproducing test,
and is likewise deferred rather than undertaken inside a single /loop iteration.

### Disposition

Routed here per the /loop rule (non-reproducible core → tried-and-rejected citing
the finding ID). Per the X16 / O4 / O5 / O6 / O9 / O17 sub-fix precedent, the
safe behaviour-neutral slice is applied: a comment block at the Pass-1 sentinel
reset documenting that `0` is deliberately overloaded as both "uncached" and a
possible genuine surface value, that the resulting recompute is benign (same
value), and why a distinguishing sentinel was rejected; plus a one-line pointer
at each of the two Pass-2 cache-check sites. No behaviour change.

Evidence: `algo.rs:955` (first-touch `wf[e] = 0` sentinel), `:983-988` /
`:1015-1018` (the `if wf[e] == 0` recompute sites), `:664-666`
(`amf_wf_surface`, zero at `dext == 2*deg-1`). `cargo test -p
feral-ordering-core` green (comment-only, proves no behaviour change). Journal:
dev/journal/2026-06-10-01.org.

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
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
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
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/lu_dense.rs
tests/lu_scaling.rs
tests/lu_sparse.rs
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
