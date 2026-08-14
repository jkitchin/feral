# FERAL Context (auto-generated)

Generated: 2026-08-14T12:01:35Z

## Latest Session
File: dev/sessions/2026-08-14-01.md
```
# Session 2026-08-14-01

## Goal

Fix issue #168 — the claim recorded at `895ef65` that `peel + dense_bump_max_dim
4096` passes the downstream `bchoco06_illcond_scaled_path_recovers_bound_649`,
reported from the discopt side as not reproducing.

## Accomplished

**The issue is right. The claim does not reproduce, and I reproduced the
non-reproduction independently before touching the record.**

Harness: discopt worktree at `bce881ff` unmodified, feral worktrees at `e00aa70`
and `895ef65`, `[patch.crates-io] feral = { path = … }`, one arm per run,
`cargo test -p discopt-core --lib bchoco06`. Both feral revs are version
`0.15.1`, so the patch applies with no version override.

| feral rev | ordering | cap | result | `PROBE_DENSE_BUMP` firings |
|---|---|---|---|---|
| `e00aa70` | whole-basis AMD | 0 | **ok** | 0 |
| `e00aa70` | peel | 0 | **FAILED** — `Numerical` | 0 |
| `e00aa70` | peel | 4096 | **FAILED** — `Numerical` | **26** |
| `895ef65` | peel | 4096 | **FAILED** — `Numerical` | **26** |

Same assertion in every failure, and it is the test's ground truth rather than
its subject:

```
assertion `left == right` failed: unscaled cold solve of the bchoco06 root LP must be Optimal
  left: Numerical
 right: Optimal
```

Firing counts come from `eprintln!("PROBE_DENSE_BUMP bump_dim={}", bump_dim)`
inserted immediately after `want_dense_bump` in `sparse_factor.rs`: 26 in both
cap-4096 arms, 0 in both cap-0 arms. The failing configuration is therefore one
where the dense route provably ran, and a silently-unapplied patch is ruled out —
it would have made the cap-4096 arm identical to peel-no-cap, which it is not.
My numbers match #168's exactly, including the 26. Row 4 shows the behaviour is
identical at the commit the claim was authored on, so nothing that landed
afterwards caused it.

Record corrected in four places:

- `dev/research/lu-ordering-and-kernel-2026-08-13.md` — § "Does the fix strand
  the 1.71x?" rewritten; the answer is now **yes**.
- `src/lu/mod.rs` — `LuParams::dense_bump_max_dim` doc now states the cap does
  not recover the bound and why there is no cap-without-peel fallback.
- `CHANGELOG.md` — recorded under 0.16.0.
```

## Git Status
```
51f87ec Merge pull request #162 from jkitchin/claude/issue-161-w05b75
e00aa70 test(lu): make the density-fallback contract test depend on the contract, not the fixture
55c0ff4 docs: session checkpoint 2026-08-13-04 (issues #164 and #165)
255e2b7 docs: record what the #164 guard actually recovers
b8906a6 docs: session checkpoint 2026-08-13-03 (PR #162 review + #1008 verdict)
```

## Test Status
```
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test dense::schur_kernel::tests::schur_panel_minus_nofma_strided_quad_is_bit_exact_vs_four_singles ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 423 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.72s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-14-01.md)


Not re-run. This session changed one doc comment in `src/lu/mod.rs` and
otherwise only `dev/` and `CHANGELOG.md`; no executable code path was touched, so
the LDLᵀ bench measures the same binary behaviour as session 2026-08-13-04.
Session 04's numbers stand, including its dense-KKT p90 regression — small-frontal
1.57 → **1.61** (target ≤ 2.0, PASS) and medium 1.96 → **2.09** (target ≤ 3.0,
PASS) — which is reported here rather than omitted because it is the most recent
unfavourable comparison on record.

```

## Recent Decisions

- A dense sweep writes nonzeros into positions that are not in `pattern`, and
  `pattern` *is* the O(touched) reset list that restores `HyperWork`'s
  all-zero-between-calls contract. A nonzero left outside it silently seeds the
  next solve.
- `u_solve_sparse`'s `SingularBasis` guard is deliberately narrowed to the rows
  the solution depends on (decisions.md, earlier today). Sweeping all `m` rows
  widens it, so whether a basis is reported singular would depend on an
  unrelated right-hand side's density. Preserving the narrowing needs a per-row
  "does this position matter" test, which is the reach being abandoned.

So the fallback marks the whole accumulator (making the reset list cover the
sweep) and fills `order` with the natural topological order over `0..m`. The four
kernels are untouched — they sweep whatever is in `order` — and the fallen-back
solve is then bit-for-bit the dense entry point it is falling back to, guard
width included. That is the right semantics for a fallback: it should be
indistinguishable from the thing it falls back to, not a third behavior.

The DFS is abandoned mid-walk rather than completed and then discarded, mirroring
`ReachWork::push`'s early abort on the dense route, so an over-cap solve pays a
bounded fraction of the reach. `over_cap` is monotone within a solve (`pattern`
only grows), so the second and later kernels of a fallen-back solve cost one
`pop`, not a re-walk.

`SparseLu::sparse_rhs_fallbacks()` exists because there was no valid witness that
the guard fired: `last_sparse_solve_work()` counts factor entries traversed,
which exceeds `m` on the reach path too. Without a dedicated counter the tests
could not tell an inert guard from a working one.

Evidence: PR #162 review, findings 1 and 4; `dev/journal/2026-08-13-04.org`.

## Recent Tried-and-Rejected
**The arm is not vacuous**, which is the failure mode that would have made this
uninteresting: a counter printed immediately after `want_dense_bump` is computed
fires 26 times in both cap-4096 arms and 0 times in both cap-0 arms. The dense
route demonstrably ran in the configurations that failed.

Why the original run passed is undetermined. Vacuity is ruled out — a patch that
failed to apply would have reproduced peel-no-cap, which fails. Stale build
artifacts and an arm mix-up are the two candidates that cannot be separated after
the fact.

**Consequence.** `sparse_factor.rs` gates `want_dense_bump` on
`symbolic.triangularized`, so there is no cap-without-peel fallback: the 1.71x
and the lost dual bound are the same lever. #163's coupling argument stands, and
`895ef65` was the only evidence against it.

**Practice this changes.** The retracted measurement was recorded without any
proof that the code path under test had executed, on a route that is a *silent
fallback* — exactly the condition #162 already argued required `used_dense_bump()`
for its own tests. A pass/fail on a silent-fallback path is not evidence unless
the arm also shows the path fired. Instrument first, then measure.

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
src/lu/sparse_hyper.rs
src/lu/sparse_matrix.rs
src/lu/sparse_solve.rs
src/lu/sparse_symbolic.rs
src/lu/sparse_triangular.rs
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
tests/factors_ld_export.rs
tests/fine_grained_delay.rs
tests/fma_opt_in_roundtrip.rs
tests/golden_bits.rs
tests/growth_flag.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/issue102_intrafront_deadlock.rs
tests/issue102_ordering_escalation.rs
tests/issue107_external_ordering.rs
tests/issue112_bg_update.rs
tests/issue127_pipeline_split.rs
tests/issue128_supernode_nrow.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/issue91_preprocess_misfire.rs
tests/issue99_fma_front_gate.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/lu_adversarial_inputs.rs
tests/lu_default_ordering.rs
tests/lu_dense_bump.rs
tests/lu_dense_update_bg.rs
tests/lu_dense.rs
tests/lu_ft_widebump.rs
tests/lu_hyper_sparse.rs
tests/lu_real_bases.rs
tests/lu_scaling.rs
tests/lu_sparse_rhs.rs
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
tests/rook_rescue_kkt.rs
tests/rook_rescue.rs
tests/small_leaf_parity.rs
tests/solver_with_ordering.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/sqd_fast_path.rs
tests/static_assembly_maps.rs
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/task_plan_parity.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
