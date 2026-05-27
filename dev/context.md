# FERAL Context (auto-generated)

Generated: 2026-05-27T14:54:46Z

## Latest Session
File: dev/sessions/2026-05-26-01.md
```
# Session 2026-05-26-01

## Goal
Continue issue #54 (pounce IPM stall on `nuffield2_trap_iter1.mtx`).
Beyond the SSIDS-aligned accounting fix already shipped at 94a28bc,
localize the pivot-stability cliff under structured x-block diagonal
shifts, head-to-head against MA57, and identify a caller-side recipe
pounce can adopt.

## Accomplished
- **Commit 94a28bc (carried in from prior session):** SSIDS-aligned
  strict-zero accounting routes `|d| <= zero_tol` to the zero bucket
  rather than pos/neg. 317 lib tests pass; pre-commit clean.
- **Pivot-stability diagnosis (commit 39797ed):** instrumented
  `probe_issue54_alpha_shift.rs` with residual + Identity-scaling axes.
  Found that unpaired x-block α-shifts produce factor residuals
  1e17–1e36 across the cascade — the LDL^T itself emits garbage, not
  the inertia counter.
- **Iterative-refinement axis (commit 48ba725):** added IR and
  pivot_threshold=1e-8/1e-4 axes to the probe. Default IR does not
  rescue the unpaired-shift pathology; pivot_threshold=1e-4 makes
  matters worse (densification).
- **MA57 head-to-head (commit prior + ma57_alpha.out):** drove MA57
  on the same unpaired α-cascade. MA57 also produces 1e36 residuals
  at α=1e20 — confirming the unpaired x-block shift is pathological
  for *any* dense direct solver, not a feral defect.
- **Paired-shift sweep (commit d8e8d10):** added `shifted_paired`
  variant using δ_c = sqrt(δ_x · μ) (geometric mean). Both feral
  (default pivtol=1e-8) and MA57 cascade to machine-precision
  residuals (≤4.57e-15 by α=1e2) and lock onto the algebraically
  correct asymptote (neg=18245, pos=8404) at α≥1e2.
- **Retracted Ipopt attribution (commit d8e8d10, 14:15 journal):**
  the geometric-mean recipe was incorrectly labeled "Wächter-Biegler
  paired escalation" and "the canonical Ipopt formula" in the 13:45
  entry. ipopt-expert agent verified against IpPDPerturbationHandler.cpp:
  Ipopt's PerturbForWrongInertia escalates *only* δ_w; δ_c is set
  independently as `delta_cd_val * mu^delta_cd_exp` (defaults 1e-8 and
  0.25) by a separate code path, with no coupling to δ_w. The formula
  `δ_c = sqrt(δ_w · μ)` does not appear anywhere in Ipopt.

## Benchmark Results
```
Top 10 worst factor-ratio vs MUMPS (Phase 2.8.1):
KIRBY2_0007     458   1046µs   119µs   8.79
KIRBY2_0006     458    914µs   127µs   7.20
CRESC132_0000  5314  83698µs 12266µs   6.82
KIRBY2_0008     458    783µs   122µs   6.42
MUONSINE_0000  1537   2142µs   376µs   5.70

Dense Phase 2.8.1 exit partition:
```

## Git Status
```
7554a78 phase B(#55): symbolic-analysis-time delay budget + CB rewire
7405a13 phase A(#55): thread n_tiny through dense factor + FactorStats
f83428e phase0(#55): re-validate historical CB-on regressions at HEAD
06fdae4 session: 2026-05-26-01 checkpoint (issue #54 pivot-stability diagnosis)
d8e8d10 probe(issue54): paired delta_x + delta_c sweep + Ipopt-attribution retraction
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-30fa2f5896434378)

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
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-26-01.md)

Top 10 worst factor-ratio vs MUMPS (Phase 2.8.1):
KIRBY2_0007     458   1046µs   119µs   8.79
KIRBY2_0006     458    914µs   127µs   7.20
CRESC132_0000  5314  83698µs 12266µs   6.82
KIRBY2_0008     458    783µs   122µs   6.42
MUONSINE_0000  1537   2142µs   376µs   5.70

Dense Phase 2.8.1 exit partition:
  small-frontal (<200)  p90=1.35  target<=2.0  PASS
  medium (<500)         p90=1.70  target<=3.0  PASS

Sparse Phase 2.8.1 exit partition:
  small-frontal (<200)  p90=1.54  target<=2.0  PASS
  medium (<500)         p90=1.54  target<=3.0  PASS
All partitions PASS. No regressions vs the previous bench (this
session touched only probe binaries and notes).

```

## Recent Decisions
budget makes the trigger structural rather than numeric: CB only
fires when delay was structurally impossible, matching MUMPS's
invariant. Resolves issue #55's primary cascade-overflow failure
mode (nql180, pinene_3200) without re-introducing the inertia
regressions of issues #17 / #18 / #48.

**Convention frozen — do not change without re-running the Phase B
acceptance criteria.** Notably:
- `DELAY_CAPACITY_MULTIPLIER = 4` is the single tuning knob for
  budget tightness; lower values trade safety for tighter front
  bounds. Re-run the cascade-victim corpus before lowering.
- Root cap `min(0.05 * n, 2048)` was chosen loose; tighten only
  with corroborating telemetry.
- `cascade_break_eps = 1e-10` is the per-pivot static perturbation
  floor; the `dev/research/cascade-break-l-perturbation-2026-05-15.md`
  Weyl-bound concern is mitigated by the structural trigger but not
  eliminated. Pivots that delay could have absorbed are now absorbed
  by delay; pivots that hit CB exhausted the structural delay
  capacity.

**References.**
- `dev/research/symbolic-delay-budget-2026-05-27.md` — design,
  capacity estimate, expected impact, acceptance map.
- `dev/research/mumps-perturbation-alignment-2026-05-27.md` —
  Phase A3 audit identifying the trigger-condition gap.
- `dev/research/cb-on-default-revalidation-2026-05-27.md` — Phase 0
  evidence motivating the structural fix.
- Issue #55 — the tracked failure mode.
- MUMPS 5.8.2 `dfac_front_aux.F:1251-1331` — reference perturbation
  branch with delay-exhausted trigger.

## Recent Tried-and-Rejected
**Removing the contrib-block zero-fill — not free, not pursued.**
The 16:00 journal claim "the `resize(cdim*cdim, 0.0)` is 100%
removable, provably safe" was wrong: it checked only `extend_add` (a
lower-triangle-only reader). Grepping every reader of `.contrib` found
**three consumers that bit-compare the full contrib Vec including the
upper triangle**: the `block_ldlt32` unit test (`to_bits()` per
element), `parallel_corpus_parity.rs:70`, and `diag_par_firstdiff.rs`.
The zero-fill is what makes the upper triangle deterministically
`0.0`; deleting it naively regresses the test and breaks parity.
Removing only its cost requires `unsafe Vec::set_len` — safe Rust
cannot length a `Vec` without N initializing writes, and `src/` has no
`unsafe` in the core numeric data path. The genuinely-wasted portion
is ~2% (the lower-triangle zeros the copy overwrites anyway); the
other half is load-bearing. Decision (jrk): not worth the first
core-path `unsafe` for ~2% on an already-correct solver. Issue #44
closed.

**Lesson.** "Provably dead" requires grepping *all* consumers of the
buffer, not the one obvious algorithmic reader. Diagnostic and test
binaries that bit-compare whole buffers make "never read" false.

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
src/bin/probe_hang_loop.rs
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
```

## Test Files
```
tests/amf_corpus_oracle.rs
tests/auto_strategy.rs
tests/blocked_ldlt.rs
tests/build_row_indices_trailing_invariant.rs

(truncated from      395 lines to 350 line budget)
