# FERAL Context (auto-generated)

Generated: 2026-05-22T18:08:54Z

## Latest Session
File: dev/sessions/2026-05-22-02.md
```
# Session 2026-05-22-02

## Goal
Investigate GitHub issue #44 — `NARX_CFy` factors ~4.4× slower per-factor
in Pounce+feral than Ipopt+MA57 (Pounce 346 s / 418 iters; Ipopt
32 s / 234 iters). Find what, if anything, is fixable on the feral side.

## Accomplished

- **Schur trailing-update kernel widened** (`5f1661c`). Widened the
  deferred-Schur SIMD kernel to a quad NEON-tile inner loop.
  Micro-benchmark ~2.2–2.5×; end-to-end on the NARX loop ~3–7% — the
  kernel is one phase of a loop that is also heavily memory-bound.

- **Phase-breakdown instrumentation** (`162b6ff`, `2e9b1e0`). Added
  `dense::factor::phase_timing` ns-counters (gated by
  `PHASE_TIMING_ENABLED`, timing-only, bit-exactness preserved) and the
  `probe_narx_phases` binary. Measured — not guessed — the warm
  `NARX_CFy_0000` numeric loop (~1128 ms, stable ±0.5 pp / 3 runs):

  | phase | % loop | what |
  |---|---:|---|
  | schur | 43.5% | SIMD trailing update (already widened, BLAS-free ceiling) |
  | extend_add | 21.1% | child→parent contrib scatter-add |
  | contrib-extract | 17.7% | frontal→contrib copy (zero-fill ~3.9%) |
  | asm-residual | 7.1% | row_map populate/clear |
  | df-residual | 4.8% | dense-factor bookkeeping |
  | scalartail | 4.6% | scalar pivot tail |
  | panelfactor/other | ~2.3% | — |

  Finding: contribution-block memory traffic (extend_add +
  contrib-extract ≈ 39%) is the #2 cost, nearly the size of the Schur
  kernel itself.

- **Amalgamation refuted.** 35 430 fronts with `ncol ≤ 4` cost 0.2% of
  the loop combined; ~950 medium fronts carry 93%. Merging tiny fronts
  amortizes nothing.

- **Contrib zero-fill investigated, not removed.** The "provably dead"
  claim was corrected: three consumers bit-compare the full `contrib`
  Vec (the `block_ldlt32` test + two parity diagnostics), so the
  zero-fill is load-bearing for the deterministic `0.0` upper triangle.
  Removing its cost needs `unsafe set_len` (first core-path unsafe);
  genuine win ~2%. Per "correctness before performance", not pursued.

- **Issue #44 closed** with a wrap-up comment documenting the measured
  breakdown (comment 4521506652). feral is correct on `NARX_CFy`; the
  residual gap vs MA57 is an acknowledged structural BLAS-free gap.

`cargo test`: 314 passed, 0 failed, 5 ignored. `cargo clippy
```

## Git Status
```
2e9b1e0 perf(probe): drill the NARX numeric loop into measured sub-phases (#44)
162b6ff probe(#44): phase-breakdown probe — Schur kernel is ~45% of the NARX loop
5f1661c perf(schur): arch-gate quad-kernel unroll to 4 on aarch64 (#44)
5dcaf7b diag(#44): add diag_narx_kernel_gflops — supernode-loop flop rate
db96569 docs: retract the ULP-nondeterminism claim; close issue #49
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-06946b9447ae7dad)

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
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-22-02.md)

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
KIRBY2_0007                    458          956          119       8.03
CRESC132_0000                 5314        87492        12266       7.13
KIRBY2_0006                    458          885          127       6.97
KIRBY2_0008                    458          839          122       6.88
KIRBY2_0009                    458          773          128       6.04
MUONSINE_0000                 1537         2051          376       5.45
KIRBY2_0010                    458          694          133       5.22
KIRBY2_0011                    458          610          120       5.08
KIRBY2_0012                    458          485          118       4.11
HAHN1_0187                     715          754          201       3.75

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.32     <= 2.0     PASS
medium (<500)            152145     1.71     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.53     <= 2.0     PASS
medium (<500)            153560     1.53     <= 3.0     PASS
All exit-partition buckets PASS. No regression vs the prior session;
the NARX kernel widening is not exercised by these corpus buckets.

```

## Recent Decisions
whether the gap is a fixable feral defect.

**What was done.** Widened the deferred-Schur trailing-update SIMD
kernel to a quad NEON-tile loop (`5f1661c`; ~2.2–2.5× micro-bench,
~3–7% end-to-end). Built a phase-breakdown probe (`probe_narx_phases`,
`dense::factor::phase_timing` ns-counters; `162b6ff`, `2e9b1e0`) and
*measured* the warm numeric loop instead of guessing: schur 43.5%,
extend_add 21.1%, contrib-extract 17.7%, assembly+bookkeeping the rest.

**Decision.** Close #44. The Schur kernel (43.5%) is already widened
and at the BLAS-free ceiling. The #2 lever — contribution-block memory
traffic, `extend_add` + `contrib-extract` ≈ 39% — would need a
packed lower-triangular contrib-block refactor or `unsafe` buffer
handling (the contrib zero-fill is *not* dead: three consumers
bit-compare the full `contrib` Vec including the upper triangle, so it
is load-bearing for determinism; removing only its wasted half is
~2% and still needs the first `unsafe` in the core numeric data path).
Per the project constraint "correctness before performance, always",
none of that is warranted for an already-correct solver. The 4.4× gap
vs MA57 — a decades-tuned Fortran solver — is acknowledged as a
structural performance gap and documented in the #44 wrap-up comment
for any future revisit.

**References.**
- GitHub issue #44 — wrap-up comment 4521506652; closed.
- `dev/journal/2026-05-22-02.org` — §15:00 phase-probe headline,
  §15:20 amalgamation refuted, §16:00 measured drill-down, §17:00
  zero-fill correction + close.
- `dev/sessions/2026-05-22-02.md` — checkpoint.
- `CHANGELOG.md` — Unreleased Performance entry.

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
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
tests/delayed_pivoting.rs

(truncated from      389 lines to 350 line budget)
