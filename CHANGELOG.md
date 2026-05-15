# Changelog

All notable changes to FERAL will be documented in this file.

## [Unreleased]

### Changed — `Solver` reuses a persistent `rayon::ThreadPool` (feral#19 follow-up)

`Solver` now owns a lazily-built `rayon::ThreadPool` that is reused
across every `factor()` call dispatching the parallel multifrontal
driver. Built on first parallel-fire; persists for the `Solver`'s
lifetime. Inside `pool.install(...)` the inner driver's
`rayon::scope` / `current_thread_index` / `current_num_threads`
calls bind to this pool's workers instead of the global pool, so
the cv-wait wakeup cost that issue #19 flagged is paid once
up-front rather than per `factor()`. End-to-end measurement on M4
Pro `robot_1600` (force-parallel, 200 IPM iters): sys time dropped
28% (24.7 s → 17.9 s). No user-facing API change; the existing
`Solver::with_parallel(false)` toggle continues to skip the pool
build entirely. See `dev/sessions/2026-05-15-04.md`.

### Changed — work-aware gate in `should_parallelize_assembly` (feral#19)

`should_parallelize_assembly` (the dispatcher inside
`factorize_multifrontal_parallel_with_workspace`) now also requires
that the estimated total tree flop count clear `PAR_MIN_FLOPS = 10^8`
before firing the rayon-parallel driver. Previously the gate was
structural only (`n_snodes ≥ N_PAR_MIN` + ≥1 multi-child supernode),
which let parallel fire on small-KKT IPM control-NLP profiles
(`robot_1600`) where rayon spawn / cv-wait overhead exceeded the
parallel speedup. New public surface: `pub const PAR_MIN_FLOPS`,
`pub fn estimate_assembly_flops`, `pub fn should_parallelize_assembly_
with_threshold`, and `NumericParams::min_parallel_flops: Option<u64>`
for per-call tuning. Pounce-side env-var hook
`POUNCE_FERAL_MIN_PAR_FLOPS=<u64>` plumbs the override. See
`dev/decisions.md` (2026-05-15-03 block) and
`dev/sessions/2026-05-15-03.md`.

### Changed — `feral_solve` C ABI defaults to iterative refinement

`feral_solve` in the C ABI (`src/capi.rs`) now routes through
`Solver::solve_many_refined` against the cached factored matrix
by default. This closes the residual floor that was causing
ipopt-feral to stall in the final-tail convergence on Mittelmann
`NARX_CFy.nl` (feral#18 — ipopt-feral now Optimal at 485 iters
vs TIMEOUT before) and `robot_1600.nl` (feral#17 — ipopt-feral
now Optimal at 301 iters / 19 s vs MaxIter before). The fix is
the same in both: cascade-break perturbs the L factor enough
that the unrefined backsolve carries a ~1e-5 residual that the
IPM can't drive below the duality gap; one round of refinement
against the original matrix closes the gap. Opt out with
`FERAL_REFINE=0` in the environment. Cascade-break itself stays
enabled — it helps on the matrices it was calibrated for
(feral#8, feral#15). See `dev/decisions.md` (2026-05-15-02
block) and `dev/sessions/2026-05-15-02.md`.

### Added — Synthetic-matrix scaling benchmark vs MUMPS and MA57

New harness at `external_benchmarks/scaling/` that sweeps four
synthetic matrix families (`dense_si`, `banded_spd`, `laplace2d`,
`saddle_kkt`) across multiple sizes and compares feral against
MUMPS 5.8.2 and HSL MA57 on a derived `total_factor_us` metric
(analyse + numeric factor combined, accounting for MUMPS's bundled
`JOB=4` timing). Components: `run.py` driver (matrix generation,
RHS synthesis, solver manifests, aggregation), `plot.py` (four
PNGs per family + overview, log-log fits), `report.org` (full
write-up), and `scaling.tsv` for downstream analysis. Headline
finding: feral's numeric factor is at parity with MUMPS on every
sparse family and at parity with MA57 on `banded_spd` at the
largest `n` — but symbolic analysis is 91-96% of total wall time
on structurally-regular sparse families, making it the highest-
impact optimization target. See
`external_benchmarks/scaling/report.org` for the full discussion.

### Changed — `dev/assemble-context.sh` no longer re-runs full corpus bench

Default `./dev/assemble-context.sh` invocation now sources the
benchmark section from the latest dated session checkpoint instead
of executing `cargo run --bin bench --release` (which walks ~150k
corpus matrices and takes ~3.5 minutes). Pass `--with-bench` to
re-run for fresh numbers. Refresh time drops from ~3m30s to ~3s.
Also fixes a pre-existing glob bug where `phase-2-baseline.md` was
selected as "latest" because it sorts after `2026-...` lexically;
the glob is now `dev/sessions/[0-9]*.md`.

### Added — F2.3 iterative-refinement diagnostics

New public function `solve_sparse_refined_with_diagnostics` and
companion types `RefinementDiagnostics` / `RefinementStep`
(`src/numeric/solve.rs`). Mirrors `solve_sparse_refined` exactly
in control flow and returned iterate; additionally emits the
per-step residual / Skeel forward-error-bound trajectory plus a
shared `kappa_1_est` from the Hager-Higham 1-norm estimator. The
non-diagnostic path is unchanged — diagnostics are opt-in and add
3-5 solves (one κ̂ pass) plus an O(nnz) `||A||_1` walk only when
explicitly requested. Intended for ripopt's δ-ladder logging
(per `dev/plans/kkt-feature-gaps.md` §F2.3). Re-exported at the
crate root.

### Changed — Symbolic-arm gate on cascade-break trigger (issue #15)

The cascade-break trigger (`NumericParams::cascade_break_ratio`)
now requires `symbolic.n >= CASCADE_BREAK_MIN_N` (=4096) to fire.
Below the threshold the trigger is a guaranteed no-op regardless
of how aggressively it is configured. Cascade-break savings only
accumulate when some front can grow, via delay propagation, to
several thousand columns — and `n` is an upper bound on
achievable expanded ncol. Issue #15 reported a 14× regression on
qcqp1000-1nc (n=1154) attributed to `Some(0.5)` firing
aggressively; per-iterate data shows the trigger actually never
fires on qcqp because `n_delayed_in = 0` everywhere. The gate
makes the trigger a structural no-op on small problems while
preserving the cascade-break wins on marine_1600 / pinene_3200
(both n > 70 000). See
`dev/research/issue-15-cascade-break-symbolic-arm.md`.

## [0.3.0] - 2026-05-13

### Added — Feral C ABI for Ipopt linkage (`feral::capi`)

New `pub mod capi` (`src/capi.rs`) exposes a minimal C ABI surface
matching Ipopt's `SparseSymLinearSolverInterface` plug-in shape:
`feral_new`, `feral_free`, `feral_set_structure`, `feral_values_ptr`,
`feral_factor`, `feral_solve`, `feral_num_neg`. Matrix format is
Ipopt's `CSR_Format_0_Offset` (upper-triangle CSR, 0-based) which is
byte-identical to feral's lower-triangle CSC. Status codes mirror
Ipopt's `ESymSolverStatus` enum.

`Cargo.toml` adds `staticlib` to `crate-type` so the ABI can be
linked into the C++ Ipopt build via the `feral-ipopt-shim/` patch
(opt-in for downstream Ipopt builders; pure-Rust consumers continue
to use the `rlib`). See `dev/research/feral-ipopt-c-shim.md` and
`dev/plans/feral-ipopt-shim.md` for the design.

### Added — Ipopt 3-way NLP comparison harness

`external_benchmarks/nlp_comparison/` runs the Ipopt
ScalableProblems suite against three Ipopt 3.14.20 binaries
(`build-mumps`, `build-ma57`, `build-feral`), each linked to a
single sparse direct solver. 35 problems × 3 solvers; see
`REPORT.md` for the 2026-05-13 sweep. MUMPS 35/35 optimal, MA57
34/35, feral 34/35; geomean over triple-optimal subset: MUMPS
139 ms, feral 158 ms, MA57 162 ms. Generates `results.json` and a
Markdown report. Logs/out/RHS blobs gitignored; only the harness +
report are tracked.

### Added — MA57 oracle + 4-way cross-solver comparison

`external_benchmarks/ma57_oracle/` builds a CoinHSL MA57 benchmark
binary alongside the existing MUMPS/SSIDS oracles.
`external_benchmarks/comparison/` is extended from 3-way to 4-way
(feral + MUMPS + SSIDS + MA57), with new `run.py` / `aggregate.py`
/ `report.py` wiring MA57 into the per-matrix sample comparison and
`REPORT.md` summary.

### Added — Issue #9 Steps 2 + 3: 32×32 register-resident kernel wired into production

**Step 3 (SIMD body).** `update_1x1_block32` in
`src/dense/block_ldlt32.rs` tiles trailing destination columns in
groups of four through `schur_panel_minus_nofma_strided_quad`
(n_elim=1), with a trailing `_dual` for the 2-column tail and a final
`axpy_minus_unroll4_nofma` for the 1-column tail. Each tile packs 4
dst columns per pulp dispatch sharing one source-column load — the
intended Phase 2.4.3 register-resident pattern. Per-element output is
byte-identical to the scalar reference and to `factor::do_1x1_update`
(verified by 4 bit-parity unit tests at p=0, p=5, p=30, zero-pivot).

**Step 2 (dispatch wiring).** `do_1x1_update` and `do_2x2_update`
(factor.rs) gain an `n == 32` fast-path delegating to
`update_1x1_block32` / `update_2x2_block32`.
`factor_frontal_blocked_in_place_with_scratch` dispatches
`nrow==ncol==32` fronts to `factor_block32` (which delegates to
`factor_frontal`); the eager unblocked BK loop drives the SIMD update
via the fast-paths. This bypasses `lblt_panel_frontal` for full
32×32 fronts because, at `bs==ncol==32`, the panel's
`apply_blocked_schur_panel` quad-dispatch path is unreachable
(`j_start = k + n_elim == nrow` skips the batched trailing update),
so all trailing-update FLOPs are done by single-column peek-ahead
axpys. The eager-update path issues quad dispatches for every
trailing tile of 4 columns instead.

Bench: median small p90 1.33 (was 1.36), median medium p90 1.74
(was 1.78) across 3 runs. Modest but consistent improvement at the
better edge of the noise band. Inertia 154428/154481, byte-identical
to baseline.

**Step 4 (rank-2 SIMD body)** remains deferred — the quad kernel's
per-q sequential rounding chain is 1-ULP-divergent from
`axpy2_minus_unroll4_nofma`'s fused chain, so a custom
4-dst-column 2-src pulp dispatch is required. 2×2 pivots are rare on
the bench corpus (no measurable bench impact expected); tracked as
follow-up. **Step 5 (cross-arch CI gate)** also tracked as
follow-up.

### Changed — Per-supernode fixed-overhead reduction (#13, Phases A + B + C)

**Phase C (single-slot contrib pool).** New `pub contrib_pool:
Option<Vec<f64>>` field on `FactorScratch`. The multifrontal driver puts
the child's `ContribBlock.data` into the slot after `extend_add` consumes
it; the kernel takes at extract time, clears+resizes to `cdim*cdim`,
and writes. When the slot is empty (cold scratch, or take outpaces put),
the kernel falls back to a fresh `Vec` allocation — bit-identical to the
pre-Phase-C path. An initial multi-slot `Vec<Vec<f64>>` variant was
abandoned: it preserved bit-parity but regressed bench p90 by ~+0.19
(small) / ~+0.30 (medium) in 4 consecutive runs (growable-indirection
bookkeeping cost more than the malloc/free pairs it avoided). The
single-slot variant is bench-neutral vs Phase A+B (small p90 1.41,
medium p90 1.83–1.85) and bit-parity is preserved across all four
parity cases including the new (d) pool-hot pre-seeded case.

Phase C contributes no measurable bench movement on this corpus, but
the infrastructure is correct and ready: if a future kernel change
makes the contrib allocation a bigger fraction of factor cost, the
recycle path engages automatically. Final issue #13 standing:
criterion #1 (ns/sup reduction) MET, criterion #2 (bench p90 small <
1.30 OR medium < 1.60) **unreachable via allocation pooling on this
corpus**, criterion #3 (no correctness regression) MET, criterion #4
(bit-exact `blocked_ldlt`) MET. Per-front kernel cost (32×32 SIMD,
issue #9) is the next plausible lever for the bench-ratio gap.

### Changed — Per-supernode fixed-overhead reduction (#13, Phases A + B background)

**Phase A (`FactorScratch` pool).** New `FactorScratch { subdiag, d_panel }`
struct in `src/dense/factor.rs` pools the two internal-only working buffers
that `factor_frontal_blocked_in_place` previously allocated per supernode.
New entry point `factor_frontal_blocked_in_place_with_scratch` accepts
`&mut FactorScratch`; the existing function is now a thin wrapper that
allocates a fresh scratch and delegates. `FactorWorkspace` carries a
`factor_scratch` field that the three hot-path call sites in
`src/numeric/factorize.rs` (D.3 dense fast path, `factor_one_supernode`,
`factor_one_small_leaf`) thread through. The scratch is safe to re-warm
across different `(nrow, bs)` shapes — the kernel prologue clears and
resizes unconditionally. Bit-parity gated by
`tests/factor_scratch_parity.rs` (7-case size sweep + 6-case repeated-
calls regression) plus the 19 byte-identity `tests/blocked_ldlt.rs`
integration tests.

**Phase B (`extend_add` direct writes).** The multifrontal `extend_add`
in `src/numeric/factorize.rs` now bypasses `SymmetricMatrix::set`/`get`
and writes directly into `frontal.data` using the lower-triangle column-
major linear index. Per-cell work drops by one indirection, one branch,
and one redundant `i >= j` sanity check, with the symmetric-storage
canonicalisation preserved at the caller.

Diagnostic (`cargo run --bin diag_supernode_cost --release`): Phase A
delivered −16 % to −54 % ns/sup on the CRESC100 / ACOPR30 / HAIFAM /
KIRBY2 cluster (issue #13 acceptance criterion #1 MET). Phase B is
within run-to-run noise of Phase A on ns/sup, which is expected
because extend_add is a child-driven post-factor cost rather than
per-supernode.

Bench (`cargo run --bin bench --release`): dense small-frontal p90
1.33–1.37 and medium p90 1.75–1.78 (vs issue baseline 1.33 / 1.70).
Issue #13 acceptance criterion #2 (small p90 < 1.30 OR medium p90 <
1.60) NOT met by Phases A+B alone. 154428/154481 inertia match
preserved exactly. Phase C (return-struct pooling for `l`, `d_diag`,
`d_subdiag`, `contrib`, `perm`, `perm_inv`) is deferred to a separate
session; design choice (ABI break vs take-into vs with_capacity hints)
is unresolved.

### Added — BLAS-3 quad-column trailing-update kernel (#9, parked on #13)

`schur_panel_minus_nofma_strided_quad` in `src/dense/schur_kernel.rs`
processes four trailing columns per pulp dispatch, halving src memory
traffic vs the existing dual kernel. Wired into
`apply_blocked_schur_panel` — every front with ≥ 4 trailing columns
now routes through quad → dual → single fall-through. Bit-exact per
column with four sequential single-column dispatches (176-config
parity sweep + 19 byte-identical `blocked_ldlt` integration tests).
Zero corpus regression: dense small-frontal p90 1.33 (target ≤ 2.0
PASS), medium p90 1.70 (target ≤ 3.0 PASS); 154428/154481 inertia
match, 99.8 % residual pass.

No measurable headline-throughput win on the current corpus — the
2026-04-27 CHAINWOO_0000 root that motivated the work (1984 × 32) no
longer exists on the current build (max actual nrow = 18 after METIS-
ND on this build). The new bottleneck is per-supernode fixed
overhead, tracked as issue #13. Kernel retained as parked
infrastructure: it re-engages automatically when fronts grow tall-
skinny again. See `dev/decisions.md` 2026-05-12 (c).

### Added — block_ldlt32 scaffold and trailing-update primitives (#9)

New module `src/dense/block_ldlt32.rs` with `BLOCK_SIZE = 32`,
`factor_block32` stub (delegates to `factor_frontal` pending the
const-generic driver port), `update_1x1_block32`, `update_2x2_block32`
scalar primitives, and a bit-parity test harness diffing factors by
`to_bits()`. Signatures match the planned pulp dispatch contract; the
SIMD body swap is a surgical follow-up gated on issue #13.

## [0.2.0] - 2026-05-12

### Fixed — Honest `resolved_method` and consistent `Auto` routing (#3)

`SymbolicFactorization.resolved_method` now reflects what the symbolic
pipeline actually ran rather than what the caller requested. Two
behavior changes:

- **ScotchND silent fallback is surfaced.** When SCOTCH's nested-
  dissection recursion produces no separator for the entire graph
  (bordered-KKT shapes such as PoissonControl trigger this), the
  driver falls back to AMD via `amd_leaf` for every recursion node.
  Previously `resolved_method` still reported `ScotchND` while the
  permutation was bit-identical to AMD's. It now reports `Amd`. The
  recovery itself is unchanged — only its visibility is fixed.
- **`OrderingMethod::Auto` is consistent with the no-arg default.**
  Auto resolution now happens against the original matrix's pattern
  before any `LdltCompress` preprocessor reshapes the graph, and the
  residual branch delegates to `pick_default_method` (the rule used
  by the no-arg `symbolic_factorize`). Previously Auto could pick a
  different concrete method than the default rule on the same
  matrix, depending on whether compression triggered. Auto is now a
  strict superset: same answer as the default plus the two extra
  shape-bakeoff branches (`n>100_000` → ScotchND, `n<10_000` →
  KahipND).

Reported by independent triage on K=158 PoissonControl benchmarks.

### Fixed — `CscMatrix::from_triplets` rejects upper-triangle entries (#4)

`CscMatrix::from_triplets` and `CscMatrix::validate` now return
`FeralError::InvalidInput` when any triplet has `row < col`. Previously
upper-triangle entries were silently accepted and routed through
`sort_and_sum_duplicates`, producing a `CscMatrix` whose row indices
violated the documented "lower triangle only" invariant. Downstream
consumers (e.g. `symmetric_pattern`) assume lower-triangle storage, so
the same symmetric matrix described with upper- vs lower-triangle
triplets produced different solve results. The error message identifies
the offending triplet by index and `(row, col)`. Reported by @janosh.

### Changed

- `Solver` now defaults to the rayon-parallel multifrontal driver
  (`factorize_multifrontal_parallel_with_workspace`). The driver is
  bit-exact with the sequential supernodal path on a per-supernode
  basis and falls through to the sequential path when the supernode
  count is below `N_PAR_MIN = 32`, so small-problem latency is not
  affected. Override with `Solver::new().with_parallel(false)`.
  Closes #7. Motivation: pounce's `marine_1600` / `pinene_3200`
  Mittelmann runs were spending all their time in sequential
  `factor_one_supernode` even though the parallel driver was
  available; this wires `Solver` directly to it.

### Added

- `Solver::with_parallel(bool)` — opt out of the rayon-parallel
  driver (returns `Self` for builder chaining).
- `Solver::parallel()` — test/diagnostic accessor for the current
  flag value.
- `SymmetricMatrix::from_pooled_buf(n, buf)` constructor that zeros
  only the lower triangle when reusing a pooled buffer; cuts the
  dead upper-triangle memset out of `factor_one_supernode`'s
  per-supernode hot path. See dev/decisions.md 2026-05-12.
- `compute_infnorm_dense(&SymmetricMatrix)` in `src/scaling/infnorm.rs`
  — dense-native Knight-Ruiz iteration for the D.3/D.4 dense
  fast-path; the sparse `compute_infnorm` remains the path for the
  multifrontal driver.
- `Solver::inertia()` — `Option<&Inertia>` accessor returning the
  full inertia of the last successful factor. Complements the
  Ipopt-shaped `num_negative_eigenvalues` (which panics if no factor
  is stored) for callers that prefer to branch on `None`. Used by
  the cross-solver bench harness.

### Tooling

- Cross-solver comparison harness under
  `external_benchmarks/comparison/` (run.py, aggregate.py, report.py)
  measures feral against MUMPS 5.8.2 and HSL MA97 2.8.1 on a sampled
  SuiteSparse subset and emits `REPORT.md`. Each solver is configured
  to its production-quality (refinement-on) settings so the residual
  comparison is apples-to-apples:
  - feral driver routes through `solve_sparse_refined` (Richardson
    refinement with stagnation exit) in `src/bin/bench_one_matrix.rs`.
  - `external_benchmarks/mumps_oracle/mumps_bench.F` sets
    `ICNTL(10) = 2` (max two iterative-refinement steps) — MUMPS
    default is 0 (no refinement).
  - `external_benchmarks/hsl_bench/hsl_bench.c` wraps
    `ma97_solve_d` in a 4-step Richardson loop because MA97 has no
    native residual-based refinement entry for non-singular systems.
  Configuration is documented in the generated report's Solvers
  table.

### Performance

- Dense fast-path (`dense_fast_factor`) now runs Knight-Ruiz ∞-norm
  scaling directly on the column-major lower-triangle buffer
  produced by `to_dense_into`, removing the `row_idx[k]`
  indirection that dominated wall time on small-dense matrices.
  Routing: `ScalingStrategy::Auto` and `InfNorm` go through the
  dense KR; `Mc64Symmetric`, `Identity`, `External` are honored
  via the unchanged sparse path. Bit-exact with the sparse KR on
  every fast-path-gate matrix (`should_use_dense_fast_path` ⇒
  matrix is small enough that every column-major slot maps 1:1
  onto a CSC entry or a known-zero, and `(d_i · 0 · d_j) = 0`
  is a no-op in the max-reduction). Targets the
  `dev/results/lever-d3/stage1-stage2-2026-04-19.md` §1 finding
  that `compute_scaling` was 82% (34 of 41 µs) of dense-path
  wall time on TRO3X3_0013.
- Pooled `local_contribs` per rayon worker inside `FactorWorkspace`,
  removing a per-task `Vec<Option<ContribBlock>>` of length
  `n_snodes` from the parallel driver. Decisive on cont-201
  (sequential **–34%**, parallel-at-T=8 **–10%**); also helps
  bratu3d (**–6% / –5%**). Bit-exact. See dev/decisions.md
  2026-05-12 and dev/sessions/2026-05-12-01.md.
- Skip the upper-triangle zero on pooled frontal buffer reuse
  (`SymmetricMatrix::from_pooled_buf`). Bit-exact; 5–10% sequential
  wall reduction across mid-size matrices.

### Investigated

- Parallel multifrontal driver lock contention (T=4) — falsified.
  Added opt-in `AtomicLockStats` telemetry to `NumericParams` with
  per-task lock wait/hold counters and eight per-phase wall-time
  counters wrapping the sequential prologue/epilogue. cont-201's
  previously-reported residual headroom is **sequential symbolic
  factorize** (157 ms of a 214 ms single-shot wall), not mutex
  contention (worst-case 3.4% of body time on cont-201, 0.02% on
  c-big). On the cached-symbolic path (production / pounce-IPM
  regime) cont-201 wall drops 214 → 56 ms with body_frac jumping
  0.15× → 0.55×; remaining 1.5× headroom is inside the rayon::scope,
  not at lock sites. Full analysis in
  `dev/debugging/2026-05-12-cont201-cached-headroom.md`.

- Parallel driver within-scope localization (iteration 2) —
  rayon idle dominates. Added `task_wall_ns` (whole-closure
  bracket) and `ws_lock_wait_ns` (per-worker workspace mutex
  wait) to `AtomicLockStats`. Derived `rayon_idle = scope·T −
  task_wall_agg` quantifies the parallelism deficit attributable
  to etree dependencies. cont-201 cached: rayon_idle = 12.3 ms/T
  (78% of the gap) vs locks 1.7 ms/T (10%) and ctrl-flow 1.5
  ms/T (10%). c-big at T=4 is essentially sequential (74%
  rayon-idle capacity, 1.04× speedup vs body_agg). Conclusion:
  **assembly-tree parallelism is exhausted** on these matrices;
  within-supernode parallelism (panel-BK / threaded dense
  kernels) is the only remaining axis. Closes the cont-201
  assembly-tree investigation. See iteration 2 in
  `dev/debugging/2026-05-12-cont201-cached-headroom.md`.

- Scaling cache verification (iteration 3) — `compute_scaling_with_cache`
  works as designed. Added `solver_scaling_phase_split` test
  (`#[ignore]`) that loads the corpus and times
  `pick_scaling_strategy` + `compute_scaling_with_cache(cache=None)`
  + reorder gather. c-big picks `Mc64Symmetric` and the no-cache
  path takes **2.3 seconds** (full Hungarian); the cached path
  in production takes **2.4 ms** — 1000× speedup, cache hits.
  cont-201 and bcsstk38 pick `InfNorm`, which is values-dependent
  Knight-Ruiz iteration (~4 ms per call) and is **not cacheable**
  across IPM iterations. The 3.95 ms scaling slice on cached
  cont-201 is fundamental per-factor work, not a missed cache.
  Closes scaling probe from session 2026-05-12-02 "Next session
  should #2".

- Issue #5 (MSS1 BK inertia non-monotone under δ_w·I): triage
  complete, closed on the feral side. Landed a reproducer test
  + zero_tol/pivot_threshold sweep diagnostics in
  `src/numeric/factorize.rs::tests`. Empirically demonstrated
  that no in-kernel magnitude-floor lever cures the wandering;
  cross-checked MUMPS 5.8.2 and MA57 (via Ipopt's wrapper) and
  confirmed neither implements eigenvalue-aware 2×2 splitting.
  Recommended fix is upstream (caller-side δ_c bump matching
  Ipopt's `PerturbForSingularity`). Full analysis in
  `dev/research/issue-5-mss1-inertia-monotonicity.md` §9.

## [0.1.0] - 2026-05-06

First public release on crates.io. Seven crates published in
dependency order:

- [`feral-ordering-core`](https://crates.io/crates/feral-ordering-core)
  0.1.0 — locked contract surface (`CscPattern`, `OrderingStats`,
  `OrderingError`, `CONTRACT_VERSION`).
- [`feral-amd`](https://crates.io/crates/feral-amd) 0.1.0 — AMD
  fill-reducing ordering (Amestoy, Davis & Duff 1996, 2004).
- [`feral-amf`](https://crates.io/crates/feral-amf) 0.1.0 —
  HAMF4 / Approximate Minimum Fill (Amestoy 1999).
- [`feral-metis`](https://crates.io/crates/feral-metis) 0.1.0 —
  multilevel nested-dissection (Karypis & Kumar 1998).
- [`feral-scotch`](https://crates.io/crates/feral-scotch) 0.1.0 —
  SCOTCH-style nested-dissection (Pellegrini 1996).
- [`feral-kahip`](https://crates.io/crates/feral-kahip) 0.1.0 —
  KaHIP flow-based nested-dissection (Sanders & Schulz 2011; data
  reduction per Ost, Schulz & Strash 2021).
- [`feral`](https://crates.io/crates/feral) 0.1.0 — sparse symmetric
  indefinite direct solver with certified inertia counts.

The accumulated `Unreleased` work below is the complete content of
the 0.1.0 release; future entries will land under `## [Unreleased]`
above.

### Changed (2026-05-03) — `SupernodeParams::default().nemin` 32 → 16

`SupernodeParams::default().nemin` (src/symbolic/supernode.rs:115)
drops from 32 to 16. `nemin` is the minimum supernode size below
which the symbolic phase merges parent and child nodes during
amalgamation. A larger `nemin` yields fewer, fatter supernodes (more
BLAS-3 work per node, more pass-through row padding); a smaller
`nemin` yields more, thinner supernodes (tighter L storage, more
pivot-block boundaries).

The previous `nemin = 32` was inherited from an early dense-kernel
study and out of step with reference solvers — MUMPS uses 5,
SSIDS's canonical configuration sits in the same low band. The new
`nemin = 16` lands halfway, validated by a sweep over
{8, 16, 32, 64} on PoissonControl K=50 and K=158 (both AMD and
METIS-ND): nemin=16 is the sweet spot for `factor_nnz` and factor
wall — K=50 `factor_nnz` -26%, factor wall -18%; K=158 `factor_nnz`
-21%, factor wall ≈ par. The corpus bench retains its dense P90
ratio targets (small-frontal ≤ 2.0 PASS, medium ≤ 3.0 PASS).

The slack recovered comes from the pass-through row padding
diagnosed in `dev/research/factor-nnz-residual-gap.md`: smaller
supernodes have less inflation from rows that flow through ancestors
not pivoting on those rows.

References: `dev/research/factor-nnz-residual-gap.md`,
`dev/decisions.md` 2026-05-03 nemin entry.

### Fixed (2026-05-03) — `build_row_indices` upper-triangle pollution

`build_row_indices` (src/numeric/factorize.rs:2257-2298) now filters
trailing-row candidates with `r < first_col + own_ncol`. The
function iterates `full_pattern.col_ptr[j]` for j in the supernode's
own column range; `full_pattern` is the fully-symmetrized A pattern,
so each column carries both lower-tri (r > j) and upper-tri (r < j)
entries. Upper-tri rows correspond to columns already eliminated by
ancestors of those rows in the etree — including them as trailing
rows polluted every supernode's frontal, propagated up through child
contrib blocks, and inflated `factor_nnz` by 7-19× over the textbook
L-fill (Σ col_counts).

Performance only: numeric assembly writes nothing into the rogue
rows (they receive zeros), so they were dead weight in BK pivoting,
not a source of wrong answers. Inertia is bit-identical across the
test corpus before and after the fix.

PoissonControl K=50 factor_nnz dropped from 1,363,445 to 323,643
(4.2×) and factor time from 231,075 µs to 3,542 µs (65×). K=158
factor_nnz dropped from 46,734,661 to 4,610,269 (10×) and factor
time from seconds to 85,099 µs. All 216 lib + integration tests
pass.

A `cfg(debug_assertions)` invariant assertion at
src/numeric/factorize.rs:1469-1485 verifies the trailing-row floor
on every supernode in debug builds. New regression test file
`tests/build_row_indices_trailing_invariant.rs` (8 tests) covers
four multifrontal-path fixtures (n > N_TINY=16).

References: `dev/research/build-row-indices-fix.md`,
`dev/decisions.md` 2026-05-03 entry.

### Changed (2026-05-02) — `NumericParams::default()` `pivot_threshold = 1e-8`

`NumericParams::default()` now sets `bk.pivot_threshold = 1e-8`,
matching MA27's `cntl[1]` reference default (Ipopt's
`ma27_pivtol`). Previously the default inherited
`BunchKaufmanParams::default()`'s `0.0`, which silently disabled
the column-relative pivot rejection, rook rescue, and delayed-
pivoting rescue paths — producing exact-zero multiplier outputs on
rank-deficient KKT-augmented LS-init systems (issue #2).

`BunchKaufmanParams::default()` (the dense entry point) is
unchanged at `0.0` per the 2026-04-13 dense-vs-sparse split.
Callers that explicitly construct `BunchKaufmanParams` and pass it
to `NumericParams::with_bk` are unaffected; in-tree sparse callers
that already opt in to `0.01` (benches, parity tests) keep their
override.

The `Solver::increase_quality` cascade still includes the W5
"0.0 → 0.01" first-jump rule for callers that explicitly disable
the threshold; from `Solver::new()` the cascade now goes
1e-8 → 1e-6 → 10^-4.5 → ... → `pivtol_max = 0.5`.

References: `dev/research/issue-2-kkt-pivot-default.md`,
`dev/plans/issue-2-kkt-pivot-default.md`, issue #2.

### Added (2026-04-28) — `bench_solver_corpus` realistic-IPM perf bench

New `src/bin/bench_solver_corpus.rs` walks `data/matrices/kkt/`,
groups by family (`<FAM>_NNNN.mtx` → `FAM`), and per family runs two
scenarios — one persistent `Solver` (caches `SymbolicFactorization`
across same-pattern re-factorizations) vs the free-function loop
that re-runs symbolic on every call. Reports per-family table and
corpus-wide aggregate (geomean / p10 / p50 / p90 speedup, symbolic
call counts, implied symbolic share of freefn wall).

Initial run on 534 families × 19,410 iterates: **aggregate 1.70×,
geomean 2.86×, p50 3.00×, p90 4.08×**. Symbolic share of freefn
wall is 41.3% — the realistic figure for IPM-tail workloads, vs the
artificial 64% reported by the per-matrix `bench` which recomputes
symbolic on every of 154k matrices.

Going forward `bench_solver_corpus` is the perf-tuning ground truth
for symbolic-phase optimizations (decisions.md 2026-04-28). The
per-matrix `bench` is retained for inertia/residual correctness
sweeps and oracle ratio comparisons.

Also added: `src/bin/profile_hot.rs` (samply target for sampling
profiles of the hot pipeline) and `[profile.release] debug = true`
in `Cargo.toml` so samply runs symbolicate cleanly.

### Changed (2026-04-27) — Dense kernel B-1 dual-column DSYRK (NR=2)

Phase B-1 of `dev/plans/dense-kernel-blas3.md` (narrow first step).
Replaces the per-column rank-`n_elim` SIMD axpy in
`apply_blocked_schur_panel` with a dual-column kernel that processes
two adjacent trailing columns per dispatch and shares src loads
between both accumulator stacks — halving src memory traffic on the
bulk of the trailing update.

- New `schur_panel_minus_nofma_strided_dual` kernel: 4-way unrolled
  SIMD body holding 8 accumulator registers (4 dst0 + 4 dst1).
  Cap (column-j diagonal) processed by scalar prologue; bulk
  (dst0[1..] and dst1) shares one src load per chunk per q.
- `apply_blocked_schur_panel` walks trailing columns in pairs;
  odd-tail column falls back to the single-column kernel.
- Bit-exact with two sequential `schur_panel_minus_nofma_strided`
  calls (verified by 144-case n_elim × len0 sweep + zero-alpha
  independence test).
- All 16 `tests/blocked_ldlt.rs` byte-identity fixtures continue
  to pass — per-supernode byte identity preserved.

Bench: all four exit-partition gates still PASS; small/medium
frontals see ~+2-3% session-13 baseline regression (within noise);
wide trailing updates (qcqp1500-1c root) get the src-load-sharing
win. Phase B-2 (lift all-1×1 gate for 2×2 streams) and Phase C
(cache-blocked dense root) remain on the plan.

### Changed (2026-04-27) — Dense kernel W-2 2×2 inline (no-swap fast path)

Phase A of `dev/plans/dense-kernel-blas3.md`. The blocked panel
factorization (`lblt_panel_frontal`) now handles no-swap 2×2 pivots
inline instead of bailing to the scalar path on every 2×2 trigger.
ACOPR-style KKT panels with heavy 2×2 pivot counts (43.8% of factor
time on `qcqp1500-1c` is in `ncol=17–32` panels, 28.6% in
`ncol=33–64`) stay on the deferred-Schur fast path.

- New `PanelStatus::ScalarFallbackPeekedNext` variant signals when
  the inline 2×2 path pre-updated col+1 before bailing on
  swap/LAPACK/growth/det-floor — caller adjusts `j_start` to avoid
  a double rank-1 update.
- New `peek_ahead_replay(target_col)` primitive supports replay onto
  arbitrary trailing columns.
- Rank-bs fast path stays gated on all-1×1 panels for bit-exactness
  (axpy2's fused add-then-sub differs from the rank-bs SIMD body's
  per-q sequential mul-sub). Lifting this gate is Phase B-2.
- Bit-exactness contract: per-supernode byte-identity between
  `factor_frontal` (scalar) and `factor_frontal_blocked` is
  preserved on all 16 fixtures including 4 new W-2 2×2 fixtures.

Symmetric-swap 2×2 and rook-rescue 2×2 still bail to scalar; phase B
will address those.

### Added (2026-04-27) — F3.4 `SchurBlock::solve` convenience

Closes the F3 phase plan. Two new methods on `SchurBlock`:

- `SchurBlock::symv(x, y)` — symmetric mat-vec `y = S · x` against
  the dense full-square buffer.
- `SchurBlock::solve(rhs)` and `solve_with(rhs, params)` — factor
  `S` with the existing dense Bunch-Kaufman LDL^T solver and run a
  single solve. The convenience wraps `dense::factor::factor` +
  `dense::solve::solve`; for repeated solves with the same `S`,
  callers should drive the underlying primitives directly to
  amortise the factor cost.

Tests cover (a) hand-computed 3×3 explicit Schur block,
(b) end-to-end `factorize_multifrontal_with_schur` →
`SchurBlock::symv` → `SchurBlock::solve` round-trip on a 4×4 KKT,
(c) dimension-mismatch error path.

This is the last item in the F3 (Schur complement extraction)
phase plan from `dev/plans/kkt-feature-gaps.md`. With F3.1
(ordering hook), F3.2/2b (numeric hook + multi-supernode tail),
F3.3 (per-matrix oracle gate, 250/250 PASS), and F3.4 all
landed, F3 is closed.

### Changed (2026-04-27) — F3.3 acceptance gate switched to per-matrix oracle bound

The strict `feral-vs-MUMPS ≤ 1e-10` reading was unachievable on
ill-conditioned ACOPR-family KKTs because MUMPS itself disagrees
with a pure-Rust dense oracle by ~1e-6 on those matrices
(conditioning floor, not bug). The gate is replaced with a
per-matrix conditioning-adapted bound:

  feral-vs-oracle ≤ max(1e-10, K · MUMPS-vs-oracle)   K = 10

with corpus floor ≥ 100 matrices. New tooling:

- `src/bin/produce_dense_schur` generates `<id>.dense_schur.bin`
  oracle sidecars (partial-pivot GE on `[A_FF | A_FS]`, column-
  major, same layout as MUMPS sidecar).
- `src/bin/diag_schur_parity` now loads both MUMPS and oracle
  sidecars, reports three pairwise distributions and the per-
  matrix Option B verdict.

Result on the 250-matrix corpus subset with sidecars: 250/250
PASS. Worst feral-vs-oracle = 9.841e-7 (ACOPP30_0000); worst
MUMPS-vs-oracle = 1.037e-6 (ACOPR14_0002) — i.e. feral hits the
same conditioning floor as MUMPS, with feral marginally closer
to ground truth on the worst case. Spec text updated in
`dev/research/schur-complement.md`.

### Fixed (2026-04-27) — F3.3 forest-Schur-etree postorder bug

`schur_constrained_postorder` in `src/ordering/postorder.rs`
emitted Schur nodes via DFS over the Schur subtree, which
violates the `post[k] == k` tail identity that
`symbolic_factorize_with_schur` relies on whenever the Schur
etree is a forest (multiple Schur roots) rather than a single
ascending chain. ACOPP30_0000 hit this with 11 Schur roots plus
an internal chain — the resulting permutation reassigned the
diagonal at col 174, producing max_rel_err = 0.997 vs the dense
Schur oracle.

Phase 2 now emits Schur nodes directly in ascending etree-index
order, restoring the tail identity for arbitrary Schur-etree
topology. Regression test
`test_schur_postorder_forest_tail_identity` covers the minimal
forest case (n=8, three Schur roots + internal Schur child).

Corpus impact (`diag_schur_parity` on 250 matrices with MUMPS
Schur sidecars):

- Factor success: 68/250 → 250/250 (no factor failures).
- Median max_rel_err vs MUMPS: 6.437e-13.
- p90: 3.323e-9; p99: 2.059e-7; max: 1.033e-6 (ACOPR14_0002).
- 136/250 within 1e-10 tolerance; 114/250 above (ACOPR14 family,
  conditioning-driven BK pivot divergence — the dense oracle on
  ACOPP30_0000 shows the same 9.8e-7 disagreement with MUMPS).

### Added (2026-04-27) — F2.2 MUMPS RINFOG cross-validation harness

The MUMPS oracle (`external_benchmarks/mumps_oracle/`) now runs
with `ICNTL(11)=1` and emits `RINFOG(4..11)` (matrix infinity-
norm, solution norm, scaled residual, omegas, forward-error
bound, COND1, COND2). The canonical `.mumps.json` sidecar gains a
`conditioning` block. `src/bin/diag_cond_parity` walks the corpus
and reports `kappa_feral / RINFOG(11)` ratios.

Note: RINFOG(10)/(11) are componentwise infinity-norm condition
numbers (Arioli-Demmel-Duff, dsol_aux.F:935), not
`||A||_1 * ||A^-1||_1`. The harness is therefore a directional
cross-check; F2.1's Hilbert/KKT calibration remains the binding
numerical gate for `estimate_condition_1norm`. Corpus geomean
ratio over 165,959 matrices is 4.244e10 — ten orders of magnitude
offset, which led to the F2.2 acceptance-gate reframe in
`dev/decisions.md`.

### Changed (2026-04-27) — AMF default ordering for `n <= 10_000`

`pick_default_method` in `src/symbolic/mod.rs` now mirrors MUMPS's
`ana_set_ordering.F` AMF-vs-METIS rule: `Amf` for `n <= 10_000`,
`MetisND` for `n > 10_000`, with the existing bordered-KKT
(`n >= 5000 && nnz/n < 6`) and chain-pattern (`n >= 2000 && nnz/n
< 4`) escape hatches preserved as MetisND. Replaces the previous
"AMD by default" rule. AMD remains available via
`OrderingMethod::Amd` for diagnostic comparison.

Validated on the 183_293-sidecar Phase C corpus
(`tests/amf_corpus_oracle.rs::amf_corpus_gate`): **183_277 ok / 16
skipped / 0 fail** at the 1.10x feral-amf-vs-MUMPS-HAMF4 nnz_L
gate. Bench Phase 2.8.1 exit partition post-flip:

```
Dense small-frontal (<200)   147982   p90=1.25   target<=2.0  PASS
Dense medium       (<500)    152145   p90=1.72   target<=3.0  PASS
Sparse small-frontal (<200)  153455   p90=1.56   target<=2.0  PASS
Sparse medium       (<500)   153560   p90=1.56   target<=3.0  PASS
```

Sparse buckets moved from p90=1.52 (session 09 baseline, AMD
default) to 1.56 (+0.04). Dense buckets unchanged. Worst-ratio
tail improved dramatically: pre-flip top was NASH_0111 at 22.75x;
post-flip top is KIRBY2_0007 at 6.20x.

ORBIT2_0000 nnz_L stays at 32_105 (the AMF-clean-room headline
win — AMD's 1.4M-nnz_L factor on this kkt-expansion shape was
the original motivation).

### Docs (2026-04-27) — AMF clean-room research note + plan

`dev/research/amf-clean-room.md` and `dev/plans/amf-clean-room.md`
land the foundation for a clean-room Approximate Minimum Fill
(AMF / HAMF4) ordering as a peer of `feral-amd`. Motivation: MUMPS
picks HAMF4 as the default for SYM=2 N≤10000 (`ana_set_ordering.F:
52-78`), and we have empirical confirmation on at least one
bipartite-KKT family (ORBIT2: feral-amd 5.1M nnz_L vs MUMPS HAMF4
110k — 47×). Research note covers the Amestoy 1999 thesis fill
metric, the six inner-loop sites that change vs AMD, and the
`Metric`-trait architecture (extract shared quotient-graph
machinery; `feral-amd` and `feral-amf` become thin specializations).
Plan note breaks the work into four phases (module factoring with
bit-parity gate, AMF metric impl + fixtures, MUMPS HAMF4 oracle
plumbing, wire-up + corpus validation), 3-5 sessions total. No
code changes — pure documentation.

### Performance (2026-04-27) — rank-bs trailing-update accumulator (W-2, 1×1)

`src/dense/factor.rs::apply_blocked_schur` rewritten as a single
`pulp::WithSimd` body (`schur_panel_minus_nofma_strided`) iterating
`for j in j_start..nrow` outermost, accumulating all `n_elim`
contributions in register accumulators, then storing — replacing the
previous `O(n_elim * trailing)` SIMD dispatch pattern of `n_elim`
rank-1 axpys. Bit-exact against the rank-1 reference (no FMA, explicit
mul/sub ordering) — verified for `n_elim ∈ {1,2,4,7,8,16,31,32}` ×
`len ∈ {1,3,7,8,9,15,16,17,31,32,33,63,64,65,256,257}`. 2×2 panels
stay on the rank-1 fallback for now. Combined with W-1: CHAINWOO
driver-level total 28.7 → 4.4 ms (**6.5× speedup**). W-2 from
`dev/plans/dense-kernel-speedup.md`.

### Performance (2026-04-27) — engage blocked panel for ncol >= 8 (W-1)

`src/dense/factor.rs::factor_frontal_blocked_in_place` now dispatches
the deferred-Schur panel kernel whenever `ncol >= 8` (was `ncol > bs`
with default `bs=64`, sending every 32×32 CHAINWOO supernode to the
scalar path). Bit-parity preserved against the scalar reference for
the new fixtures `ncol ∈ {8, 12, 16, 24, 32}`. W-1 from
`dev/plans/dense-kernel-speedup.md`. Combined effect with W-2 above.

### Ordering (2026-04-27) — quasi-dense column quotient (Fix A, opt-in only)

`crates/feral-metis/src/lib.rs::metis_order_full` gained an opt-in
`MetisOptions::dense_quotient_enabled` flag (default **false**). When
enabled, columns with off-degree > `max(40, ceil(10*sqrt(n)))` are
pulled out of the ND graph, the M1–M7 pipeline runs on the
sparse-induced subgraph, and the dense columns are appended at the
end of the returned permutation. The technique is the user-guide
reading of MUMPS `ICNTL(6)` and HSL_MC68's "dense parameter"; expert
review of MUMPS and SPRAL sources (2026-04-27) found that neither
solver actually pre-strips the graph — MUMPS handles dense rows
*inside* QAMD via the `THRESM` parameter (`ana_orderings.F:5226+`)
and SSIDS doesn't special-case them at all. Empirical test on
ORBIT2_0000: enabling Fix A *increased* `nnz_L` from 1.54M to 2.25M
because removing the dense column destroys the structural signal that
makes it the natural top separator. The opt-in path is kept for
diagnostic experimentation (`src/bin/diag_orbit2_quotient.rs`); the
correct fix is a QAMD-style deferral inside `feral-amd`, deferred to a
future session. See `dev/research/orbit2-cluster-regression.md` §10
for the post-mortem.

### Tooling (2026-04-27) — bench `FERAL_KKT_FILTER` filters at load time

`src/bin/bench.rs::load_kkt_dir` now reads `FERAL_KKT_FILTER` and skips
non-matching `.mtx` stems before parsing the `.json` sidecar and
oracle-timing sidecars. Targeted single-family runs that previously
waited for the 156k-matrix corpus to load now complete in <1s
(`FERAL_KKT_ROOTS=kkt-expansion FERAL_KKT_FILTER="CHAINWOO_0000"`:
0.18s vs >60s post-load). The post-load filter is preserved as a
redundant safety net.

### Performance (2026-04-27) — move contrib into ContribBlock (W-3b)

The multifrontal driver previously cloned `ff.contrib` on every
supernode to produce a `ContribBlock` for the parent. For
CHAINWOO_0000's 1984-row root that clone is ~30 MB per factorization.
Replaced with `std::mem::take(&mut ff.contrib)` in
`factor_supernode_dense_internal` and `factor_supernode_leaf_dense`;
the saved `NodeFactors.frontal_factors.contrib` is empty afterward
(production solve paths never read it). CHAINWOO driver-level total:
~23,246 → ~22,620 us. W-3b from `dev/plans/dense-kernel-speedup.md`.

### Performance (2026-04-27) — factor frontals in place (W-3a)

Added `factor_frontal_blocked_in_place(&mut SymmetricMatrix, ...)` in
`src/dense/factor.rs` that factors directly into the caller's buffer,
eliminating the `nrow*nrow` scratch allocation + lower-triangle copy
the kernel had been performing on every supernode call. For
CHAINWOO_0000's 1984-row root supernode that is ~30 MB per call. The
multifrontal driver (`factor_kkt_dense_path`,
`factor_supernode_dense_internal`, `factor_supernode_leaf_dense`)
migrated to the in-place entry; `factor_frontal_blocked` stays as a
wrapper for tests/examples/bit-parity reference. Bit-parity preserved
(`tests/blocked_ldlt.rs` 9/9). W-3a from
`dev/plans/dense-kernel-speedup.md`.

### Fixed (2026-04-27) — 2x2 BK pivot inertia uses trace, not a00

`src/dense/factor.rs::count_2x2_inertia` decided eigenvalue signs from
`sign(a00)` alone. KKT 2x2 blocks routinely have `a00 = 0` (variable
rows have zero Hessian diagonal), so sign-of-a00 mis-attributed those
blocks. Switched to `trace = a00 + a11` per Sylvester's law -- matches
`count_2x2_inertia_val` (sparse path), rmumps's `compute_inertia`, and
canonical Fortran MUMPS. Threaded `a11` through `do_2x2_pivot`.
All 162 lib tests pass.

### Fixed (2026-04-27) — bench prefers MUMPS oracle inertia

The bench's "inertia mismatch vs MUMPS" headline previously compared
feral's factorization against the IPOPT iter-0 `.json` sidecar's
`inertia` field, which is *IPOPT's expected inertia at iteration entry*
-- NOT what MUMPS factually computed on the dumped matrix. After
regenerating 3,377 `.mumps.json` oracle files to record MUMPS-computed
inertia (11,748 sidecars total now have inertia), the bench prefers
the MUMPS oracle when present and falls back to the IPOPT sidecar
otherwise. Sparse "inertia mismatches" 837 -> 52 (~16x improvement).
Added `inertia: Option<(usize, usize, usize)>` to `OracleTiming`.

### Added (2026-04-27) — `pick_default_method` extends to MetisND for sparse low-degree large-n

`src/symbolic/mod.rs::pick_default_method` now picks `MetisND` when
`(n>=5000 && avg_deg<6) || (n>=2000 && avg_deg<4)`. This fixes the
CHAINWOO/HYDROELL/DIXMAANH 1000x factor-time regression by routing
chain-like KKT structures away from AMD (which produces 5x more fill
on these structures). Memory peak on CHAINWOO sequence: 1173 MB ->
145 MB (8.1x). Phase 2.8.1 sparse-medium p90 ratio vs MUMPS: 39.26x
-> 1.61x.

### Added (2026-04-26) — Streaming bench + `FERAL_SPARSE_MAX` cap

- `cargo run --bin bench --release` now streams matrix data through
  each loop iteration instead of loading the entire corpus upfront.
  `KktEntry` shrinks to metadata only (`name`, `mtx_path`, `sidecar`,
  oracle timings). Each loop body re-reads the `.mtx`, runs work,
  drops at end of scope. Peak RSS on `FERAL_KKT_ROOTS=all` drops from
  30+ GB (load-all design, SIGKILLed) to ~17 GB end of dense / ~36 GB
  end of sparse on the 64 GB dev laptop.
- New `FERAL_SPARSE_MAX=N` env var caps the sparse multifrontal loop
  by sidecar dimension (`n + m`). Default `usize::MAX` (no cap).
  Set to `20000` (or similar) to bound the sparse pass on the
  expanded corpus where the n>50000 tail blows the memory ceiling
  per-matrix. Skipped matrices counted as `Size-skipped (n > N)`.
- New `Parse-skipped` summary line counts matrices skipped during
  per-iteration parse (NaN/Inf entries, dim mismatch, unparseable
  `.mtx`); pass-rate denominators subtract these to preserve old
  semantics.
- First end-to-end expanded-corpus validation: 170,176 matrices,
  sparse 167,380 attempted (99.5% inertia, 99.8% residual,
  worst 2.99e8 same outlier as kkt-only POLAK6_0021).

### Fixed (2026-04-26) — `factor_nnz()` accounting matches SSIDS

`SparseFactors::factor_nnz()` and the `summary().nnz_l` accumulator
now report the SSIDS-equivalent count (lower triangle inc diagonal of
each eliminated supernode block + trailing rect rows) instead of the
full dense `nrow * nelim` block. The previous count swept in the
strict-upper triangle of the eliminated block, which is structurally
zero, and was 1.75× larger than SSIDS's `inform%num_factor` at the
median. After the fix bench reports `nnzL/SSIDS p50 = 1.00`,
geomean = 1.09 across the kkt corpus (was 1.75 / 1.79). New audit
binary at `src/bin/diag_factor_nnz_accounting.rs`. MUMPS `INFOG(9)`
ratio drops from 1.00 to 0.62 because `INFOG(9)` includes
delayed-pivot/pre-allocation overhead that feral's actual L-fill
does not.

### Changed (2026-04-25) — Phase 2.13a `AmalgamationStrategy::Auto` is now default

**Default `AmalgamationStrategy` flipped from `Renumber` to `Auto`.**
Auto is a cheap O(n) etree-shape dispatcher: path / near-path
elimination trees go to `Adjacency`, bushy trees go to `Renumber`.
Eliminates the Phase 2.12 MUONSINE regression (5.5× → 1.4× MUMPS)
while preserving the IPM-KKT tail wins from `Renumber`. Corpus
Top-10 max ratio improves 10.64 → 9.66; p99 3.45 → 3.40. To pin a
specific strategy: `SupernodeParams { amalgamation_strategy:
AmalgamationStrategy::{Adjacency,Renumber}, .. }`. See
`dev/decisions.md` Phase 2.13a entry.

### Added (2026-04-25) — Phase 2.13a etree-shape dispatch

- `AmalgamationStrategy::Auto` variant — new `#[default]`. Resolved
  to `Adjacency` or `Renumber` via `pick_amalgamation_strategy`
  before `find_supernodes`.
- `feral::symbolic::pick_amalgamation_strategy(&etree)` — public
  resolver, also exported.
- `feral::symbolic::AUTO_MULTI_CHILD_FRAC_THRESHOLD` — public const
  (0.05) for downstream consumers wanting the same predicate.
- `src/bin/diag_etree_shape.rs` — predicate-design probe over the
  7 known-answer matrices.
- `tests/auto_strategy.rs` — 7 dispatch unit tests (path, complete
  binary tree, empty, leaf-only forest, near-path, fan-at-root).

### Changed (2026-04-25) — Phase 2.12 SSIDS column renumbering is now default

**Default `AmalgamationStrategy` flipped from `Adjacency` to
`Renumber`.** Cuts factor time 30-67% on IPM-KKT tail matrices
(ACOPR30/CRESC100/LAKES/NELSON/SWOPF) at the cost of ~10% on the
corpus median for small CUTEst-Hessian matrices. Net win for feral's
spec-stated mission (IPM/KKT solves). See `dev/decisions.md` Phase
2.12 entries for the workload-weighted reasoning. To revert per-call:
`SupernodeParams { amalgamation_strategy: AmalgamationStrategy::Adjacency, .. }`.

### Added (2026-04-25) — Phase 2.12 SSIDS column renumbering machinery

- `SupernodeParams::amalgamation_strategy: AmalgamationStrategy` field —
  new opt-in enum. Default `Adjacency` (matches every prior release);
  `Renumber` runs an SSIDS-style merge-prediction + biased-postorder
  pass that places desired-merge children adjacent to their parents
  before `find_supernodes`.
- `src/symbolic/supernode.rs::predict_merges` — symbolic merge oracle
  matching the SSIDS trivial-chain + size rule.
- `src/ordering/postorder.rs::biased_postorder` — emits non-merging
  children first, merging children last; merging child sits adjacent
  to its parent.
- Reverse iteration in `find_supernodes` Step 2 under Renumber: parent
  `first_col` shrinks monotonically, opening adjacency for the next
  lower-index sibling.
- `tests/column_renumbering.rs` — 4 structural tests (arrow / bushy
  fan collapse to 1 supernode; tridiagonal stays at 1; perm bijection).
- `tests/column_renumbering_parity.rs` — 3 numeric parity tests
  (inertia + refined residual match across strategies on arrow SPD,
  bordered KKT, ACOPR30_0067).
- `src/bin/diag_strategy_compare.rs` — 5-run-median Phase-2.10
  profiler comparison.

Default stays `Adjacency`. Renumber cuts factor time 60-67% on
ACOPR30/CRESC100 tail and 2-3× supernode count on every tiny-IPM tail
matrix, but adds ~10% to the corpus median sparse factor ratio vs
MUMPS — outside the ±5% graduation budget. Decision recorded in
`dev/decisions.md` (Phase 2.12 entry).

### Added (2026-04-21) — Phase 2.6.5 LDLᵀ-aware ordering (opt-in)

- `src/symbolic/ldlt_compress.rs` — port of MUMPS `ICNTL(12) = 2`
  (Duff-Pralet symmetric matching + quotient-graph compression).
  Public API: `SuperMap`, `build_supermap`, `compress_pattern`,
  `expand_permutation`.
- `SupernodeParams::preprocess: OrderingPreprocess` field — new
  opt-in enum. Default `None`; `LdltCompress` runs MC64 matching,
  contracts each matched pair into one super-variable, orders the
  compressed graph, and expands the super-permutation before handing
  it to the rest of the symbolic pipeline.
- `src/bin/diag_compression_bench.rs` — corpus bench. On the
  worst-10 high-leverage matrices + stride-5000 sample (40 total):
  factor-time geomean `cmp/base = 0.604` (40% faster), 0 inertia
  mismatches. MUONSINE / VESUVIO / HAHN1 / GAUSS2 family factor
  47-60% faster with compression on.
- `tests/ldlt_compress.rs` — 3 integration tests; 9 unit tests in
  the module.

Default stays `OrderingPreprocess::None`. Flip is deferred to a
follow-up session with a larger-scale residual-parity-tracking
bench (plan: `dev/plans/phase-2.6.5-ldlt-compressed-graph.md`).

### Added (2026-04-21) — Phase 2.5.2 parallel multifrontal driver (live)

- `factor_one_supernode` helper — shared per-supernode body for
  sequential and parallel drivers.
- `factorize_multifrontal_supernodal_parallel` — rayon `scope` +
  `AtomicUsize` pending-children task-graph driver. Bit-exact parity
  with the sequential driver (0 / 38 878 KKT-corpus mismatches in
  `diag_par_repeat`). Dispatcher
  `factorize_multifrontal_parallel[_with_workspace]` routes to this
  driver when `should_parallelize_assembly` returns true.
- `should_parallelize_assembly` predicate — true when
  `n_snodes >= N_PAR_MIN (32)` and at least one supernode has ≥ 2
  children.
- `src/bin/parallel_corpus_parity.rs` — full 169 585-matrix audit.
- `src/bin/diag_acopr.rs` / `diag_par_repeat` / `diag_par_firstdiff`
  / `diag_par_frontal_hash` / `diag_parent_unique` — diagnostic
  binaries retained for future debugging.
- `tests/parallel_parity.rs` — 6 KKT parity tests (all pass).

### Fixed (2026-04-21) — Parallel driver seeding race

- Seed loop in `factorize_multifrontal_supernodal_parallel` used a
  dynamic `pending[i].load() == 0` predicate inside `rayon::scope`.
  Workers running spawned leaves decrement parents' counters
  concurrently with the seed loop; a non-leaf whose final child
  completed mid-seed could be spawned twice (once by the caller,
  once by the last child's `fetch_sub==1` trampoline). Replaced
  with a static "no children" filter captured before the scope.
  Root cause found via `FERAL_HASH_FRONTAL` instrumentation on
  ACOPR14_0003.

### Added (2026-04-20) — Phase 2.4.1b scaffolding (RED)

- `BunchKaufmanParams::block_size: usize` (default `64`). Consulted
  only by the forthcoming Phase 2.4.1b blocked-panel path; the
  scalar `factor_frontal` ignores it.
- `pub fn factor_frontal_blocked(..)` — stub that currently returns
  `FeralError::InvalidInput("…Phase 2.4.1b not yet implemented")`.
  Will be implemented in Step 4 (faer-style peek-ahead panel) and
  must produce byte-identical `(L, D, perm, inertia, contrib)` to
  `factor_frontal`, enforced by the six tests in
  `tests/blocked_ldlt.rs` (currently RED by design).

No runtime behavior change: all existing call sites use
`factor_frontal`; the scalar pivot body was extracted into an
internal `scalar_pivot_step` helper, verified byte-identical by the
118/118 lib tests and the 31 integration tests that exercise
`factor_frontal` (`delayed_pivoting`, `dense_ldlt`,
`pivot_rejection`, `dense_fast_path`).

### Changed (2026-04-20) — Bench harness multi-sample denoise

`cargo run --release --bin bench` now resamples per-matrix
factor+solve timings K=5 cold reps for any matrix whose MUMPS oracle
sidecar reports `factor_us < 200 µs`. Reduction: `min` across reps
for factor, `median` for solve. No library behavior change — this
is a test-harness change only. Inertia and residual validation still
run once on the first factor.

**Motivation.** Single-shot wall time at tens-of-µs produced 10–100×
noise excursions that dominated the top-N worst-ratio report (HS85
80×, CERI651BLS 57×, PALMER2ANE 202× — all diagnosed as noise in
session 2026-04-20-01).

**Effect.** Three-run sparse max across `cargo run --bin bench`:

| | pre-denoise | post-denoise |
|-|-----------:|-------------:|
| max spread over 3 runs | 11.81 / 102.07 / 285.80 (24×) | 13.38 / 11.36 / 27.09 (2.4×) |
| sparse p90 | 1.77 | 1.65 |
| sparse p99 | 3.76 | 3.52 |

**Cost.** Bench wall-time 2:15 → 4:00 (+78%), bench runs once per
session.

See `dev/decisions.md` and `dev/results/bench-denoise/summary.md`.

### Added (2026-04-20) — D.4 tiny-n disjunct in dense fast-path gate

`should_use_dense_fast_path` now accepts any matrix with
`n ≤ N_TINY = 16` regardless of density, in addition to the
existing D.3 density-gated disjunct. The `dense_fast_factor`
implementation is unchanged — only the gate predicate is broadened.
Motivated by the finding that at tiny `n` the multifrontal path is
dominated by symbolic-phase overhead, not floating-point work;
D.4 lets tiny matrices skip symbolic entirely.

**Per-call evidence:** six observed top-10 tiny-n rows (HS73,
PALMER1E, HATFLDH, PALMER1A, KIRBY2LS, HEART6LS) show 1.17–1.53×
p50 speedup over forced multifrontal and beat MUMPS by 2–4×.
**Corpus evidence:** sparse factor/MUMPS geomean stable at
0.38–0.39 across three runs (pre-D.4 reference 0.37, within noise).
D.4's unique class (`n ≤ 16 ∧ ρ < 0.25`) appears empty on the
current IPM corpus — observable corpus impact is small, but the
gate is now a correct primitive for any such matrices that arise.

See `dev/decisions.md` (2026-04-20 D.4 entry) and
`dev/plans/sparse-tail-d4.md`.

### Added (2026-04-19) — D.3 dense fast-path for small-dense matrices

`factorize_multifrontal_with_workspace` now routes matrices with
`n ≤ 128` and lower-triangle density `≥ 0.25` to a new
`dense_fast_factor` entry point that skips symbolic analysis +
supernodal assembly entirely: densifies to a `SymmetricMatrix`,
applies global symmetric scaling in place, runs the dense BK kernel
on all `n` columns, and wraps the result in a single-supernode
`SparseFactors` shape-compatible with `solve_sparse`. Out-of-gate
matrices follow a bit-identical multifrontal path — no regression.

Two new public API entry points —
`factorize_multifrontal_supernodal` and
`factorize_multifrontal_supernodal_with_workspace` — provide
explicit gate-bypass for tests and callers that need to force the
multifrontal body on an in-gate matrix.

**Corpus:** sparse factor/MUMPS geomean 0.46 → 0.37 (−20 %),
max ratio 128.34 → 80.22 (−37 %). Ex-ante target was ≤ 0.44,
exceeded with 0.07 margin. Phase 2.8.1 exit partitions still PASS.

See `dev/decisions.md` (2026-04-19 D.3 entry) and
`dev/plans/sparse-tail-d3.md`.

### Added (2026-04-19) — `FactorWorkspace` caller-owned scratch pool

New `FactorWorkspace` struct pools scratch allocations across
`factorize_multifrontal_with_workspace` calls and across supernodes
within a call: `row_map`, the per-supernode frontal matrix
`data` Vec, and the `build_row_indices` scratch buffers.
`Solver::factor` retains one workspace so IPM-style consumers
amortise heap traffic across iterations.

`factorize_multifrontal` still exists and allocates a fresh
workspace per call — callers that don't need amortisation see no
API change.

**Corpus:** sparse factor/MUMPS geomean 0.48 → 0.46. VESUVIO
reallocs collapsed 99 % (2053 → 13). See `dev/decisions.md`
(2026-04-19 D.1 entry) and `dev/plans/factor-workspace.md`.

### Changed (2026-04-19) — `ScalingStrategy::Auto` is the default; Policy 4 fallback

`ScalingStrategy::default()` now returns `Auto` (was `InfNorm`).
`Auto` routes matrices with the arrow-KKT signature
(`diag_only / n >= 0.30`) to `Mc64Symmetric` and everything
else to `InfNorm`. The trade landed an 8× tail compression on
the IPM corpus (worst factor/MUMPS 83× → 10×) at a net change
of -9 in residual_pass count out of 154 588 matrices; 14 of
the 21 regressions are oracle-`numerically_intractable` and 5
are tolerance-edge (residuals 1e-10 → 1e-9).

`Auto` includes a Policy 4 post-scaling diagnostic that falls
back to `InfNorm` when MC64 has demonstrably misfired on a
matrix where InfNorm would do fine. Three conditions must all
fire:

1. `raw_diag_range(matrix) < 1e6` — raw matrix already
   well-conditioned (so MC64 has nothing to recover).
2. `mc_off > 1e6` — MC64's scaled `max(|off|/|diag|)` is
   large.
3. `mc_off / in_off > 1e5` — and is much larger than InfNorm's.

This recovers the MSS1_0009 residual regression (6e-12 →
1e-6 → 6e-12) without sacrificing the VESUVIO/CRESC factor
wins or the MEYER3NE parity tests. Final corpus residual_pass
post-Policy 4: 154 233 / 154 588.

`InfNorm` and `Mc64Symmetric` remain available as opt-in.
Caller code that explicitly passed `&ScalingStrategy::InfNorm`
sees no change.

See `dev/decisions.md` (2026-04-19 entries) and
`dev/research/policy-4-scaling-fallback.md`.

### Added (2026-04-19) — POUNCE integration interface (`Solver`)

A stateful `Solver` handle that mirrors Ipopt's `SymLinearSolver`
contract: factor → check inertia → escalate quality → re-factor.
Built on the just-shipped β refactor so the cached
`SymbolicFactorization` is reusable across every quality level
without invalidation.

Public API additions:
- `Solver`, `FactorStatus`, `QualityLevel` (re-exported from crate
  root).
- `Solver::new() / with_params / factor / solve / solve_refined /
  increase_quality / num_negative_eigenvalues / provides_inertia /
  factors / quality_level / pivot_threshold / scaling_strategy /
  symbolic_call_count`.
- `FactorStatus::{Success, Singular, WrongInertia { actual, expected
  }, FatalError}` — `WrongInertia` does NOT clear the stored factor,
  matching Ipopt `SYMSOLVER_WRONG_INERTIA` semantics.
- `QualityLevel::{Baseline, ScalingEnabled, PivotRaised, Exhausted}`.
- `FeralError::NoFactor` for `solve()` / `solve_refined()` called
  before a successful factor.

Two-stage quality escalation (MA27-style defaults: `pivtol_max =
0.5`, `exponent = 0.75`):
- Stage 1: flip `Identity` scaling to `InfNorm` (skipped if scaling
  is already non-Identity).
- Stage 2: pivot threshold from 0.0 jumps to 0.01, then geometric
  `min(pivtol_max, t^0.75)` until cap; transitions to `Exhausted`.

Cache reuse: `Solver::factor` caches the `SymbolicFactorization` and
re-runs `symbolic_factorize` only when the matrix's
`(n, col_ptr.len(), row_idx.len())` fingerprint changes. The IPM
caller pattern (refactor on the same pattern with new values) pays
the symbolic cost exactly once.

Test coverage: `tests/pounce_interface.rs` (11 integration tests
including the canonical IPM loop pattern) plus 5 unit tests
`U1`-`U5` in `src/numeric/solver.rs` for the escalation state
machine. All 208 tests pass.

### Refactor (2026-04-19) — scaling moved from symbolic to numeric phase (β)

`SymbolicFactorization` is now purely structural and cacheable across
IPM iterations of structurally identical KKTs. MC64/InfNorm scaling
has moved from `symbolic_factorize` to `factorize_multifrontal`.

Public API changes:
- `SupernodeParams` no longer has a `scaling_strategy` field.
- `SymbolicFactorization` no longer carries `scaling`,
  `scaling_pivot_order`, or `scaling_info`.
- New `NumericParams { bk: BunchKaufmanParams, scaling: ScalingStrategy }`
  bundle is the per-numeric input to `factorize_multifrontal`.
- New `NumericParams::with_bk(bk)` constructor for default scaling.
- New `SparseFactors.scaling` and `SparseFactors.scaling_info` fields.

Migration: callsites that previously passed a `BunchKaufmanParams` to
`factorize_multifrontal` now pass `NumericParams::with_bk(bk)`. Sites
that wired per-iteration scaling via `SupernodeParams::scaling_strategy`
now build `NumericParams { bk, scaling }` and pass it on the numeric
side; the symbolic call uses default `SupernodeParams`.

Why: the IPM driver re-factorizes the same sparsity pattern many times
with new values. Tying scaling to the symbolic phase forced a
re-symbolic pass per numeric refactor, defeating cache reuse. New
structural test
`factorize_multifrontal_with_two_strategies_on_one_symbolic` proves
one `SymbolicFactorization` can drive two `ScalingStrategy` values.

### Documentation (2026-04-18) — KaHIP driver-integration decision pinned

A planning pass against the deferred priority "KaHIP K1 data
reduction integration into driver" (sessions 04, 06, 07) ran the
41-matrix shape bake-off with `KahipND` included and decided
**not** to add a dispatcher rule for KaHIP.

Bake-off evidence (`bench_orderings`):

| ordering | fill / AMD geomean | total symbolic time |
|----------|--------------------|---------------------|
| AMD      | 1.000              | 14.4 s              |
| METIS    | 1.024              | 68.2 s              |
| SCOTCH   | 1.038              | 15.3 s              |
| KaHIP    | 1.023              | 81.1 s              |
| Auto     | 0.988              | 14.7 s (mostly AMD) |

KaHIP-with-K1 ties METIS on fill but at 4-6× the per-call setup
cost, and never strictly beats METIS on any of the 41 matrices.
On the 154 588-matrix IPM bench it would only match METIS where
the existing `n>=5000 && nnz/n<6 → MetisND` rule already fires.

Changes:
- `OrderingMethod::KahipND` docstring documents why the dispatcher
  does not select it and where it is reachable.
- New unit test `pick_default_method_never_returns_kahip` pins the
  decision so a future opt-in change must be conscious.
- Research note: `dev/research/ordering-kahip-driver-integration.md`.
- Plan: `dev/plans/ordering-kahip-driver-integration.md`.

No code paths or default behavior change. KaHIP remains reachable
via `symbolic_factorize_with_method(.., KahipND)` and via
`OrderingMethod::Auto`.

### Changed (2026-04-18) — sparse solve: workspace reuse across refinement steps

`solve_sparse` now drives a private `solve_sparse_core_into` that
takes caller-owned scratch (`y_buf` length `n`, `w_buf` length
`max_nrow`). `solve_sparse_refined` builds one `SolveWorkspace`
per call and reuses it across the (up to) 11 internal solves, plus
reuses `r`, `dx`, and `best_x` buffers across iterations and updates
`x` in place rather than constructing `x_new` per step.

Behavior is unchanged — same residuals, same convergence, same best-
iterate semantics. The win is purely allocator pressure on tiny KKT
matrices where per-call `vec![0.0; n]` calls dominated the solve
phase. Microbench (10 000 iters, ns/call) before → after the refined
refactor (which sits on top of the earlier `solve_sparse_core`
workspace fix):

| matrix     |   n |  refined before | refined after |
|------------|-----|-----------------|---------------|
| ALLINITC   |   7 |          2226ns |        1845ns |
| MCONCON    |  48 |          2961ns |        2746ns |
| AVION2     |  94 |         10654ns |        9251ns |
| BATCH      | 121 |         67275ns |       55253ns |
| HAHN1      | 715 |        410527ns |      336387ns |

Corpus impact (154 588 IPM matrices): solve/SSIDS geomean 1.30 →
1.15. Residual pass and worst residual unchanged.

### Changed (2026-04-18) — `symbolic_factorize` default: bordered-KKT fallback to MetisND

`symbolic_factorize` (the default entry point) now applies a narrow
heuristic before dispatching:

  - `n >= 5000 && stored_nnz/n < 6` → `MetisND`
  - everything else                  → `Amd`

The rule catches bordered-KKT structures like CUTEst CRESC132 where
AMD orders the constraint block into a near-dense root frontal that
swallows ~96% of n and drives a ~5000-column delay cascade.

Diagnosis: `cresc_diag` showed CRESC132_0000 with AMD produced a
5084×4854 root frontal (96% of n=5314) and 4846 cumulative delays;
the same matrix with MetisND produced a 2661×2200 root and 1333
delays. Factor time 5.4 s → 480 ms (11× win on that one matrix).
Symbolic max-frontal-width does NOT predict actual width because
delays are a numeric artifact, so a posteriori check on symbolic
alone has no signal — see `dev/journal/2026-04-18-06.org`.

`OrderingMethod::Auto` was rejected previously (bench regression
from per-call KaHIP overhead on n<500 matrices). The new rule is
narrow enough that no IPM-corpus family with n<5000 is affected.

Bench evidence (154,588 IPM matrices):

| metric                  | AMD-only | heuristic |
|-------------------------|----------|-----------|
| factor/MUMPS geomean    | 0.45     | 0.42      |
| factor/MUMPS max ratio  | 521      | 85        |
| solve/SSIDS geomean     | 1.33     | 1.30      |
| residual pass           | 154241   | 154241    |
| worst residual          | 2.69e-4  | 2.69e-4   |

Callers wanting literal AMD with no dispatcher should call
`symbolic_factorize_with_method(matrix, params, OrderingMethod::Amd)`
explicitly. `src/bin/bench.rs` honors that escape hatch via
`FERAL_ORDERING=amd`; with the env var unset, the bench routes
through the new heuristic.

### Changed (2026-04-18) — sparse refinement: 2-strike plateau exit

`solve_sparse_refined` now exits after two consecutive non-improving
steps instead of running all 10 iterations to the unreachable
`ε·√n` relative target. `profile_sparse` showed the prior loop was
adding 11.5× overhead on the bare solve because the relative target
sits below double-precision floor noise on most KKT matrices.

The 2-strike rule preserves bouncing-into-basin behavior on
borderline matrices (a 1-strike rule killed it). Bench evidence
(154,588 IPM matrices):

| metric           | before | after |
|------------------|--------|-------|
| residual pass    | 154327 | 154241 |
| worst residual   | 2.69e-4| 2.69e-4|
| solve/SSIDS      | 1.82   | 1.33  |
| factor/MUMPS     | 0.44   | 0.45  |

86 marginal matrices regress at the residual-pass threshold (still
99.8% pass), worst residual exactly matches the prior bench, and
solve time drops 27% against SSIDS.

Also: `lib.rs` re-exports `solve_sparse` (was internal). New
`src/bin/profile_sparse.rs` per-stage profiler. Full per-stage
numbers and the bug history (a misordered `improved` check that
silently capped iterations to 1) in
`dev/journal/2026-04-18-06.org`.

### Notes (2026-04-18) — `OrderingMethod::Auto` is opt-in only

Closed-loop end-to-end bench (154,588 IPM KKT matrices) showed `Auto`
regresses sparse factor/MUMPS geomean from 0.44 (AMD) to 0.58. Root
cause: thousands of small (n<500) iteration dumps get routed to KaHIP
where K1 + multilevel setup costs 2-3× per call vs AMD. The 0.988
fill geomean from the 41-matrix shape bakeoff is real but does not
amortize on workloads dominated by tiny matrices.

Resolution: `symbolic_factorize` keeps the `Amd` default; `Auto`
remains available via `symbolic_factorize_with_method` for callers
whose workloads are dominated by large or `cresc132`-class matrices.
The doc comment on `OrderingMethod::Auto` warns callers. Full
evidence in `dev/tried-and-rejected.md`.

The `FERAL_ORDERING={amd,auto,metis,scotch,kahip}` env var added to
`src/bin/bench.rs` stays as harness for future ordering experiments.

### Added (2026-04-18) — `OrderingMethod::Auto` adaptive ordering dispatcher

- `src/symbolic/mod.rs`: new `OrderingMethod::Auto` variant. Picks a
  concrete ordering per-matrix from cheap `CscPattern` features:
  - `n > 100_000 && nnz/n < 5`  → `ScotchND` (large-sparse / arrow).
  - `n <  10_000 && nnz/n < 15` → `KahipND` (K1 reductions dominate).
  - otherwise                   → `Amd`.
  Internal helper `choose_adaptive` resolves `Auto` to a concrete
  method at dispatch time; non-`Auto` inputs pass through unchanged.
- `src/bin/bench_orderings.rs`: adds an Auto column to the per-matrix
  table and the summary.
- Unit test `choose_adaptive_rules` covers the four branches;
  `symbolic_factorize_auto_produces_valid_perm` verifies the
  end-to-end dispatch produces a valid permutation.
- Bakeoff (41 matrices):
  - geomean fill: AMD 1.000, METIS 1.024, SCOTCH 1.038, KaHIP 1.023,
    **Auto 0.988** (best on average, beats every fixed method).
  - min-fill wins: AMD 37, METIS 31, SCOTCH 28, KaHIP 37, **Auto 41**
    (Auto never strictly loses).
  - total symbolic time: AMD 15.64s, METIS 71.8s, SCOTCH 16.1s,
    KaHIP 83.7s, **Auto 15.37s** (fastest — dispatches SCOTCH on
    c-big, which is both faster and lower fill than AMD there).

### Changed (2026-04-18) — `feral-kahip` K1 wired into driver; Rule-1-only preset

- `crates/feral-kahip/src/node_nd.rs`: `kahip_nd_order` now runs K1
  data reduction as a pre-pass (via `reduce_graph`) and expands the
  reduced-graph permutation back to original indices via
  `expand_permutation`. The inner nested-dissection pipeline is
  factored into `kahip_nd_inner`.
- `crates/feral-kahip/src/data_reduction.rs`: added `ReduceOptions`
  struct with per-rule toggles (`degree2_simplicial`,
  `degree2_nonsimplicial`, `twins`, `subset`). `::conservative()`
  enables only Rule 1 (degree-1 cascading); `::full()` enables all
  rules. The driver uses `::conservative()`; unit tests use
  `::full()` so all four rules remain covered.
- Fixed a Rule-2 expansion bug: path interiors were anchored only to
  endpoint `u`, but fill-preservation requires them to be eliminated
  before BOTH endpoints. When `pos(w) < pos(u)` in the reduced perm,
  the old expansion produced extra fill through the still-alive path.
  Fix: at expansion time, anchor the path to whichever of the two
  endpoints' ultimate (path-compressed) anchors has the lower
  reduced-perm position. This fix alone improved geomean fill from
  2.094 to 1.876 but did not recover three regressions (vesuvio /
  vesuviou / cresc132) that were 40-50× AMD.
- Rules 2-4 remain implemented and unit-tested but are disabled in
  the driver. Empirically they cause 40-50× fill regressions on the
  bench corpus; root cause is unresolved. See
  `dev/tried-and-rejected.md` for details.
- Bakeoff over the full parity + large corpus (41 matrices):
  - geomean fill: AMD 1.000, METIS 1.024, SCOTCH 1.038, **KaHIP 1.023**
    (was 1.032 pre-K1; KaHIP is now the best on average)
  - min-fill wins: AMD 37, METIS 31, SCOTCH 28, **KaHIP 37** (tied
    with AMD, up from 30)
  - total symbolic time (us): AMD 15.1M, METIS 71.4M, SCOTCH 16.0M,
    KaHIP 84.0M — KaHIP time dropped from 147.6M to 84.0M because
    Rule-1 cascading shrinks the graph fed to the flow refinement.
  - `c-big` (n=345241) KaHIP fill 3.29× → 2.59× (improved but still
    not competitive with SCOTCH's 1.00×; adaptive dispatch or further
    tuning are open follow-ups).

### Added (2026-04-18) — `OrderingMethod::KahipND` solver-side dispatch

- `src/symbolic/mod.rs`: added `OrderingMethod::KahipND` variant;
  `run_external_ordering` dispatches to `feral_kahip::kahip_order`.
  Test `symbolic_factorize_kahip_produces_valid_perm` mirrors the
  existing METIS/SCOTCH perm-bijection checks on the 5×5 grid.
- `src/bin/bench_orderings.rs`: extended the 4-way bakeoff
  (AMD / METIS / SCOTCH / KaHIP), including per-row fill and time
  columns plus a KaHIP win-count / geomean / total-time summary.
- `Cargo.toml`: added `feral-kahip` as a workspace path dep.
- Bakeoff over the full parity + large corpus (41 matrices):
  - geomean fill: AMD 1.000, METIS 1.024, SCOTCH 1.038, KaHIP 1.032
  - min-fill wins: AMD 40, METIS 32, SCOTCH 28, KaHIP 30 (ties count
    for all at min)
  - total symbolic time (us): AMD 14.8M, METIS 77.9M, SCOTCH 16.6M,
    KaHIP 147.6M — KaHIP is the slowest (flow-based refinement at
    every level carries ~10× the per-ordering overhead of AMD/SCOTCH).
  - Notable: `c-big` (n=345241) KaHIP fill is 3.29× AMD — worse than
    METIS 2.69× and SCOTCH 1.00× (tied with AMD). Data point for the
    adaptive dispatcher follow-up.

### Added (2026-04-18) — `feral-kahip` phases K5+K6 (multilevel controller + ND driver)

- New module `crates/feral-kahip/src/cycle.rs` implementing K5
  multilevel edge bisection: reuses `feral_metis::internals`
  (coarsen, fm_refine, initial_partition, rng) for the multilevel
  plumbing, swaps METIS's FM-only refinement for a FM bootstrap +
  K3 flow refinement at each uncoarsening level. Mode tuning
  (Fast/Eco/Strong) controls `n_sep_trials`, `coarsen_floor`,
  `amd_switch`, `fm_pass_cap`, `bnd_distance`, and how many K3
  iterations run at each level.
- `graph_to_undirected` bridge from `feral_metis::internals::Graph`
  (i32-indexed CSR) to `UndirectedGraph` (usize-indexed CSR) so K3
  and K4 can consume subgraphs produced by the multilevel pipeline.
- New module `crates/feral-kahip/src/node_nd.rs` implementing K6
  recursive nested-dissection driver: connected-components walk,
  per-component recursion, AMD leaf fallback
  (`feral_amd::amd_order`) for subgraphs ≤ `amd_switch`, K5
  bisection + K4 `flow_node_separator` lift + separator-last
  numbering for larger subgraphs.
- `kahip_order_full` wired end-to-end; returns contract-conforming
  `(perm, OrderingStats, KahipStats)`. Status updated from
  "pre-implementation scaffold" to "K2-K6 complete".
- `feral-kahip/Cargo.toml`: added `feral-amd` and `feral-metis` path
  deps (same pattern as `feral-scotch`).
- 61/61 feral-kahip tests pass (+12 new: 5 K5, 7 K6). Coverage:
  trivial 10-vertex graph, determinism, balance within slack,
  Fast/Eco/Strong on 12x12 grid, graph bridge preservation,
  diagonal pattern, 10x10 grid → AMD leaf path, 16x16 grid →
  multilevel path, disconnected components, empty graph. Clippy
  clean under `-D warnings`.
- Research note `dev/research/ordering-kahip-k5-k6.md` with the
  combined K5/K6 architecture, mode-parameter mapping, and out-of-
  scope items (full V/F-cycle re-coarsening, K1 integration,
  `OrderingMethod::KahipND` solver dispatch).

### Added (2026-04-18) — `feral-kahip` phase K3 (flow-based edge refinement)

- New shared module `crates/feral-kahip/src/graph.rs`:
  `UndirectedGraph` CSR type (n, xadj, adjncy, eweight) with
  `cut_weight`, `neighbors`, `eweights`, and `from_csc_unit_weights`.
  Infrastructure shared by K3/K4/K5/K6.
- New module `crates/feral-kahip/src/flow_refine.rs` (internal to
  the crate until K5/K6 consume it) implementing one iteration of
  flow-based bisection refinement per Sanders-Schulz 2011 §4:
  - Boundary detection, BFS band extraction with configurable
    `bnd_distance` (plan audit item 12).
  - Undirected edges modeled as anti-parallel directed pairs with
    the full edge weight as capacity on each direction (audit
    item 10).
  - Fixed-node pinning at `pin_depth = min(max_dist_in_part,
    bnd_distance)` per side — pins all band vertices at that
    depth to super-source (part 0) or super-sink (part 1) with
    INF_CAP = `(sum_band_edge_weight / 2) + 1` (audit item 2;
    fallback covers small parts inside the BFS ball).
  - Two-cut most-balanced-min-cut v1: solve max-flow normally +
    reversed; pick the candidate with lower cut weight satisfying
    the balance tolerance. Full MBMC (residual-flow manipulation)
    deferred to K5/K6 (audit item 3).
  - Strict improvement acceptance only.
- 40/40 tests pass (`cargo test -p feral-kahip`); clippy clean.
  Coverage: empty/degenerate inputs, pre-optimal path-midpoint
  cut, suboptimal 7x7 diagonal improvement (cut 12 → 8 with
  bnd_distance=2, ε=0.4), determinism across repeated calls,
  balance-constraint rejection, non-worsening on a random 40-node
  graph, fixed-node pinning invariant on a path graph.
- Research note `dev/research/ordering-kahip-k3.md` with the
  formal algorithm, band/fixed-node definitions, two-cut MBMC v1
  scope, and the 8-item test-oracle construction.

### Added (2026-04-18) — `feral-kahip` phase K2 (push-relabel max-flow)

- Implemented push-relabel max-flow / min-cut in
  `crates/feral-kahip/src/flow.rs` (internal to the crate until phase
  K3 consumes it):
  - Goldberg-Tarjan 1988 preflow algorithm with highest-label active-
    vertex selection (buckets indexed by height, FIFO within a bucket).
  - Gap relabeling per Cherkassky-Goldberg 1995, required by the
    K3 band refinement budget. Gap detection is gated on
    `0 < g < n` (a gap at height 0 would falsely disconnect the
    sink); lifted vertices with residual excess are re-inserted into
    `bucket[n+1]` so stranded flow drains back to source via reverse
    edges.
  - Deterministic tie-breaking (lowest-index admissible neighbor,
    FIFO within same-height bucket) satisfying audit item 16 of
    `dev/plans/ordering-kahip.md`.
  - Min-cut extraction via residual BFS from source.
  - Rejects `MalformedInput` on `source == sink`, out-of-bounds
    endpoints, or negative capacities. Self-loops are ignored.
    Parallel edges are preserved (residual capacity stacks correctly).
- Crate-public surface unchanged: `kahip_order` / `kahip_order_full`
  still return `OrderingError::Internal`. No `OrderingMethod::KahipND`
  yet — dispatch lands with K6.
- 29/29 tests pass (`cargo test -p feral-kahip`); clippy clean.
  Coverage includes malformed-input rejection, unit-capacity path,
  parallel edges, self-loop ignore, diamond bottleneck, CLRS 3e
  Figure 26.1 (max-flow = 23), k×k grid horizontal cut (f = k for
  k ∈ {2, 3, 4, 5}), K_{3,3} bipartite matching (f = 3), cut-
  saturation invariant on a random 30-node graph (disconnected case,
  f = 0) and a hand-laid connected 6-node network (f = 10),
  disconnected-sink zero-flow, and determinism across repeated runs.
- Research note `dev/research/ordering-kahip-k2.md` with the formal
  algorithm, gap-relabeling proof sketch, data-structure layout, and
  the full test-oracle construction.

### Added (2026-04-18) — `feral-kahip` phase K1 (data reduction)

- Implemented Ost-Schulz-Strash 2021 data reduction rules in
  `crates/feral-kahip/src/data_reduction.rs` (internal to the crate
  until the K2–K6 pipeline lands):
  - Degree-1 elimination with cascading and order-preserving op stack.
  - Degree-2 path compression handling both simplicial (endpoints
    adjacent — zero fill) and non-simplicial (one fill edge added)
    sub-cases. Skips pure-cycle chains with a per-pass `skip` array
    so subsequent seeds find other chains.
  - Open and closed twin detection using canonical sorted
    signatures; closed twins (common in KKT diagonal blocks) are
    processed before open twins.
  - Subset elimination (mark-array) as a conservative capstone rule.
  - Path-compressed anchor union-find for permutation expansion.
- Crate-public surface is unchanged: `kahip_order` and
  `kahip_order_full` still return `OrderingError::Internal` because
  the full K1–K6 pipeline is not yet wired. `OrderingMethod::KahipND`
  is not introduced; dispatch wiring lands with phase K6 per
  `dev/plans/ordering-kahip.md`.
- 15/15 tests pass (`cargo test -p feral-kahip`); clippy clean.
  Coverage includes bijection, CSC invariants, cascading, closed
  twins on K4, open-twin-via-degree-2 on K_{2,3}, and a Rule 2
  firing test between two distinct hubs.
- Research note: `dev/research/ordering-kahip-k1.md`.

### Changed (2026-04-18) — `OrderingMethod::Amd` now routes through `feral-amd`

- Default AMD is now the full Amestoy/Davis/Duff AMD in the `feral-amd`
  workspace crate (approximate external degree + aggressive element
  absorption + supervariable detection), replacing the simplified
  exact-external-degree implementation at `src/ordering/amd.rs` in the
  dispatch path.
- Fill and time improvement on the large-matrix corpus: fill 17-23%
  lower on `c-big`, `cont-201`, `bratu3d`; time 18-88× faster.
  Parity-corpus fill is a statistical tie (geomean 1.001).
- `src/ordering/amd.rs` remains on disk as a reference implementation
  and still exports `permute_pattern`. See `dev/decisions.md`
  (2026-04-18 entry) and `dev/journal/2026-04-18-03.org`.
- Parity panel regenerated via `select_parity_panel`: 17 pass + 9
  ignored (was 27 + 1). The additional ignores are rank-deficient
  KKT matrices that now fall on the zero/tiny-signed pivot
  classification boundary; residual quality is preserved (all
  feral residuals ≤ ~1e-8, matching or beating MUMPS).

### Added (2026-04-18) — OrderingMethod enum dispatch wires METIS and SCOTCH into symbolic factorization

- `feral::symbolic::OrderingMethod::{Amd, MetisND, ScotchND}` (default
  `Amd`) selects which fill-reducing ordering
  `symbolic_factorize_with_method` uses.
- `symbolic_factorize` is preserved as a thin delegate that passes
  `OrderingMethod::Amd`, so existing callers are unchanged.
- Cross-crate adapter converts the main crate's owned-usize
  `CscPattern` to the ordering-contract's borrowed-i32 view
  (`i32::try_from` overflow-checks the matrix size) and maps
  `OrderingError → FeralError::InvalidInput` with perm validation
  (length, non-negative, bounded).
- `Cargo.toml` now depends on `feral-metis` and `feral-scotch`
  directly (previously only transitively through
  `feral-ordering-core`).
- The in-tree `src/ordering/amd.rs` is retained as the `Amd`
  implementation pending separate retirement work per
  `dev/decisions.md`.

### Added (2026-04-18) — Comparative ordering bake-off binary and corpora

- New binary `cargo run --release --bin bench_orderings` runs
  `symbolic_factorize_with_method` three times per matrix (AMD /
  METIS / SCOTCH) and reports per-matrix fill + symbolic time
  plus geomean ratios and win counts. Walks `tests/data/parity/`
  (one representative per family) and `tests/data/large/` (flat
  layout) when present.
- Large-matrix corpus: pinned SuiteSparse manifest in
  `dev/scripts/large_matrices.txt` + fetch script
  `dev/scripts/fetch_large_matrices.sh`; four matrices spanning
  n=8k–345k covering symmetric indefinite and KKT regimes.
  `tests/data/large/` gitignored.
- Results and analysis: `dev/research/ordering-bakeoff-2026-04-18.md`.

### Added (2026-04-18) — Adversarial A1-A10 regression tests for FM refinement

- 9 new tests in `crates/feral-metis/src/fm_refine.rs` cover the
  edge cases enumerated in `dev/research/metis-fm-sign-bug.md` §5:
  paths, cycles, checkerboards, K_{m,k} imbalance, bridges,
  empty-side and singleton/empty inputs. Every test enforces the
  I1 bookkeeping invariant `returned_cut == cut_size(labels)`.

### Added (2026-04-18) — I1 bookkeeping-invariant sweep on existing FM tests

- 21 existing FM-style tests across `feral-metis` (fm_refine),
  `feral-scotch` (halo_fm, band_fm, vertex_separator) now enforce
  the I1 invariant `returned_cut == cut_size(labels)` after the
  FM pass. This is the assertion the metis sign bug (fixed in
  `ba31609`) cannot survive.

### Added (2026-04-18) — feral-scotch SCOTCH-style nested dissection (S1-S5 complete)

- `feral-scotch::scotch_order(pattern)` and
  `feral-scotch::scotch_order_full(pattern, opts)` ship as the
  contract-conforming public API (matches `feral-amd::amd_order*` /
  `feral-metis::metis_order*` shape under
  `dev/plans/ordering-crate-contract.md`).
- Pipeline: optional graph compression (S1) at the top level →
  connected-component split → multilevel coarsening (shared with
  feral-metis through `internals`) → best-of-`n_sep_trials` initial
  bisection scored on post-FM cut → halo-FM uncoarsening at every
  projected level (S3) → direct vertex separator via two-sided FM
  (S2, instead of König's min vertex cover) → recursion with AMD
  leaf at `amd_switch`. Band FM (S4) is available as
  `band_fm::band_fm_refine` for callers that want frontier-only FM
  with anchor-supervertex balance accounting.
- 43 unit tests in feral-scotch; clippy clean; deterministic for a
  given `ScotchOptions::seed`.

### Fixed (2026-04-18) — feral-metis FM neighbour-update sign

- `feral_metis::internals::fm_refine::refine_bisection` had flipped
  signs at the `gain[u] ± 2w` neighbour update vs. the
  `gain = ed - id` convention used by `compute_gains` and
  `cur_cut -= gain[v]`. Corrupted `cur_cut` made FM effectively a
  no-op on graphs requiring real moves; the bug was hidden by all
  four existing tests starting from already-optimal cuts or
  blocked-by-balance configurations.
- Added `fm_sign_invariant_on_alternating_path` regression test
  enforcing the I1 invariant `returned_cut == cut_size(graph,
  labels)` (the assertion the bug cannot survive). Pre-fix
  produced `-1143` on P_10 with alternating ABAB labels (cut = 9);
  post-fix returns a small non-negative cut consistent with the
  new labels.
- Full analysis and follow-up adversarial set in
  `dev/research/metis-fm-sign-bug.md`.

### Changed (2026-04-17) — Ordering crate boundary locked (2.6.0)

- New workspace crate `feral-ordering-core`: defines the shared
  contract (`CscPattern<'a>`, `OrderingStats`, `OrderingError`,
  `CONTRACT_VERSION = 1`) that all four ordering crates will
  implement. Zero deps beyond `std`.
- **Breaking:** `feral-amd`'s public surface is retrofitted onto the
  contract.
  - `CscPattern` and error type now re-exported from
    `feral-ordering-core`; `AmdError` removed (use `OrderingError`).
  - `CscPattern` borrows `&[i32]` (was `&[usize]`);
    `amd_order*` returns `Vec<i32>` (was `Vec<usize>`).
  - All public entry points now return
    `Result<_, OrderingError>`.
  - New `amd_order_full(pattern, opts) -> (perm, OrderingStats,
    AmdStats)` — the contract-conforming three-tuple variant;
    `OrderingStats.time_us` is populated, fill/flop estimates are
    `None` pending analysis-phase work.
- Rationale: lock the boundary before implementing METIS, SCOTCH,
  KaHIP so all four backends plug into Ipopt against the same
  surface. See `dev/plans/ordering-crate-contract.md` and
  `dev/decisions.md` entry of 2026-04-17.
- Evidence: all 12 SuiteSparse AMD oracle fixtures still reproduce
  bit-for-bit after the retrofit (perm, ncmpa, ndiv, nms_ldl,
  nms_lu, n_dense_deferred); 29 lib tests pass; clippy clean;
  clean-room check still passes.

### Added (2026-04-17) — feral-amd standalone crate

- New workspace member `crates/feral-amd`: clean-room Approximate
  Minimum Degree (AMD) fill-reducing ordering, Amestoy-Davis-Duff
  quotient graph variant. Full Slice A (correctness) and Slice B
  (mass elimination + supervariable detection) landed under
  `dev/plans/ordering-amd-upgrade.md`.
- Public API: `amd_order`, `amd_order_with_stats`, `amd_order_opts`;
  `CscPattern`, `AmdOptions`, `AmdStats`, `AmdError`.
- Binaries: `feral-amd` (triplet-file CLI) and `feral-amd-bench`
  (arrow/band/grid fixture suite).
- External-oracle match: byte-for-byte agreement with the
  SuiteSparse AMD Rust crate (`amd` 0.2.2) on the pinned
  `tests/data/amd_oracle/*.txt` fixtures
  (diag_4, tridiag_10, arrow_5, arrow_200, band_20_3, grid_7x7,
  amd_demo_24), covering permutation and flop counters.
- Not yet integrated into `feral`. Integration is deferred to
  `dev/plans/ordering-integration.md`.

### Known issues (Phase 2 in progress)

- **The sparse path produces catastrophically wrong residuals on
  matrices with n > 500.** Phase 1 validation only measured
  matrices with n ≤ 500 (the bench harness enforced this via a
  Phase 1a hold-over filter that was not removed until Phase 2).
  When the filter was lifted in Phase 2.1.2, the sparse path
  produced residuals 10⁴ to 10¹⁴ on larger matrices already
  present in the corpus (CHWIRUT1 through CRESC132 at n=5314),
  while canonical MUMPS and SPRAL/SSIDS produced residuals at
  machine precision. Root cause: missing global MC64
  matching-based scaling. Fix in progress as Phase 2.2.1. Until
  it lands, do not use feral on matrices the dense path cannot
  handle.
- **Phase 1 residual pass rate is not a numerical quality
  measurement**, it is a measurement against the bench tolerance
  `n · ε · 10⁶`. On small matrices this tolerance is loose enough
  (≈ 10⁻⁷ at n=500) to accept feral residuals that are already
  6–8 orders of magnitude worse than canonical solvers. Phase 1's
  99.7% sparse residual pass rate survives this re-reading; what
  does not survive is any implicit claim that feral is numerically
  comparable to canonical solvers at those residual levels.

### Phase 2.4 performance (2026-04-14)

- Dense Schur update now uses a pulp-dispatched NEON SIMD kernel
  with 4-way loop unrolling and independent accumulators
  (`src/dense/schur_kernel.rs`). The kernel uses separate
  `mul_f64s` + `sub_f64s` (no FMA) so per-lane rounding is
  bit-identical to the scalar reference; this is verified by
  `assert_eq!` unit tests across a length sweep up to 1024. The
  kernel is wired into `do_1x1_update` and `do_2x2_update` in
  `src/dense/factor.rs` with no runtime A/B flag.
- KKT corpus bench vs MUMPS oracle (n ≤ 500 dense, full sparse
  corpus): dense factor p90 **2.27 → 1.86** (−18.1%); sparse
  factor p90 **3.18 → 2.82** (−11.3%). Both Phase 2.8 exit
  criteria (dense ≤ 2.0, sparse ≤ 3.0) now met.
- Inertia and residual-pass counts are bit-identical to the
  pre-SIMD scalar baseline: dense 152911/154481 inertia, sparse
  153009/154588 inertia, sparse 154329/154588 residual pass. Zero
  correctness regressions.
- An earlier attempt (Phase 2.4.2) wired an FMA-based unroll4
  kernel and caused 4 sparse inertia mismatches from 1-ULP pivot
  classification flips at the `zero_tol` boundary; reverted and
  replaced with the bit-exact non-FMA variant. See
  `dev/tried-and-rejected.md` and `dev/decisions.md` Phase 2.4.3.

### Phase 2.8.1 exit partition check (2026-04-14)

**Correction to the "both exit criteria met" claim above.** The
Phase 2.4 entry measures against the overall `factor/MUMPS` p90
aggregate. The spec exit criterion in `FERAL-PROJECT-SPEC.md` §1747
and `dev/plans/phase-2-planning.md` §2.8.1 is stricter: it asks
"within 2× of MUMPS on small-frontal KKT set, within 3× on medium
set", with explicit bucket definitions (small-frontal: max frontal
dim < 200 AND n ≤ 10³; medium: max frontal dim < 500 AND n ≤ 10⁴).

Applying the partition:

| bucket              |  count | p90  | target | verdict |
|---------------------|-------:|-----:|-------:|:-------:|
| Dense small-frontal | 147982 | 1.39 | ≤ 2.0  | PASS    |
| Dense medium        | 152145 | 1.74 | ≤ 3.0  | PASS    |
| Sparse small-frontal| 153455 | 2.81 | ≤ 2.0  | **FAIL**|
| Sparse medium       | 153560 | 2.81 | ≤ 3.0  | PASS    |

Dense meets both bars cleanly. **Sparse small-frontal fails** the
strict partition with p90 = 2.81 (target ≤ 2.0). Phase 2 cannot
exit formally until this is resolved.

Profile evidence (`examples/profile_sparse_smallfront.rs`, 152128
small-frontal matrices) locates the bottleneck at `amd_order`:
39.8% of total time with a fat tail of ~9 ms on n=234 matrices
(DISCS family). The plan's Phase 2.5.1 target (Liu row-subtree
column counts) is only 2.6% of the budget and is demoted. The new
Phase 2.5.1 priority is diagnosing and fixing AMD. See
`dev/decisions.md` 2026-04-14 "Phase 2.5 priority reordered".

### Phase 2.5.1′ AMD + symbolic fixes (2026-04-14)

Six surgical fixes, identified by an instrumented triage binary
(`examples/triage_discs_amd.rs`) that counted per-phase µs and
scalar `contains` / insert calls:

- **AMD mark array** (`src/ordering/amd.rs`). Replaced
  `adj[a].contains(&b)` inside the fill-edge loop with a scratch
  `Vec<bool>` of size n reused across steps. Marks the current
  adjacency once, checks/inserts with O(1) lookups, unmarks before
  the next outer iteration. Drops the fill phase from O(deg³) to
  O(deg²) per step. Root cause of the pathology: on near-dense
  inputs (DISCS_0012, DMN15103_0000 fully dense) the reachable set
  was already a clique so every `contains` returned `true` after
  scanning the full adjacency vector — 778k lookups for zero inserts
  on DISCS_0012.
- **AMD dense-clique shortcut** (`src/ordering/amd.rs`). When the
  pivot's live neighbors equal all remaining live nodes, eliminating
  it forms a clique among survivors: push them in any order and
  return. Short-circuits DMN15103_0000 entirely and cuts DISCS_0012
  to just the first few steps.
- **Counting-sort `permute_pattern`** (`src/ordering/amd.rs`).
  Replaced `Vec<Vec<usize>>` + sort + dedup with a two-pass
  counting-sort layout (count, prefix sum, fill) plus one per-column
  `sort_unstable` to preserve the sorted-column invariant. ~7×
  faster on DMN15103_0000. Each off-diagonal entry is copied exactly
  once instead of twice then deduped.
- **Dead loop in supernode detection** (`src/symbolic/supernode.rs`).
  Removed a `for child_s in 0..n_snodes` loop that called
  `find_root` on every candidate and did nothing with the result
  (empty body). O(n²) wasted work per matrix. Snode max time
  dropped 507→68 µs; share 7.3% → 1.2%. GROUPING family fell off
  the top-30 worst offenders list.
- **Etree renumbering from postorder** (`src/symbolic/mod.rs`).
  Replaced the second `EliminationTree::from_pattern` call with an
  O(n) renumbering of the AMD-permuted etree through the postorder.
  Postorder is a topological relabeling of the elimination tree,
  so the tree structure is preserved and only node labels change.
  ~3% sparse small-frontal p90 improvement on 3-run median.
- **Dead transpose call** (`src/numeric/factorize.rs`). Removed
  `let _ = build_csc_transpose(&permuted);` and the helper function
  — the value was computed and immediately discarded. Full O(nnz)
  pass per matrix for nothing.

**Phase 2.8.1 exit criterion now satisfied.** All four partitions
PASS on the full KKT bench (154588 matrices):

| bucket              | count  |  p90 | target | verdict |
|---------------------|-------:|-----:|-------:|:-------:|
| Dense small-frontal | 147982 | 1.56 | ≤ 2.0  | PASS    |
| Dense medium        | 152145 | 1.96 | ≤ 3.0  | PASS    |
| Sparse small-frontal| 153455 | 1.99 | ≤ 2.0  | PASS    |
| Sparse medium       | 153560 | 2.00 | ≤ 3.0  | PASS    |

3-run medians on sparse small-frontal: **2.00 / 1.98 / 2.00**
(target ≤ 2.0). Tight margin — run-to-run noise is ~3–5%, so the
next regression in this band could push it back over the gate.
Flagged for monitoring in Phase 3+.

All 93 library tests pass. Inertia and residual counts unchanged.
Zero correctness regressions. See `dev/sessions/2026-04-14-04.md`
and `dev/decisions.md` Phase 2.5.1′ entries.

### Phase 1b Exit (2026-04-12)

Phase 1b closed under the multi-source consensus exit criterion on
the n ≤ 500 subset of the KKT corpus. Feral matches canonical
Fortran MUMPS 5.8.2 on **99.97%** of that subset's inertia — higher
than the agreement between canonical MUMPS and canonical SPRAL/SSIDS
(98.25%). See `dev/sessions/2026-04-12-01.md` and the Known issues
above for the limits of this claim.

### Added
- Sparse multifrontal LDLᵀ solver (`factorize_multifrontal`,
  `solve_sparse`, `solve_sparse_refined`)
- CSC sparse matrix infrastructure (`CscMatrix`, `CscPattern`)
- AMD ordering, elimination tree, postorder, column counts, supernode
  detection with nemin amalgamation (CHOLMOD-style pipeline)
- Symbolic factorization (`symbolic_factorize`) with postorder
  composition of AMD permutation
- Bench failure analysis: family-grouped failure tables, top-worst
  residual lists, dense ∩ sparse cross-comparison
- Bench `FERAL_EMIT_SIDECARS` environment variable: emits canonical
  `.feral.json` sidecars alongside each matrix for consensus analysis
- External benchmark infrastructure (`external_benchmarks/`):
  - Native Fortran MUMPS 5.8.2 oracle (build from `ref/mumps`,
    manifest-based driver, Python JSON wrapper)
  - Native Fortran SPRAL/SSIDS oracle (meson + METIS build, same
    driver pattern)
  - Multi-source consensus computation (Python), applies
    Definitive / Borderline / NumericallyIntractable / Excluded
    verdicts per matrix across four oracles
- Dense LDLᵀ factorization with Bunch-Kaufman pivoting (scalar, unblocked)
- Full 7-step solve sequence with equilibration
- Iterative refinement (`solve_refined`) with best-iterate strategy
- Iterative infinity-norm equilibration (Knight-Ruiz)
- Benchmark harness with built-in dense matrix timing
- CI workflow (test, clippy, fmt, no-unwrap)
- Property-based tests and stress tests (121 total tests)
- Fused update+argmax optimization (halves memory traffic per pivot step)

### Fixed
- **Phase 2.3 — delayed pivoting + sign-preservation fix**: the
  sparse multifrontal path now delays rejected pivots (both 1×1
  column-relative and 2×2 Duff-Reid growth-bound) from non-root
  supernodes to their parent, giving them a landing zone where
  child contributions have been assembled and the block is more
  likely to pivot cleanly. At root supernodes where no further
  delay is possible, `try_reject_1x1_frontal` preserves the
  pivot's sign in the `ForceAccept` fallback: small-but-nonzero
  pivots are accepted with `inertia.positive`/`negative` (not
  counted as zero) and flagged for iterative refinement. Only
  `|d| <= zero_tol ≈ eps` counts as a zero pivot. Evidence:
  sparse KKT sweep worst residual `2.31e+11 → 3.22e-4` (15 orders
  of magnitude across Phase 2.3), sparse-only failure count
  `3328 → 64`, parity panel `11/28 → 22/28`. Dense KKT numbers
  unchanged (99.0% inertia, 99.7% residual pass, 3.99e-2 worst
  on ACOPP30_0002) because the sparse-only `pivot_threshold =
  0.01` config is scoped to `params_kkt_sparse` and
  `BunchKaufmanParams::default()` stays at `0.0`. See
  `dev/sessions/2026-04-13-02.md`, `03.md`, and `04.md`.
- **Phase 2.3 — refinement termination fix**: `solve_sparse_refined`
  (and `dense::solve_refined`) now iterate up to 10 steps (was 3)
  and terminate on a residual-based criterion `||r|| <
  eps*sqrt(n)*||b||` instead of the old `|dx|/|x|` threshold.
  Under `ForceAccept` factorizations the trajectory is non-
  monotone — corrections produce small `dx` without reducing `r`,
  so `dx` is a false convergence signal and the old loop exited
  before reaching the machine-precision basin. The `||b|| = 0`
  case is handled with an absolute threshold; `||b||` is NOT
  clamped to a floor, which would defeat the relative criterion
  on small-RHS matrices (e.g. CERI651C with `||b|| = 3.238e-5`).
  Evidence: parity panel `22/28 → 27/28` (un-ignored AVION2_0510,
  CERI651C_0746, CERI651ELS_1482, HAHN1_0004, MEYER3NE_0253),
  sparse residual pass `154237 → 154329`, worst sparse residual
  `3.22e-4 → 2.50e-4`. Only SSI_2597 remains ignored as a
  pathological factorization-level case deferred to Phase 2.4.
- **Phase 2.2.2 — ACOPP30 MC64 regression**: Phase 2.2.1 MC64
  scaling improved 6 of 7 sanity-panel matrices but pushed
  ACOPP30_0000 from a pre-MC64 residual of `2.84e+16` to
  `2.27e+46` — a 30-order-of-magnitude regression caused by 5
  forced-zero pivots in the `ForceAccept` branch interacting with
  the unscaled residual recompose. Phase 2.2.2 adds
  `BunchKaufmanParams::pivot_threshold` (a column-relative 1×1
  rejection clause matching MUMPS CNTL(1) / SSIDS `options%u`,
  default `0.01`) plus the Duff-Reid 2×2 growth bound. MC64
  callers (`tests/mc64_regression.rs::ldlt_params`,
  `src/bin/bench.rs::params_kkt`,
  `examples/triage_large_cresc132.rs`) opt in at `u = 0.01`.
  ACOPP30_0000 residual drops `2.27e+46 → 1.076e-1` (47 orders),
  now ~17 orders better than the pre-MC64 Identity baseline. The
  remaining 3 regression targets (CHWIRUT1, CRESC100, CRESC132)
  are unchanged — their inertia is already exact or ±2, so the
  column-relative rejection has nothing to fire on. Full closure
  of the MC64 residual gap requires delayed pivoting (Phase 2.3).
  Validation: `dev/validation/phase-2.2.2-pivot-rejection.md`.
- **Postorder pipeline bug**: `symbolic_factorize` did not apply
  postorder to the elimination tree before supernode amalgamation,
  causing merged supernodes to have non-contiguous columns while
  downstream code assumed contiguous ranges. Closed MGH10S_0000
  (inertia (50,1,0) → (35,16,0), residual 2.61e21 → 1.10e-16).
- **Pivot threshold mismatch**: factor flagged pivots as zero at
  `100*eps` while solve divided by them at `eps*1e-10`. The band in
  between produced catastrophic cancellation. `Factors` and
  `FrontalFactors` now carry `zero_tol`/`zero_tol_2x2`; both solve
  paths skip any pivot the factor counted as zero. Closed POLAK6_0021
  (residual 8.97e-1 → 4.6e-17).
- **Best-iterate refinement**: `solve_refined` and
  `solve_sparse_refined` now track the smallest `||r||` across
  refinement steps and return the corresponding `x`, guaranteeing the
  refined answer is no worse than the unrefined one on rank-deficient
  matrices where ForceAccept produced a wrong `A⁻¹`.
- **`zero_tol` default lowered** from `100 * EPSILON` to `EPSILON`.
  The 100× safety margin was flagging tiny-but-legitimately-positive
  pivots as zero on small SPD matrices. Verified against canonical
  Fortran MUMPS, SPRAL/SSIDS, and rmumps on CERI651DLS_0534 and
  FBRAIN3LS_0788. Closed the final 32 Definitive feral failures.

### Changed
- Phase 1b exit criterion redefined from "100% correct inertia +
  solution vs rmumps" to multi-source consensus across feral, rmumps,
  canonical MUMPS 5.8.2, and SPRAL/SSIDS. Recorded in
  `dev/decisions.md` (entry 2026-04-12) with a reconsideration clause.
- Bench no longer prints per-row PASS lines for the 153k KKT corpus
  (~153k lines removed from stdout, runtime reduced). The bench now
  emits summary tables with family-grouped failure analysis and a
  dense ∩ sparse cross-comparison.
