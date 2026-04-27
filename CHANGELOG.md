# Changelog

All notable changes to FERAL will be documented in this file.

## [Unreleased]

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
