# Policy API Design — v2

**Status:** Proposed (revision of `policy-traits-api.md`)
**Drafted:** April 2026
**Supersedes:** `policy-traits-api.md` (v1) — kept for historical context.

## 0. Why this revision exists

v1 proposed five `Send + Sync` traits (`OrderingPolicy`, `ScalingPolicy`,
`PivotPolicy`, `SupernodeStrategy`, `ParallelScheduler`) plus a builder.
Expert review (faer / MUMPS / SSIDS / IPOPT consultations, April 2026) was
unanimous that the five-trait shape is heavier than any reference codebase
supports and motivated only by goals that don't require it.

Revised goals (explicit):

1. Support standard existing approaches (AMD/METIS/Scotch/KaHIP × identity/
   inf-norm/MC64 × static Bunch-Kaufman). Same as today.
2. Support a learned **meta-policy** that, given KKT features, picks among
   the standard approaches.
3. Support a future RL-trained GNN that **actively participates** in the
   ordering / scaling / pivoting / factorization-path decisions.

Key insight from the discussion: goal 2 lives **entirely in the caller**
(ripopt or a meta-layer). It needs zero new feral API. Goal 3 motivates
function-shaped extension points at three stages — but per faer's idiom,
those are `Custom(...)` enum variants, not `dyn Trait` objects.

The bulk of what ripopt actually needs from feral in the next 12 months is
**Solver-level surface** (analyze/factorize/solve separation, FactorStatus,
inertia reporting on singularity, IncreaseQuality ratchet, iterative
refinement, stats). That work is independent of policy-API shape and is
the load-bearing part of this plan.

## 1. Goals and non-goals

**Goals.**

1. ripopt can drive feral with the IPOPT-style analyze-once / refactor-many /
   multi-RHS-solve pattern with explicit inertia and quality ratcheting.
2. Each pluggable stage (ordering, scaling, pivot params) accepts a
   precomputed result *or* a callback, so a learned policy plugs in without
   core changes.
3. Default behavior unchanged: `Solver::new()` reproduces current Phase 2
   pipeline bit-for-bit.
4. RL training is feasible: deterministic execution given fixed inputs +
   fixed policy decisions, plus per-stage stats sufficient for reward
   computation.

**Non-goals.**

1. Five `Send + Sync` traits. We use enums with `Custom*` variants instead
   (faer pattern, validated by all four expert consultations).
2. Per-pivot dynamic dispatch inside `factor_frontal`. Per-pivot RL inside
   Bunch-Kaufman is open research; the multifrontal data-flow makes
   "delayed pivot" not a clean callback site (MUMPS expert).
3. A `ParallelScheduler` trait. faer's `Par` enum is the proven shape; we
   defer until a second backend exists.
4. A bundled `SupernodeStrategy` (v1 §2.4). SSIDS and MUMPS both treat
   `nemin` (amalgamation) and the LDLᵀ-aware preprocess as **independent**
   decisions; bundling was a v1 mistake.
5. GNN policy implementations. We design the seams; we don't train models.

## 2. The shape

### 2.1 Stage 1 — Ordering

```rust
pub enum Ordering {
    Amd,
    Metis,
    Scotch,
    Kahip,
    Auto,
    Custom(Vec<i32>),
    CustomFn(Arc<dyn Fn(&CscPattern) -> Result<Vec<i32>, OrderingError> + Send + Sync>),
}
```

- `Custom(perm)` is the GNN/external escape hatch — exact analog of faer's
  `SymmetricOrdering::Custom(PermRef)` at
  `faer/src/sparse/linalg/cholesky.rs:486–495`.
- `CustomFn` is for callers that want feral to compute the ordering on
  demand (e.g. when the symbolic factorization is invalidated mid-IPM).
- `Auto` retains the existing `choose_adaptive` heuristic.
- The LDLᵀ-aware **preprocess decision** moves here too (it lives with
  ordering in SSIDS, not with supernode amalgamation):

```rust
pub struct OrderingConfig {
    pub method: Ordering,
    pub preprocess: OrderingPreprocess,  // None | LdltCompressed | Auto
}
```

### 2.2 Stage 2 — Scaling

```rust
pub enum Scaling {
    Identity,
    InfNorm,
    Mc64 { cache: Option<Mc64Cache> },
    Auto,
    Custom(Vec<f64>),
    CustomFn(Arc<dyn Fn(&CscMatrix) -> Result<Vec<f64>, ScalingError> + Send + Sync>),
}

impl Scaling {
    pub fn invalidate_cache(&mut self);                  // explicit, not auto
    pub fn cache_state(&self) -> Option<CacheStamp>;     // for ripopt to verify
}
```

The cache lives in the variant payload (faer-style), not behind interior
mutability. **Explicit invalidation** is critical — IPOPT consumer review
flagged auto-invalidation by pattern hash as O(nnz) and incorrect for
restoration-phase entry. ripopt is responsible for calling
`invalidate_cache()` at the right moments (restoration entry/exit,
warm-start with structural change).

### 2.3 Stage 3 — Pivot params

```rust
pub struct FrontalContext {
    pub size: usize,
    pub depth_in_etree: u32,
    pub estimated_max_growth: Option<f64>,
    pub is_root: bool,
    _non_exhaustive: (),
}

pub enum PivotStrategy {
    Static(BunchKaufmanParams),
    PerFrontal(Arc<dyn Fn(&FrontalContext) -> BunchKaufmanParams + Send + Sync>),
}
```

Defaults to `Static(BunchKaufmanParams::default())`. The
`PerFrontal(Box<dyn Fn>)` variant is the GNN injection site.

`FrontalContext` is the GNN's input feature vector at pivoting time — it
must be designed carefully and `non_exhaustive`-marked so fields can be
added without breaking model-trained-against-old-schema users.

**Frontal granularity, not per-pivot.** SSIDS uses one global `u`; MUMPS
derives per-front modifiers but per-pivot decisions are inside the inner
loop. Per-frontal is the largest granularity at which a learned policy can
plausibly intervene without rewriting the dense kernel.

**No `on_delayed_pivot` callback.** v1 proposed one; MUMPS expert was
emphatic that delayed pivoting is structural data flow, not a decision
point. Failed-pivot behavior is a parameter (`failed_pivot_method:
{TPP, PassToParent, Regularize}`), not a callback.

### 2.4 Stage 4 — Amalgamation

```rust
pub enum Amalgamation {
    Nemin(usize),                                       // default Nemin(32) (SSIDS)
    Custom(Vec<MergeDecision>),                          // per-tree-edge replay
    CustomFn(Arc<dyn Fn(&EliminationTree, &[usize])
                       -> Vec<Supernode> + Send + Sync>),
}

pub enum MergeDecision { Merge, Split }
```

The `Custom` and `CustomFn` variants are required by Phase 5 (companion
research program at `~/Dropbox/projects/big-ideas/kkt-ordering-discovery-phase5-design.md`),
which models supernode amalgamation as a sequence of **per-elimination-tree-edge
merge/split decisions** — a learned head's output, not a single global threshold.

`Nemin(usize)` reproduces SSIDS behavior and is the default. The variant exists
because every reference solver feral currently competes against uses a
threshold-style rule; `Custom*` exists for Phase 5 imitation/replay/learned
policies.

(Earlier draft of this plan collapsed this stage to a plain
`AmalgamationParams { nemin }` struct on the SSIDS expert's recommendation
that "no real user overrides nemin." Phase 5 *is* that user. Restored.)

### 2.5 Stage 5 — Parallel layout (`Par` enum, not a trait)

```rust
pub enum Par {
    Seq,
    Rayon(NonZeroUsize),
    DeterministicRayon(NonZeroUsize),  // §4
    // --- Phase 5.4 will land these; non-breaking via enum extension ---
    // MultifrontalTask(NonZeroUsize),
    // SupernodalDag(NonZeroUsize),
    // Hybrid { ... },
}
```

faer's pattern (`faer/src/lib.rs:929-936`): a value, not a trait. Zero v-table
overhead, simple `match` codegen, and new variants are non-breaking.

Phase 5 §3 lists `{serial, multifrontal-task, supernodal-DAG, hybrid}` as
discrete actions in stage 6 of the joint policy. Those variants land when
Phase 5.4 needs them; the enum approach absorbs them without API change.

Earlier draft of this plan said "defer until a second backend exists" — Phase 5.4
is that second use case. We acknowledge it here so the shape and naming are
chosen now.

### 2.6 What's deliberately absent

- **A `ParallelScheduler` trait.** `Par` enum is the proven shape (faer).
- **A `SupernodeStrategy` trait.** Split into `Amalgamation` (§2.4) and
  `OrderingConfig.preprocess` (§2.1) — the two decisions are independent in
  SSIDS and MUMPS.
- **A `PivotPolicy::on_delayed_pivot` callback.** MUMPS expert: delayed
  pivoting is structural multifrontal data flow, not a decision point. Failed-
  pivot behavior is a parameter on `BunchKaufmanParams`, not a callback.

## 3. Solver-level surface (the load-bearing part)

This is what ripopt actually calls into. The policy enums above are knobs
on the configuration; the API below is the runtime contract.

```rust
pub struct Solver { /* … */ }

pub enum FactorStatus {
    Ok,
    Singular { n_zero: usize },
    WrongInertia { actual_neg: usize, expected_neg: usize },
    CallAgain { reason: CallAgainReason },   // dynamic-workspace resize, etc.
}

pub struct Inertia { pub n_pos: usize, pub n_neg: usize, pub n_zero: usize }

impl Solver {
    pub fn builder() -> SolverBuilder;
    pub fn new() -> Self;                                          // current defaults

    pub fn analyze(&mut self, pattern: &CscPattern) -> Result<(), FeralError>;
    pub fn factorize(&mut self, values: &[f64], expected_neg: Option<usize>) -> Result<FactorStatus, FeralError>;
    pub fn solve(&self, rhs: &mut [f64], nrhs: usize) -> Result<(), FeralError>;
    pub fn solve_with_refinement(
        &self,
        matrix: &CscMatrix,
        rhs: &mut [f64],
        max_iters: usize,
        tol: f64,
    ) -> Result<RefinementResult, FeralError>;

    pub fn inertia(&self) -> Option<Inertia>;
    pub fn provides_inertia(&self) -> bool;

    pub fn increase_quality(&mut self) -> bool;       // pivtol → pivtol^0.75 ratchet
    pub fn reset_quality(&mut self);

    pub fn stats(&self) -> &SolverStats;
    pub fn action_trace(&self) -> &ActionTrace;
}

pub struct SolverStats {
    pub symbolic_us: u64,
    pub factor_us: u64,
    pub solve_us: u64,
    pub refine_us: u64,
    pub peak_memory_bytes: usize,
    pub reallocations: u32,
}

/// Full record of every decision feral made, suitable for Phase 5 imitation
/// training data and RL credit assignment. Distinct from `SolverStats`
/// (timing/memory) because traces are extraction targets while stats are
/// observability. Both are populated on every factorization.
pub struct ActionTrace {
    pub ordering_used: Vec<i32>,                 // even when Auto picked it
    pub scaling_used: Vec<f64>,                  // even when Auto/MC64 produced it
    pub supernode_merges: Vec<MergeDecision>,    // per-tree-edge, even from Nemin
    pub per_frontal: Vec<FrontalTrace>,
    pub iterative_refinement_iters: u32,
    pub residual_history: Vec<f64>,
}

pub struct FrontalTrace {
    pub size: usize,
    pub depth_in_etree: u32,
    pub bk_params_used: BunchKaufmanParams,
    pub n_delayed: u32,
    pub n_two_by_two: u32,
    pub n_one_by_one: u32,
    pub flops: u64,
    pub time_us: u64,
    pub max_growth: f64,
}
```

Provenance:

- `analyze`/`factorize`/`solve` separation — IPOPT pattern at
  `IpMa57TSolverInterface.cpp:423-456`.
- `FactorStatus { WrongInertia, CallAgain }` — mirrors IPOPT's
  `ESymSolverStatus`; `CallAgain` covers MA27-style dynamic workspace
  realloc (`IpSparseSymLinearSolverInterface.hpp:172-181`).
- `increase_quality` — IpMa57TSolverInterface.cpp:821-834
  (`pivtol = min(pivtolmax, pivtol^0.75)`).
- Inertia on singular factorizations — needed for IPOPT's
  `PerturbForSingularity` (IpPDPerturbationHandler.hpp:63-68).
- Per-frontal stats — required for RL reward computation (goal 4 above).
- `ActionTrace` — required for Phase 5.1's imitation phase: extracts every
  decision feral made (including those derived by `Auto`, `Nemin`, `Mc64`)
  so the trace can be replayed via `Custom*` variants for testing or used
  as imitation training data.

## 4. Determinism guarantees

For RL training (goal 4 of §1) we commit to:

- Given the same matrix, same `Ordering`, same `Scaling`, same
  `PivotStrategy`, and same `Par` setting, the factorization produces
  bit-identical permutations, scalings, pivots, and numerical results
  across runs.
- `Par::Seq` is unconditionally deterministic. `Par::Rayon(N)` is
  deterministic *modulo floating-point reduction order*; we provide a
  `Par::DeterministicRayon(N)` mode that fixes assembly order at the cost
  of some parallelism.
- Stats are deterministic in count (flops, n_delayed, n_two_by_two) but
  not in timing (`*_us` fields).

Test gate: a parity test that runs the same factorization under each
`Par` mode and verifies count-stats agree.

## 5. The builder

```rust
let solver = Solver::builder()
    .ordering(OrderingConfig { method: Ordering::Amd, preprocess: OrderingPreprocess::Auto })
    .scaling(Scaling::Mc64 { cache: None })
    .pivot(PivotStrategy::Static(BunchKaufmanParams::default()))
    .amalgamation(Amalgamation::Nemin(32))
    .par(Par::Rayon(NonZeroUsize::new(8).unwrap()))
    .build();
```

`Solver::new()` is `Self::builder().build()` with all defaults. Existing
`Solver::with_params(NumericParams { ... })` is retained and translates
internally.

## 6. Implementation PR sequence

Each PR independently mergeable, ships its own tests, leaves `main`
green. The ordering reflects ripopt's blocking dependencies first, then
GNN seams, then nice-to-haves.

1. **PR 1: `Solver` API surface** (~3 days) — load-bearing for ripopt.
   - `analyze` / `factorize` / `solve` separation.
   - `FactorStatus` enum, inertia-on-singular, `increase_quality`.
   - `solve_with_refinement` with iter/residual reporting.
   - `SolverStats` plumbing (timing first; per-frontal trace in PR 4).
   - `ActionTrace` skeleton (ordering/scaling fields populated; per-frontal
     and merge-decision fields stubbed and filled in by PRs 4 and 5).
   - Test: parity test against current Solver behavior; multi-RHS solve;
     IncreaseQuality ratchet test.

2. **PR 2: `Ordering` enum + `OrderingConfig`** (~2 days)
   - Replace `OrderingMethod` enum at `symbolic/mod.rs:306–311`.
   - Add `Ordering::Custom(Vec<i32>)` and `Ordering::CustomFn(Box<dyn Fn>)`.
   - Move `OrderingPreprocess` decision into `OrderingConfig`.
   - Test: existing ordering tests pass byte-for-byte; new test for
     `Custom(perm)` injection; new test for `CustomFn` dispatch count
     (called once per analyze, not per factorize).

3. **PR 3: `Scaling` enum + explicit cache invalidation** (~2 days)
   - Replace `ScalingStrategy` at `scaling/mod.rs:177–195`.
   - `Scaling::Mc64 { cache }` carries cache in payload.
   - `invalidate_cache()` and `cache_state()` methods.
   - Test: existing scaling tests; MC64 cache reuse across two factorizes;
     cache invalidation produces fresh scaling.

4. **PR 4: `PivotStrategy` + `FrontalContext` + per-frontal trace** (~3 days)
   - Define `FrontalContext` (non_exhaustive) and `PivotStrategy`.
   - Refactor `factor_frontal` to call `policy.params(ctx)` once at top.
   - Plumb `FrontalTrace` collection (size, depth, bk_params_used,
     n_delayed, n_two_by_two, n_one_by_one, flops, time_us, max_growth).
   - Test: existing pivoting tests; synthetic test where `PerFrontal`
     returns different params at depth 0 vs leaves; verify dispatch
     happens exactly once per frontal (not per pivot).

5. **PR 5: `Amalgamation` enum with `Custom*` variants** (~2 days)
   - Define `Amalgamation::{Nemin(usize), Custom(Vec<MergeDecision>),
     CustomFn(...)}`.
   - Refactor existing nemin code as `Nemin(32)` default.
   - Plumb `supernode_merges: Vec<MergeDecision>` into `ActionTrace` so the
     merges actually performed (whether by `Nemin` or `Custom*`) are recorded.
   - Test: existing supernode tests; replay test (extract merges from a
     `Nemin(32)` run, replay them via `Custom(merges)`, verify identical
     supernode structure); a `CustomFn` test that emits a learned-style
     decision sequence.

6. **PR 6: Determinism mode** (~2 days)
   - Add `Par::DeterministicRayon(N)` ordering of assembly tasks.
   - Determinism test gate (run × N, verify identical inertia + pivots +
     factor norms; counts in `SolverStats` identical).

Total: ~14 working days, 6 PRs. PR 1 unblocks ripopt independently of
the rest. PRs 2–5 are the Phase 5 seams (ordering, scaling, pivot,
supernode amalgamation — Phase 5's stages 1, 2, 4, 5 in §3 of its
design). PR 6 is RL prep (determinism). Phase 5's stages 3 (symbolic),
6 (parallel layout), and 7 (iterative refinement) are handled
respectively by deterministic-after-ordering symbolic factor (no API
needed), the `Par` enum (§2.5, future variants land non-breakingly
in Phase 5.4), and `Solver::solve_with_refinement` (§3, in PR 1).

## 7. Migration to ripopt

Once PR 1 lands:

- ripopt's `LinearSolver` trait (analog of IPOPT's
  `SparseSymLinearSolverInterface`) calls `feral::Solver::analyze /
  factorize / solve / increase_quality`.
- ripopt retires rmumps in favor of feral.
- ripopt's `PerturbationHandler` (analog of `IpPDPerturbationHandler`)
  drives `increase_quality` and `Solver::factorize(values, expected_neg)`.

Once PRs 2–4 land:

- ripopt can configure feral with arbitrary `Ordering` / `Scaling` /
  `PivotStrategy` per IPM run, including `Custom(...)` variants supplied
  by a meta-policy or a learned model.

## 8. Path to learned policies (goal 3)

For each stage, the learned-policy implementation is:

- **Ordering (one-shot)**: GNN runs outside feral, produces `Vec<i32>`,
  passed via `Ordering::Custom(perm)`. No feral changes after PR 2.
- **Scaling (one-shot)**: GNN runs outside feral, produces `Vec<f64>`,
  passed via `Scaling::Custom(scale)`. No feral changes after PR 3.
- **Pivoting (per-frontal)**: GNN runs *inside* feral via a closure that
  takes `FrontalContext` and returns `BunchKaufmanParams`. Wrapped as
  `PivotStrategy::PerFrontal(Arc::new(move |ctx| model.predict(ctx)))`.
  No feral changes after PR 4.
- **Supernode amalgamation (per-edge)**: GNN runs outside feral, emits a
  `Vec<MergeDecision>`, passed via `Amalgamation::Custom(merges)`. Or, if
  the model needs to see derived supernode geometry, `Amalgamation::CustomFn`
  takes the etree + col_counts and emits supernodes directly. No feral
  changes after PR 5.

For RL training:

- The training driver runs `Solver::factorize` many times with different
  policy decisions and reads `SolverStats` and `ActionTrace` for the
  reward signal and credit assignment.
- `Par::DeterministicRayon` (PR 6) makes rollouts reproducible.
- Per-frontal stats give per-decision credit assignment.
- `ActionTrace` extracted from each rollout becomes (a) imitation training
  data for that policy or (b) input to off-policy RL.

For the meta-policy (goal 2 of §0):

- Lives outside feral. Looks at KKT features
  (size, density, condition estimate, …), picks a config, calls
  `Solver::builder()`. No feral changes ever.

## 9. Open questions

1. **Where does the meta-policy live in ripopt?** Probably `ripopt::policy`
   or a separate `ripopt-policy` crate. Out of scope for this plan; design
   when ripopt migration starts.

2. **`Arc<dyn Fn>` vs `Box<dyn Fn>` in `Custom*` variants?** `Arc` lets
   callers clone the strategy across builder calls and across IPM iterations
   without re-constructing. Default to `Arc`; revisit if it becomes a hot
   path issue (it won't — these are O(1)/factorize).

3. **Should `FrontalContext` carry the front data itself (matrix view) or
   only summaries?** Decision for now: summaries only, for two reasons:
   (a) the matrix view's lifetime tangles with the factor loop; (b) GNN
   policies will be trained on summaries anyway since per-element data
   is too high-dimensional. Add a `front_view()` accessor later if needed.

4. **Failed-pivot policy as parameter or callback?** Parameter:
   `failed_pivot_method: {TppPass, PassToParent, Regularize}` on
   `BunchKaufmanParams`. SSIDS evidence (only two values exist in
   practice) and MUMPS evidence (delayed-pivot is structural, not a
   decision point) both rule out a callback.

## 10. Tests gates

Every PR ships with parity tests against the pre-PR behavior. Beyond that:

- **PR 1**: ripopt-style usage test (analyze → factorize → solve → IncreaseQuality → factorize) on a small KKT system.
- **PR 2**: `Custom(perm)` injection produces predictable downstream symbolic factor; `CustomFn` is called exactly once per analyze.
- **PR 3**: MC64 cache hit test (analyze + 2× factorize without invalidation: scaling computed once); cache invalidation test (after invalidate, scaling recomputed).
- **PR 4**: per-frontal dispatch count test (PerFrontal closure called exactly N_fronts times per factorize, never inside pivot loop).
- **PR 5**: `Nemin(32)` reproduces the existing supernode structure on the
  test corpus; `Custom(merges)` round-trip (extract → replay) produces
  identical supernodes; `CustomFn` exercises a non-`Nemin`-shaped policy
  on a synthetic etree.
- **PR 6**: 10× rollouts under `Par::DeterministicRayon(8)` produce identical pivots, identical inertia, identical `SolverStats` counts.

The existing consensus-vs-MUMPS framework keeps running on every PR.

## 11. Summary of changes from v1

| v1 proposal | v2 decision | Reason |
|---|---|---|
| 5 traits (`OrderingPolicy` etc.) | Enums with `Custom*` variants | faer pattern; SSIDS/MUMPS validate enum-with-menu shape |
| `ParallelScheduler` doc-hidden trait | Defer; use `Par` enum if/when needed | One impl for years; faer precedent at `faer/src/lib.rs:929` |
| `SupernodeStrategy` bundles nemin + preprocess | Split: `Amalgamation::{Nemin, Custom, CustomFn}` + `OrderingConfig.preprocess` | SSIDS/MUMPS treat them as independent; Phase 5 needs per-edge merge as a learned action so the `Custom*` variants are non-negotiable |
| `on_delayed_pivot` callback | Drop; `failed_pivot_method` is a parameter | MUMPS: delayed pivoting is structural data flow, not a decision point |
| `Mutex<Mc64Cache>` interior mutability | Cache lives in `Scaling::Mc64 { cache }` payload + explicit `invalidate_cache()` | IPOPT needs explicit invalidation; auto-detection by pattern hash is wrong for restoration entry |
| No `Solver`-level surface specified | `analyze/factorize/solve` + `FactorStatus` + `increase_quality` + `solve_with_refinement` + `SolverStats` | IPOPT consumer requirement; load-bearing for ripopt |
| No determinism story | `Par::DeterministicRayon(N)` + count-stable stats | Required for RL training rollouts |

## 12. Phase 5 alignment

This plan is the infrastructure prerequisite for Phase 5 (the learned
joint-factorization-policy research program at
`~/Dropbox/projects/big-ideas/kkt-ordering-discovery-phase5-design.md`).
Phase 5 §3 enumerates seven stages in the joint action space; the table
below maps each stage to its v2 mechanism.

| Phase 5 stage | Action shape | v2 mechanism | Status |
|---|---|---|---|
| 1. Scaling | discrete + δ | `Scaling::{Identity, InfNorm, Mc64, Auto, Custom, CustomFn}` | Custom-ready (PR 3) |
| 2. Ordering | permutation or heuristic | `Ordering::{Amd, Metis, Scotch, Kahip, Auto, Custom, CustomFn}` | Custom-ready (PR 2) |
| 3. Symbolic factor | deterministic from ordering | implicit (no API; recomputed each `analyze`) | n/a |
| 4. Pivot strategy | per-frontal `(strategy, threshold)` | `PivotStrategy::{Static, PerFrontal}` + `FrontalContext` | Custom-ready (PR 4) |
| 5. Supernode merging | per-tree-edge merge/split | `Amalgamation::{Nemin, Custom, CustomFn}` | Custom-ready (PR 5) |
| 6. Parallel layout | discrete | `Par` enum; Phase 5.4 lands `MultifrontalTask` / `SupernodalDag` / `Hybrid` non-breakingly | Future-extensible (§2.5) |
| 7. Iterative refinement | `{off, k, tol}` | `Solver::solve_with_refinement(max_iters, tol)` | Custom-ready (PR 1) |

Phase 5.1's gate ("harness can reproduce each reference solver's behavior
by replaying its action trace through feral") requires both ends of the
trace pipeline:

- **Replay** — supplied by `Custom*` variants on every stage above.
- **Extraction** — supplied by `ActionTrace` returned from every
  factorization. The trace records `ordering_used`, `scaling_used`,
  `supernode_merges`, per-frontal `bk_params_used`, and refinement history,
  whether feral derived them via `Auto`/`Nemin`/`Mc64` or consumed them
  from the caller's `Custom*` variant.

The Phase 5 deliverable (per its §12) is "a set of trait-impl-shaped
artifacts ripopt loads at startup." Under v2 those become precomputed
`Vec<i32>`/`Vec<f64>` payloads (one-shot stages) and
`Arc<dyn Fn(...) -> ...>` closures (per-frontal and per-edge stages),
loaded into the corresponding enum's `Custom*` variant. Phase 5 §7's
citation of v1's "five new traits" is superseded by this v2 plan; the
infrastructure cost is comparable (≈14 working days vs. v1's ≈11) and
the additional cost buys the IPOPT-aligned `Solver` surface ripopt
needs anyway.

Phase 5 stages **not addressed by feral** (and therefore not by this
plan):

- IPM iterate-state conditioning (Σ_x, Σ_s magnitudes) is the caller's
  input to the policy network, not a feral concern.
- Hardware-aware reward (V100/A100/TPU). feral is CPU-only; future GPU
  backends would land as new `Par` variants.
- The 50,000-graph corpus + reference-solver action traces. External
  data infrastructure.

## 13. Status

Drafted; not yet implemented. Revisit when ready to start PR 1.
