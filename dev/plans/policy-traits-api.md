# Policy-Traits API Design

**Status:** Proposed
**Author drafted:** April 2026
**Motivation:** Enable fine-grained control over the factorization pipeline (ordering, scaling, pivoting, supernodal grouping, parallel scheduling) for ripopt integration *and* leave a clean path for learned (GNN-based) policies in a future research phase. ripopt is being refactored to align with IPOPT and will replace its current rmumps backend with feral; this is the moment to design the API right.

## 1. Goals and non-goals

**Goals.**

1. Each pipeline stage (ordering, scaling, pivoting, supernodal grouping, parallel scheduling) becomes a *trait* whose default impls reproduce feral's current behavior bit-for-bit.
2. The new traits are the canonical extension point. The existing `OrderingMethod` / `ScalingStrategy` enums become thin convenience constructors that produce trait objects internally — one code path, not two.
3. ripopt can drive feral by passing trait objects, and the same API surface accepts a future GNN-based policy with no further core changes.
4. Default behavior unchanged: `Solver::new()` still produces the current Phase 2 default pipeline.
5. Backwards-source-compatible *for any caller that constructs enum variants and passes them in* (no caller currently `match`es on these enums externally).

**Non-goals.**

- We do not redesign `CscMatrix`, `CscPattern`, `SymbolicFactorization`, or the multifrontal kernels.
- We do not introduce per-pivot dynamic dispatch in the inner factor loop. Pivoting policy operates at the *frontal* granularity, not the per-pivot granularity.
- We do not implement any GNN policies in this work — only the trait shape they will plug into.
- We do not yet expose `ParallelScheduler` as a fully pluggable trait; we wrap the existing rayon driver behind a trait-shaped seam so a future replacement is non-breaking.

## 2. The five traits

All traits are `Send + Sync` so they can be shared across worker threads in parallel benchmarking and IPM workloads.

### 2.1 `OrderingPolicy`

```rust
pub trait OrderingPolicy: Send + Sync {
    fn order(&self, pattern: &CscPattern) -> Result<OrderingResult, OrderingError>;
    fn name(&self) -> &str;
}

pub struct OrderingResult {
    pub permutation: Vec<i32>,
    pub stats: OrderingStats,
}
```

- `OrderingResult` reuses the existing `OrderingStats` from `feral-ordering-core`.
- `permutation` is `Vec<i32>` to match the existing in-tree contract; an internal helper converts to/from `Vec<usize>`.

**Adapters provided.**

- `AmdPolicy` — wraps `feral_amd::amd_order`.
- `MetisPolicy` — wraps `feral_metis::metis_order`.
- `ScotchPolicy` — wraps `feral_scotch::scotch_order`.
- `KahipPolicy` — wraps `feral_kahip::kahip_order`.
- `AutoOrderingPolicy { sub_policies, router }` — embeds the existing `choose_adaptive` logic, dispatches to one of the sub-policies. This is what `OrderingMethod::Auto` becomes internally.

**Future learned impl (sketch only):**

```rust
pub struct GnnOrderingPolicy {
    model: Arc<dyn GnnInference>,
    fallback: Box<dyn OrderingPolicy>,
    confidence_threshold: f32,
}
// Predict; if confidence < threshold, fall back. Ships safely.
```

### 2.2 `ScalingPolicy`

```rust
pub trait ScalingPolicy: Send + Sync {
    fn scale(&self, matrix: &CscMatrix) -> Result<ScalingResult, ScalingError>;
    fn name(&self) -> &str;
}

pub struct ScalingResult {
    pub scale: Vec<f64>,
    pub info: ScalingInfo,
}
```

The existing `ScalingCache` (used by MC64 across IPM iterations) is hidden inside the policy via interior mutability (`Mutex<Option<Mc64Cache>>`). External users see a stateless interface; the policy itself manages cross-call reuse.

**Adapters provided.**

- `IdentityScalingPolicy` (no-op).
- `InfNormScalingPolicy` (Knight–Ruiz ∞-norm).
- `Mc64ScalingPolicy` (matching-based; owns the cache).
- `ExternalScalingPolicy { scale: Vec<f64> }` (user-supplied vector).
- `AutoScalingPolicy { sub_policies, router }` — embeds the current adaptive routing.

### 2.3 `PivotPolicy`

```rust
pub trait PivotPolicy: Send + Sync {
    fn params_for_frontal(&self, ctx: &FrontalContext) -> BunchKaufmanParams;
    fn name(&self) -> &str;
}

pub struct FrontalContext {
    pub size: usize,
    pub depth_in_etree: u32,
    pub estimated_max_growth: Option<f64>,
    pub is_root: bool,
}
```

**Why frontal-granularity, not per-pivot.** The current `factor_frontal` consumes a single `BunchKaufmanParams` for the whole frontal. Per-pivot policy would require deep refactoring of the inner loop and adds dispatch cost at the hottest code path. Frontal-level dispatch lets a learned policy pick (α, threshold, regularization, on_zero_pivot action) per supernode — which is where the variation that matters in practice lives — without touching the dense kernel.

**Adapters provided.**

- `StaticPivotPolicy { params: BunchKaufmanParams }` — returns the same params for every frontal. This is the current behavior. Default.
- `AdaptivePivotPolicy` — small built-in heuristic that loosens the threshold for tall-skinny frontals. Stub for now.

**Path to delayed pivoting (Phase 2.3).** The `PivotPolicy` trait is the natural seat for delayed-pivot decisions; we add a second method when Phase 2.3 lands:

```rust
fn on_delayed_pivot(&self, ctx: &DelayedPivotContext) -> DelayedPivotAction;
```

…with a default impl that preserves the current `ForceAccept` behavior. This is forward-compatible.

### 2.4 `SupernodeStrategy`

```rust
pub trait SupernodeStrategy: Send + Sync {
    fn amalgamate(
        &self,
        etree: &EliminationTree,
        col_counts: &[usize],
    ) -> Vec<Supernode>;

    fn preprocess(
        &self,
        matrix: &CscMatrix,
    ) -> OrderingPreprocess;

    fn name(&self) -> &str;
}
```

Two responsibilities: the amalgamation rule (currently SSIDS-nemin) and the LDLᵀ-aware preprocess decision (currently the `OrderingPreprocess` enum). Bundled because both decisions are about supernode structure and they interact.

**Adapters provided.**

- `SsidsSupernodeStrategy { nemin: usize, preprocess: OrderingPreprocess }` — current behavior, default `nemin = 32`, `preprocess = None`.
- `AutoSupernodeStrategy` — wraps the existing `pick_ordering_preprocess` heuristic.

### 2.5 `ParallelScheduler` (seam, not full extension point)

```rust
pub trait ParallelScheduler: Send + Sync {
    fn execute_assembly_tree<F>(
        &self,
        plan: &AssemblyPlan,
        per_node: F,
    ) -> Result<(), FeralError>
    where
        F: Fn(usize) -> Result<(), FeralError> + Send + Sync;

    fn name(&self) -> &str;
}
```

We do not promise extensibility here yet — the trait exists so the rayon driver is behind a named seam, not so external users plug in alternative schedulers. A future `MpiScheduler` or `GpuScheduler` would impl this trait, but Phase 2 ships with one impl: `RayonScheduler` (current behavior). Marked `#[doc(hidden)]` for now to signal that the interface may change.

## 3. The `Solver` builder

```rust
let solver = Solver::builder()
    .ordering(AmdPolicy::default())
    .scaling(Mc64ScalingPolicy::default())
    .pivot(StaticPivotPolicy::default())
    .supernode(SsidsSupernodeStrategy::default())
    .scheduler(RayonScheduler::default())
    .build();
```

Each builder method takes `impl OrderingPolicy + 'static` etc., boxes internally. Defaults reproduce the current `Solver::new()` exactly. `Solver::new()` itself becomes:

```rust
impl Solver {
    pub fn new() -> Self {
        Self::builder().build()
    }
}
```

For users who prefer the enum API, the existing `Solver::with_params(NumericParams { … })` constructor is retained and translated internally to a builder call. This keeps every existing test green.

## 4. Internal dispatch unification

The existing match statements in `symbolic/mod.rs:306–311`, `scaling/mod.rs:177–195`, and `symbolic/mod.rs:372–399` are replaced by a single trait method call each. Concretely:

- `run_external_ordering` becomes a thin wrapper that calls `policy.order(&pattern)`. The enum-to-policy translation happens once, at builder time.
- `compute_scaling` becomes `policy.scale(matrix)`.
- The supernode/preprocess match becomes two trait method calls.

This is the win the Explore report flagged: one canonical code path, no parallel enum-and-trait dispatch.

## 5. Migration plan

**Phase A — additive trait introduction (this plan).** Land the five traits, the adapters, and the builder. Keep `OrderingMethod`, `ScalingStrategy`, `BunchKaufmanParams`, `SupernodeParams`, `OrderingPreprocess` as public enums/structs that the builder accepts and translates internally. Ship per-PR as in §6. Do not touch ripopt yet.

**Phase B — ripopt migration.** Once Phase A is on `main`, refactor ripopt's solver-facing layer to call `feral::Solver::builder()` with trait objects directly. Retire rmumps. ripopt's IPOPT-aligned refactor pulls in feral's policy hooks for fine-grained control.

**Phase C — enum cleanup (later).** Once no in-tree caller relies on the enum constructors, deprecate them and move to trait-only at the public API. Optional; not blocking.

**Phase D — learned policies (Phase 5 of the KKT research program).** A `GnnOrderingPolicy`, `GnnScalingPolicy`, `GnnPivotPolicy` are added as new trait impls in their own crate (`feral-learned`?). They plug into the same `Solver::builder()` API with no core changes.

## 6. Implementation PR sequence

Each PR is independently mergeable, ships its own tests, and leaves `main` working.

1. **PR 1: `OrderingPolicy` trait + adapters** (~2 days)
   - Define trait in `feral-ordering-core` (the existing shared-contract crate).
   - Implement `AmdPolicy`, `MetisPolicy`, `ScotchPolicy`, `KahipPolicy`, `AutoOrderingPolicy`.
   - Replace the match in `symbolic/mod.rs:306–311` with a single trait call.
   - Test: existing ordering tests must pass byte-for-byte.

2. **PR 2: `ScalingPolicy` trait + adapters** (~2 days)
   - Define trait in `scaling/policy.rs`.
   - Adapters for the five existing strategies.
   - Replace the match in `scaling/mod.rs:177–195`.
   - Test: existing scaling tests + IPM-iteration MC64 caching test.

3. **PR 3: `Solver` builder API** (~1 day)
   - Add `Solver::builder()` with `.ordering(...)` and `.scaling(...)` methods.
   - Translate `OrderingMethod` and `ScalingStrategy` enum variants to trait-object adapters internally.
   - Existing `Solver::new()` and `Solver::with_params(…)` unchanged behaviorally.

4. **PR 4: `PivotPolicy` trait + frontal-context plumbing** (~3 days)
   - Define trait + `FrontalContext`.
   - Implement `StaticPivotPolicy`.
   - Refactor `factor_frontal` to call `policy.params_for_frontal(ctx)` once per frontal.
   - Add `Solver::builder().pivot(...)`.
   - Test: existing pivoting tests + a synthetic test where pivot params vary by frontal depth.

5. **PR 5: `SupernodeStrategy` trait** (~2 days)
   - Define trait.
   - Implement `SsidsSupernodeStrategy`, `AutoSupernodeStrategy`.
   - Replace match at `symbolic/mod.rs:372–399`.
   - Add `Solver::builder().supernode(...)`.

6. **PR 6: `ParallelScheduler` seam** (~1 day)
   - Define trait, implement `RayonScheduler` (current behavior).
   - Mark trait `#[doc(hidden)]`.
   - Wire through builder.

Total: ~11 working days, 6 PRs. Mergeable in any order after PR 1 lands the shared shape.

## 7. Test strategy

Every PR includes a *parity test*: the same matrix factorized via the old enum API and via the new trait API must produce identical permutations / scalings / inertia / final residuals. This is structural assurance that the refactor is non-behavioral.

The existing consensus-vs-MUMPS framework keeps running on every PR. The 99.97% inertia agreement target on n ≤ 500 is the regression gate.

## 8. Path to GNN policies (forward-compatibility check)

For each trait, a sketch of the future learned impl. None of these are implemented in this plan; they exist to verify the trait shapes are right.

```rust
// Ordering: GNN predicts a permutation; falls back to METIS if confidence is low.
pub struct GnnOrderingPolicy { model, fallback, threshold }
impl OrderingPolicy for GnnOrderingPolicy { /* … */ }

// Scaling: learn equilibration on top of MC64.
pub struct GnnScalingPolicy { model, base: Mc64ScalingPolicy }
impl ScalingPolicy for GnnScalingPolicy { /* … */ }

// Pivot: per-frontal threshold prediction.
pub struct GnnPivotPolicy { model }
impl PivotPolicy for GnnPivotPolicy { /* params_for_frontal returns model.predict(ctx) */ }

// Supernode: learn merge thresholds from elimination-tree shape.
pub struct GnnSupernodeStrategy { model }
impl SupernodeStrategy for GnnSupernodeStrategy { /* … */ }
```

The `FrontalContext` struct in `PivotPolicy` is the input feature vector for the GNN at pivoting time — that's why it carries `depth_in_etree`, `size`, and `estimated_max_growth` rather than just the frontal data alone. Designed to be extensible (more fields can be added without breaking trait impls — old impls just ignore the new fields).

The fallback-on-low-confidence pattern in each learned impl makes shipping safe: a poorly-trained or out-of-distribution GNN degrades to the production heuristic, never below it.

## 9. Open questions

1. Should `OrderingPolicy::order` see the matrix values, not just the pattern? AMD/METIS only need the pattern, but a GNN policy might benefit from numerical magnitudes. **Decision for now:** pattern-only. Add a second method `order_with_values(matrix)` later if needed (default-routes to `order(matrix.pattern())`).

2. Should `PivotPolicy` and `SupernodeStrategy` be one trait? They're related (supernode structure affects pivoting). **Decision for now:** keep separate, since they're invoked at different stages. Revisit if the dependency between them becomes painful.

3. Should the trait crate be a separate `feral-policy` crate, or live in `feral-ordering-core`? **Decision for now:** put `OrderingPolicy` in `feral-ordering-core` (it's already the shared-contract crate) and put `ScalingPolicy`, `PivotPolicy`, `SupernodeStrategy`, `ParallelScheduler` in the main `feral` crate alongside their existing implementations. Promote to `feral-policy` if a third party wants to ship a policy crate without depending on full feral.

4. Naming: `OrderingPolicy` vs. `Orderer` vs. `OrderingStrategy`? **Decision for now:** `*Policy` everywhere, since "policy" is the term used in the IPM/ML research framing this is meant to support.
