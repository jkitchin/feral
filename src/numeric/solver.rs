//! Stateful linear-solver handle (`Solver`) for the POUNCE IPM
//! integration. Mirrors Ipopt's `SymLinearSolver` contract: factor →
//! check inertia → escalate quality → re-factor.
//!
//! The free functions in `factorize` / `solve` remain the primary
//! entry points; this is a thin coordinator that owns persistent
//! quality-escalation state and a cached `SymbolicFactorization`
//! for refactor-on-same-pattern reuse.
//!
//! See `dev/research/pounce-integration-interface.md` and
//! `dev/plans/pounce-integration-interface.md`.

use crate::error::FeralError;
use crate::inertia::Inertia;
use crate::numeric::condition::estimate_condition_1norm;
use crate::numeric::factorize::{
    factorize_multifrontal_parallel_with_workspace, factorize_multifrontal_with_workspace,
    FactorWorkspace, NumericParams, SparseFactors,
};
use crate::numeric::solve::{solve_sparse, solve_sparse_many, solve_sparse_refined};
use crate::scaling::{
    mc64_value_bound_passes, precompute_mc64_validity, Mc64CacheValidity, ScalingStrategy,
};
use crate::sparse::csc::CscMatrix;
use crate::symbolic::supernode::SupernodeParams;
use crate::symbolic::{symbolic_factorize_with_method, OrderingMethod, SymbolicFactorization};

/// Result of a single `Solver::factor` attempt.
#[derive(Debug)]
pub enum FactorStatus {
    /// Factorization succeeded. If `check_inertia` was supplied, the
    /// actual inertia matched.
    Success,
    /// Numerically singular: factor encountered a zero pivot under
    /// `ZeroPivotAction::Fail`, or scaling reported `PartialSingular`.
    Singular,
    /// Inertia was checked and disagreed with the expected count.
    /// The factor is still stored — `solve()` may proceed.
    WrongInertia { actual: Inertia, expected: Inertia },
    /// Unrecoverable error (dimension mismatch, alloc failure,
    /// symbolic-analysis failure).
    FatalError(FeralError),
}

/// Quality-escalation state. Mirrors Ipopt's two-stage
/// `IncreaseQuality` (scaling, then pivot threshold).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityLevel {
    /// Factory defaults; no escalation has fired yet.
    Baseline,
    /// Stage-1 fired: scaling flipped from `Identity` to `InfNorm`.
    ScalingEnabled,
    /// Stage-2 fired one or more times: pivot threshold raised.
    PivotRaised,
    /// Both stages exhausted; `pivot_threshold` is at `pivtol_max`.
    Exhausted,
}

/// Structural fingerprint used to detect when the cached
/// `SymbolicFactorization` is stale. Two genuinely identical
/// patterns produce the same fingerprint by construction; the
/// `structural_hash` field hashes both `col_ptr` and `row_idx`
/// so two matrices that share `n` and `nnz` but differ in
/// per-column degree distribution or per-column row indices
/// fingerprint differently.
///
/// Hash collisions between distinct patterns are mathematically
/// possible but cryptographically improbable (`u64` SipHash via
/// `DefaultHasher`). The IPM use case never relies on this:
/// successive iterates have *byte-identical* `col_ptr` / `row_idx`,
/// so the equality test fires before any hash collision could
/// matter. The structural hash is a defensive measure for
/// general callers who might hand `Solver` two structurally
/// distinct matrices that happen to share `(n, nnz)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PatternFingerprint {
    n: usize,
    nnz: usize,
    structural_hash: u64,
}

impl PatternFingerprint {
    fn of(matrix: &CscMatrix) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        matrix.col_ptr.hash(&mut h);
        matrix.row_idx.hash(&mut h);
        Self {
            n: matrix.n,
            nnz: matrix.row_idx.len(),
            structural_hash: h.finish(),
        }
    }
}

/// Solver-scope value-bounded MC64 scaling cache (Track B2).
///
/// An IPM driver replays `factor()` on a bit-identical sparsity
/// pattern with drifting values. The MC64 Hungarian that produces
/// the scaling vector is pattern-dominated, so the iter-0 scaling
/// `D₀` is usually still good for iter N. This cache stores `D₀`
/// plus the baseline diagonal-dominance fingerprint; on each warm
/// `factor()` `mc64_value_bound_passes` decides reuse-vs-recompute
/// in O(nnz) without rerunning the Hungarian (~4 s on
/// `rocket_12800`). See `dev/plans/mc64-value-bounded-cache.md`.
///
/// Distinct from `SymbolicFactorization::cached_mc64`, the one-shot
/// post-symbolic cache that is still cleared after every factor for
/// issue #38 — this lives at `Solver` scope and is gated, so reusing
/// it cannot reintroduce the #38 inertia drift.
struct Mc64ScalingCache {
    /// Pattern this cache was built for. Reuse is rejected unless the
    /// current matrix fingerprints identically.
    fingerprint: PatternFingerprint,
    /// User-order MC64 scaling vector `D₀` from the cache-baseline
    /// factor. Injected as `ScalingStrategy::External` on a hit.
    scaling: Vec<f64>,
    /// Baseline diagonal-dominance fingerprint for the value bound.
    validity: Mc64CacheValidity,
}

/// Stateful linear-solver handle. Mirrors Ipopt `SymLinearSolver`.
///
/// Owns quality-escalation state and a cached `SymbolicFactorization`
/// so repeated `factor()` calls on structurally identical matrices
/// reuse the symbolic analysis. The β refactor (scaling moved from
/// symbolic to numeric phase) makes this cache reuse correct even
/// across stage-1 quality escalation.
pub struct Solver {
    numeric_params: NumericParams,
    snode_params: SupernodeParams,
    pivtol_max: f64,
    quality_level: QualityLevel,
    last_symbolic: Option<SymbolicFactorization>,
    last_factors: Option<SparseFactors>,
    last_inertia: Option<Inertia>,
    last_pattern_fingerprint: Option<PatternFingerprint>,
    /// Diagnostic counter: number of times `symbolic_factorize` was
    /// called from this `Solver`. Used by integration tests to
    /// verify the cache-reuse property and by future telemetry.
    symbolic_call_count: usize,
    /// Diagnostic counter: number of `factor()` calls whose
    /// resulting `ScalingInfo` was `Mc64FallbackToInfnorm`. Issue #24:
    /// surfaces the previously-silent MC64 → InfNorm fallback so
    /// IPM drivers can warn / log a structured diagnostic without
    /// reaching into per-factor state. Increments once per
    /// `factor()` that fell back; resets on `Solver::new()`.
    mc64_fallback_count: usize,
    /// Pooled scratch for the numeric phase. Retained across
    /// `factor` calls so IPM-style re-factorizations (same
    /// pattern, new values; or bumped pivot threshold) do not
    /// re-allocate per-supernode buffers. Cleared to a
    /// well-defined initial state on every
    /// `factorize_multifrontal_with_workspace` entry, so stale
    /// data cannot leak between factor attempts.
    workspace: FactorWorkspace,
    /// Route `factor()` through the rayon-parallel multifrontal
    /// driver when `true`. Default `true`. The parallel driver is
    /// bit-exact with the sequential supernodal driver and falls
    /// through to the sequential path via
    /// `should_parallelize_assembly` when the supernode count is
    /// below `N_PAR_MIN`, so default-on does not regress small-
    /// problem latency. See issue #7.
    use_parallel: bool,
    /// Lazily-built rayon `ThreadPool` reused across every
    /// `factor()` call that dispatches the parallel multifrontal
    /// driver. `None` until first parallel-fire; once built, the
    /// pool's worker threads persist for the `Solver`'s lifetime
    /// (or until `Drop`).
    ///
    /// Reusing the pool eliminates the per-call thread-spawn and
    /// initial cv-wait-wakeup cost that issue #19 flagged at 53%
    /// sys time on `robot_1600`. Once warm, the workers stay
    /// parked between calls and resume on the next `install`
    /// without re-entering the kernel scheduler for each new task.
    ///
    /// Thread count: matches `rayon::current_num_threads()` at
    /// first build (honors `RAYON_NUM_THREADS` and the default
    /// `num_cpus`). Wrapped in `Arc` so `install` can run inside
    /// the same `&mut self` borrow that touches `numeric_params`
    /// and `workspace` — `install` only needs `&ThreadPool`.
    parallel_pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    /// Fill-reducing ordering method passed to every
    /// `symbolic_factorize_with_method` call this `Solver` issues.
    /// Default `OrderingMethod::Auto` — the same dispatcher
    /// `symbolic_factorize` uses, so unconfigured `Solver`s match
    /// the free-function default exactly. Library consumers override
    /// via `with_ordering` to experimentally try ND on banded /
    /// 1D-banded KKTs (#33 suggested action §3, supernode-shape
    /// thesis).
    ordering: OrderingMethod,
    /// Warm cascade-break auto-arm threshold. `Some(β)` means: at the
    /// start of `factor()`, if the previous factor on this pattern
    /// generated a supernode with `n_delayed_in >= β·n`, locally set
    /// `cascade_break_ratio = 0.5` and `cascade_break_eps = 1e-10`
    /// for this call. `None` (default) disables the auto-arm.
    ///
    /// Symbolic-time auto-arm was disproved on 2026-05-17 (see
    /// `dev/research/warm-state-cascade-amplification-2026-05-17.md`
    /// step 3): the cascade is a runtime property of delayed pivots,
    /// not a pattern property. Warm auto-arm pays the cost on the
    /// first factor and rescues iter 2+ on the same pattern.
    auto_cascade_break_beta: Option<f64>,
    /// Maximum per-supernode `n_delayed_in` observed in the previous
    /// `factor()` call. Reset to `None` on pattern change.
    prev_max_n_delayed_in: Option<usize>,
    /// Latch: once the warm auto-arm has fired on this pattern,
    /// stay armed for every subsequent factor on the same pattern.
    /// Reset to `false` on pattern change. Without latching the
    /// trigger fires once, suppresses the cascade so the next factor
    /// sees small delays, and disarms — letting a fresh cascade hit
    /// next iter. With latching we pay the CB cost permanently after
    /// the first observed cascade rather than oscillating.
    auto_arm_latched: bool,
    /// Track B2 master switch for the value-bounded MC64 scaling
    /// cache. Default `true`; flip off with `with_mc64_cache(false)`
    /// for tests/probes that need every factor to recompute scaling
    /// from scratch. When `false`, `mc64_scaling_cache` stays `None`.
    mc64_cache_enabled: bool,
    /// Solver-scope value-bounded MC64 scaling cache. `Some` only
    /// after a factor on the current pattern produced an
    /// `ScalingInfo::Applied` MC64 scaling. Cleared on pattern
    /// change. See [`Mc64ScalingCache`].
    mc64_scaling_cache: Option<Mc64ScalingCache>,
    /// Diagnostic counter: number of `factor()` calls that reused
    /// the value-bounded MC64 scaling cache. Exposed via
    /// [`Solver::mc64_cache_hit_count`].
    mc64_cache_hit_count: usize,
}

impl Solver {
    /// Construct a `Solver` with default `NumericParams` and
    /// `SupernodeParams`, MA27-style `pivtol_max = 0.5`.
    pub fn new() -> Self {
        Self::with_params(NumericParams::default(), SupernodeParams::default())
    }

    /// Construct a `Solver` with explicit parameters.
    pub fn with_params(np: NumericParams, sn: SupernodeParams) -> Self {
        Self {
            numeric_params: np,
            snode_params: sn,
            pivtol_max: 0.5,
            quality_level: QualityLevel::Baseline,
            last_symbolic: None,
            last_factors: None,
            last_inertia: None,
            last_pattern_fingerprint: None,
            symbolic_call_count: 0,
            mc64_fallback_count: 0,
            workspace: FactorWorkspace::new(),
            use_parallel: true,
            parallel_pool: None,
            ordering: OrderingMethod::Auto,
            auto_cascade_break_beta: None,
            prev_max_n_delayed_in: None,
            auto_arm_latched: false,
            mc64_cache_enabled: true,
            mc64_scaling_cache: None,
            mc64_cache_hit_count: 0,
        }
    }

    /// Build (lazily) the rayon `ThreadPool` that the parallel
    /// multifrontal driver will execute inside, and return an
    /// `Arc` to it. The pool is constructed on the first call and
    /// reused on every subsequent one — see the field doc on
    /// `parallel_pool` for the issue #19 motivation.
    ///
    /// On `ThreadPoolBuilder::build` failure the caller is given
    /// `None` so the dispatcher can fall through to the global
    /// rayon pool. In practice the builder only fails on bad
    /// `num_threads` configuration (e.g. zero) which we never
    /// pass.
    fn ensure_parallel_pool(&mut self) -> Option<std::sync::Arc<rayon::ThreadPool>> {
        if self.parallel_pool.is_none() {
            let n = rayon::current_num_threads().max(1);
            match rayon::ThreadPoolBuilder::new().num_threads(n).build() {
                Ok(pool) => {
                    self.parallel_pool = Some(std::sync::Arc::new(pool));
                }
                Err(_) => return None,
            }
        }
        self.parallel_pool.as_ref().map(std::sync::Arc::clone)
    }

    /// Toggle the rayon-parallel multifrontal driver. Default is
    /// `true`; pass `false` to force the sequential supernodal
    /// driver (useful for determinism studies or single-threaded
    /// benchmarks). The two drivers are bit-exact equal on every
    /// supernode — flipping this only affects scheduling, not
    /// numerics.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.use_parallel = parallel;
        self
    }

    /// Toggle the FMA opt-in dispatch on dense trailing-update and
    /// panel-update kernels. Default `false` keeps the bit-exact
    /// `*_nofma` path; pass `true` to dispatch through the FMA
    /// siblings (`schur_panel_minus_fma_strided*`,
    /// `axpy_minus_unroll4`, `axpy2_minus_unroll4`) for ~2x arithmetic
    /// throughput on aarch64 NEON and x86 V3 AVX2+FMA. Trade-off
    /// detailed on [`NumericParams::fma`].
    pub fn with_fma(mut self, fma: bool) -> Self {
        self.numeric_params.fma = fma;
        self
    }

    /// Disable delayed pivoting. When `on = true`, every supernode
    /// runs as if it were the root: pivots failing the BK threshold
    /// or 2×2 Duff–Reid test are force-accepted in place via
    /// `ZeroPivotAction::ForceAccept` rather than being delayed up
    /// to the parent. FERAL analogue of MA57's `cntl[4]` static-
    /// pivoting fallback. Use only when iterative refinement is
    /// available to recover the lost accuracy on accepted small
    /// pivots — appropriate for IPM KKT systems where outer
    /// regularization (δ_c, δ_x) and refinement absorb the residual.
    pub fn with_static_pivoting(mut self, on: bool) -> Self {
        self.numeric_params.allow_delayed_pivots = !on;
        self
    }

    /// Adaptive cascade-break trigger. At any non-root supernode
    /// whose front is at least `ratio` delayed columns from below
    /// (`n_delayed_in / expanded_ncol >= ratio`), flip that one
    /// supernode to `may_delay = false` with a locally-overridden
    /// `ZeroPivotAction::ForceAccept` policy. Light-delay nodes
    /// keep SSIDS-style delayed pivoting; heavy-delay nodes absorb
    /// the perturbation rather than passing 10^4–10^5 delays into
    /// the root front.
    ///
    /// Pass `0.5` for a starting threshold ("front is at least 50%
    /// delays"). Set to `0.0` to force-break at any non-root node
    /// that received any delay (equivalent to `with_static_pivoting`
    /// except the root still uses the configured `on_zero_pivot`).
    /// Pass `None` semantics by not calling this builder.
    pub fn with_cascade_break(mut self, ratio: f64) -> Self {
        self.numeric_params.cascade_break_ratio = Some(ratio);
        self
    }

    /// Warm auto-arm for cascade-break. With `Some(β)`, the next
    /// `factor()` call on the same pattern after a factor that
    /// produced a supernode with `n_delayed_in ≥ β·n` locally arms
    /// `cascade_break_ratio = 0.5` and `cascade_break_eps = 1e-10`
    /// for the duration of that call only. `numeric_params` is not
    /// mutated; the user-set `cascade_break_*` knobs (if any) take
    /// precedence and disable the auto-arm.
    ///
    /// Recommended `β` ≈ 0.05 — 5% of `n` as `n_delayed_in` at any
    /// supernode means the cascade is well underway. Below ~0.02
    /// the trigger fires on benign light-delay nodes.
    ///
    /// Symbolic-time auto-arm was disproved on 2026-05-17 (cascade
    /// is a runtime property; see
    /// `dev/research/warm-state-cascade-amplification-2026-05-17.md`).
    /// Warm auto-arm is the cheapest viable single-shot rescue: the
    /// first factor pays the cascade cost, iter 2+ rides the rescue
    /// without user intervention.
    pub fn with_auto_cascade_break(mut self, beta: f64) -> Self {
        self.auto_cascade_break_beta = Some(beta);
        self
    }

    /// Toggle the value-bounded MC64 scaling cache (Track B2).
    /// Default `true`.
    ///
    /// When on, a `factor()` that runs the MC64 Hungarian
    /// (`ScalingInfo::Applied`) caches the resulting scaling vector
    /// at `Solver` scope. Subsequent `factor()` calls on the same
    /// sparsity pattern reuse it — skipping the Hungarian, ~4 s on
    /// `rocket_12800` — whenever an O(nnz) value-bound check confirms
    /// the current matrix's values have not drifted past the
    /// matching's diagonal-dominance guarantee. On drift the check
    /// rejects and the Hungarian reruns fresh (the pre-B2 behaviour).
    ///
    /// Pass `false` for tests and probes that must observe a fresh
    /// MC64 on every factor; this also keeps `mc64_scaling_cache`
    /// permanently `None`. See `dev/plans/mc64-value-bounded-cache.md`.
    pub fn with_mc64_cache(mut self, on: bool) -> Self {
        self.mc64_cache_enabled = on;
        if !on {
            self.mc64_scaling_cache = None;
        }
        self
    }

    /// Enable static-pivot perturbation for cascade-break supernodes.
    /// At triggered supernodes the BK policy switches to
    /// `ZeroPivotAction::PerturbToEps { abs_floor: eps }`: each
    /// rejected pivot becomes `sign(d) * max(|d|, eps)` and is
    /// counted by sign rather than absorbed as zero.
    ///
    /// The factor satisfies `L · D · L^T = A + Δ` for the L and D
    /// produced, but `Δ` is *not* bounded by `eps`. See the
    /// `ZeroPivotAction::PerturbToEps` docstring and
    /// `dev/research/cascade-break-l-perturbation-2026-05-15.md`
    /// for the correct bound (the implicit Δ flows through the
    /// trailing Schur update and is bounded in `||A||² / eps`).
    /// On IPM KKT matrices the unrefined residual stays small in
    /// practice; iterative refinement against unperturbed `A` is
    /// recommended for tight tolerances.
    ///
    /// Without this knob, cascade-break uses the legacy unbounded
    /// `ForceAccept` (zero-out L and D at the rejected pivot;
    /// inertia is not preserved in general). The 0.94–0.95 ratio
    /// sweet spot reported on `pinene_3200_0009` is matrix-specific
    /// — see `dev/journal/2026-05-13-03.org` §01:15.
    ///
    /// Recommended `eps`: `1e-8` to `1e-10` for KKT systems with
    /// `||A||_∞` in the `O(1)–O(10³)` range typical of IPM iterates.
    /// Callers should multiply by an estimate of `||A||_∞` for
    /// non-normalized matrices.
    pub fn with_cascade_break_eps(mut self, eps: f64) -> Self {
        self.numeric_params.cascade_break_eps = Some(eps);
        self
    }

    /// Enable MA57-style static-pivot perturbation (issue #38). On
    /// every `factor()` call the solver computes `||A||_∞` once and
    /// propagates an absolute floor
    /// `static_pivot_floor = t * ||A||_∞` into the BK pivot kernels.
    /// Every accepted 1×1 / 2×2 pivot whose magnitude (for 2×2:
    /// smallest |eigenvalue|) is below the floor is perturbed up to
    /// the floor and counted by sign. The factor satisfies
    /// `LDL^T = A + Δ` with `||Δ||_F ≤ floor` per perturbed pivot.
    ///
    /// Inertia is then reported for the perturbed `A + Δ`, not `A`.
    /// On rocket_12800 iter 1 the true matrix has 38402 negative
    /// eigenvalues but the IPM expects 38400; with `t = 1e-8` and
    /// `||A||_∞ ≈ 25`, the floor of `2.5e-7` bends two near-zero
    /// negative pivots over into the positive bucket, matching the
    /// IPM expectation and cutting ipopt-feral's δ_w escalation
    /// retries.
    ///
    /// Default `None` (disabled). Recommended starting value: `1e-8`
    /// (matches MA57's `cntl[0]` default in Ipopt). Iterative
    /// refinement against unperturbed `A` recovers solve accuracy.
    ///
    /// See `dev/research/static-pivot-perturbation-2026-05-17.md`
    /// and the C ABI's `FERAL_STATIC_PIVOT` env var.
    pub fn with_static_pivot_threshold(mut self, t: f64) -> Self {
        self.numeric_params.static_pivot_threshold = Some(t);
        self
    }

    /// Opt into the MC64 partial-singular stderr breadcrumb. When
    /// `on = true`, each `factor()` whose scaling phase leaves
    /// variables unmatched (`ScalingInfo::PartialSingular`) emits a
    /// one-line `warning:` to stderr.
    ///
    /// Default `false`. `PartialSingular` is routine and benign for
    /// IPM hosts, which factorize structurally rank-deficient KKT
    /// systems on the first attempt of most iterations; an
    /// unconditional stderr write floods host logs for behavior that
    /// is expected and recovered downstream. The same fact is always
    /// available structurally via [`Solver::scaling_info`] (and as a
    /// count via [`Solver::mc64_fallback_count`] for the `Auto`-
    /// fallback case), so this toggle is a diagnostic convenience,
    /// not a correctness signal. The `FERAL_WARN_PARTIAL_SINGULAR`
    /// env var sets it for the C ABI. Issue #43.
    pub fn with_partial_singular_warning(mut self, on: bool) -> Self {
        self.numeric_params.warn_partial_singular = on;
        self
    }

    /// Enable the symmetric-quasi-definite (SQD) fast-path. When
    /// `on = true`, the caller asserts the input KKT has Vanderbei
    /// (1995) structure `K = [[-E, A^T], [A, F]]` with `E, F` SPD —
    /// the common case in IPOPT after the first inertia correction
    /// sets `δ_w, δ_c > 0`, and structural in IP-PMM
    /// (Pougkakiotis-Gondzio 2020). Under this contract every
    /// symmetric permutation admits an `LDL^T` with diagonal `D`
    /// (Vanderbei Thm 2.1), so the per-supernode Bunch-Kaufman
    /// 1x1-vs-2x2 search is skipped entirely.
    ///
    /// Default `off`. Mutually exclusive with delayed pivoting and
    /// cascade-break: enabling SQD also clears
    /// `allow_delayed_pivots` (no delayed pivots — every supernode
    /// runs root-style with diagonal-only kernel) and
    /// `cascade_break_ratio` (no cascade absorption — diagonal-only
    /// kernel has no rejected pivots to absorb). The
    /// `cascade_break_eps` knob is also cleared for the same reason.
    /// Disabling SQD (`on = false`) restores neither: callers that
    /// want delayed pivoting back must re-enable explicitly.
    ///
    /// Contract violations at runtime surface as
    /// `FeralError::SqdContractViolated { column, pivot }` — loud
    /// failure, never silent BK fallback. See
    /// `dev/research/sqd-fast-path.md`, `dev/decisions.md`
    /// 2026-05-16 entry, and issue #34.
    pub fn with_sqd_mode(mut self, on: bool) -> Self {
        self.numeric_params.sqd_mode = on;
        if on {
            self.numeric_params.allow_delayed_pivots = false;
            self.numeric_params.cascade_break_ratio = None;
            self.numeric_params.cascade_break_eps = None;
        }
        self
    }

    /// Override the fill-reducing ordering method used at the next
    /// (and subsequent) symbolic factorization. Default
    /// `OrderingMethod::Auto` matches the free-function
    /// `symbolic_factorize` dispatcher exactly.
    ///
    /// Motivating use case: issue #33's 1D-banded Mittelmann panel
    /// (clnlbeam) bottlenecked at 97% main-thread in the scalar 1×1
    /// pivot path because AMD produces thin supernodes on banded
    /// structure. Nested-dissection orderings (`ScotchND`,
    /// `MetisND`, `KahipND`) tend to produce squarer fronts on such
    /// problems, letting more work batch through the blocked panel
    /// kernel. Library consumers can switch in
    /// `Solver::new().with_ordering(OrderingMethod::ScotchND)` and
    /// re-time to test the supernode-shape thesis without rebuilding.
    ///
    /// Takes effect at the next `factor()` call that triggers a
    /// symbolic re-factorization. A cached symbolic from a prior
    /// `factor()` with a different ordering is not invalidated by
    /// this call alone — pattern-fingerprint mismatch is what
    /// triggers symbolic invalidation. To force a re-symbolic with
    /// the new method on the same pattern, call this *before* the
    /// first `factor()`, or pass a structurally-different matrix.
    pub fn with_ordering(mut self, method: OrderingMethod) -> Self {
        self.ordering = method;
        self
    }

    /// Factor `matrix`. If `check_inertia` is `Some(expected)`,
    /// returns `WrongInertia { actual, expected }` on mismatch
    /// without invalidating the stored factor (caller may still
    /// `solve` against it). See plan §`factor()` flow.
    pub fn factor(&mut self, matrix: &CscMatrix, check_inertia: Option<Inertia>) -> FactorStatus {
        // Step 0: reject non-finite input. A single +∞ / -∞ / NaN
        // entry sends the BK pivot-search loop into pathological
        // behavior (every threshold test fails against the inf
        // column max; pivots cascade indefinitely). Caught upstream:
        // dtoc2 iter 1 was shipping 103 inf entries from IPOPT's
        // δ_w bump that overflowed beyond 1e308 — feral spun until
        // timeout. Fail fast with a clear message so the IPM driver
        // can decide to back off δ_w. O(nnz), allocation-free.
        // See `dev/journal/2026-05-17-01.org` §08:00 (dtoc2 root
        // cause), §11:50 (input-validation fix).
        for (k, &v) in matrix.values.iter().enumerate() {
            if !v.is_finite() {
                return FactorStatus::FatalError(FeralError::InvalidInput(format!(
                    "matrix value at nnz index {k} is non-finite ({v}); \
                     fix the upstream computation before calling factor()"
                )));
            }
        }
        // Step 1: pattern fingerprint.
        let fp = PatternFingerprint::of(matrix);

        // Step 2: invalidate cache on pattern change.
        if self.last_pattern_fingerprint != Some(fp) {
            self.last_symbolic = None;
            self.last_factors = None;
            self.last_inertia = None;
            self.last_pattern_fingerprint = None;
            // Warm auto-arm signal is pattern-bound: a new pattern
            // means the prior n_delayed observation no longer applies.
            self.prev_max_n_delayed_in = None;
            self.auto_arm_latched = false;
            // Track B2: the MC64 scaling cache is keyed on the
            // pattern fingerprint; a new pattern voids it.
            self.mc64_scaling_cache = None;
        }

        // Step 3: ensure symbolic is cached.
        if self.last_symbolic.is_none() {
            match symbolic_factorize_with_method(matrix, &self.snode_params, self.ordering) {
                Ok(sym) => {
                    self.symbolic_call_count += 1;
                    self.last_symbolic = Some(sym);
                    self.last_pattern_fingerprint = Some(fp);
                }
                Err(e) => return FactorStatus::FatalError(e),
            }
        }
        // Step 3.5: ensure the parallel `ThreadPool` is built when
        // we're about to dispatch the parallel driver. Done *before*
        // the immutable `symbolic` borrow below so the mutable borrow
        // on `self.parallel_pool` doesn't collide. The pool persists
        // across `factor()` calls — see the field doc on
        // `parallel_pool` for the issue #19 motivation. The clone is
        // an `Arc::clone` (one atomic refcount bump) — the pool
        // itself stays put.
        let pool = if self.use_parallel {
            self.ensure_parallel_pool()
        } else {
            None
        };

        // Safe: just-set above or already Some.
        let symbolic = match &self.last_symbolic {
            Some(s) => s,
            None => unreachable!("symbolic just populated"),
        };

        // Step 3.6: warm cascade-break auto-arm. If the prior factor
        // on this pattern produced a supernode with n_delayed_in >=
        // β·n, locally arm cascade_break for this call. User-set
        // cascade_break_* takes precedence (auto-arm only fills in
        // when both are None). The effective params is a clone — we
        // never mutate self.numeric_params here so the user's
        // configuration survives across factor() calls.
        let auto_arm_fires = match (
            self.auto_cascade_break_beta,
            self.prev_max_n_delayed_in,
            self.numeric_params.cascade_break_ratio,
        ) {
            (Some(_), _, None) if self.auto_arm_latched => true,
            (Some(beta), Some(prev), None) => {
                let fires = (prev as f64) >= beta * (symbolic.n as f64);
                if fires {
                    self.auto_arm_latched = true;
                }
                fires
            }
            _ => false,
        };
        let effective_params: NumericParams = if auto_arm_fires {
            if std::env::var("FERAL_AUTO_CB_DEBUG").is_ok() {
                eprintln!(
                    "[auto-cb] armed: prev_max_n_delayed_in={} n={} β={:.3}",
                    self.prev_max_n_delayed_in.unwrap_or(0),
                    symbolic.n,
                    self.auto_cascade_break_beta.unwrap_or(0.0),
                );
            }
            let mut p = self.numeric_params.clone();
            p.cascade_break_ratio = Some(0.5);
            if p.cascade_break_eps.is_none() {
                p.cascade_break_eps = Some(1e-10);
            }
            p
        } else {
            if std::env::var("FERAL_AUTO_CB_DEBUG").is_ok()
                && self.auto_cascade_break_beta.is_some()
            {
                eprintln!(
                    "[auto-cb] not armed: prev_max_n_delayed_in={:?} n={} β={:.3}",
                    self.prev_max_n_delayed_in,
                    symbolic.n,
                    self.auto_cascade_break_beta.unwrap_or(0.0),
                );
            }
            self.numeric_params.clone()
        };

        // Step 3.7: issue #38 — MA57-style static-pivot perturbation.
        // When `static_pivot_threshold = Some(t)`, compute `||A||_∞`
        // once (cost: O(nnz)) and propagate the absolute floor
        // `static_pivot_floor = t * ||A||_∞` to the BK params for
        // this factor call. The dense pivot kernels then enforce the
        // floor on every accepted 1×1 / 2×2 pivot. We compute this
        // AFTER effective_params is built so cascade-break auto-arm
        // (which may overwrite BK fields) takes precedence on its
        // own knobs, and AFTER non-finite validation (Step 0 above
        // already ran) so the norm scan is well-defined.
        let mut effective_params = effective_params;
        if let Some(t) = effective_params.static_pivot_threshold {
            if t > 0.0 {
                let norm_inf = matrix_inf_norm(matrix);
                let floor = t * norm_inf;
                if floor.is_finite() && floor > 0.0 {
                    effective_params.bk.static_pivot_floor = floor;
                }
            }
        }

        // Step 3.8: Track B2 — value-bounded MC64 scaling cache. When
        // a prior factor on this exact pattern produced an MC64
        // scaling (`ScalingInfo::Applied`) and the current matrix's
        // values have not drifted past the matching's diagonal-
        // dominance bound, reuse the cached scaling vector instead of
        // rerunning the Hungarian (~4 s on rocket_12800). The reuse
        // is injected as `ScalingStrategy::External`, which the
        // numeric prologue's `compute_scaling_with_cache` resolves in
        // O(n). The value-bound check (`mc64_value_bound_passes`) is
        // the correctness gate that keeps this from reintroducing the
        // #38 iter-0-scaling-on-iter-N inertia drift: on drift it
        // rejects and we fall through to a fresh MC64 — same path as
        // the cache being absent. See
        // `dev/plans/mc64-value-bounded-cache.md`.
        let scaling_cache_hit = if self.mc64_cache_enabled {
            match &self.mc64_scaling_cache {
                Some(c)
                    if c.fingerprint == fp
                        && mc64_value_bound_passes(matrix, &c.scaling, &c.validity) =>
                {
                    effective_params.scaling = ScalingStrategy::External(c.scaling.clone());
                    true
                }
                _ => false,
            }
        } else {
            false
        };
        if scaling_cache_hit {
            self.mc64_cache_hit_count += 1;
        }

        // Step 4: numeric factor via the pooled workspace; map errors.
        // Both drivers share the same signature and a bit-exact
        // contract; pick by the `use_parallel` toggle. When parallel
        // is on and a pool was successfully built, run the
        // dispatcher inside `pool.install(...)` so any `rayon::scope`
        // / `rayon::current_thread_index` calls inside the parallel
        // driver use this pool's workers instead of the global pool.
        // The dispatcher inside
        // `factorize_multifrontal_parallel_with_workspace` may still
        // route to the sequential driver via
        // `should_parallelize_assembly`; in that case `install` is a
        // no-op on the inner code that doesn't touch rayon.
        let result = if self.use_parallel {
            if let Some(p) = pool.as_ref() {
                p.install(|| {
                    factorize_multifrontal_parallel_with_workspace(
                        matrix,
                        symbolic,
                        &effective_params,
                        &mut self.workspace,
                    )
                })
            } else {
                factorize_multifrontal_parallel_with_workspace(
                    matrix,
                    symbolic,
                    &effective_params,
                    &mut self.workspace,
                )
            }
        } else {
            factorize_multifrontal_with_workspace(
                matrix,
                symbolic,
                &effective_params,
                &mut self.workspace,
            )
        };
        // Issue #38: invalidate the one-shot MC64 cache that the
        // symbolic phase populated for the immediately-following
        // numeric reuse. The cache stores the iter-0 Hungarian
        // matching, dual variables, and column maxes; an IPM driver
        // calls `factor()` repeatedly on the same pattern with new
        // values, and reusing iter-0 scaling on iter-N matrix silently
        // corrupts inertia and (eventually) explodes factor cost. See
        // `dev/journal/2026-05-16-30.org` §17:25 and the rocket_12800
        // / pinene_3200 reproducers. The cache stays valid for the
        // *first* numeric call after symbolic (values match by
        // construction); clearing it here means call #2+ falls through
        // to a fresh `compute_symmetric(matrix)` against current values.
        // Cost: one extra MC64 (~100–200 ms on n≈1e5) per warm refactor
        // when scaling resolves to `Mc64Symmetric`; correctness wins.
        if let Some(s) = self.last_symbolic.as_mut() {
            s.cached_mc64 = None;
        }
        match result {
            Ok((factors, inertia)) => {
                // Step 5: stash; PartialSingular maps to Singular.
                let partial_singular = matches!(
                    factors.scaling_info,
                    crate::scaling::ScalingInfo::PartialSingular { .. }
                );
                // Issue #24: bump the MC64-fallback counter so callers
                // can poll a single number to detect "Auto promised
                // matching, actually got InfNorm" without inspecting
                // per-factor state.
                if factors.scaling_info.is_mc64_fallback() {
                    self.mc64_fallback_count += 1;
                }
                // Record the max per-supernode n_delayed_in seen on
                // this factor so the next factor() on the same
                // pattern can warm-arm cascade_break. See
                // `with_auto_cascade_break` for the policy.
                self.prev_max_n_delayed_in = Some(
                    factors
                        .node_factors
                        .iter()
                        .map(|nf| nf.n_delayed_in)
                        .max()
                        .unwrap_or(0),
                );
                // Track B2: maintain the value-bounded MC64 scaling
                // cache. This block runs only on a cache *miss*
                // (`!scaling_cache_hit`); the injected-`External` reuse
                // path keeps its existing entry untouched. On a miss,
                // if this factor ran the MC64 Hungarian to completion
                // (`ScalingInfo::Applied`), install a fresh entry so
                // the next warm factor on this pattern can skip the
                // Hungarian. A cheap InfNorm / fallback / Identity
                // scaling (anything reporting not-`Applied`) is not
                // worth caching — drop the cache so a later genuine
                // MC64 reinstalls it. `PartialSingular` is excluded:
                // a partial matching on a singular matrix is not a
                // scaling worth reusing.
                if self.mc64_cache_enabled && !scaling_cache_hit {
                    if matches!(factors.scaling_info, crate::scaling::ScalingInfo::Applied) {
                        let scaling = factors.scaling.clone();
                        let validity = precompute_mc64_validity(matrix, &scaling);
                        self.mc64_scaling_cache = Some(Mc64ScalingCache {
                            fingerprint: fp,
                            scaling,
                            validity,
                        });
                    } else {
                        self.mc64_scaling_cache = None;
                    }
                }
                self.last_factors = Some(factors);
                self.last_inertia = Some(inertia.clone());
                if partial_singular {
                    FactorStatus::Singular
                } else if let Some(expected) = check_inertia {
                    if inertia == expected {
                        FactorStatus::Success
                    } else {
                        // Keep the factor stored — caller may
                        // still solve() against it. Mirrors Ipopt
                        // SYMSOLVER_WRONG_INERTIA semantics.
                        FactorStatus::WrongInertia {
                            actual: inertia,
                            expected,
                        }
                    }
                } else {
                    FactorStatus::Success
                }
            }
            Err(FeralError::NumericallyRankDeficient) => {
                self.last_factors = None;
                self.last_inertia = None;
                FactorStatus::Singular
            }
            Err(e) => {
                self.last_factors = None;
                self.last_inertia = None;
                FactorStatus::FatalError(e)
            }
        }
    }

    /// Solve `A x = b` against the most recent stored factor.
    /// Returns `FeralError::NoFactor` if no factor is stored.
    /// `WrongInertia` does *not* clear the factor, so this remains
    /// callable in that state (caller's choice).
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, FeralError> {
        match &self.last_factors {
            Some(f) => solve_sparse(f, rhs),
            None => Err(FeralError::NoFactor),
        }
    }

    /// Solve with iterative refinement against the original matrix
    /// and the stored factor. Returns `FeralError::NoFactor` if no
    /// factor is stored.
    pub fn solve_refined(&self, matrix: &CscMatrix, rhs: &[f64]) -> Result<Vec<f64>, FeralError> {
        match &self.last_factors {
            Some(f) => solve_sparse_refined(matrix, f, rhs),
            None => Err(FeralError::NoFactor),
        }
    }

    /// Solve `A · X = B` for `X` against the most recent stored factor,
    /// where `B` and `X` are column-major `n × nrhs` matrices stored
    /// as flat slices of length `n * nrhs`. Returns
    /// `FeralError::NoFactor` if no factor is stored.
    ///
    /// Equivalent to `nrhs` independent `solve` calls but shares
    /// workspace and the supernodal traversal across columns.
    /// Mehrotra predictor-corrector IPM uses `nrhs = 2`. See
    /// `dev/plans/kkt-feature-gaps.md` F1.
    pub fn solve_many(&self, rhs: &[f64], nrhs: usize) -> Result<Vec<f64>, FeralError> {
        match &self.last_factors {
            Some(f) => solve_sparse_many(f, rhs, nrhs),
            None => Err(FeralError::NoFactor),
        }
    }

    /// Multi-RHS solve with per-column iterative refinement against
    /// the original matrix and the stored factor. Each column is
    /// refined independently — convergence is per-column, not all-
    /// at-once, matching the predictor-corrector use case where
    /// the two columns target different residual basins.
    pub fn solve_many_refined(
        &self,
        matrix: &CscMatrix,
        rhs: &[f64],
        nrhs: usize,
    ) -> Result<Vec<f64>, FeralError> {
        let factors = match &self.last_factors {
            Some(f) => f,
            None => return Err(FeralError::NoFactor),
        };
        if nrhs == 0 {
            return Ok(Vec::new());
        }
        let n = factors.n;
        if rhs.len() != n * nrhs {
            return Err(FeralError::DimensionMismatch {
                expected: n * nrhs,
                got: rhs.len(),
            });
        }
        let mut out = vec![0.0; n * nrhs];
        for c in 0..nrhs {
            let src = &rhs[c * n..(c + 1) * n];
            let xc = solve_sparse_refined(matrix, factors, src)?;
            out[c * n..(c + 1) * n].copy_from_slice(&xc);
        }
        Ok(out)
    }

    /// Estimate `kappa_1(A) = ||A||_1 * ||A^{-1}||_1` via the
    /// Hager-Higham 1-norm power iteration. Cost: 3-5 solves with the
    /// stored factor. Returns `FeralError::NoFactor` if no factor is
    /// stored. See `dev/research/condition-estimate.md` and F2 of
    /// `dev/plans/kkt-feature-gaps.md`.
    pub fn estimate_condition_1norm(&self, matrix: &CscMatrix) -> Result<f64, FeralError> {
        match &self.last_factors {
            Some(f) => estimate_condition_1norm(matrix, f),
            None => Err(FeralError::NoFactor),
        }
    }

    /// Two-stage quality escalation. Persistent across `factor()`
    /// calls. Returns `false` when both stages are exhausted.
    /// Mirrors `IpTSymLinearSolver::IncreaseQuality`.
    ///
    /// Stage 1 (`Baseline → ScalingEnabled`): if scaling strategy
    /// is `Identity`, flip to `InfNorm` (FERAL default). Skipped
    /// if scaling is already non-`Identity`.
    ///
    /// Stage 2 (`* → PivotRaised → Exhausted`): bump
    /// `bk.pivot_threshold`. From 0.0 jump to 0.01 (W5 special
    /// case, kept for callers that explicitly disable the threshold
    /// via `with_bk` + `BunchKaufmanParams::default`); else
    /// `min(pivtol_max, threshold^0.75)`. When the new threshold
    /// reaches `pivtol_max`, transition to `Exhausted` for the
    /// *next* call.
    ///
    /// `NumericParams::default()` already starts at
    /// `pivot_threshold = 1e-8` (MA27 default, issue #2), so for
    /// `Solver::new()` callers the W5 special case is dead and the
    /// cascade goes 1e-8 → 1e-6 → 10^-4.5 → ... → `pivtol_max`.
    pub fn increase_quality(&mut self) -> bool {
        const FIRST_PIVOT_THRESHOLD: f64 = 0.01;
        const PIVOT_EXPONENT: f64 = 0.75;
        const EPS_CAP: f64 = 1e-12;

        match self.quality_level {
            QualityLevel::Exhausted => false,
            QualityLevel::Baseline => {
                // Stage 1: flip Identity → InfNorm if applicable.
                if matches!(self.numeric_params.scaling, ScalingStrategy::Identity) {
                    self.numeric_params.scaling = ScalingStrategy::InfNorm;
                    self.quality_level = QualityLevel::ScalingEnabled;
                    true
                } else {
                    // Stage 1 is a no-op; fall through to stage 2.
                    self.bump_pivot_threshold(FIRST_PIVOT_THRESHOLD, PIVOT_EXPONENT, EPS_CAP);
                    true
                }
            }
            QualityLevel::ScalingEnabled | QualityLevel::PivotRaised => {
                self.bump_pivot_threshold(FIRST_PIVOT_THRESHOLD, PIVOT_EXPONENT, EPS_CAP);
                true
            }
        }
    }

    /// Apply the stage-2 pivot rule and update `quality_level`.
    /// Caller has already decided that stage 2 should fire and
    /// that `Exhausted` is not the current state.
    fn bump_pivot_threshold(&mut self, first_jump: f64, exponent: f64, eps_cap: f64) {
        let pivtol = &mut self.numeric_params.bk.pivot_threshold;
        if *pivtol == 0.0 {
            *pivtol = first_jump;
        } else {
            *pivtol = pivtol.powf(exponent).min(self.pivtol_max);
        }
        self.quality_level = if *pivtol >= self.pivtol_max - eps_cap {
            QualityLevel::Exhausted
        } else {
            QualityLevel::PivotRaised
        };
    }

    /// Test/diagnostic accessor for the current pivot threshold.
    pub fn pivot_threshold(&self) -> f64 {
        self.numeric_params.bk.pivot_threshold
    }

    /// Test/diagnostic accessor for the current scaling strategy.
    pub fn scaling_strategy(&self) -> &ScalingStrategy {
        &self.numeric_params.scaling
    }

    /// Whether `factor()` is configured to use the rayon-parallel
    /// multifrontal driver. Default `true`. See `with_parallel`.
    pub fn parallel(&self) -> bool {
        self.use_parallel
    }

    /// Whether the SQD (symmetric quasi-definite) fast-path is enabled.
    /// See `with_sqd_mode`.
    pub fn sqd_mode(&self) -> bool {
        self.numeric_params.sqd_mode
    }

    /// Number of negative eigenvalues from the last factor.
    /// Panics if no factor has been performed yet (mirrors Ipopt
    /// `NumberOfNegEVals()`, which has the same precondition).
    pub fn num_negative_eigenvalues(&self) -> usize {
        match &self.last_inertia {
            Some(i) => i.negative,
            None => panic!("num_negative_eigenvalues called before factor()"),
        }
    }

    /// Whether the solver provides inertia. Always `true` for FERAL.
    pub fn provides_inertia(&self) -> bool {
        true
    }

    /// Full inertia of the last successful factor, if any. Returns
    /// `None` if no factor is stored. See `num_negative_eigenvalues`
    /// for the Ipopt-shaped accessor that panics on a missing factor.
    pub fn inertia(&self) -> Option<&Inertia> {
        self.last_inertia.as_ref()
    }

    /// Minimum eigenvalue of D over the most recent factor's pivots.
    /// Returns `None` if no factor is stored. Mirrors Ipopt
    /// `SymLinearSolver::MinDiagonal` for the unconstrained
    /// inertia-correction shortcut. See
    /// [`SparseFactors::min_diagonal`].
    pub fn min_diagonal(&self) -> Option<f64> {
        self.last_factors.as_ref().and_then(|f| f.min_diagonal())
    }

    /// Smallest accepted pivot magnitude `min|λ(D)|` over the most
    /// recent factor — FERAL's near-singularity signal, the analog of
    /// MA57's `CNTL(2)` small-pivot threshold. Returns `None` if no
    /// factor is stored. Unlike [`min_diagonal`](Self::min_diagonal)
    /// (signed smallest eigenvalue), this is the smallest-in-magnitude
    /// pivot regardless of sign. Pair with
    /// [`max_pivot_magnitude`](Self::max_pivot_magnitude) for the
    /// scale-free ratio. See [`SparseFactors::min_pivot_magnitude`] and
    /// `dev/research/near-singularity-signal.md`.
    pub fn min_pivot_magnitude(&self) -> Option<f64> {
        self.last_factors
            .as_ref()
            .and_then(|f| f.min_pivot_magnitude())
    }

    /// Largest accepted pivot magnitude `max|λ(D)|` over the most
    /// recent factor. Returns `None` if no factor is stored. See
    /// [`min_pivot_magnitude`](Self::min_pivot_magnitude) and
    /// [`SparseFactors::max_pivot_magnitude`].
    pub fn max_pivot_magnitude(&self) -> Option<f64> {
        self.last_factors
            .as_ref()
            .and_then(|f| f.max_pivot_magnitude())
    }

    /// Borrow the most recent successful factor, if any. Lets a
    /// caller drive `solve_sparse_refined` directly when needed.
    pub fn factors(&self) -> Option<&SparseFactors> {
        self.last_factors.as_ref()
    }

    /// Current quality-escalation level.
    pub fn quality_level(&self) -> QualityLevel {
        self.quality_level
    }

    /// Number of times `symbolic_factorize` has been invoked from
    /// this `Solver`. Increments on the first `factor()` call after
    /// `Solver::new()` and on any subsequent `factor()` whose
    /// matrix pattern differs from the cached one. Diagnostic /
    /// test-facing counter.
    pub fn symbolic_call_count(&self) -> usize {
        self.symbolic_call_count
    }

    /// `ScalingInfo` from the most recent successful `factor()`.
    /// Returns `None` if no factor is stored. Use this to detect
    /// `Mc64FallbackToInfnorm` (issue #24) or `PartialSingular`
    /// after factoring without re-deriving it from a `factors()`
    /// borrow.
    pub fn scaling_info(&self) -> Option<&crate::scaling::ScalingInfo> {
        self.last_factors.as_ref().map(|f| &f.scaling_info)
    }

    /// Number of `factor()` calls on this `Solver` whose resulting
    /// `ScalingInfo` was `Mc64FallbackToInfnorm`. Issue #24:
    /// `ScalingStrategy::Auto` can silently fall back from MC64 to
    /// InfNorm in two cases (`InfNormSpreadAcceptable`,
    /// `Mc64WorseThanInfnorm` — see [`Mc64FallbackReason`]). This
    /// counter lets long-running IPM drivers detect the fallback
    /// without inspecting each factor's `scaling_info`.
    ///
    /// [`Mc64FallbackReason`]: crate::scaling::Mc64FallbackReason
    pub fn mc64_fallback_count(&self) -> usize {
        self.mc64_fallback_count
    }

    /// Number of `factor()` calls on this `Solver` that reused the
    /// value-bounded MC64 scaling cache (Track B2) instead of
    /// rerunning the Hungarian. Resets on `Solver::new()`; stays `0`
    /// when `with_mc64_cache(false)` is set. Lets tests and replay
    /// probes report the cache hit rate without inspecting
    /// per-factor state. See `dev/plans/mc64-value-bounded-cache.md`.
    pub fn mc64_cache_hit_count(&self) -> usize {
        self.mc64_cache_hit_count
    }

    /// Drop the cached symbolic factorisation and its associated
    /// pattern fingerprint, forcing the next `factor()` call to
    /// re-run `symbolic_factorize_with_method` from scratch.
    ///
    /// The pooled numeric `workspace`, the cached `last_factors`,
    /// and the persistent `parallel_pool` are NOT touched. This is
    /// the bisection hook for the warm-state amplification
    /// investigation (`dev/research/warm-state-cascade-amplification-2026-05-17.md`):
    /// it lets a probe distinguish "warm everything" from
    /// "warm everything *except* symbolic" without rebuilding the
    /// whole `Solver`.
    pub fn invalidate_symbolic_cache(&mut self) {
        self.last_symbolic = None;
        self.last_pattern_fingerprint = None;
    }
}

/// Compute `||A||_∞` for a symmetric matrix stored as a CSC lower
/// (or upper) triangle. Uses the symmetric definition:
/// `||A||_∞ = max_i Σ_j |a_ij|`. Iterates the stored half once and
/// reflects off-diagonal entries to the opposite row sum. Returns
/// `0.0` for `n = 0`. Used by `Solver::factor` to derive the
/// absolute floor for `NumericParams::static_pivot_threshold`.
fn matrix_inf_norm(matrix: &CscMatrix) -> f64 {
    let n = matrix.n;
    if n == 0 {
        return 0.0;
    }
    let mut row_sums = vec![0.0_f64; n];
    for j in 0..n {
        let start = matrix.col_ptr[j];
        let end = matrix.col_ptr[j + 1];
        for p in start..end {
            let i = matrix.row_idx[p];
            let v = matrix.values[p].abs();
            row_sums[i] += v;
            if i != j {
                row_sums[j] += v;
            }
        }
    }
    let mut m = 0.0_f64;
    for s in row_sums {
        if s > m {
            m = s;
        }
    }
    m
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::factor::BunchKaufmanParams;

    fn solver_with_scaling(scaling: ScalingStrategy) -> Solver {
        let np = NumericParams {
            bk: BunchKaufmanParams::default(),
            scaling,
            small_leaf: Default::default(),
            profiler: None,
            parallel_telemetry: None,
            fma: false,
            allow_delayed_pivots: true,
            cascade_break_ratio: None,
            cascade_break_eps: None,
            min_parallel_flops: None,
            sqd_mode: false,
            static_pivot_threshold: None,
            warn_partial_singular: false,
        };
        Solver::with_params(np, SupernodeParams::default())
    }

    /// #43 — the MC64 partial-singular stderr breadcrumb is off by
    /// default (feral stays quiet as a library should) and the
    /// `with_partial_singular_warning` builder toggles it.
    #[test]
    fn partial_singular_warning_default_off_and_builder_toggles() {
        assert!(
            !NumericParams::default().warn_partial_singular,
            "default must be quiet (#43)"
        );
        assert!(
            !NumericParams::with_bk(BunchKaufmanParams::default()).warn_partial_singular,
            "with_bk constructor must also default quiet (#43)"
        );

        let off = Solver::new();
        assert!(!off.numeric_params.warn_partial_singular);

        let on = Solver::new().with_partial_singular_warning(true);
        assert!(on.numeric_params.warn_partial_singular);

        let back_off = Solver::new()
            .with_partial_singular_warning(true)
            .with_partial_singular_warning(false);
        assert!(!back_off.numeric_params.warn_partial_singular);
    }

    /// Regression for the dtoc2 iter-1 hang (2026-05-17 §08:00,
    /// §11:50): IPOPT's δ_w bump overflowed beyond 1e308 on 103
    /// diagonal entries, shipping a matrix with literal `+inf`
    /// values to feral. Without input validation the BK pivot
    /// search loops forever (every threshold test fails against
    /// the inf column max). With Step 0 validation in `factor()`,
    /// the call must return `FatalError(InvalidInput)` in O(nnz)
    /// rather than hang.
    #[test]
    fn factor_rejects_non_finite_values() {
        let n = 10;
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        col_ptr.push(0);
        for j in 0..n {
            row_idx.push(j);
            values.push(if j == 5 { f64::INFINITY } else { 2.0 });
            col_ptr.push(row_idx.len());
        }
        let csc = CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        };
        let mut s = Solver::new();
        match s.factor(&csc, None) {
            FactorStatus::FatalError(FeralError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("non-finite"),
                    "error message should explain the trip: got {msg}"
                );
            }
            other => panic!("expected FatalError(InvalidInput), got {other:?}"),
        }

        // NaN must also be rejected.
        let mut values2: Vec<f64> = (0..n).map(|_| 2.0_f64).collect();
        values2[3] = f64::NAN;
        let csc2 = CscMatrix {
            n,
            col_ptr: csc.col_ptr.clone(),
            row_idx: csc.row_idx.clone(),
            values: values2,
        };
        match s.factor(&csc2, None) {
            FactorStatus::FatalError(FeralError::InvalidInput(_)) => {}
            other => panic!("expected FatalError(InvalidInput) for NaN, got {other:?}"),
        }
    }

    /// Issue #24: an arrow KKT with uniform absolute values triggers
    /// the `Auto` MC64 routing (high diag_only ratio + a dense arrow
    /// head of size > 32) but the pre-MC64 InfNorm trial passes the
    /// IN_SPREAD_GUARD, so the fallback fires. After `factor()` returns
    /// Success, `Solver::scaling_info` reports the new variant and
    /// `mc64_fallback_count` is 1.
    ///
    /// Construction (mirrors the scaling-module test
    /// `auto_surfaces_infnorm_spread_fallback_on_uniform_diag`):
    /// n=200 with column 0 dense (200 stored entries of value 2.0)
    /// and columns 1..199 degree-1 with the diagonal value 2.0.
    /// All stored absolute values are 2.0, so InfNorm converges to a
    /// uniform `d` and the IN_SPREAD_GUARD fires. The dense-column
    /// gate (>32) added on 2026-05-17 to `pick_scaling_strategy` is
    /// satisfied by column 0.
    ///
    /// Note: `n` must exceed the dense fast-path gate
    /// (`should_use_dense_fast_path`, n=128 with density<0.25). With
    /// nnz ≈ 2n = 400, density = 400 / n² = 1% << 25% so the sparse
    /// path is taken and Auto's fallback fires.
    #[test]
    fn mc64_fallback_surfaces_via_solver_api() {
        let n = 200;
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        // Column 0: dense, uniform |a| = 2.0.
        for i in 0..n {
            row_idx.push(i);
            values.push(2.0);
        }
        col_ptr.push(row_idx.len());
        // Columns 1..n: degree-1 diagonal, value 2.0.
        for j in 1..n {
            row_idx.push(j);
            values.push(2.0);
            col_ptr.push(row_idx.len());
        }
        let csc = CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        };
        // Precondition: not on the dense fast-path so Auto's
        // fallback logic actually runs.
        assert!(
            !crate::numeric::factorize::should_use_dense_fast_path(csc.n, csc.row_idx.len()),
            "test setup error: matrix would take dense fast-path"
        );

        let mut s = solver_with_scaling(ScalingStrategy::Auto);
        assert_eq!(s.mc64_fallback_count(), 0);
        assert!(s.scaling_info().is_none());

        let status = s.factor(&csc, None);
        assert!(
            matches!(status, FactorStatus::Success),
            "factor must succeed on a positive-definite uniform diagonal, got {:?}",
            status
        );

        // The new ScalingInfo variant is surfaced through the
        // `scaling_info` accessor, and the fallback counter is
        // bumped exactly once.
        match s.scaling_info() {
            Some(crate::scaling::ScalingInfo::Mc64FallbackToInfnorm {
                reason: crate::scaling::Mc64FallbackReason::InfNormSpreadAcceptable,
            }) => {}
            other => panic!(
                "expected Mc64FallbackToInfnorm{{InfNormSpreadAcceptable}}, got {:?}",
                other
            ),
        }
        assert_eq!(s.mc64_fallback_count(), 1);

        // Re-factoring the same matrix bumps the counter again
        // (every fallback fire is counted, not just unique
        // patterns) — same pattern, so symbolic is cached.
        let _ = s.factor(&csc, None);
        assert_eq!(s.mc64_fallback_count(), 2);
        assert_eq!(
            s.symbolic_call_count(),
            1,
            "symbolic must be cached across re-factor on same pattern"
        );
    }

    /// Issue #38: the `OrderingPreprocess::LdltCompress` path populates
    /// `SymbolicFactorization::cached_mc64` for the immediately-following
    /// numeric reuse. In an IPM driver the same `Solver` is fed values-
    /// drifting matrices on the same pattern, and reusing the iter-0
    /// cache on iter-N silently corrupts inertia and explodes per-factor
    /// cost (rocket_12800 reproducer: 43.2 s wrong vs 1.6 s correct).
    ///
    /// The fix in `Solver::factor` clears `cached_mc64` after every
    /// numeric call. This test locks the invariant directly by
    /// inspecting the field, since the downstream symptom (wrong
    /// inertia) only manifests on large arrow-KKT matrices where the
    /// scaling change destabilises Bunch-Kaufman pivoting — Sylvester's
    /// law keeps inertia invariant under any symmetric scaling on
    /// small well-conditioned matrices, so a behavioural test on a
    /// 4×4 reproducer is insensitive.
    #[test]
    fn mc64_cache_invalidated_after_factor_issue_38() {
        use crate::symbolic::OrderingPreprocess;
        // 4×4 block-anti-diagonal: MC64 matches (0,2) and (1,3) on
        // the large off-diagonals, so `LdltCompress` populates
        // `cached_mc64` (ncmp=2 < n=4 makes the compression path
        // take the cache-storing branch).
        let csc =
            CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 0, 1], &[1.0, 1.0, 10.0, 10.0])
                .expect("valid CSC");

        let np = NumericParams {
            scaling: ScalingStrategy::Mc64Symmetric,
            ..NumericParams::default()
        };
        let sn = SupernodeParams {
            nemin: 1,
            preprocess: OrderingPreprocess::LdltCompress,
            ..SupernodeParams::default()
        };
        let mut s = Solver::with_params(np, sn);

        let status = s.factor(&csc, None);
        assert!(
            matches!(status, FactorStatus::Success),
            "factor must succeed on block-antidiag, got {:?}",
            status
        );
        let sym = s
            .last_symbolic
            .as_ref()
            .expect("symbolic must be stored after Success");
        assert!(
            sym.cached_mc64.is_none(),
            "cached_mc64 must be cleared after factor() (issue #38: IPM \
             reuse of iter-0 MC64 cache on iter-N matrix silently corrupts \
             inertia and explodes factor cost on real arrow-KKTs)"
        );
    }

    /// Symmetric tridiagonal `n×n` CSC (lower triangle) with `diag`
    /// on the diagonal and `off` on the first sub-diagonal. When
    /// `diag > 2·|off|` the matrix is strictly diagonally dominant
    /// with a positive diagonal, hence SPD with inertia `(n, 0, 0)`.
    fn tridiag(n: usize, diag: f64, off: f64) -> CscMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(diag);
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
                vals.push(off);
            }
        }
        CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("valid CSC")
    }

    fn mc64_solver() -> Solver {
        let np = NumericParams {
            scaling: ScalingStrategy::Mc64Symmetric,
            ..NumericParams::default()
        };
        Solver::with_params(np, SupernodeParams::default())
    }

    /// Track B2: re-factoring an identical matrix reuses the
    /// value-bounded MC64 scaling cache — zero value drift, the
    /// value bound passes, the Hungarian is skipped on call 2.
    #[test]
    fn mc64_scaling_cache_hit_on_identical_refactor() {
        let a = tridiag(6, 10.0, 1.0); // SPD → inertia (6,0,0)
        let mut s = mc64_solver();

        let st1 = s.factor(&a, Some(Inertia::new(6, 0, 0)));
        assert!(matches!(st1, FactorStatus::Success), "factor 1: {:?}", st1);
        assert_eq!(s.mc64_cache_hit_count(), 0, "first factor cannot hit");

        let st2 = s.factor(&a, Some(Inertia::new(6, 0, 0)));
        assert!(matches!(st2, FactorStatus::Success), "factor 2: {:?}", st2);
        assert_eq!(
            s.mc64_cache_hit_count(),
            1,
            "identical refactor must reuse the cached MC64 scaling"
        );
    }

    /// Track B2: a cache hit must produce a bit-identical
    /// factorization to the cache-off path. Oracle = the cache-off
    /// fresh-MC64 path (pre-B2 code). On identical values fresh MC64
    /// is deterministic, so reusing `D₀` applies the same congruence.
    #[test]
    fn mc64_cache_hit_bit_matches_cache_off() {
        let a = tridiag(6, 10.0, 1.0);
        let rhs: Vec<f64> = (1..=6).map(|x| x as f64).collect();

        let mut on = mc64_solver(); // cache on by default
        let mut off = mc64_solver().with_mc64_cache(false);

        for call in 0..3 {
            let so = on.factor(&a, None);
            let sf = off.factor(&a, None);
            assert!(
                matches!(so, FactorStatus::Success) && matches!(sf, FactorStatus::Success),
                "call {call}: on={:?} off={:?}",
                so,
                sf
            );
            let xo = on.solve(&rhs).expect("cache-on solve");
            let xf = off.solve(&rhs).expect("cache-off solve");
            assert_eq!(
                xo, xf,
                "call {call}: cache-on solve must bit-match cache-off"
            );
            assert_eq!(
                on.last_inertia, off.last_inertia,
                "call {call}: inertia must match"
            );
        }
        assert_eq!(
            on.mc64_cache_hit_count(),
            2,
            "calls 1 and 2 reuse the call-0 cache"
        );
        assert_eq!(
            off.mc64_cache_hit_count(),
            0,
            "with_mc64_cache(false) never hits"
        );
    }

    /// Track B2: a pattern change voids the cache; it is rebuilt for
    /// the new pattern and hits on that pattern's repeated factor.
    #[test]
    fn mc64_cache_rebuilt_on_pattern_change() {
        let a6 = tridiag(6, 10.0, 1.0);
        let a8 = tridiag(8, 10.0, 1.0); // distinct pattern (n differs)
        let mut s = mc64_solver();

        assert!(matches!(
            s.factor(&a6, Some(Inertia::new(6, 0, 0))),
            FactorStatus::Success
        ));
        assert!(matches!(
            s.factor(&a6, Some(Inertia::new(6, 0, 0))),
            FactorStatus::Success
        ));
        assert_eq!(s.mc64_cache_hit_count(), 1, "repeated a6 factor hits");

        assert!(
            matches!(
                s.factor(&a8, Some(Inertia::new(8, 0, 0))),
                FactorStatus::Success
            ),
            "a different pattern must factor cleanly"
        );
        assert_eq!(
            s.mc64_cache_hit_count(),
            1,
            "a pattern change cannot be a cache hit"
        );
        assert!(matches!(
            s.factor(&a8, Some(Inertia::new(8, 0, 0))),
            FactorStatus::Success
        ));
        assert_eq!(
            s.mc64_cache_hit_count(),
            2,
            "the rebuilt cache hits on the repeated a8 factor"
        );
    }

    /// Track B2 — the #38 guard. When the matrix values drift far
    /// past the matching's diagonal-dominance bound, the value-bound
    /// check must reject the stale cache and rerun the Hungarian.
    /// This is the property that keeps the B2 cache from
    /// reintroducing the #38 iter-0-scaling-on-iter-N inertia drift.
    ///
    /// `a0` is diagonally dominant SPD → inertia (4,0,0). `a1` shares
    /// the pattern but blows the off-diagonal 50× past the diagonal.
    /// `a1` is a Toeplitz tridiagonal with eigenvalues
    /// `λ_k = 10 + 100·cos(kπ/5)` = {90.9, 40.9, −20.9, −70.9}
    /// → inertia (2,2,0) (hand oracle).
    #[test]
    fn mc64_cache_rejected_on_value_drift_issue_38_guard() {
        let a0 = tridiag(4, 10.0, 1.0);
        let a1 = tridiag(4, 10.0, 50.0);
        let mut s = mc64_solver();

        assert!(matches!(
            s.factor(&a0, Some(Inertia::new(4, 0, 0))),
            FactorStatus::Success
        ));
        assert_eq!(
            s.mc64_cache_hit_count(),
            0,
            "first factor installs the cache"
        );

        // a1's values have drifted far past the dominance bound
        // under a0's scaling: the value-bound check must reject, the
        // Hungarian reruns fresh, and inertia stays correct.
        let st = s.factor(&a1, Some(Inertia::new(2, 2, 0)));
        assert!(
            matches!(st, FactorStatus::Success),
            "drifted matrix must factor to its correct inertia (2,2,0), got {:?}",
            st
        );
        assert_eq!(
            s.mc64_cache_hit_count(),
            0,
            "value drift past the dominance bound must reject the stale cache"
        );
    }

    /// Track B2: `with_mc64_cache(false)` keeps every factor on the
    /// fresh-MC64 path — no hits ever, and the cache stays `None`.
    #[test]
    fn mc64_cache_disabled_never_hits() {
        let a = tridiag(6, 10.0, 1.0);
        let mut s = mc64_solver().with_mc64_cache(false);
        for _ in 0..4 {
            assert!(matches!(
                s.factor(&a, Some(Inertia::new(6, 0, 0))),
                FactorStatus::Success
            ));
        }
        assert_eq!(s.mc64_cache_hit_count(), 0, "disabled cache never hits");
        assert!(
            s.mc64_scaling_cache.is_none(),
            "disabled cache stays unpopulated"
        );
    }

    /// U1 — Baseline + Identity scaling: stage 1 fires.
    #[test]
    fn u1_increase_quality_baseline_identity_to_scaling_enabled() {
        let mut s = solver_with_scaling(ScalingStrategy::Identity);
        assert_eq!(s.quality_level(), QualityLevel::Baseline);
        assert_eq!(s.pivot_threshold(), 0.0);

        assert!(s.increase_quality());

        assert!(matches!(s.scaling_strategy(), ScalingStrategy::InfNorm));
        assert_eq!(s.pivot_threshold(), 0.0, "stage 1 must not touch pivot");
        assert_eq!(s.quality_level(), QualityLevel::ScalingEnabled);
    }

    /// U2 — Baseline + non-Identity scaling: stage 1 is a no-op,
    /// fall through to stage 2.
    #[test]
    fn u2_increase_quality_baseline_nonidentity_skips_to_pivot_raised() {
        let mut s = solver_with_scaling(ScalingStrategy::InfNorm);
        assert_eq!(s.quality_level(), QualityLevel::Baseline);

        assert!(s.increase_quality());

        assert_eq!(s.pivot_threshold(), 0.01, "first jump rule");
        assert_eq!(s.quality_level(), QualityLevel::PivotRaised);
    }

    /// U3 — Subsequent pivot bumps follow the geometric rule.
    #[test]
    fn u3_increase_quality_pivot_geometric_rule() {
        let mut s = solver_with_scaling(ScalingStrategy::InfNorm);
        s.numeric_params.bk.pivot_threshold = 0.01;
        s.quality_level = QualityLevel::PivotRaised;

        assert!(s.increase_quality());
        let want = 0.01_f64.powf(0.75);
        assert!(
            (s.pivot_threshold() - want).abs() < 1e-15,
            "got {}",
            s.pivot_threshold()
        );
        assert_eq!(s.quality_level(), QualityLevel::PivotRaised);
    }

    /// U4 — Pivot bump caps at `pivtol_max` and transitions to
    /// `Exhausted`; the next call returns `false`.
    #[test]
    fn u4_increase_quality_caps_at_pivtol_max_then_exhausts() {
        let mut s = solver_with_scaling(ScalingStrategy::InfNorm);
        s.numeric_params.bk.pivot_threshold = 0.49;
        s.quality_level = QualityLevel::PivotRaised;

        // 0.49^0.75 ≈ 0.585, capped to pivtol_max = 0.5.
        assert!(s.increase_quality());
        assert_eq!(s.pivot_threshold(), 0.5);
        assert_eq!(s.quality_level(), QualityLevel::Exhausted);

        assert!(!s.increase_quality());
        assert_eq!(s.pivot_threshold(), 0.5);
        assert_eq!(s.quality_level(), QualityLevel::Exhausted);
    }

    /// U5 — Repeated calls always terminate at `Exhausted` in
    /// finitely many steps.
    #[test]
    fn u5_increase_quality_exhausted_returns_false() {
        let mut s = solver_with_scaling(ScalingStrategy::Identity);
        let mut steps = 0;
        while s.increase_quality() {
            steps += 1;
            assert!(steps < 20, "did not exhaust within 20 steps");
        }
        assert_eq!(s.quality_level(), QualityLevel::Exhausted);
    }

    /// F1 — same pattern fingerprints equal, structural hash stable
    /// across value changes.
    #[test]
    fn f1_fingerprint_same_pattern_equal() {
        let a = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
        let b = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[7.0, 11.0, 13.0]).unwrap();
        let fa = PatternFingerprint::of(&a);
        let fb = PatternFingerprint::of(&b);
        assert_eq!(
            fa, fb,
            "byte-identical patterns must fingerprint identically"
        );
    }

    /// F2 — pre-existing footgun closed: two matrices with identical
    /// `(n, nnz)` but different sparsity patterns now fingerprint
    /// differently. Under the legacy `(n, col_ptr_len, row_idx_len)`
    /// scheme these collided silently.
    #[test]
    fn f2_fingerprint_distinguishes_same_n_nnz_different_pattern() {
        // Two 3x3 matrices, both with 3 nonzeros (lower-triangle
        // CSC), but completely different patterns:
        //
        //   A = diag(2, 3, 5)          B = [[2 . .]
        //                                    [1 3 .]
        //                                    [. 1 .]]   (zero-diag last col)
        //
        // Both have n=3, nnz=3. Under the old fingerprint they would
        // collide. The new structural hash must separate them.
        let a = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
        let b = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
        // Sanity: B before mutation matches A.
        assert_eq!(PatternFingerprint::of(&a), PatternFingerprint::of(&b));

        // Now build a structurally different matrix with same (n, nnz)
        // — same column pointers (one entry per column) but different
        // row indices: [2, 2, 2] instead of [0, 1, 2].
        let c = CscMatrix::from_triplets(3, &[2, 2, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
        assert_eq!(c.n, a.n);
        assert_eq!(c.row_idx.len(), a.row_idx.len());
        assert_eq!(c.col_ptr.len(), a.col_ptr.len());
        assert_ne!(
            PatternFingerprint::of(&a),
            PatternFingerprint::of(&c),
            "same (n, nnz) but different row_idx must fingerprint differently"
        );
    }

    /// F3 — different col_ptr distribution at same `(n, nnz)`
    /// fingerprints differently.
    #[test]
    fn f3_fingerprint_distinguishes_different_col_ptr() {
        // A: 4x4 diagonal, col_ptr = [0,1,2,3,4], nnz=4.
        let a = CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        // B: 4x4 with same nnz=4 but two entries in column 0 and one
        // each in cols 1, 2 — different col_ptr.
        let b = CscMatrix::from_triplets(4, &[0, 1, 1, 2], &[0, 0, 1, 2], &[1.0, 0.5, 2.0, 3.0])
            .unwrap();
        assert_eq!(a.n, b.n);
        assert_eq!(a.row_idx.len(), b.row_idx.len());
        assert_ne!(
            PatternFingerprint::of(&a),
            PatternFingerprint::of(&b),
            "different col_ptr distribution must fingerprint differently"
        );
    }

    // -- Issue #7: parallel driver exposure on `Solver` -----------------

    /// `Solver::new()` defaults to the rayon-parallel multifrontal
    /// driver. The parallel driver internally falls through to the
    /// sequential supernodal path on small problems via
    /// `should_parallelize_assembly` so default-on does not regress
    /// small-problem latency.
    #[test]
    fn solver_parallel_default_is_on() {
        let solver = Solver::new();
        assert!(
            solver.parallel(),
            "Solver::new() should default to use_parallel = true"
        );
    }

    /// `Solver::with_parallel` toggles the driver flag in both
    /// directions.
    #[test]
    fn solver_with_parallel_toggles() {
        let solver = Solver::new().with_parallel(false);
        assert!(!solver.parallel());
        let solver = solver.with_parallel(true);
        assert!(solver.parallel());
    }

    /// Issue #19 follow-up: the `Solver`-owned rayon `ThreadPool` is
    /// built lazily on first parallel-fire and reused on subsequent
    /// `factor()` calls. Verify by Arc-pointer identity that the
    /// pool stored across two factorizations is the same instance.
    ///
    /// Construct an indefinite tridiagonal that is large enough
    /// (`n = 4096`) to clear the `should_parallelize_assembly`
    /// structural gate's `N_PAR_MIN = 32` supernode count, but small
    /// enough that the new flop gate may or may not fire — pool
    /// construction is independent of whether the gate ends up
    /// dispatching parallel for the inner driver, since `factor()`
    /// always calls `ensure_parallel_pool()` when `use_parallel`
    /// is on.
    #[test]
    fn solver_reuses_thread_pool_across_factors() {
        // Indefinite tridiagonal: `2 -1 0 ... ; -1 2 -1 ... ; ...`
        // shifted by `-2.5*I` to push some eigenvalues negative,
        // so it doesn't get rejected by an SPD-only fast path.
        let n = 256usize;
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0 - 2.5);
            if i + 1 < n {
                rows.push(i + 1);
                cols.push(i);
                vals.push(-1.0);
            }
        }
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("matrix");

        let mut s = Solver::new();
        assert!(s.parallel_pool.is_none(), "pool must be lazy");

        let r1 = s.factor(&m, None);
        assert!(
            matches!(r1, FactorStatus::Success),
            "first factor must succeed, got {:?}",
            r1
        );
        let p1 = s
            .parallel_pool
            .as_ref()
            .expect("pool must be built after first parallel factor")
            .clone();

        let r2 = s.factor(&m, None);
        assert!(matches!(r2, FactorStatus::Success));
        let p2 = s.parallel_pool.as_ref().expect("pool persists").clone();

        // `Arc::ptr_eq` confirms it's the same `ThreadPool`
        // instance — not just a structurally-equivalent rebuild.
        assert!(
            std::sync::Arc::ptr_eq(&p1, &p2),
            "ThreadPool must be reused across factor() calls (issue #19 follow-up)"
        );
    }

    /// When `use_parallel` is false, the lazy pool must never be
    /// built — there's no need for one and we don't want to spawn
    /// worker threads we won't use.
    #[test]
    fn solver_with_parallel_false_does_not_build_pool() {
        let n = 32usize;
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0);
            if i + 1 < n {
                rows.push(i + 1);
                cols.push(i);
                vals.push(-1.0);
            }
        }
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("matrix");

        let mut s = Solver::new().with_parallel(false);
        let r = s.factor(&m, None);
        assert!(matches!(r, FactorStatus::Success));
        assert!(
            s.parallel_pool.is_none(),
            "with_parallel(false) must not build a thread pool"
        );
    }

    /// Amdahl-ceiling breakdown for the parallel driver. For each
    /// large matrix, runs the sequential driver with a `Profiler`
    /// attached, reports the supernode-time histogram, and computes
    /// the Amdahl ceiling = `total_seq / max_snode_seq`. Combined
    /// with the wall-clock A/B from
    /// `solver_parallel_speedup_largematrices`, this localises
    /// whether the remaining gap to the ceiling is from
    /// non-supernode work (assembly, mutex, allocation) or from
    /// being already at the ceiling (Amdahl-bound).
    ///
    /// `#[ignore]`'d — same data-dir contract as the speedup test.
    /// Invoke under release with:
    ///
    /// ```text
    /// cargo test --release solver_parallel_profile_breakdown \
    ///     -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn solver_parallel_profile_breakdown() {
        use crate::numeric::factorize::Profiler;
        use crate::read_mtx;
        use std::path::PathBuf;
        use std::sync::{Arc, Mutex};
        use std::time::Instant;

        let dir = PathBuf::from("tests/data/large");
        if !dir.is_dir() {
            eprintln!("SKIP: {} not found.", dir.display());
            return;
        }
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            eprintln!("SKIP: no .mtx in {}.", dir.display());
            return;
        }

        for path in &paths {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .trim_end_matches(".mtx")
                .to_string();
            let mtx = match read_mtx(path) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("[{}] SKIP read: {:?}", name, e);
                    continue;
                }
            };
            let csc = match mtx.to_csc() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("[{}] SKIP csc: {:?}", name, e);
                    continue;
                }
            };

            // Sequential with profiler.
            let prof = Arc::new(Mutex::new(Profiler::new()));
            let np = NumericParams {
                profiler: Some(prof.clone()),
                ..NumericParams::default()
            };
            let mut seq = Solver::with_params(np, SupernodeParams::default()).with_parallel(false);
            let t0 = Instant::now();
            let seq_status = seq.factor(&csc, None);
            let seq_ms = t0.elapsed().as_secs_f64() * 1e3;

            // Parallel A/B (fresh solver, no profiler — driver does
            // not record timings).
            let mut par = Solver::new();
            let t0 = Instant::now();
            let par_status = par.factor(&csc, None);
            let par_ms = t0.elapsed().as_secs_f64() * 1e3;

            let prof = match prof.lock() {
                Ok(p) => p.clone(),
                Err(e) => {
                    eprintln!("[{}] profiler poisoned: {}", name, e);
                    continue;
                }
            };
            let report = prof.report();
            let timings = prof.timings();
            let max_us = timings.iter().map(|t| t.us).max().unwrap_or(0);
            let top: Vec<_> = {
                let mut v: Vec<_> = timings.iter().collect();
                v.sort_by_key(|t| std::cmp::Reverse(t.us));
                v.into_iter().take(5).collect()
            };
            let amdahl_ceiling_ms = if max_us > 0 {
                seq_ms / ((report.total_us as f64) / (max_us as f64))
            } else {
                f64::INFINITY
            };

            let ok_seq = matches!(seq_status, FactorStatus::Success);
            let ok_par = matches!(par_status, FactorStatus::Success);

            eprintln!();
            eprintln!(
                "=== {} (n={}, nnz={}) [seq={}, par={}]",
                name,
                csc.n,
                csc.row_idx.len(),
                if ok_seq { "OK" } else { "FAIL" },
                if ok_par { "OK" } else { "FAIL" }
            );
            eprintln!(
                "  seq wall:       {:>9.1} ms   par wall: {:>9.1} ms   measured speedup: {:.2}×",
                seq_ms,
                par_ms,
                if par_ms > 0.0 { seq_ms / par_ms } else { 0.0 }
            );
            eprintln!(
                "  n_supernodes:   {:>9}     loop_us: {} us   prologue: {} us   epilogue: {} us   overhead: {:.1}%",
                report.n_supernodes,
                report.loop_us,
                report.prologue_us,
                report.epilogue_us,
                report.overhead_pct
            );
            eprintln!(
                "  Amdahl ceiling: par >= {:>5.1} ms  ⇒ max speedup ≈ {:.2}×  (largest single snode = {} us = {:.1}% of total)",
                amdahl_ceiling_ms,
                if max_us > 0 {
                    (report.total_us as f64) / (max_us as f64)
                } else {
                    0.0
                },
                max_us,
                if report.total_us > 0 {
                    100.0 * (max_us as f64) / (report.total_us as f64)
                } else {
                    0.0
                }
            );
            eprintln!("  top-5 supernodes by us:");
            for t in &top {
                eprintln!(
                    "      snode #{:6}  nrow={:6}  ncol={:6}  us={:>10}  ({:.1}% of total)",
                    t.snode_idx,
                    t.nrow,
                    t.ncol,
                    t.us,
                    if report.total_us > 0 {
                        100.0 * (t.us as f64) / (report.total_us as f64)
                    } else {
                        0.0
                    }
                );
            }
            eprintln!("  size histogram:");
            for b in &report.buckets {
                if b.count == 0 {
                    continue;
                }
                eprintln!(
                    "      nrow {:>6}   count={:>6}   sum_us={:>10}   {:5.1}% of loop   avg_us={:.0}",
                    b.range, b.count, b.sum_us, b.pct_of_total, b.avg_us
                );
            }

            // Critical-path analysis: the TRUE parallel ceiling is
            // `total_work / longest_weighted_path_through_etree`, not
            // `total_work / max_single_snode`. The naive ceiling above
            // is an upper bound; the weighted-path ceiling is what an
            // ideal scheduler with infinite workers can actually
            // reach. If `true_ceiling ≈ measured_speedup`, the etree
            // topology is the limit and no scheduler change will
            // help — only restructuring (e.g. intra-supernode
            // parallelism) can. Re-runs `symbolic_factorize` because
            // the Solver consumed its symbolic; the call is
            // deterministic.
            let symbolic =
                match crate::symbolic::symbolic_factorize(&csc, &SupernodeParams::default()) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("  (skip critical path: symbolic_factorize failed: {:?})", e);
                        continue;
                    }
                };
            let n_snodes = symbolic.supernodes.len();
            let mut work_us = vec![0u64; n_snodes];
            for t in timings {
                if t.snode_idx < n_snodes {
                    work_us[t.snode_idx] = t.us;
                }
            }
            // Postorder property: child indices < parent index, so a
            // single forward pass computes earliest-finish bottom-up.
            let mut earliest_finish = vec![0u64; n_snodes];
            for (i, snode) in symbolic.supernodes.iter().enumerate() {
                let max_child = snode
                    .children
                    .iter()
                    .filter(|&&c| c < n_snodes)
                    .map(|&c| earliest_finish[c])
                    .max()
                    .unwrap_or(0);
                earliest_finish[i] = max_child + work_us[i];
            }
            let critical_path_us = earliest_finish.iter().max().copied().unwrap_or(0);
            let total_us = work_us.iter().sum::<u64>();
            let true_ceiling = if critical_path_us > 0 {
                (total_us as f64) / (critical_path_us as f64)
            } else {
                0.0
            };
            // Depth from root (root = 0). Build parent table from
            // children, then walk parents in reverse postorder.
            let mut parent: Vec<Option<usize>> = vec![None; n_snodes];
            for (i, s) in symbolic.supernodes.iter().enumerate() {
                for &c in &s.children {
                    if c < n_snodes {
                        parent[c] = Some(i);
                    }
                }
            }
            let mut depth = vec![0usize; n_snodes];
            for i in (0..n_snodes).rev() {
                if let Some(p) = parent[i] {
                    depth[i] = depth[p] + 1;
                }
            }
            let max_depth = *depth.iter().max().unwrap_or(&0);
            let mut level_count = vec![0usize; max_depth + 1];
            let mut level_work_us = vec![0u64; max_depth + 1];
            for i in 0..n_snodes {
                level_count[depth[i]] += 1;
                level_work_us[depth[i]] += work_us[i];
            }
            eprintln!(
                "  CRITICAL PATH: {} us = {:.1} ms   total_work: {} us = {:.1} ms",
                critical_path_us,
                (critical_path_us as f64) / 1000.0,
                total_us,
                (total_us as f64) / 1000.0
            );
            eprintln!(
                "  TRUE parallel ceiling: {:.2}× (total_work / critical_path)",
                true_ceiling
            );
            eprintln!(
                "  etree depth: max={}  upper-tree level distribution (top 15 levels from root):",
                max_depth
            );
            for d in 0..=(max_depth.min(14)) {
                if level_count[d] == 0 {
                    continue;
                }
                eprintln!(
                    "      depth {:>4}  count={:>6}  work_us={:>10}  ({:.1}% of total)",
                    d,
                    level_count[d],
                    level_work_us[d],
                    if total_us > 0 {
                        100.0 * (level_work_us[d] as f64) / (total_us as f64)
                    } else {
                        0.0
                    }
                );
            }
        }
        eprintln!();
    }

    /// Wall-clock A/B between the parallel and sequential drivers on
    /// the four matrices in `tests/data/large/`. `#[ignore]`'d
    /// because it requires the large-matrix data dir (gitignored)
    /// and is a measurement, not a correctness gate.
    ///
    /// Invoke under release with:
    ///
    /// ```text
    /// cargo test --release solver_parallel_speedup_largematrices \
    ///     -- --ignored --nocapture
    /// ```
    ///
    /// Prints per-matrix wall-clock for `Solver::new()` (parallel)
    /// vs `Solver::new().with_parallel(false)` (sequential), plus
    /// the inertia check across both modes. Output is parsed by
    /// `dev/sessions/*.md` checkpoints — keep the format stable.
    #[test]
    #[ignore]
    fn solver_parallel_speedup_largematrices() {
        use crate::read_mtx;
        use std::path::PathBuf;
        use std::time::Instant;

        let dir = PathBuf::from("tests/data/large");
        if !dir.is_dir() {
            eprintln!("SKIP: {} not found.", dir.display());
            return;
        }

        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            eprintln!("SKIP: no .mtx in {}.", dir.display());
            return;
        }

        eprintln!(
            "\n  matrix                          n       nnz   par_ms   seq_ms  speedup  inertia_eq"
        );
        eprintln!(
            "  ------------------------------ -------- -------- -------- -------- -------- ----------"
        );
        for path in &paths {
            let mtx = match read_mtx(path) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  SKIP {}: {:?}", path.display(), e);
                    continue;
                }
            };
            let csc = match mtx.to_csc() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("  SKIP {}: csc {:?}", path.display(), e);
                    continue;
                }
            };
            let nnz = csc.row_idx.len();
            let n = csc.n;

            let mut par = Solver::new();
            let t0 = Instant::now();
            let par_status = par.factor(&csc, None);
            let par_ms = t0.elapsed().as_secs_f64() * 1e3;

            let mut seq = Solver::new().with_parallel(false);
            let t0 = Instant::now();
            let seq_status = seq.factor(&csc, None);
            let seq_ms = t0.elapsed().as_secs_f64() * 1e3;

            let par_ok = matches!(par_status, FactorStatus::Success);
            let seq_ok = matches!(seq_status, FactorStatus::Success);
            let inertia_eq = if par_ok && seq_ok {
                par.num_negative_eigenvalues() == seq.num_negative_eigenvalues()
            } else {
                false
            };

            let speedup = if par_ms > 0.0 { seq_ms / par_ms } else { 0.0 };
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .trim_end_matches(".mtx");
            eprintln!(
                "  {:30} {:8} {:8} {:8.1} {:8.1} {:7.2}× {:>10}",
                name,
                n,
                nnz,
                par_ms,
                seq_ms,
                speedup,
                if inertia_eq {
                    "yes"
                } else if par_ok && seq_ok {
                    "NO"
                } else {
                    "(failed)"
                }
            );
        }
        eprintln!();
    }

    /// Thread-count sweep: factor each large corpus matrix under the
    /// parallel driver with `RAYON_NUM_THREADS=1,2,4,8` (a custom
    /// rayon pool is built per row). Used to discriminate between
    /// compute-bound and memory-bandwidth-bound regimes — if speedup
    /// flattens at 4→8 threads on a matrix, the inner kernel has
    /// saturated DRAM bandwidth, not lock contention or per-task
    /// overhead.
    ///
    /// `#[ignore]` for the same reason as
    /// `solver_parallel_speedup_largematrices`: requires the
    /// gitignored large-matrix data dir and is a measurement, not a
    /// correctness gate.
    ///
    /// Invoke under release with:
    ///
    /// ```text
    /// cargo test --release solver_parallel_threadcount_sweep \
    ///     -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn solver_parallel_threadcount_sweep() {
        use crate::read_mtx;
        use std::path::PathBuf;
        use std::time::Instant;

        let dir = PathBuf::from("tests/data/large");
        if !dir.is_dir() {
            eprintln!("SKIP: {} not found.", dir.display());
            return;
        }

        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            eprintln!("SKIP: no .mtx in {}.", dir.display());
            return;
        }

        let thread_counts: &[usize] = &[1, 2, 4, 8];
        eprintln!(
            "\n  matrix                          n       nnz    T=1_ms   T=2_ms   T=4_ms   T=8_ms   sp_2   sp_4   sp_8"
        );
        eprintln!(
            "  ------------------------------ -------- -------- -------- -------- -------- -------- ------ ------ ------"
        );
        for path in &paths {
            let mtx = match read_mtx(path) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  SKIP {}: {:?}", path.display(), e);
                    continue;
                }
            };
            let csc = match mtx.to_csc() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("  SKIP {}: csc {:?}", path.display(), e);
                    continue;
                }
            };
            let nnz = csc.row_idx.len();
            let n = csc.n;

            let mut wall_ms: Vec<f64> = Vec::with_capacity(thread_counts.len());
            for &nt in thread_counts {
                let pool = match rayon::ThreadPoolBuilder::new().num_threads(nt).build() {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!(
                            "  SKIP {}: failed to build rayon pool with {} threads: {}",
                            path.display(),
                            nt,
                            e
                        );
                        continue;
                    }
                };
                let elapsed_ms = pool.install(|| {
                    let mut solver = Solver::new();
                    let t0 = Instant::now();
                    let _ = solver.factor(&csc, None);
                    t0.elapsed().as_secs_f64() * 1e3
                });
                wall_ms.push(elapsed_ms);
            }

            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .trim_end_matches(".mtx");
            let t1 = wall_ms.first().copied().unwrap_or(f64::NAN);
            let t2 = wall_ms.get(1).copied().unwrap_or(f64::NAN);
            let t4 = wall_ms.get(2).copied().unwrap_or(f64::NAN);
            let t8 = wall_ms.get(3).copied().unwrap_or(f64::NAN);
            let sp2 = if t2 > 0.0 { t1 / t2 } else { f64::NAN };
            let sp4 = if t4 > 0.0 { t1 / t4 } else { f64::NAN };
            let sp8 = if t8 > 0.0 { t1 / t8 } else { f64::NAN };
            eprintln!(
                "  {:30} {:8} {:8} {:8.1} {:8.1} {:8.1} {:8.1} {:5.2}× {:5.2}× {:5.2}×",
                name, n, nnz, t1, t2, t4, t8, sp2, sp4, sp8
            );
        }
        eprintln!();
    }

    /// Diagnostic: profile rayon-parallel lock contention across the
    /// large-matrix corpus. Wires
    /// `NumericParams::parallel_telemetry` and reports per-matrix
    /// wait/hold time on the two global mutexes
    /// (`contrib_blocks` and `node_factors_out`) plus the aggregate
    /// time spent inside `factor_one_supernode`. Aggregated body time
    /// across N workers can exceed wall time by up to N×, which
    /// reveals worker idleness when (body / N) < wall.
    ///
    /// Motivation: post-perf session 2026-05-12-01, cont-201 sits at
    /// ~30% of its 4.83× node-level parallel ceiling. Two suspects
    /// are global-mutex contention and rayon task-spawn overhead;
    /// this test produces evidence for/against the mutex hypothesis.
    ///
    /// Ignored by default — same gating as
    /// `solver_parallel_threadcount_sweep`.
    ///
    /// Invoke under release with:
    ///
    /// ```text
    /// cargo test --release solver_parallel_lock_breakdown \
    ///     -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn solver_parallel_lock_breakdown() {
        use crate::numeric::factorize::AtomicLockStats;
        use crate::read_mtx;
        use std::path::PathBuf;
        use std::sync::Arc;
        use std::time::Instant;

        let dir = PathBuf::from("tests/data/large");
        if !dir.is_dir() {
            eprintln!("SKIP: {} not found.", dir.display());
            return;
        }

        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            eprintln!("SKIP: no .mtx in {}.", dir.display());
            return;
        }

        // Use a single fixed pool size so the breakdown is
        // apples-to-apples across matrices. 4 threads strikes a
        // balance: enough to surface contention, not so many that
        // worker idleness obscures it.
        let n_threads = 4usize;
        let pool = match rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
        {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIP: rayon pool build failed: {}", e);
                return;
            }
        };

        eprintln!(
            "\n  Parallel lock-contention + phase breakdown (T={} threads)",
            n_threads
        );
        eprintln!(
            "  matrix                 wall_ms  body_ms_agg  body/T   contrib_wait_ms  contrib_hold_ms  nf_wait_ms  nf_hold_ms  n_tasks  body_frac  wait_frac"
        );
        eprintln!(
            "  ---------------------- -------- ----------- -------- ---------------- ---------------- ----------- ----------- -------- --------- ---------"
        );

        for path in &paths {
            let mtx = match read_mtx(path) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  SKIP {}: {:?}", path.display(), e);
                    continue;
                }
            };
            let csc = match mtx.to_csc() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("  SKIP {}: csc {:?}", path.display(), e);
                    continue;
                }
            };
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .trim_end_matches(".mtx");

            let stats = Arc::new(AtomicLockStats::default());
            let np = NumericParams {
                parallel_telemetry: Some(stats.clone()),
                fma: false,
                allow_delayed_pivots: true,
                cascade_break_ratio: None,
                cascade_break_eps: None,
                min_parallel_flops: None,
                ..NumericParams::default()
            };

            // First call pays the symbolic-analyze cost; second call
            // hits the Solver's pattern-fingerprint cache so wall ≈
            // pure numeric. This matches the pounce/IPM use case
            // where many factors reuse the same SymbolicFactorization.
            // We report the SECOND call's stats so the breakdown
            // reflects the production hot path.
            let (wall_ms, snap, wall_first_ms) = pool.install(|| {
                let mut solver = Solver::with_params(np, SupernodeParams::default());
                let t_first = Instant::now();
                let _ = solver.factor(&csc, None);
                let wall_first = t_first.elapsed().as_secs_f64() * 1e3;
                // Reset telemetry so the snapshot reflects only the
                // second call.
                stats
                    .contrib_wait_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .contrib_hold_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .node_factors_wait_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .node_factors_hold_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .factor_body_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .task_wall_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .ws_lock_wait_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats.n_tasks.store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .phase_scaling_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .phase_permute_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .phase_symmetric_pattern_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .phase_tree_setup_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .phase_thread_ws_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .phase_leaves_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .phase_scope_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                stats
                    .phase_collect_ns
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                let t0 = Instant::now();
                let _ = solver.factor(&csc, None);
                let wall = t0.elapsed().as_secs_f64() * 1e3;
                (wall, stats.snapshot(), wall_first)
            });

            eprintln!(
                "  --- {} (cold wall={:.1} ms, cached/2nd wall={:.1} ms) ---",
                name, wall_first_ms, wall_ms
            );

            let body_ms_agg = (snap.factor_body_ns as f64) / 1e6;
            let body_per_t = body_ms_agg / (n_threads as f64);
            let body_frac = if wall_ms > 0.0 {
                body_per_t / wall_ms
            } else {
                0.0
            };
            let total_wait_ms = (snap.contrib_wait_ns + snap.node_factors_wait_ns) as f64 / 1e6;
            let wait_frac_agg = if body_ms_agg > 0.0 {
                total_wait_ms / body_ms_agg
            } else {
                0.0
            };

            eprintln!(
                "  {:22} {:8.1} {:11.1} {:8.1} {:16.3} {:16.3} {:11.3} {:11.3} {:8} {:8.2}× {:8.2}%",
                name,
                wall_ms,
                body_ms_agg,
                body_per_t,
                snap.contrib_wait_ns as f64 / 1e6,
                snap.contrib_hold_ns as f64 / 1e6,
                snap.node_factors_wait_ns as f64 / 1e6,
                snap.node_factors_hold_ns as f64 / 1e6,
                snap.n_tasks,
                body_frac,
                wait_frac_agg * 100.0
            );
            // Per-phase breakdown of the (sequential) driver wrapper
            // — these run on the calling thread before/after the
            // rayon::scope. They form the "non-loop" floor that
            // bounds achievable parallel speedup, independent of how
            // many threads you give it.
            let scaling = snap.phase_scaling_ns as f64 / 1e6;
            let permute = snap.phase_permute_ns as f64 / 1e6;
            let sympat = snap.phase_symmetric_pattern_ns as f64 / 1e6;
            let tree = snap.phase_tree_setup_ns as f64 / 1e6;
            let tws = snap.phase_thread_ws_ns as f64 / 1e6;
            let leaves = snap.phase_leaves_ns as f64 / 1e6;
            let scope = snap.phase_scope_ns as f64 / 1e6;
            let collect = snap.phase_collect_ns as f64 / 1e6;
            let phase_sum = scaling + permute + sympat + tree + tws + leaves + scope + collect;
            let non_loop = phase_sum - scope;
            eprintln!(
                "    phases (ms): scaling={:.2} permute={:.2} sympat={:.2} tree={:.2} thread_ws={:.2} leaves={:.2} scope={:.2} collect={:.2}",
                scaling, permute, sympat, tree, tws, leaves, scope, collect,
            );
            eprintln!(
                "    sum_phases={:.2} ms,  non_loop (everything except rayon::scope)={:.2} ms,  scope/wall={:.2}",
                phase_sum,
                non_loop,
                if wall_ms > 0.0 { scope / wall_ms } else { 0.0 },
            );

            // Within-scope breakdown: where does the rayon::scope
            // wall time go? `scope` is the wall time of the
            // rayon::scope on the calling thread. We measure
            // `task_wall_agg`, the aggregate wall time of the
            // `scope.spawn` closure body across all tasks (includes
            // lock waits + factor_body + per-task control flow). The
            // gap `scope * T - task_wall_agg` is rayon idle (a
            // worker has no eligible task and is waiting), which
            // upper-bounds the parallelism deficit attributable to
            // the etree topology + scheduler. Within each task,
            // `task_wall_per_t - body_per_t - (contrib + nf + ws)`
            // is the per-task control-flow floor.
            let task_wall_agg = snap.task_wall_ns as f64 / 1e6;
            let task_wall_per_t = task_wall_agg / (n_threads as f64);
            let ws_wait = snap.ws_lock_wait_ns as f64 / 1e6;
            let scope_capacity = scope * (n_threads as f64);
            let rayon_idle = (scope_capacity - task_wall_agg).max(0.0);
            let in_task_locks = (snap.contrib_wait_ns
                + snap.contrib_hold_ns
                + snap.node_factors_wait_ns
                + snap.node_factors_hold_ns
                + snap.ws_lock_wait_ns) as f64
                / 1e6;
            let ctrl_flow_agg = (task_wall_agg - body_ms_agg - in_task_locks).max(0.0);
            eprintln!(
                "    within-scope: task_wall_agg={:.2} ms  task_wall/T={:.2} ms  ws_wait_agg={:.3} ms  in_task_locks_agg={:.2} ms  ctrl_flow_agg={:.2} ms  rayon_idle (scope·T − task_wall)={:.2} ms ({:.0}% of capacity)",
                task_wall_agg,
                task_wall_per_t,
                ws_wait,
                in_task_locks,
                ctrl_flow_agg,
                rayon_idle,
                if scope_capacity > 0.0 {
                    100.0 * rayon_idle / scope_capacity
                } else {
                    0.0
                },
            );
        }
        eprintln!();
    }

    /// Probe: what does `pick_scaling_strategy` return for each
    /// corpus matrix, and where does the wall time inside
    /// `phase_scaling_ns` actually live? Splits the 3.95 ms cont-201
    /// cached-mode scaling slice into (strategy pick) +
    /// (compute_scaling) + (scaling_pivot_order build).
    ///
    /// The hypothesis under test: the scaling phase's per-factor cost
    /// is unavoidable per-iteration value-dependent work (InfNorm
    /// must re-run because it depends on values, not pattern), NOT a
    /// missed cache. If true, the 3.95 ms is fundamental and not
    /// recoverable for the IPM hot path. If false (e.g. the
    /// strategy-pick or scaling_pivot_order build dominates),
    /// there is engineering work available.
    #[test]
    #[ignore]
    fn solver_scaling_phase_split() {
        use crate::read_mtx;
        use crate::scaling::{compute_scaling_with_cache, pick_scaling_strategy};
        use std::path::PathBuf;
        use std::time::Instant;

        let dir = PathBuf::from("tests/data/large");
        if !dir.is_dir() {
            eprintln!("SKIP: {} not found.", dir.display());
            return;
        }

        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            eprintln!("SKIP: no .mtx in {}.", dir.display());
            return;
        }

        eprintln!("\n  Scaling-phase split (Auto strategy default)");
        eprintln!(
            "  matrix                 n       nnz     picked        pick_ms  scale_ms  reorder_ms  total_ms"
        );
        eprintln!(
            "  ---------------------- ------- ------- ------------- -------  --------  ----------  --------"
        );

        for path in &paths {
            let mtx = match read_mtx(path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let csc = match mtx.to_csc() {
                Ok(c) => c,
                Err(_) => continue,
            };
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .trim_end_matches(".mtx");

            // Strategy pick: scans col_ptr (O(n)) for diag-only count.
            let t0 = Instant::now();
            let picked = pick_scaling_strategy(&csc);
            let pick_ms = t0.elapsed().as_secs_f64() * 1e3;

            // Compute scaling itself with the resolved strategy.
            // We deliberately pass `cache = None` here so the timing
            // reflects the path the Solver hits when no MC64 cache
            // was built (most non-arrow matrices). For MC64 cases we
            // would need the cache; documented below.
            let t1 = Instant::now();
            let (scaling_vec, _info) = match compute_scaling_with_cache(&csc, &picked, None) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let scale_ms = t1.elapsed().as_secs_f64() * 1e3;

            // Reorder: O(n) gather under symbolic.perm. We don't have
            // the symbolic factorize here; use identity perm to time
            // the gather kernel itself. This upper-bounds the cache-
            // friendly case; the real path has a non-identity perm.
            let perm: Vec<usize> = (0..csc.n).collect();
            let t2 = Instant::now();
            let _reordered: Vec<f64> = perm.iter().map(|&old| scaling_vec[old]).collect();
            let reorder_ms = t2.elapsed().as_secs_f64() * 1e3;

            let total = pick_ms + scale_ms + reorder_ms;
            let picked_label = format!("{:?}", picked);
            eprintln!(
                "  {:22} {:7} {:7} {:13} {:7.3}  {:7.3}   {:7.3}     {:7.3}",
                name,
                csc.n,
                csc.nnz(),
                picked_label,
                pick_ms,
                scale_ms,
                reorder_ms,
                total
            );
        }
        eprintln!();
    }

    /// Bit-exact parity: factoring the same matrix under the
    /// parallel driver and the sequential driver must produce
    /// identical summed inertia and identical `solve(rhs)` output.
    /// The parallel driver documents bit-exact parity (same FP sum
    /// order per supernode, per-thread workspaces, mutex only on
    /// the contribution-block store), so this is asserted with
    /// `==`, not a tolerance. Per CLAUDE.md hard rules, do not
    /// loosen this to a tolerance without recorded justification.
    ///
    /// Fixture: 64 independent 2×2 indefinite blocks `[[1, 2],
    /// [2, 1]]` give n = 128 with 64 disjoint elimination trees,
    /// well above the `N_PAR_MIN = 32` gate so the parallel driver
    /// actually dispatches the rayon task graph rather than falling
    /// through to the sequential path.
    #[test]
    fn solver_parallel_factor_matches_sequential() {
        const N_BLOCKS: usize = 64;
        let n = 2 * N_BLOCKS;
        let mut rows = Vec::with_capacity(3 * N_BLOCKS);
        let mut cols = Vec::with_capacity(3 * N_BLOCKS);
        let mut vals = Vec::with_capacity(3 * N_BLOCKS);
        for b in 0..N_BLOCKS {
            let i = 2 * b;
            // Lower triangle of [[1, 2], [2, 1]] per block.
            rows.push(i);
            cols.push(i);
            vals.push(1.0);
            rows.push(i + 1);
            cols.push(i);
            vals.push(2.0);
            rows.push(i + 1);
            cols.push(i + 1);
            vals.push(1.0);
        }
        let csc = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();

        // Deterministic RHS: 1..=n as f64.
        let rhs: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let mut par = Solver::new();
        assert!(par.parallel());
        assert!(matches!(par.factor(&csc, None), FactorStatus::Success));
        let par_factors = par.factors().expect("parallel factors");
        let par_inertia =
            par_factors
                .node_factors
                .iter()
                .fold((0usize, 0usize, 0usize), |(p, ng, z), nf| {
                    (
                        p + nf.inertia.positive,
                        ng + nf.inertia.negative,
                        z + nf.inertia.zero,
                    )
                });
        let par_n_supernodes = par_factors.node_factors.len();
        assert!(
            par_n_supernodes >= crate::numeric::factorize::N_PAR_MIN,
            "fixture should produce >= N_PAR_MIN supernodes, got {}",
            par_n_supernodes
        );
        let par_neg = par.num_negative_eigenvalues();
        let par_x = par.solve(&rhs).expect("parallel solve");

        let mut seq = Solver::new().with_parallel(false);
        assert!(!seq.parallel());
        assert!(matches!(seq.factor(&csc, None), FactorStatus::Success));
        let seq_inertia = seq
            .factors()
            .expect("sequential factors")
            .node_factors
            .iter()
            .fold((0usize, 0usize, 0usize), |(p, ng, z), nf| {
                (
                    p + nf.inertia.positive,
                    ng + nf.inertia.negative,
                    z + nf.inertia.zero,
                )
            });
        let seq_neg = seq.num_negative_eigenvalues();
        let seq_x = seq.solve(&rhs).expect("sequential solve");

        assert_eq!(par_inertia, seq_inertia, "summed inertia mismatch");
        assert_eq!(par_neg, seq_neg, "num_negative_eigenvalues mismatch");
        for (i, (a, b)) in par_x.iter().zip(seq_x.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "solve[{}] differs: parallel = {} ({:#x}), sequential = {} ({:#x})",
                i,
                a,
                a.to_bits(),
                b,
                b.to_bits()
            );
        }
    }
}
