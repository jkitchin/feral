use crate::dense::factor::{
    factor, factor_frontal_blocked_in_place_with_scratch, factor_frontal_diagonal_in_place,
    phase_timing, BunchKaufmanParams, FactorScratch, FrontalFactors, ZeroPivotAction,
};
use crate::dense::matrix::SymmetricMatrix;
use crate::error::FeralError;
use crate::inertia::Inertia;
use crate::scaling::{compute_scaling_dense_fast, compute_scaling_with_cache, ScalingStrategy};
use crate::sparse::csc::CscMatrix;
use crate::symbolic::{Supernode, SymbolicFactorization};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Per-supernode tracing gate. Set `FERAL_TRACE_SUPERNODE=1` (or any
/// non-empty value) in the environment to log one line per
/// supernode factor call: `[sn-trace] sn=<idx> nrow=<r> exp_ncol=<c>
/// n_del_in=<d> may_del=<m> cb=<cb> nelim=<n> n_del_out=<dout>
/// rook_rescues=<rr> ms=<ms>`. Off by default — the OnceLock keeps
/// the check at one atomic load per supernode call. Used by the
/// wide-supernode cascade investigation
/// (`dev/research/warm-state-cascade-amplification-2026-05-17.md`)
/// to identify which supernode burns time on cascade-prone matrices.
fn supernode_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FERAL_TRACE_SUPERNODE")
            .map(|v| !v.is_empty() && v != "0" && v != "off" && v != "false")
            .unwrap_or(false)
    })
}

/// Symbolic-arm gate for the cascade-break trigger
/// (`NumericParams::cascade_break_ratio`). When the symbolic factor
/// reports `symbolic.n < CASCADE_BREAK_MIN_N`, the trigger is
/// guaranteed to be a no-op regardless of how it is configured —
/// cascade-break savings only accumulate when some front can grow,
/// via delay propagation, to several thousand columns, and the
/// achievable expanded ncol is bounded above by `n`. Issue #15
/// (2026-05-14). See
/// `dev/research/issue-15-cascade-break-symbolic-arm.md`.
pub const CASCADE_BREAK_MIN_N: usize = 4096;

/// Numeric-phase parameters bundle.
///
/// Groups the dense Bunch-Kaufman pivot configuration with the
/// global symmetric scaling strategy. Both are numeric-time
/// choices — they depend on the matrix values, not the sparsity
/// pattern. Keeping them together at the numeric entry point
/// (rather than splitting `bk` into the BK call and `scaling`
/// into the symbolic call) lets the symbolic factorization stay
/// value-agnostic and therefore reusable across multiple numeric
/// factorizations of structurally identical KKTs (the IPM use
/// case). See `dev/research/pounce-integration-interface.md` and
/// `dev/plans/scaling-in-numeric.md` (β refactor).
#[derive(Debug, Clone)]
pub struct NumericParams {
    /// Dense BK kernel parameters.
    pub bk: BunchKaufmanParams,
    /// Global symmetric scaling strategy applied at the start of
    /// numeric factorization.
    pub scaling: ScalingStrategy,
    /// Phase 2.9 small-leaf-subtree batching gate. Default `Off`
    /// preserves the reference per-supernode driver. When `On` the
    /// driver processes `SymbolicFactorization::small_leaf_groups`
    /// via `factor_one_small_leaf` instead of the generic
    /// `factor_one_supernode`, skipping the per-leaf
    /// `build_row_indices` call. See
    /// `dev/plans/phase-2.9-small-leaf-subtree.md`.
    pub small_leaf: SmallLeafBatch,
    /// Phase 2.10 per-supernode profiler. When `Some`, the sequential
    /// driver records per-supernode timings, plus prologue/epilogue
    /// costs, into the shared `Profiler`. When `None` (default), no
    /// timing work runs — zero overhead in production. See
    /// `dev/plans/phase-2.10-supernode-profiler.md`.
    pub profiler: Option<Arc<Mutex<Profiler>>>,
    /// Optional lock-contention telemetry for the rayon-parallel
    /// multifrontal driver. When `Some`, the driver records the
    /// wait+hold time spent on the shared `contrib_blocks` and
    /// `node_factors_out` mutexes, plus a task count. When `None`
    /// (default) the driver performs a single Option-is_none check
    /// per lock acquire — branch-predicted away in production.
    /// Diagnostic only; the values reported are not part of any
    /// correctness contract. See `dev/sessions/2026-05-12-01.md`
    /// "Next Session Should" for the cont-201 investigation that
    /// motivates this hook.
    pub parallel_telemetry: Option<Arc<AtomicLockStats>>,
    /// Opt-in FMA dispatch on the dense trailing-update / panel-update
    /// kernels. Default `false` preserves the cross-arch bit-exactness
    /// invariant of the production `*_nofma` kernels (one rounding per
    /// `mul` plus one per `sub`); when `true`, the dense factor /
    /// block_ldlt32 paths dispatch to the FMA siblings
    /// (`schur_panel_minus_fma_strided*`, `axpy_minus_unroll4`,
    /// `axpy2_minus_unroll4`) which fuse the multiply-accumulate into
    /// one `mul_add` per step (single rounding, ~2x arithmetic
    /// throughput on aarch64 NEON and x86 V3 AVX2+FMA).
    ///
    /// Trade-off: the FMA path is **not** bit-exact with the non-FMA
    /// path; per-element drift is bounded by `n_elim * ULP` (see
    /// `dense::schur_kernel::fma_vs_nofma_panel_kernels_within_n_elim_ulps`
    /// for the contract). Inertia is unchanged on well-conditioned
    /// matrices; residuals match within `64 * EPS`. Opt in when
    /// throughput matters more than cross-policy bit-identity (large
    /// supernode workloads like Mittelmann's pinene_3200 NLP — see
    /// issue #8 and `dev/research/fma-kernel-opt-in.md`).
    pub fma: bool,

    /// When `true` (default), non-root supernodes run with
    /// `may_delay = true` — pivots that fail the column-relative
    /// threshold or the 2×2 Duff-Reid growth bound are pushed up
    /// the elimination tree to the parent (SSIDS-style delayed
    /// pivoting). When `false`, every supernode runs as if it
    /// were the root (`may_delay = false`): failing pivots are
    /// force-accepted in place via the existing
    /// `ZeroPivotAction::ForceAccept` path, with iterative
    /// refinement to recover residual.
    ///
    /// Disabling delayed pivoting is the FERAL analogue of MA57's
    /// `cntl[4]` static-pivoting fallback. Issue #8 (Mittelmann
    /// `pinene_3200_0009`) hits a delayed-pivot cascade: 118k
    /// pivots delayed up to ~14k-column root supernodes, yielding
    /// an 87s factor on an otherwise sub-second problem. Setting
    /// `allow_delayed_pivots = false` breaks the cascade at the
    /// cost of bounded L growth (`O(1/|d|)` per force-accepted
    /// small pivot), which iterative refinement is expected to
    /// recover.
    ///
    /// Default `true` preserves the SSIDS-canonical behavior the
    /// FERAL corpus is verified against. Flip per-call via
    /// `Solver::with_static_pivoting(true)` for the issue #8 fast
    /// path. See `dev/journal/2026-05-13-03.org` 22:55 entry for
    /// the root-cause analysis.
    pub allow_delayed_pivots: bool,

    /// Adaptive static-pivoting trigger. When `Some(r)`, a non-root
    /// supernode whose `n_delayed_in / expanded_ncol >= r` flips to
    /// `may_delay = false` for that one supernode only, with a
    /// locally-overridden `on_zero_pivot = ForceAccept` to absorb
    /// failures in place. The rest of the etree keeps SSIDS-style
    /// delayed pivoting. Default `None` (disabled).
    ///
    /// Issue #8 motivation: on `pinene_3200_0009`, METIS-ND
    /// concentrates 118k delays into three ~14k-column expanded
    /// fronts (87s factor). The cascade-break trigger lets the
    /// easy iterates keep their cheap delayed-pivot path while
    /// breaking the cliff iterates at the overloaded node.
    /// A starting value of `0.5` ("front is at least 50% delayed
    /// columns") catches the cascade signature without firing on
    /// light-delay nodes — calibrate against the corpus before
    /// promoting to default.
    ///
    /// Symbolic-arm gate (issue #15, 2026-05-14): the trigger is
    /// additionally guarded by `symbolic.n >= CASCADE_BREAK_MIN_N`,
    /// because cascade-break savings only accumulate when some
    /// front can grow (via delay propagation) to several thousand
    /// columns — bounded above by `n`. Below the threshold the
    /// trigger is a no-op even when armed. See
    /// `dev/research/issue-15-cascade-break-symbolic-arm.md`.
    pub cascade_break_ratio: Option<f64>,

    /// Per-pivot perturbation floor for cascade-break supernodes.
    /// When `cascade_break_ratio` fires AND this is `Some(eps)`, the
    /// triggered supernode runs with
    /// `on_zero_pivot = PerturbToEps { abs_floor: eps }`, replacing
    /// each tiny pivot by `sign(d) * max(|d|, eps)` and counting it
    /// by sign rather than zero. The factor then satisfies
    /// `LDL^T = A + Δ` with `||Δ||_∞ <= eps` per perturbed pivot —
    /// inertia is preserved provided every nonzero eigenvalue of
    /// `A` exceeds the cumulative perturbation. Default `None`
    /// keeps the legacy `ForceAccept` semantics (unbounded Δ,
    /// counts perturbed pivots as zero). See
    /// `dev/journal/2026-05-13-03.org` 01:15 for the matrix-specific
    /// "sweet spot" pathology that motivates bounded-Δ
    /// perturbation.
    pub cascade_break_eps: Option<f64>,

    /// Override the minimum estimated tree-flop count at which
    /// [`should_parallelize_assembly`] is willing to dispatch the
    /// rayon-parallel driver. `None` (default) uses [`PAR_MIN_FLOPS`].
    ///
    /// Issue #19 follow-up: rayon spawn / cv-wait overhead is
    /// hardware-dependent; the default const is calibrated for the
    /// reporter's machine. Consumers that have measured their own
    /// break-even point can override here. Set to `Some(0)` to
    /// disable the flop gate entirely (still subject to `N_PAR_MIN`
    /// and the structural multi-child gate); set to `Some(u64::MAX)`
    /// to force the gate to always reject (functionally equivalent
    /// to `Solver::with_parallel(false)` for tree-level dispatch).
    pub min_parallel_flops: Option<u64>,

    /// Opt-in symmetric-quasi-definite (SQD) fast-path. When `true`,
    /// the caller asserts the input KKT has Vanderbei (1995) structure
    /// `K = [[-E, A^T], [A, F]]` with `E, F` symmetric positive
    /// definite — the common case in IPOPT after the first inertia
    /// correction sets `δ_w, δ_c > 0`, and structural in IP-PMM
    /// (Pougkakiotis-Gondzio 2020). Under this contract every
    /// symmetric permutation admits an `LDL^T` with **purely
    /// diagonal `D`** (Vanderbei Thm 2.1), so the per-supernode
    /// Bunch-Kaufman 1x1-vs-2x2 search can be skipped entirely.
    ///
    /// Default `false` preserves the unconditional BK + delayed-pivot
    /// path that every existing caller is verified against. Mutually
    /// exclusive with `allow_delayed_pivots = true` and with
    /// `cascade_break_ratio = Some(_)`; the `Solver::with_sqd_mode`
    /// builder enforces the invariant by clearing both fields when
    /// `sqd_mode` is enabled. Contract violations at runtime surface
    /// as `FeralError::SqdContractViolated` (loud failure, no silent
    /// BK fallback) — see commit (e) of the M7 phasing.
    ///
    /// See `dev/research/sqd-fast-path.md`, `dev/decisions.md`
    /// 2026-05-16 entry, and issue #34.
    pub sqd_mode: bool,

    /// MA57-style static-pivot perturbation threshold (issue #38).
    /// When `Some(t)`, `Solver::factor` derives an absolute floor
    /// `static_pivot_floor = t * ||D·A·D||_∞` from the *scaled* matrix
    /// norm (post-scaling, via `scaled_matrix_infnorm` +
    /// `apply_post_scaling_overrides`; N2) and propagates it into
    /// `BunchKaufmanParams.static_pivot_floor` for that factor call.
    /// Every accepted 1×1 / 2×2 pivot whose magnitude (for 2×2:
    /// smallest |eigenvalue|) is below the floor is perturbed up to
    /// the floor and counted by sign. The factor satisfies
    /// `LDL^T = A + Δ` with `||Δ||_F ≤ floor` per perturbed pivot.
    ///
    /// Default `None` (disabled). Recommended starting value for IPM
    /// use: `1e-8` (matches MA57's `cntl[0]` default). The C ABI
    /// reads `FERAL_STATIC_PIVOT=<float>` to set this without a
    /// rebuild.
    ///
    /// Inertia is then reported for the perturbed `A + Δ`, not `A` —
    /// this is the whole point: bend small-magnitude pivots so the
    /// returned inertia matches the IPM's expectation, cutting the
    /// PDPerturbationHandler δ_w escalation cost. Iterative
    /// refinement against unperturbed `A` (the default in `Solver::
    /// solve_many_refined`) recovers solve accuracy.
    ///
    /// See `dev/research/static-pivot-perturbation-2026-05-17.md`
    /// and `dev/journal/2026-05-17-01.org` §16:30 / §17:10.
    pub static_pivot_threshold: Option<f64>,

    /// When `true`, the numeric drivers emit a one-line stderr
    /// `warning:` whenever MC64 matching leaves variables unmatched
    /// (`ScalingInfo::PartialSingular`) and scaling falls back to
    /// identity on those rows/columns. Default `false`.
    ///
    /// `PartialSingular` is routine and benign for IPM hosts, which
    /// factorize structurally rank-deficient KKT systems on the first
    /// attempt of most iterations; an unconditional stderr write
    /// floods host logs for behavior that is expected and recovered
    /// downstream. The same information is always available
    /// structurally via `Solver::scaling_info()` (and as a count via
    /// `Solver::mc64_fallback_count` for the `Auto`-fallback case),
    /// so the stderr line is an opt-in diagnostic breadcrumb, not a
    /// correctness signal. Default `false` keeps feral quiet as a
    /// library should be; enable it via
    /// `Solver::with_partial_singular_warning(true)` or the
    /// `FERAL_WARN_PARTIAL_SINGULAR` env var (C ABI). Issue #43.
    pub warn_partial_singular: bool,

    /// Issue #56 Lever A.2: when `true`, the sequential and Schur numeric
    /// drivers consult `FactorWorkspace::permute_cache` to skip the
    /// `CscMatrix::from_triplets` rebuild inside `permute_csc_values`,
    /// reusing the cached `(col_ptr, row_idx, value_map)` and scattering
    /// only the values. Set by `Solver::factor` to `pattern_reused` —
    /// the same fingerprint-equality signal that drives symbolic-cache
    /// reuse, which guarantees the cached permute structure is still
    /// valid. Default `false` keeps direct callers
    /// (`factorize_multifrontal_supernodal_with_workspace` used without
    /// `Solver`) on the canonical from-triplets path. NOTE: the parallel
    /// driver (the default on the large matrices this targets) does not
    /// engage the cache regardless of this flag — it always rebuilds via
    /// `permute_csc_values`; closing that gap is the open N3 facet tracked
    /// in `dev/decisions.md`.
    pub pattern_reused_hint: bool,
}

/// Gate for Phase 2.9 small-leaf-subtree batching.
///
/// When `Off` (default), `factorize_multifrontal_supernodal_with_
/// workspace` runs the generic per-supernode body on every
/// supernode. When `On`, leaf supernodes that were grouped at
/// symbolic time are routed through the batched path.
///
/// Default is `Off`. Phase 2.11 attempted a default flip after a
/// single-run measurement appeared to show 24-27% reduction on the
/// tiny-IPM tail; a 5-run repeat showed the effect was within
/// ~5% measurement noise (see `dev/tried-and-rejected.md` Phase
/// 2.11). The flip was reverted; the tail gap is structural
/// (bushy elimination tree) and needs a column-renumbering
/// refactor (see `dev/plans/phase-2.12-*` once written).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SmallLeafBatch {
    #[default]
    Off,
    On,
}

/// One supernode's timing record. Phase 2.10
/// (`dev/plans/phase-2.10-supernode-profiler.md`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SupernodeTiming {
    pub snode_idx: usize,
    pub nrow: usize,
    pub ncol: usize,
    pub us: u64,
    /// Phase-breakdown deltas (issue #44 phase-probe). All zero unless
    /// `dense::factor::PHASE_TIMING_ENABLED` is set. `assembly_us` covers
    /// `build_row_indices` + original-entry scatter + child extend-add;
    /// `densefactor_us` is the whole dense frontal factor and decomposes
    /// into `panelfactor_us` (panel/diagonal BK) + `schur_us` (deferred
    /// Schur trailing update) + `scalartail_us` (scalar pivot tail);
    /// `densefactor_us - panel - schur - scalartail` is dense-factor
    /// bookkeeping overhead.
    #[serde(default)]
    pub assembly_us: u64,
    #[serde(default)]
    pub densefactor_us: u64,
    #[serde(default)]
    pub panelfactor_us: u64,
    #[serde(default)]
    pub schur_us: u64,
    #[serde(default)]
    pub scalartail_us: u64,
}

/// Per-sub-phase wallclock breakdown of the numeric prologue —
/// everything `factorize_multifrontal_supernodal_with_workspace` does
/// before the per-supernode loop. Populated only when a `Profiler` is
/// attached; every field is zero on the default (un-profiled) path.
///
/// Added for the per-factor cost cluster, Track B1
/// (`dev/plans/per-factor-cost-cluster.md`): on `rocket_12800` the
/// prologue is 99.5% of factor wall (3.4 s vs a 10 ms supernode loop)
/// and the cause is not visible from static reading, so it has to be
/// attributed empirically.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PrologueBreakdown {
    /// `row_map` clear + resize to `n`.
    pub row_map_us: u64,
    /// `compute_scaling_with_cache` (MC64 / InfNorm / cache replay).
    pub scaling_us: u64,
    /// Building the pivot-order scaling vector.
    pub scaling_pivot_order_us: u64,
    /// `permute_csc_values` (P·A·Pᵀ rebuild) — total.
    pub permute_us: u64,
    /// The `CscMatrix::from_triplets` sub-call inside `permute_csc_values`.
    /// Subset of `permute_us`; the prime suspect for the prologue cost.
    pub permute_from_triplets_us: u64,
    /// `scaled_matrix_infnorm` + `apply_post_scaling_overrides`.
    pub infnorm_tol_us: u64,
    /// `CscMatrix::symmetric_pattern` (full pattern for row indices).
    pub symmetric_pattern_us: u64,
    /// `is_root` flags + `contrib_blocks` / `node_factors` allocation.
    pub setup_us: u64,
}

/// Per-invocation profiler for `factorize_multifrontal_supernodal_with_workspace`.
///
/// Attached to `NumericParams::profiler` as `Some(Arc<Mutex<Profiler>>)`
/// to record per-supernode timings, prologue and epilogue costs. When
/// the field is `None` the driver does no timing work — zero overhead.
///
/// The profiler is a diagnostic, not a correctness path. A poisoned
/// mutex (only possible if a panic happened while holding the lock,
/// which the driver code paths do not do) is silently ignored: the
/// affected sample is dropped, factorization continues, and the
/// `report()` validation invariants surface the gap.
#[derive(Debug, Clone, Default)]
pub struct Profiler {
    timings: Vec<SupernodeTiming>,
    prologue_us: u64,
    prologue_breakdown: PrologueBreakdown,
    epilogue_us: u64,
    total_us: u64,
}

impl Profiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of supernode timing samples recorded.
    pub fn len(&self) -> usize {
        self.timings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timings.is_empty()
    }

    /// Raw per-supernode timings in driver order.
    pub fn timings(&self) -> &[SupernodeTiming] {
        &self.timings
    }

    /// Compute the bucketed report from accumulated samples.
    pub fn report(&self) -> ProfileReport {
        const RANGES: &[(&str, usize, usize)] = &[
            ("<=8", 0, 8),
            ("9-16", 9, 16),
            ("17-32", 17, 32),
            ("33-64", 33, 64),
            ("65-128", 65, 128),
            (">128", 129, usize::MAX),
        ];

        let mut buckets: Vec<BucketStats> = RANGES
            .iter()
            .map(|&(range, _, _)| BucketStats {
                range,
                count: 0,
                sum_us: 0,
                pct_of_total: 0.0,
                avg_us: 0.0,
            })
            .collect();

        for t in &self.timings {
            for (i, &(_, lo, hi)) in RANGES.iter().enumerate() {
                if t.nrow >= lo && t.nrow <= hi {
                    buckets[i].count += 1;
                    buckets[i].sum_us += t.us;
                    break;
                }
            }
        }

        let loop_us: u64 = buckets.iter().map(|b| b.sum_us).sum();

        let mut warnings: Vec<String> = Vec::new();
        let count_sum: usize = buckets.iter().map(|b| b.count).sum();
        if count_sum != self.timings.len() {
            warnings.push(format!(
                "bucket count sum {} != timings len {}",
                count_sum,
                self.timings.len()
            ));
        }
        if self.total_us > 0 && loop_us + self.prologue_us + self.epilogue_us > self.total_us {
            warnings.push(format!(
                "loop+prologue+epilogue ({}) exceeds total ({})",
                loop_us + self.prologue_us + self.epilogue_us,
                self.total_us
            ));
        }

        for b in &mut buckets {
            if loop_us > 0 {
                b.pct_of_total = (b.sum_us as f64) * 100.0 / (loop_us as f64);
            }
            if b.count > 0 {
                b.avg_us = (b.sum_us as f64) / (b.count as f64);
            }
        }

        let overhead_pct = if self.total_us > 0 {
            ((self.prologue_us + self.epilogue_us) as f64) * 100.0 / (self.total_us as f64)
        } else {
            0.0
        };

        ProfileReport {
            n_supernodes: self.timings.len(),
            prologue_us: self.prologue_us,
            prologue_breakdown: self.prologue_breakdown.clone(),
            epilogue_us: self.epilogue_us,
            loop_us,
            total_us: self.total_us,
            overhead_pct,
            buckets,
            validation_warnings: warnings,
        }
    }
}

/// One front-size bucket in the profile histogram.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BucketStats {
    pub range: &'static str,
    pub count: usize,
    pub sum_us: u64,
    pub pct_of_total: f64,
    pub avg_us: f64,
}

/// Aggregated profile report. Serializable for diagnostic dumps.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProfileReport {
    pub n_supernodes: usize,
    pub prologue_us: u64,
    /// Sub-phase attribution of `prologue_us` (Track B1). Zero on the
    /// un-profiled path; sub-phases sum to slightly less than
    /// `prologue_us` (a few un-timed O(1) checks are not attributed).
    pub prologue_breakdown: PrologueBreakdown,
    pub epilogue_us: u64,
    /// Sum of per-supernode timings — the inner-loop wallclock.
    pub loop_us: u64,
    /// Total wallclock for the entire driver call.
    pub total_us: u64,
    pub overhead_pct: f64,
    pub buckets: Vec<BucketStats>,
    pub validation_warnings: Vec<String>,
}

impl Default for NumericParams {
    /// Sparse-multifrontal default. Sets `bk.pivot_threshold = 1e-8`
    /// to match MA27's `cntl[1]` default, which is also Ipopt's
    /// `ma27_pivtol` default. This activates the column-relative
    /// pivot rejection (and downstream rook rescue / delayed
    /// pivoting) on rank-deficient KKT-augmented systems while
    /// staying conservative enough not to reject legitimate pivots
    /// on Identity-scaled (un-equilibrated) matrices that consumers
    /// like ripopt feed in directly.
    ///
    /// `BunchKaufmanParams::default()` (the dense entry point used
    /// directly by `dense::factor::factor`) intentionally stays at
    /// `pivot_threshold = 0.0` per the 2026-04-13 dense-vs-sparse
    /// split: dense has no delayed-pivoting / rook-rescue
    /// infrastructure to land rejected pivots in. See
    /// `dev/decisions.md:325-344` and
    /// `dev/research/issue-2-kkt-pivot-default.md`.
    ///
    /// Why `1e-8` instead of the SSIDS/MUMPS canonical `0.01`:
    /// `0.01` was validated on MC64-equilibrated matrices where
    /// `|d| >= 0.01 * col_max` rejects pivots that are tiny relative
    /// to a normalized column. ripopt's KKT path runs Identity
    /// scaling (it owns scaling at a higher layer to preserve the
    /// inertia signal — see ripopt `feral_direct.rs:84-91`), so the
    /// threshold fires on raw-value ratios that have not been
    /// equilibrated. `1e-8` matches Ipopt's reference choice for
    /// exactly this configuration; the in-tree sparse callers that
    /// run with MC64/InfNorm scaling continue to override
    /// explicitly to `0.01`.
    ///
    /// Issue #2 surfaced the gap: ripopt and other consumers that
    /// build `NumericParams::default()` were inheriting `0.0` (via
    /// `BunchKaufmanParams::default()`), which silently disabled
    /// every saddle-point rescue path on rank-deficient KKT-augmented
    /// LS-init systems and caused exact-zero multipliers on
    /// non-structurally-zero rows.
    fn default() -> Self {
        Self {
            // `on_zero_pivot: ForceAccept` overrides the dense
            // `BunchKaufmanParams::default()` `Fail` — see F-03
            // (`dev/research/f03-bloweybl-rank-rejection.md`, issue
            // #32). The sparse multifrontal path has delayed-pivot
            // infrastructure that lets the eliminator recover from
            // an isolated zero pivot at the root (the dense entry
            // point does not), and the two reference oracles MUMPS
            // and MA57 both produce a usable factor with one zero
            // pivot on `GHS_indef/bloweybl`. Counting the zero in
            // `inertia.zero` matches MUMPS `INFOG(28)` and MA57
            // semantics. Dense callers that want abort-on-zero
            // continue to get it via `BunchKaufmanParams::default()`;
            // sparse callers that want it must opt in explicitly.
            bk: BunchKaufmanParams {
                pivot_threshold: 1e-8,
                on_zero_pivot: ZeroPivotAction::ForceAccept,
                ..BunchKaufmanParams::default()
            },
            scaling: ScalingStrategy::default(),
            small_leaf: SmallLeafBatch::default(),
            profiler: None,
            parallel_telemetry: None,
            fma: false,
            allow_delayed_pivots: true,
            // Phase B (issue #55): cascade-break is armed by default
            // as the recovery path for symbolic-analysis-time delay
            // budget exhaustion. The presence of
            // `cascade_break_ratio = Some(_)` means "CB is armed";
            // the numeric ratio value is only consulted on the legacy
            // path (unbudgeted supernodes, `delayed_capacity ==
            // usize::MAX`). On the budgeted path the trigger is
            // `n_delayed_in > delayed_capacity`, mirroring MUMPS's
            // `dfac_front_aux.F:1251-1331` "delay capacity exhausted"
            // perturbation branch.
            //
            // The earlier disarm (cascade-break off by default) was
            // motivated by the Weyl-bound concern documented in
            // `dev/research/cascade-break-l-perturbation-2026-05-15.md`:
            // per-pivot `||Δ||_∞ <= eps` does not hold strictly when
            // L is scaled by `1/d_new`. Phase B closes that gap not
            // by fixing the Weyl bound (it cannot be tightened
            // without changing the L kernel) but by ensuring CB
            // only fires when delay was structurally impossible —
            // matching MUMPS's invariant. See issue #55 and
            // `dev/research/symbolic-delay-budget-2026-05-27.md`.
            cascade_break_ratio: Some(0.5),
            cascade_break_eps: Some(1e-10),
            min_parallel_flops: None,
            // SQD fast-path off by default. Opt in via
            // `Solver::with_sqd_mode(true)`; see `sqd_mode` doc and
            // `dev/research/sqd-fast-path.md`.
            sqd_mode: false,
            // Issue #38: static-pivot perturbation is opt-in. Default
            // None preserves the canonical BK inertia (matches MUMPS /
            // SSIDS / rmumps for non-perturbed problems). Enable per
            // call via `Solver::with_static_pivot_threshold(t)` or via
            // the `FERAL_STATIC_PIVOT=<float>` env var (C ABI path).
            static_pivot_threshold: None,
            // Issue #43: the MC64 partial-singular notice is a
            // routine, benign condition for IPM hosts. Default `false`
            // keeps the library quiet; the structured signal is always
            // on `Solver::scaling_info()`. Enable via
            // `Solver::with_partial_singular_warning(true)` or
            // `FERAL_WARN_PARTIAL_SINGULAR` (C ABI).
            warn_partial_singular: false,
            // Issue #56 Lever A.2: opt-in permute cache. Solver flips
            // this on per-call when the symbolic cache reports
            // `pattern_reused`.
            pattern_reused_hint: false,
        }
    }
}

impl NumericParams {
    /// Construct a `NumericParams` from a `BunchKaufmanParams`,
    /// using the default scaling strategy. Convenience for
    /// callers that only customize BK behavior. The supplied `bk`
    /// is used verbatim — no `pivot_threshold` override is applied,
    /// in contrast to `Default::default()`.
    pub fn with_bk(bk: BunchKaufmanParams) -> Self {
        Self {
            bk,
            scaling: ScalingStrategy::default(),
            small_leaf: SmallLeafBatch::default(),
            profiler: None,
            parallel_telemetry: None,
            fma: false,
            allow_delayed_pivots: true,
            // Match `Default::default()` — CB armed as the recovery
            // path for delay-budget exhaustion. See the comment there
            // for rationale.
            cascade_break_ratio: Some(0.5),
            cascade_break_eps: Some(1e-10),
            min_parallel_flops: None,
            sqd_mode: false,
            static_pivot_threshold: None,
            // Quiet by default — see `Default::default()` and #43.
            warn_partial_singular: false,
            // Issue #56 Lever A.2: opt-in permute cache.
            pattern_reused_hint: false,
        }
    }
}

/// Lock-contention + phase telemetry for the rayon-parallel
/// multifrontal driver. Atomic fields aggregate across worker
/// threads via `fetch_add`. Driver-phase fields are written once by
/// the calling thread and use atomics only for uniformity (no
/// contention there). Snapshot via `snapshot()` after the factor
/// returns. See `NumericParams::parallel_telemetry`.
#[derive(Default, Debug)]
pub struct AtomicLockStats {
    /// Cumulative time spent waiting to acquire `contrib_blocks`.
    pub contrib_wait_ns: std::sync::atomic::AtomicU64,
    /// Cumulative time spent holding `contrib_blocks` (excluding wait).
    pub contrib_hold_ns: std::sync::atomic::AtomicU64,
    /// Cumulative time spent waiting to acquire `node_factors_out`.
    pub node_factors_wait_ns: std::sync::atomic::AtomicU64,
    /// Cumulative time spent holding `node_factors_out` (excluding wait).
    pub node_factors_hold_ns: std::sync::atomic::AtomicU64,
    /// Cumulative time spent inside `factor_one_supernode` itself,
    /// excluding the lock brackets above. Aggregated across workers,
    /// so on an 8-thread run this can exceed wall time by ~8×.
    pub factor_body_ns: std::sync::atomic::AtomicU64,
    /// Cumulative wall time of the entire `scope.spawn(...)` closure
    /// body across all tasks (includes lock waits, `factor_body`, and
    /// per-task control flow like the snode lookup, fast-exit check,
    /// pending-decrement, and recursive spawn). Aggregated across
    /// workers. Compute `task_wall_agg / T` to compare against
    /// `phase_scope_ns`: the gap is rayon idle (worker waiting for an
    /// eligible task) or scope spawn/join overhead.
    pub task_wall_ns: std::sync::atomic::AtomicU64,
    /// Cumulative wait time on the per-worker
    /// `Mutex<FactorWorkspace>`. Expected to be near zero (each
    /// worker has its own slot); non-zero values mean rayon scheduled
    /// two tasks onto the same worker queue before the first finished
    /// (an event that would also imply a worker idle elsewhere).
    pub ws_lock_wait_ns: std::sync::atomic::AtomicU64,
    /// Number of parallel tasks executed.
    pub n_tasks: std::sync::atomic::AtomicU64,
    /// `compute_scaling_with_cache` + scaling_pivot_order build.
    pub phase_scaling_ns: std::sync::atomic::AtomicU64,
    /// `permute_csc_values` (P·A·Pᵀ rebuild).
    pub phase_permute_ns: std::sync::atomic::AtomicU64,
    /// `permuted.symmetric_pattern()`.
    pub phase_symmetric_pattern_ns: std::sync::atomic::AtomicU64,
    /// `is_root` + `parents` + `pending` atomic counters +
    /// `contrib_blocks` / `node_factors_out` mutex setup.
    pub phase_tree_setup_ns: std::sync::atomic::AtomicU64,
    /// Per-worker `FactorWorkspace` provisioning, including
    /// `local_contribs.resize_with(n_snodes, ...)`.
    pub phase_thread_ws_ns: std::sync::atomic::AtomicU64,
    /// Leaves collection (single linear pass over supernodes).
    pub phase_leaves_ns: std::sync::atomic::AtomicU64,
    /// The `rayon::scope` itself — sums to the parallel hot-loop
    /// wall time plus rayon overhead. Compare against
    /// `factor_body_ns / T` to see worker utilization.
    pub phase_scope_ns: std::sync::atomic::AtomicU64,
    /// Final epilogue: `node_factors_out.into_inner()` + the
    /// postorder iteration that builds `final_nodes` and aggregates
    /// inertia.
    pub phase_collect_ns: std::sync::atomic::AtomicU64,
}

/// Plain-data snapshot of `AtomicLockStats` (non-atomic, easy to print).
#[derive(Default, Debug, Clone, Copy)]
pub struct ParallelLockStats {
    pub contrib_wait_ns: u64,
    pub contrib_hold_ns: u64,
    pub node_factors_wait_ns: u64,
    pub node_factors_hold_ns: u64,
    pub factor_body_ns: u64,
    pub task_wall_ns: u64,
    pub ws_lock_wait_ns: u64,
    pub n_tasks: u64,
    pub phase_scaling_ns: u64,
    pub phase_permute_ns: u64,
    pub phase_symmetric_pattern_ns: u64,
    pub phase_tree_setup_ns: u64,
    pub phase_thread_ws_ns: u64,
    pub phase_leaves_ns: u64,
    pub phase_scope_ns: u64,
    pub phase_collect_ns: u64,
}

impl AtomicLockStats {
    pub fn snapshot(&self) -> ParallelLockStats {
        use std::sync::atomic::Ordering;
        ParallelLockStats {
            contrib_wait_ns: self.contrib_wait_ns.load(Ordering::Relaxed),
            contrib_hold_ns: self.contrib_hold_ns.load(Ordering::Relaxed),
            node_factors_wait_ns: self.node_factors_wait_ns.load(Ordering::Relaxed),
            node_factors_hold_ns: self.node_factors_hold_ns.load(Ordering::Relaxed),
            factor_body_ns: self.factor_body_ns.load(Ordering::Relaxed),
            task_wall_ns: self.task_wall_ns.load(Ordering::Relaxed),
            ws_lock_wait_ns: self.ws_lock_wait_ns.load(Ordering::Relaxed),
            n_tasks: self.n_tasks.load(Ordering::Relaxed),
            phase_scaling_ns: self.phase_scaling_ns.load(Ordering::Relaxed),
            phase_permute_ns: self.phase_permute_ns.load(Ordering::Relaxed),
            phase_symmetric_pattern_ns: self.phase_symmetric_pattern_ns.load(Ordering::Relaxed),
            phase_tree_setup_ns: self.phase_tree_setup_ns.load(Ordering::Relaxed),
            phase_thread_ws_ns: self.phase_thread_ws_ns.load(Ordering::Relaxed),
            phase_leaves_ns: self.phase_leaves_ns.load(Ordering::Relaxed),
            phase_scope_ns: self.phase_scope_ns.load(Ordering::Relaxed),
            phase_collect_ns: self.phase_collect_ns.load(Ordering::Relaxed),
        }
    }
}

/// Dense Schur complement block returned by
/// [`factorize_multifrontal_with_schur`] (F3.2b).
///
/// Layout: column-major full-square `dim × dim` (the `dim²` buffer is
/// dense; both upper and lower triangles are populated by mirroring the
/// computed lower triangle, per `dev/research/schur-complement.md` D5).
/// Row/column ordering matches the user-supplied `schur_indices` exactly.
///
/// The mathematical content is `S = A_SS − A_FS^T A_FF^{-1} A_FS` where
/// `A_FF` is the eliminated (non-Schur) block, `A_FS` is the coupling,
/// and `A_SS` is the Schur block. Inertia is *not* computed for `S` —
/// callers wanting an inertia-correct read of the full system must
/// account for the Schur block separately (see F3.0 D7 prominent doc).
#[derive(Debug, Clone)]
pub struct SchurBlock {
    /// Side length of the Schur block (`= schur_indices.len()`).
    pub dim: usize,
    /// `dim × dim` column-major full-square dense buffer.
    pub data: Vec<f64>,
}

impl SchurBlock {
    /// Read the `(i, j)` entry. `0 <= i, j < dim`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[j * self.dim + i]
    }

    /// Symmetric mat-vec `y = S · x`. `x.len() == y.len() == self.dim`.
    /// Uses the full square buffer (both triangles populated).
    pub fn symv(&self, x: &[f64], y: &mut [f64]) -> Result<(), FeralError> {
        if x.len() != self.dim || y.len() != self.dim {
            return Err(FeralError::DimensionMismatch {
                expected: self.dim,
                got: x.len().max(y.len()),
            });
        }
        for yi in y.iter_mut() {
            *yi = 0.0;
        }
        for (j, &xj) in x.iter().enumerate().take(self.dim) {
            let col = &self.data[j * self.dim..(j + 1) * self.dim];
            for (i, &v) in col.iter().enumerate() {
                y[i] += v * xj;
            }
        }
        Ok(())
    }

    /// F3.4 — Convenience solve `S · x = rhs` against the dense Schur
    /// block. Factors `S` with the dense Bunch-Kaufman LDL^T solver
    /// and runs a single solve. For repeated solves with the same `S`,
    /// callers should drive `dense::factor::factor` and
    /// `dense::solve::solve` directly to amortise the factor cost.
    ///
    /// `S` is treated as symmetric (the lower triangle of the stored
    /// full-square buffer is used; the upper triangle is mirrored by
    /// the factorization). The factorization uses the default
    /// [`BunchKaufmanParams`]; pass parameters explicitly via
    /// [`SchurBlock::solve_with`] for non-default thresholds.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, FeralError> {
        self.solve_with(rhs, &BunchKaufmanParams::default())
    }

    /// As [`SchurBlock::solve`], but with explicit Bunch-Kaufman
    /// parameters (zero-pivot action, pivot threshold, etc.).
    pub fn solve_with(
        &self,
        rhs: &[f64],
        params: &BunchKaufmanParams,
    ) -> Result<Vec<f64>, FeralError> {
        if rhs.len() != self.dim {
            return Err(FeralError::DimensionMismatch {
                expected: self.dim,
                got: rhs.len(),
            });
        }
        let s_mat = SymmetricMatrix::from_column_major(self.dim, self.data.clone())?;
        let (factors, _inertia) = factor(&s_mat, params)?;
        crate::dense::solve::solve(&factors, rhs)
    }
}

/// Stored factors from a sparse multifrontal LDL^T factorization.
#[derive(Debug)]
pub struct SparseFactors {
    /// Matrix dimension.
    pub n: usize,

    /// Fill-reducing permutation (new-to-old).
    pub perm: Vec<usize>,
    /// Inverse permutation (old-to-new).
    pub perm_inv: Vec<usize>,

    /// Per-supernode factor data. Each entry contains:
    /// - L factor columns (nrow × ncol column-major, unit diagonal implicit)
    /// - D block diagonal values (ncol entries for 1×1 blocks)
    /// - D block subdiagonal values (for 2×2 blocks)
    /// - Pivot sequence (which columns used 1×1 vs 2×2 pivots)
    /// - Row indices of the frontal matrix
    pub node_factors: Vec<NodeFactors>,

    /// Whether iterative refinement is recommended.
    pub needs_refinement: bool,

    /// Global symmetric scaling vector in **user-order** indexing.
    /// Length `n`. The matrix actually factored is `D · A · D` with
    /// `D = diag(scaling)`, so solve must pre-scale the RHS and
    /// post-scale the solution with the same vector. Cloned from
    /// `SymbolicFactorization::scaling` at the end of
    /// `factorize_multifrontal` so the solve path can reach it
    /// without a back-pointer to the symbolic analysis.
    pub scaling: Vec<f64>,

    /// Diagnostic info about how `scaling` was produced. Mirrored
    /// from `SymbolicFactorization::scaling_info` for telemetry.
    pub scaling_info: crate::scaling::ScalingInfo,

    /// Concrete fill-reducing ordering method actually used. Mirrored
    /// from `SymbolicFactorization::resolved_method`. Resolves
    /// `OrderingMethod::Auto` to the dispatched method.
    pub resolved_method: crate::symbolic::OrderingMethod,
    /// Concrete amalgamation strategy actually used. Mirrored
    /// from `SymbolicFactorization::resolved_amalgamation`.
    pub resolved_amalgamation: crate::symbolic::AmalgamationStrategy,
    /// Concrete ordering preprocessor actually used. Mirrored
    /// from `SymbolicFactorization::resolved_preprocess`.
    pub resolved_preprocess: crate::symbolic::OrderingPreprocess,
}

/// A flat, factorization-order export of the LDLᵀ factors.
///
/// Produced by [`SparseFactors::ldlt_export`]. The supernodal frontal
/// factors are reassembled into a single global unit-lower-triangular
/// `L` (CSC) and a block-diagonal `D` (`d_diag` + `d_subdiag`), both
/// indexed by **factorization order** `e ∈ [0, n)` — the order in which
/// pivots are eliminated (postorder over supernodes, Bunch-Kaufman pivot
/// order within each front). In this order `L` is genuinely
/// unit-lower-triangular (`L[e,e] = 1`, entries only for row > col) and
/// every 2×2 `D` block occupies consecutive positions `(e, e+1)` with
/// `d_subdiag[e] ≠ 0`.
///
/// `perm` maps factorization order to the original matrix index:
/// `perm[e]` is the user-space row/column eliminated at position `e`.
/// `perm_inv` is its inverse. The reconstruction identity, with the
/// global symmetric scaling `s = SparseFactors::scaling` (user-order),
/// is
///
/// ```text
/// M = L · D · Lᵀ            (in factorization order)
/// A[perm[i], perm[j]] = M[i, j] / (s[perm[i]] · s[perm[j]])
/// ```
///
/// i.e. `L D Lᵀ` reconstructs the *scaled, permuted* input matrix, and
/// undoing the permutation and scaling recovers the original `A`.
#[derive(Debug, Clone)]
pub struct LdltExport {
    /// Factorization order → original matrix index (length `n`).
    pub perm: Vec<usize>,
    /// Original matrix index → factorization order (length `n`).
    pub perm_inv: Vec<usize>,
    /// CSC column pointers of `L` (length `n + 1`).
    pub l_indptr: Vec<usize>,
    /// CSC row indices of `L`, sorted within each column.
    pub l_indices: Vec<usize>,
    /// CSC values of `L` (unit diagonal stored explicitly).
    pub l_values: Vec<f64>,
    /// Block-diagonal of `D` in factorization order (length `n`).
    pub d_diag: Vec<f64>,
    /// Sub-diagonal of `D` (length `n`); `d_subdiag[e] ≠ 0` marks the
    /// top-left of a 2×2 block coupling positions `e` and `e + 1`.
    pub d_subdiag: Vec<f64>,
}

impl SparseFactors {
    /// Reassemble the supernodal frontal factors into a single global
    /// `L` (CSC) and `D`, in factorization order. See [`LdltExport`]
    /// for the layout and the reconstruction identity.
    ///
    /// O(nnz(L)) time and memory. Each global pivot is eliminated in
    /// exactly one supernode (where it is fully summed), so its `L`
    /// column is produced exactly once; trailing-row entries scatter
    /// into the columns of the fronts that own them. The
    /// factorization-order index `e` is assigned by a first pass over
    /// `node_factors` (postorder) and the within-front BK permutation
    /// `frontal_factors.perm`, so a second pass can emit CSC columns in
    /// strictly increasing order with sorted rows.
    pub fn ldlt_export(&self) -> LdltExport {
        let n = self.n;

        // Pass 1: assign factorization order. `e_of_g[g]` is the
        // factorization position of permuted-global index `g`; each `g`
        // is eliminated exactly once across all supernodes.
        let mut e_of_g = vec![usize::MAX; n];
        let mut perm = vec![0usize; n];
        let mut perm_inv = vec![0usize; n];
        let mut d_diag = vec![0.0f64; n];
        let mut d_subdiag = vec![0.0f64; n];
        let mut e = 0usize;
        for node in &self.node_factors {
            let ff = &node.frontal_factors;
            for j in 0..ff.nelim {
                let g = node.row_indices[ff.perm[j]];
                e_of_g[g] = e;
                // `self.perm` is fill-reducing new→old; `g` is the
                // permuted index, so `self.perm[g]` is the original.
                perm[e] = self.perm[g];
                perm_inv[self.perm[g]] = e;
                d_diag[e] = ff.d_diag[j];
                d_subdiag[e] = ff.d_subdiag[j];
                e += 1;
            }
        }
        debug_assert_eq!(e, n, "every index eliminated exactly once");

        // Pass 2: build L columns in factorization order. A supernode's
        // eliminated columns form a contiguous, increasing `e` range, so
        // iterating nodes in order then `j` in `0..nelim` emits columns
        // in ascending order — exactly CSC column order.
        let mut l_indptr = Vec::with_capacity(n + 1);
        l_indptr.push(0);
        let mut l_indices = Vec::new();
        let mut l_values = Vec::new();
        let mut col: Vec<(usize, f64)> = Vec::new();
        for node in &self.node_factors {
            let ff = &node.frontal_factors;
            let nrow = ff.nrow;
            for j in 0..ff.nelim {
                col.clear();
                // Unit diagonal, stored explicitly.
                let diag_e = e_of_g[node.row_indices[ff.perm[j]]];
                col.push((diag_e, 1.0));
                for i in (j + 1)..nrow {
                    let v = ff.l[j * nrow + i];
                    if v != 0.0 {
                        let row_e = e_of_g[node.row_indices[ff.perm[i]]];
                        col.push((row_e, v));
                    }
                }
                col.sort_unstable_by_key(|&(r, _)| r);
                for &(r, v) in &col {
                    l_indices.push(r);
                    l_values.push(v);
                }
                l_indptr.push(l_indices.len());
            }
        }

        LdltExport {
            perm,
            perm_inv,
            l_indptr,
            l_indices,
            l_values,
            d_diag,
            d_subdiag,
        }
    }

    /// One-line diagnostic summary of the strategies and pivot counts
    /// that produced these factors. Suitable for logging one record
    /// per factorization in monitoring drivers.
    ///
    /// Format:
    /// `n=<n> | <ordering> | <amalg> | preproc=<preproc> |
    ///  scaling=<scaling_info> | n_supernodes=<k> | nnz_L=<nL> |
    ///  n_2x2=<n2> | n_delayed=<nd> | inertia=(p,n,z)`
    ///
    /// Aggregated from `node_factors` so it is O(supernodes) and
    /// allocation-light. The inertia summed here equals the
    /// `Inertia` returned from `factorize_multifrontal`.
    pub fn summary(&self) -> String {
        let mut n_2x2 = 0usize;
        let mut n_delayed = 0usize;
        let mut nnz_l = 0usize;
        let mut inertia = crate::inertia::Inertia::new(0, 0, 0);
        for nf in &self.node_factors {
            let ff = &nf.frontal_factors;
            n_delayed += ff.n_delayed;
            // Match factor_nnz() accounting (lower-tri inc diag of
            // eliminated block + trailing rect).
            let trailing = ff.nrow.saturating_sub(ff.nelim) * ff.nelim;
            nnz_l += ff.nelim * (ff.nelim + 1) / 2 + trailing;
            let nelim = ff.nelim;
            let mut k = 0;
            while k < nelim {
                let two_by_two = k + 1 < nelim && ff.d_subdiag[k] != 0.0;
                if two_by_two {
                    n_2x2 += 1;
                    k += 2;
                } else {
                    k += 1;
                }
            }
            inertia.positive += nf.inertia.positive;
            inertia.negative += nf.inertia.negative;
            inertia.zero += nf.inertia.zero;
        }
        format!(
            "n={} | ord={:?} | amalg={:?} | preproc={:?} | scaling={:?} | n_supernodes={} | nnz_L={} | n_2x2={} | n_delayed={} | inertia=({},{},{})",
            self.n,
            self.resolved_method,
            self.resolved_amalgamation,
            self.resolved_preprocess,
            self.scaling_info,
            self.node_factors.len(),
            nnz_l,
            n_2x2,
            n_delayed,
            inertia.positive,
            inertia.negative,
            inertia.zero,
        )
    }

    /// Total real entries used in the L factor across all supernodes.
    ///
    /// Per supernode the L block is `nrow × nelim` column-major with
    /// unit-lower-triangular structure in the leading `nelim × nelim`
    /// eliminated block. The strict-upper triangle of that block is
    /// structurally zero and excluded from the count. The unit
    /// diagonal *is* counted.
    ///
    /// Per-supernode count:
    /// `nelim * (nelim + 1) / 2 + (nrow - nelim) * nelim`
    ///   = (eliminated lower-tri inc diagonal) + (trailing rect rows).
    ///
    /// This matches SSIDS's `inform%num_factor` accounting exactly at
    /// the median across the kkt corpus (verified by
    /// `src/bin/diag_factor_nnz_accounting.rs`). MUMPS's `INFOG(9)`
    /// uses a different accounting that includes additional entries
    /// for delayed pivots and pre-allocation; nnzL/MUMPS ratios will
    /// therefore be < 1 typically.
    ///
    /// The D entries are not counted here (`nelim + n_2x2` extra
    /// scalars; negligible for fill-ratio analysis on large fronts).
    ///
    /// Use case: fill-ratio diagnostics. `factor_nnz() / csc.nnz()` is
    /// a quick proxy for ordering quality. Values <10× on KKT-style
    /// matrices indicate a healthy ordering; values >50× suggest the
    /// resolved `OrderingMethod` is mismatched to the structure.
    pub fn factor_nnz(&self) -> usize {
        self.node_factors
            .iter()
            .map(|nf| {
                let nrow = nf.frontal_factors.nrow;
                let nelim = nf.frontal_factors.nelim;
                let trailing = nrow.saturating_sub(nelim) * nelim;
                let eliminated_lower_with_diag = nelim * (nelim + 1) / 2;
                eliminated_lower_with_diag + trailing
            })
            .sum()
    }

    /// Minimum eigenvalue of D over all eliminated pivots.
    ///
    /// 1×1 pivots contribute `d_diag[k]` directly. 2×2 blocks
    /// contribute the smaller eigenvalue of
    /// `[[d_diag[k], d_subdiag[k]], [d_subdiag[k], d_diag[k+1]]]`,
    /// computed as `(trace - sqrt(trace^2 - 4*det)) / 2`.
    ///
    /// 2×2 detection follows the solve-path convention
    /// (`src/numeric/solve.rs:217`): `d_subdiag[k] != 0.0` with the
    /// bounds check `k + 1 < nelim`.
    ///
    /// Returns `None` when no pivots were eliminated (n=0 or every
    /// supernode skipped). Used by ipopt-style unconstrained
    /// inertia correction (`-min_d + eps` as a direct delta_w).
    pub fn min_diagonal(&self) -> Option<f64> {
        let mut min_d = f64::INFINITY;
        let mut any = false;
        for nf in &self.node_factors {
            let ff = &nf.frontal_factors;
            let nelim = ff.nelim;
            let mut k = 0;
            while k < nelim {
                let two_by_two = k + 1 < nelim && ff.d_subdiag[k] != 0.0;
                let eig = if two_by_two {
                    let a = ff.d_diag[k];
                    let b = ff.d_subdiag[k];
                    let c = ff.d_diag[k + 1];
                    let trace = a + c;
                    let det = a * c - b * b;
                    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
                    (trace - disc) * 0.5
                } else {
                    ff.d_diag[k]
                };
                if eig < min_d {
                    min_d = eig;
                }
                any = true;
                k += if two_by_two { 2 } else { 1 };
            }
        }
        if any {
            Some(min_d)
        } else {
            None
        }
    }

    /// Smallest accepted pivot magnitude `min|λ(D)|` over every
    /// eliminated 1×1 and 2×2 block. This is FERAL's near-singularity
    /// signal — the analog of MA57's `CNTL(2)` small-pivot threshold.
    ///
    /// A 1×1 pivot contributes `|d_diag[k]|`. A 2×2 block contributes
    /// the *smaller-magnitude* eigenvalue of
    /// `[[d_diag[k], d_subdiag[k]], [d_subdiag[k], d_diag[k+1]]]`,
    /// computed as `|det| / larger_magnitude` to stay cancellation-free
    /// when the block is near-singular.
    ///
    /// Distinct from [`min_diagonal`](Self::min_diagonal), which returns
    /// the *signed* smallest eigenvalue (the most-negative one) for the
    /// unconstrained inertia-correction shortcut. Near-singularity needs
    /// the smallest-in-*magnitude* pivot regardless of sign.
    ///
    /// The value lives in the scaled space of the matrix actually
    /// factored (`S·A·S`); pair it with [`max_pivot_magnitude`] to form
    /// the scale-free ratio `min/max ≈ 1/κ(D)`. See
    /// `dev/research/near-singularity-signal.md`.
    ///
    /// Returns `None` when no pivots were eliminated.
    pub fn min_pivot_magnitude(&self) -> Option<f64> {
        self.pivot_magnitude_extent().map(|(min, _max)| min)
    }

    /// Largest accepted pivot magnitude `max|λ(D)|` over every
    /// eliminated 1×1 and 2×2 block. Provided so a caller can form the
    /// scale-free near-singularity ratio `min_pivot_magnitude() /
    /// max_pivot_magnitude()` without recomputing `||S·A·S||`.
    ///
    /// Returns `None` when no pivots were eliminated. See
    /// [`min_pivot_magnitude`](Self::min_pivot_magnitude).
    pub fn max_pivot_magnitude(&self) -> Option<f64> {
        self.pivot_magnitude_extent().map(|(_min, max)| max)
    }

    /// Sum of MUMPS-style static-perturbation events
    /// (`INFO(25)` / NBTINYW equivalent) across every supernode.
    /// Counts each time a pivot was rounded to `sign(d)·floor` via
    /// `perturb_to_floor` (1×1) or `perturb_2x2_to_floor` (2×2).
    /// Diagnostic only — does not affect inertia, solve behavior,
    /// or any acceptance gate. See [`FrontalFactors::n_tiny`].
    pub fn n_tiny(&self) -> usize {
        self.node_factors
            .iter()
            .map(|nf| nf.frontal_factors.n_tiny)
            .sum()
    }

    /// Single pass over the D blocks returning `(min|λ|, max|λ|)`.
    /// Shared by [`min_pivot_magnitude`](Self::min_pivot_magnitude) and
    /// [`max_pivot_magnitude`](Self::max_pivot_magnitude). 2×2 detection
    /// follows the solve-path convention (`d_subdiag[k] != 0.0`).
    fn pivot_magnitude_extent(&self) -> Option<(f64, f64)> {
        let mut min_mag = f64::INFINITY;
        let mut max_mag = 0.0f64;
        let mut any = false;
        for nf in &self.node_factors {
            let ff = &nf.frontal_factors;
            let nelim = ff.nelim;
            let mut k = 0;
            while k < nelim {
                let two_by_two = k + 1 < nelim && ff.d_subdiag[k] != 0.0;
                let (smaller, larger) = if two_by_two {
                    // Eigenvalues of [[a,b],[b,c]]: λ± = (t ± √(t²−4Δ))/2.
                    // The larger magnitude (|t|+√disc)/2 is
                    // cancellation-free; the smaller is |Δ|/larger
                    // since |λ₊·λ₋| = |Δ|.
                    let a = ff.d_diag[k];
                    let b = ff.d_subdiag[k];
                    let c = ff.d_diag[k + 1];
                    let trace = a + c;
                    let det = a * c - b * b;
                    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
                    let larger = (trace.abs() + disc) * 0.5;
                    let smaller = if larger > 0.0 {
                        det.abs() / larger
                    } else {
                        0.0
                    };
                    (smaller, larger)
                } else {
                    let m = ff.d_diag[k].abs();
                    (m, m)
                };
                if smaller < min_mag {
                    min_mag = smaller;
                }
                if larger > max_mag {
                    max_mag = larger;
                }
                any = true;
                k += if two_by_two { 2 } else { 1 };
            }
        }
        if any {
            Some((min_mag, max_mag))
        } else {
            None
        }
    }
}

/// Factor data for a single supernode.
#[derive(Debug)]
pub struct NodeFactors {
    /// First column index (in permuted numbering).
    pub first_col: usize,
    /// Attempted column count (`snode.ncol() + n_delayed_in`). This is
    /// the `ncol` argument that was passed to `factor_frontal` and may
    /// exceed the supernode's native column count when children delayed
    /// pivots up into this node. Solve paths that iterate over
    /// eliminated columns must use `frontal_factors.nelim`, not `ncol`.
    pub ncol: usize,
    /// Number of pivots actually eliminated at this node
    /// (`ncol - n_delayed_out`). Mirror of `frontal_factors.nelim` for
    /// convenience in the solve path.
    pub nelim: usize,
    /// Number of delayed columns that entered this node from its
    /// children during parent assembly (sum of `child.contrib.n_delayed`
    /// over all children). These occupy positions
    /// `[snode.ncol() .. snode.ncol() + n_delayed_in)` of `row_indices`
    /// and are fed to `factor_frontal` as additional fully-summed
    /// columns on top of the supernode's native column count.
    pub n_delayed_in: usize,
    /// Total number of rows in the frontal.
    pub nrow: usize,
    /// Row indices of the frontal (length nrow).
    pub row_indices: Vec<usize>,
    /// The frontal factors from partial BK factorization.
    pub frontal_factors: FrontalFactors,
    /// Inertia of this node's eliminated pivots.
    pub inertia: Inertia,
}

/// Caller-owned scratch pool for sparse numeric factorization.
///
/// Reusing a single workspace across multiple calls of
/// [`factorize_multifrontal_with_workspace`] amortises per-call
/// allocation — the alloc-probe evidence in
/// `dev/research/sparse-tail-perf-2026-04-19.md` §9 shows 17–23
/// allocations per supernode, many of which are scratch buffers
/// that can be pooled.
///
/// Each field grows monotonically: the first call sizes the field
/// to what the matrix needs; subsequent calls on larger matrices
/// grow via `resize`, and subsequent calls on smaller matrices
/// reuse the existing capacity without shrinking.
///
/// The scratch buffers are NOT populated across calls — every call
/// clears them to a well-defined initial state on entry. The
/// workspace exists purely to retain heap capacity between calls,
/// not to carry data.
///
/// Invariant for `row_map`: at function entry every entry is
/// `usize::MAX`. The per-supernode loop in
/// `factorize_multifrontal_with_workspace` writes and then clears
/// exactly `row_indices.len()` entries per iteration, preserving
/// the invariant between iterations. At call entry the invariant
/// is re-established unconditionally by clearing and re-filling
/// `row_map` so prior error paths (which skip the clear) cannot
/// corrupt subsequent calls.
#[derive(Debug, Default)]
pub struct FactorWorkspace {
    /// Global→local row-index map. Length grows to `matrix.n`;
    /// entries are maintained in the all-`usize::MAX` state outside
    /// the per-supernode critical section.
    row_map: Vec<usize>,
    /// Pooled storage for the per-supernode frontal
    /// `SymmetricMatrix::data` buffer. Length resized per supernode
    /// to `nrow * nrow`; the allocation is reused across supernodes
    /// and across calls. Left empty when ownership is temporarily
    /// borrowed by an in-flight `SymmetricMatrix`.
    frontal_values: Vec<f64>,
    /// Scratch for `build_row_indices`: delayed-column globals
    /// accumulated from children of the current supernode.
    build_delayed: Vec<usize>,
    /// Scratch for `build_row_indices`: trailing (non-fully-summed)
    /// row globals for the current supernode, collected via a
    /// `build_seen`-based dedup and sorted at the end to match the
    /// pre-pool BTreeSet traversal order.
    build_trailing: Vec<usize>,
    /// Scratch for `build_row_indices`: global→`bool` membership
    /// marker. Length grows to `matrix.n`; entries are maintained
    /// in the all-`false` state outside the call (touched indices
    /// are cleared before return).
    build_seen: Vec<bool>,
    /// Pooled `n * n` f64 storage for the D.3/D.4 dense fast-path
    /// densify of the input `CscMatrix`. Reused across calls via
    /// `std::mem::take` + `CscMatrix::to_dense_into` so the
    /// fast-path no longer reallocates `n * n` doubles per call.
    /// Left empty when ownership is temporarily borrowed by an
    /// in-flight `SymmetricMatrix`.
    dense_values: Vec<f64>,
    /// Per-worker scratch for the parallel driver: a sparse Option
    /// lookup keyed by supernode index. The parallel task drains its
    /// children's `ContribBlock`s into this vec (under the shared
    /// `contrib_blocks` mutex), passes it to `factor_one_supernode`,
    /// then takes its own slot out into the shared store. All slots
    /// are guaranteed `None` between tasks, so no clearing is needed.
    /// Pre-sized to `n_snodes` once per worker by the parallel driver;
    /// sequential drivers don't touch this field.
    local_contribs: Vec<Option<ContribBlock>>,
    /// Issue #13 Phase A: pooled `subdiag` + `d_panel` working buffers
    /// for `factor_frontal_blocked_in_place_with_scratch`. Reused
    /// across supernodes within a single workspace lifetime; the
    /// kernel `clear()`s and `resize()`s on entry, preserving capacity.
    pub factor_scratch: FactorScratch,
    /// Issue #56 Lever A.2: cached permute structure for fast warm
    /// reconstruction of P·A·Pᵀ. Populated on the first cold call to
    /// `permute_csc_values_with_cache`; consulted on subsequent calls
    /// when `NumericParams::pattern_reused_hint` is `true` and the
    /// input `(n, nnz)` matches. The warm path scatters values via a
    /// precomputed `value_map` instead of rebuilding triplets and
    /// re-sorting through `CscMatrix::from_triplets`.
    pub(crate) permute_cache: Option<PermuteCache>,
}

/// Cached permute structure. See `FactorWorkspace::permute_cache`.
///
/// The structure (`col_ptr`, `row_idx`) of P·A·Pᵀ depends only on the
/// input sparsity pattern and the permutation; both are invariant
/// across IPM iterations that share a `Solver`. The `value_map`
/// records, for each input value position `k`, which slot in the
/// permuted `values` array that value occupies (summing duplicates
/// when the input contains both `(i, j)` and `(j, i)`). Warm calls
/// allocate only the output `values` buffer and run a single O(nnz)
/// scatter pass — no triplet construction, no sort.
#[derive(Debug, Default)]
pub(crate) struct PermuteCache {
    /// Input matrix order at cache build.
    input_n: usize,
    /// Input matrix nnz at cache build (`matrix.values.len()`).
    input_nnz: usize,
    /// Input `col_ptr` at cache build. The permuted structure and
    /// `value_map` are a pure function of the input *pattern*
    /// (`col_ptr` + `row_idx`) and `perm_inv`; the warm path is valid
    /// only if all three are byte-identical to the build-time inputs.
    /// REG-1: `(n, nnz)` alone is NOT a sufficient key — two distinct
    /// patterns sharing `(n, nnz)`, or the same pattern under a changed
    /// permutation, would otherwise scatter values through a stale
    /// structure and return a wrong factorization. Stored (not hashed)
    /// so a fingerprint collision can never reintroduce that silent
    /// wrong answer; the compare is O(n + nnz), still strictly cheaper
    /// than the `from_triplets` sort the warm path skips.
    input_col_ptr: Vec<usize>,
    /// Input `row_idx` at cache build (see `input_col_ptr`).
    input_row_idx: Vec<usize>,
    /// `perm_inv` at cache build (see `input_col_ptr`).
    input_perm_inv: Vec<usize>,
    /// `col_ptr` of the cached permuted CSC.
    permuted_col_ptr: Vec<usize>,
    /// `row_idx` of the cached permuted CSC.
    permuted_row_idx: Vec<usize>,
    /// `value_map[k]` is the index into the permuted `values` vector
    /// where `matrix.values[k]` contributes (via `+=`). When the input
    /// has duplicate `(i, j)` / `(j, i)` entries they map to the same
    /// slot, matching `CscMatrix::from_triplets` duplicate-summing.
    value_map: Vec<usize>,
}

impl FactorWorkspace {
    /// Construct an empty workspace. Equivalent to `default()`.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Gate predicate for the D.3 dense fast-path.
///
/// Returns `true` when the input qualifies for the dense fast-path.
///
/// Two disjuncts:
///   1. **D.4 tiny-n** — `n ≤ N_TINY` unconditionally (density is
///      irrelevant; the multifrontal scaffolding cost dominates at
///      these sizes).
///   2. **D.3 small-dense** — `n ≤ N_MAX` and
///      `nnz_lower / (n * (n + 1) / 2) ≥ ρ_MIN`. The density threshold
///      is expressed as the integer inequality
///      `nnz_lower * ρ_DEN ≥ n * (n + 1) / 2 * ρ_NUM` so the check
///      costs a handful of integer ops with no division or FP.
///
/// Authoritative entry point for the gate; callers must not
/// roll their own. Thresholds may be tuned post-measurement
/// (see `dev/plans/sparse-tail-d3.md` stage 2 for D.3 and
/// `dev/plans/sparse-tail-d4.md` stage 2 for D.4).
///
/// Thresholds (`N_TINY = 16`, `N_MAX = 128`, `ρ_MIN = 1/4`) are
/// initial values from the research note
/// `dev/research/sparse-tail-d3-d4-2026-04-19.md`. Update all three
/// together if a future sweep tunes them.
#[inline]
pub fn should_use_dense_fast_path(n: usize, nnz_lower: usize) -> bool {
    // D.4 tiny-n: unconditional.
    const N_TINY: usize = 16;
    // D.3 small-dense: density-gated.
    const N_MAX: usize = 128;
    // ρ_MIN = ρ_NUM / ρ_DEN = 1/4 = 0.25
    const RHO_NUM: usize = 1;
    const RHO_DEN: usize = 4;
    if n == 0 {
        return false;
    }
    if n <= N_TINY {
        return true;
    }
    if n > N_MAX {
        return false;
    }
    let lower_cells = n * (n + 1) / 2;
    // nnz_lower / lower_cells >= RHO_NUM / RHO_DEN, i.e.
    // nnz_lower * RHO_DEN >= lower_cells * RHO_NUM.
    nnz_lower * RHO_DEN >= lower_cells * RHO_NUM
}

/// Fast-path factorization for small-and-dense matrices.
///
/// Skips symbolic analysis entirely: densifies the CSC into a
/// `SymmetricMatrix`, applies the usual global symmetric scaling,
/// runs the dense BK kernel on all `n` columns, and wraps the
/// `FrontalFactors` in a single-supernode `SparseFactors` that is
/// shape-compatible with `solve_sparse`.
///
/// Should only be called on matrices for which
/// [`should_use_dense_fast_path`] returns `true`. The production
/// dispatch path in `factorize_multifrontal_with_workspace` enforces
/// this; direct callers (tests, benches) must observe it themselves.
///
pub fn dense_fast_factor(
    matrix: &CscMatrix,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError> {
    let mut ws = FactorWorkspace::new();
    dense_fast_factor_with_workspace(matrix, params, &mut ws)
}

/// Pooled-buffer variant of [`dense_fast_factor`].
///
/// The `n * n` dense-densify buffer is drawn from (and returned
/// to) `ws.dense_values`, so repeated calls across a single
/// `FactorWorkspace` lifetime amortise the `n * n` f64 allocation.
/// See `dev/research/phase-2.5.x-to-dense-pooling.md`.
pub fn dense_fast_factor_with_workspace(
    matrix: &CscMatrix,
    params: &NumericParams,
    ws: &mut FactorWorkspace,
) -> Result<(SparseFactors, Inertia), FeralError> {
    let n = matrix.n;
    if n == 0 {
        return Err(FeralError::InvalidInput(
            "dense_fast_factor: matrix dimension is zero".to_string(),
        ));
    }

    // Densify the CSC into a SymmetricMatrix (lower-triangle populated
    // at data[j*n + i] for i >= j). The densify happens *before*
    // scaling so the dense-native Knight-Ruiz path can iterate
    // the column-major buffer directly. Pool the `n * n` buffer:
    // hand the caller-owned Vec to `to_dense_into`, use it, then
    // return it to `ws.dense_values` before falling out of the
    // function.
    let dense_buf = std::mem::take(&mut ws.dense_values);
    let mut sym = matrix.to_dense_into(dense_buf);

    // Global symmetric scaling — same contract as the multifrontal
    // path. Perm is identity here so user-order == pivot-order.
    // `compute_scaling_dense_fast` routes `Auto`/`InfNorm` to the
    // dense-native KR iteration over `sym`; other strategies use
    // the sparse `compute_scaling` path.
    let (scaling, scaling_info) = compute_scaling_dense_fast(matrix, &sym, &params.scaling)?;
    if params.warn_partial_singular {
        if let crate::scaling::ScalingInfo::PartialSingular { n_unmatched } = &scaling_info {
            eprintln!(
                "warning: MC64 matching left {} of {} variables unmatched; \
                 scaling is identity on those rows/columns",
                n_unmatched, n
            );
        }
    }

    // Apply D · A · D in place on the dense buffer.
    for (j, &s_j) in scaling.iter().enumerate() {
        let col = j * n;
        for (i, &s_i) in scaling.iter().enumerate().skip(j) {
            sym.data[col + i] *= s_i * s_j;
        }
    }

    // F-01: raise the BK kernel's `zero_tol` to the Wilkinson backward
    // error floor `n · EPS · ||A_scaled||_inf` so rank-deficiency
    // pivots that land above EPS but in the noise band are classified
    // as zero (case a in `try_reject_1x1_frontal`) rather than as
    // small-but-real (case b). No-op under `on_zero_pivot == Fail`
    // (preserves the absolute-tolerance contract on dense
    // abort-on-zero callers). See
    // `dev/research/f01-rankdef-underreporting.md`.
    let local_params =
        apply_post_scaling_overrides(params, scaled_matrix_infnorm_dense(&sym.data, n), n);
    let params: &NumericParams = local_params.as_ref().unwrap_or(params);

    // Factor the full n columns. `may_delay = false` matches the
    // multifrontal root-supernode behavior: ForceAccept absorbs any
    // unstable pivot instead of carrying it forward (there is no
    // ancestor in a single-node factorization).
    // Factor in place into `sym.data` (W-3a). `sym.data` content is
    // undefined on return, but the buffer itself is reusable; return it
    // to the pool.
    //
    // Issue #34 phase (d): SQD fast-path dispatch. When the caller has
    // declared the input symmetric quasi-definite via
    // `Solver::with_sqd_mode(true)`, skip the BK 1x1-vs-2x2 search and
    // use the diagonal-only kernel. Vanderbei 1995 Theorem 2.1
    // guarantees a diagonal D exists for any SQD matrix in the order
    // we receive it.
    let ff = if params.sqd_mode {
        factor_frontal_diagonal_in_place(&mut sym, n, &params.bk)?
    } else {
        factor_frontal_blocked_in_place_with_scratch(
            &mut sym,
            n,
            false,
            &params.bk,
            &mut ws.factor_scratch,
        )?
    };
    ws.dense_values = sym.data;

    let inertia = ff.inertia.clone();
    let needs_refinement = ff.needs_refinement;

    // Synthesize a single-supernode SparseFactors with identity perm.
    // `solve_sparse` iterates node_factors applying each node's
    // FrontalFactors to its slice; with row_indices = 0..n and
    // perm/perm_inv identity, this reduces exactly to the dense solve.
    let perm: Vec<usize> = (0..n).collect();
    let perm_inv: Vec<usize> = (0..n).collect();
    let row_indices: Vec<usize> = (0..n).collect();

    let node = NodeFactors {
        first_col: 0,
        ncol: n,
        nelim: ff.nelim,
        n_delayed_in: 0,
        nrow: n,
        row_indices,
        frontal_factors: ff,
        inertia: inertia.clone(),
    };

    Ok((
        SparseFactors {
            n,
            perm,
            perm_inv,
            node_factors: vec![node],
            needs_refinement,
            scaling,
            scaling_info,
            // Dense fast-path skips symbolic analysis; no ordering /
            // amalgamation / preprocess actually ran. Record the
            // concrete "did-nothing" values rather than the
            // `Auto` sentinels (which are dispatch tokens, not
            // resolutions). The single-supernode
            // `node_factors.len() == 1` is the identifying signal
            // that the fast path was taken.
            resolved_method: crate::symbolic::OrderingMethod::Amd,
            resolved_amalgamation: crate::symbolic::AmalgamationStrategy::Adjacency,
            resolved_preprocess: crate::symbolic::OrderingPreprocess::None,
        },
        inertia,
    ))
}

/// Forced-supernodal variant of [`factorize_multifrontal`].
///
/// Bypasses the D.3 dense fast-path gate and runs the multifrontal
/// supernodal path regardless of input shape. Intended for test
/// oracles (the solve-parity suite in `tests/dense_fast_path.rs`)
/// that need to compare the dense-path factor against the
/// multifrontal factor on an in-gate matrix.
pub fn factorize_multifrontal_supernodal(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError> {
    let mut ws = FactorWorkspace::new();
    factorize_multifrontal_supernodal_with_workspace(matrix, symbolic, params, &mut ws)
}

/// Perform multifrontal numeric factorization.
///
/// Takes the original sparse matrix and the symbolic factorization,
/// performs numeric factorization by traversing supernodes in postorder:
///
/// 1. Assemble original matrix entries into the frontal matrix
/// 2. Assemble child contribution blocks (extend-add)
/// 3. Factor the frontal with the dense BK kernel
/// 4. Extract the contribution block (Schur complement)
/// 5. Accumulate inertia
///
/// This entry point allocates a fresh `FactorWorkspace` on every
/// call. Callers amortising factorization across multiple
/// invocations (e.g. IPM iterations) should use
/// [`factorize_multifrontal_with_workspace`] instead and retain
/// the workspace between calls.
pub fn factorize_multifrontal(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError> {
    let mut ws = FactorWorkspace::new();
    factorize_multifrontal_with_workspace(matrix, symbolic, params, &mut ws)
}

/// Numeric multifrontal factorization with a partial Schur extraction (F3.2b).
///
/// `symbolic` must have been produced by
/// [`crate::symbolic::symbolic_factorize_with_schur`]; otherwise this
/// returns `InvalidInput`. The matching invariant — `is_schur_tail ==
/// Some(n_schur) > 0` — is the only structural precondition.
///
/// Pipeline divergence from [`factorize_multifrontal`]:
///
/// 1. Per-supernode `nvschur[s]` is computed from `is_schur_tail` and
///    the supernode column ranges. Only supernodes whose column range
///    intersects `[n - n_schur, n)` have `nvschur > 0`. Those
///    supernodes are necessarily root(s) of the etree post-F3.2a (Schur
///    columns occupy the highest etree-index positions).
///
/// 2. At each Schur-bearing root, the Bunch-Kaufman pivot loop
///    eliminates only `expanded_ncol − nvschur` columns; the remaining
///    `nvschur` Schur columns end up un-eliminated in the contribution
///    block at positions `[0, nvschur) × [0, nvschur)` (col-major
///    lower-triangle dense). This matches MUMPS
///    `dfac_front_LDLT_type1.F:193-205`'s `NPIV ≤ NASS − NVSCHUR`
///    bound (see dev/research/schur-complement.md D4-D6).
///
/// 3. After the postorder loop, the dense `n_schur × n_schur` Schur
///    block is read out of the root supernode's `ContribBlock`,
///    mirrored lower→upper, and returned as a [`SchurBlock`].
///
/// **Constraint**: the Schur columns must form a single contiguous tail
/// and the tail-bearing supernode must be a single root whose last
/// column is at position `n - 1`. The F3.2a symbolic pipeline now
/// guarantees the single-supernode invariant by force-merging any
/// Schur-bearing supernodes before this entry point sees them
/// (see [`crate::symbolic::symbolic_factorize_with_schur`] step 8b,
/// mirroring MUMPS's HALO-SCHUR amalgamation in
/// `ana_orderings.F:9187-9220`). Forest-structured Schur sets (the
/// matrix has multiple connected components, each contributing a Schur
/// root) are rejected at the symbolic phase with `InvalidInput`.
///
/// Returned `Inertia` reflects the inertia of the *eliminated* block
/// `A_FF` only — the Schur block's spectrum is not factored.
pub fn factorize_multifrontal_with_schur(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia, SchurBlock), FeralError> {
    let n_schur = symbolic.is_schur_tail.ok_or_else(|| {
        FeralError::InvalidInput(
            "factorize_multifrontal_with_schur requires symbolic produced by \
             symbolic_factorize_with_schur (is_schur_tail is None)"
                .to_string(),
        )
    })?;
    if n_schur == 0 {
        return Err(FeralError::InvalidInput(
            "is_schur_tail = Some(0); use factorize_multifrontal instead".to_string(),
        ));
    }

    let n = symbolic.n;
    let n_snodes = symbolic.supernodes.len();

    // Per-supernode nvschur. Schur columns occupy global perm positions
    // [n - n_schur, n), so a supernode's nvschur is the size of its
    // column-range intersection with that interval. The
    // F3.2a postorder pins these positions to the tail of the supernode
    // sequence, so only the last contiguous run of supernodes has
    // nvschur > 0.
    let mut nvschur_per_snode = vec![0usize; n_snodes];
    let schur_lo = n - n_schur;
    for (s, snode) in symbolic.supernodes.iter().enumerate() {
        let col_lo = snode.first_col;
        let col_hi = col_lo + snode.ncol();
        if col_hi <= schur_lo || col_lo >= n {
            continue;
        }
        let lo = col_lo.max(schur_lo);
        let hi = col_hi.min(n);
        nvschur_per_snode[s] = hi - lo;
    }

    // F3.2b scope guard: require the Schur tail to live entirely in
    // one supernode whose last column is at position n - 1. Multi-
    // supernode Schur tails are deferred to F3.3.
    let last_snode = n_snodes
        .checked_sub(1)
        .ok_or_else(|| FeralError::InvalidInput("symbolic has zero supernodes".to_string()))?;
    let last = &symbolic.supernodes[last_snode];
    if last.first_col + last.ncol() != n {
        return Err(FeralError::InvalidInput(
            "Schur path expects last supernode to end at n-1".to_string(),
        ));
    }
    if nvschur_per_snode[last_snode] != n_schur {
        return Err(FeralError::InvalidInput(format!(
            "F3.2b scope: Schur tail must lie in a single root supernode \
             (last snode covers {} of {} Schur columns); see \
             dev/research/schur-complement.md F3.3",
            nvschur_per_snode[last_snode], n_schur
        )));
    }
    for &k in &nvschur_per_snode[..last_snode] {
        debug_assert_eq!(k, 0, "nvschur > 0 outside last snode violates F3.2b scope");
    }

    let mut ws = FactorWorkspace::new();
    factorize_multifrontal_with_schur_inner(matrix, symbolic, params, &mut ws, &nvschur_per_snode)
}

/// F3.2b inner driver: a Schur-aware specialization of
/// [`factorize_multifrontal_supernodal_with_workspace`]. Sequential.
/// Skips the dense fast-path (incompatible with partial elimination)
/// and the small-leaf batch path (leaves cannot be Schur-bearing under
/// the F3.2a layout, but we route everything through the generic
/// `factor_one_supernode` to keep the nvschur threading explicit).
fn factorize_multifrontal_with_schur_inner(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
    ws: &mut FactorWorkspace,
    nvschur_per_snode: &[usize],
) -> Result<(SparseFactors, Inertia, SchurBlock), FeralError> {
    let n = symbolic.n;
    let n_snodes = symbolic.supernodes.len();

    ws.row_map.clear();
    ws.row_map.resize(n, usize::MAX);

    let (scaling_user, scaling_info) = crate::scaling::compute_scaling(matrix, &params.scaling)?;
    let scaling_pivot_order: Vec<f64> =
        symbolic.perm.iter().map(|&old| scaling_user[old]).collect();

    let (permuted, _) = permute_csc_values_with_cache(
        matrix,
        &symbolic.perm,
        &symbolic.perm_inv,
        false,
        params.pattern_reused_hint,
        &mut ws.permute_cache,
    )?;
    // Issue #56 Lever A.1: `symbolic.permuted_pattern` is the full
    // symmetric pattern of P·A·Pᵀ — identical (up to sort, which
    // `permute_pattern` enforces) to what `permuted.symmetric_pattern()`
    // returns. Use it directly instead of recomputing every numeric call.
    let full_pattern = &symbolic.permuted_pattern;

    let mut is_root = vec![true; n_snodes];
    for snode in &symbolic.supernodes {
        for &child_idx in &snode.children {
            if child_idx < n_snodes {
                is_root[child_idx] = false;
            }
        }
    }

    let mut contrib_blocks: Vec<Option<ContribBlock>> = (0..n_snodes).map(|_| None).collect();
    let mut node_factors: Vec<NodeFactors> = Vec::with_capacity(n_snodes);
    let mut total_inertia = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    let mut needs_refinement = false;

    for (snode_idx, &nvschur) in nvschur_per_snode.iter().enumerate() {
        let node = factor_one_supernode(
            snode_idx,
            symbolic,
            &permuted,
            full_pattern,
            &scaling_pivot_order,
            &is_root,
            params,
            ws,
            &mut contrib_blocks,
            nvschur,
        )?;
        total_inertia.positive += node.inertia.positive;
        total_inertia.negative += node.inertia.negative;
        total_inertia.zero += node.inertia.zero;
        if node.frontal_factors.needs_refinement {
            needs_refinement = true;
        }
        node_factors.push(node);
    }

    // Extract the dense Schur block from the last (Schur-bearing) root
    // supernode's contribution block. The first nvschur rows/cols of
    // contrib are the Schur columns in user-supplied order — see
    // factor_one_supernode (nvschur > 0) plus the BK pivot gate at
    // src/dense/factor.rs:1670 (positions ≥ ncol_eff are never swapped).
    let n_schur = nvschur_per_snode[n_snodes - 1];
    debug_assert!(n_schur > 0);
    let contrib = contrib_blocks[n_snodes - 1].take().ok_or_else(|| {
        FeralError::InvalidInput(
            "Schur path: root supernode produced no contribution block".to_string(),
        )
    })?;
    if contrib.dim < n_schur {
        return Err(FeralError::InvalidInput(format!(
            "Schur extraction: root contrib dim {} < n_schur {}",
            contrib.dim, n_schur
        )));
    }

    // Extract leading n_schur × n_schur subblock. ContribBlock data is
    // col-major dim × dim with valid data in the lower triangle (per
    // factor_frontal_blocked_in_place / factor_frontal). Mirror to a
    // full-square output buffer.
    let mut out = vec![0.0f64; n_schur * n_schur];
    for j in 0..n_schur {
        for i in 0..n_schur {
            let val = if i >= j {
                contrib.data[j * contrib.dim + i]
            } else {
                contrib.data[i * contrib.dim + j]
            };
            out[j * n_schur + i] = val;
        }
    }

    let factors = SparseFactors {
        n,
        perm: symbolic.perm.clone(),
        perm_inv: symbolic.perm_inv.clone(),
        node_factors,
        needs_refinement,
        scaling: scaling_user,
        scaling_info,
        resolved_method: symbolic.resolved_method,
        resolved_amalgamation: symbolic.resolved_amalgamation,
        resolved_preprocess: symbolic.resolved_preprocess,
    };
    let schur = SchurBlock {
        dim: n_schur,
        data: out,
    };

    Ok((factors, total_inertia, schur))
}

/// Gated dispatcher: routes to the D.3 dense fast-path when
/// [`should_use_dense_fast_path`] fires, otherwise runs the
/// multifrontal supernodal body in
/// [`factorize_multifrontal_supernodal_with_workspace`].
///
/// Semantics are byte-identical to `factorize_multifrontal`: the
/// returned `SparseFactors` and `Inertia` are the same for the
/// same inputs. Scratch allocations are drawn from (and returned
/// to) `ws` instead of the global allocator, so repeated calls
/// with different matrices amortise heap traffic.
///
/// On a gate hit the dense path draws its `n * n` densify buffer
/// from `ws.dense_values` (pooled via `to_dense_into`) — see
/// `dev/research/phase-2.5.x-to-dense-pooling.md`.
pub fn factorize_multifrontal_with_workspace(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
    ws: &mut FactorWorkspace,
) -> Result<(SparseFactors, Inertia), FeralError> {
    if should_use_dense_fast_path(matrix.n, matrix.row_idx.len()) {
        return dense_fast_factor_with_workspace(matrix, params, ws);
    }
    factorize_multifrontal_supernodal_with_workspace(matrix, symbolic, params, ws)
}

/// Workspace-reusing supernodal body (un-gated).
///
/// See [`factorize_multifrontal_supernodal`] for the entry point
/// that bypasses the D.3 gate. Directly callable from tests that
/// need forced-multifrontal behavior on an in-gate matrix.
///
/// See `dev/plans/factor-workspace.md` for the rollout plan and
/// `tests/factor_workspace_parity.rs` for the guardrail tests
/// enforcing bit-level equivalence with the no-workspace path.
pub fn factorize_multifrontal_supernodal_with_workspace(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
    ws: &mut FactorWorkspace,
) -> Result<(SparseFactors, Inertia), FeralError> {
    // Phase 2.10 profiler. When `params.profiler.is_none()`, every
    // `Instant::now()` below is gated out, so the production path
    // does no timing work.
    let t_total = params.profiler.as_ref().map(|_| Instant::now());
    let t_prologue = params.profiler.as_ref().map(|_| Instant::now());

    // Track B1 (`dev/plans/per-factor-cost-cluster.md`): per-sub-phase
    // attribution of the prologue. `profiling` gates every
    // `Instant::now()` below, so the un-profiled production path does
    // no extra timing work.
    let profiling = params.profiler.is_some();
    let mut bd = PrologueBreakdown::default();
    let tic = || profiling.then(Instant::now);
    let toc = |t: Option<Instant>| t.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);

    let n = symbolic.n;
    let n_snodes = symbolic.supernodes.len();

    // Re-establish the `row_map` invariant (all entries `usize::MAX`,
    // length >= n) unconditionally, so a prior error-exit that
    // skipped the per-supernode clear cannot leak state into this
    // call. `clear()` keeps capacity; `resize` rewrites entries —
    // cost is O(n), not O(n_snodes * n) as the pre-workspace code
    // paid.
    let t_phase = tic();
    ws.row_map.clear();
    ws.row_map.resize(n, usize::MAX);
    bd.row_map_us = toc(t_phase);

    // β refactor: scaling is a numeric-phase concern, computed
    // here against the live matrix values, not cached on the
    // value-agnostic `SymbolicFactorization`. Returns the user-
    // order scaling vector and a diagnostic info enum.
    //
    // Phase 2.4.4: if the symbolic phase ran `LdltCompress`, it
    // already produced an `Mc64Cache` that we reuse here when the
    // scaling strategy also resolves to MC64 — O(n) post-processing
    // instead of a second Hungarian.
    let t_phase = tic();
    let (scaling_user, scaling_info) =
        compute_scaling_with_cache(matrix, &params.scaling, symbolic.cached_mc64.as_ref())?;
    bd.scaling_us = toc(t_phase);
    // `PartialSingular` is routine for IPM hosts factorizing
    // structurally rank-deficient KKT systems; structurally singular
    // matrices are allowed to proceed — they typically surface the
    // issue as a zero pivot during numeric factorization, the right
    // layer to reject. The stderr breadcrumb is opt-in (#43): default
    // off so feral stays quiet as a library; the same fact is always
    // available structurally via `Solver::scaling_info()`.
    if params.warn_partial_singular {
        if let crate::scaling::ScalingInfo::PartialSingular { n_unmatched } = &scaling_info {
            eprintln!(
                "warning: MC64 matching left {} of {} variables unmatched; \
                 scaling is identity on those rows/columns",
                n_unmatched, n
            );
        }
    }
    // Pivot-order cache of `scaling_user`: for each pivot index k,
    // `scaling_pivot_order[k] == scaling_user[symbolic.perm[k]]`.
    // This matches the assembly-time lookup pattern below where the
    // permuted CSC is indexed in pivot positions.
    let t_phase = tic();
    let scaling_pivot_order: Vec<f64> =
        symbolic.perm.iter().map(|&old| scaling_user[old]).collect();
    bd.scaling_pivot_order_us = toc(t_phase);
    debug_assert_eq!(scaling_pivot_order.len(), n);

    // Permute the matrix values into the new ordering. The
    // `from_triplets` rebuild inside is timed separately (B1 prime
    // suspect) and returned as `from_triplets_us`.
    //
    // Issue #56 Lever A.2: when `params.pattern_reused_hint` is set
    // (Solver flips it on per-call when the symbolic-cache fingerprint
    // matches), the cached permute structure on `ws.permute_cache` is
    // used to scatter values in O(nnz) — skipping the triplet sort
    // that dominated the prologue on Thomson IPM trajectories.
    let t_phase = tic();
    let (permuted, from_triplets_us) = permute_csc_values_with_cache(
        matrix,
        &symbolic.perm,
        &symbolic.perm_inv,
        profiling,
        params.pattern_reused_hint,
        &mut ws.permute_cache,
    )?;
    bd.permute_us = toc(t_phase);
    bd.permute_from_triplets_us = from_triplets_us;

    // F-01: raise the per-supernode BK `zero_tol` to the Wilkinson
    // backward error floor `n · EPS · ||A_scaled||_inf` so
    // rank-deficiency pivots that surface during elimination are
    // classified as zero rather than as small-but-real. No-op under
    // `on_zero_pivot == Fail`. See
    // `dev/research/f01-rankdef-underreporting.md`.
    let t_phase = tic();
    let local_params = apply_post_scaling_overrides(
        params,
        scaled_matrix_infnorm(&permuted, &scaling_pivot_order),
        n,
    );
    let params: &NumericParams = local_params.as_ref().unwrap_or(params);
    bd.infnorm_tol_us = toc(t_phase);

    // Issue #56 Lever A.1: reuse `symbolic.permuted_pattern` (which is
    // `permute_pattern(&matrix.symmetric_pattern(), &perm)`) instead of
    // recomputing `permuted.symmetric_pattern()` here. Both produce the
    // full symmetric pattern of P·A·Pᵀ with sorted columns. The recompute
    // was ~5% of total factor wall on Thomson n=100 and ~4% at n=200.
    let t_phase = tic();
    let full_pattern = &symbolic.permuted_pattern;
    bd.symmetric_pattern_us = toc(t_phase);

    // Phase 2.3 Step 5: identify root supernodes (no parent in the etree
    // forest). A node is a root iff no other supernode lists it as a
    // child. Roots must run with `may_delay = false` so
    // `ZeroPivotAction::ForceAccept` absorbs any unstable pivots instead
    // of delaying them to a non-existent ancestor. On disconnected
    // matrices the forest has multiple roots — this handles them
    // uniformly.
    let t_phase = tic();
    let mut is_root = vec![true; n_snodes];
    for snode in &symbolic.supernodes {
        for &child_idx in &snode.children {
            if child_idx < n_snodes {
                is_root[child_idx] = false;
            }
        }
    }

    // Storage for contribution blocks (one per supernode, freed after parent assembly)
    let mut contrib_blocks: Vec<Option<ContribBlock>> = (0..n_snodes).map(|_| None).collect();

    let mut node_factors: Vec<NodeFactors> = Vec::with_capacity(n_snodes);
    let mut total_inertia = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    let mut needs_refinement = false;
    bd.setup_us = toc(t_phase);

    // Process supernodes in postorder (children before parents).
    // Phase 2.5.2 Step B: per-supernode body extracted into
    // `factor_one_supernode` so a parallel task-graph driver (Step C)
    // can invoke it independently per-supernode. Sequential behaviour
    // is bit-exact against the pre-extraction loop — the helper is a
    // direct lift of the original loop body.
    //
    // Phase 2.9 (`dev/plans/phase-2.9-small-leaf-subtree.md`): when
    // `params.small_leaf == On`, leaf supernodes that were grouped
    // at symbolic time are dispatched to `factor_one_small_leaf` in
    // a single batched sweep per group. Group members are
    // postorder-consecutive indices, so we advance `snode_idx` past
    // the whole group after processing it; non-grouped supernodes
    // take the generic path exactly as before. The gate is `Off`
    // by default.
    let use_small_leaf =
        params.small_leaf == SmallLeafBatch::On && !symbolic.small_leaf_groups.is_empty();
    let mut snode_idx = 0usize;

    let prologue_us = t_prologue.map(|t| t.elapsed().as_micros() as u64);

    while snode_idx < n_snodes {
        if use_small_leaf {
            if let Some(gid) = symbolic.snode_group[snode_idx] {
                let group = &symbolic.small_leaf_groups[gid];
                debug_assert_eq!(
                    group.members.first(),
                    Some(&snode_idx),
                    "group members must start at current snode_idx"
                );
                for (i, &m) in group.members.iter().enumerate() {
                    let t_snode = params.profiler.as_ref().map(|_| Instant::now());
                    let node = factor_one_small_leaf(
                        m,
                        &group.member_rows[i],
                        symbolic,
                        &permuted,
                        &scaling_pivot_order,
                        &is_root,
                        params,
                        ws,
                        &mut contrib_blocks,
                    )?;
                    if let (Some(arc), Some(t)) = (params.profiler.as_ref(), t_snode) {
                        let snode = &symbolic.supernodes[m];
                        let timing = SupernodeTiming {
                            snode_idx: m,
                            nrow: snode.nrow,
                            ncol: snode.ncol,
                            us: t.elapsed().as_micros() as u64,
                            // Phase breakdown is not instrumented on the
                            // small-leaf batch path (`factor_one_small_leaf`).
                            assembly_us: 0,
                            densefactor_us: 0,
                            panelfactor_us: 0,
                            schur_us: 0,
                            scalartail_us: 0,
                        };
                        if let Ok(mut prof) = arc.lock() {
                            prof.timings.push(timing);
                        }
                    }
                    total_inertia.positive += node.inertia.positive;
                    total_inertia.negative += node.inertia.negative;
                    total_inertia.zero += node.inertia.zero;
                    if node.frontal_factors.needs_refinement {
                        needs_refinement = true;
                    }
                    node_factors.push(node);
                }
                snode_idx += group.members.len();
                continue;
            }
        }

        let t_snode = params.profiler.as_ref().map(|_| Instant::now());
        // Issue #44 phase-probe: snapshot the global phase counters
        // before/after so each supernode's record carries its own delta.
        // The counters are process-global `AtomicU64`s; this driver loop
        // is single-threaded so before/after differencing is exact.
        let phase_before = phase_timing::snapshot();
        let node = factor_one_supernode(
            snode_idx,
            symbolic,
            &permuted,
            full_pattern,
            &scaling_pivot_order,
            &is_root,
            params,
            ws,
            &mut contrib_blocks,
            0, // nvschur: standard path has no Schur tail
        )?;
        if let (Some(arc), Some(t)) = (params.profiler.as_ref(), t_snode) {
            let snode = &symbolic.supernodes[snode_idx];
            let phase_after = phase_timing::snapshot();
            let timing = SupernodeTiming {
                snode_idx,
                nrow: snode.nrow,
                ncol: snode.ncol,
                us: t.elapsed().as_micros() as u64,
                assembly_us: (phase_after.0 - phase_before.0) / 1000,
                densefactor_us: (phase_after.1 - phase_before.1) / 1000,
                panelfactor_us: (phase_after.2 - phase_before.2) / 1000,
                schur_us: (phase_after.3 - phase_before.3) / 1000,
                scalartail_us: (phase_after.4 - phase_before.4) / 1000,
            };
            if let Ok(mut prof) = arc.lock() {
                prof.timings.push(timing);
            }
        }

        total_inertia.positive += node.inertia.positive;
        total_inertia.negative += node.inertia.negative;
        total_inertia.zero += node.inertia.zero;
        if node.frontal_factors.needs_refinement {
            needs_refinement = true;
        }
        node_factors.push(node);
        snode_idx += 1;
    }

    let t_epilogue = params.profiler.as_ref().map(|_| Instant::now());

    let result = Ok((
        SparseFactors {
            n,
            perm: symbolic.perm.clone(),
            perm_inv: symbolic.perm_inv.clone(),
            node_factors,
            needs_refinement,
            // β refactor: scaling vector + diagnostic info are
            // produced by `compute_scaling` at the top of this
            // function (no longer cached on `SymbolicFactorization`).
            // Solve operates at the user API boundary so it needs
            // user-order indexing, not the pivot-order cache used
            // at assembly time.
            scaling: scaling_user,
            scaling_info,
            resolved_method: symbolic.resolved_method,
            resolved_amalgamation: symbolic.resolved_amalgamation,
            resolved_preprocess: symbolic.resolved_preprocess,
        },
        total_inertia,
    ));

    if let Some(arc) = params.profiler.as_ref() {
        if let Ok(mut prof) = arc.lock() {
            prof.prologue_us = prologue_us.unwrap_or(0);
            prof.prologue_breakdown = bd;
            prof.epilogue_us = t_epilogue
                .map(|t| t.elapsed().as_micros() as u64)
                .unwrap_or(0);
            prof.total_us = t_total.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);
        }
    }

    result
}

/// Factor a single supernode in isolation.
///
/// Phase 2.5.2 Step B: extracted from
/// [`factorize_multifrontal_supernodal_with_workspace`]'s per-supernode
/// loop body so the same code path can be reused by a future parallel
/// task-graph driver (Step C). Preserves the exact semantics of the
/// original loop iteration:
///
/// * Takes child contribution blocks out of `contrib_blocks` (via
///   `Option::take`) — children must have been produced by a prior
///   call for this same `contrib_blocks`.
/// * Writes the produced contribution block (if any) into
///   `contrib_blocks[snode_idx]`.
/// * Uses `ws.row_map`, `ws.frontal_values`, `ws.build_delayed`,
///   `ws.build_trailing`, `ws.build_seen` as scratch; respects the
///   same entry/exit invariants (row_map all `usize::MAX`, build_seen
///   all `false`).
///
/// Returns the `NodeFactors` for the supernode. The caller accumulates
/// inertia / `needs_refinement` from it.
#[allow(clippy::too_many_arguments)]
fn factor_one_supernode(
    snode_idx: usize,
    symbolic: &SymbolicFactorization,
    permuted: &CscMatrix,
    full_pattern: &crate::sparse::csc::CscPattern,
    scaling_pivot_order: &[f64],
    is_root: &[bool],
    params: &NumericParams,
    ws: &mut FactorWorkspace,
    contrib_blocks: &mut [Option<ContribBlock>],
    nvschur: usize,
) -> Result<NodeFactors, FeralError> {
    let snode = &symbolic.supernodes[snode_idx];
    let own_ncol = snode.ncol();
    let nrow = snode.nrow;

    if nrow == 0 || own_ncol == 0 {
        return Ok(NodeFactors {
            first_col: snode.first_col,
            ncol: 0,
            nelim: 0,
            n_delayed_in: 0,
            nrow: 0,
            row_indices: Vec::new(),
            frontal_factors: FrontalFactors {
                nrow: 0,
                ncol: 0,
                nelim: 0,
                l: Vec::new(),
                d_diag: Vec::new(),
                d_subdiag: Vec::new(),
                perm: Vec::new(),
                perm_inv: Vec::new(),
                contrib: Vec::new(),
                contrib_dim: 0,
                n_delayed: 0,
                inertia: Inertia {
                    positive: 0,
                    negative: 0,
                    zero: 0,
                },
                needs_refinement: false,
                n_rook_rescues: 0,
                n_tiny: 0,
                zero_tol: params.bk.zero_tol,
                zero_tol_2x2: params.bk.zero_tol_2x2,
            },
            inertia: Inertia {
                positive: 0,
                negative: 0,
                zero: 0,
            },
        });
    }

    // Phase 2.3 Step 5: count delayed columns arriving from each
    // child. Children that were processed under `may_delay = true`
    // may have left `n_delayed` fully-summed columns un-eliminated
    // in the top-left of their contribution block; these re-enter
    // pivot search at this node as additional fully-summed columns
    // on top of `snode.ncol()`.
    let n_delayed_in: usize = snode
        .children
        .iter()
        .filter_map(|&c| contrib_blocks[c].as_ref())
        .map(|c| c.n_delayed)
        .sum();
    let expanded_ncol = own_ncol + n_delayed_in;

    // Phase B3+B5 (issue #55): symbolic-analysis-time delay budget.
    // When the supernode received more delayed pivots from its children
    // than `delayed_capacity` permits, two dispositions are possible:
    //   - CB armed (`cascade_break_ratio.is_some()`) → engage the
    //     sign-preserving static-perturbation fallback at this
    //     supernode, mirroring MUMPS's `INFO(2)` recovery path.
    //   - CB disarmed → return `DelayBudgetExceeded` so the caller can
    //     restart with a larger budget multiplier or fall back to a
    //     different solver path.
    // Root supernodes are exempt from the error path: by the time
    // delays reach the root the frontal size is already committed and
    // there is no further delay target — the root must factor what it
    // received.
    let budget_exceeded =
        snode.delayed_capacity != usize::MAX && n_delayed_in > snode.delayed_capacity;
    let cb_armed = params.cascade_break_ratio.is_some();
    if budget_exceeded && !cb_armed && !is_root[snode_idx] {
        return Err(FeralError::DelayBudgetExceeded {
            supernode: snode_idx,
            required: n_delayed_in,
            capacity: snode.delayed_capacity,
        });
    }

    // Build the row indices for this frontal. The default layout is
    // [own native cols (own_ncol) | delayed cols from children (n_delayed_in) | trailing rows].
    let _pt_asm = phase_timing::start();
    let _pt_br = phase_timing::start();
    let mut row_indices = build_row_indices(
        snode,
        full_pattern,
        contrib_blocks,
        &mut ws.build_delayed,
        &mut ws.build_trailing,
        &mut ws.build_seen,
    );
    phase_timing::stop(&phase_timing::BUILDROW_NS, _pt_br);
    let actual_nrow = row_indices.len();
    debug_assert!(
        actual_nrow >= expanded_ncol,
        "row_indices ({}) must cover the expanded fully-summed block ({})",
        actual_nrow,
        expanded_ncol
    );
    // Trailing-row invariant: every row index at positions
    // [expanded_ncol..actual_nrow) must be strictly above the supernode's
    // own columns. Rows < first_col + own_ncol indicate either upper-triangle
    // pollution from a symmetrized pattern or a stale contrib block leaking
    // into the parent's frontal. See dev/research/build-row-indices-fix.md.
    #[cfg(debug_assertions)]
    {
        let first_col = snode.first_col;
        let own_last = first_col + own_ncol;
        for (pos, &r) in row_indices.iter().enumerate().skip(expanded_ncol) {
            debug_assert!(
                r >= own_last,
                "trailing row {} at position {} of supernode (first_col={}, own_ncol={}, expanded_ncol={}) is < first_col+own_ncol={}",
                r, pos, first_col, own_ncol, expanded_ncol, own_last
            );
        }
    }
    // F3.2b layout fix: when this is a Schur supernode (`nvschur > 0`)
    // that received delayed columns from descendants (`n_delayed_in > 0`),
    // the default layout above places own (Schur) cols at frontal
    // positions [0, own_ncol) and delayed cols at [own_ncol, expanded_ncol).
    // The BK pivot loop in `factor_frontal_blocked_in_place` eliminates
    // positions [0, ncol_eff) where ncol_eff = expanded_ncol - nvschur =
    // n_delayed_in, so without this swap the Schur cols sit inside the
    // eliminable range and get factored out — which is exactly the
    // opposite of what we want. Swap so delayed cols come first
    // (eliminable) and Schur cols come after (excluded from pivoting,
    // per the BK gate at src/dense/factor.rs:1670).
    let own_col_offset = if nvschur > 0 && n_delayed_in > 0 {
        let mut swapped = Vec::with_capacity(actual_nrow);
        swapped.extend_from_slice(&row_indices[own_ncol..expanded_ncol]);
        swapped.extend_from_slice(&row_indices[..own_ncol]);
        swapped.extend_from_slice(&row_indices[expanded_ncol..]);
        row_indices = swapped;
        n_delayed_in
    } else {
        0
    };

    // Populate the pooled `ws.row_map`. Invariant on entry: every entry
    // is `usize::MAX`. Mirror-clear at the end restores it.
    for (local, &global) in row_indices.iter().enumerate() {
        ws.row_map[global] = local;
    }

    // Step 1: Assemble original matrix entries into frontal, applying
    // symmetric scaling D·A·D in place. Own cols sit at frontal positions
    // [own_col_offset, own_col_offset + own_ncol); for the standard path
    // own_col_offset = 0, for the Schur swap path it's n_delayed_in.
    let scaling = scaling_pivot_order;
    let frontal_buf = std::mem::take(&mut ws.frontal_values);
    let mut frontal = SymmetricMatrix::from_pooled_buf(actual_nrow, frontal_buf);
    let _pt_sc = phase_timing::start();
    for (k_local, &gj) in row_indices[own_col_offset..own_col_offset + own_ncol]
        .iter()
        .enumerate()
    {
        let local_j = own_col_offset + k_local;
        let s_j = scaling[gj];
        for k in permuted.col_ptr[gj]..permuted.col_ptr[gj + 1] {
            let gi = permuted.row_idx[k];
            let local_i = ws.row_map[gi];
            if local_i != usize::MAX {
                let val = permuted.values[k] * scaling[gi] * s_j;
                frontal.set(local_i, local_j, val);
            }
        }
    }
    phase_timing::stop(&phase_timing::SCATTER_NS, _pt_sc);

    // Step 2: Assemble child contribution blocks (extend-add).
    //
    // Phase C: once `extend_add` has consumed `contrib`, move its `data`
    // Vec into the factor scratch's single-slot pool. The next
    // supernode's factor kernel will take it instead of allocating a
    // fresh Vec. If the slot is already occupied (multi-child front),
    // the new buffer overwrites the slot — the old one is freed
    // normally. This keeps bookkeeping to one branch per child.
    let _pt_ea = phase_timing::start();
    for &child_idx in &snode.children {
        if let Some(mut contrib) = contrib_blocks[child_idx].take() {
            extend_add(&contrib, &ws.row_map, &mut frontal);
            ws.factor_scratch.contrib_pool = Some(std::mem::take(&mut contrib.data));
        }
    }
    phase_timing::stop(&phase_timing::EXTENDADD_NS, _pt_ea);
    phase_timing::stop(&phase_timing::ASSEMBLY_NS, _pt_asm);

    // Step 3: Factor the frontal in place (W-3a). `frontal.data`
    // content is undefined on return; the buffer goes back to the pool.
    //
    // F3.2b: `nvschur` Schur columns at positions
    // `[expanded_ncol - nvschur, expanded_ncol)` are excluded from the
    // eliminable range. The BK pivot loop only swaps within
    // `[0, ncol_eff)` (see `dense::factor` r_is_fully_summed gate at
    // src/dense/factor.rs:1670), so Schur columns stay at their
    // original positions and end up in the contribution block in the
    // user-supplied order. nvschur > 0 implies is_root (Schur tail at
    // top of etree post-F3.2a), so may_delay is forced to false.
    debug_assert!(
        nvschur == 0 || is_root[snode_idx],
        "nvschur > 0 only valid at root supernodes (Schur tail invariant)"
    );
    debug_assert!(nvschur <= expanded_ncol);
    // Phase B5 (issue #55): cascade-break trigger is the symbolic
    // delay budget. When `budget_exceeded` and CB is armed,
    // perturb-in-place at this supernode rather than propagate the
    // overflow upward — the structural analogue of MUMPS's "delay
    // capacity exhausted ⇒ static perturbation" branch
    // (`dfac_front_aux.F:1251-1331`). The legacy heuristic
    // (`n_delayed_in/expanded_ncol ≥ ratio`) is retained as a
    // secondary trigger only when the symbolic capacity is unbounded
    // (`delayed_capacity == usize::MAX` — old / non-budgeted paths)
    // so callers that hand-tune `cascade_break_ratio` retain
    // backward-compatible behavior.
    //
    // Defaults when callers opt in are empirical; the budget-based
    // trigger removes the perf-tax on cascade-victim problems while
    // ensuring CB never perturbs a pivot that MUMPS would delay
    // (issue #55 Phase 0 evidence: marine_1600_0017, nuffield2_trap).
    let cascade_break = match params.cascade_break_ratio {
        Some(r) if !is_root[snode_idx] && params.allow_delayed_pivots && expanded_ncol > 0 => {
            if snode.delayed_capacity != usize::MAX {
                budget_exceeded
            } else {
                symbolic.n >= CASCADE_BREAK_MIN_N
                    && (n_delayed_in as f64) / (expanded_ncol as f64) >= r
            }
        }
        _ => false,
    };
    let may_delay = !is_root[snode_idx] && params.allow_delayed_pivots && !cascade_break;
    let local_bk;
    let bk_ref: &BunchKaufmanParams = if cascade_break {
        // When `cascade_break_eps` is set, use the bounded-Δ
        // perturbation path; otherwise fall back to the legacy
        // unbounded `ForceAccept`. The eps is treated as an absolute
        // floor per pivot — callers should pre-multiply by an
        // estimate of `||A||_∞` when working with non-normalized
        // matrices.
        let on_zero = match params.cascade_break_eps {
            Some(eps) => ZeroPivotAction::PerturbToEps { abs_floor: eps },
            None => ZeroPivotAction::ForceAccept,
        };
        local_bk = BunchKaufmanParams {
            on_zero_pivot: on_zero,
            ..params.bk.clone()
        };
        &local_bk
    } else {
        &params.bk
    };
    let eliminable = expanded_ncol - nvschur;
    // Issue #34 phase (d): SQD fast-path. The diagonal kernel ignores
    // `may_delay` (SQD never delays — it either accepts the diagonal
    // pivot or trips `SqdContractViolated`) and `bk_ref` overrides
    // (no cascade-break logic applies since 2x2 pivots aren't formed).
    let trace = supernode_trace_enabled();
    let t_sn = trace.then(Instant::now);
    let _pt_df = phase_timing::start();
    let mut ff = if params.sqd_mode {
        factor_frontal_diagonal_in_place(&mut frontal, eliminable, &params.bk)?
    } else {
        factor_frontal_blocked_in_place_with_scratch(
            &mut frontal,
            eliminable,
            may_delay,
            bk_ref,
            &mut ws.factor_scratch,
        )?
    };
    phase_timing::stop(&phase_timing::DENSEFACTOR_NS, _pt_df);
    if let Some(t0) = t_sn {
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "[sn-trace] sn={snode_idx} nrow={actual_nrow} exp_ncol={expanded_ncol} \
             elim={eliminable} n_del_in={n_delayed_in} may_del={may_delay} cb={cascade_break} \
             nelim={} n_del_out={} rook_rescues={} pos={} neg={} zero={} ms={ms:.3}",
            ff.nelim,
            ff.n_delayed,
            ff.n_rook_rescues,
            ff.inertia.positive,
            ff.inertia.negative,
            ff.inertia.zero,
        );
    }
    ws.frontal_values = frontal.data;

    let node_inertia = ff.inertia.clone();
    let node_nelim = ff.nelim;
    let node_n_delayed = ff.n_delayed;

    // Step 4: Store contribution block for parent. Move
    // `ff.contrib` directly into `ContribBlock::data` (W-3b: avoid the
    // 30 MB clone on CHAINWOO root). After this move,
    // `frontal_factors.contrib` in the saved `NodeFactors` is empty —
    // production solve paths only read `l`, `d_diag`, `d_subdiag`,
    // `perm`, `perm_inv` from `frontal_factors`; `contrib` is consumed
    // by the parent supernode during assembly and is dead data
    // afterward.
    if ff.contrib_dim > 0 {
        let cdim = ff.contrib_dim;
        let mut contrib_row_indices = Vec::with_capacity(cdim);
        for cj in 0..cdim {
            contrib_row_indices.push(row_indices[ff.perm[node_nelim + cj]]);
        }
        let contrib_data = std::mem::take(&mut ff.contrib);
        contrib_blocks[snode_idx] = Some(ContribBlock {
            row_indices: contrib_row_indices,
            data: contrib_data,
            dim: cdim,
            n_delayed: node_n_delayed,
        });
    }

    // Restore the `row_map` invariant.
    for &global in &row_indices {
        ws.row_map[global] = usize::MAX;
    }

    Ok(NodeFactors {
        first_col: snode.first_col,
        ncol: expanded_ncol,
        nelim: node_nelim,
        n_delayed_in,
        nrow: actual_nrow,
        row_indices,
        frontal_factors: ff,
        inertia: node_inertia,
    })
}

/// Factor a single true-leaf supernode that was pre-qualified for the
/// SmallLeafSubtree batched path (phase 2.9).
///
/// This is a leaf specialisation of `factor_one_supernode` — true leaves
/// have no children, so:
///
/// * `n_delayed_in == 0`, `expanded_ncol == own_ncol`.
/// * No extend-add pass (the children loop is empty anyway).
/// * `row_indices` is passed in pre-computed at symbolic time
///   (`SmallLeafGroup::member_rows`), saving the per-front
///   `build_row_indices` call and its `build_delayed`/`build_seen`
///   scratch churn on every leaf.
///
/// All other semantics — scaling, `factor_frontal_blocked`, contribution
/// block deposit, `row_map` write/restore — match the generic path
/// byte-for-byte, which is what the parity tests in
/// `tests/small_leaf_parity.rs` verify.
#[allow(clippy::too_many_arguments)]
fn factor_one_small_leaf(
    snode_idx: usize,
    precomputed_rows: &[usize],
    symbolic: &SymbolicFactorization,
    permuted: &CscMatrix,
    scaling_pivot_order: &[f64],
    is_root: &[bool],
    params: &NumericParams,
    ws: &mut FactorWorkspace,
    contrib_blocks: &mut [Option<ContribBlock>],
) -> Result<NodeFactors, FeralError> {
    let snode = &symbolic.supernodes[snode_idx];
    debug_assert!(
        snode.children.is_empty(),
        "factor_one_small_leaf called on non-leaf supernode {}",
        snode_idx
    );

    let own_ncol = snode.ncol();
    let nrow = snode.nrow;

    if nrow == 0 || own_ncol == 0 {
        return Ok(NodeFactors {
            first_col: snode.first_col,
            ncol: 0,
            nelim: 0,
            n_delayed_in: 0,
            nrow: 0,
            row_indices: Vec::new(),
            frontal_factors: FrontalFactors {
                nrow: 0,
                ncol: 0,
                nelim: 0,
                l: Vec::new(),
                d_diag: Vec::new(),
                d_subdiag: Vec::new(),
                perm: Vec::new(),
                perm_inv: Vec::new(),
                contrib: Vec::new(),
                contrib_dim: 0,
                n_delayed: 0,
                inertia: Inertia {
                    positive: 0,
                    negative: 0,
                    zero: 0,
                },
                needs_refinement: false,
                n_rook_rescues: 0,
                n_tiny: 0,
                zero_tol: params.bk.zero_tol,
                zero_tol_2x2: params.bk.zero_tol_2x2,
            },
            inertia: Inertia {
                positive: 0,
                negative: 0,
                zero: 0,
            },
        });
    }

    let row_indices = precomputed_rows.to_vec();
    let actual_nrow = row_indices.len();
    let expanded_ncol = own_ncol;
    debug_assert!(actual_nrow >= expanded_ncol);

    for (local, &global) in row_indices.iter().enumerate() {
        ws.row_map[global] = local;
    }

    let scaling = scaling_pivot_order;
    let frontal_buf = std::mem::take(&mut ws.frontal_values);
    let mut frontal = SymmetricMatrix::from_pooled_buf(actual_nrow, frontal_buf);
    for (local_j, &gj) in row_indices[..own_ncol].iter().enumerate() {
        let s_j = scaling[gj];
        for k in permuted.col_ptr[gj]..permuted.col_ptr[gj + 1] {
            let gi = permuted.row_idx[k];
            let local_i = ws.row_map[gi];
            if local_i != usize::MAX {
                let val = permuted.values[k] * scaling[gi] * s_j;
                frontal.set(local_i, local_j, val);
            }
        }
    }

    // No extend-add: leaves have no children.

    // W-3a: factor in place; pool returns the (now-undefined) buffer.
    let may_delay = !is_root[snode_idx] && params.allow_delayed_pivots;
    // Issue #34 phase (d): SQD fast-path; see factor_one_supernode for
    // the rationale on bypassing `may_delay` and the BK overrides.
    let mut ff = if params.sqd_mode {
        factor_frontal_diagonal_in_place(&mut frontal, expanded_ncol, &params.bk)?
    } else {
        factor_frontal_blocked_in_place_with_scratch(
            &mut frontal,
            expanded_ncol,
            may_delay,
            &params.bk,
            &mut ws.factor_scratch,
        )?
    };
    ws.frontal_values = frontal.data;

    let node_inertia = ff.inertia.clone();
    let node_nelim = ff.nelim;
    let node_n_delayed = ff.n_delayed;

    // W-3b: move `ff.contrib` rather than clone (see internal variant
    // for the full contract).
    if ff.contrib_dim > 0 {
        let cdim = ff.contrib_dim;
        let mut contrib_row_indices = Vec::with_capacity(cdim);
        for cj in 0..cdim {
            contrib_row_indices.push(row_indices[ff.perm[node_nelim + cj]]);
        }
        let contrib_data = std::mem::take(&mut ff.contrib);
        contrib_blocks[snode_idx] = Some(ContribBlock {
            row_indices: contrib_row_indices,
            data: contrib_data,
            dim: cdim,
            n_delayed: node_n_delayed,
        });
    }

    for &global in &row_indices {
        ws.row_map[global] = usize::MAX;
    }

    Ok(NodeFactors {
        first_col: snode.first_col,
        ncol: expanded_ncol,
        nelim: node_nelim,
        n_delayed_in: 0,
        nrow: actual_nrow,
        row_indices,
        frontal_factors: ff,
        inertia: node_inertia,
    })
}

/// Minimum supernode count below which the parallel driver falls
/// through to sequential. Phase 2.5.2 Step D. Tentative value
/// (conservative): reassess after the corpus bench in Step E shows
/// where per-task overhead breaks even.
pub const N_PAR_MIN: usize = 32;

/// Minimum total assembly-tree flop estimate (sum of
/// `ncol * nrow^2` per supernode) at which the parallel driver is
/// expected to amortise rayon spawn / cv-wait overhead. Issue #19
/// follow-up to the Phase 2.5.2 Step E reassessment that never
/// closed.
///
/// Empirical calibration on Apple M4 Pro (14 rayon threads) across
/// two workload families:
///
/// - Poisson-KKT (`calibrate_par_min_flops`, sessions 2026-05-15-05
///   and -06): break-even at `est_flops ≈ 6×10⁶`.
/// - `robot_1600` KKT (`probe_issue_19`, session 2026-05-15-06):
///   iter 0000 (4.75×10⁶ flops) → parallel hurts 0.67×; iters 0001
///   /0003 (1.13×10⁷ flops) → parallel wins 1.42-1.48×; iter 0006
///   (9.43×10⁶ flops) → parallel wins 1.24×. Same crossover ≈ 6×10⁶.
///
/// The threshold is set just above break-even (10⁷ ≈ 1.7× safety
/// margin) so the gate fires sequential on iter 0000-class problems
/// where rayon overhead dominates, and parallel on iter 0001/0003-
/// class problems where parallel wins ≥1.4×. The intermediate iter
/// 0006 (9.43×10⁶) sits below the threshold and stays sequential —
/// we trade its 1.24× win for one decimal of safety margin against
/// the break-even.
///
/// The pre-issue-#19 history: this const was originally 10⁸ on a
/// freehand "100 µs spawn + 10 GFLOP/s" argument, which was ~10×
/// conservative. The persistent rayon `ThreadPool` reuse landed in
/// `91e028a` cut the per-call cv-wait overhead enough that the
/// original 12× wall regression on `robot_1600` no longer
/// reproduces; the gate's remaining job is to veto parallel on
/// problems below break-even, not to be a blunt safety belt.
///
/// Consumers on non-M4 hardware (or with non-Poisson tree shapes)
/// can override per-call via [`NumericParams::min_parallel_flops`]
/// (or `POUNCE_FERAL_MIN_PAR_FLOPS=<u64>` from pounce-feral).
pub const PAR_MIN_FLOPS: u64 = 10_000_000;

/// Cheap O(n_supernodes) flop-cost estimate for the entire assembly
/// tree. Used as the work gate inside [`should_parallelize_assembly`].
///
/// Per-supernode proxy: `ncol * nrow^2`. This overestimates a touch
/// (the true cost is `ncol^3/3 + ncol^2*nrow_below + ncol*nrow_
/// below^2`) but dominates the true cost by a small constant whenever
/// `nrow > 2*ncol`, which holds for all but trivial supernodes. The
/// proxy is order-of-magnitude correct, which is what the threshold
/// gate needs.
///
/// Returns `u64` because a non-trivial factorization can produce
/// flop counts well above `usize::MAX` on a 32-bit platform; on
/// 64-bit it's equivalent. Saturating arithmetic on the per-node
/// multiply so wildly pathological inputs can't panic — at `nrow ≥
/// 2^22` the cube already saturates and we'll definitely choose
/// parallel anyway.
pub fn estimate_assembly_flops(supernodes: &[Supernode]) -> u64 {
    supernodes
        .iter()
        .map(|s| {
            let ncol = s.ncol as u64;
            let nrow = s.nrow as u64;
            ncol.saturating_mul(nrow).saturating_mul(nrow)
        })
        .fold(0u64, |acc, x| acc.saturating_add(x))
}

/// Predicate: does the symbolic factorization present enough structure
/// and work for the parallel driver to win?
///
/// Three conditions, all required:
/// 1. `n_snodes >= N_PAR_MIN` — enough tasks to amortise thread-pool
///    overhead.
/// 2. At least one supernode has ≥ 2 children — a pure postorder
///    chain has zero sibling parallelism, so the parallel driver
///    would add only overhead.
/// 3. Estimated total tree flops `>= PAR_MIN_FLOPS` — gate the issue
///    #19 case where the assembly tree is structurally rich but every
///    individual supernode is too small to absorb rayon overhead
///    (small-KKT IPM control-NLP profile, e.g. robot_1600).
///
/// Uses the const [`PAR_MIN_FLOPS`] as the flop threshold. To override
/// per-call (e.g. consumers that have calibrated their hardware), see
/// [`should_parallelize_assembly_with_threshold`] or pass
/// `NumericParams::min_parallel_flops`.
pub fn should_parallelize_assembly(symbolic: &SymbolicFactorization) -> bool {
    should_parallelize_assembly_with_threshold(symbolic, PAR_MIN_FLOPS)
}

/// `should_parallelize_assembly` with the flop threshold supplied
/// explicitly. Useful when `NumericParams::min_parallel_flops` is set
/// or in tests that exercise threshold-edge behavior.
pub fn should_parallelize_assembly_with_threshold(
    symbolic: &SymbolicFactorization,
    min_flops: u64,
) -> bool {
    if symbolic.supernodes.len() < N_PAR_MIN {
        return false;
    }
    if !symbolic.supernodes.iter().any(|s| s.children.len() >= 2) {
        return false;
    }
    estimate_assembly_flops(&symbolic.supernodes) >= min_flops
}

/// Gated parallel entry point. Phase 2.5.2 Step D.
///
/// Dispatch order:
/// 1. [`should_use_dense_fast_path`] — dense fast-path takes precedence.
/// 2. If [`should_parallelize_assembly`] returns true, dispatch to
///    the rayon driver ([`factorize_multifrontal_supernodal_parallel`]).
/// 3. Otherwise, sequential multifrontal.
///
/// The workspace `ws` is used by the dense fast-path and the
/// sequential fall-through; the parallel driver owns its own
/// per-thread scratch internally.
pub fn factorize_multifrontal_parallel_with_workspace(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
    ws: &mut FactorWorkspace,
) -> Result<(SparseFactors, Inertia), FeralError> {
    if should_use_dense_fast_path(matrix.n, matrix.row_idx.len()) {
        return dense_fast_factor_with_workspace(matrix, params, ws);
    }
    let threshold = params.min_parallel_flops.unwrap_or(PAR_MIN_FLOPS);
    if should_parallelize_assembly_with_threshold(symbolic, threshold) {
        return factorize_multifrontal_supernodal_parallel(matrix, symbolic, params);
    }
    factorize_multifrontal_supernodal_with_workspace(matrix, symbolic, params, ws)
}

/// Fresh-workspace variant of [`factorize_multifrontal_parallel_with_workspace`].
pub fn factorize_multifrontal_parallel(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError> {
    let mut ws = FactorWorkspace::new();
    factorize_multifrontal_parallel_with_workspace(matrix, symbolic, params, &mut ws)
}

/// Rayon task-graph parallel driver for the multifrontal assembly tree.
///
/// Phase 2.5.2 Step C. Bit-exact parity with
/// [`factorize_multifrontal_supernodal_with_workspace`] on each
/// supernode, because:
///
/// * Each supernode's assembly (extend-add over children) happens in
///   a single task with children iterated in `snode.children` order —
///   the same FP sum order as sequential.
/// * Each task uses a per-thread `FactorWorkspace` drawn from
///   `thread_ws[rayon::current_thread_index()]`, so scratch buffers
///   are never shared across threads.
/// * The shared contribution-block store is mutex-protected; each
///   task only locks it briefly to stage its own children into a
///   local `Vec<Option<ContribBlock>>` and, later, to deposit its
///   own block. No mutex is held during the dense kernel.
///
/// Entry points that dispatch to this driver must check
/// `n_snodes >= N_PAR_MIN` first and otherwise fall through to the
/// sequential path (see `factorize_multifrontal_parallel_with_workspace`).
///
/// The `FactorWorkspace` passed in by the caller is **not** used for
/// per-task scratch — it is reserved for the caller's amortisation
/// semantics (future extension). Per-task workspaces are owned
/// internally, one per rayon worker thread.
pub fn factorize_multifrontal_supernodal_parallel(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError> {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;

    // Lever 1.1: enable intra-front (node-level) parallelism in the
    // per-front dense Schur update. This is set ONLY here, on the
    // parallel driver, so a serial backend — `Solver::with_parallel(false)`,
    // which routes to `factorize_multifrontal_supernodal_with_workspace`
    // — leaves `intrafront_parallel = false` and never spawns nested
    // rayon work (the pounce#79 oversubscription guarantee). The dense
    // kernel further gates on `INTRAFRONT_MIN_AREA`, so only wide fronts
    // actually fork. Bit-exact regardless of thread count (see
    // `apply_schur_panel_range`). See
    // `dev/research/lever-1.1-intrafront-parallel-schur.md`.
    // `FERAL_INTRAFRONT=0|off|false` disables Lever 1.1 for A/B
    // benchmarking and as a safety override; default on. Read once per
    // factorization (negligible beside the factor cost), mirroring the
    // `FERAL_PARALLEL` diagnostic affordance.
    let intrafront_on = !matches!(
        std::env::var("FERAL_INTRAFRONT").as_deref(),
        Ok("0") | Ok("off") | Ok("false") | Ok("no")
    );
    let params = &{
        let mut p = params.clone();
        p.bk.intrafront_parallel = intrafront_on;
        p
    };

    let n = symbolic.n;
    let n_snodes = symbolic.supernodes.len();
    let telemetry = params.parallel_telemetry.as_deref();

    // Setup — mirrors the sequential driver. Reuse the symbolic-phase
    // MC64 cache if present (see the sequential driver for details).
    let t_phase = telemetry.map(|_| std::time::Instant::now());
    let (scaling_user, scaling_info) =
        compute_scaling_with_cache(matrix, &params.scaling, symbolic.cached_mc64.as_ref())?;
    if params.warn_partial_singular {
        if let crate::scaling::ScalingInfo::PartialSingular { n_unmatched } = &scaling_info {
            eprintln!(
                "warning: MC64 matching left {} of {} variables unmatched; \
                 scaling is identity on those rows/columns",
                n_unmatched, n
            );
        }
    }
    let scaling_pivot_order: Vec<f64> =
        symbolic.perm.iter().map(|&old| scaling_user[old]).collect();
    if let (Some(t), Some(start)) = (telemetry, t_phase) {
        t.phase_scaling_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let t_phase = telemetry.map(|_| std::time::Instant::now());
    let (permuted, _) = permute_csc_values(matrix, &symbolic.perm, &symbolic.perm_inv, false)?;
    if let (Some(t), Some(start)) = (telemetry, t_phase) {
        t.phase_permute_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // F-01: raise the per-supernode BK `zero_tol` to the Wilkinson
    // backward error floor `n · EPS · ||A_scaled||_inf`. See sequential
    // driver above and `dev/research/f01-rankdef-underreporting.md`.
    let local_params = apply_post_scaling_overrides(
        params,
        scaled_matrix_infnorm(&permuted, &scaling_pivot_order),
        n,
    );
    let params: &NumericParams = local_params.as_ref().unwrap_or(params);

    // Issue #56 Lever A.1: reuse `symbolic.permuted_pattern` — see the
    // sequential driver above for the rationale.
    let t_phase = telemetry.map(|_| std::time::Instant::now());
    let full_pattern = &symbolic.permuted_pattern;
    if let (Some(t), Some(start)) = (telemetry, t_phase) {
        t.phase_symmetric_pattern_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let t_phase = telemetry.map(|_| std::time::Instant::now());

    let mut is_root = vec![true; n_snodes];
    for snode in &symbolic.supernodes {
        for &child_idx in &snode.children {
            if child_idx < n_snodes {
                is_root[child_idx] = false;
            }
        }
    }

    // Parent table: parents[c] == i iff i's children include c.
    let mut parents: Vec<Option<usize>> = vec![None; n_snodes];
    for (i, snode) in symbolic.supernodes.iter().enumerate() {
        for &c in &snode.children {
            if c < n_snodes {
                parents[c] = Some(i);
            }
        }
    }

    // Pending-children atomic counter per supernode. A supernode is
    // ready to process when its counter hits zero.
    let pending: Vec<AtomicUsize> = symbolic
        .supernodes
        .iter()
        .map(|s| {
            let cnt = s.children.iter().filter(|&&c| c < n_snodes).count();
            AtomicUsize::new(cnt)
        })
        .collect();

    // Shared state: contrib blocks, result slots, first error.
    let contrib_blocks: Mutex<Vec<Option<ContribBlock>>> =
        Mutex::new((0..n_snodes).map(|_| None).collect());
    let node_factors_out: Mutex<Vec<Option<NodeFactors>>> =
        Mutex::new((0..n_snodes).map(|_| None).collect());
    let first_error: Mutex<Option<FeralError>> = Mutex::new(None);
    if let (Some(t), Some(start)) = (telemetry, t_phase) {
        t.phase_tree_setup_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let t_phase = telemetry.map(|_| std::time::Instant::now());
    // Per-thread workspaces. Provision one workspace per rayon
    // worker PLUS one extra slot for the calling thread, which may
    // also execute tasks inside `rayon::scope` and has
    // `current_thread_index() == None`. Bin the calling thread's
    // tasks into the extra slot (index `num_threads`) to avoid
    // mutex-serializing it against worker 0 — and to prevent the
    // caller and worker 0 from time-sharing a single workspace.
    let num_threads = rayon::current_num_threads().max(1);
    let thread_ws: Vec<Mutex<FactorWorkspace>> = (0..num_threads + 1)
        .map(|_| {
            let mut w = FactorWorkspace::new();
            w.row_map.resize(n, usize::MAX);
            w.build_seen.resize(n, false);
            // Per-worker children-contrib lookup. Allocated once here
            // instead of per-task; slots are always `None` after each
            // task drains children + takes own (see run_parallel_task).
            w.local_contribs.resize_with(n_snodes, || None);
            Mutex::new(w)
        })
        .collect();
    if let (Some(t), Some(start)) = (telemetry, t_phase) {
        t.phase_thread_ws_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let t_phase = telemetry.map(|_| std::time::Instant::now());
    // Collect the true leaves (supernodes with no children) BEFORE
    // entering the scope. Using `pending[i].load() == 0` as the
    // seeding predicate is unsound: once the scope is live, workers
    // execute previously-spawned tasks concurrently with this loop
    // and decrement parents' counters. A non-leaf whose pending
    // counter just hit zero would then be spawned twice — once here
    // by the caller, and again by the final child via the
    // fetch_sub==1 trampoline in `run_parallel_task`.
    let leaves: Vec<usize> = symbolic
        .supernodes
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            if s.children.iter().all(|&c| c >= n_snodes) {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    if let (Some(t), Some(start)) = (telemetry, t_phase) {
        t.phase_leaves_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let t_phase = telemetry.map(|_| std::time::Instant::now());
    rayon::scope(|scope| {
        for &leaf_idx in &leaves {
            run_parallel_task(
                scope,
                leaf_idx,
                symbolic,
                &permuted,
                full_pattern,
                &scaling_pivot_order,
                &is_root,
                params,
                &parents,
                &pending,
                &contrib_blocks,
                &node_factors_out,
                &first_error,
                &thread_ws,
            );
        }
    });
    if let (Some(t), Some(start)) = (telemetry, t_phase) {
        t.phase_scope_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let t_phase = telemetry.map(|_| std::time::Instant::now());
    // Surface any first-error that the tasks captured.
    let err_opt = match first_error.into_inner() {
        Ok(v) => v,
        Err(p) => p.into_inner(),
    };
    if let Some(err) = err_opt {
        return Err(err);
    }

    // Collect node_factors in postorder (same order as sequential).
    let nodes_vec = match node_factors_out.into_inner() {
        Ok(v) => v,
        Err(p) => p.into_inner(),
    };
    let mut final_nodes: Vec<NodeFactors> = Vec::with_capacity(n_snodes);
    let mut total_inertia = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    let mut needs_refinement = false;
    for opt in nodes_vec.into_iter() {
        let node = match opt {
            Some(n) => n,
            None => {
                return Err(FeralError::InvalidInput(
                    "parallel driver: supernode was not processed (graph stall)".to_string(),
                ));
            }
        };
        total_inertia.positive += node.inertia.positive;
        total_inertia.negative += node.inertia.negative;
        total_inertia.zero += node.inertia.zero;
        if node.frontal_factors.needs_refinement {
            needs_refinement = true;
        }
        final_nodes.push(node);
    }
    if let (Some(t), Some(start)) = (telemetry, t_phase) {
        t.phase_collect_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    Ok((
        SparseFactors {
            n,
            perm: symbolic.perm.clone(),
            perm_inv: symbolic.perm_inv.clone(),
            node_factors: final_nodes,
            needs_refinement,
            scaling: scaling_user,
            scaling_info,
            resolved_method: symbolic.resolved_method,
            resolved_amalgamation: symbolic.resolved_amalgamation,
            resolved_preprocess: symbolic.resolved_preprocess,
        },
        total_inertia,
    ))
}

/// Spawn a single supernode factorization task into the rayon scope.
///
/// On completion, decrements the parent's pending counter and — if
/// the parent becomes ready — recursively spawns the parent into the
/// same scope. The top-level call seeds all leaf supernodes.
///
/// # Worker-stack depth is O(1) in tree height
///
/// The leaf→root climb is **trampolined through rayon's task queue**,
/// not native call-stack recursion. The whole body of this function is
/// a `scope.spawn(move |s| { … })`; the "recursive" call at the bottom
/// (`run_parallel_task(s, parent_idx, …)`) only *enqueues* the parent's
/// closure and returns — the parent's actual factorization runs in a
/// freshly spawned task after the current task's frame has popped, never
/// nested on top of it. So a deep / path-like elimination tree does
/// **not** drive native stack depth proportional to tree height; each
/// task uses a small constant frame plus per-front dense-kernel scratch
/// (bounded by front size, also independent of height).
///
/// This is measured, not assumed: factoring the deepest corpus matrix
/// (`c-big`, n = 345 241, supernode-tree height 1521) on this driver
/// succeeds on a worker stack as small as 32 KiB — far below the rayon
/// ~2 MiB default — and a synthetic tridiagonal chain (supernode-tree
/// height ~500) factors likewise; see
/// `tests/parallel_parity.rs::deep_chain_tree_no_stack_overflow` and
/// `dev/research/parallel-stack-depth-pounce79.md`. As a consequence
/// `ensure_parallel_pool` (src/numeric/solver.rs) does not need an
/// enlarged `stack_size`; the default worker stack suffices regardless
/// of tree shape. (The worker-stack overflow a downstream consumer once
/// hit came from *oversubscription* — running this parallel driver
/// nested inside the caller's own rayon `par_iter` — not from tree
/// depth; the fix is a serial inner backend via
/// `Solver::with_parallel(false)`, not a bigger stack.)
#[allow(clippy::too_many_arguments)]
fn run_parallel_task<'a>(
    scope: &rayon::Scope<'a>,
    snode_idx: usize,
    symbolic: &'a SymbolicFactorization,
    permuted: &'a CscMatrix,
    full_pattern: &'a crate::sparse::csc::CscPattern,
    scaling_pivot_order: &'a [f64],
    is_root: &'a [bool],
    params: &'a NumericParams,
    parents: &'a [Option<usize>],
    pending: &'a [std::sync::atomic::AtomicUsize],
    contrib_blocks: &'a std::sync::Mutex<Vec<Option<ContribBlock>>>,
    node_factors_out: &'a std::sync::Mutex<Vec<Option<NodeFactors>>>,
    first_error: &'a std::sync::Mutex<Option<FeralError>>,
    thread_ws: &'a [std::sync::Mutex<FactorWorkspace>],
) {
    use std::sync::atomic::Ordering;
    scope.spawn(move |s| {
        let telemetry = params.parallel_telemetry.as_deref();
        let t_task_start = telemetry.map(|_| std::time::Instant::now());
        if let Some(t) = telemetry {
            t.n_tasks.fetch_add(1, Ordering::Relaxed);
        }

        // Fast-exit if a prior task errored; the scope will still
        // drain, we just skip actual work.
        {
            let err_guard = match first_error.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if err_guard.is_some() {
                if let (Some(t), Some(start)) = (telemetry, t_task_start) {
                    t.task_wall_ns
                        .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                return;
            }
        }

        let snode = &symbolic.supernodes[snode_idx];
        let n_snodes = symbolic.supernodes.len();

        // Pick a per-thread workspace slot. `current_thread_index`
        // returns Some(worker_idx) when this task runs on a rayon
        // worker, and None when it runs on the calling thread
        // (rayon::scope donates the caller's thread to execute
        // tasks while it waits). The caller gets the last slot
        // (`thread_ws.len() - 1`) so it does not contend with any
        // worker's slot.
        let thread_idx = rayon::current_thread_index().unwrap_or(thread_ws.len() - 1);
        let thread_idx = thread_idx.min(thread_ws.len() - 1);
        let ws_mtx = &thread_ws[thread_idx];

        // N3: per-supernode profiler wall time. Independent of
        // `parallel_telemetry` (which measures driver-internal lock/wait
        // costs) — this mirrors the sequential driver's
        // `params.profiler` recording so `Solver::with_profiling(true)`
        // yields a populated report on the parallel dispatch too. Set
        // inside the factor block, consumed in the `Ok` arm below.
        let mut prof_us: Option<u64> = None;
        let (result, own_contrib) = {
            let t_ws_wait = telemetry.map(|_| std::time::Instant::now());
            let mut ws_guard = match ws_mtx.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if let (Some(t), Some(start)) = (telemetry, t_ws_wait) {
                t.ws_lock_wait_ns
                    .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }

            // Move the pooled local_contribs out of the workspace so
            // we can hand `&mut FactorWorkspace` and `&mut Vec<...>`
            // independently to `factor_one_supernode`. The vec returns
            // to the workspace at the end of this block. All slots are
            // guaranteed `None` going in (postcondition of the prior
            // task on this worker took both children's slots and the
            // own_contrib slot below).
            let mut local_contribs = std::mem::take(&mut ws_guard.local_contribs);

            // Stage child contributions: shared lock held only for
            // the drain, not across the factor body.
            {
                let t_wait_start = telemetry.map(|_| std::time::Instant::now());
                let mut shared = match contrib_blocks.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                if let (Some(t), Some(start)) = (telemetry, t_wait_start) {
                    t.contrib_wait_ns
                        .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                let hold_start = telemetry.map(|_| std::time::Instant::now());
                for &c in &snode.children {
                    if c < n_snodes {
                        local_contribs[c] = shared[c].take();
                    }
                }
                if let (Some(t), Some(start)) = (telemetry, hold_start) {
                    t.contrib_hold_ns
                        .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            }

            let factor_start = telemetry.map(|_| std::time::Instant::now());
            let prof_start = params.profiler.as_ref().map(|_| std::time::Instant::now());
            let res = factor_one_supernode(
                snode_idx,
                symbolic,
                permuted,
                full_pattern,
                scaling_pivot_order,
                is_root,
                params,
                &mut ws_guard,
                &mut local_contribs,
                0, // nvschur: parallel path is not used by Schur API
            );
            if let (Some(t), Some(start)) = (telemetry, factor_start) {
                t.factor_body_ns
                    .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            if let Some(start) = prof_start {
                prof_us = Some(start.elapsed().as_micros() as u64);
            }

            let own = local_contribs[snode_idx].take();
            // Return the pooled vec to the workspace. All slots are
            // `None` (children were taken into us above; own was just
            // taken out), so no clearing is needed.
            ws_guard.local_contribs = local_contribs;
            (res, own)
        };

        match result {
            Ok(node) => {
                // N3: record this supernode's profiler timing (completion
                // order, not postorder — the bucketed `report()` is
                // order-independent). The phase-breakdown fields are left
                // zero: they are derived from process-global phase atomics
                // that cannot be safely differenced across concurrent tasks
                // (finding N9), so only the wall `us` is meaningful here.
                if let (Some(arc), Some(us)) = (params.profiler.as_ref(), prof_us) {
                    let timing = SupernodeTiming {
                        snode_idx,
                        nrow: snode.nrow,
                        ncol: snode.ncol,
                        us,
                        assembly_us: 0,
                        densefactor_us: 0,
                        panelfactor_us: 0,
                        schur_us: 0,
                        scalartail_us: 0,
                    };
                    if let Ok(mut prof) = arc.lock() {
                        prof.timings.push(timing);
                    }
                }
                {
                    let t_wait_start = telemetry.map(|_| std::time::Instant::now());
                    let mut shared = match contrib_blocks.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    if let (Some(t), Some(start)) = (telemetry, t_wait_start) {
                        t.contrib_wait_ns
                            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    let hold_start = telemetry.map(|_| std::time::Instant::now());
                    shared[snode_idx] = own_contrib;
                    if let (Some(t), Some(start)) = (telemetry, hold_start) {
                        t.contrib_hold_ns
                            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                }
                {
                    let t_wait_start = telemetry.map(|_| std::time::Instant::now());
                    let mut nf = match node_factors_out.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    if let (Some(t), Some(start)) = (telemetry, t_wait_start) {
                        t.node_factors_wait_ns
                            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    let hold_start = telemetry.map(|_| std::time::Instant::now());
                    nf[snode_idx] = Some(node);
                    if let (Some(t), Some(start)) = (telemetry, hold_start) {
                        t.node_factors_hold_ns
                            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                }
                if let Some(parent_idx) = parents[snode_idx] {
                    let prev = pending[parent_idx].fetch_sub(1, Ordering::AcqRel);
                    if prev == 1 {
                        run_parallel_task(
                            s,
                            parent_idx,
                            symbolic,
                            permuted,
                            full_pattern,
                            scaling_pivot_order,
                            is_root,
                            params,
                            parents,
                            pending,
                            contrib_blocks,
                            node_factors_out,
                            first_error,
                            thread_ws,
                        );
                    }
                }
            }
            Err(e) => {
                let mut err_guard = match first_error.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                if err_guard.is_none() {
                    *err_guard = Some(e);
                }
            }
        }

        if let (Some(t), Some(start)) = (telemetry, t_task_start) {
            t.task_wall_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    });
}

/// Row-sum infinity norm of the symmetrically-scaled matrix `D · A · D`,
/// where `A` is stored as a CSC lower-triangle in pivot order and
/// `scaling[k]` is `D_kk` in pivot order.
///
/// Used by the F-01 fix to compute a post-scaling relative null-pivot
/// threshold for the BK kernel. See
/// `dev/research/f01-rankdef-underreporting.md`. O(nnz).
fn scaled_matrix_infnorm(permuted: &CscMatrix, scaling: &[f64]) -> f64 {
    let n = permuted.n;
    let mut row_sum = vec![0.0_f64; n];
    for j in 0..n {
        let s_j = scaling[j];
        for k in permuted.col_ptr[j]..permuted.col_ptr[j + 1] {
            let i = permuted.row_idx[k];
            let m = (permuted.values[k] * scaling[i] * s_j).abs();
            row_sum[i] += m;
            if i != j {
                row_sum[j] += m;
            }
        }
    }
    row_sum.into_iter().fold(0.0_f64, f64::max)
}

/// Row-sum infinity norm of a symmetric matrix stored densely in the
/// `SymmetricMatrix` convention used by the dense fast path:
/// column-major, lower triangle at `data[j*n + i]` for `i >= j`. The
/// upper triangle (`i < j`) may hold stale buffer contents and is
/// ignored.
fn scaled_matrix_infnorm_dense(data: &[f64], n: usize) -> f64 {
    let mut row_sum = vec![0.0_f64; n];
    for j in 0..n {
        for i in j..n {
            let m = data[j * n + i].abs();
            row_sum[i] += m;
            if i != j {
                row_sum[j] += m;
            }
        }
    }
    row_sum.into_iter().fold(0.0_f64, f64::max)
}

/// Compute the BK kernel's null-pivot tolerance from the scaled
/// matrix's infinity norm. Returns `sqrt(n) · EPS · ||A_scaled||_inf`,
/// matching the MUMPS 5.8.2 default for `CNTL(3)` under
/// `ICNTL(24) = 1` (null-pivot detection enabled).
///
/// Tradeoff: `n · EPS · ‖A‖` (the full Wilkinson backward-error bound
/// for LDLᵀ) catches more rank-deficient pivots but mis-classifies
/// genuine small eigenvalues on ill-conditioned non-singular matrices
/// (e.g. `synth/ill_cond_e14` n=100 cond=1e14: pivots at ~1e-14 are
/// real, not zero). The `sqrt(n)` formula keeps the floor strictly
/// below such pivots while still raising it ~`sqrt(n)`× above the
/// dense default of `EPS`. MA57 and SSIDS use `sqrt(EPS)` absolute,
/// which is much looser but configurable. See F-01 research note.
#[inline]
fn null_pivot_floor(scaled_infnorm: f64, n: usize) -> f64 {
    (n as f64).sqrt() * f64::EPSILON * scaled_infnorm
}

/// Apply the post-scaling pivot-tolerance overrides (F-01 null-pivot
/// floor and N2 static-pivot floor), both derived from the scaled
/// ∞-norm `‖D·A·D‖∞` the BK kernels actually operate on.
///
/// Returns `Some(local)` when the caller should use a cloned
/// `NumericParams` with raised `bk.null_pivot_tol` /
/// `bk.null_pivot_tol_2x2` and/or a scaled `bk.static_pivot_floor`, or
/// `None` when the original `params` is sufficient (no static threshold
/// set, and either the null-pivot floor is below the existing
/// `null_pivot_tol`, or `on_zero_pivot == Fail` preserves the
/// absolute-threshold contract for abort-on-zero callers).
///
/// Only `null_pivot_tol` is bumped — `zero_tol` (the solve-time
/// divide-skip floor, propagated to `Factors.zero_tol`) stays at the
/// strict EPS default. This keeps ill-conditioned but non-singular
/// matrices (e.g. `synth/ill_cond_e14`) usable at solve time while
/// the factor-time inertia count honestly reports rank deficiency.
/// See `dev/research/f01-rankdef-underreporting.md`.
///
/// The 2×2 block threshold is set to `floor · ‖A_scaled‖_inf` rather
/// than `floor²`. A near-singular 2×2 block of a symmetric matrix has
/// one eigenvalue at the pivot floor and one at the matrix scale, so
/// its determinant has magnitude ~ `floor · ‖A‖`. The default
/// `floor²` would only catch blocks with *both* eigenvalues at the
/// floor — orders of magnitude smaller than the actual
/// rank-deficiency signature.
fn apply_post_scaling_overrides(
    params: &NumericParams,
    scaled_infnorm: f64,
    n: usize,
) -> Option<NumericParams> {
    // N2 (`dev/research/repo-review-2026-06-09.md`): the MA57-style
    // static-pivot floor must be derived from the SCALED matrix the BK
    // kernels actually operate on (`D·A·D`), not the unscaled user
    // matrix. `static_pivot_threshold = t` is a *relative* threshold;
    // the absolute floor enforced on scaled pivots is `t · ‖D·A·D‖∞`.
    // The solver funnel previously computed `t · ‖A‖∞` from the
    // unscaled matrix, so under a norm-normalizing scaling (InfNorm /
    // MC64) `t` behaved like a different value in pivot space by the
    // scaling-induced norm ratio — e.g. `γ·A` equilibrates to the same
    // matrix as `A` under InfNorm yet received a `γ`× larger floor.
    // Unlike the F-01 null-pivot override below, the static floor
    // applies regardless of `on_zero_pivot` (static pivoting is
    // independent of the abort-on-zero contract).
    let static_floor = params
        .static_pivot_threshold
        .filter(|t| *t > 0.0)
        .map(|t| t * scaled_infnorm)
        .filter(|f| f.is_finite() && *f > 0.0);

    // F-01: raise the BK kernel's null-pivot tolerance to the Wilkinson
    // backward-error floor `sqrt(n) · EPS · ‖A_scaled‖∞`. No-op under
    // `on_zero_pivot == Fail` (preserves the absolute-tolerance contract
    // for abort-on-zero callers) or when the floor is already at/below
    // the configured `null_pivot_tol`. See
    // `dev/research/f01-rankdef-underreporting.md`.
    let null_floor = if matches!(params.bk.on_zero_pivot, ZeroPivotAction::Fail) {
        None
    } else {
        let floor = null_pivot_floor(scaled_infnorm, n);
        (floor > params.bk.null_pivot_tol).then_some(floor)
    };

    if static_floor.is_none() && null_floor.is_none() {
        return None;
    }
    let mut local = params.clone();
    if let Some(sf) = static_floor {
        local.bk.static_pivot_floor = sf;
    }
    if let Some(floor) = null_floor {
        local.bk.null_pivot_tol = floor;
        local.bk.null_pivot_tol_2x2 = (floor * scaled_infnorm).max(floor * floor);
    }
    Some(local)
}

/// Permute a CSC matrix: compute the lower triangle of P·A·Pᵀ.
/// Permute the CSC matrix values into pivot order.
///
/// Returns the permuted matrix and, when `profile` is set, the wall
/// microseconds spent inside the `CscMatrix::from_triplets` rebuild
/// (Track B1: that rebuild re-sorts every entry and is the prime
/// suspect for the prologue cost). `profile == false` returns `0` for
/// the timing with no `Instant::now()` calls — the production path.
fn permute_csc_values(
    matrix: &CscMatrix,
    _perm: &[usize],
    perm_inv: &[usize],
    profile: bool,
) -> Result<(CscMatrix, u64), FeralError> {
    let n = matrix.n;

    // Collect permuted entries in lower triangle
    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(matrix.nnz());

    for old_j in 0..n {
        let new_j = perm_inv[old_j];
        for k in matrix.col_ptr[old_j]..matrix.col_ptr[old_j + 1] {
            let old_i = matrix.row_idx[k];
            let new_i = perm_inv[old_i];
            let val = matrix.values[k];

            // Store in lower triangle of permuted matrix
            if new_i >= new_j {
                triplets.push((new_i, new_j, val));
            } else {
                triplets.push((new_j, new_i, val));
            }
        }
    }

    let rows: Vec<usize> = triplets.iter().map(|t| t.0).collect();
    let cols: Vec<usize> = triplets.iter().map(|t| t.1).collect();
    let vals: Vec<f64> = triplets.iter().map(|t| t.2).collect();

    let t_ft = profile.then(Instant::now);
    let permuted = CscMatrix::from_triplets(n, &rows, &cols, &vals)?;
    let from_triplets_us = t_ft.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);
    Ok((permuted, from_triplets_us))
}

/// Permute a CSC matrix into pivot order with an optional structural
/// cache (issue #56 Lever A.2).
///
/// On the warm path (`pattern_reused_hint == true` and `cache` matches
/// the input `(n, nnz)`), the cached `permuted_col_ptr` /
/// `permuted_row_idx` / `value_map` are used to scatter values into a
/// fresh `values` vector in a single O(nnz) pass — skipping the
/// triplet construction and `CscMatrix::from_triplets` sort that
/// dominate `permute_csc_values` for IPM-like workloads with thousands
/// of factor calls on the same pattern.
///
/// On the cold path (cache empty, hint off, or `(n, nnz)` mismatch),
/// falls through to `permute_csc_values` and then rebuilds the cache
/// from the freshly-permuted matrix so the next warm call can take the
/// fast path. Cache rebuild failures (e.g., a row index unexpectedly
/// missing from the permuted column — should not occur with valid
/// inputs) clear the cache rather than propagating — the canonical
/// permuted matrix has already been produced.
fn permute_csc_values_with_cache(
    matrix: &CscMatrix,
    perm: &[usize],
    perm_inv: &[usize],
    profile: bool,
    pattern_reused_hint: bool,
    cache: &mut Option<PermuteCache>,
) -> Result<(CscMatrix, u64), FeralError> {
    let n = matrix.n;
    let nnz = matrix.nnz();

    // Warm path: cache hit — scatter values into the cached structure.
    // REG-1: the cache is keyed on the full input pattern AND `perm_inv`,
    // not just `(n, nnz)`. A different pattern sharing `(n, nnz)`, or the
    // same pattern under a changed permutation (e.g. AutoRace reselecting
    // an ordering after a symbolic-cache invalidation), must fall through
    // to the cold rebuild rather than scatter values through a stale
    // structure.
    if pattern_reused_hint {
        if let Some(c) = cache.as_ref() {
            if c.input_n == n
                && c.input_nnz == nnz
                && c.value_map.len() == nnz
                && c.input_col_ptr == matrix.col_ptr
                && c.input_row_idx == matrix.row_idx
                && c.input_perm_inv.as_slice() == perm_inv
            {
                let permuted_nnz = c.permuted_row_idx.len();
                let mut values = vec![0.0f64; permuted_nnz];
                for k in 0..nnz {
                    let dest = c.value_map[k];
                    debug_assert!(dest < permuted_nnz);
                    values[dest] += matrix.values[k];
                }
                let permuted = CscMatrix {
                    n,
                    col_ptr: c.permuted_col_ptr.clone(),
                    row_idx: c.permuted_row_idx.clone(),
                    values,
                };
                return Ok((permuted, 0));
            }
        }
    }

    // Cold path: canonical from-triplets rebuild.
    let (permuted, from_triplets_us) = permute_csc_values(matrix, perm, perm_inv, profile)?;

    // N7: only build the value-map cache when a warm reuse is actually
    // anticipated. `pattern_reused_hint == false` is the one-shot path —
    // the warm branch above is gated on the same hint, so a cache built
    // here would never be read. Building it anyway costs an O(nnz · log)
    // scan plus three O(nnz) vectors per factorization that are thrown
    // away. Skip the build for one-shot callers; the warm-reuse caller
    // (hint == true) still pays the build exactly once on its first,
    // cache-cold call and reaps it on every subsequent reuse.
    if pattern_reused_hint {
        // Refresh cache so the next warm call can take the fast path. Best
        // effort — on the (unexpected) row-lookup failure the cache is
        // cleared and subsequent calls take the cold path until the next
        // successful refresh.
        match build_permute_value_map(matrix, perm_inv, &permuted) {
            Ok(value_map) => {
                *cache = Some(PermuteCache {
                    input_n: n,
                    input_nnz: nnz,
                    input_col_ptr: matrix.col_ptr.clone(),
                    input_row_idx: matrix.row_idx.clone(),
                    input_perm_inv: perm_inv.to_vec(),
                    permuted_col_ptr: permuted.col_ptr.clone(),
                    permuted_row_idx: permuted.row_idx.clone(),
                    value_map,
                });
            }
            Err(_) => {
                *cache = None;
            }
        }
    } else {
        // N7/REG-1 defence in depth: a one-shot (hint == false) call must
        // not leave a stale cache from a previous pattern that a later
        // warm call could trust. Clear it (O(1)). The warm path also
        // validates the pattern+perm fingerprint, so this is belt-and-
        // suspenders, not the sole guard.
        *cache = None;
    }

    Ok((permuted, from_triplets_us))
}

/// Build the `value_map` for `PermuteCache` by replaying the
/// input-iteration logic of `permute_csc_values` and binary-searching
/// each lower-triangle target `(lr, lc)` against the freshly-built
/// permuted CSC's sorted column. One scan over input nnz, one binary
/// search per nonzero — O(nnz · log(max_col_nnz)).
fn build_permute_value_map(
    matrix: &CscMatrix,
    perm_inv: &[usize],
    permuted: &CscMatrix,
) -> Result<Vec<usize>, FeralError> {
    let n = matrix.n;
    let nnz = matrix.nnz();
    let mut value_map = vec![0usize; nnz];
    for old_j in 0..n {
        let new_j = perm_inv[old_j];
        #[allow(clippy::needless_range_loop)]
        for k in matrix.col_ptr[old_j]..matrix.col_ptr[old_j + 1] {
            let old_i = matrix.row_idx[k];
            let new_i = perm_inv[old_i];
            let (lr, lc) = if new_i >= new_j {
                (new_i, new_j)
            } else {
                (new_j, new_i)
            };
            let col_start = permuted.col_ptr[lc];
            let col_end = permuted.col_ptr[lc + 1];
            let dest_off = permuted.row_idx[col_start..col_end]
                .binary_search(&lr)
                .map_err(|_| {
                    FeralError::InvalidInput(format!(
                        "permute cache build: row {} not found in permuted column {} \
                         (col range {}..{}). Should not occur on valid CSC input.",
                        lr, lc, col_start, col_end
                    ))
                })?;
            value_map[k] = col_start + dest_off;
        }
    }
    Ok(value_map)
}

/// Build row indices for a frontal matrix.
///
/// Returns indices laid out as:
///
/// ```text
/// [own native cols (own_ncol)]
/// [delayed cols inherited from children (n_delayed_in)]
/// [trailing non-fully-summed rows, sorted]
/// ```
///
/// The first two regions together form the fully-summed block that
/// `factor_frontal` is allowed to pivot over. Delayed column global
/// indices come from each child's `ContribBlock.row_indices[..n_delayed]`
/// in child-iteration order; duplicates across children cannot arise
/// because each matrix column belongs to exactly one supernode.
/// Trailing rows are deduplicated against the fully-summed set so a
/// delayed column that also shows up as a pattern row of a parent
/// column (via the full symmetric pattern) does not appear twice.
fn build_row_indices(
    snode: &crate::symbolic::supernode::Supernode,
    full_pattern: &crate::sparse::csc::CscPattern,
    contrib_blocks: &[Option<ContribBlock>],
    build_delayed: &mut Vec<usize>,
    build_trailing: &mut Vec<usize>,
    build_seen: &mut Vec<bool>,
) -> Vec<usize> {
    let own_ncol = snode.ncol();
    let first_col = snode.first_col;
    let n = full_pattern.n;

    // Grow `build_seen` on demand; caller maintains the all-`false`
    // invariant outside this function.
    if build_seen.len() < n {
        build_seen.resize(n, false);
    }

    // Collect delayed columns from each child, preserving child-iteration
    // order. Bit-for-bit equivalent to the old `Vec::new() + extend` path;
    // the Vec is pooled across supernodes so only its capacity growth
    // allocates.
    build_delayed.clear();
    for &child_idx in &snode.children {
        if let Some(contrib) = &contrib_blocks[child_idx] {
            build_delayed.extend_from_slice(&contrib.row_indices[..contrib.n_delayed]);
        }
    }

    // Mark own native + delayed columns as "fully summed" in the seen
    // bitmap so the trailing scan skips them. Duplicates across children
    // cannot arise (each matrix column belongs to exactly one supernode).
    for seen in build_seen.iter_mut().skip(first_col).take(own_ncol) {
        *seen = true;
    }
    for &c in build_delayed.iter() {
        build_seen[c] = true;
    }

    // Trailing row set via seen-based dedup. Same role as the previous
    // BTreeSet<usize> but with O(1) insert and a single O(m log m) sort
    // at the end to match the BTreeSet iteration order that callers
    // (and the parity tests) depend on.
    //
    // Filter `r < first_col + own_ncol` (the supernode's own col range
    // upper bound): `full_pattern` is the fully-symmetrized A pattern
    // (csc.rs:186), so iterating column j gives both lower-tri (r > j,
    // legitimate trailing) and upper-tri (r < j, columns already
    // eliminated by ancestors of those rows) entries. Upper-tri rows are
    // not legitimate trailing rows in a multifrontal frontal — they
    // would be padded with zeros and inflate factor_nnz without
    // contributing structurally. The same filter on the children's
    // contrib loop is defensive: a clean child cannot produce trailing
    // rows < parent.first_col, but the filter guards against historical
    // contribs built by an older buggy build_row_indices.
    let own_last = first_col + own_ncol;
    build_trailing.clear();
    for j in first_col..first_col + own_ncol {
        for k in full_pattern.col_ptr[j]..full_pattern.col_ptr[j + 1] {
            let r = full_pattern.row_idx[k];
            if r < own_last {
                continue;
            }
            if !build_seen[r] {
                build_seen[r] = true;
                build_trailing.push(r);
            }
        }
    }
    for &child_idx in &snode.children {
        if let Some(contrib) = &contrib_blocks[child_idx] {
            for &r in &contrib.row_indices[contrib.n_delayed..] {
                if r < own_last {
                    continue;
                }
                if !build_seen[r] {
                    build_seen[r] = true;
                    build_trailing.push(r);
                }
            }
        }
    }
    build_trailing.sort_unstable();

    let total = own_ncol + build_delayed.len() + build_trailing.len();
    let mut result = Vec::with_capacity(total);
    result.extend(first_col..first_col + own_ncol);
    result.extend_from_slice(build_delayed);
    result.extend_from_slice(build_trailing);

    // Restore the all-`false` invariant on `build_seen` by clearing
    // only the entries we touched. Cheaper than a full `resize` and
    // keeps the invariant auditable.
    for seen in build_seen.iter_mut().skip(first_col).take(own_ncol) {
        *seen = false;
    }
    for &c in build_delayed.iter() {
        build_seen[c] = false;
    }
    for &r in build_trailing.iter() {
        build_seen[r] = false;
    }

    result
}

/// Contribution block from a child supernode.
///
/// Under delayed pivoting the top-left `n_delayed × n_delayed` block
/// holds the child's un-eliminated fully-summed columns (which must
/// re-enter pivot search at the parent as additional fully-summed
/// columns), and the bottom-right `(dim - n_delayed) × (dim - n_delayed)`
/// block is the classic Schur complement over the non-fully-summed
/// trailing rows. The cross block (rows = trailing, cols = delayed)
/// carries the mixed interactions. `row_indices[..n_delayed]` are
/// the global row indices of the delayed columns in the parent's
/// numbering; `row_indices[n_delayed..]` are the trailing rows.
#[derive(Debug)]
struct ContribBlock {
    /// Row indices of the contribution block (global).
    /// First `n_delayed` entries are delayed fully-summed columns;
    /// the remainder are the trailing non-fully-summed rows (sorted).
    row_indices: Vec<usize>,
    /// Dense symmetric matrix data (lower triangle, column-major).
    /// Dimension: row_indices.len() × row_indices.len()
    data: Vec<f64>,
    /// Dimension of the contribution block.
    dim: usize,
    /// Number of delayed fully-summed columns carried in this block
    /// (top-left `n_delayed × n_delayed` sub-matrix). Zero for nodes
    /// whose BK sweep succeeded on every attempted column. Consumed
    /// by the parent's `build_row_indices` and the Step 5 assembly
    /// which places these columns in the parent's fully-summed region.
    n_delayed: usize,
}

/// Extend-add: assemble a child's contribution block into the parent frontal.
///
/// Issue #13 Phase B: bypasses `SymmetricMatrix::set`/`get` and writes
/// directly into `frontal.data`. The lower-triangle column-major linear
/// index for cell `(row, col)` with `row >= col` is `col * n + row`; the
/// `parent_i >= parent_j` test below canonicalises which of the two
/// indices is the row. This removes the redundant `i >= j` branch that
/// `set`/`get` perform on every cell and eliminates one read+write call
/// frame per cell. With ~38 children per medium front each contributing
/// up to `cdim*(cdim+1)/2` cells, this trims a measurable per-front cost.
fn extend_add(contrib: &ContribBlock, parent_row_map: &[usize], frontal: &mut SymmetricMatrix) {
    let cdim = contrib.dim;
    let fn_ = frontal.n;
    let f_data: &mut [f64] = frontal.data.as_mut_slice();
    for cj in 0..cdim {
        let parent_j = parent_row_map[contrib.row_indices[cj]];
        if parent_j == usize::MAX {
            continue;
        }
        for ci in cj..cdim {
            let parent_i = parent_row_map[contrib.row_indices[ci]];
            if parent_i == usize::MAX {
                continue;
            }
            let val = contrib.data[cj * cdim + ci];
            if val == 0.0 {
                continue;
            }
            // Canonicalise to lower triangle: ensure row >= col.
            let (row, col) = if parent_i >= parent_j {
                (parent_i, parent_j)
            } else {
                (parent_j, parent_i)
            };
            f_data[col * fn_ + row] += val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::factor::ZeroPivotAction;
    use crate::symbolic::{symbolic_factorize, SupernodeParams};

    fn make_params() -> NumericParams {
        NumericParams::with_bk(BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            ..BunchKaufmanParams::default()
        })
    }

    #[test]
    fn test_factorize_diagonal() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let (factors, inertia) = factorize_multifrontal(&m, &sym, &make_params()).unwrap();

        assert_eq!(inertia.positive, 3);
        assert_eq!(inertia.negative, 0);
        assert_eq!(inertia.zero, 0);
        assert_eq!(factors.n, 3);
    }

    #[test]
    fn test_summary_one_liner() {
        // Tridiagonal n=32 — large enough to bypass the dense
        // fast-path (N_TINY=16, N_MAX=128 with density gate) so the
        // multifrontal path runs and the resolved-field mirroring
        // from `SymbolicFactorization` is exercised.
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
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let (factors, inertia) = factorize_multifrontal(&m, &sym, &make_params()).unwrap();

        let s = factors.summary();
        assert!(s.contains("ord="), "summary missing ord field: {}", s);
        assert!(s.contains("amalg="), "summary missing amalg field: {}", s);
        assert!(
            s.contains("preproc="),
            "summary missing preproc field: {}",
            s
        );
        assert!(
            s.contains("scaling="),
            "summary missing scaling field: {}",
            s
        );
        let nnz_l = factors.factor_nnz();
        assert!(nnz_l > 0, "tridiagonal factor_nnz must be > 0");
        assert!(
            s.contains(&format!("nnz_L={}", nnz_l)),
            "summary nnz_L mismatch: got {}, want nnz_L={}",
            s,
            nnz_l
        );
        let expected = format!(
            "inertia=({},{},{})",
            inertia.positive, inertia.negative, inertia.zero
        );
        assert!(
            s.contains(&expected),
            "summary inertia mismatch: got {}, want substring {}",
            s,
            expected
        );
        // `Auto` is a dispatch sentinel, never a resolved value.
        assert_ne!(
            factors.resolved_amalgamation,
            crate::symbolic::AmalgamationStrategy::Auto
        );
        assert_ne!(
            factors.resolved_method,
            crate::symbolic::OrderingMethod::Auto
        );
        assert_ne!(
            factors.resolved_preprocess,
            crate::symbolic::OrderingPreprocess::Auto
        );
        // Mirror invariant: the numeric factors agree with the
        // symbolic factorization on the resolved strategies.
        assert_eq!(factors.resolved_method, sym.resolved_method);
        assert_eq!(factors.resolved_amalgamation, sym.resolved_amalgamation);
        assert_eq!(factors.resolved_preprocess, sym.resolved_preprocess);
    }

    #[test]
    fn test_factorize_tridiagonal() {
        // [2 -1  0]
        // [-1 2 -1]
        // [0 -1  2]
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[2.0, -1.0, 2.0, -1.0, 2.0],
        )
        .unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let (factors, inertia) = factorize_multifrontal(&m, &sym, &make_params()).unwrap();

        // This matrix is SPD
        assert_eq!(inertia.positive, 3);
        assert_eq!(inertia.negative, 0);
        assert_eq!(inertia.zero, 0);
        assert_eq!(factors.n, 3);
    }

    #[test]
    fn test_factorize_matches_dense() {
        // Factor a small matrix with both dense and sparse, compare inertia
        // [2 -1  0]
        // [-1 3 -1]
        // [0 -1  4]
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[2.0, -1.0, 3.0, -1.0, 4.0],
        )
        .unwrap();

        // Dense factorization
        let dense_mat = m.to_dense();
        let params = make_params();
        let (_, dense_inertia) = factor(&dense_mat, &params.bk).unwrap();

        // Sparse factorization
        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let (_, sparse_inertia) = factorize_multifrontal(&m, &sym, &params).unwrap();

        assert_eq!(sparse_inertia, dense_inertia);
    }

    #[test]
    fn test_factorize_kkt() {
        // KKT matrix: [H A^T; A -delta*I]
        // H = [[2,0],[0,3]], A = [1,1], delta = 1e-8
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 2, 2, 2],
            &[0, 1, 0, 1, 2],
            &[2.0, 3.0, 1.0, 1.0, -1e-8],
        )
        .unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let params = make_params();
        let (_, inertia) = factorize_multifrontal(&m, &sym, &params).unwrap();

        // Should have 2 positive (H block), 1 negative (constraint block)
        assert_eq!(inertia.positive, 2);
        assert_eq!(inertia.negative, 1);
        assert_eq!(inertia.zero, 0);
    }

    #[test]
    fn test_factorize_indefinite() {
        // Indefinite: [[1,2],[2,1]]
        let m = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[1.0, 2.0, 1.0]).unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let params = make_params();
        let (_, inertia) = factorize_multifrontal(&m, &sym, &params).unwrap();

        // Eigenvalues: 3, -1 → 1 positive, 1 negative
        assert_eq!(inertia.positive, 1);
        assert_eq!(inertia.negative, 1);
        assert_eq!(inertia.zero, 0);
    }

    /// Structural goal of the β refactor: a single SymbolicFactorization
    /// is reusable across NumericParams that select different scaling
    /// strategies. The same `sym` factors twice — once with InfNorm,
    /// once with Identity — and both calls succeed and produce the
    /// expected inertia (1 positive, 2 negative for a saddle-point
    /// system with one constraint).
    #[test]
    fn factorize_multifrontal_with_two_strategies_on_one_symbolic() {
        use crate::scaling::ScalingStrategy;

        // Saddle-point KKT: [[2 0 -1], [0 2 -1], [-1 -1 0]].
        // Inertia: H = 2I_2 contributes 2 positive; constraint Schur
        // is -[-1 -1]·(I/2)·[-1 -1]^T = -1, so 1 negative.
        let m = CscMatrix::from_triplets(
            3,
            &[0, 2, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[2.0, -1.0, 2.0, -1.0, 0.0],
        )
        .unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();

        let infnorm = NumericParams {
            bk: BunchKaufmanParams {
                on_zero_pivot: ZeroPivotAction::ForceAccept,
                ..BunchKaufmanParams::default()
            },
            scaling: ScalingStrategy::InfNorm,
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
            pattern_reused_hint: false,
        };
        let identity = NumericParams {
            bk: infnorm.bk.clone(),
            scaling: ScalingStrategy::Identity,
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
            pattern_reused_hint: false,
        };

        let (_, i_inf) = factorize_multifrontal(&m, &sym, &infnorm).unwrap();
        let (_, i_id) = factorize_multifrontal(&m, &sym, &identity).unwrap();

        assert_eq!(i_inf.positive, 2);
        assert_eq!(i_inf.negative, 1);
        assert_eq!(i_id.positive, 2);
        assert_eq!(i_id.negative, 1);
    }

    /// Dense fast-path: `ScalingStrategy::Auto` short-circuits to
    /// the dense-native KR. The factor must be bit-exact with an
    /// explicit `ScalingStrategy::InfNorm` call on the same input.
    /// Guards against accidental re-introduction of Auto's MC64
    /// branch in the fast path.
    #[test]
    fn dense_fast_factor_auto_matches_explicit_infnorm_bitwise() {
        use crate::scaling::ScalingStrategy;
        // n=24 dense diagonally-dominant lower-triangular pattern;
        // fires the D.3 gate (density well above 0.25, n <= 128).
        let n = 24usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(10.0 * (j as f64 + 1.0));
            for i in (j + 1)..n {
                rows.push(i);
                cols.push(j);
                vals.push(1.0 + 0.1 * (i - j) as f64);
            }
        }
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        assert!(
            should_use_dense_fast_path(m.n, m.row_idx.len()),
            "test setup: dense fast-path gate must fire"
        );

        let auto = NumericParams {
            bk: BunchKaufmanParams {
                on_zero_pivot: ZeroPivotAction::ForceAccept,
                ..BunchKaufmanParams::default()
            },
            scaling: ScalingStrategy::Auto,
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
            pattern_reused_hint: false,
        };
        let infnorm = NumericParams {
            scaling: ScalingStrategy::InfNorm,
            ..auto.clone()
        };

        let (f_auto, i_auto) = dense_fast_factor(&m, &auto).unwrap();
        let (f_inf, i_inf) = dense_fast_factor(&m, &infnorm).unwrap();

        assert_eq!(i_auto, i_inf, "inertia diverged under Auto vs InfNorm");

        // Bit-equality of the per-node `D` values is the strongest
        // signature: the BK kernel ran on identical scaled inputs.
        assert_eq!(
            f_auto.node_factors.len(),
            f_inf.node_factors.len(),
            "node count diverged",
        );
        for (k, (na, ni)) in f_auto
            .node_factors
            .iter()
            .zip(f_inf.node_factors.iter())
            .enumerate()
        {
            assert_eq!(
                na.frontal_factors.d_diag.len(),
                ni.frontal_factors.d_diag.len(),
                "node {} D-diag length diverged",
                k,
            );
            for (j, (da, di)) in na
                .frontal_factors
                .d_diag
                .iter()
                .zip(ni.frontal_factors.d_diag.iter())
                .enumerate()
            {
                assert_eq!(
                    da.to_bits(),
                    di.to_bits(),
                    "node {} D_diag[{}] bits differ: auto={} infnorm={}",
                    k,
                    j,
                    da,
                    di,
                );
            }
        }
    }

    /// 6×6 KKT for F3.2b Schur extraction tests. Same shape as the
    /// hand-built oracle in src/symbolic tests:
    /// - Non-Schur block diag(1,2,3,4) at positions 0..4
    /// - Schur block (positions 4,5):
    ///   [1.5, 0.2; 0.2, 2.5]
    /// - Coupling A_FS:
    ///   col 4 has rows {0:0.5, 2:0.7}; col 5 has rows {1:0.3, 3:0.9}
    fn small_kkt_6x6_for_schur() -> CscMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..4 {
            rows.push(i);
            cols.push(i);
            vals.push((i + 1) as f64);
        }
        rows.push(4);
        cols.push(0);
        vals.push(0.5);
        rows.push(4);
        cols.push(2);
        vals.push(0.7);
        rows.push(5);
        cols.push(1);
        vals.push(0.3);
        rows.push(5);
        cols.push(3);
        vals.push(0.9);
        rows.push(4);
        cols.push(4);
        vals.push(1.5);
        rows.push(5);
        cols.push(4);
        vals.push(0.2);
        rows.push(5);
        cols.push(5);
        vals.push(2.5);
        CscMatrix::from_triplets(6, &rows, &cols, &vals).unwrap()
    }

    /// Hand-computed Schur complement S = A_SS − A_FS^T A_FF^{-1} A_FS:
    ///   A_FF = diag(1, 2, 3, 4) ⇒ A_FF^{-1} = diag(1, 0.5, 1/3, 0.25)
    ///   A_FS^T A_FF^{-1} A_FS:
    ///     (4,4) = 0.5²·1 + 0.7²·(1/3) = 0.25 + 0.49/3
    ///     (4,5) = 0  (no shared row between col 4 and col 5)
    ///     (5,5) = 0.3²·0.5 + 0.9²·0.25 = 0.045 + 0.2025
    ///   S = [[1.5 − (0.25 + 0.49/3), 0.2],
    ///        [0.2, 2.5 − (0.045 + 0.2025)]]
    fn hand_computed_schur_2x2() -> [[f64; 2]; 2] {
        let s00 = 1.5 - (0.25 + 0.49 / 3.0);
        let s11 = 2.5 - (0.045 + 0.2025);
        let s01 = 0.2;
        [[s00, s01], [s01, s11]]
    }

    #[test]
    fn schur_block_matches_hand_computed_for_small_kkt() {
        let m = small_kkt_6x6_for_schur();
        let params = crate::symbolic::SupernodeParams::default();
        // ScalingStrategy::Identity is required because compute_scaling
        // is called against the original matrix in the Schur driver,
        // and the hand-computed S assumes no scaling. Default Auto on
        // a 6-row matrix would route to InfNorm and rescale entries.
        let sym = crate::symbolic::symbolic_factorize_with_schur(&m, &params, &[4, 5]).unwrap();
        let nparams = NumericParams {
            scaling: crate::scaling::ScalingStrategy::Identity,
            ..NumericParams::default()
        };
        let (_factors, _inertia, schur) =
            factorize_multifrontal_with_schur(&m, &sym, &nparams).unwrap();
        assert_eq!(schur.dim, 2);
        let expected = hand_computed_schur_2x2();
        let tol = 1e-12;
        for i in 0..2 {
            for j in 0..2 {
                let got = schur.get(i, j);
                let want = expected[i][j];
                assert!(
                    (got - want).abs() < tol,
                    "S({},{}) got {} want {} diff {}",
                    i,
                    j,
                    got,
                    want,
                    (got - want).abs()
                );
            }
        }
    }

    #[test]
    fn schur_block_full_square_storage() {
        // S(i,j) == S(j,i) at every entry — full-square storage with
        // mirror to upper triangle.
        let m = small_kkt_6x6_for_schur();
        let params = crate::symbolic::SupernodeParams::default();
        let sym = crate::symbolic::symbolic_factorize_with_schur(&m, &params, &[4, 5]).unwrap();
        let nparams = NumericParams {
            scaling: crate::scaling::ScalingStrategy::Identity,
            ..NumericParams::default()
        };
        let (_, _, schur) = factorize_multifrontal_with_schur(&m, &sym, &nparams).unwrap();
        for i in 0..schur.dim {
            for j in 0..schur.dim {
                assert!((schur.get(i, j) - schur.get(j, i)).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn schur_user_order_preserved_when_reversed() {
        // Reverse user order: schur_indices = [5, 4]. Then S
        // reported has rows/cols permuted so that out(0,0) corresponds
        // to original index 5 (= S_hand(1,1)).
        let m = small_kkt_6x6_for_schur();
        let params = crate::symbolic::SupernodeParams::default();
        let sym = crate::symbolic::symbolic_factorize_with_schur(&m, &params, &[5, 4]).unwrap();
        let nparams = NumericParams {
            scaling: crate::scaling::ScalingStrategy::Identity,
            ..NumericParams::default()
        };
        let (_, _, schur) = factorize_multifrontal_with_schur(&m, &sym, &nparams).unwrap();
        let hand = hand_computed_schur_2x2();
        // Mapping: out(i,j) = hand(map(i), map(j)) with map = [1, 0]
        let map = [1usize, 0usize];
        let tol = 1e-12;
        for i in 0..2 {
            for j in 0..2 {
                let got = schur.get(i, j);
                let want = hand[map[i]][map[j]];
                assert!(
                    (got - want).abs() < tol,
                    "reversed S({},{}) got {} want {}",
                    i,
                    j,
                    got,
                    want
                );
            }
        }
    }

    #[test]
    fn schur_rejects_symbolic_without_schur_tail() {
        let m = small_kkt_6x6_for_schur();
        let params = crate::symbolic::SupernodeParams::default();
        let sym = crate::symbolic::symbolic_factorize(&m, &params).unwrap(); // no Schur
        assert_eq!(sym.is_schur_tail, None);
        let nparams = NumericParams::default();
        let r = factorize_multifrontal_with_schur(&m, &sym, &nparams);
        assert!(matches!(r, Err(FeralError::InvalidInput(_))));
    }

    /// F3.2b multi-supernode Schur tail. Builds a problem where the
    /// pre-merge symbolic phase produces multiple Schur-bearing
    /// supernodes (size > nemin and with structurally distinct row
    /// patterns); after the symbolic merge step (see
    /// `merge_schur_tail_supernodes` in `symbolic/mod.rs`), the numeric
    /// driver must accept the symbolic and produce the correct Schur
    /// block. Verified against an oracle that solves
    /// `A_FF * X = A_FS` densely and computes
    /// `S = A_SS - A_FS^T * X`.
    #[test]
    fn schur_multi_supernode_tail_matches_oracle() {
        // Two coupled subblocks A and B with their own dense Schur
        // tail, plus a tridiagonal cross-link across the entire Schur
        // set so the etree has a single Schur root (forest Schur is
        // unsupported per F3.2a).
        let half = 25usize;
        let k_each = 40usize;
        let n = 2 * half + 2 * k_each;

        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0 + i as f64);
        }
        for i in 0..half {
            for s in 0..k_each {
                let j = 2 * half + s;
                rows.push(j);
                cols.push(i);
                vals.push(0.1);
            }
        }
        for i in half..2 * half {
            for s in 0..k_each {
                let j = 2 * half + k_each + s;
                rows.push(j);
                cols.push(i);
                vals.push(0.1);
            }
        }
        for s in 0..k_each {
            for t in 0..s {
                rows.push(2 * half + s);
                cols.push(2 * half + t);
                vals.push(0.05);
                rows.push(2 * half + k_each + s);
                cols.push(2 * half + k_each + t);
                vals.push(0.05);
            }
        }
        for s in (2 * half + 1)..n {
            rows.push(s);
            cols.push(s - 1);
            vals.push(0.03);
        }
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let schur: Vec<usize> = (2 * half..n).collect();
        let n_schur = schur.len();

        let params = crate::symbolic::SupernodeParams::default();
        let sym = crate::symbolic::symbolic_factorize_with_schur(&m, &params, &schur).unwrap();
        let nparams = NumericParams {
            scaling: crate::scaling::ScalingStrategy::Identity,
            ..NumericParams::default()
        };
        let (_, _, schur_block) = factorize_multifrontal_with_schur(&m, &sym, &nparams).unwrap();
        assert_eq!(schur_block.dim, n_schur);

        // Build A as dense and the f-index list.
        let mut is_schur = vec![false; n];
        for &i in &schur {
            is_schur[i] = true;
        }
        let f_indices: Vec<usize> = (0..n).filter(|i| !is_schur[*i]).collect();
        let nf = f_indices.len();
        let mut f_inv = vec![usize::MAX; n];
        for (k, &i) in f_indices.iter().enumerate() {
            f_inv[i] = k;
        }
        let mut a = vec![0.0f64; n * n];
        for j in 0..n {
            for k in m.col_ptr[j]..m.col_ptr[j + 1] {
                let i = m.row_idx[k];
                a[j * n + i] = m.values[k];
                if i != j {
                    a[i * n + j] = m.values[k];
                }
            }
        }

        // Factor A_FF (sparse).
        let mut tr = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            if is_schur[j] {
                continue;
            }
            for k in m.col_ptr[j]..m.col_ptr[j + 1] {
                let i = m.row_idx[k];
                if !is_schur[i] {
                    tr.0.push(f_inv[i]);
                    tr.1.push(f_inv[j]);
                    tr.2.push(m.values[k]);
                }
            }
        }
        let a_ff = CscMatrix::from_triplets(nf, &tr.0, &tr.1, &tr.2).unwrap();
        let sym_ff = crate::symbolic::symbolic_factorize(&a_ff, &params).unwrap();
        let (factors_ff, _) = factorize_multifrontal(&a_ff, &sym_ff, &nparams).unwrap();

        // S = A_SS - A_FS^T A_FF^{-1} A_FS via column-by-column solve.
        let mut s_ref = vec![0.0f64; n_schur * n_schur];
        for (si, &i) in schur.iter().enumerate() {
            for (sj, &j) in schur.iter().enumerate() {
                s_ref[sj * n_schur + si] = a[j * n + i];
            }
        }
        for (sj, &j) in schur.iter().enumerate() {
            let mut rhs = vec![0.0f64; nf];
            for &fi in &f_indices {
                rhs[f_inv[fi]] = a[j * n + fi];
            }
            let x = crate::numeric::solve::solve_sparse(&factors_ff, &rhs).unwrap();
            for (si, &i) in schur.iter().enumerate() {
                let mut acc = 0.0;
                for &fi in &f_indices {
                    acc += a[i * n + fi] * x[f_inv[fi]];
                }
                s_ref[sj * n_schur + si] -= acc;
            }
        }

        let mut max_rel = 0.0f64;
        for sj in 0..n_schur {
            for si in 0..n_schur {
                let want = s_ref[sj * n_schur + si];
                let got = schur_block.get(si, sj);
                let denom = want.abs().max(1e-14);
                let rel = (got - want).abs() / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        assert!(
            max_rel < 1e-10,
            "Schur block max relative error {} exceeds 1e-10",
            max_rel
        );
    }

    /// F3.4 — `SchurBlock::solve` factors the dense Schur block and
    /// solves `S · x = rhs`. Verified against an explicit 3×3
    /// symmetric indefinite Schur block with a hand-picked rhs.
    #[test]
    fn schur_block_solve_small_explicit() {
        // S = [[ 4, -1,  0],
        //      [-1,  3, -1],
        //      [ 0, -1,  2]]   (SPD, used here just because hand-checked).
        // x_true = [1, 2, 3]
        // S · x = [4·1 + -1·2,  -1·1 + 3·2 + -1·3,  -1·2 + 2·3]
        //       = [2, 2, 4]
        let dim = 3;
        let s = vec![
            4.0, -1.0, 0.0, // col 0
            -1.0, 3.0, -1.0, // col 1
            0.0, -1.0, 2.0, // col 2
        ];
        let block = SchurBlock {
            dim,
            data: s.clone(),
        };
        let rhs = vec![2.0, 2.0, 4.0];
        let x = block.solve(&rhs).expect("solve must succeed");
        let want = [1.0, 2.0, 3.0];
        for i in 0..dim {
            assert!(
                (x[i] - want[i]).abs() < 1e-12,
                "x[{i}] = {} != {} (rel diff {:.3e})",
                x[i],
                want[i],
                (x[i] - want[i]).abs()
            );
        }

        // Round-trip: symv(S, x) == rhs.
        let mut rhs_check = vec![0.0; dim];
        block.symv(&x, &mut rhs_check).expect("symv");
        for i in 0..dim {
            assert!((rhs_check[i] - rhs[i]).abs() < 1e-12);
        }
    }

    /// F3.4 — End-to-end: factorize a small KKT with a Schur tail,
    /// pick a target `x_S`, compute `rhs_S = S · x_S` via `symv`, and
    /// recover `x_S` via `SchurBlock::solve`. Exercises the dim and
    /// rhs validation paths.
    #[test]
    fn schur_block_solve_roundtrip_after_factorize() {
        // Reuse the 4-row 4-col KKT from the hand-computed test above.
        // A = [[ 2, 0, 0, 1],
        //      [ 0, 3, 0, 1],
        //      [ 0, 0, 4, 1],
        //      [ 1, 1, 1, 0]]   (last col/row is the Schur slot).
        let entries = [
            (0, 0, 2.0),
            (1, 1, 3.0),
            (2, 2, 4.0),
            (3, 0, 1.0),
            (3, 1, 1.0),
            (3, 2, 1.0),
        ];
        let n = 4;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for &(r, c, v) in &entries {
            rows.push(r);
            cols.push(c);
            vals.push(v);
        }
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let schur_indices = vec![3];
        let snode = crate::symbolic::SupernodeParams::default();
        let sym =
            crate::symbolic::symbolic_factorize_with_schur(&m, &snode, &schur_indices).unwrap();
        let nparams = NumericParams {
            scaling: crate::scaling::ScalingStrategy::Identity,
            ..NumericParams::default()
        };
        let (_, _, schur_block) = factorize_multifrontal_with_schur(&m, &sym, &nparams).unwrap();

        // Hand: S = A_SS - A_SF · A_FF^{-1} · A_FS
        //         = 0 - [1 1 1] · diag(1/2, 1/3, 1/4) · [1; 1; 1]
        //         = -(1/2 + 1/3 + 1/4) = -13/12
        let want_s = -(1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0);
        assert!((schur_block.get(0, 0) - want_s).abs() < 1e-12);

        // Pick x_S, build rhs_S, recover x_S.
        let x_target = vec![1.5_f64];
        let mut rhs_s = vec![0.0; 1];
        schur_block.symv(&x_target, &mut rhs_s).unwrap();
        let x_solved = schur_block.solve(&rhs_s).unwrap();
        assert!((x_solved[0] - x_target[0]).abs() < 1e-12);
    }

    /// F3.4 — Dimension mismatch on rhs is reported, not silently
    /// truncated.
    #[test]
    fn schur_block_solve_dim_mismatch() {
        let block = SchurBlock {
            dim: 2,
            data: vec![1.0, 0.0, 0.0, 1.0],
        };
        let rhs = vec![1.0, 2.0, 3.0];
        let r = block.solve(&rhs);
        assert!(matches!(r, Err(FeralError::DimensionMismatch { .. })));
    }

    /// Issue #5 reproducer — sweep δ_w on the first 90 diagonal
    /// entries (the (1,1) block) of MSS1_0000 and observe inertia.
    /// Mirrors `ripopt::feral_direct.rs`'s configuration exactly.
    /// See `dev/research/issue-5-mss1-inertia-monotonicity.md`.
    fn issue5_mss1_inertia_sweep_with(
        zero_tol: f64,
        pivot_threshold: f64,
    ) -> Option<Vec<(f64, Inertia)>> {
        let path = std::path::Path::new("data/matrices/kkt/MSS1/MSS1_0000.mtx");
        let mtx = match crate::io::mtx::read_mtx(path) {
            Ok(m) => m,
            Err(_) => return None, // fixture not present — skip
        };
        let n = mtx.n;
        // The (1,1) block is the first n_x rows/cols. MSS1: n_x = 90,
        // n_eq = 73, total 163 (per .json sidecar).
        let n_x = 90usize;
        assert_eq!(n, 163, "MSS1_0000 expected n=163");

        // ripopt's exact configuration from feral_direct.rs:
        let mut bk = BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            zero_tol,
            zero_tol_2x2: zero_tol * zero_tol,
            ..BunchKaufmanParams::default()
        };
        bk.pivot_threshold = pivot_threshold;

        let params = NumericParams {
            bk,
            scaling: ScalingStrategy::Identity,
            small_leaf: SmallLeafBatch::default(),
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
            pattern_reused_hint: false,
        };

        let deltas = [0.0, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12];

        let mut results = Vec::with_capacity(deltas.len());
        for &dw in &deltas {
            // Add δ_w to the first n_x diagonal entries. Build via
            // triplets so we don't have to deal with whether the
            // diagonal entries already exist (from_triplets sums
            // duplicates).
            let mut rows: Vec<usize> = Vec::with_capacity(mtx.entries.len() + n_x);
            let mut cols: Vec<usize> = Vec::with_capacity(mtx.entries.len() + n_x);
            let mut vals: Vec<f64> = Vec::with_capacity(mtx.entries.len() + n_x);
            for &(r, c, v) in &mtx.entries {
                // The corpus matrix already has -1e-8 baked into the
                // (2,2) block (Ipopt's default static regularization).
                // The reporter's `dc=0` trace is over the *unperturbed*
                // KKT, so strip it here. Add δ_c=0 (i.e. nothing) back.
                if r >= n_x && c >= n_x && r == c {
                    continue;
                }
                rows.push(r);
                cols.push(c);
                vals.push(v);
            }
            for i in 0..n_x {
                rows.push(i);
                cols.push(i);
                vals.push(dw);
            }
            let csc = CscMatrix::from_triplets(n, &rows, &cols, &vals)
                .expect("MSS1_0000 + δ_w·I CSC build");

            let sym = symbolic_factorize(&csc, &SupernodeParams::default())
                .expect("symbolic factorize MSS1_0000");
            let (_factors, inertia) = factorize_multifrontal(&csc, &sym, &params)
                .expect("numeric factorize MSS1_0000 + δ_w·I");
            results.push((dw, inertia));
        }
        Some(results)
    }

    /// Issue #5 — regression guard for the BK 1×1/2×2 boundary
    /// instability on MSS1_0000.
    ///
    /// **Disposition (2026-05-10): closed on the feral side.** Per
    /// `dev/research/issue-5-mss1-inertia-monotonicity.md` §9, the
    /// wandering is caused by BK pivot-ordering ambiguity on a
    /// structurally rank-deficient J. Neither the in-kernel
    /// magnitude-floor levers (`zero_tol`, `pivot_threshold`) nor
    /// the canonical Fortran solvers (MUMPS, MA57) handle this in
    /// the linear-solver layer — they keep 2×2 atomic and rely on
    /// the IPM driver to escalate δ_w / δ_c. The recommended fix
    /// is upstream: ripopt should add a `PerturbForSingularity`
    /// δ_c bump to its inertia-correction loop.
    ///
    /// This test stays as a regression guard: it asserts the
    /// wandering pattern persists under ripopt's exact
    /// configuration. If a future change happens to flip it to
    /// monotone non-decreasing, that's the signal to revisit the
    /// disposition — flip the assertion to
    /// `assert!(positives.windows(2).all(|w| w[1] >= w[0]))` and
    /// update the research note.
    ///
    /// Skipped silently if `data/matrices/kkt/MSS1/MSS1_0000.mtx`
    /// is not present in the working tree.
    fn issue5_mss1_inertia_sweep() -> Option<Vec<(f64, Inertia)>> {
        // Default: ripopt's exact pre-issue-#2 configuration —
        // `zero_tol = 1e-10`, `pivot_threshold = 0.0`.
        issue5_mss1_inertia_sweep_with(1e-10, 0.0)
    }

    #[test]
    fn issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep() {
        let Some(results) = issue5_mss1_inertia_sweep() else {
            return;
        };

        // Print the trace on test failure — this is the diagnostic
        // payload the issue is about. Captured by `cargo test --
        // --nocapture`.
        for (dw, inertia) in &results {
            eprintln!(
                "δ_w = {:>8.0e}  →  inertia(+{:3}, -{:3}, 0:{:>3})",
                dw, inertia.positive, inertia.negative, inertia.zero
            );
        }

        // After stripping the (2,2)-block -1e-8 (Ipopt's static
        // regularization), the constraint Jacobian J has structural
        // rank deficiency: rank(J) ≈ 45, m-rank(J) ≈ 28. The Schur
        // complement of [[(1+δ_w)·I, J^T], [J, 0]] is then
        // S = -(1+δ_w)^-1·J·J^T which has rank 45 for any finite
        // δ_w. So the *correct* inertia at every δ_w in the sweep
        // is (n_x + rank(J), rank(J), m - rank(J)) = (135, 45, 28),
        // independent of δ_w. The PDPerturbationHandler in ripopt
        // assumes inertia is monotone non-decreasing in δ_w and
        // ramps it expecting to see n+ → n_x as δ_w grows. That
        // assumption holds for non-degenerate KKT but fails here —
        // and the wandering observed in the BK 1×1/2×2 boundary
        // makes the failure dramatic.
        //
        // Lock in the symptom: positive count is *non-monotone*
        // across the sweep. Once Option B (norm-relative pivot
        // floor / SEUIL analog) lands, this assertion flips to
        // `assert!(positives are monotone non-decreasing)` per
        // dev/research/issue-5-mss1-inertia-monotonicity.md.
        let positives: Vec<usize> = results.iter().map(|(_, i)| i.positive).collect();
        let any_decrease = positives.windows(2).any(|w| w[1] < w[0]);
        assert!(
            any_decrease,
            "issue #5 regression: expected non-monotone positive count, got monotone: \
             {:?}. If this assertion fails, the in-kernel pivot behavior has changed; \
             revisit the disposition in dev/research/issue-5-mss1-inertia-monotonicity.md \
             §9 and flip the assertion to monotone non-decreasing.",
            positives
        );
    }

    /// Issue #5 diagnostic — does raising `zero_tol` alone (no other
    /// code change) cure the wandering? Empirical answer (2026-05-10):
    /// **no**. zero_tol from 1e-10 to 1e-2 produces identical n+
    /// `[93, 91, 94, 90, 92, ...]`. The wandering pivots have
    /// magnitudes O(1) — the BK 1×1/2×2 boundary instability is
    /// *above* the absolute pivot floor regime. Option B's SEUIL
    /// floor will not fix the wandering; it only stabilises the
    /// large-δ_w underflow tail.
    ///
    /// Diagnostic-only — always passes; prints the sweep table.
    #[test]
    fn issue_5_mss1_zero_tol_sweep_diagnostic() {
        let tols = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2];
        for &tol in &tols {
            let Some(results) = issue5_mss1_inertia_sweep_with(tol, 0.0) else {
                return;
            };
            let positives: Vec<usize> = results.iter().map(|(_, i)| i.positive).collect();
            let negatives: Vec<usize> = results.iter().map(|(_, i)| i.negative).collect();
            let zeros: Vec<usize> = results.iter().map(|(_, i)| i.zero).collect();
            let monotone = positives.windows(2).all(|w| w[1] >= w[0]);
            eprintln!(
                "zero_tol = {:>5.0e}: monotone(+) = {}\n  +: {:?}\n  -: {:?}\n  0: {:?}",
                tol, monotone, positives, negatives, zeros
            );
        }
    }

    /// Issue #5 diagnostic — does raising `pivot_threshold` (issue
    /// #2's lever) cure the wandering? `pivot_threshold` gates the
    /// 2×2 Duff-Reid growth bound, rejecting marginal 2×2 pivots
    /// in favor of 1×1 splits. If the wandering is driven by 2×2
    /// pivots flickering across the α-test boundary, raising
    /// pivot_threshold should stabilise the trace at issue #2's
    /// recommended `1e-8` default.
    ///
    /// Diagnostic-only — always passes; prints the sweep table.
    #[test]
    fn issue_5_mss1_pivot_threshold_sweep_diagnostic() {
        let us = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 0.5];
        for &u in &us {
            let Some(results) = issue5_mss1_inertia_sweep_with(1e-10, u) else {
                return;
            };
            let positives: Vec<usize> = results.iter().map(|(_, i)| i.positive).collect();
            let negatives: Vec<usize> = results.iter().map(|(_, i)| i.negative).collect();
            let zeros: Vec<usize> = results.iter().map(|(_, i)| i.zero).collect();
            let monotone = positives.windows(2).all(|w| w[1] >= w[0]);
            eprintln!(
                "pivot_threshold = {:>5.0e}: monotone(+) = {}\n  +: {:?}\n  -: {:?}\n  0: {:?}",
                u, monotone, positives, negatives, zeros
            );
        }
    }

    /// Build a small Vec<Supernode> with the given (ncol, nrow,
    /// n_children) for testing `estimate_assembly_flops` and
    /// `should_parallelize_assembly` in isolation. Neither
    /// function reads `row_indices` or the contents of `children`
    /// — only `ncol`, `nrow`, `children.len()`, and the
    /// supernodes-vec length. `row_indices` is therefore left
    /// empty (not `vec![0; nrow]`) so the saturation test can
    /// pass `nrow = 1 << 32` without OOMing on a 32 GiB
    /// allocation — see CI failure on commit 8a2a8e1
    /// (`memory allocation of 34359738368 bytes failed` on the
    /// GH Actions runner).
    fn make_supernodes(specs: &[(usize, usize, usize)]) -> Vec<Supernode> {
        specs
            .iter()
            .map(|&(ncol, nrow, n_children)| Supernode {
                first_col: 0,
                ncol,
                nrow,
                row_indices: Vec::new(),
                children: vec![0; n_children],
                delayed_capacity: usize::MAX,
            })
            .collect()
    }

    /// Minimal `SymbolicFactorization` carrying only the fields
    /// `should_parallelize_assembly` inspects (`supernodes`). All
    /// other fields are zeroed/empty defaults; do not pass this
    /// stub to anything that touches the etree, pattern, or
    /// permutation.
    fn stub_symbolic(n: usize, snodes: Vec<Supernode>) -> SymbolicFactorization {
        SymbolicFactorization {
            n,
            perm: Vec::new(),
            perm_inv: Vec::new(),
            supernodes: snodes,
            factor_nnz_estimate: 0,
            factor_slack: 1.2,
            contrib_sizes: Vec::new(),
            peak_contrib_bytes: 0,
            etree: crate::ordering::elimination_tree::EliminationTree {
                parent: Vec::new(),
                n: 0,
            },
            permuted_pattern: crate::sparse::csc::CscPattern {
                n: 0,
                col_ptr: vec![0],
                row_idx: Vec::new(),
            },
            col_counts: Vec::new(),
            small_leaf_groups: Vec::new(),
            snode_group: Vec::new(),
            cached_mc64: None,
            resolved_method: crate::symbolic::OrderingMethod::Amd,
            resolved_amalgamation: crate::symbolic::AmalgamationStrategy::Adjacency,
            resolved_preprocess: crate::symbolic::OrderingPreprocess::None,
            is_schur_tail: None,
        }
    }

    #[test]
    fn estimate_assembly_flops_empty_tree_is_zero() {
        assert_eq!(estimate_assembly_flops(&[]), 0);
    }

    #[test]
    fn estimate_assembly_flops_sums_ncol_times_nrow_squared() {
        // Two supernodes: (2, 4) → 2*4*4 = 32, (3, 5) → 3*5*5 = 75.
        // Total expected: 107.
        let snodes = make_supernodes(&[(2, 4, 0), (3, 5, 0)]);
        assert_eq!(estimate_assembly_flops(&snodes), 107);
    }

    #[test]
    fn estimate_assembly_flops_saturates_on_pathological_input() {
        // ncol = 1, nrow = u32::MAX as usize on a 64-bit host: nrow^2
        // = ~1.8e19 which fits in u64; ncol * that fits too. Make it
        // bigger: 100 nodes of (1, 2^32) — each contributes 2^64
        // which saturates to u64::MAX, and the sum saturates to
        // u64::MAX. The point is `should_parallelize_assembly` must
        // return true (high-flop regime) without panicking.
        let snodes = make_supernodes(&[(1, 1usize << 32, 0); 4]);
        // Per-node: 1 * 2^32 * 2^32 = 2^64 → saturates to u64::MAX.
        // Sum: 4 × u64::MAX → saturates to u64::MAX.
        assert_eq!(estimate_assembly_flops(&snodes), u64::MAX);
    }

    /// `should_parallelize_assembly` end-to-end on a tridiagonal
    /// n=64 matrix. The elimination tree is a pure chain (each
    /// supernode has 0 or 1 children) — even though the per-node
    /// flops may be small the structural-chain gate alone keeps the
    /// parallel driver off. Pre-issue-#19 the gate was already
    /// rejecting this; the test pins the existing behavior.
    #[test]
    fn should_parallelize_assembly_rejects_tridiagonal_chain() {
        let n = 64usize;
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
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        assert!(
            !should_parallelize_assembly(&sym),
            "tridiagonal n=64 has a chain etree (or trivially small \
             supernode count); parallel driver must not fire"
        );
    }

    /// New issue-#19 gate: hand-built tree with N_PAR_MIN multi-child
    /// supernodes but per-node flops below the threshold. The
    /// structural gates pass; the flop gate must veto.
    #[test]
    fn should_parallelize_assembly_rejects_low_flop_multi_child_tree() {
        // 64 supernodes, every other one has 2 children — structural
        // gate passes. ncol=2, nrow=4 ⇒ 32 flops/node × 64 = 2048
        // flops total. Way below PAR_MIN_FLOPS (10⁷ post-calibration).
        let mut specs = vec![(2usize, 4usize, 2usize); 64];
        for (i, s) in specs.iter_mut().enumerate() {
            if i % 2 == 1 {
                s.2 = 0;
            }
        }
        let snodes = make_supernodes(&specs);
        assert!(snodes.len() >= N_PAR_MIN);
        assert!(snodes.iter().any(|s| s.children.len() >= 2));
        assert!(estimate_assembly_flops(&snodes) < PAR_MIN_FLOPS);
        // Build a SymbolicFactorization with these supernodes. Most
        // fields are unused by should_parallelize_assembly — use
        // empty placeholders.
        let sym = stub_symbolic(128, snodes);
        assert!(
            !should_parallelize_assembly(&sym),
            "tree with {} supernodes and {} flops < PAR_MIN_FLOPS = {} \
             must dispatch sequentially per issue #19",
            sym.supernodes.len(),
            estimate_assembly_flops(&sym.supernodes),
            PAR_MIN_FLOPS,
        );
    }

    /// Counterpart to the previous test: same structure but per-node
    /// flops large enough to clear the gate.
    #[test]
    fn should_parallelize_assembly_accepts_high_flop_multi_child_tree() {
        // 64 supernodes, ncol=64, nrow=256 ⇒ 64 * 65536 = 4.19e6
        // flops/node × 64 ≈ 2.7e8 total. Above PAR_MIN_FLOPS.
        let mut specs = vec![(64usize, 256usize, 2usize); 64];
        for (i, s) in specs.iter_mut().enumerate() {
            if i % 2 == 1 {
                s.2 = 0;
            }
        }
        let snodes = make_supernodes(&specs);
        assert!(estimate_assembly_flops(&snodes) >= PAR_MIN_FLOPS);
        let sym = stub_symbolic(64 * 256, snodes);
        assert!(
            should_parallelize_assembly(&sym),
            "tree with structural gates passed and {} flops >= {} \
             must dispatch parallel",
            estimate_assembly_flops(&sym.supernodes),
            PAR_MIN_FLOPS,
        );
    }

    // ---- Track B1: prologue sub-phase instrumentation ----

    #[test]
    fn permute_csc_values_profile_flag_gates_timing_not_values() {
        // 3x3 symmetric matrix stored lower-triangle: diag (4,5,6),
        // one off-diagonal (1,0)=2. A non-identity permutation
        // (perm_inv reverses the order) makes the rebuild non-trivial.
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1], &[0, 1, 2, 0], &[4.0, 5.0, 6.0, 2.0])
            .unwrap();
        let perm = [2usize, 1, 0];
        let perm_inv = [2usize, 1, 0];

        let (off, t_off) = permute_csc_values(&m, &perm, &perm_inv, false).unwrap();
        let (on, t_on) = permute_csc_values(&m, &perm, &perm_inv, true).unwrap();

        // The `profile` flag is timing-only: the permuted matrix must
        // be bit-identical regardless of whether timing is collected.
        assert_eq!(off.n, on.n);
        assert_eq!(off.col_ptr, on.col_ptr);
        assert_eq!(off.row_idx, on.row_idx);
        assert_eq!(off.values, on.values);

        // `profile == false` does no `Instant::now()` call, so the
        // reported `from_triplets_us` is deterministically zero.
        assert_eq!(t_off, 0, "profile=false must report zero timing");
        // `profile == true` may legitimately round to zero on a tiny
        // matrix — only the gating direction is asserted here.
        let _ = t_on;
    }

    #[test]
    fn prologue_breakdown_subphases_sum_within_prologue() {
        // Tridiagonal SPD n=64 (diag 2, off-diag -1): large enough to
        // exercise the supernodal driver. Inertia oracle is by hand —
        // a symmetric diagonally-dominant tridiagonal matrix with
        // positive diagonal is SPD, so inertia is (64, 0, 0).
        let n = 64usize;
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
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();

        let prof = Arc::new(Mutex::new(Profiler::new()));
        let mut params = make_params();
        params.profiler = Some(prof.clone());
        let (_factors, inertia) = factorize_multifrontal(&m, &sym, &params).unwrap();
        assert_eq!(
            (inertia.positive, inertia.negative, inertia.zero),
            (64, 0, 0),
            "SPD tridiagonal must factor with all-positive inertia"
        );

        let report = match prof.lock() {
            Ok(p) => p.report(),
            Err(_) => panic!("profiler mutex poisoned"),
        };
        let bd = &report.prologue_breakdown;

        // `from_triplets` is a sub-call of `permute_csc_values`, so its
        // wall is a subset of `permute_us`.
        assert!(
            bd.permute_from_triplets_us <= bd.permute_us,
            "from_triplets ({} us) cannot exceed permute total ({} us)",
            bd.permute_from_triplets_us,
            bd.permute_us
        );

        // The seven sub-phases are disjoint segments of the prologue
        // span, so the sum of their (floored) wall cannot exceed the
        // (floored) prologue wall.
        let subphase_sum = bd.row_map_us
            + bd.scaling_us
            + bd.scaling_pivot_order_us
            + bd.permute_us
            + bd.infnorm_tol_us
            + bd.symmetric_pattern_us
            + bd.setup_us;
        assert!(
            subphase_sum <= report.prologue_us,
            "prologue sub-phases sum to {} us, exceeding prologue {} us",
            subphase_sum,
            report.prologue_us
        );
    }

    /// N7: a one-shot caller (`pattern_reused_hint == false`) must not
    /// pay to build the value-map cache — the warm fast path is gated on
    /// the same hint, so a cache built on a cold one-shot call would
    /// never be read. The cold path must still produce the correct
    /// permuted matrix; only the (wasted) cache build is skipped.
    #[test]
    fn permute_cache_not_built_for_one_shot_caller() {
        // Lower-triangular SPD-ish pattern, identity permutation so the
        // permuted matrix equals the input and is trivial to verify.
        let n = 4usize;
        let rows = vec![0usize, 1, 1, 2, 2, 3, 3];
        let cols = vec![0usize, 0, 1, 1, 2, 2, 3];
        let vals = vec![4.0f64, -1.0, 4.0, -1.0, 4.0, -1.0, 4.0];
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let perm: Vec<usize> = (0..n).collect();
        let perm_inv: Vec<usize> = (0..n).collect();

        // One-shot: hint off, empty cache.
        let mut cache: Option<PermuteCache> = None;
        let (permuted, _us) =
            permute_csc_values_with_cache(&m, &perm, &perm_inv, false, false, &mut cache).unwrap();

        // Permuted matrix is correct (identity perm ⇒ equals input).
        assert_eq!(permuted.n, m.n);
        assert_eq!(permuted.col_ptr, m.col_ptr);
        assert_eq!(permuted.row_idx, m.row_idx);
        assert_eq!(permuted.values, m.values);

        // N7: the cache must remain unbuilt for a one-shot caller.
        // Pre-fix this is `Some(..)` (the cold path populated it
        // unconditionally); post-fix it stays `None`.
        assert!(
            cache.is_none(),
            "one-shot caller (hint=false) should not build the value-map cache"
        );

        // Sanity: a warm-reuse caller (hint=true) still builds the cache
        // on its first cold call, so the gating is conditional, not a
        // blanket disable.
        let mut warm_cache: Option<PermuteCache> = None;
        let (_permuted2, _us2) =
            permute_csc_values_with_cache(&m, &perm, &perm_inv, true, true, &mut warm_cache)
                .unwrap();
        assert!(
            warm_cache.is_some(),
            "warm-reuse caller (hint=true) should build the value-map cache on its cold call"
        );
    }

    /// REG-1 (repo-review-2026-06-09-verification.md): a stale
    /// `PermuteCache` left by an earlier pattern must NOT be reused for a
    /// different input pattern that merely shares `(n, nnz)`. N7
    /// (`131a6de`) made the cold-path rebuild conditional on
    /// `hint == true`, so a `hint == false` call no longer refreshes the
    /// cache; combined with the warm path validating only
    /// `(n, nnz, value_map.len())` this let a later warm call scatter new
    /// values through the previous pattern's structure and return Success
    /// with a wrong factorization. The warm path must validate the actual
    /// input pattern before it fires.
    #[test]
    fn permute_cache_rejects_stale_pattern_same_n_nnz() {
        let n = 4usize;
        let perm: Vec<usize> = (0..n).collect();
        let perm_inv: Vec<usize> = (0..n).collect();

        // Pattern A and pattern B share (n, nnz) = (4, 7) but differ
        // structurally (different row_idx / col_ptr).
        let a = CscMatrix::from_triplets(
            n,
            &[0, 1, 1, 2, 2, 3, 3],
            &[0, 0, 1, 1, 2, 2, 3],
            &[4.0, -1.0, 4.0, -1.0, 4.0, -1.0, 4.0],
        )
        .unwrap();
        let b = CscMatrix::from_triplets(
            n,
            &[0, 1, 2, 2, 3, 3, 3],
            &[0, 1, 0, 2, 1, 3, 0],
            &[4.0, 4.0, -1.0, 4.0, -1.0, 4.0, -2.0],
        )
        .unwrap();
        assert_eq!(a.nnz(), b.nnz(), "patterns must share nnz for the probe");

        // Warm-reuse caller builds A's cache (hint=true cold call).
        let mut cache: Option<PermuteCache> = None;
        permute_csc_values_with_cache(&a, &perm, &perm_inv, false, true, &mut cache).unwrap();
        assert!(cache.is_some());

        // A one-shot (hint=false) call on pattern B follows.
        permute_csc_values_with_cache(&b, &perm, &perm_inv, false, false, &mut cache).unwrap();

        // Warm call on pattern B must equal the canonical permutation of
        // B — NOT B's values scattered through A's structure.
        let (warm_b, _) =
            permute_csc_values_with_cache(&b, &perm, &perm_inv, false, true, &mut cache).unwrap();
        let (canon_b, _) = permute_csc_values(&b, &perm, &perm_inv, false).unwrap();
        assert_eq!(warm_b.col_ptr, canon_b.col_ptr, "stale-pattern col_ptr");
        assert_eq!(warm_b.row_idx, canon_b.row_idx, "stale-pattern row_idx");
        assert_eq!(warm_b.values, canon_b.values, "stale-pattern values");
    }

    /// REG-1 second route: the same input pattern but a changed
    /// permutation (AutoRace selecting a different ordering after a
    /// symbolic-cache invalidation) must also bypass the stale cache —
    /// `hint` stays `true` throughout, so the cold-path clear does not
    /// help; the stored `perm_inv` is what guards it.
    #[test]
    fn permute_cache_rejects_stale_permutation() {
        let n = 4usize;
        // Arrow pattern (dense first column): its permuted structure is
        // NOT invariant under reordering, so a stale-perm warm hit is
        // structurally visible. (A constant tridiagonal matrix would be
        // reversal-invariant and hide the bug.)
        let m = CscMatrix::from_triplets(
            n,
            &[0, 1, 2, 3, 1, 2, 3],
            &[0, 0, 0, 0, 1, 2, 3],
            &[10.0, 2.0, 3.0, 4.0, 20.0, 30.0, 40.0],
        )
        .unwrap();

        // Permutation 1 (identity) builds the cache.
        let perm1: Vec<usize> = (0..n).collect();
        let perm1_inv: Vec<usize> = (0..n).collect();
        let mut cache: Option<PermuteCache> = None;
        permute_csc_values_with_cache(&m, &perm1, &perm1_inv, false, true, &mut cache).unwrap();
        assert!(cache.is_some());

        // Permutation 2 (reversal) on the SAME pattern, hint still true.
        let perm2: Vec<usize> = vec![3, 2, 1, 0];
        let mut perm2_inv = vec![0usize; n];
        for (i, &p) in perm2.iter().enumerate() {
            perm2_inv[p] = i;
        }
        let (warm, _) =
            permute_csc_values_with_cache(&m, &perm2, &perm2_inv, false, true, &mut cache).unwrap();
        let (canon, _) = permute_csc_values(&m, &perm2, &perm2_inv, false).unwrap();
        assert_eq!(warm.col_ptr, canon.col_ptr, "stale-perm col_ptr");
        assert_eq!(warm.row_idx, canon.row_idx, "stale-perm row_idx");
        assert_eq!(warm.values, canon.values, "stale-perm values");
    }
}
