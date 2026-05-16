use crate::dense::rook::{rook_rescue, RookKind};
use crate::error::FeralError;
use crate::inertia::Inertia;

// Phase 2.4.3: the rank-1 / rank-2 Schur-update inner loops dispatch to
// the 4-way unrolled non-FMA pulp kernels in `crate::dense::schur_kernel`.
// The non-FMA variants reproduce the scalar loop's rounding bit-for-bit
// (two IEEE 754 roundings per element) so inertia counts are identical
// to the scalar path — verified by bit-exact unit tests across a length
// sweep and by the full KKT bench. The ILP win comes from 4 independent
// accumulators exposing parallelism that the single-accumulator
// autovectorized scalar loop could not. The kernel itself dispatches
// per-arch (NEON on aarch64, AVX2 on x86_64-v3, scalar fallback
// elsewhere); see commit 18194807.
use crate::dense::schur_kernel;

/// Phase 2.4.1c triage flag. When set to `true`, `factor_frontal_blocked`
/// delegates to the scalar `factor_frontal` unconditionally. Used by
/// `examples/triage_sparse_kernel_diff.rs` to A/B-compare the two
/// kernels across the full KKT corpus. Default `false` preserves
/// production dispatch; setting this is a diagnostic affordance only.
pub static FORCE_SCALAR_FRONTAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Phase A diagnostic flag (dev/plans/dense-kernel-blas3.md). When set to
/// `true`, `lblt_panel_frontal` skips the inline 2×2 acceptance and
/// returns `ScalarFallback` on every 2×2 trigger (matching the pre-W-2
/// 2×2 behavior). Used to bisect parity regressions specifically to the
/// inline 2×2 path. Default `false` preserves the W-2 2×2 fast path.
pub static DISABLE_PANEL_INLINE_2X2: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Phase B-1.5 diagnostic flag
/// (`dev/research/dense-kernel-attribution-2026-04-28.md`). When `true`,
/// the panel driver and `lblt_panel_frontal` increment the counters in
/// `panel_diag` to attribute panel-vs-scalar work and bail reasons.
/// Default `false`; the load is one relaxed atomic + branch per call
/// site (well-predicted at "false") so production overhead is
/// negligible.
pub static PANEL_DIAG_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Counters populated when `PANEL_DIAG_ENABLED` is on. All `Relaxed`
/// reads/writes — the diagnostic binary clears them between matrices
/// and reads totals at the end of the run. Phase B-1.5
/// (`dev/research/dense-kernel-attribution-2026-04-28.md`).
pub mod panel_diag {
    use std::sync::atomic::AtomicU64;
    /// Panels that committed all `bs` requested pivots inline.
    pub static PANEL_FULL: AtomicU64 = AtomicU64::new(0);
    /// Panels that bailed with `n_elim < bs` via `ScalarFallback*`.
    pub static PANEL_PARTIAL: AtomicU64 = AtomicU64::new(0);
    /// Panels that broke on a delayed pivot (SSIDS may_delay path).
    pub static PANEL_DELAYED: AtomicU64 = AtomicU64::new(0);
    /// 2×2 trigger but argmax row r != col+1 (swap required), or
    /// col+1 >= ncol, or panel cap exhausted. Inline 2×2 declined
    /// before any peek-ahead; ScalarFallback returned.
    pub static FALLBACK_2X2_NEED_SWAP_OR_BOUND: AtomicU64 = AtomicU64::new(0);
    /// 2×2 trigger, no swap, but `arr >= alpha_bk * gamma_r` so the
    /// scalar path's swap-1×1 alternative would win. Bail.
    pub static FALLBACK_2X2_SWAP_1X1_WINS: AtomicU64 = AtomicU64::new(0);
    /// 2×2 trigger, no swap, but LAPACK-extension 1×1 alternative
    /// `akk * gamma_r >= alpha * gamma0^2` would win. Bail.
    pub static FALLBACK_2X2_LAPACK_1X1_WINS: AtomicU64 = AtomicU64::new(0);
    /// 2×2 candidate failed the Duff-Reid growth bound or the SSIDS
    /// scale-invariant det floor. Bail to scalar (which has the
    /// may_delay/ForceAccept escalation paths).
    pub static FALLBACK_2X2_GROWTH_OR_DET: AtomicU64 = AtomicU64::new(0);
    /// Scalar tail steps (panel disengaged because `remaining < PANEL_MIN_NCOL`).
    pub static SCALAR_TAIL_STEPS: AtomicU64 = AtomicU64::new(0);
    /// Pivots accepted inside the panel (committed via deferred-Schur).
    pub static PIVOTS_INLINE: AtomicU64 = AtomicU64::new(0);
    /// Pivots performed by `scalar_pivot_step` (after a fallback or in
    /// the scalar tail).
    pub static PIVOTS_SCALAR: AtomicU64 = AtomicU64::new(0);
    /// Phase A2 (`dev/plans/dense-kernel-w2-2x2-swap.md`): swap-required
    /// 2×2 pivots committed inline at c==0. Counts a successful
    /// `swap_rows_cols(col+1, r)` followed by an inline 2×2 accept.
    pub static INLINE_2X2_SWAP_OK: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        for c in [
            &PANEL_FULL,
            &PANEL_PARTIAL,
            &PANEL_DELAYED,
            &FALLBACK_2X2_NEED_SWAP_OR_BOUND,
            &FALLBACK_2X2_SWAP_1X1_WINS,
            &FALLBACK_2X2_LAPACK_1X1_WINS,
            &FALLBACK_2X2_GROWTH_OR_DET,
            &SCALAR_TAIL_STEPS,
            &PIVOTS_INLINE,
            &PIVOTS_SCALAR,
            &INLINE_2X2_SWAP_OK,
        ] {
            c.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn snapshot() -> [(&'static str, u64); 11] {
        use std::sync::atomic::Ordering::Relaxed;
        [
            ("panel_full", PANEL_FULL.load(Relaxed)),
            ("panel_partial", PANEL_PARTIAL.load(Relaxed)),
            ("panel_delayed", PANEL_DELAYED.load(Relaxed)),
            (
                "fallback_2x2_need_swap_or_bound",
                FALLBACK_2X2_NEED_SWAP_OR_BOUND.load(Relaxed),
            ),
            (
                "fallback_2x2_swap_1x1_wins",
                FALLBACK_2X2_SWAP_1X1_WINS.load(Relaxed),
            ),
            (
                "fallback_2x2_lapack_1x1_wins",
                FALLBACK_2X2_LAPACK_1X1_WINS.load(Relaxed),
            ),
            (
                "fallback_2x2_growth_or_det",
                FALLBACK_2X2_GROWTH_OR_DET.load(Relaxed),
            ),
            ("scalar_tail_steps", SCALAR_TAIL_STEPS.load(Relaxed)),
            ("pivots_inline", PIVOTS_INLINE.load(Relaxed)),
            ("pivots_scalar", PIVOTS_SCALAR.load(Relaxed)),
            ("inline_2x2_swap_ok", INLINE_2X2_SWAP_OK.load(Relaxed)),
        ]
    }
}

/// Issue #13 Phase A + C — caller-supplied scratch.
///
/// **Phase A**: two internal-only working buffers (`subdiag`, `d_panel`)
/// reused across supernodes. The kernel `clear()`s then `resize()`s on
/// entry, preserving capacity across calls of differing `nrow` / `bs`.
///
/// **Phase C**: `contrib_pool` is a stack of recyclable `Vec<f64>`
/// buffers that the multifrontal driver feeds back after `extend_add`
/// has consumed a child's contribution block. The kernel pops one at
/// extract time, `clear()`+`resize()`s to `cdim*cdim`, and writes the
/// trailing Schur block. When the pool is empty the kernel falls back
/// to a fresh allocation — `FrontalFactors` always owns its `contrib`
/// Vec, the pool is purely a malloc-amortisation channel.
#[derive(Default, Debug, Clone)]
pub struct FactorScratch {
    /// Working subdiagonal of D, length `nrow` per call.
    pub subdiag: Vec<f64>,
    /// Panel-local D values, length `bs` per call.
    pub d_panel: Vec<f64>,
    /// Recyclable contribution-block buffer (Phase C, single-slot).
    /// The factor kernel takes at extract time; the multifrontal driver
    /// puts after `extend_add` consumes a child contribution block.
    /// `None` means the kernel falls back to a fresh `Vec` allocation.
    /// A single-slot pool keeps bookkeeping cost to one branch +
    /// `Option::take` / `=Some(...)` per supernode; a multi-slot
    /// `Vec<Vec<f64>>` pool was tried and regressed bench p90 by ~0.2
    /// (small) / ~0.3 (medium) — see issue #13 Phase C investigation.
    pub contrib_pool: Option<Vec<f64>>,
}

impl FactorScratch {
    /// Construct an empty scratch. Equivalent to `default()`.
    pub fn new() -> Self {
        Self::default()
    }
}

#[inline(always)]
fn diag_inc(counter: &std::sync::atomic::AtomicU64) {
    if PANEL_DIAG_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[inline(always)]
fn diag_add(counter: &std::sync::atomic::AtomicU64, n: u64) {
    if PANEL_DIAG_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        counter.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Dead-zero absolute floor for the 2×2 pivot cancellation test.
/// Matches SPRAL SSIDS `datatypes.f90:260` default (`small = 1e-20`),
/// used by `ldlt_tpp.cxx:98,106`. This is a true zero-detection floor
/// (near the underflow boundary), **not** a stability threshold — the
/// scale-invariant cancellation test at `factor_frontal`'s 2×2 gate
/// handles stability via ratios against the local block max.
const SSIDS_DET_SMALL: f64 = 1e-20;

/// Pivot-growth threshold above which a factor is flagged for iterative
/// refinement. A well-pivoted BK factor with unit-diagonal L satisfies
/// |L_ij| ≤ 1/(1 − α) ≈ 2.78; values substantially above this indicate
/// that some accepted pivot was small relative to its column max.
///
/// With `pivot_threshold = 0.0` (the BK default), the alpha-test is
/// satisfied vacuously when both the candidate diagonal and its column
/// off-diagonals are simultaneously tiny, producing a "successful"
/// factor whose plain forward/back substitution cannot reach machine
/// precision. Bratu3d under default params reaches max|L| ≈ 8e16 and
/// returns a plain residual of 4.66e6; setting `pivot_threshold = 0.01`
/// keeps max|L| ≈ 27 with plain residual 1.25e-14. See
/// dev/journal/2026-04-25-02.org.
///
/// 1e6 is a conservative trigger: matrices with growth in [2.78, 1e6]
/// converge in 1–2 IR steps without being flagged here; matrices above
/// 1e6 are catastrophic and need IR for any reasonable accuracy.
const L_GROWTH_THRESHOLD: f64 = 1e6;

/// Sets `*needs_refinement = true` when any `|L_ij| > L_GROWTH_THRESHOLD`.
/// Called at every dense-factor exit path so callers using plain
/// `Solver::solve` (rather than `solve_refined`) get a programmatic
/// signal — `factors.needs_refinement` — that the factor is too
/// unstable for plain forward/back substitution.
fn flag_growth_for_refinement(l: &[f64], needs_refinement: &mut bool) {
    if *needs_refinement {
        return;
    }
    for &v in l {
        if v.abs() > L_GROWTH_THRESHOLD {
            *needs_refinement = true;
            return;
        }
    }
}

/// Threshold-partial-pivoting acceleration mode.
///
/// `Plain` (default) computes the column AMAX from scratch at every
/// pivot iteration. `Maxfromm` reuses MUMPS's MAXFROMM trick (see
/// `dfac_front_aux.F` `DMUMPS_FAC_I_LDLT` / `DMUMPS_FAC_MQ_LDLT`):
/// after a 1×1 rank-1 trailing update at pivot `k`, scan the freshly
/// updated column `k+1` once and stash its max. The next iteration
/// then accepts at pivot `k+1` without re-scanning iff
/// `|a_{k+1,k+1}| >= alpha * cached_max`. The acceptance predicate is
/// bit-identical to `Plain`'s BK test, so inertia and L/D are
/// unchanged — only the AMAX scan is skipped. Falls back to a full
/// scan on rejection / delay / 2×2.
///
/// See `dev/research/issue-10-app-vs-maxfromm.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TppMethod {
    /// Recompute the AMAX scan at every pivot (Phase ≤ 2.11 behavior).
    #[default]
    Plain,
    /// Capture column k+1's AMAX as a byproduct of the rank-1 trailing
    /// update at pivot k; reuse on the next pivot when the diagonal
    /// dominates by `alpha`.
    Maxfromm,
}

/// Parameters controlling Bunch-Kaufman factorization behavior.
#[derive(Debug, Clone)]
pub struct BunchKaufmanParams {
    /// Pivot threshold α. BK standard: (1 + sqrt(17)) / 8 ≈ 0.6404.
    pub alpha: f64,

    /// A 1×1 pivot |d| <= zero_tol is considered "truly zero" — the L
    /// column is zeroed (or the pivot is perturbed under PerturbToEps)
    /// and the solve will skip dividing by `d_diag[k]` when checking
    /// `|d| > factors.zero_tol`. Default: f64::EPSILON ≈ 2.22e-16.
    ///
    /// Rationale: for a well-equilibrated matrix with ||A|| ~ 1, the
    /// rounding error floor is ~eps. Any pivot more than eps above zero
    /// has a reliable sign and should be counted as positive/negative,
    /// not zero. The previous default of 100*eps (2.22e-14) was too
    /// aggressive and flagged legitimate small-positive pivots as zero
    /// on SPD matrices — verified by triage against canonical MUMPS,
    /// SSIDS, and rmumps on CERI651DLS_0534 and FBRAIN3LS_0788
    /// (2026-04-12, dev/journal/2026-04-12-01.org).
    ///
    /// This is the *solve-time* floor (propagated to Factors.zero_tol).
    /// For factor-time rank-deficiency detection on scaled matrices,
    /// see `null_pivot_tol`.
    pub zero_tol: f64,

    /// A 2×2 pivot block is "truly singular" when |det| <= zero_tol_2x2.
    /// Default: zero_tol². Solve-time floor; see `null_pivot_tol_2x2`
    /// for the factor-time rank-deficiency analogue.
    pub zero_tol_2x2: f64,

    /// Factor-time rank-deficiency floor for 1×1 pivots. A pivot with
    /// `zero_tol < |d| <= null_pivot_tol` is counted as zero in the
    /// inertia signature (rank deficiency) but the factor is left
    /// untouched: L stays divided by `d`, the trailing update fires,
    /// and the solve uses the strict `zero_tol` to decide whether to
    /// divide. Default equals `zero_tol` (no rank-deficiency band).
    ///
    /// The sparse multifrontal driver overrides this to
    /// `sqrt(n) · EPS · ‖A_scaled‖_∞` (MUMPS CNTL(3)-style) when
    /// `on_zero_pivot != Fail`, so rank-deficient KKT systems report
    /// honest inertia without degrading solve quality on
    /// ill-conditioned but non-singular matrices. See
    /// `dev/research/f01-rankdef-underreporting.md`.
    pub null_pivot_tol: f64,

    /// Factor-time rank-deficiency floor for 2×2 pivot blocks. Mirrors
    /// `null_pivot_tol` for the 2×2 determinant check. Default equals
    /// `zero_tol_2x2`.
    pub null_pivot_tol_2x2: f64,

    /// What to do when the selected pivot is numerically zero.
    pub on_zero_pivot: ZeroPivotAction,

    /// Column-relative pivot threshold `u` (MUMPS `CNTL(1)`, SSIDS `options%u`).
    /// A 1×1 candidate pivot `a_kk` is accepted only if
    /// `|a_kk| >= u * max_{i>k}(|a_ik|)`, i.e. the pivot must dominate its
    /// column by at least a factor of `1/u`. Additionally, a 2×2 pivot block
    /// is accepted only if the Duff-Reid growth bound
    /// `(|a22|*RMAX + AMAX*TMAX)*u <= |det|` (and its symmetric partner)
    /// holds, where RMAX/TMAX are column maxes of the two pivot columns
    /// *beyond* the 2×2 block and AMAX is the cross term.
    ///
    /// Default `0.0` preserves Phase 1 behavior (no threshold check — every
    /// non-zero pivot is accepted). Callers opting into MC64 scaling should
    /// set this to `0.01` (MUMPS/SSIDS default) so that after symmetric
    /// equilibration, candidate pivots that are more than 100× smaller than
    /// their column max are rejected and flushed through the existing
    /// `ForceAccept` path. See dev/plans/scaling-aware-pivot-rejection.md
    /// and MUMPS `dfac_front_aux.F:1494-1606` for the reference formulas.
    pub pivot_threshold: f64,

    /// Panel width for the blocked dense Schur update. Consulted by the
    /// Phase 2.4.1b blocked-panel path in `factor_frontal`; ignored when
    /// `remaining <= block_size` or when `may_delay == true` routes the
    /// factor through the scalar path. Default 64 matches faer's
    /// `factor.rs:722` crossover. See `dev/plans/phase-2.4.1-blocked-ldlt.md`.
    pub block_size: usize,

    /// Opt-in FMA dispatch on the dense trailing-update / panel-update
    /// kernels. Default `false` keeps the cross-arch bit-exact non-FMA
    /// path; `true` switches to the FMA siblings for ~2x arithmetic
    /// throughput on aarch64 NEON and x86 V3 AVX2+FMA. Mirrors
    /// `NumericParams::fma`; the sparse multifrontal driver copies
    /// `NumericParams::fma` here when constructing per-supernode BK
    /// params. See `dev/research/fma-kernel-opt-in.md`, issue #8.
    pub fma: bool,

    /// Threshold-partial-pivoting acceleration. See [`TppMethod`].
    /// Default `Plain` (no acceleration); the multifrontal driver may
    /// flip this per-front via `NumericParams::tpp_method`.
    pub tpp_method: TppMethod,
}

/// Action to take when a near-zero pivot is encountered.
#[derive(Debug, Clone)]
pub enum ZeroPivotAction {
    /// Accept the tiny pivot at face value: zero the L column, count
    /// as one extra zero in the inertia signature, and flag for
    /// iterative refinement. The perturbation magnitude is unbounded —
    /// effectively the difference between the true pivot and zero.
    /// Use this only when downstream code can tolerate sign-loss in
    /// the perturbed positions (e.g. callers that re-check inertia
    /// against an expected signature and refactor on mismatch).
    ForceAccept,
    /// Return FeralError::NumericallyRankDeficient.
    Fail,
    /// Replace the tiny pivot with `sign(d) * max(|d|, abs_floor)`,
    /// keep the L column live, and count the perturbed pivot by its
    /// sign (positive or negative — never zero). Sign of `0.0` is
    /// treated as `+1.0`.
    ///
    /// The factor satisfies `L · D · L^T = A + Δ` exactly (within
    /// roundoff) for the L and D produced, but `Δ` is *not* localised
    /// to the perturbed diagonal entry. `L[i,k] = A[i,k] / d_new`
    /// stays live, so the trailing Schur update reduces `A[i,j]` by
    /// `A[i,k] · A[j,k] / d_new` rather than the "true" reduction
    /// `A[i,k] · A[j,k] / d_orig`. The implicit `Δ` per perturbed
    /// pivot therefore scales with `||A[k+1:,k]||² · |1/d_new −
    /// 1/d_orig|`, which is bounded by `||A[:,k]||² / abs_floor`
    /// in the worst case — *not* by `abs_floor` as a naive Weyl
    /// reading might suggest. On numerically reasonable IPM KKT
    /// matrices the unrefined residual stays small in practice
    /// (e.g. `~1e-5` on `robot_1600_0004`), but downstream code
    /// should drive iterative refinement against unperturbed `A`
    /// for tight tolerances. Sets `needs_refinement = true`.
    ///
    /// Closest published precedent: LAPACK static pivoting (Trefethen
    /// & Bau §22) and MA57's `cntl(4)` static-pivot replacement. A
    /// typical `abs_floor` recipe is `eps_rel · ||A||_∞` with
    /// `eps_rel` between `1e-12` and `1e-8` for IPM KKT systems.
    ///
    /// History: `dev/journal/2026-05-13-03.org` §01:15 (motivation),
    /// `dev/research/cascade-break-l-perturbation-2026-05-15.md`
    /// (forensics), `dev/tried-and-rejected.md` "L-zeroing fix"
    /// (rejected attempt at making the bound match the original
    /// docstring).
    PerturbToEps { abs_floor: f64 },
}

/// Compute the perturbed pivot for `ZeroPivotAction::PerturbToEps`.
/// `sign(0) = +1`, so a true zero pivot becomes `+abs_floor`.
#[inline]
fn perturb_to_floor(d: f64, abs_floor: f64) -> f64 {
    let mag = d.abs().max(abs_floor);
    if d < 0.0 {
        -mag
    } else {
        mag
    }
}

impl Default for BunchKaufmanParams {
    fn default() -> Self {
        let zero_tol = f64::EPSILON;
        Self {
            alpha: (1.0 + 17f64.sqrt()) / 8.0, // ≈ 0.6404
            zero_tol,
            zero_tol_2x2: zero_tol * zero_tol,
            null_pivot_tol: zero_tol,
            null_pivot_tol_2x2: zero_tol * zero_tol,
            on_zero_pivot: ZeroPivotAction::Fail,
            pivot_threshold: 0.0,
            block_size: 64,
            fma: false,
            tpp_method: TppMethod::Plain,
        }
    }
}

/// Factorization result: P·L·D_bk·Lᵀ·Pᵀ = D_eq·A·D_eq.
#[derive(Debug)]
pub struct Factors {
    pub n: usize,
    /// Unit lower triangular L in full n×n column-major storage.
    /// Diagonal entries are 1.0 (stored explicitly).
    pub l: Vec<f64>,
    /// D_bk diagonal entries in pivot order. Length n.
    pub d_diag: Vec<f64>,
    /// D_bk sub-diagonal entries. Length n. Zero for 1×1 pivots.
    pub d_subdiag: Vec<f64>,
    /// BK pivot permutation (forward). Length n.
    /// perm[i] = j means original row j was moved to pivot position i.
    pub perm: Vec<usize>,
    /// Inverse permutation. perm_inv[perm[i]] == i for all i.
    pub perm_inv: Vec<usize>,
    /// Equilibration scaling diagonal D_eq. Length n.
    pub d_eq: Vec<f64>,
    /// True when ZeroPivotAction::ForceAccept fired during factorization.
    pub needs_refinement: bool,
    /// 1×1 pivot threshold copied from BunchKaufmanParams at factor time.
    /// `solve` consults this to decide whether to divide by `d_diag[k]`:
    /// pivots `|d| <= zero_tol` were force-accepted as numerically zero
    /// during factorization and must be skipped (left as-is) by the
    /// D-block solve. Otherwise dividing by a tiny pivot produces
    /// catastrophic error. See dev/plans/threshold-mismatch-fix.md.
    pub zero_tol: f64,
    /// 2×2 pivot block threshold (matches BunchKaufmanParams::zero_tol_2x2).
    pub zero_tol_2x2: f64,
}

/// Factor a symmetric indefinite matrix using Bunch-Kaufman pivoting.
/// Applies equilibration transparently before factoring.
pub fn factor(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    params: &BunchKaufmanParams,
) -> Result<(Factors, Inertia), FeralError> {
    matrix.validate()?;
    let n = matrix.n;

    // Apply equilibration
    let d_eq = crate::dense::equilibrate::equilibrate_scaling(matrix);

    // Copy the lower triangle into a working array, applying equilibration
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            a[j * n + i] = d_eq[i] * matrix.data[j * n + i] * d_eq[j];
        }
    }

    // Initialize permutation as identity
    let mut perm: Vec<usize> = (0..n).collect();

    // Storage for D block subdiagonal
    let mut subdiag = vec![0.0; n];

    // Inertia counts
    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut zero = 0usize;
    let mut needs_refinement = false;

    let alpha = params.alpha;
    let mut k = 0;

    // Fused update+argmax: the previous pivot's update computes γ₀ and r
    // for the next column, avoiding a redundant O(n) scan. On the first
    // iteration (or after a swap invalidates fused values), we fall back
    // to column_offdiag_max.
    let mut fused_gamma0 = 0.0f64;
    let mut fused_r = 0usize;
    let mut have_fused = false;

    while k < n {
        let remaining = n - k;

        if remaining == 1 {
            // Last pivot: always 1×1
            let d = a[k * n + k];
            if d.abs() <= params.zero_tol {
                match params.on_zero_pivot {
                    ZeroPivotAction::ForceAccept => {
                        needs_refinement = true;
                        zero += 1;
                    }
                    ZeroPivotAction::Fail => {
                        return Err(FeralError::NumericallyRankDeficient);
                    }
                    ZeroPivotAction::PerturbToEps { abs_floor } => {
                        let d_new = perturb_to_floor(d, abs_floor);
                        a[k * n + k] = d_new;
                        needs_refinement = true;
                        if d_new > 0.0 {
                            pos += 1;
                        } else {
                            neg += 1;
                        }
                    }
                }
            } else if matches!(params.on_zero_pivot, ZeroPivotAction::ForceAccept)
                && d.abs() <= params.null_pivot_tol
            {
                // F-01 rank-deficiency band on the trailing 1×1: count
                // zero, leave d intact for the solve.
                needs_refinement = true;
                zero += 1;
            } else if d > 0.0 {
                pos += 1;
            } else {
                neg += 1;
            }
            k += 1;
            continue;
        }

        // Step 1: Find γ₀ = max off-diagonal magnitude in column k
        // Use fused values from previous update when available.
        let (gamma0, r) = if have_fused {
            have_fused = false;
            (fused_gamma0, fused_r)
        } else {
            column_offdiag_max(&a, n, k)
        };

        if gamma0 == 0.0 {
            // Column is zero off-diagonal: 1×1 pivot (matrix reducible)
            count_1x1_inertia(
                &mut a,
                n,
                k,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            set_l_column_identity(&mut a, n, k);
            // No fused values — next column wasn't updated
            k += 1;
            continue;
        }

        // Step 3: Test if A[k,k] is acceptable as 1×1 pivot
        let akk = a[k * n + k].abs();
        if akk >= alpha * gamma0 {
            // Accept A[k,k] as 1×1 pivot, no swap — fused values are valid
            let (ng, nr) = do_1x1_pivot(
                &mut a,
                n,
                k,
                gamma0,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            fused_gamma0 = ng;
            fused_r = nr;
            have_fused = k + 1 < n;
            k += 1;
            continue;
        }

        // Step 4: Compute γᵣ = max off-diagonal magnitude in symmetric row/column r
        let gamma_r = symmetric_row_offdiag_max(&a, n, k, r);

        // Step 5: Test if A[r,r] is acceptable as 1×1 pivot (swap k↔r)
        let arr = a[r * n + r].abs();
        if arr >= alpha * gamma_r {
            // Swap invalidates any fused column — re-scan next iteration
            swap_rows_cols(&mut a, n, k, r, &mut perm);
            let (ng, nr) = do_1x1_pivot(
                &mut a,
                n,
                k,
                gamma_r,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            fused_gamma0 = ng;
            fused_r = nr;
            have_fused = k + 1 < n;
            k += 1;
            continue;
        }

        // Step 6: LAPACK extension — test if A[k,k] still usable
        if akk * gamma_r >= alpha * gamma0 * gamma0 {
            // No swap — fused values are valid
            let (ng, nr) = do_1x1_pivot(
                &mut a,
                n,
                k,
                gamma0,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            fused_gamma0 = ng;
            fused_r = nr;
            have_fused = k + 1 < n;
            k += 1;
            continue;
        }

        // Step 7: 2×2 pivot using rows/columns {k, r}
        if r != k + 1 {
            swap_rows_cols(&mut a, n, k + 1, r, &mut perm);
        }

        // Duff-Reid 2×2 growth bound (MUMPS dfac_front_aux.F:1599-1606).
        // See the corresponding comment in factor_frontal().
        let d11_v = a[k * n + k];
        let d21_v = a[k * n + (k + 1)];
        let d22_v = a[(k + 1) * n + (k + 1)];
        let det_v = d11_v * d22_v - d21_v * d21_v;
        let absdet = det_v.abs();
        let mut rmax = 0.0f64;
        let mut tmax = 0.0f64;
        for i in (k + 2)..n {
            let v0 = a[k * n + i].abs();
            if v0 > rmax {
                rmax = v0;
            }
            let v1 = a[(k + 1) * n + i].abs();
            if v1 > tmax {
                tmax = v1;
            }
        }
        let amax = d21_v.abs();
        let u = params.pivot_threshold;
        let growth_fail = (d22_v.abs() * rmax + amax * tmax) * u > absdet
            || (d11_v.abs() * tmax + amax * rmax) * u > absdet;

        if growth_fail {
            // 2×2 rejected by the Duff-Reid growth bound. Fall back to a
            // single 1×1 at k with the column-relative threshold. The
            // second position (k+1) is revisited on the next iteration.
            let (ng, nr) = do_1x1_pivot(
                &mut a,
                n,
                k,
                gamma0,
                params,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
            )?;
            fused_gamma0 = ng;
            fused_r = nr;
            have_fused = k + 1 < n;
            k += 1;
            continue;
        }

        let (ng, nr) = do_2x2_pivot(
            &mut a,
            n,
            k,
            &mut subdiag,
            params,
            &mut pos,
            &mut neg,
            &mut zero,
            &mut needs_refinement,
        )?;
        fused_gamma0 = ng;
        fused_r = nr;
        have_fused = k + 2 < n;
        k += 2;
    }

    // Extract L and D from the working array.
    // For 2×2 blocks, the off-diagonal a[k*n+(k+1)] is the D block subdiag
    // (already stored in subdiag), NOT an L entry. L entries for a 2×2 block
    // at {k, k+1} start at row k+2.
    let mut l = vec![0.0; n * n];
    let mut d_diag = vec![0.0; n];

    let mut j = 0;
    while j < n {
        d_diag[j] = a[j * n + j];
        l[j * n + j] = 1.0;

        if j + 1 < n && subdiag[j] != 0.0 {
            // 2×2 block at (j, j+1): L entries start at row j+2
            d_diag[j + 1] = a[(j + 1) * n + (j + 1)];
            l[(j + 1) * n + (j + 1)] = 1.0;
            for i in (j + 2)..n {
                l[j * n + i] = a[j * n + i];
                l[(j + 1) * n + i] = a[(j + 1) * n + i];
            }
            j += 2;
        } else {
            // 1×1 block: L entries start at row j+1
            for i in (j + 1)..n {
                l[j * n + i] = a[j * n + i];
            }
            j += 1;
        }
    }

    // Compute inverse permutation
    let mut perm_inv = vec![0usize; n];
    for (i, &p) in perm.iter().enumerate() {
        perm_inv[p] = i;
    }

    let inertia = Inertia::new(pos, neg, zero);

    flag_growth_for_refinement(&l, &mut needs_refinement);

    Ok((
        Factors {
            n,
            l,
            d_diag,
            d_subdiag: subdiag,
            perm,
            perm_inv,
            d_eq,
            needs_refinement,
            zero_tol: params.zero_tol,
            zero_tol_2x2: params.zero_tol_2x2,
        },
        inertia,
    ))
}

/// Factor a dense symmetric indefinite matrix by treating it as a single
/// fully-summed front and delegating to `factor_frontal(may_delay=false)`.
///
/// Unlike `factor()`, this entry point inherits `factor_frontal`'s safe
/// rejection fallback (via `try_reject_1x1_frontal`): when the 2×2
/// Duff-Reid growth bound fails, the kernel does not divide by a zero
/// pivot, and pivots below the column-relative threshold are either
/// accepted with their correct sign or force-zeroed, with iterative
/// refinement flagged.
///
/// Knight-Ruiz equilibration is applied before the factorization
/// (matching `factor()`'s preprocessing) and `d_eq` is carried on the
/// returned `Factors` for the solve to un-equilibrate.
///
/// Rationale: per `dev/research/task-19-dense-acopp30-expert-consultation.md`,
/// the dense `factor()` entry point is under-constrained for pathological
/// KKT matrices (natural order + u=0 + no `|det|==0` rejection). MUMPS
/// 5.8.2, SPRAL SSIDS, and faer all route such matrices through a single
/// multifrontal / frontal code path. This wrapper gives the bench and
/// other dense callers access to the same safe kernel the sparse path
/// uses, without needing a full symbolic analysis (no AMD/METIS).
pub fn factor_single_front(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    params: &BunchKaufmanParams,
) -> Result<(Factors, Inertia), FeralError> {
    matrix.validate()?;
    let n = matrix.n;

    let d_eq = crate::dense::equilibrate::equilibrate_scaling(matrix);

    // Build an equilibrated scratch SymmetricMatrix for factor_frontal.
    let mut eq_data = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            eq_data[j * n + i] = d_eq[i] * matrix.data[j * n + i] * d_eq[j];
        }
    }
    let eq_matrix = crate::dense::matrix::SymmetricMatrix { n, data: eq_data };

    let front = factor_frontal_blocked(&eq_matrix, n, false, params)?;

    // With may_delay=false and ncol=n, nelim==n, contrib is empty, and
    // the FrontalFactors fields map 1:1 to Factors plus d_eq.
    debug_assert_eq!(front.nelim, n);
    debug_assert_eq!(front.n_delayed, 0);
    debug_assert_eq!(front.contrib_dim, 0);

    let inertia = front.inertia;
    let factors = Factors {
        n,
        l: front.l,
        d_diag: front.d_diag,
        d_subdiag: front.d_subdiag,
        perm: front.perm,
        perm_inv: front.perm_inv,
        d_eq,
        needs_refinement: front.needs_refinement,
        zero_tol: front.zero_tol,
        zero_tol_2x2: front.zero_tol_2x2,
    };

    Ok((factors, inertia))
}

/// Result of partial frontal factorization for the multifrontal solver.
#[derive(Debug)]
pub struct FrontalFactors {
    /// Number of rows in the frontal (nrow).
    pub nrow: usize,
    /// Attempted column count (the `ncol` argument passed to `factor_frontal`).
    /// When `may_delay = true` and a pivot is rejected, the kernel may stop
    /// early with `nelim < ncol`; the leftover `ncol - nelim` columns are
    /// carried in the contribution block as delayed pivots. For the root
    /// supernode (`may_delay = false`) this always equals `nelim`.
    pub ncol: usize,
    /// Actually eliminated column count (`nelim ≤ ncol`). Solve loops use
    /// `nelim` as the upper bound of the D-block sweep.
    pub nelim: usize,
    /// L factor: nrow × nelim column-major. Unit diagonal (implicit).
    /// L[j*nrow + i] for i in [0, nrow), j in [0, nelim).
    pub l: Vec<f64>,
    /// D block diagonal (length nelim).
    pub d_diag: Vec<f64>,
    /// D block subdiagonal for 2×2 pivots (length nelim).
    pub d_subdiag: Vec<f64>,
    /// BK pivot permutation within the first nelim rows.
    /// perm[i] = j means original row j was moved to pivot position i.
    /// Only indices 0..nelim are permuted; nelim..nrow are identity.
    pub perm: Vec<usize>,
    /// Inverse permutation.
    pub perm_inv: Vec<usize>,
    /// Schur complement / delayed-pivot block: cdim × cdim column-major
    /// where cdim = nrow - nelim. Lower triangle only. For the first
    /// `ncol - nelim` positions this holds the un-eliminated (delayed)
    /// fully-summed columns; the remaining `nrow - ncol` positions hold
    /// the non-fully-summed trailing rows. When `nelim == ncol` the whole
    /// block is the classic Schur complement S = A22 - L21 * D * L21^T.
    ///
    /// The multifrontal driver may move this `Vec` into the parent's
    /// `ContribBlock` (W-3b) — production paths must read `contrib` only
    /// before storing the `FrontalFactors` into a `NodeFactors`. Direct
    /// callers of `factor_frontal*` (tests, examples, the bit-parity
    /// reference) see the populated `contrib` as documented.
    pub contrib: Vec<f64>,
    /// Dimension of the contribution block (`nrow - nelim`).
    pub contrib_dim: usize,
    /// Number of delayed fully-summed columns in the contribution block,
    /// i.e. `ncol - nelim`. These occupy positions `0..n_delayed` of the
    /// contrib block; positions `n_delayed..contrib_dim` are the
    /// non-fully-summed trailing rows.
    pub n_delayed: usize,
    /// Inertia of the `nelim` eliminated pivots.
    pub inertia: Inertia,
    /// Whether ForceAccept fired during factorization.
    pub needs_refinement: bool,
    /// Number of pivots rescued by rook search after BK-partial's column-
    /// relative threshold test rejected them (Phase 2.4.3). Zero on
    /// well-conditioned matrices. Aggregated per-front; the multifrontal
    /// driver sums across supernodes.
    pub n_rook_rescues: usize,
    /// 1×1 pivot threshold from BunchKaufmanParams (see Factors::zero_tol).
    pub zero_tol: f64,
    /// 2×2 pivot threshold from BunchKaufmanParams.
    pub zero_tol_2x2: f64,
}

/// Outcome of an attempt to accept a 1×1 pivot via `try_reject_1x1_frontal`.
/// The caller uses this to decide whether to continue, force-accept, or
/// break out of the BK loop (SSIDS-style delayed pivoting).
#[derive(Debug, Clone, Copy, PartialEq)]
enum PivotOutcome {
    /// Pivot clears the column-relative threshold; do the rank-1 update.
    Accepted,
    /// Pivot is below threshold; L column has been zeroed and the zero
    /// has been counted. Caller increments k and continues. Only produced
    /// when `may_delay == false`.
    Rejected,
    /// Pivot is below threshold; caller should `break` the BK loop and
    /// let the parent supernode retry this column. Only produced when
    /// `may_delay == true`. The kernel has not mutated any state for
    /// the failed pivot.
    Delayed,
    /// Rook rescue (Phase 2.4.3) found a 2×2 block pivot at `{k, k+1}`
    /// after applying symmetric swaps. The gates (SSIDS det floor,
    /// Duff-Reid growth bound) were checked inside `rook_rescue`. The
    /// caller must count 2×2 inertia, record `subdiag[k] = d21`, apply
    /// `do_2x2_update`, and advance `k` by 2. Emitted only by
    /// `try_reject_1x1_with_rook_rescue`; panel path never sees this.
    AcceptedRook2x2 { d11: f64, d21: f64, d22: f64 },
}

/// Result of one iteration of the scalar BK pivot loop in
/// `factor_frontal`. `scalar_pivot_step` is the extracted per-step body;
/// the caller translates `Advanced(n)` into `k += n` and `Delayed` into
/// a `break` to keep the pre-extraction control flow byte-identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PivotStepResult {
    /// Eliminated `n` columns (1 for a 1×1 pivot, 2 for a 2×2 block).
    Advanced(usize),
    /// Pivot was delayed (SSIDS-style); caller breaks the loop.
    Delayed,
}

/// Outcome of one `lblt_panel_frontal` invocation. The panel processes a
/// run of pure 1×1 pivots (no row/column swap, no 2×2) using faer-style
/// peek-ahead; any deviation from that path (2×2 candidate, swap
/// candidate, scalar-force condition) terminates the panel and asks the
/// caller to run `scalar_pivot_step` once before re-entering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PanelStatus {
    /// Panel eliminated `bs` 1×1 pivots cleanly.
    Full,
    /// Panel terminated early because the next pivot needed capabilities
    /// the panel doesn't support (2×2 or swap). Caller runs one scalar
    /// pivot step and may re-enter the panel afterwards. The panel
    /// peek-ahead'd column `k + n_elim` (one column) before bailing, so
    /// `apply_blocked_schur` must use `j_start = k + n_elim + 1` to
    /// avoid a double rank-1 update.
    ScalarFallback,
    /// Like `ScalarFallback`, but the panel ALSO peek-ahead'd column
    /// `k + n_elim + 1` while evaluating the no-swap 2×2 fast path
    /// (Phase A, dev/plans/dense-kernel-blas3.md). The 2×2 candidate
    /// failed one of the bail-out tests (swap-1×1 alternative,
    /// LAPACK-extension 1×1 alternative, growth bound, or det floor).
    /// Caller runs one scalar pivot step and uses
    /// `j_start = k + n_elim + 2` in `apply_blocked_schur` to avoid a
    /// double rank-1 update on either pre-updated column.
    ScalarFallbackPeekedNext,
    /// Panel terminated early because the next 1×1 pivot was rejected
    /// under the SSIDS `may_delay == true` contract — the parent
    /// supernode will absorb the delayed columns. Caller applies the
    /// deferred Schur update to trailing columns and breaks out of the
    /// outer loop (no further pivots in this supernode). Only produced
    /// when the caller passed `may_delay == true`.
    Delayed,
}

/// Factor a frontal matrix, eliminating only the first `ncol` columns.
///
/// This is the key dense kernel for the multifrontal solver. Unlike `factor()`,
/// pivot search is RESTRICTED to the first `ncol` rows/columns. Rows ncol..nrow
/// are never swapped into pivot positions, preserving their ordering for the
/// contribution block.
///
/// When `may_delay == true`, the first pivot that fails the column-relative
/// threshold (or the 2×2 Duff-Reid growth bound) causes the kernel to stop
/// early: the leftover `(ncol - nelim)` columns are carried forward in the
/// contribution block as delayed fully-summed columns. The SSIDS `ldlt_tpp`
/// kernel uses this "break on first failure" model — see
/// `dev/research/phase-2.3-delayed-pivoting.md` for the reference.
///
/// When `may_delay == false` (the root supernode), the existing
/// `ZeroPivotAction::ForceAccept` path handles failed pivots by zeroing the
/// L column and counting a zero pivot, exactly as before.
///
/// After eliminating `nelim` pivots, the `(nrow - nelim) × (nrow - nelim)`
/// trailing block of the working matrix is extracted as the contribution
/// block. When `nelim == ncol` this is the classic Schur complement
/// `S = A22 - L21 * D * L21^T`. When `nelim < ncol` the first
/// `(ncol - nelim)` rows/columns of that block are delayed fully-summed
/// columns. No equilibration is applied.
/// Diagnostic per-phase timing sink for `factor_frontal_with_profile`.
/// Populated only when a caller opts in; the `factor_frontal` wrapper
/// passes `None` so production paths are branchless. Used by
/// `src/bin/diag_leaf_profile.rs` (Phase 2.9.2 Step A) to sub-time
/// the kernel and decide whether the arena refactor is worthwhile.
#[doc(hidden)]
#[derive(Default, Debug, Clone, Copy)]
pub struct FrontalProfile {
    pub alloc_copy_ns: u128,
    pub setup_ns: u128,
    pub pivot_loop_ns: u128,
    pub extract_ns: u128,
    pub n_calls: u64,
}

pub fn factor_frontal(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
) -> Result<FrontalFactors, FeralError> {
    factor_frontal_with_profile(matrix, ncol, may_delay, params, None)
}

#[doc(hidden)]
pub fn factor_frontal_with_profile(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
    mut profile: Option<&mut FrontalProfile>,
) -> Result<FrontalFactors, FeralError> {
    matrix.validate()?;
    let nrow = matrix.n;

    if ncol > nrow {
        return Err(FeralError::InvalidInput(format!(
            "ncol {} > nrow {}",
            ncol, nrow
        )));
    }
    if ncol == 0 {
        return Ok(FrontalFactors {
            nrow,
            ncol: 0,
            nelim: 0,
            l: Vec::new(),
            d_diag: Vec::new(),
            d_subdiag: Vec::new(),
            perm: (0..nrow).collect(),
            perm_inv: (0..nrow).collect(),
            contrib: matrix.data.clone(),
            contrib_dim: nrow,
            n_delayed: 0,
            inertia: Inertia {
                positive: 0,
                negative: 0,
                zero: 0,
            },
            needs_refinement: false,
            n_rook_rescues: 0,
            zero_tol: params.zero_tol,
            zero_tol_2x2: params.zero_tol_2x2,
        });
    }

    // Phase: alloc + copy
    let t0 = profile.as_ref().map(|_| std::time::Instant::now());
    let mut a = vec![0.0; nrow * nrow];
    for j in 0..nrow {
        for i in j..nrow {
            a[j * nrow + i] = matrix.data[j * nrow + i];
        }
    }
    if let (Some(p), Some(t)) = (profile.as_deref_mut(), t0) {
        p.alloc_copy_ns += t.elapsed().as_nanos();
    }

    // Phase: setup (perm, subdiag, counters)
    let t0 = profile.as_ref().map(|_| std::time::Instant::now());
    let mut perm: Vec<usize> = (0..nrow).collect();
    let mut subdiag = vec![0.0; nrow];
    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut zero = 0usize;
    let mut needs_refinement = false;
    let mut n_rook_rescues = 0usize;
    if let (Some(p), Some(t)) = (profile.as_deref_mut(), t0) {
        p.setup_ns += t.elapsed().as_nanos();
    }

    let t_pivot = profile.as_ref().map(|_| std::time::Instant::now());
    let mut k = 0;
    // MAXFROMM cache: live across `scalar_pivot_step` calls within
    // this front. Always start empty (no prior pivot for `k=0`).
    let mut cached_maxfromm: Option<f64> = None;

    // Factor only the first ncol columns. Pivot search restricted to [k, ncol).
    // Per-step body is extracted as `scalar_pivot_step` so the Phase 2.4.1b
    // blocked-panel path can share the rejection/delay fallback with the
    // unblocked driver. This loop is behavior-preserving byte-for-byte vs
    // the pre-extraction body.
    while k < ncol {
        match scalar_pivot_step(
            &mut a,
            nrow,
            ncol,
            k,
            may_delay,
            params,
            &mut perm,
            &mut subdiag,
            &mut pos,
            &mut neg,
            &mut zero,
            &mut needs_refinement,
            &mut n_rook_rescues,
            &mut cached_maxfromm,
        )? {
            PivotStepResult::Advanced(n) => k += n,
            PivotStepResult::Delayed => break,
        }
    }

    if let (Some(p), Some(t)) = (profile.as_deref_mut(), t_pivot) {
        p.pivot_loop_ns += t.elapsed().as_nanos();
    }
    let t_extract = profile.as_ref().map(|_| std::time::Instant::now());

    // At the break point, `k` is the number of successfully eliminated
    // pivots. When `may_delay == false` this equals `ncol`; when
    // `may_delay == true` and a pivot was delayed, `nelim < ncol`.
    let nelim = k;
    let n_delayed = ncol - nelim;

    // Extract L (nrow × nelim), D diagonal, and contribution block
    let mut l = vec![0.0; nrow * nelim];
    let mut d_diag = vec![0.0; nelim];

    let mut j = 0;
    while j < nelim {
        d_diag[j] = a[j * nrow + j];
        l[j * nrow + j] = 1.0; // unit diagonal

        if j + 1 < nelim && subdiag[j] != 0.0 {
            // 2×2 block
            d_diag[j + 1] = a[(j + 1) * nrow + (j + 1)];
            l[(j + 1) * nrow + (j + 1)] = 1.0;
            for i in (j + 2)..nrow {
                l[j * nrow + i] = a[j * nrow + i];
                l[(j + 1) * nrow + i] = a[(j + 1) * nrow + i];
            }
            j += 2;
        } else {
            for i in (j + 1)..nrow {
                l[j * nrow + i] = a[j * nrow + i];
            }
            j += 1;
        }
    }

    // Extract contribution block: trailing (nrow-nelim) × (nrow-nelim) of a.
    // When `nelim < ncol` the first `n_delayed` rows/columns of this block
    // are delayed fully-summed columns; the remaining positions hold the
    // non-fully-summed trailing rows exactly as before.
    let cdim = nrow - nelim;
    let mut contrib = vec![0.0; cdim * cdim];
    for cj in 0..cdim {
        for ci in cj..cdim {
            contrib[cj * cdim + ci] = a[(nelim + cj) * nrow + (nelim + ci)];
        }
    }

    let mut perm_inv = vec![0usize; nrow];
    for (i, &p) in perm.iter().enumerate() {
        perm_inv[p] = i;
    }

    flag_growth_for_refinement(&l, &mut needs_refinement);

    let result = FrontalFactors {
        nrow,
        ncol,
        nelim,
        l,
        d_diag,
        d_subdiag: subdiag[..nelim].to_vec(),
        perm,
        perm_inv,
        contrib,
        contrib_dim: cdim,
        n_delayed,
        inertia: Inertia::new(pos, neg, zero),
        needs_refinement,
        n_rook_rescues,
        zero_tol: params.zero_tol,
        zero_tol_2x2: params.zero_tol_2x2,
    };
    if let (Some(p), Some(t)) = (profile, t_extract) {
        p.extract_ns += t.elapsed().as_nanos();
        p.n_calls += 1;
    }
    Ok(result)
}

/// Blocked-panel BK LDLᵀ variant of `factor_frontal` (Phase 2.4.1b).
///
/// **Status: Step 4b GREEN (peek-ahead panel + deferred Schur).**
/// Implements the faer-style blocked kernel described in
/// `dev/plans/phase-2.4.1-blocked-ldlt.md`: a panel processes up to
/// `params.block_size` 1×1 pivots. Before each pivot search, the
/// current column is updated via **replay** — pending rank-1 updates
/// from prior panel pivots are applied in ascending pivot index using
/// the same `schur_kernel::axpy_minus_unroll4_nofma` kernel scalar
/// uses, which makes the per-element accumulation order identical to
/// `factor_frontal`. After the panel, the deferred rank-1 updates are
/// applied to the remaining trailing columns in the same pivot-outer
/// order, again via the bit-exact axpy kernel.
///
/// **Bit-parity guarantee.** Since (i) replay traverses `(i, j)` with
/// updates applied in ascending pivot index — identical to scalar's
/// pivot-outer/column-inner loop — and (ii) both paths use the same
/// axpy kernel, scalar and blocked produce byte-identical
/// `(L, D, perm, inertia, contrib)`. Enforced by `tests/blocked_ldlt.rs`.
///
/// **Fallbacks.** Any of the following routes through `scalar_pivot_step`
/// instead of the panel:
/// - `may_delay == true` (SSIDS-style delayed pivoting — Step 5 target).
/// - `ncol <= params.block_size` (small-front scalar fast path).
/// - Panel encounters a 2×2 candidate (`akk < alpha * gamma0`) or any
///   other non-trivial BK branch — panel returns `ScalarFallback`,
///   caller runs one `scalar_pivot_step` then re-enters the panel.
///
/// The scalar oracle (`factor_frontal`) is retained for correctness
/// and serves as the bit-parity reference.
pub fn factor_frontal_blocked(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
) -> Result<FrontalFactors, FeralError> {
    matrix.validate()?;
    let nrow = matrix.n;
    // Clone the lower triangle into a fresh scratch buffer so the
    // in-place entry can factor without aliasing the caller's borrow.
    // Hot-path callers in the multifrontal driver should call
    // `factor_frontal_blocked_in_place` directly to skip this copy.
    // (W-3a from `dev/plans/dense-kernel-speedup.md`.)
    let mut scratch_data = vec![0.0; nrow * nrow];
    for j in 0..nrow {
        for i in j..nrow {
            scratch_data[j * nrow + i] = matrix.data[j * nrow + i];
        }
    }
    let mut scratch = crate::dense::matrix::SymmetricMatrix {
        n: nrow,
        data: scratch_data,
    };
    factor_frontal_blocked_in_place(&mut scratch, ncol, may_delay, params)
}

/// In-place blocked-panel BK LDLᵀ. Factors directly into `matrix.data`
/// (which is treated as scratch storage); on return the lower-triangle
/// content of `matrix.data` is undefined. Skips the input `validate()`
/// scan; the caller is responsible for ensuring the lower triangle is
/// finite. The multifrontal driver assembles fronts from a value-checked
/// CSC, so per-front re-validation is redundant.
///
/// W-3a from `dev/plans/dense-kernel-speedup.md`: eliminates the
/// `nrow * nrow` duplicate allocation + lower-triangle copy that the
/// pre-W-3a `factor_frontal_blocked` performed on every call. For
/// CHAINWOO_0000's 1984-row root supernode that copy is 30 MB
/// per supernode call.
pub fn factor_frontal_blocked_in_place(
    matrix: &mut crate::dense::matrix::SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
) -> Result<FrontalFactors, FeralError> {
    let mut scratch = FactorScratch::new();
    factor_frontal_blocked_in_place_with_scratch(matrix, ncol, may_delay, params, &mut scratch)
}

/// Issue #13 Phase A — pooled-scratch variant of
/// [`factor_frontal_blocked_in_place`]. The caller-supplied `scratch`
/// holds the `subdiag` and `d_panel` working buffers; repeated calls
/// across a single `FactorScratch` lifetime amortise those two
/// allocations per supernode. The returned `FrontalFactors` still
/// owns its `l`/`d_diag`/`d_subdiag`/`perm`/`perm_inv`/`contrib`
/// Vecs — phase C (deferred) would extend the pooling there.
///
/// Bit-exact with [`factor_frontal_blocked_in_place`]: the scratch
/// buffers are `clear()`+`resize()`d on entry so a warm scratch from
/// a prior call (even of a different `nrow`) is indistinguishable
/// from a fresh `FactorScratch::new()`.
pub fn factor_frontal_blocked_in_place_with_scratch(
    matrix: &mut crate::dense::matrix::SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
    scratch: &mut FactorScratch,
) -> Result<FrontalFactors, FeralError> {
    let nrow = matrix.n;

    if ncol > nrow {
        return Err(FeralError::InvalidInput(format!(
            "ncol {} > nrow {}",
            ncol, nrow
        )));
    }
    if ncol == 0 {
        return Ok(FrontalFactors {
            nrow,
            ncol: 0,
            nelim: 0,
            l: Vec::new(),
            d_diag: Vec::new(),
            d_subdiag: Vec::new(),
            perm: (0..nrow).collect(),
            perm_inv: (0..nrow).collect(),
            contrib: matrix.data.clone(),
            contrib_dim: nrow,
            n_delayed: 0,
            inertia: Inertia {
                positive: 0,
                negative: 0,
                zero: 0,
            },
            needs_refinement: false,
            n_rook_rescues: 0,
            zero_tol: params.zero_tol,
            zero_tol_2x2: params.zero_tol_2x2,
        });
    }

    // Phase 2.4.1c triage hook. When `FORCE_SCALAR_FRONTAL` is set,
    // delegate to `factor_frontal` unconditionally so a binary can
    // run the multifrontal driver with either kernel without
    // patching call sites. This is a diagnostic flag only — the
    // default (false) preserves the production dispatch.
    if FORCE_SCALAR_FRONTAL.load(std::sync::atomic::Ordering::Relaxed) {
        return factor_frontal(matrix, ncol, may_delay, params);
    }

    // Issue #9 Step 2 dispatch: 32×32 fully-summed fronts go through
    // `factor_block32`. That function delegates to `factor_frontal`,
    // whose eager `do_1x1_update` / `do_2x2_update` now route to the
    // block-32 SIMD body (`update_1x1_block32` quad/dual/single tiling)
    // at n==32. The panel path (`lblt_panel_frontal`) was leaving the
    // batched-source quad kernel unused at bs==ncol==32 because
    // `j_start = k + n_elim == nrow` skips `apply_blocked_schur_panel`;
    // the eager-update path uses the quad kernel for every trailing
    // tile of 4 columns. Bit-parity: factor_frontal is the documented
    // oracle for both lblt_panel_frontal and the block-32 SIMD body.
    if nrow == crate::dense::block_ldlt32::BLOCK_SIZE
        && ncol == crate::dense::block_ldlt32::BLOCK_SIZE
    {
        return crate::dense::block_ldlt32::factor_block32(matrix, ncol, may_delay, params);
    }

    // Fallback conditions where the panel offers no advantage.
    // Delegation preserves parity trivially.
    //
    // W-1 (`dev/plans/dense-kernel-speedup.md`): engage the deferred-Schur
    // panel for any `ncol >= PANEL_MIN_NCOL`. Previously the gate was
    // `ncol > bs` (where `bs = params.block_size`, default 64), which
    // sent every 32×32 CHAINWOO root supernode (62% of factor time on
    // CHAINWOO_0000) through the scalar `factor_frontal` path. The
    // panel kernel `lblt_panel_frontal` already handles small-`bs`
    // dispatches; we widen the gate and clamp the working block size
    // to `min(params.block_size, ncol)` so a single panel pass covers
    // the entire elimination range when `ncol <= params.block_size`.
    const PANEL_MIN_NCOL: usize = 8;
    let bs = params.block_size.min(ncol);
    if bs < 2 || ncol < PANEL_MIN_NCOL {
        return factor_frontal(matrix, ncol, may_delay, params);
    }

    // Factor in place into the caller's buffer — the kernel reads and
    // writes only the lower triangle (strict upper is never touched),
    // so reusing `matrix.data` is safe. After the panel loop and
    // L/D/contrib extract phases we discard `matrix.data` content.
    let a: &mut [f64] = matrix.data.as_mut_slice();

    let mut perm: Vec<usize> = (0..nrow).collect();
    // Issue #13 Phase A: pooled `subdiag` and `d_panel` from caller-supplied
    // scratch. `clear()` + `resize()` preserves capacity across calls so a
    // warm scratch (even with a larger prior `nrow`/`bs`) reuses the heap
    // allocation. Zero-initialised — both buffers are read before write in
    // the may_delay-rejection and panel-bail paths.
    scratch.subdiag.clear();
    scratch.subdiag.resize(nrow, 0.0);
    scratch.d_panel.clear();
    scratch.d_panel.resize(bs, 0.0);
    let subdiag: &mut [f64] = scratch.subdiag.as_mut_slice();
    let d_panel: &mut [f64] = scratch.d_panel.as_mut_slice();
    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut zero = 0usize;
    let mut needs_refinement = false;
    let mut n_rook_rescues = 0usize;
    // MAXFROMM cache: live across both scalar-tail steps and the
    // post-panel fallback step within this front. Panel paths do not
    // populate it (the panel processes its own AMAX); the cache is
    // simply None after a panel pass, which forces the next scalar
    // step to do a full scan exactly as it does today.
    let mut cached_maxfromm: Option<f64> = None;

    let mut k = 0;
    while k < ncol {
        let remaining = ncol - k;
        // Scalar tail engages when too few columns are left to amortize
        // the deferred-Schur dispatch. With W-1 the first panel may
        // process all `ncol` columns when `ncol <= bs`; subsequent
        // iterations fall into the scalar tail only once fewer than
        // `PANEL_MIN_NCOL` columns remain (matching the entry gate).
        if remaining < PANEL_MIN_NCOL {
            // Scalar tail: process remaining pivots one at a time.
            // Reborrow `a` (a `&mut [f64]`) so the same binding can be
            // passed across multiple call sites.
            diag_inc(&panel_diag::SCALAR_TAIL_STEPS);
            match scalar_pivot_step(
                &mut *a,
                nrow,
                ncol,
                k,
                may_delay,
                params,
                &mut perm,
                &mut *subdiag,
                &mut pos,
                &mut neg,
                &mut zero,
                &mut needs_refinement,
                &mut n_rook_rescues,
                &mut cached_maxfromm,
            )? {
                PivotStepResult::Advanced(n) => {
                    diag_add(&panel_diag::PIVOTS_SCALAR, n as u64);
                    k += n;
                }
                PivotStepResult::Delayed => break,
            }
            continue;
        }

        // Clamp the panel cap to the remaining elimination range so the
        // last partial panel never peeks past column `ncol - 1`. Without
        // this clamp, when `ncol % bs != 0` the last panel call would
        // read/write columns in the contribution-block region.
        let panel_cap = bs.min(remaining);
        // The panel rewrites column `k` and beyond; any cached MAXFROMM
        // from a prior scalar fallback no longer describes the post-panel
        // column. Clear so the next scalar step falls back to a full
        // AMAX scan exactly as Plain would.
        cached_maxfromm = None;
        let (n_elim, status) = lblt_panel_frontal(
            &mut *a,
            nrow,
            ncol,
            k,
            panel_cap,
            may_delay,
            params,
            &mut pos,
            &mut neg,
            &mut zero,
            &mut needs_refinement,
            &mut *d_panel,
            &mut *subdiag,
            &mut perm,
        )?;
        // On ScalarFallback and Delayed the panel peek-ahead'd column
        // `k+n_elim` (applied pivots 0..n_elim-1 to it) before deciding
        // it could not pivot. In scalar semantics that column's state at
        // break time already matches what pivots 0..n_elim-1 produce via
        // eager updates, so `apply_blocked_schur` must skip it to avoid
        // a double rank-1 update. On Full the column at `k+n_elim` was
        // not peek-ahead'd, so the deferred update starts there normally.
        let j_start = match status {
            PanelStatus::Full => k + n_elim,
            PanelStatus::ScalarFallback | PanelStatus::Delayed => k + n_elim + 1,
            // Phase A (dev/plans/dense-kernel-blas3.md): the no-swap
            // 2×2 inline path peek-ahead'd col+1 before bailing, so
            // BOTH col=k+n_elim and col+1=k+n_elim+1 already carry the
            // deferred updates from pivots 0..n_elim-1. Skip them
            // both to avoid a double rank-1 update.
            PanelStatus::ScalarFallbackPeekedNext => k + n_elim + 2,
        };
        apply_blocked_schur(
            &mut *a, nrow, k, n_elim, j_start, &*d_panel, &*subdiag, params.fma,
        );
        k += n_elim;

        // Phase B-1.5 attribution: count panel outcome and committed pivots.
        diag_add(&panel_diag::PIVOTS_INLINE, n_elim as u64);
        match status {
            PanelStatus::Full => diag_inc(&panel_diag::PANEL_FULL),
            PanelStatus::ScalarFallback | PanelStatus::ScalarFallbackPeekedNext => {
                diag_inc(&panel_diag::PANEL_PARTIAL)
            }
            PanelStatus::Delayed => diag_inc(&panel_diag::PANEL_DELAYED),
        }

        match status {
            PanelStatus::Full => {}
            PanelStatus::ScalarFallback | PanelStatus::ScalarFallbackPeekedNext => {
                // One scalar step to handle the 2×2/swap case the panel declined.
                if k >= ncol {
                    break;
                }
                match scalar_pivot_step(
                    &mut *a,
                    nrow,
                    ncol,
                    k,
                    may_delay,
                    params,
                    &mut perm,
                    &mut *subdiag,
                    &mut pos,
                    &mut neg,
                    &mut zero,
                    &mut needs_refinement,
                    &mut n_rook_rescues,
                    &mut cached_maxfromm,
                )? {
                    PivotStepResult::Advanced(n) => {
                        diag_add(&panel_diag::PIVOTS_SCALAR, n as u64);
                        k += n;
                    }
                    PivotStepResult::Delayed => break,
                }
            }
            PanelStatus::Delayed => break,
        }
    }

    let nelim = k;
    let n_delayed = ncol - nelim;

    // Extract L, D, contrib — identical logic to `factor_frontal`.
    let mut l = vec![0.0; nrow * nelim];
    let mut d_diag = vec![0.0; nelim];

    let mut j = 0;
    while j < nelim {
        d_diag[j] = a[j * nrow + j];
        l[j * nrow + j] = 1.0;

        if j + 1 < nelim && subdiag[j] != 0.0 {
            d_diag[j + 1] = a[(j + 1) * nrow + (j + 1)];
            l[(j + 1) * nrow + (j + 1)] = 1.0;
            for i in (j + 2)..nrow {
                l[j * nrow + i] = a[j * nrow + i];
                l[(j + 1) * nrow + i] = a[(j + 1) * nrow + i];
            }
            j += 2;
        } else {
            for i in (j + 1)..nrow {
                l[j * nrow + i] = a[j * nrow + i];
            }
            j += 1;
        }
    }

    // Phase C: take the recyclable buffer from the scratch pool (single
    // slot). `clear()` resets logical length while preserving capacity;
    // `resize(_, 0.0)` zero-fills back to `cdim*cdim` so the cell writes
    // below see the same starting state as `vec![0.0; cdim*cdim]` did.
    // When the slot is empty, `take()` returns `None` and
    // `unwrap_or_default()` gives a fresh empty Vec — bit-identical to
    // the previous `vec![0.0; ...]` path.
    let cdim = nrow - nelim;
    let mut contrib = scratch.contrib_pool.take().unwrap_or_default();
    contrib.clear();
    contrib.resize(cdim * cdim, 0.0);
    for cj in 0..cdim {
        for ci in cj..cdim {
            contrib[cj * cdim + ci] = a[(nelim + cj) * nrow + (nelim + ci)];
        }
    }

    let mut perm_inv = vec![0usize; nrow];
    for (i, &p) in perm.iter().enumerate() {
        perm_inv[p] = i;
    }

    flag_growth_for_refinement(&l, &mut needs_refinement);

    Ok(FrontalFactors {
        nrow,
        ncol,
        nelim,
        l,
        d_diag,
        d_subdiag: subdiag[..nelim].to_vec(),
        perm,
        perm_inv,
        contrib,
        contrib_dim: cdim,
        n_delayed,
        inertia: Inertia::new(pos, neg, zero),
        needs_refinement,
        n_rook_rescues,
        zero_tol: params.zero_tol,
        zero_tol_2x2: params.zero_tol_2x2,
    })
}

// =============================================================
// SQD (Symmetric Quasi-Definite) fast-path — issue #34
// =====================================================
//
// Vanderbei (1995) Theorem 2.1: every symmetric permutation of
// K = [[-E, A^T], [A, F]] with E, F ≻ 0 admits an LDL^T with purely
// diagonal D. The caller asserts the contract by opting in via
// Solver::with_sqd_mode(true); these kernels skip the Bunch-Kaufman
// 1x1-vs-2x2 selection and run a pure diagonal-pivot loop. Mutually
// exclusive with delayed pivoting (the builder enforces).
//
// Contract enforcement (phase e): each pivot is checked against two
// bounds derived from Gill-Saunders-Shinnerl 1996:
//   (i)  |d_kk|       > zero_tol        (near-zero guard)
//   (ii) max |l_{ik}| <= 1 / sqrt(EPS)  (column-growth guard,
//        ≈ 6.7e7 in f64)
// A trip on either surfaces `FeralError::SqdContractViolated
// { column, pivot }` immediately — never a silent BK fallback.
//
// References: dev/research/sqd-fast-path.md, dev/decisions.md
// 2026-05-16 entry, issue #34.

/// SQD fast-path top-level entry — analog of [`factor`] for a caller
/// who has asserted the Vanderbei (1995) quasi-definite contract.
///
/// Returns the same `Factors` shape as [`factor`] with `d_subdiag`
/// all zero (diagonal D), so existing solve paths handle the result
/// without modification. Equilibration is applied identically to
/// [`factor`].
///
/// Contract violation: `|d| <= params.zero_tol` or the column-growth
/// bound `max |l_{ik}| > 1/sqrt(EPS)` at any column k returns
/// `Err(FeralError::SqdContractViolated { column: k, pivot: d })`.
/// `params.on_zero_pivot` is *ignored*: SQD treats `zero_tol` as a
/// hard contract trip, not a force-accept threshold.
pub fn factor_diagonal(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    params: &BunchKaufmanParams,
) -> Result<(Factors, Inertia), FeralError> {
    matrix.validate()?;
    let n = matrix.n;

    let d_eq = crate::dense::equilibrate::equilibrate_scaling(matrix);

    let mut scratch_data = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            scratch_data[j * n + i] = d_eq[i] * matrix.data[j * n + i] * d_eq[j];
        }
    }
    let mut scratch = crate::dense::matrix::SymmetricMatrix {
        n,
        data: scratch_data,
    };

    let frontal = factor_frontal_diagonal_in_place(&mut scratch, n, params)?;

    // Reshape FrontalFactors into top-level Factors. SQD eliminates
    // every column (nelim == ncol == nrow == n) so the contrib block
    // is empty and perm is identity.
    debug_assert_eq!(frontal.nelim, n, "SQD must eliminate every column");
    debug_assert_eq!(frontal.contrib_dim, 0, "SQD leaves no contribution");
    Ok((
        Factors {
            n,
            l: frontal.l,
            d_diag: frontal.d_diag,
            d_subdiag: frontal.d_subdiag,
            perm: frontal.perm,
            perm_inv: frontal.perm_inv,
            d_eq,
            needs_refinement: frontal.needs_refinement,
            zero_tol: params.zero_tol,
            zero_tol_2x2: params.zero_tol_2x2,
        },
        frontal.inertia,
    ))
}

/// SQD fast-path supernode kernel — analog of
/// [`factor_frontal_blocked_in_place_with_scratch`] for the
/// diagonal-D contract. Factors the first `ncol` columns of
/// `matrix.data` (column-major lower triangle) into `(L, D)` with
/// purely diagonal `D`, leaving the trailing
/// `(nrow - ncol) × (nrow - ncol)` Schur complement in the
/// contribution block. The shared rank-1 trailing-update kernel
/// `do_1x1_update` is reused unchanged — only the per-pivot driver
/// differs (no `column_offdiag_max`, no `symmetric_row_offdiag_max`,
/// no `try_reject_1x1_frontal`, no 2x2 branch).
///
/// `may_delay` is implicitly `false` (the SQD-vs-delayed builder
/// invariant is enforced one level up by `Solver::with_sqd_mode`);
/// the kernel has no delayed-pivot machinery.
pub fn factor_frontal_diagonal_in_place(
    matrix: &mut crate::dense::matrix::SymmetricMatrix,
    ncol: usize,
    params: &BunchKaufmanParams,
) -> Result<FrontalFactors, FeralError> {
    let nrow = matrix.n;

    if ncol > nrow {
        return Err(FeralError::InvalidInput(format!(
            "ncol {} > nrow {}",
            ncol, nrow
        )));
    }
    if ncol == 0 {
        return Ok(FrontalFactors {
            nrow,
            ncol: 0,
            nelim: 0,
            l: Vec::new(),
            d_diag: Vec::new(),
            d_subdiag: Vec::new(),
            perm: (0..nrow).collect(),
            perm_inv: (0..nrow).collect(),
            contrib: matrix.data.clone(),
            contrib_dim: nrow,
            n_delayed: 0,
            inertia: Inertia {
                positive: 0,
                negative: 0,
                zero: 0,
            },
            needs_refinement: false,
            n_rook_rescues: 0,
            zero_tol: params.zero_tol,
            zero_tol_2x2: params.zero_tol_2x2,
        });
    }

    let a = matrix.data.as_mut_slice();

    let mut pos = 0usize;
    let mut neg = 0usize;
    // SQD treats `zero_tol` as a contract trip, so the FMA flag is
    // the only `params` field that genuinely matters for the inner
    // kernel. Cache it once.
    let fma = params.fma;

    // Diagonal-only BK substitute. For each pivot:
    //   1. d = a[k,k]; if |d| <= zero_tol, contract violated.
    //   2. Rank-1 update the trailing block via the shared
    //      `do_1x1_update` kernel (also scales L[:, k] by 1/d).
    //   3. Check the L-column growth bound (Gill-Saunders-Shinnerl
    //      1996): max_{i>k} |l_{ik}| <= 1/sqrt(EPS) ≈ 6.7e7. A
    //      violation means the diagonal pivot was too small relative
    //      to its column even though it cleared `zero_tol` — abort
    //      with SqdContractViolated rather than carry forward a
    //      back-error blow-up.
    //   4. Count inertia from the sign of d.
    let l_growth_bound = 1.0 / f64::EPSILON.sqrt();
    for k in 0..ncol {
        let d = a[k * nrow + k];
        if d.abs() <= params.zero_tol {
            return Err(FeralError::SqdContractViolated {
                column: k,
                pivot: d,
            });
        }
        do_1x1_update(a, nrow, k, fma);
        // Post-update L-growth check: a[i, k] for i > k now holds l_{ik}.
        let col_base = k * nrow;
        let mut max_l: f64 = 0.0;
        for i in (k + 1)..nrow {
            let v = a[col_base + i].abs();
            if v > max_l {
                max_l = v;
            }
        }
        if max_l > l_growth_bound {
            return Err(FeralError::SqdContractViolated {
                column: k,
                pivot: d,
            });
        }
        if d > 0.0 {
            pos += 1;
        } else {
            neg += 1;
        }
    }

    // Extract L (nrow × ncol), D diagonal, and contribution block —
    // the shape matches `factor_frontal_blocked_in_place_with_scratch`
    // so downstream consumers (solver, contribution-block assembler)
    // see no structural difference.
    let nelim = ncol;
    let mut l = vec![0.0; nrow * nelim];
    let mut d_diag = vec![0.0; nelim];
    for j in 0..nelim {
        d_diag[j] = a[j * nrow + j];
        l[j * nrow + j] = 1.0;
        for i in (j + 1)..nrow {
            l[j * nrow + i] = a[j * nrow + i];
        }
    }

    let cdim = nrow - nelim;
    let mut contrib = vec![0.0; cdim * cdim];
    for cj in 0..cdim {
        for ci in cj..cdim {
            contrib[cj * cdim + ci] = a[(nelim + cj) * nrow + (nelim + ci)];
        }
    }

    let perm: Vec<usize> = (0..nrow).collect();
    let perm_inv = perm.clone();

    let mut needs_refinement = false;
    flag_growth_for_refinement(&l, &mut needs_refinement);

    Ok(FrontalFactors {
        nrow,
        ncol,
        nelim,
        l,
        d_diag,
        d_subdiag: vec![0.0; nelim],
        perm,
        perm_inv,
        contrib,
        contrib_dim: cdim,
        n_delayed: 0,
        inertia: Inertia::new(pos, neg, 0),
        needs_refinement,
        n_rook_rescues: 0,
        zero_tol: params.zero_tol,
        zero_tol_2x2: params.zero_tol_2x2,
    })
}

/// Process one blocked panel of up to `bs` pure 1×1 pivots starting at
/// global column `k`. Applies per-column peek-ahead (replay of pending
/// rank-1 updates from prior panel pivots) before each pivot search so
/// the BK test sees the same column state scalar would. Terminates early
/// on any condition the panel cannot handle without a full-state view
/// (2×2 candidate, swap candidate, or — when `may_delay == true` — a
/// delayed pivot from the SSIDS threshold test).
///
/// On return, `a[k..k+n_elim]` columns hold scaled L columns (or zeroed
/// L columns for rejected pivots), `d_panel[0..n_elim]` holds the
/// pre-scaling diagonals (or 0 for rejected/zero-gamma0 pivots).
/// Columns `[k+n_elim, nrow)` are stale — the caller must apply
/// `apply_blocked_schur` before running further pivot searches or
/// extracting the contribution block.
#[allow(clippy::too_many_arguments)]
fn lblt_panel_frontal(
    a: &mut [f64],
    nrow: usize,
    ncol: usize,
    k: usize,
    bs: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
    d_panel: &mut [f64],
    subdiag: &mut [f64],
    // Phase A2 (`dev/plans/dense-kernel-w2-2x2-swap.md`): the panel
    // owns the swap-required 2×2 path; it commits row/col swaps via
    // `swap_rows_cols(a, nrow, col+1, r, perm)` and the change must
    // be visible to the caller's L extract step.
    perm: &mut [usize],
) -> Result<(usize, PanelStatus), FeralError> {
    let alpha_bk = params.alpha;
    let cap = bs;
    let mut c = 0usize;
    while c < cap {
        let col = k + c;

        // Peek-ahead: apply pending rank-1 (and rank-2 for accepted
        // panel-internal 2×2's) updates from pivots 0..c.
        peek_ahead_column(a, nrow, k, c, d_panel, subdiag, params.fma);

        // Compute gamma0 + argmax row over rows (col+1)..nrow
        // (unrestricted — matches scalar's gamma0 search, which
        // includes rows ncol..nrow for the BK test even in frontal
        // mode). The argmax row `r` is needed for the no-swap 2×2
        // fast path below.
        let mut gamma0 = 0.0f64;
        let mut r = col + 1;
        for i in (col + 1)..nrow {
            let v = a[col * nrow + i].abs();
            if v > gamma0 {
                gamma0 = v;
                r = i;
            }
        }

        if gamma0 == 0.0 {
            // Zero-column: matches scalar's gamma0==0 branch exactly.
            count_1x1_inertia(a, nrow, col, params, pos, neg, zero, needs_refinement)?;
            set_l_column_identity(a, nrow, col);
            // The L column is all zeros (below diagonal); the diagonal is
            // unchanged (or perturbed in place by `count_1x1_inertia`
            // under `PerturbToEps`). Subsequent replay with
            // alpha = (0 * d) = 0 is a no-op, matching scalar.
            d_panel[c] = a[col * nrow + col];
            c += 1;
            continue;
        }

        let akk = a[col * nrow + col].abs();
        if akk < alpha_bk * gamma0 {
            // 2×2 candidate. Try the no-swap 2×2 fast path inline
            // (Phase A of dev/plans/dense-kernel-blas3.md). The
            // conditions to accept inline mirror scalar_pivot_step's
            // 2×2 branch but require zero swap and zero rejection:
            //
            //   - argmax row `r` is exactly `col + 1` (so the rank-2
            //     pivot is over consecutive columns; no swap with a
            //     fully-summed row needed)
            //   - `col + 1 < ncol` (the 2×2 stays inside the
            //     fully-summed range)
            //   - `c + 1 < cap` (panel has room for two slots)
            //   - swap-1×1 alternative fails (`arr < alpha * gamma_r`
            //     where r = col+1)
            //   - LAPACK-extension 1×1 alternative fails
            //     (`akk * gamma_r < alpha * gamma0^2`)
            //   - Duff-Reid growth bound passes
            //   - SSIDS scale-invariant det floor passes
            //
            // Any condition fails → bail to scalar_pivot_step which
            // handles all the swap/rejection/rook-rescue branches the
            // panel deliberately does not.
            // Phase A2 (`dev/plans/dense-kernel-w2-2x2-swap.md`):
            // handle the swap-required 2×2 case (`r > col + 1`) inline
            // when `c == 0`. At `c == 0` no pivots are committed, so
            // the deferred state is identical to the scalar state and
            // peek-ahead is a no-op. Reading `arr` at the original `r`
            // and computing `gamma_r` over the un-mutated row matches
            // scalar's pre-swap reads bit-for-bit. Mid-panel
            // (`c > 0`) swap-2×2 still bails to scalar — it requires
            // peek-ahead-of-r plus a row-r-replay primitive that is
            // out of scope for A2.
            let need_swap = r > col + 1;
            // Scalar requires `r < ncol` for any swap (line 2120
            // `r_is_fully_summed = r < ncol`); the swap-2×2 path is
            // gated on the same predicate at line 2167. The no-swap
            // path doesn't need this check (r == col+1 < ncol already).
            let bounds_ok = col + 1 < ncol
                && c + 1 < cap
                && (!need_swap || r < ncol)
                && !DISABLE_PANEL_INLINE_2X2.load(std::sync::atomic::Ordering::Relaxed);
            let allowed = bounds_ok && (!need_swap || c == 0);
            if !allowed {
                diag_inc(&panel_diag::FALLBACK_2X2_NEED_SWAP_OR_BOUND);
                return Ok((c, PanelStatus::ScalarFallback));
            }

            if need_swap {
                // c == 0: state is scalar-equivalent; read arr/gamma_r
                // at the original `r` row, then bail or commit swap.
                // Bail returns `ScalarFallback` (no peek-ahead, no
                // swap committed) — caller resumes at `j_start = k`
                // and scalar redoes gamma0/r/swap identically.
                let gamma_r_pre = symmetric_row_offdiag_max(a, nrow, col, r);
                let arr_pre = a[r * nrow + r].abs();
                if arr_pre >= alpha_bk * gamma_r_pre {
                    diag_inc(&panel_diag::FALLBACK_2X2_SWAP_1X1_WINS);
                    return Ok((c, PanelStatus::ScalarFallback));
                }
                if akk * gamma_r_pre >= alpha_bk * gamma0 * gamma0 {
                    diag_inc(&panel_diag::FALLBACK_2X2_LAPACK_1X1_WINS);
                    return Ok((c, PanelStatus::ScalarFallback));
                }
                // Commit the symmetric swap. After this, the state at
                // column `col` is identical to scalar's post-swap
                // state: a[col, col+1] holds the old `gamma0`, the
                // (col+1, col+1) diagonal holds the old `arr`, and
                // perm records the col+1 ↔ r swap.
                swap_rows_cols(a, nrow, col + 1, r, perm);
                // Fall through to the no-swap accept path; growth/det
                // bail below uses `ScalarFallback` (no peek-ahead) —
                // scalar restarts at `col`, sees the swapped state
                // (its r will now be col+1), redoes the same checks,
                // and lands in may_delay/ForceAccept identically.
            } else {
                // No-swap path: peek-ahead col+1 with committed pivots
                // 0..c-1. Scalar's eager updates have already mutated
                // col+1 by this point (rank-1 from each prior pivot,
                // rank-2 from any prior panel-internal 2×2). The
                // deferred panel state has not. Replay onto col+1 so
                // the upcoming reads of a[r_idx*..], gamma_r,
                // growth-bound terms (rmax/tmax), and the L scaling
                // step see the same state scalar would.
                //
                // CRITICAL: any bail-out in this branch MUST return
                // `ScalarFallbackPeekedNext` (not `ScalarFallback`)
                // so the caller knows to use `j_start = k + n_elim + 2`
                // in `apply_blocked_schur` — col+1's deferred updates
                // have already been applied here.
                let r_idx = col + 1;
                peek_ahead_replay(a, nrow, k, c, r_idx, d_panel, subdiag, params.fma);
                let gamma_r = symmetric_row_offdiag_max(a, nrow, col, r_idx);
                let arr = a[r_idx * nrow + r_idx].abs();
                if arr >= alpha_bk * gamma_r {
                    diag_inc(&panel_diag::FALLBACK_2X2_SWAP_1X1_WINS);
                    return Ok((c, PanelStatus::ScalarFallbackPeekedNext));
                }
                if akk * gamma_r >= alpha_bk * gamma0 * gamma0 {
                    diag_inc(&panel_diag::FALLBACK_2X2_LAPACK_1X1_WINS);
                    return Ok((c, PanelStatus::ScalarFallbackPeekedNext));
                }
            }

            // 2×2 accepted at (col, col+1) with no swap. Apply the
            // same growth + det-floor checks scalar applies.
            let d11 = a[col * nrow + col];
            let d21 = a[col * nrow + (col + 1)];
            let d22 = a[(col + 1) * nrow + (col + 1)];
            let det = d11 * d22 - d21 * d21;

            // Duff-Reid 2×2 growth bound — same predicate as
            // scalar_pivot_step:1755-1756.
            let mut rmax = 0.0f64;
            let mut tmax = 0.0f64;
            for i in (col + 2)..nrow {
                let v0 = a[col * nrow + i].abs();
                if v0 > rmax {
                    rmax = v0;
                }
                let v1 = a[(col + 1) * nrow + i].abs();
                if v1 > tmax {
                    tmax = v1;
                }
            }
            let amax = d21.abs();
            let absdet = det.abs();
            let u = params.pivot_threshold;
            let growth_fail = (d22.abs() * rmax + amax * tmax) * u > absdet
                || (d11.abs() * tmax + amax * rmax) * u > absdet;

            // SSIDS scale-invariant det floor — same predicate as
            // scalar_pivot_step:1776-1788.
            let max_piv = d11.abs().max(d21.abs()).max(d22.abs());
            let det_floor_fail = if max_piv < SSIDS_DET_SMALL {
                true
            } else {
                let det_scale = 1.0 / max_piv;
                let detpiv0 = (d11 * det_scale) * d22;
                let detpiv1 = (d21 * det_scale) * d21;
                let detpiv = detpiv0 - detpiv1;
                let cancel_floor = SSIDS_DET_SMALL
                    .max(detpiv0.abs() * 0.5)
                    .max(detpiv1.abs() * 0.5);
                detpiv.abs() < cancel_floor
            };

            if growth_fail || det_floor_fail {
                // Rejection means scalar will run its
                // may_delay/ForceAccept branches; the panel can't
                // reproduce that here without duplicating bookkeeping.
                diag_inc(&panel_diag::FALLBACK_2X2_GROWTH_OR_DET);
                if need_swap {
                    // Swap branch: no peek-ahead was done. Scalar
                    // restarts at `k + n_elim`; its own gamma0 search
                    // on the post-swap state finds argmax at col+1
                    // (so `r == col+1`, no second swap) and lands in
                    // the same growth/det reject branch.
                    return Ok((c, PanelStatus::ScalarFallback));
                }
                // No-swap branch: col+1 was peek-ahead'd above —
                // caller must skip it (`j_start = k + n_elim + 2`).
                return Ok((c, PanelStatus::ScalarFallbackPeekedNext));
            }

            // Inline accept: record D values, scale L block, update
            // inertia, defer the rank-2 trailing update.
            if need_swap {
                diag_inc(&panel_diag::INLINE_2X2_SWAP_OK);
            }
            let pivot_inertia = count_2x2_inertia_val(d11, d21, d22);
            *pos += pivot_inertia.positive;
            *neg += pivot_inertia.negative;
            *zero += pivot_inertia.zero;

            d_panel[c] = d11;
            d_panel[c + 1] = d22;
            subdiag[k + c] = d21;

            // Scale the L block — same operation as do_2x2_update's
            // first loop (factor.rs:2075-2080). The trailing rank-2
            // update is deferred; replay happens via peek_ahead and
            // apply_blocked_schur honoring `subdiag[k+c] != 0`.
            if det.abs() != 0.0 {
                let inv_det = 1.0 / det;
                for i in (col + 2)..nrow {
                    let a_ik = a[col * nrow + i];
                    let a_ik1 = a[(col + 1) * nrow + i];
                    a[col * nrow + i] = (d22 * a_ik - d21 * a_ik1) * inv_det;
                    a[(col + 1) * nrow + i] = (d11 * a_ik1 - d21 * a_ik) * inv_det;
                }
            }

            c += 2;
            continue;
        }

        // 1×1 pivot at col, no swap. Try the column-relative threshold.
        let outcome = try_reject_1x1_frontal(
            a,
            nrow,
            col,
            gamma0,
            may_delay,
            params,
            pos,
            neg,
            zero,
            needs_refinement,
        )?;
        match outcome {
            PivotOutcome::Accepted => {
                // Scale L column and record d. Matches `do_1x1_update`'s
                // scale-then-update; rank-1 is deferred to replay.
                let d = a[col * nrow + col];
                if d.abs() != 0.0 {
                    let inv_d = 1.0 / d;
                    for i in (col + 1)..nrow {
                        a[col * nrow + i] *= inv_d;
                    }
                }
                d_panel[c] = d;
            }
            PivotOutcome::Rejected => {
                // `try_reject_1x1_frontal` has zeroed the L column and
                // diagonal. d = 0 → replay alpha = 0 → no-op, matching
                // scalar's `do_1x1_update` early-return on d == 0.
                d_panel[c] = 0.0;
            }
            PivotOutcome::Delayed => {
                // SSIDS break-on-first-failure contract: the pivot was
                // below threshold and `may_delay == true`. The rejection
                // routine did NOT mutate state for this column, so we
                // return with `n_elim = c` and let the caller apply the
                // deferred Schur to columns `[k+c+1, nrow)`, then break
                // out of the outer loop. Column `k+c` retains its
                // peek-ahead state (pivots 0..c-1 applied), which
                // matches scalar's column state at break time exactly.
                return Ok((c, PanelStatus::Delayed));
            }
            PivotOutcome::AcceptedRook2x2 { .. } => {
                unreachable!("panel path never enables rook rescue")
            }
        }

        c += 1;
    }

    Ok((c, PanelStatus::Full))
}

/// Apply pivot `q`'s deferred rank-1 update to a single trailing column.
/// This is the **replay** primitive that makes the blocked path
/// bit-exact with scalar: for each (i, j) with j = `col`, `i >= j`,
/// scalar applies rank-1 updates in ascending pivot index; replay does
/// the same via repeated calls to this helper in order `q = 0..c-1`.
/// The inner axpy uses `schur_kernel::axpy_minus_unroll4_nofma` — the
/// same kernel as `do_1x1_update` — so the per-lane rounding matches.
fn peek_ahead_column(
    a: &mut [f64],
    nrow: usize,
    k: usize,
    c: usize,
    d_panel: &[f64],
    subdiag: &[f64],
    fma: bool,
) {
    peek_ahead_replay(a, nrow, k, c, k + c, d_panel, subdiag, fma);
}

/// Replay primitive: apply the first `n_committed` panel pivots
/// (those at panel positions `0..n_committed`) to `target_col` in
/// q-ascending order, honoring `subdiag[k+q] != 0` as a 2×2 marker.
/// Bit-exact with the eager scalar path applied to one trailing
/// column. Used by `peek_ahead_column` (where `target_col = k+c`)
/// and by the no-swap 2×2 inline path (where the panel must
/// peek-ahead `target_col = col+1` before reading the second pivot's
/// D values, growth-bound terms, and L scaling inputs).
#[allow(clippy::too_many_arguments)]
fn peek_ahead_replay(
    a: &mut [f64],
    nrow: usize,
    k: usize,
    n_committed: usize,
    target_col: usize,
    d_panel: &[f64],
    subdiag: &[f64],
    fma: bool,
) {
    let col = target_col;
    let mut q = 0usize;
    while q < n_committed {
        // Phase A 2×2 (dev/plans/dense-kernel-blas3.md): when pivot
        // q is the start of an inline 2×2 block, replay the rank-2
        // contribution via axpy2 — bit-exact with scalar
        // do_2x2_update:2094.
        if q + 1 < n_committed && subdiag[k + q] != 0.0 {
            let d11 = d_panel[q];
            let d22 = d_panel[q + 1];
            let d21 = subdiag[k + q];
            let q_col = k + q;
            let q1_col = k + q + 1;
            let l_jq = a[q_col * nrow + col];
            let l_jq1 = a[q1_col * nrow + col];
            let dl_jq = d11 * l_jq + d21 * l_jq1;
            let dl_jq1 = d21 * l_jq + d22 * l_jq1;
            // dst = column `col` rows col..nrow; src0/src1 =
            // columns q_col, q1_col rows col..nrow. Disjoint because
            // q1_col < col.
            let (before, rest) = a.split_at_mut(col * nrow);
            let src0 = &before[q_col * nrow + col..q_col * nrow + nrow];
            let src1 = &before[q1_col * nrow + col..q1_col * nrow + nrow];
            let dst = &mut rest[col..nrow];
            if fma {
                schur_kernel::axpy2_minus_unroll4(dst, src0, dl_jq, src1, dl_jq1);
            } else {
                schur_kernel::axpy2_minus_unroll4_nofma(dst, src0, dl_jq, src1, dl_jq1);
            }
            q += 2;
            continue;
        }
        // 1×1 contribution. Scalar's `do_1x1_update` returns early
        // when d == 0; skipping here preserves that no-op behavior
        // bit-exactly.
        let d_q = d_panel[q];
        if d_q.abs() == 0.0 {
            q += 1;
            continue;
        }
        let q_col = k + q;
        let l_jk = a[q_col * nrow + col];
        let alpha = l_jk * d_q;
        if alpha == 0.0 {
            q += 1;
            continue;
        }
        let (before, rest) = a.split_at_mut(col * nrow);
        let src = &before[q_col * nrow + col..q_col * nrow + nrow];
        let dst = &mut rest[col..nrow];
        if fma {
            schur_kernel::axpy_minus_unroll4(dst, src, alpha);
        } else {
            schur_kernel::axpy_minus_unroll4_nofma(dst, src, alpha);
        }
        q += 1;
    }
}

/// Apply the `n_elim` panel pivots' deferred rank-1 updates to the
/// trailing columns `[j_start, nrow)`.
///
/// W-2 (`dev/plans/dense-kernel-speedup.md`): when every accepted
/// pivot in the panel is a 1×1 (no 2×2 pivots and no rejected
/// pivots), use the rank-`n_elim` accumulator
/// `schur_kernel::schur_panel_minus_nofma` to issue one pulp dispatch
/// per trailing column instead of `n_elim` per column. Per-element
/// accumulation order — and hence bit-pattern — is preserved by
/// applying the per-q `mul + sub` sequentially in register
/// accumulators inside the kernel body.
///
/// The fallback path (rank-1 axpy in q-outer, j-inner order) remains
/// the bit-exact reference. We route to it whenever any pivot is
/// "rejected to zero" (rank-bs would still match because alpha=0 is
/// skipped, but exercising the simpler reference here keeps the
/// fallback honest) or when `n_elim < W2_RANK_BS_MIN` so the
/// per-column alpha-precompute overhead is amortised.
///
/// `j_start` is typically `k + n_elim`, except when the caller had the
/// panel peek-ahead one extra column (`ScalarFallback`), in which case
/// `j_start = k + n_elim + 1` to avoid double-updating the peeked column.
#[allow(clippy::too_many_arguments)]
fn apply_blocked_schur(
    a: &mut [f64],
    nrow: usize,
    k: usize,
    n_elim: usize,
    j_start: usize,
    d_panel: &[f64],
    subdiag: &[f64],
    fma: bool,
) {
    if n_elim == 0 || j_start >= nrow {
        return;
    }

    // W-2 fast path: when the panel produced n_elim 1×1 pivots all
    // with non-zero d AND no 2×2 pivots, run the rank-`n_elim`
    // accumulator. The 2×2 gate is a correctness requirement (Phase
    // A, dev/plans/dense-kernel-blas3.md): the rank-bs kernel
    // accumulates contributions per-q sequentially (`acc -= a_q *
    // s_q`), which is bit-exact with sequential rank-1 axpys but NOT
    // with axpy2's fused `(a_0*s_0 + a_1*s_1)` add-then-sub
    // ordering. Lifting this gate is Phase B-2.
    const W2_RANK_BS_MIN: usize = 2;
    let any_zero_d = d_panel.iter().take(n_elim).any(|&d| d.abs() == 0.0);
    let has_2x2 = subdiag[k..k + n_elim].iter().any(|&s| s != 0.0);

    if !any_zero_d && !has_2x2 && n_elim >= W2_RANK_BS_MIN {
        apply_blocked_schur_panel(a, nrow, k, n_elim, j_start, d_panel, fma);
        return;
    }

    // Fallback: pivot-pair-or-singleton outer loop, kernel inner.
    // Per-element accumulation order matches scalar:
    //   - 1×1 at q: axpy with `alpha = l_jk * d_q` — bit-exact with
    //     do_1x1_update:2057.
    //   - 2×2 at (q, q+1): axpy2 with `(dl_j0, dl_j1)` derived from
    //     the d11/d21/d22 block — bit-exact with do_2x2_update:2085.
    let mut q = 0usize;
    while q < n_elim {
        if q + 1 < n_elim && subdiag[k + q] != 0.0 {
            // 2×2 contribution from cols (q, q+1).
            let d11 = d_panel[q];
            let d22 = d_panel[q + 1];
            let d21 = subdiag[k + q];
            let q_col = k + q;
            let q1_col = k + q + 1;
            for j in j_start..nrow {
                let l_jq = a[q_col * nrow + j];
                let l_jq1 = a[q1_col * nrow + j];
                let dl_jq = d11 * l_jq + d21 * l_jq1;
                let dl_jq1 = d21 * l_jq + d22 * l_jq1;
                // dst, src0, src1 are pairwise disjoint because
                // q_col < q1_col < j.
                let (before, rest) = a.split_at_mut(j * nrow);
                let src0 = &before[q_col * nrow + j..q_col * nrow + nrow];
                let src1 = &before[q1_col * nrow + j..q1_col * nrow + nrow];
                let dst = &mut rest[j..nrow];
                if fma {
                    schur_kernel::axpy2_minus_unroll4(dst, src0, dl_jq, src1, dl_jq1);
                } else {
                    schur_kernel::axpy2_minus_unroll4_nofma(dst, src0, dl_jq, src1, dl_jq1);
                }
            }
            q += 2;
        } else {
            let d_q = d_panel[q];
            if d_q.abs() == 0.0 {
                q += 1;
                continue;
            }
            let q_col = k + q;
            for j in j_start..nrow {
                let l_jk = a[q_col * nrow + j];
                let alpha = l_jk * d_q;
                if alpha == 0.0 {
                    continue;
                }
                let (before, rest) = a.split_at_mut(j * nrow);
                let src = &before[q_col * nrow + j..q_col * nrow + nrow];
                let dst = &mut rest[j..nrow];
                if fma {
                    schur_kernel::axpy_minus_unroll4(dst, src, alpha);
                } else {
                    schur_kernel::axpy_minus_unroll4_nofma(dst, src, alpha);
                }
            }
            q += 1;
        }
    }
}

/// Rank-`n_elim` deferred-Schur trailing-update path (W-2). Walks
/// trailing columns `[j_start, nrow)` in the outer loop and, for each
/// `j`, builds a length-`n_elim` `alphas` vector then issues a single
/// `schur_kernel::schur_panel_minus_nofma` dispatch covering all
/// `n_elim` rank-1 contributions to the column.
///
/// Caller invariants:
/// - All `d_panel[0..n_elim]` are non-zero (1×1 pivots only). When any
///   d_q is zero, route to the rank-1 fallback in
///   `apply_blocked_schur` to keep the scalar reference honest. (The
///   kernel itself does skip zero alphas, so this gate is a perf
///   knob.)
/// - `j_start <= nrow` and `n_elim > 0`.
///
/// Bit-exactness contract: per-element, the SIMD body issues
/// `acc <- round(acc - round(alpha_q * src_q[i]))` for q in ascending
/// order, the same sequence as the rank-1 reference.
fn apply_blocked_schur_panel(
    a: &mut [f64],
    nrow: usize,
    k: usize,
    n_elim: usize,
    j_start: usize,
    d_panel: &[f64],
    fma: bool,
) {
    // Stack-friendly upper bound on n_elim. The current panel is
    // gated at `params.block_size` which defaults to 64; if that
    // ceiling ever changes, increase MAX_N_ELIM accordingly. Using a
    // fixed-size buffer here avoids a per-column heap allocation in
    // the hot path.
    const MAX_N_ELIM: usize = 64;
    debug_assert!(
        n_elim <= MAX_N_ELIM,
        "apply_blocked_schur_panel: n_elim {} exceeds MAX_N_ELIM {}",
        n_elim,
        MAX_N_ELIM
    );

    let mut alphas0_buf = [0.0f64; MAX_N_ELIM];
    let mut alphas1_buf = [0.0f64; MAX_N_ELIM];
    let mut alphas2_buf = [0.0f64; MAX_N_ELIM];
    let mut alphas3_buf = [0.0f64; MAX_N_ELIM];

    // Phase 2.4.3 (issue #9, re-scoped): walk trailing columns in
    // groups of 4, dispatching the quad-column kernel that shares
    // each src-vector load across 4 destination accumulators. Each
    // quad is bit-exact with four sequential single-column dispatches
    // (verified by the quad kernel's bit-exactness sweep).
    //
    // Fall-through: 2-or-3-column remainder uses dual + (optional)
    // single; 0-or-1-column remainder uses single.
    let mut j = j_start;
    while j + 3 < nrow {
        let alphas0 = &mut alphas0_buf[..n_elim];
        let alphas1 = &mut alphas1_buf[..n_elim];
        let alphas2 = &mut alphas2_buf[..n_elim];
        let alphas3 = &mut alphas3_buf[..n_elim];
        let mut all_zero = true;
        for q in 0..n_elim {
            let q_col = k + q;
            let base = q_col * nrow;
            let l_jk0 = a[base + j];
            let l_jk1 = a[base + j + 1];
            let l_jk2 = a[base + j + 2];
            let l_jk3 = a[base + j + 3];
            let d_q = d_panel[q];
            let alpha0 = l_jk0 * d_q;
            let alpha1 = l_jk1 * d_q;
            let alpha2 = l_jk2 * d_q;
            let alpha3 = l_jk3 * d_q;
            alphas0[q] = alpha0;
            alphas1[q] = alpha1;
            alphas2[q] = alpha2;
            alphas3[q] = alpha3;
            if alpha0 != 0.0 || alpha1 != 0.0 || alpha2 != 0.0 || alpha3 != 0.0 {
                all_zero = false;
            }
        }
        if !all_zero {
            // Carve out four disjoint mutable slices. The src pivot
            // columns [k, k+n_elim) all precede column j in memory,
            // so a single split at j*nrow gives src in `before`.
            // Three nested splits inside `rest` separate columns
            // j, j+1, j+2, j+3.
            let (before, rest) = a.split_at_mut(j * nrow);
            let (col_j, rest1) = rest.split_at_mut(nrow);
            let (col_j1, rest2) = rest1.split_at_mut(nrow);
            let (col_j2, col_j3_and_after) = rest2.split_at_mut(nrow);
            let dst0 = &mut col_j[j..];
            let dst1 = &mut col_j1[(j + 1)..nrow];
            let dst2 = &mut col_j2[(j + 2)..nrow];
            let dst3 = &mut col_j3_and_after[(j + 3)..nrow];
            if fma {
                schur_kernel::schur_panel_minus_fma_strided_quad(
                    dst0, dst1, dst2, dst3, before, k, n_elim, nrow, j, alphas0, alphas1, alphas2,
                    alphas3,
                );
            } else {
                schur_kernel::schur_panel_minus_nofma_strided_quad(
                    dst0, dst1, dst2, dst3, before, k, n_elim, nrow, j, alphas0, alphas1, alphas2,
                    alphas3,
                );
            }
        }
        j += 4;
    }

    // 2-or-3-column remainder: use dual to consume the next pair, if
    // any. After this, at most 1 column remains.
    if j + 1 < nrow {
        let alphas0 = &mut alphas0_buf[..n_elim];
        let alphas1 = &mut alphas1_buf[..n_elim];
        let mut all_zero = true;
        for q in 0..n_elim {
            let q_col = k + q;
            let l_jk0 = a[q_col * nrow + j];
            let l_jk1 = a[q_col * nrow + j + 1];
            let d_q = d_panel[q];
            let alpha0 = l_jk0 * d_q;
            let alpha1 = l_jk1 * d_q;
            alphas0[q] = alpha0;
            alphas1[q] = alpha1;
            if alpha0 != 0.0 || alpha1 != 0.0 {
                all_zero = false;
            }
        }
        if !all_zero {
            let (before, rest) = a.split_at_mut(j * nrow);
            let (col_j, after_j) = rest.split_at_mut(nrow);
            let dst0 = &mut col_j[j..];
            let dst1 = &mut after_j[(j + 1)..nrow];
            if fma {
                schur_kernel::schur_panel_minus_fma_strided_dual(
                    dst0, dst1, before, k, n_elim, nrow, j, alphas0, alphas1,
                );
            } else {
                schur_kernel::schur_panel_minus_nofma_strided_dual(
                    dst0, dst1, before, k, n_elim, nrow, j, alphas0, alphas1,
                );
            }
        }
        j += 2;
    }

    if j < nrow {
        // Odd remainder: one trailing column. Fall back to the
        // single-column kernel.
        let alphas = &mut alphas0_buf[..n_elim];
        let mut all_zero_alpha = true;
        for q in 0..n_elim {
            let q_col = k + q;
            let l_jk = a[q_col * nrow + j];
            let d_q = d_panel[q];
            let alpha = l_jk * d_q;
            alphas[q] = alpha;
            if alpha != 0.0 {
                all_zero_alpha = false;
            }
        }
        if !all_zero_alpha {
            let trailing_len = nrow - j;
            let (before, rest) = a.split_at_mut(j * nrow);
            let dst = &mut rest[j..nrow];
            if fma {
                schur_kernel::schur_panel_minus_fma_strided(
                    dst,
                    before,
                    k,
                    n_elim,
                    nrow,
                    j,
                    trailing_len,
                    alphas,
                );
            } else {
                schur_kernel::schur_panel_minus_nofma_strided(
                    dst,
                    before,
                    k,
                    n_elim,
                    nrow,
                    j,
                    trailing_len,
                    alphas,
                );
            }
        }
    }
}

/// MAXFROMM capture: scan column `col` of the trailing submatrix
/// (rows `col+1..n`) for its max absolute value. Returns `None` when
/// the column has no off-diagonal entries (last pivot).
///
/// Cheap because the column was just written by the rank-1 trailing
/// update at pivot `k = col - 1` — the cache line containing
/// `a[col*n + col+1 ..]` is hot.
#[inline]
fn capture_maxfromm_col(a: &[f64], n: usize, col: usize) -> Option<f64> {
    if col + 1 >= n {
        return None;
    }
    let mut mf = 0.0f64;
    for i in (col + 1)..n {
        let v = a[col * n + i].abs();
        if v > mf {
            mf = v;
        }
    }
    Some(mf)
}

/// Translate a `PivotOutcome` from `try_reject_1x1_with_rook_rescue`
/// (or plain `try_reject_1x1_frontal`) into a `PivotStepResult` and
/// perform the required trailing-update. Used at every scalar BK
/// 1×1 call site to centralize the `AcceptedRook2x2` dispatch.
///
/// `maxfromm` is the MAXFROMM cache slot: when `Some`, on
/// `Accepted` the freshly updated column `k+1` is scanned and the
/// result stored; on every other outcome the slot is cleared (next
/// iteration must re-scan). When `None`, no capture happens.
#[inline]
#[allow(clippy::too_many_arguments)]
fn finish_1x1_outcome(
    outcome: PivotOutcome,
    a: &mut [f64],
    nrow: usize,
    k: usize,
    subdiag: &mut [f64],
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    fma: bool,
    maxfromm: Option<&mut Option<f64>>,
) -> PivotStepResult {
    match outcome {
        PivotOutcome::Accepted => {
            do_1x1_update(a, nrow, k, fma);
            if let Some(slot) = maxfromm {
                *slot = capture_maxfromm_col(a, nrow, k + 1);
            }
            PivotStepResult::Advanced(1)
        }
        PivotOutcome::Rejected => {
            if let Some(slot) = maxfromm {
                *slot = None;
            }
            PivotStepResult::Advanced(1)
        }
        PivotOutcome::Delayed => {
            if let Some(slot) = maxfromm {
                *slot = None;
            }
            PivotStepResult::Delayed
        }
        PivotOutcome::AcceptedRook2x2 { d11, d21, d22 } => {
            let inertia = count_2x2_inertia_val(d11, d21, d22);
            *pos += inertia.positive;
            *neg += inertia.negative;
            *zero += inertia.zero;
            subdiag[k] = d21;
            do_2x2_update(a, nrow, k, d11, d21, d22, fma);
            if let Some(slot) = maxfromm {
                *slot = None;
            }
            PivotStepResult::Advanced(2)
        }
    }
}

/// One iteration of the scalar BK pivot loop for `factor_frontal`.
///
/// Extracted verbatim from the pre-extraction in-line loop body so the
/// Phase 2.4.1b blocked-panel path can share the rejection/delay fallback
/// with the unblocked driver. Byte-identical behavior with the original
/// body is required — see `dev/plans/phase-2.4.1-blocked-ldlt.md` §2.
///
/// Returns `Advanced(1)` or `Advanced(2)` on success; `Delayed` when the
/// SSIDS-style delay path fires (only possible when `may_delay == true`).
#[allow(clippy::too_many_arguments)]
fn scalar_pivot_step(
    a: &mut [f64],
    nrow: usize,
    ncol: usize,
    k: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
    perm: &mut [usize],
    subdiag: &mut [f64],
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
    n_rook_rescues: &mut usize,
    cached_maxfromm: &mut Option<f64>,
) -> Result<PivotStepResult, FeralError> {
    let alpha = params.alpha;
    let remaining = ncol - k;
    let use_maxfromm = matches!(params.tpp_method, TppMethod::Maxfromm);
    // Take ownership of any cached value: it belongs to *this* pivot.
    // Subsequent paths will re-populate it (or leave it None).
    let cached = cached_maxfromm.take();

    if remaining == 1 {
        // Last eliminated pivot: always 1×1. Compute the column max
        // over rows (k+1..nrow) for the column-relative threshold.
        // Rook rescue cannot fire here (needs ncol-k >= 2), so call
        // the rejection routine directly.
        let col_max = if let Some(mf) = cached {
            mf
        } else {
            let mut m = 0.0f64;
            for i in (k + 1)..nrow {
                let v = a[k * nrow + i].abs();
                if v > m {
                    m = v;
                }
            }
            m
        };
        let outcome = try_reject_1x1_frontal(
            a,
            nrow,
            k,
            col_max,
            may_delay,
            params,
            pos,
            neg,
            zero,
            needs_refinement,
        )?;
        match outcome {
            PivotOutcome::Accepted => do_1x1_update(a, nrow, k, params.fma),
            PivotOutcome::Rejected => {}
            PivotOutcome::Delayed => return Ok(PivotStepResult::Delayed),
            PivotOutcome::AcceptedRook2x2 { .. } => {
                unreachable!("remaining==1 never triggers rook rescue")
            }
        }
        return Ok(PivotStepResult::Advanced(1));
    }

    // MAXFROMM short-circuit: if the previous 1×1 stash gives a
    // gamma0-equivalent for this pivot AND the diagonal already
    // dominates by `alpha`, skip the AMAX scan and routing 1×1-at-k.
    // gamma0 is `max_{i>k} |a[i,k]|` — for the in-place column-k
    // factor, MAXFROMM captured exactly this set after pivot k-1.
    // Bit-identical to the full-scan BK 1×1 path that follows.
    if use_maxfromm {
        if let Some(mf) = cached {
            let akk = a[k * nrow + k].abs();
            if akk >= alpha * mf {
                let outcome = try_reject_1x1_with_rook_rescue(
                    a,
                    nrow,
                    ncol,
                    k,
                    mf,
                    may_delay,
                    params,
                    perm,
                    pos,
                    neg,
                    zero,
                    needs_refinement,
                    n_rook_rescues,
                )?;
                return Ok(finish_1x1_outcome(
                    outcome,
                    a,
                    nrow,
                    k,
                    subdiag,
                    pos,
                    neg,
                    zero,
                    params.fma,
                    Some(cached_maxfromm),
                ));
            }
            // Diagonal too small relative to cached MAXFROMM — fall
            // through to full scan. `cached_maxfromm` is already
            // cleared (we `.take()`d it above).
        }
    }

    // Find max |A[i,k]| for i in (k, ncol) — restricted to fully-summed rows
    let (gamma0, r) = {
        let mut max_val = 0.0f64;
        let mut max_row = k + 1;
        // Search within fully-summed rows first
        for i in (k + 1)..ncol {
            let v = a[k * nrow + i].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        // Also check sub-diagonal rows (they contribute to gamma0 for
        // the BK pivot test, but are never swapped into pivot position)
        for i in ncol..nrow {
            let v = a[k * nrow + i].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        (max_val, max_row)
    };

    if gamma0 == 0.0 {
        count_1x1_inertia(a, nrow, k, params, pos, neg, zero, needs_refinement)?;
        set_l_column_identity(a, nrow, k);
        return Ok(PivotStepResult::Advanced(1));
    }

    let akk = a[k * nrow + k].abs();

    if akk >= alpha * gamma0 {
        // 1×1 pivot at k, no swap
        let outcome = try_reject_1x1_with_rook_rescue(
            a,
            nrow,
            ncol,
            k,
            gamma0,
            may_delay,
            params,
            perm,
            pos,
            neg,
            zero,
            needs_refinement,
            n_rook_rescues,
        )?;
        return Ok(finish_1x1_outcome(
            outcome,
            a,
            nrow,
            k,
            subdiag,
            pos,
            neg,
            zero,
            params.fma,
            if use_maxfromm {
                Some(&mut *cached_maxfromm)
            } else {
                None
            },
        ));
    }

    // gamma_r: max off-diagonal in symmetric row r
    let gamma_r = symmetric_row_offdiag_max(a, nrow, k, r);
    let arr = a[r * nrow + r].abs();

    // Can we swap r into pivot position? Only if r < ncol (fully summed)
    let r_is_fully_summed = r < ncol;

    if r_is_fully_summed && arr >= alpha * gamma_r {
        // 1×1 pivot at r, swap r↔k
        swap_rows_cols(a, nrow, k, r, perm);
        let outcome = try_reject_1x1_with_rook_rescue(
            a,
            nrow,
            ncol,
            k,
            gamma_r,
            may_delay,
            params,
            perm,
            pos,
            neg,
            zero,
            needs_refinement,
            n_rook_rescues,
        )?;
        return Ok(finish_1x1_outcome(
            outcome,
            a,
            nrow,
            k,
            subdiag,
            pos,
            neg,
            zero,
            params.fma,
            if use_maxfromm {
                Some(&mut *cached_maxfromm)
            } else {
                None
            },
        ));
    }

    if akk * gamma_r >= alpha * gamma0 * gamma0 {
        // 1×1 pivot at k (LAPACK extension), no swap
        let outcome = try_reject_1x1_with_rook_rescue(
            a,
            nrow,
            ncol,
            k,
            gamma0,
            may_delay,
            params,
            perm,
            pos,
            neg,
            zero,
            needs_refinement,
            n_rook_rescues,
        )?;
        return Ok(finish_1x1_outcome(
            outcome,
            a,
            nrow,
            k,
            subdiag,
            pos,
            neg,
            zero,
            params.fma,
            if use_maxfromm {
                Some(&mut *cached_maxfromm)
            } else {
                None
            },
        ));
    }

    if r_is_fully_summed && k + 1 < ncol {
        // 2×2 pivot using {k, r}, both fully summed
        if r != k + 1 {
            swap_rows_cols(a, nrow, k + 1, r, perm);
        }
        let d11 = a[k * nrow + k];
        let d21 = a[k * nrow + (k + 1)];
        let d22 = a[(k + 1) * nrow + (k + 1)];
        let det = d11 * d22 - d21 * d21;

        // Duff-Reid 2×2 growth bound (MUMPS dfac_front_aux.F:1599-1606):
        //
        //   reject iff  (|a22|*RMAX + AMAX*TMAX) * u  >  |det|
        //        OR     (|a11|*TMAX + AMAX*RMAX) * u  >  |det|
        //
        // where RMAX = max |a[i, k]| for i > k+1
        //       TMAX = max |a[i, k+1]| for i > k+1
        //       AMAX = |a[k+1, k]| = |d21|
        // (i.e., RMAX and TMAX are the column maxes of the two pivot
        // columns *beyond* the 2×2 block; AMAX is the cross term.)
        //
        // When pivot_threshold == 0.0 the growth bound is always
        // satisfied (0 <= |det|), preserving Phase 1 behavior.
        let mut rmax = 0.0f64;
        let mut tmax = 0.0f64;
        for i in (k + 2)..nrow {
            let v0 = a[k * nrow + i].abs();
            if v0 > rmax {
                rmax = v0;
            }
            let v1 = a[(k + 1) * nrow + i].abs();
            if v1 > tmax {
                tmax = v1;
            }
        }
        let amax = d21.abs();
        let absdet = det.abs();
        let u = params.pivot_threshold;
        let growth_fail = (d22.abs() * rmax + amax * tmax) * u > absdet
            || (d11.abs() * tmax + amax * rmax) * u > absdet;

        // Scale-invariant cancellation-aware determinant floor, ported
        // from SSIDS `src/ssids/cpu/kernels/ldlt_tpp.cxx:98-106`:
        //
        //   maxpiv    = max(|a11|, |a21|, |a22|)
        //   detscale  = 1 / maxpiv
        //   detpiv0   = (a11 * detscale) * a22
        //   detpiv1   = (a21 * detscale) * a21
        //   detpiv    = detpiv0 - detpiv1      (== det / maxpiv)
        //   reject iff maxpiv < small
        //           OR |detpiv| < max(small, |detpiv0|/2, |detpiv1|/2)
        //
        // This replaces the prior absolute `|det| <= zero_tol_2x2` floor,
        // which was only meaningful on equilibrated matrices. The test
        // is scale-invariant by construction: the ratio `|detpiv|` vs
        // fractions of `|detpiv0|`, `|detpiv1|` does not depend on
        // the absolute magnitude of the block. `SSIDS_DET_SMALL = 1e-20`
        // is a dead-zero underflow floor, NOT a stability threshold.
        // See dev/research/ssids-scale-invariant-det-floor.md.
        let max_piv = d11.abs().max(d21.abs()).max(d22.abs());
        let det_floor_fail = if max_piv < SSIDS_DET_SMALL {
            true
        } else {
            let det_scale = 1.0 / max_piv;
            let detpiv0 = (d11 * det_scale) * d22;
            let detpiv1 = (d21 * det_scale) * d21;
            let detpiv = detpiv0 - detpiv1;
            let cancel_floor = SSIDS_DET_SMALL
                .max(detpiv0.abs() * 0.5)
                .max(detpiv1.abs() * 0.5);
            detpiv.abs() < cancel_floor
        };

        if growth_fail || det_floor_fail {
            // 2×2 rejected. SSIDS-style delayed pivoting: when
            // `may_delay == true`, break out immediately so the parent
            // supernode can retry this pivot with a larger pivot search
            // window. Otherwise fall back to a single 1×1 at k with the
            // column-relative threshold, which triggers the existing
            // ForceAccept path.
            if may_delay {
                return Ok(PivotStepResult::Delayed);
            }
            if det_floor_fail {
                match params.on_zero_pivot {
                    ZeroPivotAction::Fail => {
                        return Err(FeralError::NumericallyRankDeficient);
                    }
                    ZeroPivotAction::ForceAccept | ZeroPivotAction::PerturbToEps { .. } => {
                        // PerturbToEps falls back to ForceAccept-style
                        // accounting for a near-singular 2×2 block; the
                        // subsequent 1×1 attempt below will apply the
                        // bounded-Δ perturbation per pivot.
                        *needs_refinement = true;
                    }
                }
            }
            let outcome = try_reject_1x1_with_rook_rescue(
                a,
                nrow,
                ncol,
                k,
                gamma0,
                may_delay,
                params,
                perm,
                pos,
                neg,
                zero,
                needs_refinement,
                n_rook_rescues,
            )?;
            return Ok(finish_1x1_outcome(
                outcome,
                a,
                nrow,
                k,
                subdiag,
                pos,
                neg,
                zero,
                params.fma,
                if use_maxfromm {
                    Some(&mut *cached_maxfromm)
                } else {
                    None
                },
            ));
        }

        let pivot_inertia = count_2x2_inertia_val(d11, d21, d22);
        *pos += pivot_inertia.positive;
        *neg += pivot_inertia.negative;
        *zero += pivot_inertia.zero;

        subdiag[k] = d21;
        // 2×2 update — cache for pivot k+2 cannot be derived from a
        // 2×2 trailing update without extra bookkeeping; clear so the
        // next iteration falls back to a full scan. (Phase 4 may wire
        // 2×2 capture; see issue-10 research note.)
        do_2x2_update(a, nrow, k, d11, d21, d22, params.fma);
        Ok(PivotStepResult::Advanced(2))
    } else {
        // Can't do 2×2 (r is not fully summed or only 1 column left).
        // Last-resort 1×1 at k with column-relative rejection.
        let outcome = try_reject_1x1_with_rook_rescue(
            a,
            nrow,
            ncol,
            k,
            gamma0,
            may_delay,
            params,
            perm,
            pos,
            neg,
            zero,
            needs_refinement,
            n_rook_rescues,
        )?;
        Ok(finish_1x1_outcome(
            outcome,
            a,
            nrow,
            k,
            subdiag,
            pos,
            neg,
            zero,
            params.fma,
            if use_maxfromm {
                Some(&mut *cached_maxfromm)
            } else {
                None
            },
        ))
    }
}

/// Apply the column-relative pivot threshold to a frontal 1×1 candidate at
/// position `k` with column max `col_max`. Returns:
///
/// - `Accepted` — pivot clears the threshold; caller should apply the
///   rank-1 update.
/// - `Rejected` — pivot is below threshold AND `may_delay == false`; the
///   L column has been zeroed and a zero pivot has been counted via
///   `ZeroPivotAction::ForceAccept`. Caller increments `k` and continues.
/// - `Delayed` — pivot is below threshold AND `may_delay == true`; no
///   state has been mutated. Caller should break the BK loop so the parent
///   supernode can retry this column.
///
/// `ZeroPivotAction::Fail` short-circuits to `Err(NumericallyRankDeficient)`
/// regardless of `may_delay`.
#[allow(clippy::too_many_arguments)]
fn try_reject_1x1_frontal(
    a: &mut [f64],
    nrow: usize,
    k: usize,
    col_max: f64,
    may_delay: bool,
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
) -> Result<PivotOutcome, FeralError> {
    let d = a[k * nrow + k];
    // Reject-gate uses `null_pivot_tol` so rank-deficiency pivots in
    // the [zero_tol, null_pivot_tol] band are routed through the case
    // split below. When `null_pivot_tol == zero_tol` (default) this
    // collapses to the historical floor.
    let threshold = (params.pivot_threshold * col_max).max(params.null_pivot_tol);

    if d.abs() <= threshold {
        if may_delay {
            return Ok(PivotOutcome::Delayed);
        }
        // At the root (may_delay=false) we have no parent to absorb
        // the rejected pivot. Three branches by absolute magnitude:
        //
        //  (a)  |d| <= zero_tol  — truly numerically zero. Set d=0
        //       so solve skips this position; count as zero. This is
        //       the strict floor (default EPS).
        //
        //  (a') zero_tol < |d| <= null_pivot_tol — rank-deficiency
        //       band per Wilkinson's backward error bound. Count as
        //       zero in inertia (F-01) but leave d_diag and L intact
        //       so the solve can still divide by `d` (its strict
        //       `factors.zero_tol` check on the divide stays at EPS).
        //       `needs_refinement = true` guards the residual.
        //
        //  (b)  null_pivot_tol < |d| <= u*col_max — small but clearly
        //       nonzero by the relative-scale test. Accept with
        //       correct sign, request iterative refinement. Matches
        //       SSIDS/MUMPS convention for degenerate LPs (e.g.
        //       DEGENLPA where MUMPS reports (20, 15, 0)).
        if d.abs() <= params.zero_tol {
            match params.on_zero_pivot {
                ZeroPivotAction::ForceAccept => {
                    *needs_refinement = true;
                    *zero += 1;
                    for i in (k + 1)..nrow {
                        a[k * nrow + i] = 0.0;
                    }
                    a[k * nrow + k] = 0.0;
                    return Ok(PivotOutcome::Rejected);
                }
                ZeroPivotAction::Fail => return Err(FeralError::NumericallyRankDeficient),
                ZeroPivotAction::PerturbToEps { abs_floor } => {
                    let d_new = perturb_to_floor(d, abs_floor);
                    a[k * nrow + k] = d_new;
                    *needs_refinement = true;
                    if d_new > 0.0 {
                        *pos += 1;
                    } else {
                        *neg += 1;
                    }
                    // Fall through to Accepted: caller runs do_1x1_update
                    // with the perturbed pivot.
                    return Ok(PivotOutcome::Accepted);
                }
            }
        }
        // Case (a'): rank-deficiency band — count as zero, keep factor
        // intact so the solve can divide by `d`. Only fires under
        // ForceAccept; Fail and PerturbToEps fall through to case (b)
        // for consistency with their pre-F-01 semantics.
        if matches!(params.on_zero_pivot, ZeroPivotAction::ForceAccept)
            && d.abs() <= params.null_pivot_tol
        {
            *needs_refinement = true;
            *zero += 1;
            return Ok(PivotOutcome::Accepted);
        }
        // Case (b): small but nonzero — accept with correct sign.
        *needs_refinement = true;
        if d > 0.0 {
            *pos += 1;
        } else {
            *neg += 1;
        }
        return Ok(PivotOutcome::Accepted);
    }

    // Accept: sign-based inertia.
    if d > 0.0 {
        *pos += 1;
    } else {
        *neg += 1;
    }
    Ok(PivotOutcome::Accepted)
}

/// Phase 2.4.3 splice: attempt a 1×1 pivot with rook rescue on rejection.
///
/// Fast path: if `|d| > threshold`, delegate to `try_reject_1x1_frontal`
/// unchanged (well-conditioned matrices pay zero rook cost, matching the
/// plan's "rescue, not top-level" design).
///
/// Slow path: if the column-relative threshold rejects the pivot at `k`,
/// call `rook_rescue` before falling through to delay / force-accept.
/// On rook success, apply the symmetric swap sequence via
/// `swap_rows_cols` (updating `perm`), increment `n_rook_rescues`, and
/// return either `Accepted` (1×1 rescue — caller runs `do_1x1_update`)
/// or `AcceptedRook2x2` (2×2 rescue — caller runs the 2×2 update path).
///
/// Panel path (`lblt_panel_frontal`) cannot safely apply mid-panel
/// swaps (they would invalidate `d_panel` replay state), so it keeps
/// calling the original `try_reject_1x1_frontal` directly and never
/// enters this wrapper. See plan §"Blocked-panel interaction".
#[allow(clippy::too_many_arguments)]
fn try_reject_1x1_with_rook_rescue(
    a: &mut [f64],
    nrow: usize,
    ncol: usize,
    k: usize,
    col_max: f64,
    may_delay: bool,
    params: &BunchKaufmanParams,
    perm: &mut [usize],
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
    n_rook_rescues: &mut usize,
) -> Result<PivotOutcome, FeralError> {
    let d = a[k * nrow + k];
    let threshold = (params.pivot_threshold * col_max).max(params.null_pivot_tol);

    // Well-conditioned fast path: pivot clears the threshold. Delegate
    // verbatim so accounting stays byte-identical to the pre-rook path.
    if d.abs() > threshold {
        return try_reject_1x1_frontal(
            a,
            nrow,
            k,
            col_max,
            may_delay,
            params,
            pos,
            neg,
            zero,
            needs_refinement,
        );
    }

    // Threshold failed — try rook rescue before delay/force-accept.
    if let Some(pivot) = rook_rescue(a, nrow, ncol, k, params) {
        *n_rook_rescues += 1;
        for idx in 0..pivot.n_swaps {
            let (p, q) = pivot.swaps[idx];
            swap_rows_cols(a, nrow, p, q, perm);
        }
        match pivot.kind {
            RookKind::Pivot1x1 => {
                let d_new = a[k * nrow + k];
                if d_new > 0.0 {
                    *pos += 1;
                } else {
                    *neg += 1;
                }
                return Ok(PivotOutcome::Accepted);
            }
            RookKind::Pivot2x2 => {
                let d11 = a[k * nrow + k];
                let d21 = a[k * nrow + (k + 1)];
                let d22 = a[(k + 1) * nrow + (k + 1)];
                return Ok(PivotOutcome::AcceptedRook2x2 { d11, d21, d22 });
            }
        }
    }

    // Rook could not rescue — delegate to the existing delay /
    // force-accept logic.
    try_reject_1x1_frontal(
        a,
        nrow,
        k,
        col_max,
        may_delay,
        params,
        pos,
        neg,
        zero,
        needs_refinement,
    )
}

/// 1×1 rank-1 update: update columns k+1..n after eliminating column k.
///
/// `fma=true` selects the FMA panel kernels (issue #8 opt-in); `fma=false`
/// preserves the cross-arch bit-exact non-FMA path.
pub(crate) fn do_1x1_update(a: &mut [f64], n: usize, k: usize, fma: bool) {
    // Issue #9 Step 2: route n==32 to the block-32 SIMD body. Bit-exact
    // per the parity sweep in `block_ldlt32::tests` (4 unit tests cover
    // p=0/5/30 + zero-pivot).
    if n == crate::dense::block_ldlt32::BLOCK_SIZE {
        crate::dense::block_ldlt32::update_1x1_block32(a, k, fma);
        return;
    }
    let d = a[k * n + k];
    if d.abs() == 0.0 {
        return;
    }
    let inv_d = 1.0 / d;
    for i in (k + 1)..n {
        a[k * n + i] *= inv_d;
    }
    for j in (k + 1)..n {
        let l_jk = a[k * n + j];
        let alpha = l_jk * d;
        // src = column k rows j..n (already scaled by inv_d above);
        // dst = column j rows j..n. Disjoint because k < j.
        let (before, rest) = a.split_at_mut(j * n);
        let src = &before[k * n + j..k * n + n];
        let dst = &mut rest[j..n];
        if fma {
            schur_kernel::axpy_minus_unroll4(dst, src, alpha);
        } else {
            schur_kernel::axpy_minus_unroll4_nofma(dst, src, alpha);
        }
    }
}

/// Rank-2 update after a 2×2 pivot at columns `k`, `k+1`.
///
/// `fma=true` selects the FMA panel kernels (issue #8 opt-in); `fma=false`
/// preserves the cross-arch bit-exact non-FMA path.
pub(crate) fn do_2x2_update(
    a: &mut [f64],
    n: usize,
    k: usize,
    d11: f64,
    d21: f64,
    d22: f64,
    fma: bool,
) {
    // Issue #9 Step 2: route n==32 to the block-32 body. The Step 4 SIMD
    // rank-2 kernel is still pending, so update_2x2_block32 is currently
    // the scalar per-column axpy2 (bit-identical to this function at
    // n==32); the indirection is a hook for Step 4.
    if n == crate::dense::block_ldlt32::BLOCK_SIZE {
        crate::dense::block_ldlt32::update_2x2_block32(a, k, d11, d21, d22, fma);
        return;
    }
    let det = d11 * d22 - d21 * d21;
    if det.abs() == 0.0 {
        return;
    }
    let inv_det = 1.0 / det;

    for i in (k + 2)..n {
        let a_ik = a[k * n + i];
        let a_ik1 = a[(k + 1) * n + i];
        a[k * n + i] = (d22 * a_ik - d21 * a_ik1) * inv_det;
        a[(k + 1) * n + i] = (d11 * a_ik1 - d21 * a_ik) * inv_det;
    }

    for j in (k + 2)..n {
        let l_j0 = a[k * n + j];
        let l_j1 = a[(k + 1) * n + j];
        let dl_j0 = d11 * l_j0 + d21 * l_j1;
        let dl_j1 = d21 * l_j0 + d22 * l_j1;
        // src0, src1 = columns k, k+1 rows j..n (scaled by the
        // rank-1 block of the 2×2 update above); dst = column j
        // rows j..n. Pairwise disjoint because k < k+1 < j.
        let (before, rest) = a.split_at_mut(j * n);
        let src0 = &before[k * n + j..k * n + n];
        let src1 = &before[(k + 1) * n + j..(k + 1) * n + n];
        let dst = &mut rest[j..n];
        if fma {
            schur_kernel::axpy2_minus_unroll4(dst, src0, dl_j0, src1, dl_j1);
        } else {
            schur_kernel::axpy2_minus_unroll4_nofma(dst, src0, dl_j0, src1, dl_j1);
        }
    }
}

/// Count inertia of a 2×2 D block, returning Inertia struct.
fn count_2x2_inertia_val(d11: f64, d21: f64, d22: f64) -> Inertia {
    let det = d11 * d22 - d21 * d21;
    let trace = d11 + d22;
    if det > 0.0 {
        if trace > 0.0 {
            Inertia::new(2, 0, 0)
        } else {
            Inertia::new(0, 2, 0)
        }
    } else if det < 0.0 {
        Inertia::new(1, 1, 0)
    } else if trace > 0.0 {
        Inertia::new(1, 0, 1)
    } else if trace < 0.0 {
        Inertia::new(0, 1, 1)
    } else {
        Inertia::new(0, 0, 2)
    }
}

/// Find max |A[i,k]| for i > k (column k, below diagonal).
/// Returns (max_value, row_index_of_max).
fn column_offdiag_max(a: &[f64], n: usize, k: usize) -> (f64, usize) {
    let mut max_val = 0.0;
    let mut max_idx = k + 1;
    for i in (k + 1)..n {
        let val = a[k * n + i].abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    (max_val, max_idx)
}

/// Compute the max off-diagonal magnitude in the full symmetric row/column r,
/// restricted to the trailing submatrix starting at column k.
/// This searches both below the diagonal (column r, rows > r) and
/// to the left of the diagonal (row r, columns k..r), excluding
/// position (r, k) which is not part of the "off-diagonal of r" in
/// the context of pivot selection — we want max over i != r.
fn symmetric_row_offdiag_max(a: &[f64], n: usize, k: usize, r: usize) -> f64 {
    let mut max_val = 0.0;

    // Below diagonal: column r, rows r+1..n
    for i in (r + 1)..n {
        let val = a[r * n + i].abs();
        if val > max_val {
            max_val = val;
        }
    }

    // Left of diagonal: row r, columns k..r (stored as a[col*n + r] for col < r)
    for j in k..r {
        let val = a[j * n + r].abs();
        if val > max_val {
            max_val = val;
        }
    }

    max_val
}

/// Swap rows and columns p and q in the lower triangle of the working matrix,
/// and update the permutation vector.
fn swap_rows_cols(a: &mut [f64], n: usize, p: usize, q: usize, perm: &mut [usize]) {
    if p == q {
        return;
    }
    // Ensure p < q
    let (p, q) = if p < q { (p, q) } else { (q, p) };

    // Swap permutation entries
    perm.swap(p, q);

    // Swap diagonal entries
    a.swap(p * n + p, q * n + q);

    // Swap columns p and q below row q (both in lower triangle)
    for i in (q + 1)..n {
        a.swap(p * n + i, q * n + i);
    }

    // Swap entries in column p (rows p+1..q-1) with entries in row q (cols p+1..q-1)
    // a[p*n + i] (col p, row i) with a[i*n + q] (col i, row q) for i in (p+1)..q
    for i in (p + 1)..q {
        a.swap(p * n + i, i * n + q);
    }

    // Swap a[p*n + q] (col p, row q) with... nothing — this is the (q, p) entry
    // In symmetric swap, the off-diagonal (p, q) entry needs special handling:
    // swap columns p row p+1..q-1 already done
    // Swap rows: for columns 0..p, swap a[j*n + p] with a[j*n + q]
    for j in 0..p {
        a.swap(j * n + p, j * n + q);
    }
}

/// Perform a 1×1 pivot at position k: compute L column, rank-1 update,
/// and fused argmax of the next column (Section 6 of research note).
/// Returns (gamma0_next, r_next) for the next pivot step's column.
///
/// `col_max` is the maximum off-diagonal magnitude in the column being
/// used as pivot (gamma0 for k-no-swap, gamma_r for k↔r swap). The pivot
/// is rejected via the ForceAccept path whenever
/// `|d| < max(zero_tol, pivot_threshold * col_max)`. This matches
/// MUMPS dfac_front_aux.F:1494-1495 and SSIDS options%u semantics:
/// the pivot must dominate its column by at least 1/u, otherwise the
/// rank-1 update would amplify rounding by ~1/|d| per position.
#[allow(clippy::too_many_arguments)]
fn do_1x1_pivot(
    a: &mut [f64],
    n: usize,
    k: usize,
    col_max: f64,
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
) -> Result<(f64, usize), FeralError> {
    let mut d = a[k * n + k];
    let threshold = (params.pivot_threshold * col_max).max(params.null_pivot_tol);

    if d.abs() <= threshold {
        // Pivot rejected (either absolute floor or column-relative
        // Duff-Reid/MUMPS threshold). Three-way split mirrors
        // try_reject_1x1_frontal: strict-zero (zero L), rank-deficiency
        // band (count zero but keep factor live), or small-but-real
        // (count by sign).
        if d.abs() <= params.zero_tol {
            // Truly-zero path: zero L, route by on_zero_pivot.
            match params.on_zero_pivot {
                ZeroPivotAction::ForceAccept => {
                    *needs_refinement = true;
                    *zero += 1;
                    for i in (k + 1)..n {
                        a[k * n + i] = 0.0;
                    }
                    // Also zero the diagonal so solve's `|d| > zero_tol`
                    // check skips this position.
                    a[k * n + k] = 0.0;
                    return Ok((0.0, k + 2));
                }
                ZeroPivotAction::Fail => return Err(FeralError::NumericallyRankDeficient),
                ZeroPivotAction::PerturbToEps { abs_floor } => {
                    d = perturb_to_floor(d, abs_floor);
                    a[k * n + k] = d;
                    *needs_refinement = true;
                    if d > 0.0 {
                        *pos += 1;
                    } else {
                        *neg += 1;
                    }
                    // Fall through to L scaling + rank-1 update with the
                    // perturbed `d` already counted.
                }
            }
        } else if matches!(params.on_zero_pivot, ZeroPivotAction::ForceAccept)
            && d.abs() <= params.null_pivot_tol
        {
            // Rank-deficiency band (F-01): count zero, keep d and L
            // live so the trailing update fires and the solve can
            // divide by the small-but-real pivot.
            *needs_refinement = true;
            *zero += 1;
        } else {
            // Small but real (case b): accept with sign.
            *needs_refinement = true;
            if d > 0.0 {
                *pos += 1;
            } else {
                *neg += 1;
            }
        }
    } else {
        // Accept: count inertia by sign.
        if d > 0.0 {
            *pos += 1;
        } else {
            *neg += 1;
        }
    }

    let d_inv = 1.0 / d;

    // Compute L column entries: L[i,k] = A[i,k] / d
    for i in (k + 1)..n {
        a[k * n + i] *= d_inv;
    }

    let mut next_gamma0 = 0.0;
    let mut next_r = k + 2;

    // Fused rank-1 update + argmax of next column (k+1).
    // Column k+1 is only updated in the j=k+1 iteration, so we handle it
    // separately to track the argmax during the same memory pass.
    if k + 1 < n {
        let j = k + 1;
        let l_jk = a[k * n + j];
        let l_jk_d = l_jk * d;
        // Update diagonal
        a[j * n + j] -= a[k * n + j] * l_jk_d;
        // Update off-diagonal and track argmax
        for i in (j + 1)..n {
            a[j * n + i] -= a[k * n + i] * l_jk_d;
            let val = a[j * n + i].abs();
            if val > next_gamma0 {
                next_gamma0 = val;
                next_r = i;
            }
        }
    }

    // Remaining columns: plain update (no argmax tracking)
    for j in (k + 2)..n {
        let l_jk = a[k * n + j];
        let l_jk_d = l_jk * d;
        for i in j..n {
            a[j * n + i] -= a[k * n + i] * l_jk_d;
        }
    }

    Ok((next_gamma0, next_r))
}

/// Perform a 2×2 pivot at positions {k, k+1}.
/// Uses the normalized computation from faer to avoid catastrophic cancellation.
/// Returns (gamma0_next, r_next) for the next pivot step's column.
#[allow(clippy::too_many_arguments)]
fn do_2x2_pivot(
    a: &mut [f64],
    n: usize,
    k: usize,
    subdiag: &mut [f64],
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
) -> Result<(f64, usize), FeralError> {
    let a00 = a[k * n + k];
    let a10 = a[k * n + (k + 1)];
    let a11 = a[(k + 1) * n + (k + 1)];

    // Store the 2×2 block subdiagonal
    subdiag[k] = a10;

    // Count inertia from the 2×2 block
    let det = a00 * a11 - a10 * a10;
    count_2x2_inertia(det, a00, a11, params, pos, neg, zero, needs_refinement)?;

    if (k + 2) >= n {
        // No trailing submatrix to update
        return Ok((0.0, 0));
    }

    // Normalized 2×2 computation (from faer, Section 4.3 of research note)
    let d10_abs = a10.abs();

    if d10_abs < f64::EPSILON * 1e-10 {
        // Degenerate 2×2 block — treat L columns as zero
        for i in (k + 2)..n {
            a[k * n + i] = 0.0;
            a[(k + 1) * n + i] = 0.0;
        }
        return Ok((0.0, k + 3));
    }

    let d00 = a00 / d10_abs;
    let d11 = a11 / d10_abs;
    let t = 1.0 / (d00 * d11 - 1.0);
    let d10 = a10 / d10_abs; // sign only (±1 for reals)
    let d = t / d10_abs;

    let mut next_gamma0 = 0.0;
    let mut next_r = k + 3;

    // Fused rank-2 update + argmax of next column (k+2).
    // Column k+2 is only updated in the j=k+2 iteration, so handle separately.
    if k + 2 < n {
        let j = k + 2;
        let x0 = a[k * n + j];
        let x1 = a[(k + 1) * n + j];
        let w0 = (x0 * d11 - x1 * d10) * d;
        let w1 = (x1 * d00 - x0 * d10) * d;

        // Update diagonal
        a[j * n + j] -= a[k * n + j] * w0 + a[(k + 1) * n + j] * w1;
        // Update off-diagonal and track argmax for column k+2
        for i in (j + 1)..n {
            a[j * n + i] -= a[k * n + i] * w0 + a[(k + 1) * n + i] * w1;
            let val = a[j * n + i].abs();
            if val > next_gamma0 {
                next_gamma0 = val;
                next_r = i;
            }
        }

        a[k * n + j] = w0;
        a[(k + 1) * n + j] = w1;
    }

    // Remaining columns: plain update (no argmax tracking)
    for j in (k + 3)..n {
        let x0 = a[k * n + j];
        let x1 = a[(k + 1) * n + j];
        let w0 = (x0 * d11 - x1 * d10) * d;
        let w1 = (x1 * d00 - x0 * d10) * d;

        for i in j..n {
            a[j * n + i] -= a[k * n + i] * w0 + a[(k + 1) * n + i] * w1;
        }

        a[k * n + j] = w0;
        a[(k + 1) * n + j] = w1;
    }

    Ok((next_gamma0, next_r))
}

/// Count inertia for a 1×1 pivot at position `k` (column-major stride
/// `stride`). When the pivot is below `zero_tol` and the configured
/// `ZeroPivotAction` is `PerturbToEps`, the diagonal entry at
/// `a[k*stride + k]` is overwritten with the perturbed pivot in place
/// so the D-block solve will divide by the bounded magnitude.
#[allow(clippy::too_many_arguments)]
fn count_1x1_inertia(
    a: &mut [f64],
    stride: usize,
    k: usize,
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
) -> Result<(), FeralError> {
    let d = a[k * stride + k];
    if d.abs() <= params.zero_tol {
        match params.on_zero_pivot {
            ZeroPivotAction::ForceAccept => {
                *needs_refinement = true;
                *zero += 1;
                Ok(())
            }
            ZeroPivotAction::Fail => Err(FeralError::NumericallyRankDeficient),
            ZeroPivotAction::PerturbToEps { abs_floor } => {
                let d_new = perturb_to_floor(d, abs_floor);
                a[k * stride + k] = d_new;
                *needs_refinement = true;
                if d_new > 0.0 {
                    *pos += 1;
                } else {
                    *neg += 1;
                }
                Ok(())
            }
        }
    } else if matches!(params.on_zero_pivot, ZeroPivotAction::ForceAccept)
        && d.abs() <= params.null_pivot_tol
    {
        // F-01 rank-deficiency band: count as zero, leave d intact so
        // the solve can divide (solve checks the strict factors.zero_tol).
        *needs_refinement = true;
        *zero += 1;
        Ok(())
    } else if d > 0.0 {
        *pos += 1;
        Ok(())
    } else {
        *neg += 1;
        Ok(())
    }
}

/// Count inertia for a 2×2 pivot block.
/// Uses determinant and trace of the block to classify eigenvalue signs
/// per Sylvester's law (matches `count_2x2_inertia_val` and the rmumps /
/// canonical-MUMPS conventions).
#[allow(clippy::too_many_arguments)]
fn count_2x2_inertia(
    det: f64,
    a00: f64,
    a11: f64,
    params: &BunchKaufmanParams,
    pos: &mut usize,
    neg: &mut usize,
    zero: &mut usize,
    needs_refinement: &mut bool,
) -> Result<(), FeralError> {
    // 2026-04-27 (Fix B in dev/research/2x2-bk-inertia-accounting.md):
    // switched the same-sign branch from `a00 > 0` to `trace > 0`. KKT
    // matrices produce 2×2 blocks where `a00 == 0` (variable rows have
    // zero Hessian diagonal) but `a11` carries the sign — the old rule
    // mis-attributed those. The trace-based rule matches canonical
    // Fortran MUMPS on ACOPP30_0000 (see dev/journal/2026-04-12-01.org
    // §15:45 "MAJOR FINDING"). The 2026-04-12 attempt was abandoned
    // because it regressed 16 dense matrices vs rmumps; the regression
    // was against an outdated rmumps oracle, not against canonical
    // MUMPS.
    let trace = a00 + a11;
    if det.abs() <= params.zero_tol_2x2 {
        // Near-singular 2×2 block: one eigenvalue near zero, the other
        // has sign of trace.
        match params.on_zero_pivot {
            ZeroPivotAction::ForceAccept | ZeroPivotAction::PerturbToEps { .. } => {
                // PerturbToEps falls back to ForceAccept-style
                // accounting for a near-singular 2×2 block: count one
                // sign + one zero. A bounded-Δ perturbation of the
                // block determinant would require modifying the
                // already-applied 2×2 update, which is out of scope
                // for the present pass. Callers driving PerturbToEps
                // should rely on the 1×1 paths for bounded perturbation
                // and treat 2×2 near-singular blocks as residual cases
                // for iterative refinement.
                *needs_refinement = true;
                if trace > 0.0 {
                    *pos += 1;
                    *zero += 1;
                } else {
                    *neg += 1;
                    *zero += 1;
                }
                Ok(())
            }
            ZeroPivotAction::Fail => Err(FeralError::NumericallyRankDeficient),
        }
    } else if matches!(params.on_zero_pivot, ZeroPivotAction::ForceAccept)
        && det.abs() <= params.null_pivot_tol_2x2
    {
        // F-01 rank-deficiency band for 2×2 blocks: count one zero, the
        // other by trace sign. The block update has already fired and
        // the solve will divide using strict factors.zero_tol_2x2.
        *needs_refinement = true;
        if trace > 0.0 {
            *pos += 1;
            *zero += 1;
        } else {
            *neg += 1;
            *zero += 1;
        }
        Ok(())
    } else if det > 0.0 {
        // Same-sign eigenvalues: sign comes from trace.
        if trace > 0.0 {
            *pos += 2;
        } else {
            *neg += 2;
        }
        Ok(())
    } else {
        // Opposite-sign eigenvalues.
        *pos += 1;
        *neg += 1;
        Ok(())
    }
}

/// Set L column at position k to the identity column (1 on diagonal, 0 below).
fn set_l_column_identity(a: &mut [f64], n: usize, k: usize) {
    for i in (k + 1)..n {
        a[k * n + i] = 0.0;
    }
}

#[cfg(test)]
mod growth_flag_tests {
    use super::*;

    #[test]
    fn growth_below_threshold_does_not_flag() {
        let l = vec![1.0, 2.78, -2.5, 0.0, 100.0, -999_999.0];
        let mut flag = false;
        flag_growth_for_refinement(&l, &mut flag);
        assert!(!flag, "max|L| = 999_999 < 1e6 should not flag");
    }

    #[test]
    fn growth_above_threshold_flags() {
        let l = vec![1.0, 2.0, 1.5e6, -3.0];
        let mut flag = false;
        flag_growth_for_refinement(&l, &mut flag);
        assert!(flag, "max|L| = 1.5e6 > 1e6 must flag");
    }

    #[test]
    fn catastrophic_growth_flags() {
        let l = vec![1.0, 1.0, 8.06e16, 1.0];
        let mut flag = false;
        flag_growth_for_refinement(&l, &mut flag);
        assert!(flag, "max|L| = 8e16 (bratu3d-class) must flag");
    }

    #[test]
    fn negative_large_entry_flags() {
        let l = vec![-2e10, 1.0];
        let mut flag = false;
        flag_growth_for_refinement(&l, &mut flag);
        assert!(flag, "negative large |L| must flag");
    }

    #[test]
    fn already_set_flag_is_preserved() {
        let l = vec![0.0, 0.0]; // would not flag on its own
        let mut flag = true; // pre-set by zero-pivot path
        flag_growth_for_refinement(&l, &mut flag);
        assert!(flag, "must not clobber pre-set flag");
    }

    #[test]
    fn empty_l_does_not_flag() {
        let l: Vec<f64> = vec![];
        let mut flag = false;
        flag_growth_for_refinement(&l, &mut flag);
        assert!(!flag);
    }

    #[test]
    fn nan_and_inf_in_l_flag() {
        // NaN.abs() is NaN; NaN > x is always false, so NaN alone does
        // not trigger. But Inf does, and a real factor with NaN almost
        // always also has Inf. This is an explicit doc of behavior.
        let l_inf = vec![1.0, f64::INFINITY];
        let mut flag = false;
        flag_growth_for_refinement(&l_inf, &mut flag);
        assert!(flag, "Inf entry must trigger");
    }
}
