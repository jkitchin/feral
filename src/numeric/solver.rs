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
use crate::numeric::factorize::{
    factorize_multifrontal_with_workspace, FactorWorkspace, NumericParams, SparseFactors,
};
use crate::numeric::solve::{solve_sparse, solve_sparse_refined};
use crate::scaling::ScalingStrategy;
use crate::sparse::csc::CscMatrix;
use crate::symbolic::supernode::SupernodeParams;
use crate::symbolic::{symbolic_factorize, SymbolicFactorization};

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

/// Conservative sparsity-pattern fingerprint used to detect when the
/// cached `SymbolicFactorization` is stale. Two genuinely identical
/// patterns produce the same fingerprint by construction; collisions
/// between distinct patterns are possible but tolerated for the IPM
/// use case (the consumer factors structurally identical KKTs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PatternFingerprint {
    n: usize,
    col_ptr_len: usize,
    row_idx_len: usize,
}

impl PatternFingerprint {
    fn of(matrix: &CscMatrix) -> Self {
        Self {
            n: matrix.n,
            col_ptr_len: matrix.col_ptr.len(),
            row_idx_len: matrix.row_idx.len(),
        }
    }
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
    /// Pooled scratch for the numeric phase. Retained across
    /// `factor` calls so IPM-style re-factorizations (same
    /// pattern, new values; or bumped pivot threshold) do not
    /// re-allocate per-supernode buffers. Cleared to a
    /// well-defined initial state on every
    /// `factorize_multifrontal_with_workspace` entry, so stale
    /// data cannot leak between factor attempts.
    workspace: FactorWorkspace,
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
            workspace: FactorWorkspace::new(),
        }
    }

    /// Factor `matrix`. If `check_inertia` is `Some(expected)`,
    /// returns `WrongInertia { actual, expected }` on mismatch
    /// without invalidating the stored factor (caller may still
    /// `solve` against it). See plan §`factor()` flow.
    pub fn factor(&mut self, matrix: &CscMatrix, check_inertia: Option<Inertia>) -> FactorStatus {
        // Step 1: pattern fingerprint.
        let fp = PatternFingerprint::of(matrix);

        // Step 2: invalidate cache on pattern change.
        if self.last_pattern_fingerprint != Some(fp) {
            self.last_symbolic = None;
            self.last_factors = None;
            self.last_inertia = None;
            self.last_pattern_fingerprint = None;
        }

        // Step 3: ensure symbolic is cached.
        if self.last_symbolic.is_none() {
            match symbolic_factorize(matrix, &self.snode_params) {
                Ok(sym) => {
                    self.symbolic_call_count += 1;
                    self.last_symbolic = Some(sym);
                    self.last_pattern_fingerprint = Some(fp);
                }
                Err(e) => return FactorStatus::FatalError(e),
            }
        }
        // Safe: just-set above or already Some.
        let symbolic = match &self.last_symbolic {
            Some(s) => s,
            None => unreachable!("symbolic just populated"),
        };

        // Step 4: numeric factor via the pooled workspace; map errors.
        match factorize_multifrontal_with_workspace(
            matrix,
            symbolic,
            &self.numeric_params,
            &mut self.workspace,
        ) {
            Ok((factors, inertia)) => {
                // Step 5: stash; PartialSingular maps to Singular.
                let partial_singular = matches!(
                    factors.scaling_info,
                    crate::scaling::ScalingInfo::PartialSingular { .. }
                );
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
    /// case); else `min(pivtol_max, threshold^0.75)`. When the
    /// new threshold reaches `pivtol_max`, transition to
    /// `Exhausted` for the *next* call.
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
        };
        Solver::with_params(np, SupernodeParams::default())
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
}
