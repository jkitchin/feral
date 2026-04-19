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
use crate::numeric::factorize::{NumericParams, SparseFactors};
use crate::sparse::csc::CscMatrix;
use crate::symbolic::supernode::SupernodeParams;
use crate::symbolic::SymbolicFactorization;

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
    // Step-1 skeleton: called from `factor()` in Step 2.
    #[allow(dead_code)]
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
// Step-1 skeleton: most fields are populated/read in Steps 2-6.
#[allow(dead_code)]
pub struct Solver {
    numeric_params: NumericParams,
    snode_params: SupernodeParams,
    pivtol_max: f64,
    quality_level: QualityLevel,
    last_symbolic: Option<SymbolicFactorization>,
    last_factors: Option<SparseFactors>,
    last_inertia: Option<Inertia>,
    last_pattern_fingerprint: Option<PatternFingerprint>,
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
        }
    }

    /// Factor `matrix`. If `check_inertia` is `Some(expected)`,
    /// returns `WrongInertia { actual, expected }` on mismatch
    /// without invalidating the stored factor (caller may still
    /// `solve` against it). See plan §`factor()` flow.
    pub fn factor(&mut self, _matrix: &CscMatrix, _check_inertia: Option<Inertia>) -> FactorStatus {
        unimplemented!("Solver::factor — implemented in Step 2")
    }

    /// Solve `A x = b` against the most recent successful factor.
    /// Returns `FeralError::NoFactor` if no factor is stored.
    pub fn solve(&self, _rhs: &[f64]) -> Result<Vec<f64>, FeralError> {
        unimplemented!("Solver::solve — implemented in Step 6")
    }

    /// Solve with iterative refinement against the original matrix.
    /// Returns `FeralError::NoFactor` if no factor is stored.
    pub fn solve_refined(&self, _matrix: &CscMatrix, _rhs: &[f64]) -> Result<Vec<f64>, FeralError> {
        unimplemented!("Solver::solve_refined — implemented in Step 6")
    }

    /// Two-stage quality escalation. Persistent across `factor()`
    /// calls. Returns `false` when both stages are exhausted.
    /// Mirrors `IpTSymLinearSolver::IncreaseQuality`.
    pub fn increase_quality(&mut self) -> bool {
        unimplemented!("Solver::increase_quality — implemented in Step 5")
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
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}
