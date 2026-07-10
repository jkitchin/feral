//! Unsymmetric LU factorization family for simplex basis factorization
//! (issue #81).
//!
//! This is a **separate** factorization family from feral's symmetric LDLᵀ
//! solver. An LU basis is unsymmetric and square, factored as `P B Q = L U`
//! with threshold partial pivoting (row permutation `P`) and, on the sparse
//! path, a fill-reducing column permutation `Q`. There is no inertia.
//!
//! The design target is a revised-simplex basis engine: cheap rank-1
//! column-replacement updates and warm `ftran` (`B⁻¹a`) / `btran` (`B⁻ᵀa`)
//! solves, not just one-shot factor/solve. See `dev/research/unsymmetric-lu.md`
//! and `dev/plans/unsymmetric-lu-epic.md`.

pub mod condition;
pub mod dense_factor;
pub mod dense_matrix;
pub mod dense_solve;
pub mod dense_update;
pub mod scaling;
pub mod sparse_factor;
pub mod sparse_matrix;
pub mod sparse_solve;
pub mod sparse_symbolic;
pub mod sparse_update;

pub use dense_factor::DenseLu;
pub use dense_matrix::GeneralMatrix;
pub use sparse_factor::SparseLu;
pub use sparse_matrix::SparseColMatrix;
pub use sparse_symbolic::SparseLuSymbolic;

/// What to do when the LU factorization hits a numerically null pivot column.
///
/// Mirrors [`crate::dense::factor::ZeroPivotAction`] for the symmetric side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LuSingularAction {
    /// Return [`FeralError::SingularBasis`](crate::error::FeralError::SingularBasis).
    Fail,
    /// Replace the pivot with `sign(d) · max(|d|, abs_floor)` and continue.
    PerturbToEps {
        /// Lower bound applied to the perturbed pivot magnitude.
        abs_floor: f64,
    },
}

/// Why a rank-1 [`SparseLu::update`]/[`DenseLu::update`] gave up and returned
/// [`FeralError::NeedsRefactor`](crate::error::FeralError::NeedsRefactor).
///
/// `update()` still returns the payload-free `Err(NeedsRefactor)` (additive,
/// non-breaking); the cause and a magnitude are recorded separately and read
/// back via [`SparseLu::last_refactor`]/[`DenseLu::last_refactor`]. This lets a
/// caller distinguish an **ill-conditioning** failure (`Growth`, `TinyPivot`,
/// `Singular`) — where iterative refine-and-retry is the right response — from a
/// mere **bookkeeping-budget** trip (`UpdateBudget`), where a plain refactor
/// suffices and refine-and-retry is wasted work (discopt#364). See issue #95.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefactorCause {
    /// The element-growth high-water ratio `‖U‖∞ / ‖U₀‖∞` exceeded
    /// [`LuParams::max_growth`]. Magnitude = the growth ratio that tripped.
    Growth,
    /// The update-count budget [`LuParams::max_updates`] was reached before this
    /// update. Magnitude = the update count that hit the cap (`= max_updates`).
    UpdateBudget,
    /// A bump/final diagonal pivot fell at or below `zero_pivot_tol · ‖U₀‖∞`, or
    /// was non-finite. Magnitude = `|pivot|` of the offending diagonal. On the
    /// dense path a linearly dependent replacement also surfaces here (it drives
    /// the final `U` diagonal to ~0); only the sparse path can report `Singular`
    /// distinctly.
    TinyPivot,
    /// The replacement column is linearly dependent on the retained basis: the
    /// sparse spike has no entry at or below its own diagonal in triangular-rank
    /// order, so no bump pivot exists. Magnitude = `0.0`. Sparse-only (the dense
    /// path reports this as `TinyPivot`).
    Singular,
}

/// Two-sided scaling strategy for the LU basis (issue #81 robustness layer).
///
/// Mirrors the spirit of [`crate::scaling::ScalingStrategy`] but keeps the row
/// and column scalings separate (the basis is unsymmetric).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LuScaling {
    /// No scaling.
    None,
    /// Two-sided Knight–Ruiz ∞-norm equilibration.
    InfNorm,
    /// Unsymmetric MC64 (max-weight bipartite matching) scaling + row
    /// permutation, with a partial-matching fall back to `InfNorm`.
    Mc64,
    /// MC64 first, then ∞-norm equilibration of the matched matrix.
    Mc64ThenInfNorm,
}

/// Tuning parameters for the LU factorization and its updates.
///
/// Mirrors the shape of [`crate::dense::factor::BunchKaufmanParams`].
#[derive(Debug, Clone)]
pub struct LuParams {
    /// Threshold partial-pivoting parameter `u ∈ (0, 1]`. `1.0` is strict
    /// partial pivoting (max stability); smaller values permit a sparser /
    /// closer-to-diagonal pivot when it is within `u·max` of the column max.
    ///
    /// Governs the **initial factorization** (both the dense and sparse paths
    /// honor it: the sparse path prefers the within-threshold diagonal row,
    /// matching CSparse `cs_lu`). The Forrest–Tomlin bump elimination in
    /// `update()` does not consult it: the default FT sweep keeps the retained
    /// pivot order (there is no pivot choice to threshold), and the opt-in
    /// pivot-searching variant ([`LuParams::update_pivot_search`]) uses
    /// strict-larger interchanges (`u = 1`) for maximum stability.
    pub pivot_threshold: f64,
    /// A pivot column with all candidates `≤ zero_pivot_tol` is singular.
    pub zero_pivot_tol: f64,
    /// Action on a singular pivot column.
    pub on_singular: LuSingularAction,
    /// `update()` returns `NeedsRefactor` when the growth monitor exceeds this.
    pub max_growth: f64,
    /// Hard cap on `updates_since_refactor` before `update()` returns
    /// `NeedsRefactor`.
    pub max_updates: usize,
    /// Bases with `m` at or below this use the dense path unconditionally.
    pub dense_threshold: usize,
    /// Scaling strategy applied before factorization.
    pub scaling: LuScaling,
    /// Maximum iterative-refinement steps in `ftran_refined`/`btran_refined`.
    pub refine_steps: usize,
    /// Stop refinement when `‖r‖/‖a‖ < refine_tol`.
    pub refine_tol: f64,
    /// Run the sparse `update()` bump elimination with Bartels–Golub row
    /// interchanges (strict-larger pivot search) instead of the fixed
    /// Forrest–Tomlin pivot order (issue #112). With it on, every elimination
    /// multiplier is bounded by 1 (the classic BG stability guarantee), which
    /// keeps element growth bounded across long update chains at the cost of
    /// extra fill in the rows the interchanges rewrite. Off (the default),
    /// updates keep the plain FT order — cheapest, and already protected
    /// against the issue #112 cancellation-to-exact-zero by the always-on
    /// compensated accumulation in the sweep. Note a pivot search can only
    /// *prevent* instability from building up; it cannot recover a pivot the
    /// fixed order has already cancelled (any interchange order's working row
    /// is exactly proportional to the fixed order's — see
    /// `dev/research/issue-112-bg-update.md`), which is why this is a
    /// trajectory choice rather than a failure rescue. Sparse path only; the
    /// dense update is unaffected.
    pub update_pivot_search: bool,
}

impl LuParams {
    /// Reject parameters outside their documented ranges before any
    /// factorization consumes them. `pivot_threshold` must lie in `(0, 1]` (its
    /// documented `u` range): `0` would disable pivoting (always prefer the
    /// diagonal), `> 1` is meaningless, and `NaN` poisons every threshold
    /// comparison. `zero_pivot_tol` is a relative floor and must be finite and
    /// in `[0, 1)`: negative is nonsensical and `≥ 1` would declare every
    /// pivot singular. Validating here keeps the dense and sparse factor paths
    /// from drifting on the same bad input.
    pub(crate) fn validate(&self) -> Result<(), crate::error::FeralError> {
        use crate::error::FeralError;
        if !(self.pivot_threshold > 0.0 && self.pivot_threshold <= 1.0) {
            return Err(FeralError::InvalidInput(format!(
                "LuParams::pivot_threshold must be in (0, 1], got {}",
                self.pivot_threshold
            )));
        }
        if !(self.zero_pivot_tol >= 0.0 && self.zero_pivot_tol < 1.0) {
            return Err(FeralError::InvalidInput(format!(
                "LuParams::zero_pivot_tol must be in [0, 1), got {}",
                self.zero_pivot_tol
            )));
        }
        Ok(())
    }
}

impl Default for LuParams {
    fn default() -> Self {
        LuParams {
            pivot_threshold: 1.0,
            zero_pivot_tol: 1e-13,
            on_singular: LuSingularAction::Fail,
            max_growth: 1e8,
            max_updates: 64,
            dense_threshold: 128,
            scaling: LuScaling::None,
            refine_steps: 0,
            refine_tol: 1e-12,
            update_pivot_search: false,
        }
    }
}

/// Auto dense/sparse routing for an `m`×`m` basis with `nnz` stored entries.
///
/// Mirrors `crate::numeric::factorize::should_use_dense_fast_path`: tiny bases
/// always go dense; small dense-enough bases (density ≥ 1/4) go dense; the rest
/// go sparse. `dense_threshold` is the upper `m` bound for the density test.
pub fn should_use_dense_lu(m: usize, nnz: usize, params: &LuParams) -> bool {
    const M_TINY: usize = 16;
    if m == 0 {
        return false;
    }
    if m <= M_TINY {
        return true;
    }
    if m > params.dense_threshold {
        return false;
    }
    // Density gate: dense when at least a quarter of the cells are nonzero.
    // Saturating arithmetic so a caller-set large `dense_threshold` (which bounds
    // `m` above) can't overflow `usize` on `m*m` or `nnz*4`.
    let cells = m.saturating_mul(m);
    nnz.saturating_mul(4) >= cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_tiny_is_dense() {
        let p = LuParams::default();
        assert!(should_use_dense_lu(8, 0, &p));
        assert!(should_use_dense_lu(16, 0, &p));
    }

    #[test]
    fn router_large_is_sparse() {
        let p = LuParams::default();
        assert!(!should_use_dense_lu(1000, 1000 * 1000, &p));
    }

    #[test]
    fn router_density_gate() {
        let p = LuParams::default();
        // 64x64 = 4096 cells; dense iff nnz >= 1024.
        assert!(!should_use_dense_lu(64, 1023, &p));
        assert!(should_use_dense_lu(64, 1024, &p));
    }
}
