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

pub mod dense_factor;
pub mod dense_matrix;
pub mod dense_solve;
pub mod dense_update;
pub mod sparse_factor;
pub mod sparse_matrix;
pub mod sparse_solve;
pub mod sparse_symbolic;

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
    let cells = m * m;
    nnz * 4 >= cells
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
