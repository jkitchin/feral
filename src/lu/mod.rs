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
pub(crate) mod sparse_hyper;
pub mod sparse_matrix;
pub mod sparse_solve;
pub mod sparse_symbolic;
pub(crate) mod sparse_triangular;
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
    /// Maximum dimension of a post-triangularization **bump** that
    /// [`SparseLu::factor`](super::SparseLu::factor) routes through the dense
    /// kernel instead of the sparse scatter kernel. `0` (the default) disables
    /// the route entirely.
    ///
    /// After a Suhl–Suhl peel the residual bump is small but its *factor* is
    /// dense — 42% on a measured QPLIB simplex basis whose bump was 566x566 at
    /// 2.2% **input** density — which is the worst case for a scalar sparse
    /// scatter kernel and the best case for a blocked dense one (6.05 ms vs
    /// 27.3 ms on that block). [`Self::dense_threshold`] cannot express this:
    /// it keys on input density and on the dimension of the *whole* basis.
    ///
    /// Cost of being wrong is bounded by memory: the route packs a `b*b`
    /// `f64` buffer, so `b = 1024` is 8 MB and `b = 4096` is 134 MB. Set this
    /// to the largest bump worth that allocation, or `0` to stay sparse.
    ///
    /// **Requires [`SparseLuSymbolic::analyze_triangularized`].** Only a
    /// symbolic that actually peeled
    /// ([`triangularized`](super::SparseLuSymbolic::triangularized)) can take
    /// this route. [`SparseLuSymbolic::analyze`] (the default),
    /// [`SparseLuSymbolic::natural`], [`SparseLuSymbolic::with_order`] and
    /// [`SparseLuSymbolic::analyze_amd_only`] report the whole basis as bump
    /// because they never looked for structure, so setting this cap alongside
    /// any of them is a no-op however large it is.
    ///
    /// Opting into the peel is worth having on its own — it is 4.2–9.8x on the
    /// symbolic step, which a simplex pays on every refactorization — but it
    /// also changes the rounding trajectory, and that is not free: see
    /// [`SparseLuSymbolic::analyze`] for both halves, including the
    /// ill-conditioned LP where it cost a downstream simplex its dual bound
    /// (issue #163) and the instance where it is a 2.6x slowdown. This cap is
    /// the other 4.28x, on the numeric side, and it needs the peel to fire.
    ///
    /// When `analyze_triangularized` peels nothing and the bump *is* the whole
    /// basis, this cap does not apply either — such a basis is bounded by
    /// [`Self::dense_threshold`] instead. The case for the dense kernel rests on
    /// the peel having stripped the structure and left an irreducible core; with
    /// nothing stripped, whole-basis dense is `dense_threshold`'s call, and it
    /// weighs density rather than dimension alone.
    ///
    /// [`SparseLuSymbolic::natural`]: super::SparseLuSymbolic::natural
    /// [`SparseLuSymbolic::with_order`]: super::SparseLuSymbolic::with_order
    /// [`SparseLuSymbolic::analyze_amd_only`]: super::SparseLuSymbolic::analyze_amd_only
    /// [`SparseLuSymbolic::analyze`]: super::SparseLuSymbolic::analyze
    /// [`SparseLuSymbolic::analyze_triangularized`]: super::SparseLuSymbolic::analyze_triangularized
    pub dense_bump_max_dim: usize,
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
    /// Density cap for the reach-limited ("hyper-sparse") triangular solves in
    /// the sparse `ftran`/`btran` (issue #161B, Hall & McKinnon 2005).
    ///
    /// The gather-form halves of the solve (`U w = s` in `ftran`, `Lᵀ v = s` in
    /// `btran`) read every row of `U` / every column of `L` on every call, so
    /// their cost tracks `nnz(factor)` rather than the number of nonzeros the
    /// solution actually has. With this set, a solve first computes the reach
    /// of the right-hand side's pattern in the factor's DAG and sweeps only
    /// those positions — the work a simplex `ftran`/`btran` against a
    /// near-unit-vector rhs actually needs.
    ///
    /// The value is the fraction of `m` at or below which the reach-limited
    /// route is taken; a reach larger than `hyper_sparse_max_density · m`
    /// aborts back to the dense sweep, so a *sparse rhs whose solution fills
    /// in* pays up to this fraction of a wasted sweep on top of the dense one.
    /// That bounded downside is the price of routing on solution density, which
    /// cannot be known without computing part of the reach.
    ///
    /// # Why the default is 0.10 and not higher
    ///
    /// **The reach DFS costs about what the sweep it replaces costs.** Walking
    /// the graph to find the reach traverses the same factor entries the numeric
    /// sweep then traverses again, so the route pays roughly twice to avoid
    /// paying once. That is a fine trade when the reach is small and a losing
    /// one when it is a sizeable fraction of `m`.
    ///
    /// Measured on QPLIB_1157 (`tests/data/lu_bases/`): at a cap of 0.25 the
    /// `btran` reach walks **68,925 graph edges per sweep against
    /// `nnz(L) = 75,084`** — 92% of what the dense `Lᵀ` sweep traverses — and
    /// the numeric sweep then walks the reached columns on top of that. At 0.10
    /// the same figure is 5%. (The `O(r log r)` sort is *not* the cost: sorting
    /// even 2000 positions is ~18 us against a ~105 us per-solve regression.)
    ///
    /// Measured on the real QPLIB_1157 simplex basis (`m = 3937`, 7.46 nnz/col,
    /// fill 6.76x), sweeping only this cap:
    ///
    /// | cap | `ftran` | `btran` | route fired |
    /// |---|---|---|---|
    /// | 0.05 | 10.59x | 0.98x | 1195 |
    /// | **0.10** | **10.81x** | **0.97x** | **1195** |
    /// | 0.25 | 11.01x | **0.69x** | 2370 |
    /// | 1.00 | 10.75x | **0.69x** | 2560 |
    ///
    /// `ftran` is flat across the whole range; `btran` falls off a cliff
    /// between 0.10 and 0.25, and the fired count nearly doubles over the same
    /// step. That basis's `btran` reaches sit between 10% and 25% of `m`, so a
    /// cap of 0.25 admits exactly the population the sort loses on — a 1.45x
    /// regression bought for no `ftran` gain.
    ///
    /// 0.10 is therefore the largest cap measured that still excludes that
    /// band. Raising it needs evidence from a basis whose solutions actually
    /// live there and *win*, not a synthetic fixture: this effect is invisible
    /// unless the solution-density profile straddles the cap, which is why the
    /// in-tree generator could not show it and the shipped default was wrong
    /// until real bases were measured.
    ///
    /// `0.0` disables the route entirely — every solve takes the dense sweep,
    /// exactly as before issue #161, and the `L` row index and reach workspace
    /// are not even allocated. Valid range `[0, 1]`.
    pub hyper_sparse_max_density: f64,
}

impl LuParams {
    /// Reject parameters outside their documented ranges before any
    /// factorization consumes them. `pivot_threshold` must lie in `(0, 1]` (its
    /// documented `u` range): `0` would disable pivoting (always prefer the
    /// diagonal), `> 1` is meaningless, and `NaN` poisons every threshold
    /// comparison. `zero_pivot_tol` is a relative floor and must be finite and
    /// in `[0, 1)`: negative is nonsensical and `≥ 1` would declare every
    /// pivot singular. `max_growth` must be `> 1.0` (finite or `+∞`; `NaN` or
    /// `≤ 1.0` rejected) and `refine_tol` finite and `> 0` — both otherwise
    /// silently disable a guard (issue #122C). Validating here keeps the dense
    /// and sparse factor paths from drifting on the same bad input.
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
        // `max_growth` gates the update refactor trigger (`growth > max_growth`).
        // A `NaN` makes that comparison always false, silently disabling the
        // growth guard in the update paths (which — unlike `should_refactor_growth`
        // — do not defend with `is_finite`); `≤ 1.0` rejects every update since
        // the monitored growth is ≥ 1. `+∞` is the documented "never trigger on
        // growth" opt-out and passes (`+∞ > 1.0`); `NaN` does not (`NaN > 1.0`
        // is false). Issue #122C.
        if self.max_growth.is_nan() || self.max_growth <= 1.0 {
            return Err(FeralError::InvalidInput(format!(
                "LuParams::max_growth must be > 1.0 (finite or +inf), got {}",
                self.max_growth
            )));
        }
        // `refine_tol` is the relative-residual stop for iterative refinement.
        // `NaN` breaks the `‖r‖/‖a‖ < refine_tol` convergence check (always
        // false → refinement runs to `refine_steps` doing no useful work or
        // spinning); `≤ 0` can never be met; `+∞` would accept any residual.
        // Require finite and strictly positive. Issue #122C.
        if !self.refine_tol.is_finite() || self.refine_tol <= 0.0 {
            return Err(FeralError::InvalidInput(format!(
                "LuParams::refine_tol must be finite and > 0, got {}",
                self.refine_tol
            )));
        }
        // `hyper_sparse_max_density` scales to a node budget `d·m`. A negative
        // or `NaN` value makes that budget meaningless (`NaN as usize` is 0 in
        // Rust, which would silently disable the route rather than error);
        // above 1.0 the cap exceeds `m` and can never abort, which defeats the
        // fallback the route relies on. `0.0` is the documented off switch.
        if !(self.hyper_sparse_max_density >= 0.0 && self.hyper_sparse_max_density <= 1.0) {
            return Err(FeralError::InvalidInput(format!(
                "LuParams::hyper_sparse_max_density must be in [0, 1], got {}",
                self.hyper_sparse_max_density
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
            dense_bump_max_dim: 0,
            scaling: LuScaling::None,
            refine_steps: 0,
            refine_tol: 1e-12,
            update_pivot_search: false,
            hyper_sparse_max_density: 0.10,
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

    /// Issue #122C: `validate()` must reject `max_growth` and `refine_tol`
    /// values that silently disable a guard, and accept the documented
    /// boundary/opt-out values. The default is valid.
    #[test]
    fn validate_max_growth_and_refine_tol() {
        use crate::error::FeralError;
        assert!(LuParams::default().validate().is_ok());

        let bad_growth = |g: f64| LuParams {
            max_growth: g,
            ..LuParams::default()
        };
        // NaN disables `growth > max_growth` silently → reject.
        assert!(matches!(
            bad_growth(f64::NAN).validate(),
            Err(FeralError::InvalidInput(_))
        ));
        // ≤ 1.0 rejects every update (growth ≥ 1) → reject.
        assert!(matches!(
            bad_growth(1.0).validate(),
            Err(FeralError::InvalidInput(_))
        ));
        assert!(matches!(
            bad_growth(0.5).validate(),
            Err(FeralError::InvalidInput(_))
        ));
        // Just past 1.0 and the +∞ opt-out are both documented-valid.
        assert!(bad_growth(1.0 + 1e-12).validate().is_ok());
        assert!(bad_growth(f64::INFINITY).validate().is_ok());

        let bad_tol = |t: f64| LuParams {
            refine_tol: t,
            ..LuParams::default()
        };
        // NaN / ≤ 0 / +∞ all break the convergence check → reject.
        for t in [f64::NAN, 0.0, -1e-12, f64::INFINITY] {
            assert!(
                matches!(bad_tol(t).validate(), Err(FeralError::InvalidInput(_))),
                "refine_tol {t} must be rejected"
            );
        }
        // A small positive finite tol is valid.
        assert!(bad_tol(1e-14).validate().is_ok());
    }
}
