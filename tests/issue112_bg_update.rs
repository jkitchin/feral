//! Issue #112: the Forrest–Tomlin bump update failed with `TinyPivot` at
//! magnitude **exactly 0.0** on nonsingular bases — the fixed-order sweep's
//! plain accumulation absorbed the true pivot's bits once an intermediate
//! grew past `|true pivot|/ε`. The fix is Neumaier-compensated accumulation
//! in the sweep (always on); the opt-in Bartels–Golub pivot search
//! (`LuParams::update_pivot_search`) additionally bounds every multiplier by
//! 1 so instability cannot build across update chains. These tests pin both.
//!
//! Oracles (no same-session self-oracle): the regression basis is a **hand
//! calculation** (full arithmetic trace at `absorption_basis`, verified
//! offline in exact rational arithmetic — the committed diagonal is asserted
//! bit-for-bit against the hand value), and accuracy is measured by the
//! oracle-free equation-residual identity `‖B'x − b‖` under the backward
//! -stability bound (the issue's "accuracy preserved" acceptance).

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{LuParams, SparseLu, SparseLuSymbolic};
use feral::FeralError;

const M: usize = 4;
/// Legal tiny retained diagonals that amplify the fixed-order sweep.
const T: f64 = 9.5367431640625e-7; // 2^-20

/// Hand-built m=4 basis whose fixed-order FT bump elimination drives the
/// working row to `2²⁰`-scale intermediates and cancels the final diagonal to
/// **exactly 0.0** under plain accumulation, while the true value is
/// `2⁻³⁵ ≠ 0` (so the replacement basis is nonsingular). Upper triangular,
/// entries in `[2⁻²⁰, 1]` (so under natural ordering `L = I`, `U = B`,
/// `perm = qcol = id`, `u_max0 = 1`, `ztol = 1e-13`, and the tiny diagonals
/// are forced legal pivots — each column's only unpivoted row is its own).
///
/// Replacing column 0 by `a = (2 + δ, 1, −(0.5 − 2⁻²¹), 1)` with `δ = 2⁻³⁵`
/// gives spike = `a` (since `L = I`) and the sweep (hand trace, exact unless
/// noted; verified offline in exact rational arithmetic):
///
/// - rank 0 (col 1): `vrc = B[0][1] = 1`, `piv = 2⁻²⁰` ⇒ `mult = 2²⁰`
///   (exact power of two). Fill: `rw[2] −= 2²⁰·B[1][2] = −2²⁰·(−2⁻²⁰) = +1`
///   ⇒ `rw[2] = 2`; spike hit: `rw[0] −= 2²⁰·a[1] = 2²⁰`:
///   `fl(2 + 2⁻³⁵ − 2²⁰) = −(2²⁰ − 2)` — the true value needs a bit at
///   `2⁻³⁵ = ulp/4` of that binade, so a plain round-to-nearest sum
///   **absorbs the δ exactly** (the compensated sum keeps it);
/// - rank 1 (col 2): `vrc = 2`, `piv = 2⁻²⁰` ⇒ `mult = 2²¹`; spike hit:
///   `rw[0] −= 2²¹·(−(0.5 − 2⁻²¹)) = +(2²⁰ − 1)` (exact) ⇒ plain `rw[0] = 1`
///   (exact);
/// - rank 2 (col 3): `vrc = 1`, `piv = 1`; spike hit `rw[0] −= 1·1` ⇒
///   plain-sum final diagonal = **0.0 exactly**; compensated = `2⁻³⁵` exactly.
///
/// Nonsingularity: `det(B') = ±(∏ retained diagonals)·2⁻³⁵ ≠ 0` in exact
/// arithmetic (hand calculation). Numerically, `σ_min(B') ≈ δ·∏(tiny
/// diagonals)` — inherent to *any* single-shot distilled absorption case —
/// so a from-scratch factorization sees an ε-relative last pivot; see
/// `updated_factors_solve_backward_stably` for how accuracy is asserted.
fn absorption_basis() -> Vec<Vec<f64>> {
    let mut cols = vec![vec![0.0; M]; M];
    cols[0][0] = 1.0;
    cols[1][0] = 1.0;
    cols[1][1] = T;
    cols[2][0] = 1.0;
    cols[2][1] = -T;
    cols[2][2] = T;
    cols[3][0] = 1.0;
    cols[3][3] = 1.0;
    cols
}

/// True value of the bump's final diagonal after the replacement below.
const TRUE_DIAG: f64 = 2.9103830456733704e-11; // 2^-35, the absorbed bit

/// The entering column. `head_extra = 2⁻³⁵` gives the nonsingular
/// cancellation case; `head_extra = 0.0` makes the same cancellation
/// mathematically exact — a genuinely singular replacement.
fn entering_column(head_extra: f64) -> Vec<f64> {
    vec![2.0 + head_extra, 1.0, -(0.5 - 2f64.powi(-21)), 1.0]
}

fn replaced_cols(head_extra: f64) -> Vec<Vec<f64>> {
    let mut cols = absorption_basis();
    cols[0] = entering_column(head_extra);
    cols
}

fn factor(cols: &[Vec<f64>], params: LuParams) -> SparseLu {
    let a = SparseColMatrix::from_dense_columns(M, cols).expect("matrix");
    SparseLu::factor(&a, &SparseLuSymbolic::natural(M), params).expect("factor")
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()))
}

/// `‖B x − b‖∞` for dense columns.
fn residual(cols: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
    let mut r = b.to_vec();
    for (j, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            r[i] -= v * x[j];
        }
    }
    inf_norm(&r)
}

/// `‖Bᵀ x − b‖∞` for dense columns.
fn residual_t(cols: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
    let mut r = b.to_vec();
    for (j, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            r[j] -= v * x[i];
        }
    }
    inf_norm(&r)
}

/// The compensated sweep must commit the update the plain sweep cancelled to
/// an exact-zero pivot (the issue #112 fingerprint), recovering the true
/// final diagonal **exactly**, with no refactorization and no stale failure
/// cause. Pre-fix this update returned `NeedsRefactor` with
/// `last_refactor() == Some((TinyPivot, 0.0))`.
#[test]
fn compensated_sweep_recovers_exactly_cancelled_pivot() {
    let mut lu = factor(&absorption_basis(), LuParams::default());
    lu.update(0, &entering_column(TRUE_DIAG))
        .expect("the compensated sweep must absorb the nonsingular replacement");
    assert_eq!(lu.updates_since_refactor(), 1, "committed as an update");
    assert_eq!(lu.last_refactor(), None, "no failure cause after a commit");
    assert_eq!(
        lu.pivot_search_swaps(),
        0,
        "the default path must not have used the pivot search"
    );
    // The committed bump diagonal is the hand-computed true value, exactly:
    // the Neumaier compensation loses nothing here (all inputs are exact
    // powers-of-two combinations).
    let diag = lu.u_dense(0, 0);
    assert_eq!(
        diag, TRUE_DIAG,
        "committed diagonal must be the true absorbed bit 2^-35"
    );
}

/// Accuracy of the committed factors: ftran and btran on the updated
/// factorization are **backward stable** — residual `≤ ε·m·‖B'‖∞·‖x‖∞` — on
/// the replacement basis (the issue's "accuracy preserved" acceptance).
///
/// Note on the issue's literal "residual ≤ the from-scratch refactor's":
/// a *single-shot* distilled absorption case is necessarily within `~δ` of a
/// singular matrix (`σ_min(B') ≤ δ·∏retained`), so a from-scratch
/// factorization of this B' sees an ε-relative last pivot and reports
/// `SingularBasis` — there is no refactor residual to compare against; the
/// updated factors here are in fact **exact** (every sweep value is an exact
/// power-of-two combination, and the compensated diagonal equals the true
/// `2⁻³⁵` bit-for-bit — asserted above). Matching the captures' healthy
/// `σ_min` needs the multi-update imbalance history only the real corpus
/// has; see `dev/research/issue-112-bg-update.md` §UPDATE.
#[test]
fn updated_factors_solve_backward_stably() {
    let mut lu = factor(&absorption_basis(), LuParams::default());
    lu.update(0, &entering_column(TRUE_DIAG)).expect("update");

    let bp = replaced_cols(TRUE_DIAG);
    let a = SparseColMatrix::from_dense_columns(M, &bp).expect("matrix");
    let bnorm = 2.0; // ‖B'‖max
    let xt = vec![1.0, 2.0, 3.0, 4.0];

    let mut b = vec![0.0; M];
    a.matvec(&xt, &mut b);
    let mut x = b.clone();
    lu.ftran(&mut x).expect("updated ftran");
    let res = residual(&bp, &x, &b);
    let bound = f64::EPSILON * (M as f64) * bnorm * inf_norm(&x);
    assert!(
        res <= bound,
        "ftran must be backward stable: residual {res:.3e} > bound {bound:.3e}"
    );

    let mut bt = vec![0.0; M];
    a.matvec_transpose(&xt, &mut bt);
    let mut y = bt.clone();
    lu.btran(&mut y).expect("updated btran");
    let rest = residual_t(&bp, &y, &bt);
    let bound_t = f64::EPSILON * (M as f64) * bnorm * inf_norm(&y);
    assert!(
        rest <= bound_t,
        "btran must be backward stable: residual {rest:.3e} > bound {bound_t:.3e}"
    );
}

/// With `head_extra = 0` the cancellation is mathematically exact — the
/// replacement basis is genuinely singular — so the compensated sweep (whose
/// final diagonal is now a true zero, not an artifact) must still reject the
/// update, and the failed update must leave the factorization untouched —
/// including under the pivot search, whose interchange path rewrites more
/// rows before discovering the failure (the swapped rows must roll back).
#[test]
fn genuinely_singular_replacement_still_rejected_and_rolled_back() {
    for pivot_search in [false, true] {
        let params = LuParams {
            update_pivot_search: pivot_search,
            ..LuParams::default()
        };
        let mut lu = factor(&absorption_basis(), params);
        let err = lu.update(0, &entering_column(0.0));
        assert!(
            matches!(err, Err(FeralError::NeedsRefactor)),
            "a truly singular replacement must be rejected (pivot_search={pivot_search}): {err:?}"
        );
        assert_eq!(lu.updates_since_refactor(), 0, "nothing committed");

        // The original basis must still solve exactly as before the failure.
        let cols = absorption_basis();
        let a = SparseColMatrix::from_dense_columns(M, &cols).expect("matrix");
        let xt = vec![1.0, -1.0, 2.0, 0.5];
        let mut b = vec![0.0; M];
        a.matvec(&xt, &mut b);
        let mut x = b.clone();
        lu.ftran(&mut x)
            .expect("ftran on the rolled-back factorization");
        let res = residual(&cols, &x, &b);
        assert!(
            res <= 1e-12 * inf_norm(&b).max(1.0),
            "rolled-back factors must solve the original basis (residual {res:.3e})"
        );
    }

    // Engine stays usable after the rolled-back failure: the nonsingular
    // variant commits — on the default path. (Under the pivot search the same
    // replacement is *correctly* refused: its interchange order's true final
    // pivot is λ·2⁻³⁵ with λ ~ 2⁻²⁰ from the dominated-diagonal swaps —
    // genuinely below ztol. Pivot re-ordering cannot express this
    // factorization; only the compensated fixed order can. See the
    // proportionality analysis in dev/research/issue-112-bg-update.md.)
    let mut lu = factor(&absorption_basis(), LuParams::default());
    let _ = lu.update(0, &entering_column(0.0));
    lu.update(0, &entering_column(TRUE_DIAG))
        .expect("engine must remain usable after the rolled-back failure");
    assert_eq!(lu.updates_since_refactor(), 1);
}

/// The opt-in Bartels–Golub pivot search: on a basis where the working row
/// strictly dominates a retained diagonal, the update must perform
/// interchanges (`FtOp::Swap` etas), commit valid factors, and keep solving —
/// including a second update whose spike solve replays the Swap etas, and
/// btran (the transposed replay). Oracle: the equation-residual identity on
/// the exactly-known replacement bases.
#[test]
fn pivot_search_interchanges_produce_valid_factors_and_chain() {
    let params = LuParams {
        update_pivot_search: true,
        ..LuParams::default()
    };
    let mut lu = factor(&absorption_basis(), params);

    // Entering column whose sweep meets dominated retained diagonals
    // (T = 2^-20 vs working-row entries of O(1)) — the pivot search must swap.
    let v1 = vec![1.0, 1.0, 0.0, 0.25];
    lu.update(0, &v1).expect("pivot-searching update commits");
    assert!(
        lu.pivot_search_swaps() >= 1,
        "dominated retained diagonals must trigger interchanges, got {}",
        lu.pivot_search_swaps()
    );

    let mut bp = absorption_basis();
    bp[0] = v1.clone();
    let a1 = SparseColMatrix::from_dense_columns(M, &bp).expect("matrix");
    let xt = vec![0.5, 1.5, -2.0, 3.0];
    let mut b = vec![0.0; M];
    a1.matvec(&xt, &mut b);
    let mut x = b.clone();
    lu.ftran(&mut x).expect("ftran after swap update");
    let res = residual(&bp, &x, &b);
    assert!(
        res <= 1e-9 * inf_norm(&b).max(1.0),
        "ftran residual {res:.3e}"
    );

    // Second update: its spike computation must replay the Swap etas.
    let v2 = vec![0.5, 0.0, 1.0, 2.0];
    lu.update(3, &v2).expect("post-swap update commits");
    bp[3] = v2;
    let a2 = SparseColMatrix::from_dense_columns(M, &bp).expect("matrix");

    let mut b2 = vec![0.0; M];
    a2.matvec(&xt, &mut b2);
    let mut x2 = b2.clone();
    lu.ftran(&mut x2).expect("ftran after chained updates");
    let res2 = residual(&bp, &x2, &b2);
    assert!(
        res2 <= 1e-9 * inf_norm(&b2).max(1.0),
        "chained ftran residual {res2:.3e}"
    );

    let mut bt = vec![0.0; M];
    a2.matvec_transpose(&xt, &mut bt);
    let mut y = bt.clone();
    lu.btran(&mut y).expect("btran after chained updates");
    let rest = residual_t(&bp, &y, &bt);
    assert!(
        rest <= 1e-9 * inf_norm(&bt).max(1.0),
        "chained btran residual {rest:.3e}"
    );
}

/// With the pivot search off (the default), the same dominated-diagonal
/// update must still commit via the plain FT order (its multipliers are large
/// but the growth monitor arbitrates), and the swap counter must stay zero —
/// the default path never deviates from the legacy pivot order.
#[test]
fn default_path_never_swaps() {
    let mut lu = factor(&absorption_basis(), LuParams::default());
    lu.update(0, &[1.0, 1.0, 0.0, 0.25])
        .expect("plain FT update commits");
    assert_eq!(lu.pivot_search_swaps(), 0, "default path must not swap");
}
