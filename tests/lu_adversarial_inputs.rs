//! Shared adversarial-input tests for the LU basis engine — the class the
//! 2026-07-10 audit flagged: **inputs the factor path validates or handles,
//! but the `update()` / solve paths do not guard.** Each such input is a
//! silent-wrong-answer or spurious-failure hazard that the existing update
//! tests (dense/tridiagonal bases replacing the last slot) never exercise.
//!
//! Three families, one per open issue:
//! - #114 non-finite entering columns (factor rejects; update commits NaN).
//! - #115 dense update's zero-superdiagonal landmine (structural zeros the
//!   sparse redesign dodges but the dense column-shift does not).
//! - #118 update ztol anchored to `u_max0` not `max|A|` (healthy pivots on
//!   high-growth bases spuriously rejected → refactor livelock).
//!
//! ## Convention for the pending fixes
//!
//! Tests that assert the **correct** post-fix behavior are `#[ignore]`d with
//! their issue number, so `cargo test` stays green while the bug is open. They
//! are executable acceptance criteria: run `cargo test --test
//! lu_adversarial_inputs -- --ignored` to see them fail against `main`, and
//! delete the `#[ignore]` line when the corresponding fix lands. Tests **not**
//! ignored assert facts that are correct today (the factor-path rejection, the
//! nonsingularity of the replacement basis via an independent fresh factor,
//! the sparse path already handling what the dense path cannot, and that
//! genuinely singular replacements stay rejected) — they establish the
//! adversarial setup is sound and guard against a fix that over-corrects.
//!
//! Oracles are external and oracle-free: NaN-must-be-rejected is a contract,
//! nonsingularity is verified by an independent from-scratch factorization,
//! and accuracy is the equation-residual identity `‖B'x − b‖`.
//!
//! Not covered here (they are frontal-kernel-internal, not black-box
//! update/solve inputs): the rook factor/solve contract (#116) and the
//! blocked/scalar rook inertia divergence (#117) — those need dense-LDLᵀ
//! frontal-level tests, tracked in their own issues.

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{DenseLu, LuParams, RefactorCause, SparseLu, SparseLuSymbolic};
use feral::FeralError;

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

/// `‖B x − b‖∞` for a basis given as dense columns (column-major `cols[j][i]`).
fn residual(cols: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
    let m = b.len();
    let mut r = b.to_vec();
    for (j, col) in cols.iter().enumerate() {
        for i in 0..m {
            r[i] -= col[i] * x[j];
        }
    }
    inf_norm(&r)
}

// ===========================================================================
// #114 — non-finite entering columns
// ===========================================================================

/// The factor path rejects a non-finite basis column loudly (sparse and
/// dense). This is the asymmetry the update path violates; it is correct today
/// and must stay correct.
#[test]
fn factor_rejects_nonfinite_columns() {
    let bad = vec![vec![f64::NAN, 0.0], vec![0.0, 1.0]];
    assert!(
        matches!(
            SparseLu::factor_dense_columns(2, &bad, LuParams::default()),
            Err(FeralError::InvalidInput(_))
        ),
        "sparse factor must reject a NaN column"
    );
    assert!(
        matches!(
            DenseLu::factor(&bad, 2, LuParams::default()),
            Err(FeralError::InvalidInput(_))
        ),
        "dense factor must reject a NaN column"
    );
    let inf = vec![vec![f64::INFINITY, 0.0], vec![0.0, 1.0]];
    assert!(SparseLu::factor_dense_columns(2, &inf, LuParams::default()).is_err());
    assert!(DenseLu::factor(&inf, 2, LuParams::default()).is_err());
}

/// #114 (sparse): `update` with a NaN entering column must be rejected as
/// invalid input, leaving the factorization unchanged — matching the factor
/// path. Currently the NaN is committed into `U` and a later `ftran` returns
/// `Ok` with a NaN solution (confirmed: `x = [NaN, 1.0]`), a silent wrong
/// answer.
#[test]
fn sparse_update_rejects_nonfinite_column() {
    let cols = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut lu = SparseLu::factor_dense_columns(2, &cols, LuParams::default()).unwrap();

    for bad in [
        vec![f64::NAN, 1.0],
        vec![1.0, f64::NAN],
        vec![f64::INFINITY, 1.0],
    ] {
        let err = lu.update(1, &bad);
        assert!(
            matches!(err, Err(FeralError::InvalidInput(_))),
            "update must reject non-finite column {bad:?}, got {err:?}"
        );
        assert_eq!(
            lu.updates_since_refactor(),
            0,
            "a rejected update must leave the factorization unchanged"
        );
    }

    // Unchanged-on-failure: the original identity basis still solves exactly.
    let mut x = vec![3.0, 5.0];
    lu.ftran(&mut x).unwrap();
    assert!((x[0] - 3.0).abs() < 1e-12 && (x[1] - 5.0).abs() < 1e-12);
}

/// #114 (dense): same contract on the dense path. Currently
/// `update(1, [NaN, 1.0])` on a 2×2 identity returns `Ok(())`, committing a
/// NaN into `U`.
#[test]
fn dense_update_rejects_nonfinite_column() {
    let cols = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut lu = DenseLu::factor(&cols, 2, LuParams::default()).unwrap();
    for bad in [vec![f64::NAN, 1.0], vec![f64::INFINITY, 1.0]] {
        let err = lu.update(1, &bad);
        assert!(
            matches!(err, Err(FeralError::InvalidInput(_))),
            "dense update must reject non-finite column {bad:?}, got {err:?}"
        );
        assert_eq!(lu.updates_since_refactor(), 0);
    }
}

// ===========================================================================
// #115 — dense update zero-superdiagonal landmine
// ===========================================================================

/// The sparse update already commits a slack-basis self-replacement (the
/// Forrest–Tomlin symmetric permutation dodges the zero-superdiagonal
/// landmine). Correct today; the contrast that shows the dense failure is a
/// path defect, not an ill-posed input.
#[test]
fn sparse_update_handles_slack_basis_self_replacement() {
    let cols = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut lu = SparseLu::factor_dense_columns(2, &cols, LuParams::default()).unwrap();
    lu.update(0, &[1.0, 0.0])
        .expect("sparse update must commit an identity self-replacement");
    assert_eq!(lu.updates_since_refactor(), 1);
    // Still solves the (unchanged) basis exactly.
    let mut x = vec![2.0, 7.0];
    lu.ftran(&mut x).unwrap();
    assert!((x[0] - 2.0).abs() < 1e-12 && (x[1] - 7.0).abs() < 1e-12);
}

/// #115: the dense update must commit a trivially valid replacement on a
/// slack/triangular basis. Currently the column-shift scheme pivots on the old
/// (zero) superdiagonal and returns `NeedsRefactor(TinyPivot, 0.0)` even for an
/// identity column replaced with itself.
#[test]
fn dense_update_commits_on_slack_basis() {
    // Identity self-replacement at every slot must be a no-op commit.
    for slot in 0..3 {
        let cols = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let mut lu = DenseLu::factor(&cols, 3, LuParams::default()).unwrap();
        let mut col = vec![0.0; 3];
        col[slot] = 1.0;
        lu.update(slot, &col)
            .unwrap_or_else(|e| panic!("dense self-replace slot {slot} must commit: {e:?}"));
    }

    // A genuine slack-basis pivot: replace column 0 of I₃ with e₀ + e₁ (a valid
    // nonsingular replacement). Must commit and solve.
    let cols = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let mut lu = DenseLu::factor(&cols, 3, LuParams::default()).unwrap();
    let new_col = vec![1.0, 1.0, 0.0];
    lu.update(0, &new_col)
        .expect("valid slack replacement must commit");
    let mut bp = cols.clone();
    bp[0] = new_col;
    let xt = [1.0, 2.0, 3.0];
    let mut b = vec![0.0; 3];
    for i in 0..3 {
        for (j, col) in bp.iter().enumerate() {
            b[i] += col[i] * xt[j];
        }
    }
    let mut x = b.clone();
    lu.ftran(&mut x).unwrap();
    assert!(residual(&bp, &x, &b) <= 1e-12 * inf_norm(&b).max(1.0));
}

// ===========================================================================
// #118 — update ztol anchored to u_max0 (growth livelock)
// ===========================================================================

/// Wilkinson growth matrix: unit diagonal, −1 strictly below, last column all
/// +1. Well-conditioned with `max|A| = 1`, but LU growth makes
/// `u_max0 = 2^(m-1)`, so the update's `ztol = zero_pivot_tol · u_max0` climbs
/// to ≈ 14 at m = 50 — dwarfing the healthy O(1) bump pivot (magnitude 0.25).
/// m = 50 is deliberate: growth `≈ 1.4e14 < 2^53`, so the ±1/integer solve
/// arithmetic stays exact (residual 0), keeping the accuracy oracle tight;
/// beyond m ≈ 53 the growth exceeds the f64 mantissa and the residual blows up
/// for genuine (backward-error) reasons, which would confound this test.
fn wilkinson(m: usize) -> Vec<Vec<f64>> {
    let mut cols = vec![vec![0.0; m]; m];
    for (j, col) in cols.iter_mut().enumerate() {
        col[j] = 1.0;
        for row in col.iter_mut().skip(j + 1) {
            *row = -1.0;
        }
    }
    for row in cols[m - 1].iter_mut() {
        *row = 1.0;
    }
    cols
}

/// The replacement basis is genuinely nonsingular: an independent from-scratch
/// factorization of `B'` succeeds. Correct today; establishes that the update's
/// rejection (below) is spurious, not a real singularity.
#[test]
fn wilkinson_replacement_basis_is_nonsingular() {
    let m = 50;
    let cols = wilkinson(m);
    let mut bp = cols.clone();
    bp[0] = {
        let mut c = vec![0.0; m];
        c[0] = 1.0;
        c[1] = 1.0;
        c
    };
    let a = SparseColMatrix::from_dense_columns(m, &bp).unwrap();
    let sym = SparseLuSymbolic::analyze(&a).unwrap();
    let mut fresh = SparseLu::factor(&a, &sym, LuParams::default())
        .expect("B' is nonsingular: fresh factor must succeed");
    // And it solves to small residual.
    let xt: Vec<f64> = (0..m).map(|i| 1.0 + (i % 4) as f64).collect();
    let mut b = vec![0.0; m];
    a.matvec(&xt, &mut b);
    let mut x = b.clone();
    fresh.ftran(&mut x).unwrap();
    assert!(residual(&bp, &x, &b) <= 1e-9 * inf_norm(&b).max(1.0));
}

/// #118 (sparse): replacing slot 0 of the Wilkinson basis with `e₀ + e₁` is a
/// valid update whose true bump pivot is O(1) (magnitude 0.25), but the update
/// rejects it as `TinyPivot` because `ztol ≈ 5.8e4`. Worse, `refactor()`
/// reproduces the identical high-growth factor, so update→refactor→retry
/// **livelocks**. Correct behavior: the update commits and solves.
#[test]
fn sparse_update_commits_on_high_growth_basis() {
    let m = 50;
    let cols = wilkinson(m);
    let a = SparseColMatrix::from_dense_columns(m, &cols).unwrap();
    let sym = SparseLuSymbolic::analyze(&a).unwrap();
    let mut lu = SparseLu::factor(&a, &sym, LuParams::default()).unwrap();

    let mut col = vec![0.0; m];
    col[0] = 1.0;
    col[1] = 1.0;
    lu.update(0, &col)
        .expect("valid O(1)-pivot update on a well-conditioned basis must commit");

    // Residual against B' with a non-trivial RHS.
    let mut bp = cols.clone();
    bp[0] = col;
    let ba = SparseColMatrix::from_dense_columns(m, &bp).unwrap();
    let xt: Vec<f64> = (0..m).map(|i| 1.0 + (i % 4) as f64).collect();
    let mut b = vec![0.0; m];
    ba.matvec(&xt, &mut b);
    let mut x = b.clone();
    lu.ftran(&mut x).unwrap();
    assert!(
        residual(&bp, &x, &b) <= 1e-8 * inf_norm(&b).max(1.0),
        "committed update must solve B' accurately"
    );
}

/// #118 (dense): the same on the dense path. (Also exercises the #115 landmine,
/// since the Wilkinson basis has zero superdiagonals — both fixes are needed
/// for this to pass.)
#[test]
fn dense_update_commits_on_high_growth_basis() {
    let m = 50;
    let cols = wilkinson(m);
    let mut lu = DenseLu::factor(&cols, m, LuParams::default()).unwrap();
    let mut col = vec![0.0; m];
    col[0] = 1.0;
    col[1] = 1.0;
    lu.update(0, &col)
        .expect("dense update on a well-conditioned basis must commit");
}

/// #118: the livelock is the real cost — a rejected update followed by
/// `refactor()` and retry never makes progress, because the refactor
/// reproduces the identical high-growth factor. Correct behavior: the first
/// update commits (loop exits at attempt 0).
#[test]
fn high_growth_update_does_not_livelock() {
    let m = 50;
    let cols = wilkinson(m);
    let a = SparseColMatrix::from_dense_columns(m, &cols).unwrap();
    let sym = SparseLuSymbolic::analyze(&a).unwrap();
    let mut lu = SparseLu::factor(&a, &sym, LuParams::default()).unwrap();
    let mut col = vec![0.0; m];
    col[0] = 1.0;
    col[1] = 1.0;

    let mut committed = false;
    for _ in 0..4 {
        if lu.update(0, &col).is_ok() {
            committed = true;
            break;
        }
        lu.refactor(&a, &sym).unwrap();
    }
    assert!(
        committed,
        "update→refactor→retry must terminate productively, not livelock"
    );
}

/// Guard against over-correcting #114/#115/#118: a genuinely singular
/// replacement (a linearly dependent column — replacing slot 1 of I₃ with a
/// duplicate of column 0) must still be rejected, on both paths. Correct today;
/// must stay correct after the fixes loosen the update's rejection criteria.
#[test]
fn singular_replacement_still_rejected() {
    let cols = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    // Replace column 1 with a copy of column 0 (e₀): columns 0 and 1 identical
    // ⇒ singular basis.
    let dependent = vec![1.0, 0.0, 0.0];

    let mut sp = SparseLu::factor_dense_columns(3, &cols, LuParams::default()).unwrap();
    assert!(
        matches!(sp.update(1, &dependent), Err(FeralError::NeedsRefactor)),
        "sparse: dependent replacement must be rejected"
    );

    let mut dn = DenseLu::factor(&cols, 3, LuParams::default()).unwrap();
    assert!(
        matches!(dn.update(1, &dependent), Err(FeralError::NeedsRefactor)),
        "dense: dependent replacement must be rejected"
    );
    // The recorded cause is a singularity/tiny-pivot signal, not a budget trip.
    assert!(matches!(
        dn.last_refactor(),
        Some((RefactorCause::TinyPivot | RefactorCause::Singular, _))
    ));
}
