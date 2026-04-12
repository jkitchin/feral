//! Phase 2.2.1 regression tests: scaling must close the residual
//! gap that the Phase 2.1.2 sanity check exposed on `n > 500`
//! matrices.
//!
//! These tests are gated `#[ignore]` because they require
//! `data/matrices/kkt/<problem>/<id>.mtx` files that are not
//! committed to git. Run with:
//!
//!     cargo test --test mc64_regression -- --ignored
//!
//! Each test loads a specific matrix, runs the sparse path
//! (symbolic + factor + refined solve), and asserts on the
//! residual against a target derived from the canonical MUMPS
//! and SSIDS oracles.
//!
//! **Pre-fix baselines** (from commit `c01235f`, before MC64):
//!   ACOPP30_0000:     feral 3.15e-2   MUMPS 5.0e-14   SSIDS 3.9e-14
//!   CRESC132_0000:    feral 2.39e+08  MUMPS 2.48e-11  SSIDS 8.90e-15
//!   CHWIRUT1_0000:    feral 1.41e+09  MUMPS ~1e-13    SSIDS ~3e-13
//!
//! These tests MUST fail before MC64 lands and MUST pass after.
//! If a test is still failing after the MC64 implementation is
//! complete, investigate before declaring Phase 2.2.1 closed.

use std::path::Path;

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn ldlt_params() -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    }
}

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        let r = ax[i] - b[i];
        rs += r * r;
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

/// Load a matrix + sidecar, run feral's sparse path end-to-end,
/// and return the relative residual against the sidecar RHS.
fn feral_residual(stem: &str) -> Option<f64> {
    let mtx_path_buf = format!("data/matrices/kkt/{}.mtx", stem);
    let json_path_buf = format!("data/matrices/kkt/{}.json", stem);
    let mtx_path = Path::new(&mtx_path_buf);
    let json_path = Path::new(&json_path_buf);
    if !mtx_path.exists() {
        eprintln!("SKIP: {} not found", mtx_path.display());
        return None;
    }

    let mtx = read_mtx(mtx_path).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let sc = read_sidecar(json_path).expect("read sidecar");
    let rhs = sc.finite_rhs().expect("finite rhs");

    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (factors, _inertia) = factorize_multifrontal(&csc, &sym, &ldlt_params()).expect("factor");
    let x = solve_sparse_refined(&csc, &factors, &rhs).expect("solve");

    Some(rel_residual(&csc, &x, &rhs))
}

/// ACOPP30_0000: n=209, the matrix identified in the Phase 1
/// retrospective as the archetype of the "ACOPP30 residual gap".
/// Pre-fix: 3.15e-2. Canonical MUMPS: 5.0e-14. Target: < 1e-8
/// (six orders of magnitude improvement from the baseline, four
/// orders from the canonical).
#[test]
#[ignore]
fn acopp30_0000_residual_under_1e_8_after_mc64() {
    let Some(res) = feral_residual("ACOPP30/ACOPP30_0000") else {
        return;
    };
    assert!(
        res < 1e-8,
        "ACOPP30_0000 residual = {:.3e}, target < 1e-8. \
         Pre-fix baseline: 3.15e-2. Canonical MUMPS: 5.0e-14.",
        res
    );
}

/// CRESC132_0000: n=5314, the largest matrix in the existing
/// corpus. Pre-fix: 2.39e+08. Canonical MUMPS: 2.48e-11. Target:
/// < 1e-6 (14 orders of magnitude improvement from the baseline).
#[test]
#[ignore]
fn cresc132_0000_residual_under_1e_6_after_mc64() {
    let Some(res) = feral_residual("CRESC132/CRESC132_0000") else {
        return;
    };
    assert!(
        res < 1e-6,
        "CRESC132_0000 residual = {:.3e}, target < 1e-6. \
         Pre-fix baseline: 2.39e+08. Canonical MUMPS: 2.48e-11.",
        res
    );
}

/// CHWIRUT1_0000: n=645, one of the smaller matrices in the
/// sanity check panel. Had correct inertia but bad residual
/// (1.41e+09) pre-fix. This is a case where MC64 scaling should
/// cleanly close the gap because the inertia bug is not masking
/// the scaling bug.
#[test]
#[ignore]
fn chwirut1_0000_residual_under_1e_8_after_mc64() {
    let Some(res) = feral_residual("CHWIRUT1/CHWIRUT1_0000") else {
        return;
    };
    assert!(
        res < 1e-8,
        "CHWIRUT1_0000 residual = {:.3e}, target < 1e-8. \
         Pre-fix baseline: 1.41e+09.",
        res
    );
}

/// CRESC100_0000: n=806, another matrix with correct inertia but
/// bad residual (2.54e+04) pre-fix. Smallest residual improvement
/// needed in the sanity panel.
#[test]
#[ignore]
fn cresc100_0000_residual_under_1e_8_after_mc64() {
    let Some(res) = feral_residual("CRESC100/CRESC100_0000") else {
        return;
    };
    assert!(
        res < 1e-8,
        "CRESC100_0000 residual = {:.3e}, target < 1e-8. \
         Pre-fix baseline: 2.54e+04.",
        res
    );
}
