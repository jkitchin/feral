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
        // Phase 2.2.2: MC64 scales A to D*A*D with unit row/col norms,
        // which shrinks the worst pivots below the absolute zero_tol
        // floor. The column-relative MUMPS/SSIDS default u=0.01
        // rejects pivots that are >100x smaller than their column max
        // and flushes them through ForceAccept, which is the designed
        // interaction with the equilibrated matrix.
        pivot_threshold: 0.01,
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
///
/// Phase 2.2.1 Step 8 status: STILL FAILING. Post-MC64 residual
/// grew to 2.27e+46 under the `solve_sparse_refined` path because
/// ForceAccept sees 5 zero pivots on the scaled matrix (feral
/// inertia (62,142,5) vs MUMPS (71,137,1), SSIDS (71,138,0)). The
/// rank-deficient solve interacts pathologically with MC64 exp
/// dual clamps. Diagnosis deferred to Phase 2.2.2+. See
/// dev/validation/phase-2.2.1-mc64-sweep.md for details.
///
/// Phase 2.2.2 Step 7 status: RECOVERED 47 ORDERS, STILL FAILING
/// TARGET. Enabling `pivot_threshold = 0.01` (MUMPS CNTL(1) /
/// SSIDS options%u) in `ldlt_params()` drops the residual from
/// 2.27e+46 to 1.076e-1.
///
/// Phase 2.2.3 status: SECONDARY REGRESSION, STILL FAILING. The
/// supernode amalgamation adjacency fix that cleared the
/// CHWIRUT1/CRESC100/CRESC132 plateau produces 117 fine-grained
/// supernodes for ACOPP30 where the pre-fix amalgamation bug
/// had accidentally fused 16 of them. With 1-2 columns per
/// supernode the BK kernel has no room to pivot and 14-31
/// pivots get ForceAccept'd as zero (inertia ~(58,137,14) to
/// (56,122,31) vs MUMPS (71,137,1)). Residual is now 1.659e+5.
/// Full recovery of this matrix requires either (a) SSIDS-style
/// column renumbering to build coarser supernodes (logged as
/// follow-up in dev/research/phase-2.2.3-plateau.md) or (b)
/// delayed pivoting (Phase 2.3). The plateau work on
/// CHWIRUT1/CRESC100/CRESC132 is a strict win and is what moved
/// the three companion tests in this file to PASS under
/// `--ignored`.
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
///
/// Phase 2.2.1 Step 8 status: IMPROVED but STILL FAILING. Residual
/// dropped 2.39e+08 → 1.37e+05.
///
/// Phase 2.2.2 Step 7 status: UNCHANGED at 1.370e+05.
///
/// Phase 2.2.3 status: PASSING. The supernode amalgamation
/// adjacency fix (commit 91e808b) drops the residual to
/// 4.43e-15 — **4 orders better than canonical MUMPS** —
/// and resolves the previously-observed ±2 inertia mismatch.
/// The inertia now matches MUMPS exactly: (2660, 2654, 0).
/// The "inertia mismatch" previously attributed to a
/// count_2x2_inertia trace-rule issue was in fact a symptom of
/// the same supernode amalgamation bug.
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

/// CHWIRUT1_0000: n=645. Pre-fix: 1.41e+09. Canonical MUMPS:
/// 9.51e-13. Target: < 1e-8.
///
/// Phase 2.2.1 Step 8: 1.41e+09 → 8.50e+02.
/// Phase 2.2.2 Step 7: UNCHANGED at 8.497e+02.
/// Phase 2.2.3 status: PASSING. The supernode adjacency fix
/// drops the residual to 8.69e-14 — **beats canonical MUMPS
/// by half an order** — with iterative refinement converging in
/// 2 steps. Inertia matches MUMPS (431, 214, 0) exactly.
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

/// CRESC100_0000: n=806. Pre-fix: 2.54e+04. Canonical MUMPS:
/// 6.15e-15. Target: < 1e-8.
///
/// Phase 2.2.1 Step 8: 2.54e+04 → 1.43e+02.
/// Phase 2.2.2 Step 7: UNCHANGED at 1.426e+02.
/// Phase 2.2.3 status: PASSING. The supernode adjacency fix
/// drops the residual to 1.75e-16 — **beats canonical MUMPS
/// by 2 orders of magnitude**. Inertia matches MUMPS
/// (606, 200, 0) exactly.
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
