//! Phase 2.2.3 follow-up — parity panel.
//!
//! For each curated matrix in `tests/data/parity/`, assert feral's
//! multi-frontal solve matches the MUMPS oracle exactly on inertia
//! and within K*MUMPS on relative residual. Regenerate this file by
//! running:
//!     cargo run --release --example select_parity_panel
//!
//! Do NOT edit tests/parity.rs by hand. The file is generated.

use std::path::Path;

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};

const K_RESIDUAL: f64 = 10.0;
const ABS_FLOOR: f64 = 1e-14;

fn ldlt_params() -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
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
        rs += (ax[i] - b[i]).powi(2);
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

fn read_oracle(path: &Path) -> (Inertia, f64) {
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read oracle"))
            .expect("parse oracle");
    let pos = data["inertia"]["positive"].as_u64().unwrap() as usize;
    let neg = data["inertia"]["negative"].as_u64().unwrap() as usize;
    let zero = data["inertia"]["zero"].as_u64().unwrap() as usize;
    let res = data["residual_2norm_relative"].as_f64().unwrap();
    (Inertia::new(pos, neg, zero), res)
}

fn run_parity(fam: &str, stem: &str) {
    let base = format!("tests/data/parity/{}/{}", fam, stem);
    let mtx = read_mtx(Path::new(&format!("{}.mtx", base))).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let sidecar = read_sidecar(Path::new(&format!("{}.json", base))).expect("sidecar");
    let rhs = sidecar.finite_rhs().expect("finite rhs");
    let (mumps_inertia, mumps_res) = read_oracle(Path::new(&format!("{}.mumps.json", base)));

    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (fac, inertia) = factorize_multifrontal(&csc, &sym, &ldlt_params()).expect("factor");
    let x = solve_sparse_refined(&csc, &fac, &rhs).expect("solve");
    let feral_res = rel_residual(&csc, &x, &rhs);

    assert_eq!(
        inertia, mumps_inertia,
        "{} inertia: feral={} mumps={}",
        stem, inertia, mumps_inertia
    );
    // Gate: feral residual must be within K*MUMPS residual, OR at or
    // below the absolute floor. The floor catches matrices where MUMPS
    // produces sub-machine-precision residuals (e.g. 1e-30) that feral
    // cannot and should not be expected to match.
    let target = (K_RESIDUAL * mumps_res).max(ABS_FLOOR);
    assert!(
        feral_res <= target,
        "{} residual: feral={:.3e} > max(K*mumps={:.3e}, floor={:.3e}) = {:.3e}",
        stem,
        feral_res,
        K_RESIDUAL * mumps_res,
        ABS_FLOOR,
        target
    );
}

// Panel snapshot: 27/28 matrices pass MUMPS parity (14 panel-time +
// 8 un-ignored after the Phase 2.3 sign-preservation fix + 5
// un-ignored after the Phase 2.3 refinement-termination fix).
// Only SSI_2597 remains ignored (factorization-level limitation,
// not a refinement issue — see Phase 2.4). Passing matrices run
// as regular tests and protect against regression. As fixes land,
// rerun `cargo run --release --example select_parity_panel` to
// refresh the panel and un-ignore the now-passing matrices.

#[test]
fn parity_acopp30_0000() {
    run_parity("acopp30", "ACOPP30_0000");
}

// Panel time: residual ratio 1.52e1 > K=10 (feral=3.86e-14, mumps=2.54e-15)
// Phase 2.3: closed by refinement-termination fix (residual-based,
// max 10 steps).
#[test]
fn parity_avion2_0510() {
    run_parity("avion2", "AVION2_0510");
}

#[test]
fn parity_bqpgasim_0012() {
    run_parity("bqpgasim", "BQPGASIM_0012");
}

// Panel time: inertia mismatch (feral=(129, 60, 1) mumps=(129, 61, 0))
// Phase 2.3 sign-preservation fix: passes.
#[test]
fn parity_ceri651a_0000() {
    run_parity("ceri651a", "CERI651A_0000");
}

// Panel time: inertia mismatch (feral=(128, 61, 1) mumps=(129, 61, 0))
// Phase 2.3 sign-preservation fix: passes.
#[test]
fn parity_ceri651a_0165() {
    run_parity("ceri651a", "CERI651A_0165");
}

// Panel time: inertia mismatch (feral=(128, 61, 1) mumps=(129, 61, 0))
// Phase 2.3 sign-preservation fix: passes.
#[test]
fn parity_ceri651a_0166() {
    run_parity("ceri651a", "CERI651A_0166");
}

// Panel time: residual ratio 5.10e2 > K=10 (feral=4.29e-13, mumps=8.40e-16)
// Phase 2.3: closed by refinement-termination fix. The plain 3-step
// refinement was exiting before the trajectory hit the machine-
// precision basin at step ~4.
#[test]
fn parity_ceri651c_0746() {
    run_parity("ceri651c", "CERI651C_0746");
}

// Panel time: residual ratio 5.12e2 > K=10 (feral=1.28e-12, mumps=2.50e-15)
// Phase 2.3: closed by refinement-termination fix (same cause as
// CERI651C_0746 — trajectory settles after ~5 steps).
#[test]
fn parity_ceri651els_1482() {
    run_parity("ceri651els", "CERI651ELS_1482");
}

#[test]
fn parity_chwirut1_0000() {
    run_parity("chwirut1", "CHWIRUT1_0000");
}

#[test]
fn parity_cresc100_0000() {
    run_parity("cresc100", "CRESC100_0000");
}

#[test]
fn parity_cresc132_0000() {
    run_parity("cresc132", "CRESC132_0000");
}

// Panel time: inertia mismatch (feral=(20, 14, 1) mumps=(20, 15, 0))
// Phase 2.3 sign-preservation fix: passes.
#[test]
fn parity_degenlpa_0065() {
    run_parity("degenlpa", "DEGENLPA_0065");
}

// Panel time: inertia mismatch (feral=(20, 14, 1) mumps=(20, 15, 0))
// Phase 2.3 sign-preservation fix: passes.
#[test]
fn parity_degenlpb_0045() {
    run_parity("degenlpb", "DEGENLPB_0045");
}

// Panel time: inertia mismatch (feral=(20, 14, 1) mumps=(20, 15, 0))
// Phase 2.3 sign-preservation fix: passes.
#[test]
fn parity_degenlpb_0046() {
    run_parity("degenlpb", "DEGENLPB_0046");
}

// Panel time: inertia mismatch (feral=(20, 14, 1) mumps=(20, 15, 0))
// Phase 2.3 sign-preservation fix: passes.
#[test]
fn parity_degenlpb_0047() {
    run_parity("degenlpb", "DEGENLPB_0047");
}

// Panel time: residual ratio 1.08e1 > K=10 (feral=3.23e-13, mumps=2.99e-14)
// Phase 2.3: closed by refinement-termination fix.
#[test]
fn parity_hahn1_0004() {
    run_parity("hahn1", "HAHN1_0004");
}

#[test]
fn parity_hahn1_0006() {
    run_parity("hahn1", "HAHN1_0006");
}

#[test]
fn parity_hahn1_0023() {
    run_parity("hahn1", "HAHN1_0023");
}

#[test]
fn parity_hatfldbne_2138() {
    run_parity("hatfldbne", "HATFLDBNE_2138");
}

#[test]
fn parity_hatfldbne_2140() {
    run_parity("hatfldbne", "HATFLDBNE_2140");
}

#[test]
fn parity_hs85_0176() {
    run_parity("hs85", "HS85_0176");
}

#[test]
fn parity_hydcar20_0000() {
    run_parity("hydcar20", "HYDCAR20_0000");
}

// Panel time: residual ratio 1.25e1 > K=10 (feral=2.50e-14, mumps=1.99e-15)
// Phase 2.3: closed by refinement-termination fix.
#[test]
fn parity_meyer3ne_0253() {
    run_parity("meyer3ne", "MEYER3NE_0253");
}

// Panel time: inertia mismatch (feral=(52, 22, 1) mumps=(52, 23, 0))
// Phase 2.3 sign-preservation fix: passes.
#[test]
fn parity_palmer2ane_0000() {
    run_parity("palmer2ane", "PALMER2ANE_0000");
}

#[test]
fn parity_roszman1_0225() {
    run_parity("roszman1", "ROSZMAN1_0225");
}

// Panel time: residual ratio 1.56e3 > K=10 (feral=1.80e-13, mumps=1.15e-16)
#[test]
#[ignore]
fn parity_ssi_2597() {
    run_parity("ssi", "SSI_2597");
}

#[test]
fn parity_swopf_0000() {
    run_parity("swopf", "SWOPF_0000");
}

#[test]
fn parity_vesuvio_0021() {
    run_parity("vesuvio", "VESUVIO_0021");
}
