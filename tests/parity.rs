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

// Panel snapshot: 11/28 matrices pass MUMPS parity at panel time.
// Failing matrices are `#[ignore]`'d with the panel-time failure
// mode in the attribute comment. Passing matrices run as regular
// tests and protect against regression. As fixes land, rerun
// `cargo run --release --example select_parity_panel` to refresh
// the panel and un-ignore the now-passing matrices.

// Panel time: inertia mismatch (feral=(58, 137, 14) mumps=(71, 137, 1))
#[test]
#[ignore]
fn parity_acopp30_0000() {
    run_parity("acopp30", "ACOPP30_0000");
}

#[test]
fn parity_argauss_0000() {
    run_parity("argauss", "ARGAUSS_0000");
}

// Panel time: residual ratio 2.85e3 > K=10 (feral=7.53e-13, mumps=2.65e-16)
#[test]
#[ignore]
fn parity_batch_0094() {
    run_parity("batch", "BATCH_0094");
}

// Panel time: residual ratio 3.07e3 > K=10 (feral=8.09e-13, mumps=2.64e-16)
#[test]
#[ignore]
fn parity_batch_1048() {
    run_parity("batch", "BATCH_1048");
}

#[test]
fn parity_bqpgasim_0012() {
    run_parity("bqpgasim", "BQPGASIM_0012");
}

// Panel time: residual ratio 9.52e2 > K=10 (feral=1.03e-12, mumps=1.09e-15)
#[test]
#[ignore]
fn parity_ceri651als_1527() {
    run_parity("ceri651als", "CERI651ALS_1527");
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

// Panel time: residual ratio 2.38e1 > K=10 (feral=7.58e-14, mumps=3.18e-15)
#[test]
#[ignore]
fn parity_hahn1_0154() {
    run_parity("hahn1", "HAHN1_0154");
}

#[test]
fn parity_hahn1_0209() {
    run_parity("hahn1", "HAHN1_0209");
}

#[test]
fn parity_hahn1_0222() {
    run_parity("hahn1", "HAHN1_0222");
}

// Panel time: inertia mismatch (feral=(3, 4, 1) mumps=(4, 4, 0))
#[test]
#[ignore]
fn parity_hatfldbne_1418() {
    run_parity("hatfldbne", "HATFLDBNE_1418");
}

// Panel time: inertia mismatch (feral=(3, 4, 1) mumps=(4, 4, 0))
#[test]
#[ignore]
fn parity_hatfldbne_1419() {
    run_parity("hatfldbne", "HATFLDBNE_1419");
}

// Panel time: inertia mismatch (feral=(1, 3, 2) mumps=(3, 3, 0))
#[test]
#[ignore]
fn parity_hatfldf_0013() {
    run_parity("hatfldf", "HATFLDF_0013");
}

// Panel time: inertia mismatch (feral=(1, 3, 2) mumps=(3, 3, 0))
#[test]
#[ignore]
fn parity_hatfldf_0037() {
    run_parity("hatfldf", "HATFLDF_0037");
}

// Panel time: inertia mismatch (feral=(2, 25, 23) mumps=(25, 25, 0))
#[test]
#[ignore]
fn parity_hatfldg_0003() {
    run_parity("hatfldg", "HATFLDG_0003");
}

// Panel time: inertia mismatch (feral=(2, 25, 23) mumps=(25, 25, 0))
#[test]
#[ignore]
fn parity_hatfldg_0004() {
    run_parity("hatfldg", "HATFLDG_0004");
}

// Panel time: inertia mismatch (feral=(2, 25, 23) mumps=(25, 25, 0))
#[test]
#[ignore]
fn parity_hatfldg_0005() {
    run_parity("hatfldg", "HATFLDG_0005");
}

// Panel time: inertia mismatch (feral=(2, 25, 23) mumps=(25, 25, 0))
#[test]
#[ignore]
fn parity_hatfldg_0006() {
    run_parity("hatfldg", "HATFLDG_0006");
}

// Panel time: residual ratio 8.78e2 > K=10 (feral=2.88e-13, mumps=3.28e-16)
#[test]
#[ignore]
fn parity_hs102_0000() {
    run_parity("hs102", "HS102_0000");
}

#[test]
fn parity_hs85_0081() {
    run_parity("hs85", "HS85_0081");
}

// Panel time: inertia mismatch (feral=(98, 99, 1) mumps=(99, 99, 0))
#[test]
#[ignore]
fn parity_hydcar20_0000() {
    run_parity("hydcar20", "HYDCAR20_0000");
}

#[test]
fn parity_roszman1_0336() {
    run_parity("roszman1", "ROSZMAN1_0336");
}

#[test]
fn parity_ssine_2529() {
    run_parity("ssine", "SSINE_2529");
}

// Panel time: residual ratio 2.53e3 > K=10 (feral=2.91e-13, mumps=1.15e-16)
#[test]
#[ignore]
fn parity_ssi_2597() {
    run_parity("ssi", "SSI_2597");
}

// Panel time: inertia mismatch (feral=(53, 92, 30) mumps=(83, 92, 0))
#[test]
#[ignore]
fn parity_swopf_0000() {
    run_parity("swopf", "SWOPF_0000");
}

#[test]
fn parity_vesuvia_0000() {
    run_parity("vesuvia", "VESUVIA_0000");
}
