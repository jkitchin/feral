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

// Panel snapshot: 11/31 matrices pass MUMPS parity at panel time.
// Failing matrices are `#[ignore]`'d with the panel-time failure
// mode in the attribute comment. Passing matrices run as regular
// tests and protect against regression. As fixes land, rerun
// `cargo run --release --example select_parity_panel` to refresh
// the panel and un-ignore the now-passing matrices.

// Panel time: inertia mismatch (feral=(56, 122, 31) mumps=(71, 137, 1))
#[test]
#[ignore]
fn parity_acopp30_0000() {
    run_parity("acopp30", "ACOPP30_0000");
}

// Panel time: inertia mismatch (feral=(397, 163, 4) mumps=(400, 164, 0))
#[test]
#[ignore]
fn parity_acopr30_0159() {
    run_parity("acopr30", "ACOPR30_0159");
}

// Panel time: inertia mismatch (feral=(397, 163, 4) mumps=(400, 164, 0))
#[test]
#[ignore]
fn parity_acopr30_0166() {
    run_parity("acopr30", "ACOPR30_0166");
}

#[test]
fn parity_argauss_0000() {
    run_parity("argauss", "ARGAUSS_0000");
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

// Panel time: residual ratio 8.16e2 > K=10 (feral=1.14e-12, mumps=1.40e-15)
#[test]
#[ignore]
fn parity_ceri651c_2107() {
    run_parity("ceri651c", "CERI651C_2107");
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
fn parity_hatfldbne_0971() {
    run_parity("hatfldbne", "HATFLDBNE_0971");
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
fn parity_hatfldbne_1586() {
    run_parity("hatfldbne", "HATFLDBNE_1586");
}

// Panel time: inertia mismatch (feral=(3, 4, 1) mumps=(4, 4, 0))
#[test]
#[ignore]
fn parity_hatfldbne_2837() {
    run_parity("hatfldbne", "HATFLDBNE_2837");
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

// Panel time: inertia mismatch (feral=(23, 23, 152) mumps=(99, 99, 0))
#[test]
#[ignore]
fn parity_hydcar20_0000() {
    run_parity("hydcar20", "HYDCAR20_0000");
}

// Panel time: inertia mismatch (feral=(19, 20, 23) mumps=(31, 31, 0))
#[test]
#[ignore]
fn parity_methanl8_0000() {
    run_parity("methanl8", "METHANL8_0000");
}

// Panel time: residual ratio 4.24e1 > K=10 (feral=1.25e-14, mumps=2.95e-16)
#[test]
#[ignore]
fn parity_meyer3ne_0202() {
    run_parity("meyer3ne", "MEYER3NE_0202");
}

// Panel time: residual ratio 1.64e1 > K=10 (feral=1.36e-13, mumps=8.29e-15)
#[test]
#[ignore]
fn parity_meyer3ne_0261() {
    run_parity("meyer3ne", "MEYER3NE_0261");
}

#[test]
fn parity_roszman1_0225() {
    run_parity("roszman1", "ROSZMAN1_0225");
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

// Panel time: inertia mismatch (feral=(49, 67, 59) mumps=(83, 92, 0))
#[test]
#[ignore]
fn parity_swopf_0000() {
    run_parity("swopf", "SWOPF_0000");
}

#[test]
fn parity_vesuvia_0000() {
    run_parity("vesuvia", "VESUVIA_0000");
}

// Panel time: inertia mismatch (feral=(2058, 1022, 3) mumps=(2058, 1025, 0))
#[test]
#[ignore]
fn parity_vesuviou_0015() {
    run_parity("vesuviou", "VESUVIOU_0015");
}
