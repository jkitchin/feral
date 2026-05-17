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

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};

const K_RESIDUAL: f64 = 10.0;
const ABS_FLOOR: f64 = 1e-14;

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    })
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

// Panel snapshot: 13/26 matrices pass MUMPS parity at panel time.
// Failing matrices are `#[ignore]`'d with the panel-time failure
// mode in the attribute comment. Passing matrices run as regular
// tests and protect against regression. As fixes land, rerun
// `cargo run --release --example select_parity_panel` to refresh
// the panel and un-ignore the now-passing matrices.

// Panel time: inertia mismatch (feral=(38, 68, 0) mumps=(37, 68, 1))
#[test]
#[ignore]
fn parity_acopp14_0001() {
    run_parity("acopp14", "ACOPP14_0001");
}

// Panel time: inertia mismatch (feral=(38, 68, 0) mumps=(37, 68, 1))
#[test]
#[ignore]
fn parity_acopp14_0003() {
    run_parity("acopp14", "ACOPP14_0003");
}

// Panel time: inertia mismatch (feral=(71, 138, 0) mumps=(71, 137, 1))
#[test]
#[ignore]
fn parity_acopp30_0000() {
    run_parity("acopp30", "ACOPP30_0000");
}

// Panel time: inertia mismatch (feral=(71, 138, 0) mumps=(72, 137, 0))
#[test]
#[ignore]
fn parity_acopp30_0001() {
    run_parity("acopp30", "ACOPP30_0001");
}

// Panel time: inertia mismatch (feral=(71, 137, 1) mumps=(72, 137, 0))
#[test]
#[ignore]
fn parity_acopp30_0005() {
    run_parity("acopp30", "ACOPP30_0005");
}

// Panel time: inertia mismatch (feral=(7, 0, 0) mumps=(6, 0, 1))
#[test]
#[ignore]
fn parity_ceri651cls_0486() {
    run_parity("ceri651cls", "CERI651CLS_0486");
}

// Panel time: inertia mismatch (feral=(7, 0, 0) mumps=(6, 0, 1))
#[test]
#[ignore]
fn parity_ceri651cls_0487() {
    run_parity("ceri651cls", "CERI651CLS_0487");
}

#[test]
fn parity_ceri651dls_0618() {
    run_parity("ceri651dls", "CERI651DLS_0618");
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

// Panel time: inertia mismatch (feral=(5, 0, 1) mumps=(6, 0, 0))
#[test]
#[ignore]
fn parity_fbrain3ls_0839() {
    run_parity("fbrain3ls", "FBRAIN3LS_0839");
}

#[test]
fn parity_hatfldbne_1418() {
    run_parity("hatfldbne", "HATFLDBNE_1418");
}

#[test]
fn parity_hatfldbne_1419() {
    run_parity("hatfldbne", "HATFLDBNE_1419");
}

#[test]
fn parity_himmelbj_0032() {
    run_parity("himmelbj", "HIMMELBJ_0032");
}

#[test]
fn parity_hydcar20_0000() {
    run_parity("hydcar20", "HYDCAR20_0000");
}

#[test]
fn parity_meyer3ne_0220() {
    run_parity("meyer3ne", "MEYER3NE_0220");
}

#[test]
fn parity_meyer3ne_0259() {
    run_parity("meyer3ne", "MEYER3NE_0259");
}

#[test]
fn parity_muonsine_0019() {
    run_parity("muonsine", "MUONSINE_0019");
}

#[test]
fn parity_muonsine_0027() {
    run_parity("muonsine", "MUONSINE_0027");
}

#[test]
fn parity_roszman1_0241() {
    run_parity("roszman1", "ROSZMAN1_0241");
}

// Panel time: residual ratio 3.39e2 > K=10 (feral=8.59e-13, mumps=2.53e-15)
#[test]
#[ignore]
fn parity_ssi_1685() {
    run_parity("ssi", "SSI_1685");
}

// Panel time: residual ratio 2.53e2 > K=10 (feral=3.02e-14, mumps=1.19e-16)
#[test]
#[ignore]
fn parity_ssi_2412() {
    run_parity("ssi", "SSI_2412");
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
fn parity_vesuviou_0030() {
    run_parity("vesuviou", "VESUVIOU_0030");
}
