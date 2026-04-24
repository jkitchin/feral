//! Phase 2.4.3 Step 7 — rook rescue regression on CRESC100 / GAUSS2.
//!
//! These two families dominate the dense factor/MUMPS tail in the
//! corpus bench (CRESC100 at 40–45x and GAUSS2 at 41–44x pre-rook,
//! per `dev/plans/phase-2.4.3-rook-rescue.md` §Motivation). Both are
//! ill-conditioned KKT matrices that force BK-partial into delayed-
//! pivot cascades; rook rescue is the numerical fix.
//!
//! For each of 10 matrices from each family this test verifies:
//!   1. Inertia matches the MUMPS 5.8.2 oracle exactly (Sylvester
//!      hard gate per CLAUDE.md).
//!   2. Feral relative residual <= 10x MUMPS relative residual OR
//!      below the ABS_FLOOR = 1e-14 (same gate as `tests/parity.rs`).
//!   3. Rook rescue fires at least once across the panel — otherwise
//!      we are not actually exercising the rescue path on these
//!      matrices and the regression check is meaningless.
//!
//! Matrices are read directly from `data/matrices/kkt/{CRESC100,GAUSS2}/`
//! (not `tests/data/parity/`), so the test lives next to the bench
//! corpus rather than the curated parity panel.

use std::path::Path;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};

const K_RESIDUAL: f64 = 10.0;
const ABS_FLOOR: f64 = 1e-14;
const SAMPLE_SIZE: usize = 10;

/// Near-maximal pivot threshold (u = 0.99) intentionally forces
/// BK-partial rejections so the rook rescue path is exercised. At the
/// default u = 0.01 (and even u = 0.5) BK-partial + LAPACK extension
/// clears every pivot on these well-scaled KKT matrices and the rescue
/// counter stays at zero, which would make the `total_rescues > 0`
/// regression gate vacuous. With u = 0.99 we get 20-30 rescues per
/// CRESC100 matrix and 0-28 per GAUSS2 matrix — enough to guarantee
/// Sylvester's law of inertia is preserved *through* the rescue splice,
/// not just on paths that avoid it.
fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.99,
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

/// Returns `None` when MUMPS itself failed on this matrix
/// (factorization_status != "ok", residual may be null). Such entries
/// cannot serve as an oracle and are skipped by the panel. Example:
/// GAUSS2_0002 has status="fail" and residual=null in the bench corpus.
fn read_oracle(path: &Path) -> Option<(Inertia, f64, u64)> {
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read oracle"))
            .expect("parse oracle");
    if data["factorization_status"].as_str() != Some("ok") {
        return None;
    }
    let pos = data["inertia"]["positive"].as_u64()? as usize;
    let neg = data["inertia"]["negative"].as_u64()? as usize;
    let zero = data["inertia"]["zero"].as_u64()? as usize;
    let res = data["residual_2norm_relative"].as_f64()?;
    let factor_us = data["factor_us"].as_u64().unwrap_or(0);
    Some((Inertia::new(pos, neg, zero), res, factor_us))
}

struct Result {
    stem: String,
    n: usize,
    inertia_match: bool,
    feral_inertia: Inertia,
    mumps_inertia: Inertia,
    residual_gate_passed: bool,
    feral_res: f64,
    mumps_res: f64,
    n_rook_rescues: usize,
    feral_factor_us: u128,
    mumps_factor_us: u64,
}

fn run_family(family_dir: &str, stems: &[&str]) -> Vec<Result> {
    let mut results = Vec::with_capacity(stems.len());
    for stem in stems {
        let base = format!("data/matrices/kkt/{}/{}", family_dir, stem);
        let mtx = read_mtx(Path::new(&format!("{}.mtx", base))).expect("read mtx");
        let csc = mtx.to_csc().expect("to_csc");
        let sidecar = read_sidecar(Path::new(&format!("{}.json", base))).expect("sidecar");
        let rhs = sidecar.finite_rhs().expect("finite rhs");
        let Some((mumps_inertia, mumps_res, mumps_factor_us)) =
            read_oracle(Path::new(&format!("{}.mumps.json", base)))
        else {
            eprintln!("  skip {} — MUMPS oracle factorization_status != ok", stem);
            continue;
        };

        let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");

        let t0 = std::time::Instant::now();
        let (fac, inertia) =
            factorize_multifrontal(&csc, &sym, &ldlt_params()).expect("factor must succeed");
        let feral_factor_us = t0.elapsed().as_micros();

        let n_rook_rescues: usize = fac
            .node_factors
            .iter()
            .map(|nf| nf.frontal_factors.n_rook_rescues)
            .sum();

        let x = solve_sparse_refined(&csc, &fac, &rhs).expect("solve");
        let feral_res = rel_residual(&csc, &x, &rhs);

        let residual_target = (K_RESIDUAL * mumps_res).max(ABS_FLOOR);
        results.push(Result {
            stem: stem.to_string(),
            n: csc.n,
            inertia_match: inertia == mumps_inertia,
            feral_inertia: inertia,
            mumps_inertia,
            residual_gate_passed: feral_res <= residual_target,
            feral_res,
            mumps_res,
            n_rook_rescues,
            feral_factor_us,
            mumps_factor_us,
        });
    }
    results
}

fn report(family: &str, results: &[Result]) {
    eprintln!(
        "\n=== {} rook rescue panel ({} matrices) ===",
        family,
        results.len()
    );
    eprintln!(
        "{:24} {:>6} {:>9} {:>9} {:>12} {:>12} {:>8}",
        "stem", "n", "feral_us", "mumps_us", "feral_res", "mumps_res", "rescues"
    );
    for r in results {
        eprintln!(
            "{:24} {:>6} {:>9} {:>9} {:>12.3e} {:>12.3e} {:>8}",
            r.stem,
            r.n,
            r.feral_factor_us,
            r.mumps_factor_us,
            r.feral_res,
            r.mumps_res,
            r.n_rook_rescues
        );
    }
}

fn assert_family(family: &str, results: &[Result]) {
    let total_rescues: usize = results.iter().map(|r| r.n_rook_rescues).sum();
    let n_inertia_mismatch = results.iter().filter(|r| !r.inertia_match).count();
    let n_residual_fail = results.iter().filter(|r| !r.residual_gate_passed).count();

    for r in results {
        assert!(
            r.inertia_match,
            "{} {}: inertia mismatch — feral={}, mumps={}",
            family, r.stem, r.feral_inertia, r.mumps_inertia
        );
        assert!(
            r.residual_gate_passed,
            "{} {}: feral_res={:.3e} exceeds max(10*mumps={:.3e}, floor={:.3e})",
            family,
            r.stem,
            r.feral_res,
            K_RESIDUAL * r.mumps_res,
            ABS_FLOOR
        );
    }

    assert_eq!(
        n_inertia_mismatch, 0,
        "{} inertia mismatches: {}",
        family, n_inertia_mismatch
    );
    assert_eq!(
        n_residual_fail, 0,
        "{} residual failures: {}",
        family, n_residual_fail
    );

    assert!(
        total_rescues > 0,
        "{} rook rescue never fired across {} matrices — test is not \
         exercising the rescue path; rescue is either already disabled or \
         BK-partial is clearing every pivot without rescue",
        family,
        results.len()
    );
}

/// 10-matrix sample from CRESC100. Picked as the first 10 by index so
/// the test is deterministic and reproducible; these are the same
/// matrices that appear in the bench's "Top 10 worst" list rotations.
#[test]
fn test_rook_rescue_cresc100_panel() {
    let stems: [&str; SAMPLE_SIZE] = [
        "CRESC100_0000",
        "CRESC100_0001",
        "CRESC100_0002",
        "CRESC100_0003",
        "CRESC100_0004",
        "CRESC100_0005",
        "CRESC100_0006",
        "CRESC100_0007",
        "CRESC100_0008",
        "CRESC100_0009",
    ];
    let results = run_family("CRESC100", &stems);
    report("CRESC100", &results);
    assert_family("CRESC100", &results);
}

/// 10-matrix sample from GAUSS2.
#[test]
fn test_rook_rescue_gauss2_panel() {
    let stems: [&str; SAMPLE_SIZE] = [
        "GAUSS2_0000",
        "GAUSS2_0001",
        "GAUSS2_0002",
        "GAUSS2_0003",
        "GAUSS2_0004",
        "GAUSS2_0005",
        "GAUSS2_0006",
        "GAUSS2_0007",
        "GAUSS2_0008",
        "GAUSS2_0009",
    ];
    let results = run_family("GAUSS2", &stems);
    report("GAUSS2", &results);
    assert_family("GAUSS2", &results);
}
