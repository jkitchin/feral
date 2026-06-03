//! Diagnostic for feral issue #23 (ACOPP30 residual plateau).
//!
//! Factor each ACOPP30 matrix under default `Solver`, run iterative
//! refinement against the JSON-supplied RHS, report:
//!   - inertia returned
//!   - relative residual after factor+solve
//!   - relative residual after solve_refined
//!   - smallest |D| from the factor (proxy for pivot health)
//!
//! Goal: confirm the 8 matrices listed in
//! dev/research/inertia-triage-2026-04-27.md (0006, 0008, 0011, 0040,
//! 0046, 0047, 0049, 0058) plateau at 1.9e-2..2.6e-2 after IR, and
//! understand which guard in factor_frontal is allowing the bad pivot.
//!
//! Usage:
//!     cargo run --release --bin diag_acopp30_residual

use std::fs;
use std::path::Path;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams, SparseFactors};
use feral::numeric::solver::Solver;
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, CscMatrix};

const CORPUS: &str = "data/matrices/kkt/ACOPP30";
const SUSPECTS: &[usize] = &[6, 8, 11, 40, 46, 47, 49, 58];

fn parse_rhs(json: &str, n: usize) -> Option<Vec<f64>> {
    let key = "\"rhs\":[";
    let i = json.find(key)? + key.len();
    let rest = &json[i..];
    let end = rest.find(']')?;
    let body = &rest[..end];
    let v: Vec<f64> = body
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    if v.len() == n {
        Some(v)
    } else {
        None
    }
}

fn parse_inertia(json: &str) -> Option<(usize, usize)> {
    let key = "\"inertia\":{";
    let i = json.find(key)? + key.len();
    let chunk = &json[i..i + 80];
    let pos = grab_int(chunk, "\"positive\":")?;
    let neg = grab_int(chunk, "\"negative\":")?;
    Some((pos, neg))
}

fn grab_int(s: &str, key: &str) -> Option<usize> {
    let i = s.find(key)? + key.len();
    let rest = &s[i..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn matvec_lower_sym(csc: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for j in 0..csc.n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = csc.values[k];
            out[i] += v * x[j];
            if i != j {
                out[j] += v * x[i];
            }
        }
    }
}

fn relative_residual(csc: &CscMatrix, x: &[f64], rhs: &[f64]) -> f64 {
    let n = csc.n;
    let mut ax = vec![0.0; n];
    matvec_lower_sym(csc, x, &mut ax);
    let mut num2 = 0.0;
    let mut den2 = 0.0;
    for i in 0..n {
        let d = ax[i] - rhs[i];
        num2 += d * d;
        den2 += rhs[i] * rhs[i];
    }
    num2.sqrt() / den2.sqrt().max(1.0)
}

fn smallest_abs_d(factors: &SparseFactors) -> f64 {
    let mut min_abs = f64::INFINITY;
    for nf in &factors.node_factors {
        let ff = &nf.frontal_factors;
        let nelim = ff.nelim;
        let mut k = 0;
        while k < nelim {
            let two_by_two = k + 1 < nelim && ff.d_subdiag[k] != 0.0;
            if two_by_two {
                let a = ff.d_diag[k];
                let b = ff.d_subdiag[k];
                let c = ff.d_diag[k + 1];
                let trace = a + c;
                let det = a * c - b * b;
                let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
                let e1 = (trace - disc) * 0.5;
                let e2 = (trace + disc) * 0.5;
                min_abs = min_abs.min(e1.abs()).min(e2.abs());
                k += 2;
            } else {
                min_abs = min_abs.min(ff.d_diag[k].abs());
                k += 1;
            }
        }
    }
    min_abs
}

fn process(idx: usize) {
    let mtx_path = format!("{CORPUS}/ACOPP30_{idx:04}.mtx");
    let json_path = format!("{CORPUS}/ACOPP30_{idx:04}.json");
    if !Path::new(&mtx_path).exists() {
        println!("[{idx:04}] SKIP: missing mtx");
        return;
    }
    let mtx = match read_mtx(Path::new(&mtx_path)) {
        Ok(m) => m,
        Err(e) => {
            println!("[{idx:04}] read_mtx FAILED: {e:?}");
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            println!("[{idx:04}] to_csc FAILED: {e:?}");
            return;
        }
    };
    let json = fs::read_to_string(&json_path).expect("read json");
    let rhs = parse_rhs(&json, csc.n).expect("parse rhs");
    let (exp_pos, exp_neg) = parse_inertia(&json).expect("parse inertia");

    let scaling = std::env::var("DIAG_SCALING").ok();
    let strategy = match scaling.as_deref() {
        Some("infnorm") => ScalingStrategy::InfNorm,
        Some("identity") => ScalingStrategy::Identity,
        Some("mc64") => ScalingStrategy::Mc64Symmetric,
        _ => ScalingStrategy::default(),
    };
    let params = NumericParams {
        scaling: strategy.clone(),
        ..NumericParams::default()
    };

    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (factors, inertia) = match factorize_multifrontal(&csc, &sym, &params) {
        Ok(p) => p,
        Err(e) => {
            println!("[{idx:04}] factor FAILED: {e:?}");
            return;
        }
    };
    let min_d = smallest_abs_d(&factors);

    let mut solver = Solver::with_params(params, SupernodeParams::default());
    let status = solver.factor(&csc, None);
    if !matches!(status, feral::numeric::solver::FactorStatus::Success) {
        println!("[{idx:04}] solver.factor returned non-Success: {status:?}");
        return;
    }

    // Unrefined solve.
    let rhs_buf = rhs.clone();
    let x_raw = match solver.solve(&rhs_buf) {
        Ok(x) => x.to_vec(),
        Err(e) => {
            println!("[{idx:04}] solve FAILED: {e:?}");
            return;
        }
    };
    let rel_raw = relative_residual(&csc, &x_raw, &rhs);

    // Refined solve.
    let x_ref = match solver.solve_refined(&csc, &rhs) {
        Ok(x) => x.to_vec(),
        Err(e) => {
            println!("[{idx:04}] solve_refined FAILED: {e:?}");
            return;
        }
    };
    let rel_ref = relative_residual(&csc, &x_ref, &rhs);

    let inertia_ok = inertia.positive == exp_pos && inertia.negative == exp_neg;
    let inertia_tag = if inertia_ok { "OK " } else { "MIS" };
    println!(
        "[{idx:04}] inertia {inertia_tag} feral=({}, {}, {}) exp=({}, {}, 0)  min|D|={:>9.2e}  rel_raw={:>9.2e}  rel_ref={:>9.2e}",
        inertia.positive, inertia.negative, inertia.zero,
        exp_pos, exp_neg, min_d, rel_raw, rel_ref,
    );
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let targets: Vec<usize> = if args.is_empty() {
        SUSPECTS.to_vec()
    } else if args[0] == "--all" {
        (0..105).collect()
    } else {
        args.iter().filter_map(|a| a.parse().ok()).collect()
    };

    println!("ACOPP30 residual diagnostic ({} matrices)\n", targets.len());
    for idx in targets {
        process(idx);
    }
    Ok(())
}
