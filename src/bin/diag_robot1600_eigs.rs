//! Diagnostic for feral issue #17 (robot_1600 WrongInertia loop).
//!
//! Takes one or more KKT `.mtx` files dumped from pounce-feral on
//! robot_1600 and prints, for each:
//!   - feral inertia under `cascade=off` and `cascade=default`
//!   - the K smallest |eigenvalue(D)| values from each BK factor
//!   - counts of pivots below thresholds 1e-12, 1e-10, 1e-8
//!
//! Together this pinpoints the near-zero-eigenvalue region where
//! sign assignment differs between configurations (and, by reference,
//! between feral and MA57/MUMPS).
//!
//! Usage:
//!     cargo run --release --bin diag_robot1600_eigs -- /path/to/iter004.mtx [...]

use feral::numeric::factorize::{factorize_multifrontal, NumericParams, SparseFactors};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, CscMatrix};
use std::path::Path;

const K_SHOW: usize = 12;

fn eigs_of_d(factors: &SparseFactors) -> Vec<f64> {
    let mut eigs = Vec::new();
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
                eigs.push((trace - disc) * 0.5);
                eigs.push((trace + disc) * 0.5);
                k += 2;
            } else {
                eigs.push(ff.d_diag[k]);
                k += 1;
            }
        }
    }
    eigs
}

fn report(label: &str, csc: &CscMatrix, params: NumericParams) {
    let snode = SupernodeParams::default();
    let sym = match symbolic_factorize(csc, &snode) {
        Ok(s) => s,
        Err(e) => {
            println!("  [{label:<10}] symbolic FAILED: {e:?}");
            return;
        }
    };
    let (factors, inertia) = match factorize_multifrontal(csc, &sym, &params) {
        Ok(p) => p,
        Err(e) => {
            println!("  [{label:<10}] factor FAILED: {e:?}");
            return;
        }
    };
    let eigs = eigs_of_d(&factors);

    let mut by_abs: Vec<(usize, f64)> = eigs.iter().copied().enumerate().collect();
    by_abs.sort_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap());

    let n_below_1e12 = eigs.iter().filter(|v| v.abs() < 1e-12).count();
    let n_below_1e10 = eigs.iter().filter(|v| v.abs() < 1e-10).count();
    let n_below_1e8 = eigs.iter().filter(|v| v.abs() < 1e-8).count();
    let n_below_1e6 = eigs.iter().filter(|v| v.abs() < 1e-6).count();

    println!(
        "  [{label:<10}] inertia={inertia}  |D|<1e-12:{n_below_1e12}  <1e-10:{n_below_1e10}  <1e-8:{n_below_1e8}  <1e-6:{n_below_1e6}"
    );
    print!("    smallest |eigs|: ");
    for (i, (_idx, v)) in by_abs.iter().take(K_SHOW).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{v:+.3e}");
    }
    println!();
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: diag_robot1600_eigs <mtx> [<mtx> ...]");
        std::process::exit(2);
    }

    for path in &args {
        let mtx = match read_mtx(Path::new(path)) {
            Ok(m) => m,
            Err(e) => {
                println!("\n[{path}] read_mtx FAILED: {e:?}");
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                println!("\n[{path}] to_csc FAILED: {e:?}");
                continue;
            }
        };
        println!("\n[{path}] n={} nnz={}", csc.n, csc.row_idx.len());
        let off = NumericParams {
            cascade_break_ratio: None,
            cascade_break_eps: None,
            ..NumericParams::default()
        };
        report("cb=off    ", &csc, off);
        report("cb=default", &csc, NumericParams::default());
        let pounce = NumericParams {
            cascade_break_ratio: Some(0.5),
            cascade_break_eps: Some(1e-10),
            ..NumericParams::default()
        };
        report("cb=pounce ", &csc, pounce);
    }
    Ok(())
}
