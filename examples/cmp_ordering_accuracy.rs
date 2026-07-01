//! Issue #102 follow-up: does the OrderingPreprocess choice change solve
//! ACCURACY (not just fill)? Factor a dumped KKT under None / LdltCompress /
//! Auto, solve the real RHS, and report the relative residual + min pivot.
//!
//! Usage: cargo run --release --example cmp_ordering_accuracy -- <kkt.mtx> [rhs.txt]

use feral::symbolic::{OrderingPreprocess, SupernodeParams};
use feral::{read_mtx, CscMatrix, NumericParams, Solver};
use std::path::Path;

/// y = A·x for a symmetric matrix stored as lower-triangle CSC.
fn sym_matvec(a: &CscMatrix, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; a.n];
    for j in 0..a.n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let v = a.values[k];
            y[i] += v * x[j];
            if i != j {
                y[j] += v * x[i];
            }
        }
    }
    y
}

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let ax = sym_matvec(a, x);
    let mut rnum = 0.0f64;
    let mut bden = 0.0f64;
    for i in 0..a.n {
        rnum = rnum.max((ax[i] - b[i]).abs());
        bden = bden.max(b[i].abs());
    }
    rnum / bden.max(1e-300)
}

fn run(name: &str, a: &CscMatrix, b: &[f64], build: impl Fn() -> Solver) {
    let mut s = build();
    let st = s.factor(a, None);
    let (nnz_l, minp, maxp) = s
        .last_factor_stats()
        .map(|f| (f.nnz_l, f.min_abs_pivot, f.max_abs_pivot))
        .unwrap_or((0, 0.0, 0.0));
    let (res, res_ref) = match s.solve(b) {
        Ok(x) => {
            let r = rel_residual(a, &x, b);
            let rr = s
                .solve_refined(a, b)
                .map(|xr| rel_residual(a, &xr, b))
                .unwrap_or(f64::NAN);
            (r, rr)
        }
        Err(e) => {
            println!("  {name:<20} factor={st:?} solve ERR {e:?}");
            return;
        }
    };
    println!(
        "  {name:<20} nnz_L={nnz_l:>9} |piv|=[{minp:.2e},{maxp:.2e}] growth={:.1e}  \
         resid={res:.3e}  refined={res_ref:.3e}  {st:?}",
        maxp / minp.max(1e-300)
    );
}

fn main() {
    let mtx = std::env::args().nth(1).expect("usage: <kkt.mtx> [rhs.txt]");
    let a = read_mtx(Path::new(&mtx))
        .and_then(|m| m.to_csc())
        .expect("read mtx");
    let b: Vec<f64> = match std::env::args().nth(2) {
        Some(p) => std::fs::read_to_string(p)
            .expect("read rhs")
            .split_whitespace()
            .map(|s| s.parse().expect("rhs f64"))
            .collect(),
        None => vec![1.0; a.n],
    };
    assert_eq!(b.len(), a.n, "rhs length != n");
    println!("matrix: n={} nnz={}", a.n, a.row_idx.len());

    let none = || {
        Solver::with_params(
            NumericParams::default(),
            SupernodeParams {
                preprocess: OrderingPreprocess::None,
                ..SupernodeParams::default()
            },
        )
    };
    let ldlt = || {
        Solver::with_params(
            NumericParams::default(),
            SupernodeParams {
                preprocess: OrderingPreprocess::LdltCompress,
                ..SupernodeParams::default()
            },
        )
    };
    run("Auto (default)", &a, &b, Solver::new);
    run("None", &a, &b, none);
    run("LdltCompress", &a, &b, ldlt);
}
