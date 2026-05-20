//! Issue #46 review probe — LDLᵀ factor time / fill on the CHO KKT.
//!
//! Runs the issue's ordering sweep on `cho_iter0_kkt.mtx` and, for each
//! `OrderingMethod`, prints factor wall time, factor nnz (fill), and the
//! solve residual. Then repeats with the structural diagonal completed
//! (an explicit `0.0` inserted for every column lacking a diagonal) —
//! the decisive test of whether the missing diagonal drives the fill.
//!
//! Usage: cargo run --release --bin probe_issue46 [-- <kkt.mtx>]

use std::path::Path;
use std::time::Instant;

use feral::symbolic::OrderingMethod;
use feral::{read_mtx, CscMatrix, Solver};

const DEFAULT_MTX: &str =
    "/Users/jkitchin/projects/pounce/benchmarks/cho/feral_repro/cho_iter0_kkt.mtx";

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Return a copy of `csc` with an explicit `0.0` diagonal entry inserted
/// for every column that lacks one — the structurally-complete diagonal
/// POUNCE's KKT pattern actually carries (issue #46).
fn complete_diagonal(csc: &CscMatrix) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..csc.n {
        let mut has_diag = false;
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            rows.push(csc.row_idx[k]);
            cols.push(j);
            vals.push(csc.values[k]);
            if csc.row_idx[k] == j {
                has_diag = true;
            }
        }
        if !has_diag {
            rows.push(j);
            cols.push(j);
            vals.push(0.0);
        }
    }
    CscMatrix::from_triplets(csc.n, &rows, &cols, &vals)
        .expect("diagonal-completed triplets are valid lower-triangle")
}

fn run(label: &str, csc: &CscMatrix, in_nnz: usize, m: OrderingMethod, rhs: &[f64]) {
    let mut s = Solver::new().with_ordering(m);
    let t = Instant::now();
    let status = s.factor(csc, None);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let fnnz = s.factors().map(|f| f.factor_nnz()).unwrap_or(0);
    let fill = fnnz as f64 / in_nnz as f64;
    let rel = match s.solve(rhs) {
        Ok(x) => {
            let mut ax = vec![0.0; csc.n];
            csc.symv(&x, &mut ax);
            let r: Vec<f64> = ax.iter().zip(rhs).map(|(&a, &b)| a - b).collect();
            norm_inf(&r) / norm_inf(rhs).max(1.0)
        }
        Err(_) => f64::NAN,
    };
    println!("{label:<28} {ms:>10.0} {fnnz:>13} {fill:>8.1}x {rel:>11.2e}  {status:?}");
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MTX.to_string());
    if !Path::new(&path).exists() {
        eprintln!("SKIP: {path} not present");
        std::process::exit(2);
    }
    let csc = match read_mtx(Path::new(&path)).and_then(|m| m.to_csc()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("load failed: {e:?}");
            std::process::exit(1);
        }
    };
    let in_nnz = csc.col_ptr[csc.n];
    let n_diag = (0..csc.n)
        .filter(|&j| csc.row_idx[csc.col_ptr[j]..csc.col_ptr[j + 1]].contains(&j))
        .count();
    println!(
        "matrix: n={}, input nnz={in_nnz}, columns with a stored diagonal={n_diag} \
         ({} missing)",
        csc.n,
        csc.n - n_diag
    );

    let completed = complete_diagonal(&csc);
    let comp_nnz = completed.col_ptr[completed.n];
    println!("diagonal-completed: nnz={comp_nnz}");

    // Unit RHS so the solve exercises the factor.
    let rhs = vec![1.0_f64; csc.n];

    println!(
        "\n{:<28} {:>10} {:>13} {:>9} {:>11}  status",
        "case", "factor_ms", "factor_nnz", "fill_x", "rel_res"
    );
    for m in [
        OrderingMethod::Auto,
        OrderingMethod::Amd,
        OrderingMethod::Amf,
        OrderingMethod::MetisND,
        OrderingMethod::ScotchND,
        OrderingMethod::KahipND,
    ] {
        run(&format!("stripped {m:?}"), &csc, in_nnz, m, &rhs);
    }
    println!("  --- structural diagonal completed (in-nnz held at stripped value) ---");
    for m in [OrderingMethod::Auto, OrderingMethod::Amf] {
        run(
            &format!("diag-completed {m:?}"),
            &completed,
            in_nnz,
            m,
            &rhs,
        );
    }
}
