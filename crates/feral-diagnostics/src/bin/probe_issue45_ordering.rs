//! Decisive experiment for the #45/#46 root cause.
//!
//! The journal experiment "complete the whole matrix, then factor it"
//! gave a fast factor (606 ms) but a garbage solve (residual 2e22).
//! That experiment changed the matrix pattern; this probe isolates
//! *what* about the pattern change actually flips the behaviour.
//!
//! `pick_scaling_strategy` (the `ScalingStrategy::Auto` router) counts
//! diagonal-only columns. Completing the diagonal can flip the router
//! from `InfNorm` to `Mc64Symmetric`. This probe crosses numeric input
//! (stripped vs diagonal-completed) with an *explicit* scaling strategy
//! so the scaling variable is held fixed, and reports factor time,
//! factor nnz, inertia, pivot magnitudes, and solve residual.
//!
//! Usage: cargo run --release --bin probe_issue45_ordering [-- <kkt.mtx> <rhs.txt>]

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse;
use feral::scaling::{compute_scaling, pick_scaling_strategy, ScalingStrategy};
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, CscMatrix, NumericParams};

const DEFAULT_MTX: &str =
    "/Users/jkitchin/projects/pounce/benchmarks/cho/feral_repro/cho_iter0_kkt.mtx";
const DEFAULT_RHS: &str =
    "/Users/jkitchin/projects/pounce/benchmarks/cho/feral_repro/cho_iter0_rhs.txt";

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Copy of `csc` with an explicit `0.0` diagonal for every column that
/// lacks a stored diagonal entry.
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

fn run(label: &str, m: &CscMatrix, scaling: ScalingStrategy, rhs: &[f64]) {
    let snode = SupernodeParams::default();
    let np = NumericParams {
        scaling,
        ..NumericParams::default()
    };
    let sym = match symbolic_factorize_with_method(m, &snode, OrderingMethod::Auto) {
        Ok(s) => s,
        Err(e) => {
            println!("{label:<34} symbolic failed: {e:?}");
            return;
        }
    };
    let t = Instant::now();
    let (factors, inertia) = match factorize_multifrontal(m, &sym, &np) {
        Ok(fi) => fi,
        Err(e) => {
            println!("{label:<34} numeric failed: {e:?}");
            return;
        }
    };
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let fnnz = factors.factor_nnz();
    let minp = factors.min_pivot_magnitude().unwrap_or(f64::NAN);
    let maxp = factors.max_pivot_magnitude().unwrap_or(f64::NAN);
    match solve_sparse(&factors, rhs) {
        Ok(x) => {
            let mut ax = vec![0.0; m.n];
            m.symv(&x, &mut ax);
            let r: Vec<f64> = ax.iter().zip(rhs).map(|(&a, &b)| a - b).collect();
            let relres = norm_inf(&r) / norm_inf(rhs).max(1.0);
            println!(
                "{label:<34} {ms:>8.0}ms  fnnz={fnnz:>9}  \
                 inertia=({},{},{})  pivot[{minp:.2e},{maxp:.2e}]  relres={relres:.3e}",
                inertia.positive, inertia.negative, inertia.zero
            );
        }
        Err(e) => println!("{label:<34} solve failed: {e:?}"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mtx_path = args.next().unwrap_or_else(|| DEFAULT_MTX.to_string());
    let rhs_path = args.next().unwrap_or_else(|| DEFAULT_RHS.to_string());
    if !Path::new(&mtx_path).exists() {
        eprintln!("SKIP: {mtx_path} not present");
        std::process::exit(2);
    }
    let stripped = match read_mtx(Path::new(&mtx_path)).and_then(|m| m.to_csc()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("load failed: {e:?}");
            std::process::exit(1);
        }
    };
    let rhs: Vec<f64> = match std::fs::read_to_string(&rhs_path) {
        Ok(t) => t
            .split_whitespace()
            .filter_map(|w| w.parse::<f64>().ok())
            .collect(),
        Err(_) => vec![1.0; stripped.n],
    };
    let rhs = if rhs.len() == stripped.n {
        rhs
    } else {
        vec![1.0; stripped.n]
    };

    let completed = complete_diagonal(&stripped);
    println!(
        "n={}, stripped nnz={}, completed nnz={}",
        stripped.n, stripped.col_ptr[stripped.n], completed.col_ptr[completed.n]
    );
    println!(
        "Auto scaling router:  stripped -> {:?},  completed -> {:?}",
        pick_scaling_strategy(&stripped),
        pick_scaling_strategy(&completed),
    );
    for sc in [ScalingStrategy::InfNorm, ScalingStrategy::Mc64Symmetric] {
        match compute_scaling(&stripped, &sc) {
            Ok((v, info)) => {
                let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = v.iter().cloned().fold(0.0_f64, f64::max);
                let n_zero = v.iter().filter(|&&x| x == 0.0).count();
                let n_tiny = v.iter().filter(|&&x| x != 0.0 && x.abs() < 1e-30).count();
                println!(
                    "scaling {sc:?}: min={mn:.3e} max={mx:.3e} \
                     zeros={n_zero} tiny(<1e-30)={n_tiny} info={info:?}"
                );
            }
            Err(e) => println!("scaling {sc:?}: failed {e:?}"),
        }
    }
    println!();

    for (mlabel, m) in [("stripped", &stripped), ("completed", &completed)] {
        for sc in [
            ScalingStrategy::Identity,
            ScalingStrategy::InfNorm,
            ScalingStrategy::Mc64Symmetric,
            ScalingStrategy::Auto,
        ] {
            run(&format!("{mlabel:<10} {sc:?}"), m, sc, &rhs);
        }
    }
}
