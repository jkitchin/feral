//! Issue #203 — why do feral and MA57 disagree on inertia?
//!
//! On three `laptime` iterates feral reports an inertia ~22-26 away
//! from MA57's, with good residuals on both sides. The leading
//! hypothesis is the pivot threshold: `NumericParams::default()` sets
//! `bk.pivot_threshold = 1e-8` (MA27's `cntl[1]`, and Ipopt's
//! `ma27_pivtol` default), while MA57's `CNTL(1)` defaults to `1e-2` —
//! six orders of magnitude tighter. A loose threshold accepts tiny
//! pivots whose sign is decided by rounding noise.
//!
//! This sweeps the threshold and prints the inertia and residual at
//! each value. If the disagreement is a threshold artefact, feral's
//! inertia should migrate toward MA57's as the threshold tightens.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin issue203_inertia_probe \
//!       -- MATRIX.mtx [EXPECT_POS EXPECT_NEG]

use feral::symbolic::OrderingMethod;
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};

fn matvec(k: &CscMatrix, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; k.n];
    for j in 0..k.n {
        for p in k.col_ptr[j]..k.col_ptr[j + 1] {
            let i = k.row_idx[p];
            let a = k.values[p];
            y[i] += a * x[j];
            if i != j {
                y[j] += a * x[i];
            }
        }
    }
    y
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: issue203_inertia_probe MATRIX.mtx [EXPECT_POS EXPECT_NEG]");
        std::process::exit(2);
    };
    let expect: Option<(usize, usize)> = match (args.get(1), args.get(2)) {
        (Some(p), Some(n)) => Some((p.parse().unwrap_or(0), n.parse().unwrap_or(0))),
        _ => None,
    };

    let mtx = read_mtx(std::path::Path::new(path)).expect("read_mtx");
    let m = mtx.to_csc().expect("to_csc");
    let x_true: Vec<f64> = (0..m.n).map(|i| 1.0 + i as f64 / m.n as f64).collect();
    let b = matvec(&m, &x_true);

    println!(
        "matrix {} n={}",
        std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path),
        m.n
    );
    if let Some((p, n)) = expect {
        println!("reference inertia: ({p}, {n}, 0)");
    }
    {
        let mut s = Solver::new();
        if matches!(s.factor(&m, None), FactorStatus::Success) {
            match s.estimate_condition_1norm(&m) {
                Ok(c) => println!("condition estimate (1-norm): {c:.3e}"),
                Err(e) => println!("condition estimate failed: {e:?}"),
            }
        }
    }
    println!(
        "\n{:<8} {:>10} {:>10} {:>9} {:>7} {:>8} {:>11}",
        "ordering", "pivtol", "pos", "neg", "zero", "d(pos)", "rel_res"
    );

    for (label, method) in [
        ("amf", OrderingMethod::Amf),
        ("metis", OrderingMethod::MetisND),
    ] {
        for tol in [1e-8f64, 1e-6, 1e-4, 1e-2, 1e-1] {
            let mut s = Solver::new().with_ordering(method.clone());
            // `NumericParams` is not exposed field-by-field on `Solver`,
            // so rebuild it and hand it over via `with_params`.
            let mut np = feral::NumericParams::default();
            np.bk.pivot_threshold = tol;
            let mut s2 = Solver::with_params(np, feral::symbolic::SupernodeParams::default())
                .with_ordering(method.clone());
            std::mem::swap(&mut s, &mut s2);
            match s.factor(&m, None) {
                FactorStatus::Success => {
                    let i = s.inertia().expect("inertia").clone();
                    let x = s.solve_refined(&m, &b).unwrap_or_default();
                    let rel = if x.len() == m.n {
                        let ax = matvec(&m, &x);
                        let rn: f64 = ax.iter().zip(&b).map(|(a, b)| (a - b) * (a - b)).sum();
                        let bn: f64 = b.iter().map(|v| v * v).sum();
                        (rn / bn).sqrt()
                    } else {
                        f64::NAN
                    };
                    let d = expect
                        .map(|(p, _)| i.positive as i64 - p as i64)
                        .unwrap_or(0);
                    println!(
                        "{label:<8} {tol:>10.0e} {:>10} {:>9} {:>7} {:>8} {rel:>11.2e}",
                        i.positive, i.negative, i.zero, d
                    );
                }
                st => println!("{label:<8} {tol:>10.0e}  FAILED {st:?}"),
            }
        }
    }
}
