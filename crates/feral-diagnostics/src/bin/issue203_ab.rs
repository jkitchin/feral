//! Issue #203 — paired wall-clock A/B across ordering methods.
//!
//! The question this answers: the ordering work in
//! `dev/research/feral-metis-node-separator-fm-2026-09-17.md` is entirely
//! *symbolic* — fewer nonzeros in L, fewer modelled flops. Does any of it
//! reach the clock?
//!
//! Protocol, per `dev/decisions.md` (2026-08-09): every arm is timed once
//! per pair in a fixed order so drift hits all arms equally, `min` over
//! pairs is the per-arm statistic, and significance is an exact two-sided
//! sign test over the per-pair winners. Arms run in one process against one
//! matrix and one right-hand side, so nothing but the ordering differs.
//!
//! **Correctness gate, first.** The matrix is expected to be the augmented
//! system `[[H, J^T], [J, -dI]]` with `H` SPD and `d > 0`, whose inertia is
//! analytically `(n_vars, n - n_vars, 0)` — the Schur complement
//! `-dI - J H^-1 J^T` is negative definite whatever `J` is, so the oracle
//! needs no rank assumption and comes from outside the solver. Any arm that
//! misses it, or returns `rel_res > 1e-8`, is reported and cannot carry a
//! timing.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin issue203_ab \
//!       -- MATRIX.mtx N_VARS [PAIRS]

use feral::numeric::factorize::{factorize_multifrontal_parallel_with_workspace, FactorWorkspace};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, CscMatrix, NumericParams};
use std::time::Instant;

fn matvec(csc: &CscMatrix, x: &[f64]) -> Vec<f64> {
    let n = csc.n;
    let mut y = vec![0.0f64; n];
    for j in 0..n {
        for p in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[p];
            let a = csc.values[p];
            y[i] += a * x[j];
            if i != j {
                y[j] += a * x[i];
            }
        }
    }
    y
}

fn rel_res(csc: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let ax = matvec(csc, x);
    let rn: f64 = ax.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum();
    let bn: f64 = b.iter().map(|v| v * v).sum();
    if bn == 0.0 {
        0.0
    } else {
        (rn / bn).sqrt()
    }
}

struct Run {
    analyse_us: u128,
    factor_us: u128,
    solve_us: u128,
    inertia: (usize, usize, usize),
    rel: f64,
    nnz_l: usize,
}

fn one(
    csc: &CscMatrix,
    b: &[f64],
    method: OrderingMethod,
    ws: &mut FactorWorkspace,
) -> Option<Run> {
    let snode = SupernodeParams::default();
    let t = Instant::now();
    let sym = symbolic_factorize_with_method(csc, &snode, method).ok()?;
    let analyse_us = t.elapsed().as_micros();

    let params = NumericParams::default();
    let t = Instant::now();
    let (factors, inertia) =
        factorize_multifrontal_parallel_with_workspace(csc, &sym, &params, ws).ok()?;
    let factor_us = t.elapsed().as_micros();

    let t = Instant::now();
    let x = solve_sparse_refined(csc, &factors, b).ok()?;
    let solve_us = t.elapsed().as_micros();

    Some(Run {
        analyse_us,
        factor_us,
        solve_us,
        inertia: (inertia.positive, inertia.negative, inertia.zero),
        rel: rel_res(csc, &x, b),
        nnz_l: sym.factor_nnz_estimate,
    })
}

/// Exact two-sided sign test: probability of `k` or more extreme wins out
/// of `n` fair coin flips.
fn sign_test(wins: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let k = wins.min(n - wins);
    let mut tail = 0.0f64;
    for i in 0..=k {
        let mut c = 1.0f64;
        for j in 0..i {
            c *= (n - j) as f64 / (j + 1) as f64;
        }
        tail += c;
    }
    let p = 2.0 * tail / 2f64.powi(n as i32);
    p.min(1.0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: issue203_ab MATRIX.mtx N_VARS [PAIRS]");
        std::process::exit(2);
    }
    let n_vars: usize = args[1].parse().expect("N_VARS");
    let pairs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);

    let mtx = read_mtx(std::path::Path::new(&args[0])).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let n = csc.n;
    let expect = (n_vars, n - n_vars, 0usize);

    // b = K x_true with x_true[i] = 1 + i/n, the chain_proxy convention.
    let x_true: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 / n as f64).collect();
    let b = matvec(&csc, &x_true);

    let arms: Vec<(&str, OrderingMethod)> = vec![
        ("auto", OrderingMethod::Auto),
        ("amf", OrderingMethod::Amf),
        ("amd", OrderingMethod::Amd),
        ("metis", OrderingMethod::MetisND),
    ];

    println!(
        "matrix {} n={} nnz={} pairs={}",
        args[0],
        n,
        csc.row_idx.len(),
        pairs
    );
    println!("inertia oracle: ({}, {}, {})", expect.0, expect.1, expect.2);

    let mut ws = FactorWorkspace::new();
    let mut factor: Vec<Vec<u128>> = vec![Vec::new(); arms.len()];
    let mut total: Vec<Vec<u128>> = vec![Vec::new(); arms.len()];
    let mut first: Vec<Option<Run>> = (0..arms.len()).map(|_| None).collect();
    let mut bad: Vec<String> = Vec::new();

    for _ in 0..pairs {
        for (a, (label, method)) in arms.iter().enumerate() {
            match one(&csc, &b, method.clone(), &mut ws) {
                Some(r) => {
                    if r.inertia != expect {
                        bad.push(format!(
                            "{label}: inertia {:?} != oracle {:?}",
                            r.inertia, expect
                        ));
                    }
                    if r.rel > 1e-8 || r.rel.is_nan() {
                        bad.push(format!("{label}: rel_res {:.3e} > 1e-8", r.rel));
                    }
                    factor[a].push(r.factor_us);
                    total[a].push(r.analyse_us + r.factor_us + r.solve_us);
                    if first[a].is_none() {
                        first[a] = Some(r);
                    }
                }
                None => bad.push(format!("{label}: run failed")),
            }
        }
    }

    bad.sort();
    bad.dedup();
    if !bad.is_empty() {
        println!("\n*** CORRECTNESS GATE FAILED — timings below cannot be quoted ***");
        for m in &bad {
            println!("  {m}");
        }
    } else {
        println!("correctness gate: all arms hit the inertia oracle, rel_res <= 1e-8");
    }

    println!(
        "\n{:<8} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "arm", "nnz_L", "analyse_us", "min_factor", "min_total", "rel_res"
    );
    for (a, (label, _)) in arms.iter().enumerate() {
        let Some(r) = &first[a] else { continue };
        let mf = factor[a].iter().copied().min().unwrap_or(0);
        let mt = total[a].iter().copied().min().unwrap_or(0);
        println!(
            "{label:<8} {:>12} {:>12} {:>12} {:>12} {:>12.2e}",
            r.nnz_l, r.analyse_us, mf, mt, r.rel
        );
    }

    // Every arm against `auto`, the current default.
    println!(
        "\n{:<8} {:>14} {:>14} {:>10} {:>9}",
        "arm", "factor vs auto", "total vs auto", "wins", "p"
    );
    for (a, (label, _)) in arms.iter().enumerate() {
        if a == 0 {
            continue;
        }
        let base_f = factor[0].iter().copied().min().unwrap_or(1).max(1);
        let base_t = total[0].iter().copied().min().unwrap_or(1).max(1);
        let mf = factor[a].iter().copied().min().unwrap_or(1).max(1);
        let mt = total[a].iter().copied().min().unwrap_or(1).max(1);
        let wins = factor[0]
            .iter()
            .zip(&factor[a])
            .filter(|(base, arm)| arm < base)
            .count();
        let np = factor[0].len().min(factor[a].len());
        println!(
            "{label:<8} {:>14.3} {:>14.3} {:>10} {:>9.4}",
            base_f as f64 / mf as f64,
            base_t as f64 / mt as f64,
            format!("{wins}/{np}"),
            sign_test(wins, np)
        );
    }
    println!("\n(ratio > 1 means the arm is faster than `auto`)");
}
