//! Probe for issue #49 — the #47 value-aware scaling routing regression
//! on the Mittelmann `ex4_2` explicit-zero-(2,2) KKT.
//!
//! Loads a KKT `.mtx` (zeros preserved by `read_mtx`/`from_triplets`),
//! prints the routing decision both ways — value-aware (current,
//! post-#47) and value-blind (pre-#47) — then factors the matrix under
//! each of `Auto`, `Mc64Symmetric`, `InfNorm` and reports factor time,
//! `nnz_L`, delayed-pivot count, inertia, scaling info, and residual.
//!
//! This is a diagnostic probe; the relaxed probe-bin convention
//! (unwrap/expect permitted) applies.
//!
//! Usage: `cargo run --release --bin probe_issue49 -- KKT.mtx [RHS.txt]`

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal_parallel_with_workspace, FactorWorkspace};
use feral::numeric::solve::solve_sparse_refined;
use feral::scaling::{pick_scaling_strategy, ScalingStrategy};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, CscMatrix, NumericParams, Solver};

/// Replicate `pick_scaling_strategy` BUT counting stored entries
/// (the pre-#47, value-blind logic) so the routing flip is visible.
fn route_value_blind(m: &CscMatrix) -> (ScalingStrategy, usize, usize) {
    let n = m.n;
    let mut diag_only = 0usize;
    let mut max_col_nnz = 0usize;
    for j in 0..n {
        let start = m.col_ptr[j];
        let end = m.col_ptr[j + 1];
        let nnz_col = end - start;
        if nnz_col > max_col_nnz {
            max_col_nnz = nnz_col;
        }
        if nnz_col == 1 && m.row_idx[start] == j {
            diag_only += 1;
        }
    }
    let has_arrow_head = max_col_nnz > 32;
    let has_slack_mass = diag_only as f64 / n as f64 >= 0.3;
    let strat = if has_arrow_head && has_slack_mass {
        ScalingStrategy::Mc64Symmetric
    } else {
        ScalingStrategy::InfNorm
    };
    (strat, diag_only, max_col_nnz)
}

/// Same shape but value-aware (mirrors current `pick_scaling_strategy`),
/// so the diag_only / max_col_nnz numbers can be printed alongside.
fn route_value_aware(m: &CscMatrix) -> (ScalingStrategy, usize, usize) {
    let n = m.n;
    let mut diag_only = 0usize;
    let mut max_col_nnz = 0usize;
    for j in 0..n {
        let start = m.col_ptr[j];
        let end = m.col_ptr[j + 1];
        let mut nnz_col = 0usize;
        let mut diag_nonzero = false;
        for k in start..end {
            if m.values[k] == 0.0 {
                continue;
            }
            nnz_col += 1;
            if m.row_idx[k] == j {
                diag_nonzero = true;
            }
        }
        if nnz_col > max_col_nnz {
            max_col_nnz = nnz_col;
        }
        if nnz_col == 1 && diag_nonzero {
            diag_only += 1;
        }
    }
    (pick_scaling_strategy(m), diag_only, max_col_nnz)
}

fn read_rhs(path: &str, n: usize) -> Vec<f64> {
    let f = File::open(path).expect("open rhs");
    let mut b = Vec::with_capacity(n);
    for line in BufReader::new(f).lines() {
        let line = line.expect("rhs line");
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        b.push(s.parse::<f64>().expect("rhs parse"));
    }
    b
}

fn rel_res_2norm(csc: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = csc.n;
    let mut r = b.iter().map(|v| -v).collect::<Vec<f64>>();
    for j in 0..n {
        for p in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[p];
            let a = csc.values[p];
            r[i] += a * x[j];
            if i != j {
                r[j] += a * x[i];
            }
        }
    }
    let rn: f64 = r.iter().map(|v| v * v).sum();
    let bn: f64 = b.iter().map(|v| v * v).sum();
    if bn == 0.0 {
        0.0
    } else {
        (rn / bn).sqrt()
    }
}

fn factor_under(csc: &CscMatrix, strat: ScalingStrategy, rhs: &Option<Vec<f64>>) {
    let label = format!("{:?}", strat);
    let snode = SupernodeParams::default();
    let sym = symbolic_factorize(csc, &snode).expect("symbolic");
    let params = NumericParams {
        scaling: strat,
        ..NumericParams::default()
    };
    let mut ws = FactorWorkspace::new();
    let t0 = Instant::now();
    let res = factorize_multifrontal_parallel_with_workspace(csc, &sym, &params, &mut ws);
    let factor_us = t0.elapsed().as_micros();
    match res {
        Ok((factors, inertia)) => {
            println!("  strategy {}", label);
            println!(
                "    factor      {} us ({:.3} s)",
                factor_us,
                factor_us as f64 / 1e6
            );
            println!("    {}", factors.summary());
            println!(
                "    inertia     ({}, {}, {})",
                inertia.positive, inertia.negative, inertia.zero
            );
            println!("    scaling_info {:?}", factors.scaling_info);
            if let Some(b) = rhs {
                if b.len() == csc.n {
                    match solve_sparse_refined(csc, &factors, b) {
                        Ok(x) => {
                            println!("    rel_res     {:.3e}", rel_res_2norm(csc, &x, b));
                        }
                        Err(e) => println!("    solve FAILED: {:?}", e),
                    }
                }
            }
        }
        Err(e) => {
            println!("  strategy {}", label);
            println!("    factor FAILED after {} us: {:?}", factor_us, e);
        }
    }
}

/// Factor `csc` `repeat` times on one warm high-level `Solver` — the
/// exact analyze-once / refactor-many path POUNCE's `pounce-feral`
/// backend drives. Exposes whether the symbolic / MC64 caches engage.
fn run_warm(csc: &CscMatrix, rhs: &Option<Vec<f64>>, repeat: usize) {
    let mut s = Solver::new();
    for call in 0..repeat {
        let t = Instant::now();
        let status = s.factor(csc, None);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let fnnz = s.factors().map(|f| f.factor_nnz()).unwrap_or(0);
        let inertia = s
            .inertia()
            .map(|i| format!("({},{},{})", i.positive, i.negative, i.zero))
            .unwrap_or_else(|| "-".to_string());
        let scaling = s
            .scaling_info()
            .map(|si| format!("{si:?}"))
            .unwrap_or_else(|| "-".to_string());
        let rel = match rhs {
            Some(b) if b.len() == csc.n => match s.solve(b) {
                Ok(x) => format!("{:.2e}", rel_res_2norm(csc, &x, b)),
                Err(_) => "solve_err".to_string(),
            },
            _ => "-".to_string(),
        };
        let tag = if call == 0 { "cold" } else { "warm" };
        println!(
            "  call {call} ({tag}) factor_ms={ms:>9.1}  factor_nnz={fnnz:<9} \
             symbolic_calls={}  mc64_cache_hits={}  mc64_fallbacks={}  \
             inertia={inertia}  scaling={scaling}  rel_res={rel}  {status:?}",
            s.symbolic_call_count(),
            s.mc64_cache_hit_count(),
            s.mc64_fallback_count(),
        );
    }
    println!();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} KKT.mtx [RHS.txt]", args[0]);
        std::process::exit(2);
    }
    let mtx = read_mtx(Path::new(&args[1])).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let rhs = args.get(2).map(|p| read_rhs(p, csc.n));

    println!("matrix {}", args[1]);
    println!("  n={} stored_nnz={}", csc.n, csc.row_idx.len());
    let zeros = csc.values.iter().filter(|&&v| v == 0.0).count();
    println!("  explicit stored zeros={}", zeros);

    let (blind, b_diag, b_mcn) = route_value_blind(&csc);
    let (aware, a_diag, a_mcn) = route_value_aware(&csc);
    println!(
        "  route value-blind (pre-#47): {:?}  [diag_only={} ({:.3}), max_col_nnz={}]",
        blind,
        b_diag,
        b_diag as f64 / csc.n as f64,
        b_mcn
    );
    println!(
        "  route value-aware (post-#47): {:?}  [diag_only={} ({:.3}), max_col_nnz={}]",
        aware,
        a_diag,
        a_diag as f64 / csc.n as f64,
        a_mcn
    );
    println!();

    println!("== low-level factorize_multifrontal_parallel_with_workspace ==");
    for strat in [
        ScalingStrategy::Auto,
        ScalingStrategy::Mc64Symmetric,
        ScalingStrategy::InfNorm,
    ] {
        factor_under(&csc, strat, &rhs);
        println!();
    }

    println!("== high-level Solver, warm (analyze-once / refactor-many, POUNCE path) ==");
    run_warm(&csc, &rhs, 5);
}
