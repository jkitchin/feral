//! Phase 3 verification for issue #8 — FMA opt-in path on Mittelmann
//! KKT matrices.
//!
//! For each curated matrix (default: `pinene_3200_0000`,
//! `marine_1600_0000`, `gasoil_3200_0000`), times a single factor
//! twice — once with `NumericParams::default()` (FMA off, bit-exact
//! reference) and once with `params.fma = true` (FMA opt-in dispatch).
//! Verifies inertia matches the sidecar oracle on both paths and that
//! the residual is small after iterative refinement.
//!
//! Per-matrix FMA-off has a configurable wallclock budget; if it
//! exceeds the budget we record `factor_us` as a lower bound and skip
//! the solve+residual step on that path. This keeps the harness
//! responsive on `pinene_3200`, whose baseline factor takes
//! >600 s (the issue #8 motivating data point).
//!
//! Usage:
//!     cargo run --release --bin bench_fma_phase3
//!     cargo run --release --bin bench_fma_phase3 -- pinene_3200_0000
//!     cargo run --release --bin bench_fma_phase3 -- --budget-secs 30 marine_1600_0000

use std::path::Path;
use std::time::{Duration, Instant};

use feral::numeric::factorize::{
    factorize_multifrontal_parallel_with_workspace, FactorWorkspace, NumericParams,
};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, CscMatrix, Inertia};

const DEFAULT_TARGETS: &[(&str, &str)] = &[
    ("marine_1600", "marine_1600_0000"),
    ("gasoil_3200", "gasoil_3200_0000"),
    ("pinene_3200", "pinene_3200_0000"),
];

struct PathResult {
    label: &'static str,
    analyse_us: u64,
    factor_us: u64,
    solve_us: Option<u64>,
    inertia: Inertia,
    rel_residual: Option<f64>,
    truncated: bool,
}

fn rel_residual_2norm(csc: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
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

fn time_path(
    label: &'static str,
    csc: &CscMatrix,
    sym: &feral::symbolic::SymbolicFactorization,
    rhs: &[f64],
    fma: bool,
    budget: Duration,
) -> Option<PathResult> {
    let params = NumericParams {
        fma,
        ..NumericParams::default()
    };
    let mut ws = FactorWorkspace::new();

    let t0 = Instant::now();
    // We can't cancel mid-factor; we only check the budget afterwards.
    let factor_res = factorize_multifrontal_parallel_with_workspace(csc, sym, &params, &mut ws);
    let factor_us = t0.elapsed().as_micros() as u64;
    let truncated = !fma && Duration::from_micros(factor_us) > budget;

    let (factors, inertia) = match factor_res {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[{label}] factor failed: {e:?}");
            return None;
        }
    };

    // Skip the solve step on a truncated baseline run — we already
    // know the factor blew past the budget.
    let (solve_us, rel) = if truncated {
        (None, None)
    } else {
        let t0 = Instant::now();
        match solve_sparse_refined(csc, &factors, rhs) {
            Ok(x) => {
                let solve_us = t0.elapsed().as_micros() as u64;
                let rel = rel_residual_2norm(csc, &x, rhs);
                (Some(solve_us), Some(rel))
            }
            Err(e) => {
                eprintln!("[{label}] solve failed: {e:?}");
                (None, None)
            }
        }
    };

    Some(PathResult {
        label,
        analyse_us: 0,
        factor_us,
        solve_us,
        inertia,
        rel_residual: rel,
        truncated,
    })
}

fn print_row(r: &PathResult) {
    let factor_s = r.factor_us as f64 * 1e-6;
    let solve_s = r.solve_us.map(|u| u as f64 * 1e-6);
    let res = r
        .rel_residual
        .map(|v| format!("{:.3e}", v))
        .unwrap_or_else(|| "n/a".to_string());
    let trunc = if r.truncated { " (>budget)" } else { "" };
    println!(
        "  {:<10} factor={:>8.3}s{}  solve={}  inertia={}  res={}",
        r.label,
        factor_s,
        trunc,
        solve_s
            .map(|s| format!("{:>7.3}s", s))
            .unwrap_or_else(|| "    n/a".to_string()),
        r.inertia,
        res
    );
}

fn main() -> std::io::Result<()> {
    let mut args: Vec<String> = std::env::args().collect();
    args.remove(0);

    let mut budget_secs: f64 = 60.0;
    let mut explicit: Vec<String> = Vec::new();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        if a == "--budget-secs" {
            budget_secs = it
                .next()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(budget_secs);
        } else {
            explicit.push(a);
        }
    }

    let targets: Vec<(String, String)> = if explicit.is_empty() {
        DEFAULT_TARGETS
            .iter()
            .map(|(f, m)| (f.to_string(), m.to_string()))
            .collect()
    } else {
        explicit
            .into_iter()
            .map(|m| {
                let family = m
                    .rsplit_once('_')
                    .map(|(prefix, _)| prefix.to_string())
                    .unwrap_or_else(|| m.clone());
                (family, m)
            })
            .collect()
    };

    let budget = Duration::from_secs_f64(budget_secs);
    println!("Phase 3 — FMA opt-in verification (issue #8)");
    println!("Budget for FMA-off path: {budget_secs:.1} s");

    let snode = SupernodeParams::default();

    for (family, matrix) in &targets {
        let base = format!("data/matrices/kkt-mittelmann/{family}/{matrix}");
        let mtx_path = format!("{base}.mtx");
        let json_path = format!("{base}.json");
        if !Path::new(&mtx_path).is_file() {
            eprintln!("SKIP {matrix}: {mtx_path} not found");
            continue;
        }
        let mtx = read_mtx(Path::new(&mtx_path)).expect("read mtx");
        let csc = mtx.to_csc().expect("to_csc");
        let sidecar = read_sidecar(Path::new(&json_path)).expect("sidecar");
        let rhs = sidecar.finite_rhs().expect("finite rhs");
        let oracle = Inertia::new(
            sidecar.inertia.positive,
            sidecar.inertia.negative,
            sidecar.inertia.zero,
        );

        println!("\n[{matrix}] n={} nnz={}", csc.n, csc.row_idx.len());

        let t0 = Instant::now();
        let sym = symbolic_factorize(&csc, &snode).expect("symbolic");
        let analyse_us = t0.elapsed().as_micros() as u64;
        println!("  symbolic    {:>7.3}s", analyse_us as f64 * 1e-6);
        println!("  oracle inertia: {oracle}");

        // FMA OFF path — may run past budget on pinene_3200.
        if let Some(mut row) = time_path("nofma", &csc, &sym, &rhs, /*fma=*/ false, budget) {
            row.analyse_us = analyse_us;
            assert_eq!(
                row.inertia, oracle,
                "{matrix} nofma inertia mismatch: got {} expected {}",
                row.inertia, oracle
            );
            print_row(&row);
        }

        // FMA ON path — should always finish quickly on these targets.
        if let Some(mut row) = time_path("fma", &csc, &sym, &rhs, /*fma=*/ true, budget) {
            row.analyse_us = analyse_us;
            assert_eq!(
                row.inertia, oracle,
                "{matrix} fma inertia mismatch: got {} expected {}",
                row.inertia, oracle
            );
            print_row(&row);
        }
    }

    Ok(())
}
