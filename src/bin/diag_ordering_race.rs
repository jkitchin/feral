//! Issue #8: AutoRace empirical validation.
//!
//! Symbolic factorization runs once per problem (sparsity pattern is
//! fixed across IPM iterations), so racing several orderings at
//! symbolic time costs ~50–500 ms one-shot for a 4-way race —
//! amortized over many numeric factors of the same pattern. This
//! binary reports per-candidate `factor_nnz_estimate` and
//! symbolic-wall, then runs `OrderingMethod::AutoRace` and reports
//! which method it resolved to plus the numeric-factor cost.
//!
//! Usage:
//!     cargo run --release --bin diag_ordering_race
//!     cargo run --release --bin diag_ordering_race -- pinene_3200_0008
//!     cargo run --release --bin diag_ordering_race -- chainwoo_3000_0000

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{
    factorize_multifrontal_parallel_with_workspace, FactorWorkspace, NumericParams,
};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, read_sidecar, CscMatrix, Inertia};

fn rel_residual_2norm(csc: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = csc.n;
    let mut r: Vec<f64> = b.iter().map(|v| -v).collect();
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

fn run(label: &str, csc: &CscMatrix, rhs: &[f64], oracle: &Inertia, method: OrderingMethod) {
    let snode = SupernodeParams::default();
    let t_sym = Instant::now();
    let sym = match symbolic_factorize_with_method(csc, &snode, method) {
        Ok(s) => s,
        Err(e) => {
            println!("  [{label:<10}]  symbolic FAILED: {e:?}");
            return;
        }
    };
    let sym_s = t_sym.elapsed().as_secs_f64();
    let nnz_l = sym.factor_nnz_estimate;
    let resolved = sym.resolved_method;

    let params = NumericParams::default();
    let mut ws = FactorWorkspace::new();
    let t0 = Instant::now();
    let res = factorize_multifrontal_parallel_with_workspace(csc, &sym, &params, &mut ws);
    let factor_s = t0.elapsed().as_secs_f64();
    let (factors, inertia) = match res {
        Ok(p) => p,
        Err(e) => {
            println!(
                "  [{label:<10}]  sym={sym_s:6.3}s  resolved={resolved:?}  nnz_L={nnz_l}  \
                 factor FAILED after {factor_s:.3}s: {e:?}"
            );
            return;
        }
    };
    let oracle_ok = inertia == *oracle;

    let t0 = Instant::now();
    let solve = solve_sparse_refined(csc, &factors, rhs);
    let solve_s = t0.elapsed().as_secs_f64();
    let rel = solve.as_ref().ok().map(|x| rel_residual_2norm(csc, x, rhs));

    println!(
        "  [{label:<10}]  sym={sym_s:6.3}s  factor={factor_s:7.3}s  solve={solve_s:6.3}s  \
         resolved={resolved:?}  nnz_L={nnz_l}  \
         inertia={inertia} (ok={oracle_ok})  rel_res={}",
        rel.map(|v| format!("{:.3e}", v))
            .unwrap_or_else(|| "n/a".to_string())
    );
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tag = args
        .first()
        .cloned()
        .unwrap_or_else(|| "pinene_3200_0009".to_string());
    let family = tag
        .rsplit_once('_')
        .map(|(p, _)| p.to_string())
        .unwrap_or_else(|| tag.clone());
    let base = format!("data/matrices/kkt-mittelmann/{family}/{tag}");
    let mtx_path = format!("{base}.mtx");
    let json_path = format!("{base}.json");

    let mtx = read_mtx(Path::new(&mtx_path)).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let sidecar = read_sidecar(Path::new(&json_path)).expect("sidecar");
    let rhs = sidecar.finite_rhs().expect("finite rhs");
    let oracle = Inertia::new(
        sidecar.inertia.positive,
        sidecar.inertia.negative,
        sidecar.inertia.zero,
    );

    println!(
        "[{tag}] n={} nnz={}  avg_deg={:.2}  oracle={oracle}",
        csc.n,
        csc.row_idx.len(),
        csc.row_idx.len() as f64 / csc.n as f64
    );

    println!("\n-- individual candidates --");
    run("AMD", &csc, &rhs, &oracle, OrderingMethod::Amd);
    run("MetisND", &csc, &rhs, &oracle, OrderingMethod::MetisND);
    run("ScotchND", &csc, &rhs, &oracle, OrderingMethod::ScotchND);
    run("KahipND", &csc, &rhs, &oracle, OrderingMethod::KahipND);

    println!("\n-- dispatcher modes --");
    run("Auto", &csc, &rhs, &oracle, OrderingMethod::Auto);
    run("AutoRace", &csc, &rhs, &oracle, OrderingMethod::AutoRace);

    Ok(())
}
