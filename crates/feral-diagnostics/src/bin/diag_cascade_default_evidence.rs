//! Evidence that the auto-armed cascade-break default
//! (`cascade_break_ratio = Some(0.5)`, `cascade_break_eps = Some(1e-10)`)
//! fixes the target matrix without breaking a regular one.
//!
//! Runs four configurations on each matrix:
//!   - `off`        : both fields = None (legacy)
//!   - `default`    : NumericParams::default() (new auto-armed)
//!
//! and reports factor wallclock, inertia (vs sidecar oracle),
//! residual after refined solve, and total delay-in (signal that
//! cascade-break absorbed delays vs propagated them).
//!
//! Matrices:
//!   - pinene_3200_0009  → target (cascade pathology, 88 s under legacy)
//!   - pinene_3200_0000  → healthy sibling from same family
//!
//! Usage:
//!     cargo run --release --bin diag_cascade_default_evidence

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{
    factorize_multifrontal_parallel_with_workspace, FactorWorkspace, NumericParams,
};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
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

fn run(label: &str, csc: &CscMatrix, rhs: &[f64], oracle: &Inertia, params: NumericParams) {
    let snode = SupernodeParams::default();
    let sym = match symbolic_factorize(csc, &snode) {
        Ok(s) => s,
        Err(e) => {
            println!("  [{label:<10}]  symbolic FAILED: {e:?}");
            return;
        }
    };

    let mut ws = FactorWorkspace::new();
    let t0 = Instant::now();
    let res = factorize_multifrontal_parallel_with_workspace(csc, &sym, &params, &mut ws);
    let factor_s = t0.elapsed().as_secs_f64();
    let (factors, inertia) = match res {
        Ok(p) => p,
        Err(e) => {
            println!("  [{label:<10}]  factor FAILED after {factor_s:.3}s: {e:?}");
            return;
        }
    };
    let oracle_ok = inertia == *oracle;

    let total_delay_in: usize = factors.node_factors.iter().map(|nf| nf.n_delayed_in).sum();

    let t0 = Instant::now();
    let solve = solve_sparse_refined(csc, &factors, rhs);
    let solve_s = t0.elapsed().as_secs_f64();
    let rel = solve.as_ref().ok().map(|x| rel_residual_2norm(csc, x, rhs));

    println!(
        "  [{label:<10}]  factor={factor_s:7.3}s  solve={solve_s:6.3}s  \
         delay_in={total_delay_in:>6}  inertia={inertia} (ok={oracle_ok})  rel_res={}",
        rel.map(|v| format!("{:.3e}", v))
            .unwrap_or_else(|| "n/a".to_string())
    );
}

fn run_matrix(tag: &str) {
    let family = tag
        .rsplit_once('_')
        .map(|(p, _)| p.to_string())
        .unwrap_or_else(|| tag.to_string());
    let base = format!("data/matrices/kkt-mittelmann/{family}/{tag}");
    let mtx_path = format!("{base}.mtx");
    let json_path = format!("{base}.json");

    let mtx = match read_mtx(Path::new(&mtx_path)) {
        Ok(m) => m,
        Err(e) => {
            println!("[{tag}] read_mtx FAILED: {e:?}");
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            println!("[{tag}] to_csc FAILED: {e:?}");
            return;
        }
    };
    let sidecar = match read_sidecar(Path::new(&json_path)) {
        Ok(s) => s,
        Err(e) => {
            println!("[{tag}] sidecar FAILED: {e:?}");
            return;
        }
    };
    let rhs = match sidecar.finite_rhs() {
        Some(r) => r,
        None => {
            println!("[{tag}] no finite rhs");
            return;
        }
    };
    let oracle = Inertia::new(
        sidecar.inertia.positive,
        sidecar.inertia.negative,
        sidecar.inertia.zero,
    );

    println!(
        "\n[{tag}] n={} nnz={}  avg_deg={:.2}  oracle={oracle}",
        csc.n,
        csc.row_idx.len(),
        csc.row_idx.len() as f64 / csc.n as f64
    );

    // Legacy: both knobs off.
    let off = NumericParams {
        cascade_break_ratio: None,
        cascade_break_eps: None,
        ..NumericParams::default()
    };
    run("off       ", &csc, &rhs, &oracle, off);
    // New auto-armed default.
    run("default   ", &csc, &rhs, &oracle, NumericParams::default());
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        for tag in &args {
            run_matrix(tag);
        }
        return Ok(());
    }
    // Issue #8 broad sweep: pinene_3200 full iterate history +
    // marine_1600 full iterate history. Both were called out in the
    // original issue body as multi-hundred-second factor outliers.
    for i in 0..10 {
        run_matrix(&format!("pinene_3200_{:04}", i));
    }
    for i in 0..18 {
        run_matrix(&format!("marine_1600_{:04}", i));
    }
    Ok(())
}
