//! Issue #43 verification: static-pivoting fallback on pinene_3200_0009.
//!
//! Compares the production path (delayed pivoting enabled) against
//! `NumericParams.allow_delayed_pivots = false` on the iterate that
//! takes 87 s with the default settings. Reports factor wallclock,
//! inertia (vs sidecar oracle), and rel residual after refined solve.
//!
//! Usage:
//!     cargo run --release --bin diag_pinene_static_pivot
//!     cargo run --release --bin diag_pinene_static_pivot -- pinene_3200_0008

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{
    factorize_multifrontal_parallel_with_workspace, FactorWorkspace, NumericParams,
};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};

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

enum Mode {
    /// Default — delayed pivoting on every non-root supernode.
    Default,
    /// Force-accept everywhere (issue #43 naive form).
    StaticAll,
    /// Adaptive: force-accept only at non-root supernodes whose
    /// front is at least `f` delayed columns from below.
    CascadeBreak(f64),
}

fn run(label: &str, csc: &CscMatrix, rhs: &[f64], oracle: &Inertia, mode: Mode) {
    let snode = SupernodeParams::default();
    let sym = symbolic_factorize(csc, &snode).expect("symbolic");

    let base = NumericParams::default();
    let bk = match mode {
        Mode::StaticAll | Mode::CascadeBreak(_) => BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            ..base.bk.clone()
        },
        Mode::Default => base.bk.clone(),
    };
    let allow_delayed = !matches!(mode, Mode::StaticAll);
    let cascade = match mode {
        Mode::CascadeBreak(r) => Some(r),
        _ => None,
    };
    let params = NumericParams {
        bk,
        allow_delayed_pivots: allow_delayed,
        cascade_break_ratio: cascade,
        ..base
    };
    let mut ws = FactorWorkspace::new();

    let t0 = Instant::now();
    let (factors, inertia) =
        factorize_multifrontal_parallel_with_workspace(csc, &sym, &params, &mut ws)
            .expect("factor");
    let factor_s = t0.elapsed().as_secs_f64();

    let oracle_ok = inertia == *oracle;

    let t0 = Instant::now();
    let solve = solve_sparse_refined(csc, &factors, rhs);
    let solve_s = t0.elapsed().as_secs_f64();
    let rel = solve.as_ref().ok().map(|x| rel_residual_2norm(csc, x, rhs));

    let total_2x2: usize = factors
        .node_factors
        .iter()
        .map(|nf| {
            let ff = &nf.frontal_factors;
            let mut k = 0usize;
            let mut n = 0usize;
            while k < ff.nelim {
                if k + 1 < ff.nelim && ff.d_subdiag[k] != 0.0 {
                    n += 1;
                    k += 2;
                } else {
                    k += 1;
                }
            }
            n
        })
        .sum();
    let total_delay_in: usize = factors.node_factors.iter().map(|nf| nf.n_delayed_in).sum();

    println!(
        "  [{label}]  factor={factor_s:7.3}s  solve={solve_s:6.3}s  \
         inertia={inertia} (ok={oracle_ok})  total_2x2={total_2x2}  \
         total_delay_in={total_delay_in}  rel_res={}",
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
        "[{tag}] n={} nnz={}  oracle={oracle}",
        csc.n,
        csc.row_idx.len()
    );

    run("static-all       ", &csc, &rhs, &oracle, Mode::StaticAll);
    run(
        "cascade-break 0.50",
        &csc,
        &rhs,
        &oracle,
        Mode::CascadeBreak(0.5),
    );
    run(
        "cascade-break 0.25",
        &csc,
        &rhs,
        &oracle,
        Mode::CascadeBreak(0.25),
    );
    // Default (delayed pivoting enabled) — may take ~87s on 0009.
    run("default            ", &csc, &rhs, &oracle, Mode::Default);

    Ok(())
}
