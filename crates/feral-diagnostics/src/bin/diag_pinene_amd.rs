//! Issue #8 follow-up: does AMD dodge the delayed-pivot cascade on
//! pinene_3200_0009 that METIS-ND falls into?
//!
//! On pinene_3200_0009 with the default (METIS-ND, picked by
//! `pick_default_method` because n>=5000 ∧ avg_deg<6), factor is
//! 88s with three ~14k-column root supernodes absorbing 118k
//! delays (see journal 22:55).
//!
//! AMD hypothesis: locally-greedy minimum-degree may produce a
//! different etree topology where delays don't all funnel to a
//! single dense root. If so, factor wallclock drops without
//! sacrificing inertia (unlike `with_static_pivoting`).
//!
//! Reports per-method: max root supernode ncol, root count
//! (post_parent == itself), total delayed_in, factor wallclock,
//! inertia vs oracle, residual after refined solve.
//!
//! Usage:
//!     cargo run --release --bin diag_pinene_amd
//!     cargo run --release --bin diag_pinene_amd -- pinene_3200_0008

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
            println!("  [{label}]  symbolic FAILED: {e:?}");
            return;
        }
    };
    let sym_s = t_sym.elapsed().as_secs_f64();

    // Etree root count and max root ncol. Roots = supernodes not
    // listed as any other supernode's child.
    let n_supers = sym.supernodes.len();
    let mut is_child = vec![false; n_supers];
    for sn in &sym.supernodes {
        for &c in &sn.children {
            is_child[c] = true;
        }
    }
    let mut root_count = 0usize;
    let mut max_root_ncol = 0usize;
    let mut max_super_ncol = 0usize;
    for (i, sn) in sym.supernodes.iter().enumerate() {
        max_super_ncol = max_super_ncol.max(sn.ncol);
        if !is_child[i] {
            root_count += 1;
            max_root_ncol = max_root_ncol.max(sn.ncol);
        }
    }

    let params = NumericParams::default();
    let mut ws = FactorWorkspace::new();
    let t0 = Instant::now();
    let res = factorize_multifrontal_parallel_with_workspace(csc, &sym, &params, &mut ws);
    let factor_s = t0.elapsed().as_secs_f64();
    let (factors, inertia) = match res {
        Ok(p) => p,
        Err(e) => {
            println!(
                "  [{label}]  sym={sym_s:.3}s  n_super={n_supers} roots={root_count} \
                 max_root_ncol={max_root_ncol} max_super_ncol={max_super_ncol}  \
                 factor FAILED after {factor_s:.3}s: {e:?}"
            );
            return;
        }
    };
    let oracle_ok = inertia == *oracle;

    let total_delay_in: usize = factors.node_factors.iter().map(|nf| nf.n_delayed_in).sum();
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

    let t0 = Instant::now();
    let solve = solve_sparse_refined(csc, &factors, rhs);
    let solve_s = t0.elapsed().as_secs_f64();
    let rel = solve.as_ref().ok().map(|x| rel_residual_2norm(csc, x, rhs));

    println!(
        "  [{label:<8}]  sym={sym_s:6.3}s  factor={factor_s:7.3}s  solve={solve_s:6.3}s  \
         n_super={n_supers}  roots={root_count}  max_root_ncol={max_root_ncol}  \
         max_super_ncol={max_super_ncol}  delay_in={total_delay_in}  n_2x2={total_2x2}  \
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

    run("AMD     ", &csc, &rhs, &oracle, OrderingMethod::Amd);
    run("MetisND ", &csc, &rhs, &oracle, OrderingMethod::MetisND);

    Ok(())
}
