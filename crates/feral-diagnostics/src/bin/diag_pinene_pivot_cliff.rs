//! Issue #8 root cause: where does the 2x2 cascade live?
//!
//! Compares pinene_3200 iterates 0008 (fast: 44ms) vs 0009 (slow: 87s).
//! Both have identical symbolic structure (same n, same nnz, same
//! supernode partitioning), so per-supernode pivot statistics expose
//! exactly which supernode(s) blow up on 0009.
//!
//! For each iterate, prints:
//!   - global summary (n_2x2, n_delayed, inertia, nnz_L)
//!   - top supernodes by per-node n_2x2
//!   - top supernodes by per-node n_delayed_in (work pushed up from children)
//!   - top supernodes by trailing-update work proxy (nelim * nrow)
//!
//! Goal: confirm whether iterate 0009 concentrates many 2x2 pivots in
//! one supernode (likely the trailing constraint Schur block) and
//! whether children delay many pivots into it.

use std::path::Path;

use feral::numeric::factorize::{
    factorize_multifrontal_parallel_with_workspace, FactorWorkspace, NumericParams, SparseFactors,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, CscMatrix, Inertia};

struct SupernodeStats {
    idx: usize,
    first_col: usize,
    ncol: usize,
    nelim: usize,
    nrow: usize,
    n_delayed_in: usize,
    n_delayed_out: usize,
    n_2x2: usize,
    work_proxy: usize,
}

fn collect_stats(factors: &SparseFactors) -> Vec<SupernodeStats> {
    factors
        .node_factors
        .iter()
        .enumerate()
        .map(|(idx, nf)| {
            let ff = &nf.frontal_factors;
            // Count 2x2 pivots within this node by walking d_subdiag.
            let nelim = ff.nelim;
            let mut n_2x2 = 0usize;
            let mut k = 0;
            while k < nelim {
                let two_by_two = k + 1 < nelim && ff.d_subdiag[k] != 0.0;
                if two_by_two {
                    n_2x2 += 1;
                    k += 2;
                } else {
                    k += 1;
                }
            }
            // Trailing-update work proxy: per eliminated column we touch
            // ~(nrow - k) entries of the trailing block, so the total
            // is roughly nelim * (nrow - nelim/2). Use that scaled by
            // (1 + n_2x2 / nelim) to mildly inflate for 2x2-heavy nodes.
            let work_proxy = nelim.saturating_mul(nf.nrow.saturating_sub(nelim / 2));
            let n_delayed_out = ff.n_delayed;
            SupernodeStats {
                idx,
                first_col: nf.first_col,
                ncol: nf.ncol,
                nelim,
                nrow: nf.nrow,
                n_delayed_in: nf.n_delayed_in,
                n_delayed_out,
                n_2x2,
                work_proxy,
            }
        })
        .collect()
}

fn print_top(
    label: &str,
    mut stats: Vec<SupernodeStats>,
    key: fn(&SupernodeStats) -> usize,
    n: usize,
) {
    stats.sort_by_key(|s| std::cmp::Reverse(key(s)));
    println!("  Top {n} by {label}:");
    println!(
        "    {:>4}  {:>7}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>14}",
        "rank", "node", "first", "ncol", "nelim", "nrow", "ndIn", "n_2x2", "work_proxy"
    );
    for (rank, s) in stats.into_iter().take(n).enumerate() {
        println!(
            "    {:>4}  {:>7}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>14}",
            rank + 1,
            s.idx,
            s.first_col,
            s.ncol,
            s.nelim,
            s.nrow,
            s.n_delayed_in,
            s.n_2x2,
            s.work_proxy,
        );
    }
}

fn factor_and_report(label: &str, csc: &CscMatrix, oracle: Inertia) {
    let snode = SupernodeParams::default();
    let sym = symbolic_factorize(csc, &snode).expect("symbolic");
    println!("\n[{label}] n={} nnz={}", csc.n, csc.row_idx.len());
    println!("  symbolic supernodes: {}", sym.supernodes.len());

    let params = NumericParams::default();
    let mut ws = FactorWorkspace::new();
    let (factors, inertia) =
        factorize_multifrontal_parallel_with_workspace(csc, &sym, &params, &mut ws)
            .expect("factor");
    assert_eq!(inertia, oracle, "{label} inertia mismatch");
    println!("  summary: {}", factors.summary());

    let stats = collect_stats(&factors);

    // Aggregates
    let n_with_2x2 = stats.iter().filter(|s| s.n_2x2 > 0).count();
    let n_with_delay = stats.iter().filter(|s| s.n_delayed_in > 0).count();
    let max_node_2x2 = stats.iter().map(|s| s.n_2x2).max().unwrap_or(0);
    let max_node_delay = stats.iter().map(|s| s.n_delayed_in).max().unwrap_or(0);
    let total_2x2: usize = stats.iter().map(|s| s.n_2x2).sum();
    let total_delay_in: usize = stats.iter().map(|s| s.n_delayed_in).sum();
    let total_delay_out: usize = stats.iter().map(|s| s.n_delayed_out).sum();
    println!(
        "  pivot stats: total_2x2={total_2x2} max_node_2x2={max_node_2x2} \
         n_supers_with_2x2={n_with_2x2}/{total} \
         total_delay_in={total_delay_in} total_delay_out={total_delay_out} \
         max_node_delay_in={max_node_delay} n_supers_with_delay={n_with_delay}",
        total = stats.len(),
    );

    print_top(
        "n_2x2 (per supernode)",
        stats
            .iter()
            .map(|s| SupernodeStats {
                idx: s.idx,
                first_col: s.first_col,
                ncol: s.ncol,
                nelim: s.nelim,
                nrow: s.nrow,
                n_delayed_in: s.n_delayed_in,
                n_delayed_out: s.n_delayed_out,
                n_2x2: s.n_2x2,
                work_proxy: s.work_proxy,
            })
            .collect(),
        |s| s.n_2x2,
        8,
    );
    print_top(
        "n_delayed_in",
        stats
            .iter()
            .map(|s| SupernodeStats {
                idx: s.idx,
                first_col: s.first_col,
                ncol: s.ncol,
                nelim: s.nelim,
                nrow: s.nrow,
                n_delayed_in: s.n_delayed_in,
                n_delayed_out: s.n_delayed_out,
                n_2x2: s.n_2x2,
                work_proxy: s.work_proxy,
            })
            .collect(),
        |s| s.n_delayed_in,
        8,
    );
    print_top(
        "trailing-update work_proxy (nelim * (nrow - nelim/2))",
        stats,
        |s| s.work_proxy,
        8,
    );
}

fn main() -> std::io::Result<()> {
    for tag in ["pinene_3200_0008", "pinene_3200_0009"] {
        let base = format!("data/matrices/kkt-mittelmann/pinene_3200/{tag}");
        let mtx = read_mtx(Path::new(&format!("{base}.mtx"))).expect("read mtx");
        let csc = mtx.to_csc().expect("to_csc");
        let sidecar = read_sidecar(Path::new(&format!("{base}.json"))).expect("sidecar");
        let oracle = Inertia::new(
            sidecar.inertia.positive,
            sidecar.inertia.negative,
            sidecar.inertia.zero,
        );
        factor_and_report(tag, &csc, oracle);
    }
    Ok(())
}
