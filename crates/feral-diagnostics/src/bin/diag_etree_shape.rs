//! Phase 2.13a — etree shape statistics probe.
//!
//! Computes O(n) shape statistics on the post-ordered etree for each
//! known-answer matrix to discover which statistic separates
//! Renumber-wins (bushy IPM-KKT trees) from Renumber-loses (path /
//! near-path trees). The output decides the dispatch predicate for
//! `AmalgamationStrategy::Auto`.
//!
//! Stats:
//!   n_internal       = |{ nodes with at least one child }|
//!   n_multi_child    = |{ nodes with >= 2 children }|
//!   max_children     = max |children(v)|
//!   mean_children    = (n - n_leaves) / n_internal
//!   multi_child_frac = n_multi_child / n_internal
//!
//! See `dev/research/phase-2.13a-amalgamation-auto.md`.

use std::path::Path;

use feral::ordering::elimination_tree::EliminationTree;
use feral::symbolic::{pick_amalgamation_strategy, symbolic_factorize, SupernodeParams};
use feral::{read_mtx, CscMatrix};

const MATRICES: &[(&str, &str, &str)] = &[
    // (label, mtx path, known best strategy)
    (
        "ACOPR30_0067",
        "data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx",
        "Renumber",
    ),
    (
        "CRESC100_0000",
        "data/matrices/kkt/CRESC100/CRESC100_0000.mtx",
        "Renumber",
    ),
    (
        "LAKES_0000",
        "data/matrices/kkt/LAKES/LAKES_0000.mtx",
        "Renumber",
    ),
    (
        "NELSON_0000",
        "data/matrices/kkt/NELSON/NELSON_0000.mtx",
        "Renumber",
    ),
    (
        "SWOPF_0000",
        "data/matrices/kkt/SWOPF/SWOPF_0000.mtx",
        "Renumber",
    ),
    (
        "MUONSINE_0000",
        "data/matrices/kkt/MUONSINE/MUONSINE_0000.mtx",
        "Adjacency",
    ),
    (
        // KIRBY2_0007 is bushy (multi_child_frac ≈ 0.97). Renumber
        // is slightly better than Adjacency but the dominant cost
        // is symbolic-phase AMD (Phase 2.13b territory), not
        // amalgamation. Expected dispatch: Renumber.
        "KIRBY2_0007",
        "data/matrices/kkt/KIRBY2/KIRBY2_0007.mtx",
        "Renumber",
    ),
];

#[derive(Debug)]
struct Shape {
    n: usize,
    n_leaves: usize,
    n_internal: usize,
    n_multi_child: usize,
    max_children: usize,
    mean_children: f64,
    multi_child_frac: f64,
}

fn shape_of(etree: &EliminationTree) -> Shape {
    let n = etree.n;
    let mut child_count = vec![0usize; n];
    for &p in &etree.parent {
        if let Some(par) = p {
            child_count[par] += 1;
        }
    }
    let n_leaves = child_count.iter().filter(|&&c| c == 0).count();
    let n_internal = n - n_leaves;
    let n_multi_child = child_count.iter().filter(|&&c| c >= 2).count();
    let max_children = child_count.iter().copied().max().unwrap_or(0);
    let mean_children = if n_internal > 0 {
        (n - n_leaves) as f64 / n_internal as f64
    } else {
        0.0
    };
    let multi_child_frac = if n_internal > 0 {
        n_multi_child as f64 / n_internal as f64
    } else {
        0.0
    };
    Shape {
        n,
        n_leaves,
        n_internal,
        n_multi_child,
        max_children,
        mean_children,
        multi_child_frac,
    }
}

fn load_csc(path: &str) -> Option<CscMatrix> {
    if !Path::new(path).exists() {
        eprintln!("SKIP missing: {}", path);
        return None;
    }
    let mtx = read_mtx(Path::new(path)).ok()?;
    mtx.to_csc().ok()
}

fn main() {
    println!(
        "{:<16} {:>5} {:>8} {:>10} {:>10} {:>13} {:>17} {:>10} {:>10} {:>5}",
        "matrix",
        "n",
        "leaves",
        "internal",
        "multichld",
        "max_children",
        "multi_child_frac",
        "best",
        "auto",
        "ok?",
    );
    for &(label, path, best) in MATRICES {
        let Some(csc) = load_csc(path) else { continue };
        let sym = match symbolic_factorize(&csc, &SupernodeParams::default()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("FAIL {}: {}", label, e);
                continue;
            }
        };
        let s = shape_of(&sym.etree);
        let auto_pick = pick_amalgamation_strategy(&sym.etree);
        let auto_str = format!("{:?}", auto_pick);
        let ok = if auto_str == best { "yes" } else { "NO" };
        println!(
            "{:<16} {:>5} {:>8} {:>10} {:>10} {:>13} {:>17.4} {:>10} {:>10} {:>5}",
            label,
            s.n,
            s.n_leaves,
            s.n_internal,
            s.n_multi_child,
            s.max_children,
            s.multi_child_frac,
            best,
            auto_str,
            ok,
        );
        let _ = s.mean_children;
    }
}
