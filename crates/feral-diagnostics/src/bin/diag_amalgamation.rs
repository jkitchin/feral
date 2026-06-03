//! Phase 2.11 diagnostic: characterize the supernode-tree shape on the
//! tiny-IPM tail to determine why feral's amalgamation
//! (`src/symbolic/supernode.rs`) leaves so many ≤8-wide supernodes
//! despite `nemin=32` (default).
//!
//! Prints, per matrix:
//!   * n_supernodes_fundamental — supernodes before amalgamation
//!   * n_supernodes_final       — supernodes after amalgamation
//!   * merges_done              — fundamental − final
//!   * snode_tree.multi_child   — internal supernodes with ≥2 children
//!     (bushy-tree signature: only one of N children can be adjacent
//!     to the parent in postorder, so the other N-1 sibling merges
//!     are blocked by the adjacency check at supernode.rs:204-236)
//!   * leaf_snodes              — supernodes with 0 children
//!   * snode width histogram (≤8, 9-16, 17-32, 33-64, >64)
//!
//! Hypothesis under test: if multi_child ≫ 0, the gap is a
//! bushy-tree-blocked-sibling-merge problem and the fix is SSIDS-style
//! column renumbering. If multi_child ≈ 0, the gap is somewhere else
//! and we need a different intervention.

use std::path::Path;

use feral::sparse::csc::CscPattern;
use feral::symbolic::{
    symbolic_factorize_with_method, AmalgamationStrategy, OrderingMethod, SmallLeafParams,
    Supernode, SupernodeParams,
};
use feral::{read_mtx, CscMatrix};

/// Replicates `find_small_leaf_groups` while counting the
/// per-event reasons a group flushes. Mirror the production
/// algorithm exactly so the counters reflect the actual behavior.
struct SmallLeafGroupingStats {
    n_qualifying: usize,
    n_close_arena: usize,
    n_close_nonqual: usize,
    n_close_end: usize,
    n_groups: usize,
    nrow_actual_p50: usize,
    nrow_actual_p95: usize,
    nrow_actual_max: usize,
}

fn analyze_small_leaf_grouping(
    supernodes: &[Supernode],
    pattern: &CscPattern,
    params: &SmallLeafParams,
) -> SmallLeafGroupingStats {
    let mut n_qualifying = 0usize;
    let mut n_close_arena = 0usize;
    let mut n_close_nonqual = 0usize;
    let mut n_close_end = 0usize;
    let mut n_groups = 0usize;
    let mut nrow_actuals: Vec<usize> = Vec::new();

    let mut seen: Vec<bool> = vec![false; pattern.n];
    let mut trailing: Vec<usize> = Vec::new();
    let mut current_arena: Option<usize> = None;

    for snode in supernodes {
        let qualifies = snode.children.is_empty()
            && snode.ncol <= params.ncol_max
            && snode.nrow <= params.nrow_max
            && snode.nrow > 0;

        if !qualifies {
            if current_arena.is_some() {
                n_close_nonqual += 1;
                n_groups += 1;
                current_arena = None;
            }
            continue;
        }

        n_qualifying += 1;

        // Compute actual rows for this leaf.
        let first_col = snode.first_col;
        let ncol = snode.ncol;
        for s in seen.iter_mut().skip(first_col).take(ncol) {
            *s = true;
        }
        trailing.clear();
        for j in first_col..first_col + ncol {
            for k in pattern.col_ptr[j]..pattern.col_ptr[j + 1] {
                let r = pattern.row_idx[k];
                if !seen[r] {
                    seen[r] = true;
                    trailing.push(r);
                }
            }
        }
        let nrow_actual = ncol + trailing.len();
        nrow_actuals.push(nrow_actual);

        // Restore seen.
        for s in seen.iter_mut().skip(first_col).take(ncol) {
            *s = false;
        }
        for &r in trailing.iter() {
            seen[r] = false;
        }

        let leaf_size = nrow_actual * nrow_actual;
        let must_close = match current_arena {
            Some(a) => a + leaf_size > params.arena_budget,
            None => false,
        };
        if must_close {
            n_close_arena += 1;
            n_groups += 1;
            current_arena = None;
        }

        match current_arena.as_mut() {
            Some(a) => *a += leaf_size,
            None => current_arena = Some(leaf_size),
        }
    }
    if current_arena.is_some() {
        n_close_end += 1;
        n_groups += 1;
    }

    nrow_actuals.sort_unstable();
    let nrow_actual_p50 = nrow_actuals
        .get(nrow_actuals.len() / 2)
        .copied()
        .unwrap_or(0);
    let nrow_actual_p95 = nrow_actuals
        .get((nrow_actuals.len() * 95) / 100)
        .copied()
        .unwrap_or(0);
    let nrow_actual_max = nrow_actuals.iter().copied().max().unwrap_or(0);

    SmallLeafGroupingStats {
        n_qualifying,
        n_close_arena,
        n_close_nonqual,
        n_close_end,
        n_groups,
        nrow_actual_p50,
        nrow_actual_p95,
        nrow_actual_max,
    }
}

const MATRICES: &[(&str, &str)] = &[
    ("ACOPR30_0067", "data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx"),
    (
        "CRESC100_0000",
        "data/matrices/kkt/CRESC100/CRESC100_0000.mtx",
    ),
    ("LAKES_0000", "data/matrices/kkt/LAKES/LAKES_0000.mtx"),
    ("NELSON_0000", "data/matrices/kkt/NELSON/NELSON_0000.mtx"),
    ("SWOPF_0000", "data/matrices/kkt/SWOPF/SWOPF_0000.mtx"),
];

fn load_csc(path: &str) -> Option<CscMatrix> {
    if !Path::new(path).exists() {
        eprintln!("SKIP: {} not present", path);
        return None;
    }
    let mtx = match read_mtx(Path::new(path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("read_mtx({}) failed: {}", path, e);
            return None;
        }
    };
    match mtx.to_csc() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("to_csc({}) failed: {}", path, e);
            None
        }
    }
}

fn analyze(label: &str, csc: &CscMatrix) {
    println!("=== {} ===", label);
    for strategy in [
        AmalgamationStrategy::Adjacency,
        AmalgamationStrategy::Renumber,
    ] {
        analyze_one(label, csc, strategy);
    }
}

fn analyze_one(label: &str, csc: &CscMatrix, strategy: AmalgamationStrategy) {
    let params = SupernodeParams {
        amalgamation_strategy: strategy,
        ..Default::default()
    };
    let sym = match symbolic_factorize_with_method(csc, &params, OrderingMethod::Amd) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: symbolic_factorize failed: {}", label, e);
            return;
        }
    };
    let snodes = &sym.supernodes;
    let n_final = snodes.len();

    // Width histogram (post-amalgamation).
    let mut buckets = [0usize; 5]; // ≤8, 9-16, 17-32, 33-64, >64
    for s in snodes {
        let w = s.ncol;
        let b = match w {
            0..=8 => 0,
            9..=16 => 1,
            17..=32 => 2,
            33..=64 => 3,
            _ => 4,
        };
        buckets[b] += 1;
    }

    // Tree shape: count children per supernode (children live on the
    // parent, so a supernode's child count is just `s.children.len()`).
    let mut child_counts = vec![0usize; n_final];
    for (i, s) in snodes.iter().enumerate() {
        child_counts[i] = s.children.len();
        for &c in &s.children {
            debug_assert!(c < i, "{}: child {} >= parent {}", label, c, i);
        }
    }

    let multi_child = child_counts.iter().filter(|&&c| c >= 2).count();
    let leaf_snodes = child_counts.iter().filter(|&&c| c == 0).count();
    let max_children = child_counts.iter().copied().max().unwrap_or(0);

    println!("--- {} strategy={:?} (n={}) ---", label, strategy, csc.n);
    println!("  n_supernodes_final  = {}", n_final);
    println!(
        "  width buckets       : ≤8={}, 9-16={}, 17-32={}, 33-64={}, >64={}",
        buckets[0], buckets[1], buckets[2], buckets[3], buckets[4]
    );
    println!("  leaf_supernodes     = {}", leaf_snodes);
    println!(
        "  multi-child snodes  = {} ({}% of internal nodes)",
        multi_child,
        if n_final > leaf_snodes {
            (multi_child * 100) / (n_final - leaf_snodes)
        } else {
            0
        }
    );
    println!("  max children       = {}", max_children);

    let blocked_sibling_estimate: usize = child_counts.iter().map(|&c| c.saturating_sub(1)).sum();
    println!(
        "  est. sibling-merges blocked by adjacency = {}",
        blocked_sibling_estimate
    );

    // Small-leaf grouping (Phase 2.9): does it already cover these
    // bushy tails?
    let n_groups = sym.small_leaf_groups.len();
    let n_grouped: usize = sym.small_leaf_groups.iter().map(|g| g.members.len()).sum();
    let n_ungrouped = sym.snode_group.iter().filter(|g| g.is_none()).count();
    println!(
        "  small_leaf_groups   : {} groups covering {} of {} supernodes ({} ungrouped)",
        n_groups, n_grouped, n_final, n_ungrouped
    );

    // Step A: identify the dominant small_leaf-group breaker.
    // Re-permute the pattern the way symbolic_factorize does.
    let pat_full = csc.symmetric_pattern();
    let permuted = feral::ordering::amd::permute_pattern(&pat_full, &sym.perm);
    let stats = analyze_small_leaf_grouping(snodes, &permuted, &SmallLeafParams::default());
    println!(
        "  qualifying leaves   : {} of {} leaves",
        stats.n_qualifying, leaf_snodes
    );
    println!(
        "  group closes        : arena={}, nonqual={}, end={} (total groups={})",
        stats.n_close_arena, stats.n_close_nonqual, stats.n_close_end, stats.n_groups
    );
    println!(
        "  nrow_actual         : p50={}, p95={}, max={}",
        stats.nrow_actual_p50, stats.nrow_actual_p95, stats.nrow_actual_max
    );
    println!();
}

fn main() {
    let mut any = false;
    for (label, path) in MATRICES {
        if let Some(csc) = load_csc(path) {
            analyze(label, &csc);
            any = true;
        }
    }
    if !any {
        eprintln!("No matrices ran. Place corpus under data/matrices/kkt/.");
    }
}
