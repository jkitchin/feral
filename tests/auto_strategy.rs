//! Phase 2.13a — `AmalgamationStrategy::Auto` dispatch tests.
//!
//! Verifies the cheap O(n) etree shape predicate routes:
//!   * pure path etree → `Adjacency` (Renumber over-merging case)
//!   * bushy etree → `Renumber` (IPM-KKT amalgamation case)
//!   * empty / leaf-only forest → `Adjacency`
//!
//! See `dev/research/phase-2.13a-amalgamation-auto.md`.

#![allow(clippy::assertions_on_constants, clippy::needless_range_loop)]
use feral::ordering::elimination_tree::EliminationTree;
use feral::symbolic::{
    pick_amalgamation_strategy, AmalgamationStrategy, AUTO_MULTI_CHILD_FRAC_THRESHOLD,
};

/// Build a path: 0 -> 1 -> 2 -> ... -> n-1 (root). Every internal
/// node has exactly one child; multi_child_frac = 0.
fn path_etree(n: usize) -> EliminationTree {
    let mut parent: Vec<Option<usize>> = vec![None; n];
    for i in 0..n.saturating_sub(1) {
        parent[i] = Some(i + 1);
    }
    EliminationTree { parent, n }
}

/// Build a complete binary tree on n nodes (last node is the root).
/// All internal nodes have 2 children, so multi_child_frac = 1.0.
/// Indexing: child(i) = 2*i+1, 2*i+2; root = 0 in tree-order, but
/// our etree convention is parent[j] > j so we relabel to put the
/// root last and children before.
fn binary_tree_etree(depth: usize) -> EliminationTree {
    // Total nodes: 2^depth + 2^(depth-1) + ... + 1 = 2^(depth+1) - 1.
    let n = (1usize << (depth + 1)) - 1;
    // Build in tree-order then relabel by depth-first postorder so
    // parent[j] > j.
    // Simpler: number nodes left-to-right, depth-first; assign
    // post-order indices.
    let parent_full = vec![None::<usize>; n];
    // Tree-order: root=0, children of i = 2*i+1, 2*i+2.
    // Build a postorder traversal -> assigns new label per visit.
    let mut new_label = vec![0usize; n];
    let mut counter = 0usize;
    fn visit(i: usize, n: usize, new_label: &mut [usize], counter: &mut usize) {
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        if l < n {
            visit(l, n, new_label, counter);
        }
        if r < n {
            visit(r, n, new_label, counter);
        }
        new_label[i] = *counter;
        *counter += 1;
    }
    visit(0, n, &mut new_label, &mut counter);

    let mut parent: Vec<Option<usize>> = vec![None; n];
    // For each tree-order node i with children l, r, set
    // parent[new_label[l/r]] = Some(new_label[i]).
    for i in 0..n {
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        if l < n {
            parent[new_label[l]] = Some(new_label[i]);
        }
        if r < n {
            parent[new_label[r]] = Some(new_label[i]);
        }
    }
    let _ = parent_full;
    EliminationTree { parent, n }
}

#[test]
fn path_dispatches_to_adjacency() {
    let etree = path_etree(100);
    assert_eq!(
        pick_amalgamation_strategy(&etree),
        AmalgamationStrategy::Adjacency,
        "pure path must dispatch Adjacency"
    );
}

#[test]
fn complete_binary_tree_dispatches_to_renumber() {
    let etree = binary_tree_etree(5); // 2^6-1 = 63 nodes
    assert_eq!(
        pick_amalgamation_strategy(&etree),
        AmalgamationStrategy::Renumber,
        "complete binary tree must dispatch Renumber"
    );
}

#[test]
fn empty_etree_dispatches_to_adjacency() {
    let etree = EliminationTree {
        parent: Vec::new(),
        n: 0,
    };
    assert_eq!(
        pick_amalgamation_strategy(&etree),
        AmalgamationStrategy::Adjacency,
    );
}

#[test]
fn leaf_only_forest_dispatches_to_adjacency() {
    let etree = EliminationTree {
        parent: vec![None; 10],
        n: 10,
    };
    assert_eq!(
        pick_amalgamation_strategy(&etree),
        AmalgamationStrategy::Adjacency,
        "no internal nodes means no merging opportunities; fall back"
    );
}

#[test]
fn auto_default_resolves_under_threshold() {
    // Tiny path of 5 nodes; multi_child_frac = 0.0 < threshold.
    let etree = path_etree(5);
    let strat = pick_amalgamation_strategy(&etree);
    assert_eq!(strat, AmalgamationStrategy::Adjacency);
    assert!(0.0 < AUTO_MULTI_CHILD_FRAC_THRESHOLD);
}

#[test]
fn near_path_with_one_branch_still_adjacency() {
    // 100-node path with a single 2-child junction: multi_child_frac
    // = 1/99 ≈ 0.010, below the 0.05 threshold.
    let n = 100;
    // Add one extra leaf branching into node 50: build an n+1 node
    // version with a single extra child of node 50.
    let n2 = n + 1;
    let mut parent2 = vec![None::<usize>; n2];
    for i in 0..n - 1 {
        parent2[i] = Some(i + 1);
    }
    parent2[n] = Some(50); // extra leaf hanging off node 50
    let etree = EliminationTree {
        parent: parent2,
        n: n2,
    };
    // child counts: node 50 has 2 children (49 and the extra),
    // every other internal has 1. internal count = 99 + 1 (node 50
    // is still internal) → 99. multi_child = 1.
    // multi_child_frac ~ 1/99 ≈ 0.010 < 0.05 → Adjacency.
    assert_eq!(
        pick_amalgamation_strategy(&etree),
        AmalgamationStrategy::Adjacency,
    );
}

#[test]
fn fan_at_root_dispatches_to_renumber() {
    // n=20, all nodes 0..18 attach directly to node 19. Single
    // multi-child internal (the root), with 19 children.
    // Internals = 1 (just the root). multi_child = 1.
    // multi_child_frac = 1.0 ≥ 0.05 → Renumber.
    let n = 20;
    let mut parent = vec![None::<usize>; n];
    for i in 0..n - 1 {
        parent[i] = Some(n - 1);
    }
    let etree = EliminationTree { parent, n };
    assert_eq!(
        pick_amalgamation_strategy(&etree),
        AmalgamationStrategy::Renumber,
    );
}
