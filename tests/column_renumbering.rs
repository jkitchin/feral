//! Phase 2.12: column-renumbering amalgamation strategy tests.
//!
//! Each test constructs a structurally simple matrix where the
//! current `Adjacency` strategy must under-merge (because the
//! postorder ordering blocks sibling merges), and asserts that the
//! `Renumber` strategy produces the SSIDS-correct number of
//! supernodes.
//!
//! Plan: `dev/plans/phase-2.12-column-renumbering.md`.
//! Research: `dev/research/phase-2.12-column-renumbering.md`.

use feral::symbolic::{
    symbolic_factorize_with_method, AmalgamationStrategy, OrderingMethod, SupernodeParams,
};
use feral::CscMatrix;

/// Arrow matrix: variables 0..n-2 are coupled only to variable n-1
/// (the "tip" of the arrow). With nemin=32, the SSIDS size rule
/// admits all sibling-merges into the parent — but the adjacency
/// check at supernode.rs blocks every merge except the one whose
/// last column equals parent_first - 1.
///
/// Construction: lower-triangle entries are `(j, j)` and `(n-1, j)`
/// for each j in 0..n-1, plus `(n-1, n-1)`.
fn arrow_matrix(n: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n - 1 {
        rows.push(j);
        cols.push(j);
        vals.push(2.0);
        rows.push(n - 1);
        cols.push(j);
        vals.push(-1.0);
    }
    rows.push(n - 1);
    cols.push(n - 1);
    vals.push((n - 1) as f64 + 0.5);
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

/// A bushy fan: same as arrow_matrix but verifies the fundamental
/// supernode/amalgamation count is collapsed all the way to 1
/// supernode. Distinct from `arrow_matrix_collapses` because the
/// arrow matrix is also a valid fan.
fn bushy_fan(n: usize) -> CscMatrix {
    arrow_matrix(n)
}

/// Tridiagonal: parent of j-1 is j, in a single chain. Postorder
/// already places columns adjacently, so renumbering should be a
/// no-op (identity bias). Both strategies must produce the same
/// supernode list.
fn tridiagonal(n: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        rows.push(j);
        cols.push(j);
        vals.push(2.0);
        if j + 1 < n {
            rows.push(j + 1);
            cols.push(j);
            vals.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

#[test]
fn arrow_matrix_collapses_under_renumber() {
    let m = arrow_matrix(8);

    // Adjacency strategy: only one merge possible. Expect ≥3
    // supernodes (specifically: many leaf singletons + the root).
    let adj_params = SupernodeParams {
        nemin: 32,
        amalgamation_strategy: AmalgamationStrategy::Adjacency,
        ..Default::default()
    };
    let adj_sym = symbolic_factorize_with_method(&m, &adj_params, OrderingMethod::Amd).unwrap();
    assert!(
        adj_sym.supernodes.len() >= 2,
        "adjacency strategy on arrow matrix should under-merge; got {} supernodes",
        adj_sym.supernodes.len()
    );

    // Renumber strategy: with nemin=32 ≥ n=8, every fundamental
    // supernode is a candidate to merge into its parent → 1
    // supernode covering all 8 columns.
    let renum_params = SupernodeParams {
        nemin: 32,
        amalgamation_strategy: AmalgamationStrategy::Renumber,
        ..Default::default()
    };
    let renum_sym = symbolic_factorize_with_method(&m, &renum_params, OrderingMethod::Amd).unwrap();
    assert_eq!(
        renum_sym.supernodes.len(),
        1,
        "renumber strategy on arrow matrix should collapse to 1 supernode"
    );
    assert_eq!(renum_sym.supernodes[0].ncol(), 8);
}

#[test]
fn bushy_fan_collapses_under_renumber() {
    // Same shape as the arrow matrix at larger n. Catches the case
    // where the bias-emit-late traversal needs to handle multiple
    // (≥10) sibling subtrees correctly.
    let m = bushy_fan(33);
    let params = SupernodeParams {
        nemin: 64,
        amalgamation_strategy: AmalgamationStrategy::Renumber,
        ..Default::default()
    };
    let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::Amd).unwrap();
    assert_eq!(
        sym.supernodes.len(),
        1,
        "bushy fan should collapse to 1 supernode under renumber"
    );
    assert_eq!(sym.supernodes[0].ncol(), 33);
}

#[test]
fn tridiagonal_renumber_is_at_least_as_aggressive() {
    // Tridiagonal under both strategies must total to n columns.
    // Renumber is strictly more aggressive (reverse iteration +
    // bias) so its supernode count must be ≤ Adjacency's.
    let m = tridiagonal(8);
    let adj_params = SupernodeParams {
        nemin: 32,
        amalgamation_strategy: AmalgamationStrategy::Adjacency,
        ..Default::default()
    };
    let renum_params = SupernodeParams {
        nemin: 32,
        amalgamation_strategy: AmalgamationStrategy::Renumber,
        ..Default::default()
    };
    let adj = symbolic_factorize_with_method(&m, &adj_params, OrderingMethod::Amd).unwrap();
    let renum = symbolic_factorize_with_method(&m, &renum_params, OrderingMethod::Amd).unwrap();
    assert!(
        renum.supernodes.len() <= adj.supernodes.len(),
        "Renumber must not produce more supernodes than Adjacency; got {} vs {}",
        renum.supernodes.len(),
        adj.supernodes.len()
    );
    let adj_total: usize = adj.supernodes.iter().map(|s| s.ncol()).sum();
    let renum_total: usize = renum.supernodes.iter().map(|s| s.ncol()).sum();
    assert_eq!(adj_total, 8);
    assert_eq!(renum_total, 8);
}

#[test]
fn perm_is_valid_bijection_under_renumber() {
    // Whatever the renumbering does, the resulting perm must be a
    // bijection on 0..n.
    for n in [4, 8, 33, 64] {
        let m = arrow_matrix(n);
        let params = SupernodeParams {
            nemin: 32,
            amalgamation_strategy: AmalgamationStrategy::Renumber,
            ..Default::default()
        };
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::Amd).unwrap();
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            (0..n).collect::<Vec<_>>(),
            "n={}: perm is not a bijection on 0..n",
            n
        );
        for i in 0..n {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
            assert_eq!(sym.perm_inv[sym.perm[i]], i);
        }
    }
}
