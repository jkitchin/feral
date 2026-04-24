//! Integration tests for Phase 2.6.5 LDLᵀ-aware ordering preprocessing.
//!
//! The module-level unit tests in `src/symbolic/ldlt_compress.rs` cover
//! the pure-algorithmic invariants on hand-crafted permutations and
//! adjacency patterns. The tests here exercise the full symbolic +
//! numeric pipeline on real matrices: they verify that turning on
//! `OrderingPreprocess::LdltCompress` produces a valid permutation and
//! that the resulting factorization's inertia matches the
//! no-preprocessing baseline bit-exactly, as required by the
//! "0 inertia regressions" criterion in the plan's Step 8.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, OrderingPreprocess, SupernodeParams};
use feral::CscMatrix;

/// 4×4 block-anti-diagonal: MC64 will match (0,2) and (1,3) on the
/// large off-diagonals. Under compression, the ordering graph is 2×2.
fn block_antidiag_4x4() -> CscMatrix {
    // Lower-triangle triplets for the symmetric matrix with zero
    // diagonals at 2,3 and large off-diagonal blocks at (2,0), (3,1):
    //   A = [ 1   0   10  0  ]
    //       [ 0   1   0   10 ]
    //       [ 10  0   0   0  ]
    //       [ 0   10  0   0  ]
    // (stored as lower-triangle: (0,0), (1,1), (2,0), (3,1).)
    let rows = vec![0, 1, 2, 3];
    let cols = vec![0, 1, 0, 1];
    let vals = vec![1.0, 1.0, 10.0, 10.0];
    CscMatrix::from_triplets(4, &rows, &cols, &vals).unwrap()
}

/// A small KKT-flavoured matrix with nontrivial MC64 matching: a 2×2
/// SPD block plus one linear constraint with a zero diagonal at the
/// multiplier row. Mirrors the tiny bordered structure that shows up
/// throughout the IPM corpus.
fn tiny_kkt_3x3() -> CscMatrix {
    // H = [[2, 1], [1, 3]]; J = [1, 1]; zero (2,2) block.
    // Full matrix:
    //   [ 2  1  1 ]
    //   [ 1  3  1 ]
    //   [ 1  1  0 ]
    // Lower triangle: (0,0)=2, (1,0)=1, (1,1)=3, (2,0)=1, (2,1)=1, (2,2)=0
    // Drop the structural zero (2,2) so the matrix has a true zero
    // diagonal at row/col 2 — MC64 will match row 2 ↔ col 0 or 1.
    let rows = vec![0, 1, 1, 2, 2];
    let cols = vec![0, 0, 1, 0, 1];
    let vals = vec![2.0, 1.0, 3.0, 1.0, 1.0];
    CscMatrix::from_triplets(3, &rows, &cols, &vals).unwrap()
}

#[test]
fn compress_produces_valid_permutation_on_block_antidiag() {
    let m = block_antidiag_4x4();
    let params = SupernodeParams {
        nemin: 1,
        preprocess: OrderingPreprocess::LdltCompress,
        ..SupernodeParams::default()
    };
    let sym = symbolic_factorize(&m, &params).unwrap();
    assert_eq!(sym.n, 4);
    let mut sorted = sym.perm.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        vec![0, 1, 2, 3],
        "compressed-path perm must be a bijection"
    );
    for i in 0..4 {
        assert_eq!(sym.perm[sym.perm_inv[i]], i, "perm/perm_inv consistency");
    }
}

#[test]
fn compress_matches_baseline_inertia_tiny_kkt() {
    let m = tiny_kkt_3x3();
    let np = NumericParams::default();

    let base_params = SupernodeParams {
        nemin: 1,
        preprocess: OrderingPreprocess::None,
        ..SupernodeParams::default()
    };
    let compress_params = SupernodeParams {
        nemin: 1,
        preprocess: OrderingPreprocess::LdltCompress,
        ..SupernodeParams::default()
    };

    let sym_base = symbolic_factorize(&m, &base_params).unwrap();
    let sym_cmp = symbolic_factorize(&m, &compress_params).unwrap();

    let (_f_base, inertia_base) = factorize_multifrontal(&m, &sym_base, &np).unwrap();
    let (_f_cmp, inertia_cmp) = factorize_multifrontal(&m, &sym_cmp, &np).unwrap();

    assert_eq!(
        inertia_base, inertia_cmp,
        "LDLT compression must not change inertia (bit-exact parity required)"
    );
}

#[test]
fn compress_on_diagonal_matrix_is_noop_equivalent() {
    // Diagonal matrix: MC64 matches every column to itself, so
    // compression is a no-op. The resulting factorization must behave
    // identically to the uncompressed path.
    let rows: Vec<usize> = (0..5).collect();
    let cols: Vec<usize> = (0..5).collect();
    let vals = vec![2.0, -3.0, 4.0, 5.0, -1.0];
    let m = CscMatrix::from_triplets(5, &rows, &cols, &vals).unwrap();
    let np = NumericParams::default();

    let base = SupernodeParams {
        nemin: 1,
        preprocess: OrderingPreprocess::None,
        ..SupernodeParams::default()
    };
    let cmp = SupernodeParams {
        nemin: 1,
        preprocess: OrderingPreprocess::LdltCompress,
        ..SupernodeParams::default()
    };
    let sb = symbolic_factorize(&m, &base).unwrap();
    let sc = symbolic_factorize(&m, &cmp).unwrap();
    let (_fb, ib) = factorize_multifrontal(&m, &sb, &np).unwrap();
    let (_fc, ic) = factorize_multifrontal(&m, &sc, &np).unwrap();
    assert_eq!(
        ib, ic,
        "compression on a diagonal matrix (trivial matching) must produce identical inertia"
    );
    // Expected inertia: 3 positive (2.0, 4.0, 5.0), 2 negative (-3.0, -1.0),
    // 0 zero.
    assert_eq!(ib.positive, 3);
    assert_eq!(ib.negative, 2);
    assert_eq!(ib.zero, 0);
}
