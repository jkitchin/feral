//! Regression tests for the postorder pipeline fix.
//!
//! These tests guard against the symbolic_factorize bug where the elimination
//! tree was not postordered before supernode detection, causing amalgamated
//! supernodes to have non-contiguous column sets while downstream code
//! assumed first_col..first_col+ncol was the eliminated set.
//!
//! See dev/research/postorder-pipeline.md for the full explanation.

#![allow(clippy::doc_lazy_continuation, clippy::needless_range_loop)]
use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    factor, read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction,
};
use std::path::Path;

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    })
}

/// Hand-built bordered KKT:
///
///     [ 1  0  0  -1 ]
///     [ 0  1  0  -1 ]
///     [ 0  0  1  -1 ]
///     [-1 -1 -1   0 ]
///
/// Inertia: H = I_3 (3 positive). The constraint row's Schur complement is
/// -A·H^-1·A^T = -[-1 -1 -1]·[-1 -1 -1]^T = -3, so 1 negative pivot.
/// Total inertia: (3, 1, 0).
///
/// This mirrors MGH10S structure at minimum size: each variable column
/// couples to a single constraint row, and the constraint row has zero
/// (or near-zero) diagonal. Elimination tree:
///   - parent[0] = 3, parent[1] = 3, parent[2] = 3, parent[3] = None
/// Cols 0, 1, 2 are siblings of constraint col 3 — non-contiguous in the
/// natural ordering. Without postorder, find_supernodes would amalgamate
/// them into a "supernode" reporting first_col=0, ncol=4 but actually
/// containing the wrong column set.
fn bordered_kkt_4x4() -> (CscMatrix, Inertia) {
    // Lower triangle only:
    //   (0,0)=1, (1,1)=1, (2,2)=1
    //   (3,0)=-1, (3,1)=-1, (3,2)=-1
    //   (3,3)=0
    let m = CscMatrix::from_triplets(
        4,
        &[0, 3, 1, 3, 2, 3, 3],
        &[0, 0, 1, 1, 2, 2, 3],
        &[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0],
    )
    .unwrap();
    let expected = Inertia {
        positive: 3,
        negative: 1,
        zero: 0,
    };
    (m, expected)
}

#[test]
fn bordered_kkt_4x4_dense_oracle() {
    // Sanity: dense path produces the expected inertia.
    // If this fails, the test oracle is wrong, not the sparse code.
    let (csc, expected) = bordered_kkt_4x4();
    let dense = csc.to_dense();
    let (_, inertia) = factor(&dense, &ldlt_params().bk).expect("dense factor");
    assert_eq!(
        inertia, expected,
        "dense oracle is wrong: got {} expected {}",
        inertia, expected
    );
}

#[test]
fn bordered_kkt_4x4_sparse_inertia_matches_dense() {
    // Regression: sparse must produce the same inertia as dense.
    // Before the postorder fix, sparse returned (4, 0, 0) here because
    // the constraint column 3 was never properly eliminated.
    let (csc, expected) = bordered_kkt_4x4();

    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (_, sparse_inertia) =
        factorize_multifrontal(&csc, &sym, &ldlt_params()).expect("sparse factor");

    assert_eq!(
        sparse_inertia, expected,
        "sparse inertia mismatch on bordered KKT: got {} expected {}. \
         This indicates the postorder pipeline is broken — see \
         dev/research/postorder-pipeline.md.",
        sparse_inertia, expected
    );
}

/// Bordered KKT with two variable groups coupled to the same constraint row,
/// matching the MGH10S structure exactly at small size:
///
///     [ 100   0    0    0   -1   0   0    0  ]
///     [  0  100    0    0    0  -1   0    0  ]
///     [  0    0  100    0    0   0  -1    0  ]
///     [  0    0    0  1e-9   0   0   0    1  ]   ← wait, this is wrong
///
/// Actually let's keep it cleaner: 2 variable rows + 1 constraint row.
/// Skipped: this is captured by the 4x4 case + MGH10S.
/// Two-constraint bordered KKT mirroring MGH10S structure at small size:
///
///     [100    0    0    0   -1   0 ]   row 0: var
///     [  0  100    0    0    0  -1 ]   row 1: var
///     [  0    0  1e-9   0    1   0 ]   row 2: slack
///     [  0    0    0  1e-9   0   1 ]   row 3: slack
///     [-1     0    1    0  -1e-8  0]   row 4: constraint 0
///     [  0   -1    0    1    0 -1e-8] row 5: constraint 1
///
/// Elimination tree (natural order):
///   parent[0] = 4, parent[2] = 4, parent[4] = None
///   parent[1] = 5, parent[3] = 5, parent[5] = None
///
/// Constraint 4's children are {0, 2} (non-adjacent in column numbering).
/// Constraint 5's children are {1, 3}. Two disjoint subtrees.
///
/// Without postorder, find_supernodes will amalgamate cols 0, 2 with 4 (and
/// 1, 3 with 5) under size-based merging, producing supernodes whose
/// first_col fields are 0 and 1 respectively but whose actual column SETS
/// are {0, 2, 4} and {1, 3, 5} — not contiguous. Downstream code then
/// interprets first_col=0 ncol=3 as cols {0, 1, 2}, eliminating the wrong
/// columns and missing the negative constraint diagonals.
///
/// Inertia: 2 vars × +100, 2 slacks × +1e-9, plus the 2 constraint rows.
/// After eliminating each variable and its slack into the constraint:
///   constraint Schur = -1e-8 - (-1)·(1/100)·(-1) - (1)·(1/1e-9)·(1) ≈ -1e9
/// Two negative pivots. Total: (4, 2, 0).
fn two_constraint_bordered_kkt() -> (CscMatrix, Inertia) {
    let m = CscMatrix::from_triplets(
        6,
        // Lower triangle:
        //   diagonals: (0,0)=100, (1,1)=100, (2,2)=1e-9, (3,3)=1e-9,
        //              (4,4)=-1e-8, (5,5)=-1e-8
        //   off-diag:  (4,0)=-1, (4,2)=1, (5,1)=-1, (5,3)=1
        &[0, 4, 1, 5, 2, 4, 3, 5, 4, 5],
        &[0, 0, 1, 1, 2, 2, 3, 3, 4, 5],
        &[100.0, -1.0, 100.0, -1.0, 1e-9, 1.0, 1e-9, 1.0, -1e-8, -1e-8],
    )
    .unwrap();
    let expected = Inertia {
        positive: 4,
        negative: 2,
        zero: 0,
    };
    (m, expected)
}

#[test]
fn two_constraint_bordered_dense_oracle() {
    let (csc, expected) = two_constraint_bordered_kkt();
    let dense = csc.to_dense();
    let (_, inertia) = factor(&dense, &ldlt_params().bk).expect("dense factor");
    assert_eq!(inertia, expected, "dense oracle wrong: got {}", inertia);
}

#[test]
fn two_constraint_bordered_sparse_matches_dense() {
    // This is the smallest case that exhibits the postorder bug.
    // Sparse must produce the same (4, 2, 0) inertia as dense.
    let (csc, expected) = two_constraint_bordered_kkt();

    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (_, sparse_inertia) =
        factorize_multifrontal(&csc, &sym, &ldlt_params()).expect("sparse factor");

    assert_eq!(
        sparse_inertia, expected,
        "sparse inertia mismatch on two-constraint bordered KKT: got {} expected {}. \
         Two disjoint constraint subtrees were amalgamated under size-based merging \
         and the merged supernodes have non-contiguous columns. See \
         dev/research/postorder-pipeline.md.",
        sparse_inertia, expected
    );
}

#[test]
#[ignore]
fn mgh10s_sparse_inertia_matches_sidecar() {
    // Real-world regression: MGH10S_0000 from the 153k KKT corpus had
    // sparse residual 2.61e21 before the postorder fix because the sparse
    // path computed inertia (50, 1, 0) instead of (35, 16, 0).
    //
    // This test is #[ignore]d because the data file is not committed.
    // Run with: cargo test --test sparse_postorder -- --ignored
    let mtx_path = Path::new("data/matrices/kkt/MGH10S/MGH10S_0000.mtx");
    let json_path = Path::new("data/matrices/kkt/MGH10S/MGH10S_0000.json");

    if !mtx_path.exists() || !json_path.exists() {
        eprintln!("SKIP: {} not found", mtx_path.display());
        return;
    }

    let mtx = read_mtx(mtx_path).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let sc = read_sidecar(json_path).expect("read sidecar");
    let expected = Inertia {
        positive: sc.inertia.positive,
        negative: sc.inertia.negative,
        zero: sc.inertia.zero,
    };

    // Dense oracle should already match
    let dense = mtx.to_dense();
    let (_, dense_inertia) = factor(&dense, &ldlt_params().bk).expect("dense factor");
    assert_eq!(
        dense_inertia, expected,
        "dense path is broken on MGH10S_0000 — investigate before postorder claims"
    );

    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let (_, sparse_inertia) =
        factorize_multifrontal(&csc, &sym, &ldlt_params()).expect("sparse factor");

    assert_eq!(
        sparse_inertia, expected,
        "sparse inertia mismatch on MGH10S_0000: got {} expected {}",
        sparse_inertia, expected
    );
}
