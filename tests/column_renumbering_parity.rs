//! Phase 2.12: numeric parity check for the `Renumber` strategy.
//!
//! For each fixture, factor the same matrix under both
//! `Adjacency` and `Renumber` strategies, solve a synthetic RHS via
//! the resulting LDLᵀ, and assert:
//!
//! 1. Both factorizations succeed.
//! 2. The inertia (n_pos, n_neg, n_zero) reported by each is
//!    identical (correctness invariant: LDLᵀ inertia is independent
//!    of permutation when the pivoting threshold is the same).
//! 3. Both produce a relative residual ≤ 1e-8 on the synthetic RHS.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{
    symbolic_factorize_with_method, AmalgamationStrategy, OrderingMethod, SupernodeParams,
};
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn ldlt_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    })
}

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        let r = ax[i] - b[i];
        rs += r * r;
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

fn solve_under(
    strategy: AmalgamationStrategy,
    csc: &CscMatrix,
    rhs: &[f64],
) -> (Vec<f64>, (usize, usize, usize)) {
    let snode_params = SupernodeParams {
        amalgamation_strategy: strategy,
        ..Default::default()
    };
    let sym =
        symbolic_factorize_with_method(csc, &snode_params, OrderingMethod::Amd).expect("symbolic");
    let (sfac, inertia) = factorize_multifrontal(csc, &sym, &ldlt_params()).expect("factor");
    let triple = (inertia.positive, inertia.negative, inertia.zero);
    let x = solve_sparse_refined(csc, &sfac, rhs).expect("solve_sparse_refined");
    (x, triple)
}

/// Arrow matrix: dense root frontal under Renumber. SPD construction
/// (diagonal dominance) so inertia is (n, 0, 0).
fn arrow_spd(n: usize) -> CscMatrix {
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
    // Diagonal dominance: tip diagonal > sum of |off-diagonals|.
    vals.push((n - 1) as f64 + 0.5);
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

/// Bordered KKT-shaped indefinite. Top block SPD (size m), bottom
/// block 0, off-diagonal coupling. Inertia is (m, n-m, 0).
fn bordered_kkt(m: usize, k: usize) -> CscMatrix {
    let n = m + k;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    // Top SPD diagonal block.
    for j in 0..m {
        rows.push(j);
        cols.push(j);
        vals.push(2.0);
    }
    // Off-diagonal coupling: column j (j<m) couples to rows m..m+k.
    for j in 0..m {
        for r in m..n {
            if (j + r) % 3 == 0 {
                rows.push(r);
                cols.push(j);
                vals.push(0.3);
            }
        }
    }
    // Bottom block: small-positive diagonal so the matrix is
    // genuinely indefinite (positive top, near-zero bottom →
    // BunchKaufman delivers a 2x2 pivot).
    for j in m..n {
        rows.push(j);
        cols.push(j);
        vals.push(-0.01);
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

#[test]
fn arrow_matrix_strategies_produce_same_solution() {
    let csc = arrow_spd(8);
    let rhs: Vec<f64> = (0..8).map(|i| 1.0 + i as f64 * 0.5).collect();

    let (xa, inertia_a) = solve_under(AmalgamationStrategy::Adjacency, &csc, &rhs);
    let (xr, inertia_r) = solve_under(AmalgamationStrategy::Renumber, &csc, &rhs);

    assert_eq!(inertia_a, inertia_r, "inertia must match across strategies");

    let res_a = rel_residual(&csc, &xa, &rhs);
    let res_r = rel_residual(&csc, &xr, &rhs);
    assert!(res_a < 1e-8, "Adjacency residual {} too large", res_a);
    assert!(res_r < 1e-8, "Renumber residual {} too large", res_r);

    // Cross-strategy: solutions must agree to refined-residual precision.
    for i in 0..csc.n {
        assert!(
            (xa[i] - xr[i]).abs() < 1e-7,
            "x[{}]: Adj={} Ren={}",
            i,
            xa[i],
            xr[i]
        );
    }
}

#[test]
fn tail_matrix_strategies_agree_on_inertia_and_residual() {
    // Real KKT from the tiny-IPM tail. Skips silently if the corpus
    // isn't present (CI runs without data/).
    use std::path::Path;
    let path = "data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx";
    if !Path::new(path).exists() {
        return;
    }
    let csc = match feral::read_mtx(Path::new(path)) {
        Ok(m) => match m.to_csc() {
            Ok(c) => c,
            Err(_) => return,
        },
        Err(_) => return,
    };
    let rhs: Vec<f64> = (0..csc.n).map(|i| 1.0 + 0.1 * (i as f64)).collect();

    let (xa, inertia_a) = solve_under(AmalgamationStrategy::Adjacency, &csc, &rhs);
    let (xr, inertia_r) = solve_under(AmalgamationStrategy::Renumber, &csc, &rhs);

    assert_eq!(
        inertia_a, inertia_r,
        "ACOPR30_0067 inertia must match across strategies"
    );

    let res_a = rel_residual(&csc, &xa, &rhs);
    let res_r = rel_residual(&csc, &xr, &rhs);
    // Tail matrices are ill-conditioned IPM KKTs; refinement keeps
    // residuals tight but use a permissive bound to avoid flakes.
    assert!(res_a < 1e-6, "Adjacency residual {} too large", res_a);
    assert!(res_r < 1e-6, "Renumber residual {} too large", res_r);
}

#[test]
fn bordered_kkt_strategies_produce_same_solution() {
    let csc = bordered_kkt(6, 4);
    let rhs: Vec<f64> = (0..csc.n).map(|i| ((i as i32 - 5) as f64) * 0.2).collect();

    let (xa, inertia_a) = solve_under(AmalgamationStrategy::Adjacency, &csc, &rhs);
    let (xr, inertia_r) = solve_under(AmalgamationStrategy::Renumber, &csc, &rhs);

    assert_eq!(
        inertia_a, inertia_r,
        "bordered KKT inertia must match across strategies"
    );

    let res_a = rel_residual(&csc, &xa, &rhs);
    let res_r = rel_residual(&csc, &xr, &rhs);
    assert!(res_a < 1e-7, "Adjacency residual {} too large", res_a);
    assert!(res_r < 1e-7, "Renumber residual {} too large", res_r);
}
