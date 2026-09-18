//! Issue #204: a-priori memory and work estimate from the symbolic analysis.
//!
//! The point of the feature is that a host can ask "what will this
//! factorization cost" *before* paying for it. The tests therefore check
//! the estimate against hand-computed oracles on matrices whose factors
//! are known in closed form, not against feral's own output.
//!
//! Two oracles:
//!
//! * **Tridiagonal**, `diag = 4`, `offdiag = -1`, order `n`. Strictly
//!   diagonally dominant, so no pivoting is needed and no fill occurs:
//!   `L` is lower bidiagonal, `nnz(L) = 2n - 1` counting the diagonal.
//! * **Dense** `n x n` SPD. `L` is a full lower triangle, so
//!   `nnz(L) = n(n+1)/2`, it is one supernode of `nrow = ncol = n`, and
//!   the root has no contribution block at all.
//!
//! Both are independent of any ordering: a permutation cannot change the
//! number of nonzeros in the factor of a dense matrix, and the
//! tridiagonal one is fill-free under any elimination order that AMD or
//! AMF would pick on a path graph.

use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{CscMatrix, FactorStatus, Solver};

/// Tridiagonal SPD, lower triangle. `nnz(L) = 2n - 1`, no fill.
fn tridiag(n: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        rows.push(j);
        cols.push(j);
        vals.push(4.0);
        if j + 1 < n {
            rows.push(j + 1);
            cols.push(j);
            vals.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("tridiag")
}

/// Dense SPD, lower triangle. `nnz(L) = n(n+1)/2`.
fn dense_spd(n: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        for i in j..n {
            rows.push(i);
            cols.push(j);
            // Diagonally dominant: diagonal n, off-diagonal -1.
            vals.push(if i == j { n as f64 + 1.0 } else { -1.0 });
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("dense")
}

fn estimate(m: &CscMatrix) -> feral::symbolic::WorkEstimate {
    let sym = symbolic_factorize_with_method(m, &SupernodeParams::default(), OrderingMethod::Amd)
        .expect("symbolic");
    sym.work_estimate()
}

#[test]
fn tridiagonal_factor_nnz_matches_the_closed_form() {
    for n in [8usize, 64, 500] {
        let e = estimate(&tridiag(n));
        assert_eq!(
            e.factor_nnz,
            2 * n - 1,
            "tridiagonal of order {n}: L is bidiagonal, so nnz(L) = 2n-1"
        );
        // `factor_bytes` budgets against the *slacked* figure, which is
        // deliberately larger than the true fill.
        assert_eq!(
            e.factor_bytes,
            e.factor_alloc_nnz * std::mem::size_of::<f64>()
        );
        assert!(
            e.factor_alloc_nnz > e.factor_nnz,
            "the allocation figure carries slack over the true fill"
        );
        assert_eq!(e.n, n);
    }
}

#[test]
fn dense_factor_nnz_and_shape_match_the_closed_form() {
    for n in [16usize, 70] {
        let e = estimate(&dense_spd(n));
        assert_eq!(
            e.factor_nnz,
            n * (n + 1) / 2,
            "dense of order {n}: L is a full lower triangle"
        );
        assert_eq!(
            e.max_front_rows, n,
            "a dense matrix is a single frontal matrix of order n"
        );
        // Σ ncol·nrow² over one supernode with ncol = nrow = n.
        let want = (n as f64).powi(3);
        assert!(
            (e.factor_flops - want).abs() <= want * 1e-12,
            "dense flop proxy should be n^3 = {want}, got {}",
            e.factor_flops
        );
        assert_eq!(
            e.peak_contrib_bytes, 0,
            "the root supernode has no contribution block"
        );
    }
}

#[test]
fn peak_bytes_is_the_factor_plus_the_transient_peak() {
    // The headline number a host budgets against must be the sum of its
    // two documented parts, on a matrix where the transient part is
    // actually non-zero.
    let e = estimate(&tridiag(2000));
    assert_eq!(e.peak_bytes, e.factor_bytes + e.peak_contrib_bytes);
    assert!(e.n_supernodes > 1, "expected a real supernode tree");
}

#[test]
fn estimate_is_a_lower_bound_on_what_the_numeric_phase_stores() {
    // `factor_nnz` counts the nonzeros of L. The supernodal numeric
    // phase stores dense frontal blocks, so what it actually allocates
    // is >= the estimate. A host budgeting on the estimate alone would
    // under-count, and this pins the direction of that error so the
    // doc comment cannot drift from the behaviour.
    for n in [64usize, 400] {
        let m = tridiag(n);
        let e = estimate(&m);
        let mut s = Solver::new().with_ordering(OrderingMethod::Amd);
        assert!(matches!(s.factor(&m, None), FactorStatus::Success));
        // Supernodal storage: each node holds a dense `nrow x nelim`
        // block, so this is what the numeric phase actually keeps.
        let stored: usize = s
            .factors()
            .expect("factors")
            .node_factors
            .iter()
            .map(|nf| nf.nrow * nf.nelim)
            .sum();
        assert!(
            stored >= e.factor_nnz,
            "order {n}: stored {stored} < estimated {}",
            e.factor_nnz
        );
    }
}

#[test]
fn estimate_grows_with_problem_size() {
    let small = estimate(&tridiag(100));
    let large = estimate(&tridiag(1000));
    assert!(large.factor_nnz > small.factor_nnz);
    assert!(large.factor_flops > small.factor_flops);
    assert!(large.peak_bytes > small.peak_bytes);
}

#[test]
fn solver_exposes_the_estimate_after_factoring_and_not_before() {
    let m = tridiag(200);
    let mut s = Solver::new();
    assert!(
        s.work_estimate().is_none(),
        "no symbolic analysis has run yet"
    );
    assert!(matches!(s.factor(&m, None), FactorStatus::Success));
    let e = s.work_estimate().expect("estimate after factor");
    assert_eq!(e.factor_nnz, 2 * 200 - 1);
}
