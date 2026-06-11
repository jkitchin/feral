//! Reconstruction tests for `SparseFactors::ldlt_export` and the
//! `Solver::symbolic()` accessor (Python-interface expansion).
//!
//! The oracle is the **input matrix itself**: we hand-build a symmetric
//! matrix `A`, factor it, reassemble `L`/`D` from the export, and verify
//! `L D Lᵀ` reproduces the *scaled, permuted* `A` (so undoing the
//! permutation and scaling recovers the original dense `A`). No part of
//! the factorization writes the oracle — `A` is supplied independently.
//!
//! Cases:
//! - `kkt_saddle_point_reconstructs`: 5×5 indefinite KKT with a zero
//!   (2,2) block, which forces 2×2 pivots and exercises the within-front
//!   Bunch-Kaufman permutation in the export's index mapping.
//! - `spd_tridiagonal_reconstructs`: 40×40 SPD tridiagonal, a
//!   multi-supernode pure-1×1 factorization.
//! - `l_is_unit_lower_triangular`: structural invariants of the export.
//! - `symbolic_accessor_after_factor`: `Solver::symbolic()` is populated
//!   and self-consistent; the predicted nnz bounds the realized nnz.

use feral::{CscMatrix, FactorStatus, Solver};

/// Dense full symmetric matrix (row-major `n×n`) from a lower-triangle
/// triplet list. Mirrors the upper triangle.
fn dense_symmetric(n: usize, rows: &[usize], cols: &[usize], vals: &[f64]) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for k in 0..rows.len() {
        let (r, c, v) = (rows[k], cols[k], vals[k]);
        a[r * n + c] = v;
        a[c * n + r] = v;
    }
    a
}

/// Reconstruct dense `A` from the LDLᵀ export and the symmetric scaling
/// vector, then assert it matches `a_ref` (row-major `n×n`).
fn assert_reconstructs(solver: &Solver, n: usize, a_ref: &[f64], tol: f64) {
    let factors = solver.factors().expect("factor present");
    // A clean (no force-accept / no static-pivot perturbation)
    // factorization must reconstruct exactly; flag otherwise so a
    // tolerance bump can never silently hide an approximate factor.
    assert!(
        !factors.needs_refinement,
        "test matrix should factor exactly (no perturbation)"
    );
    let s = &factors.scaling; // user-order, length n
    let ex = factors.ldlt_export();
    assert_eq!(ex.perm.len(), n);
    assert_eq!(ex.l_indptr.len(), n + 1);

    // M = L · D · Lᵀ in factorization order (dense).
    // First L (n×n, factorization order) and D (block-diagonal).
    let mut l = vec![0.0f64; n * n];
    for col in 0..n {
        for k in ex.l_indptr[col]..ex.l_indptr[col + 1] {
            let row = ex.l_indices[k];
            l[row * n + col] = ex.l_values[k];
        }
    }
    // D as a dense n×n block-diagonal.
    let mut d = vec![0.0f64; n * n];
    let mut e = 0usize;
    while e < n {
        let two = e + 1 < n && ex.d_subdiag[e] != 0.0;
        if two {
            d[e * n + e] = ex.d_diag[e];
            d[(e + 1) * n + (e + 1)] = ex.d_diag[e + 1];
            d[e * n + (e + 1)] = ex.d_subdiag[e];
            d[(e + 1) * n + e] = ex.d_subdiag[e];
            e += 2;
        } else {
            d[e * n + e] = ex.d_diag[e];
            e += 1;
        }
    }
    // M = L D Lᵀ.
    let mut ld = vec![0.0f64; n * n]; // L·D
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += l[i * n + k] * d[k * n + j];
            }
            ld[i * n + j] = acc;
        }
    }
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += ld[i * n + k] * l[j * n + k]; // (L)ᵀ ⇒ l[j,k]
            }
            m[i * n + j] = acc;
        }
    }
    // A[perm[i], perm[j]] = M[i,j] / (s[perm[i]] · s[perm[j]]).
    for i in 0..n {
        for j in 0..n {
            let oi = ex.perm[i];
            let oj = ex.perm[j];
            let a_rec = m[i * n + j] / (s[oi] * s[oj]);
            let a_true = a_ref[oi * n + oj];
            assert!(
                (a_rec - a_true).abs() <= tol * (1.0 + a_true.abs()),
                "reconstruction mismatch at original ({oi},{oj}): \
                 got {a_rec}, want {a_true}"
            );
        }
    }
}

#[test]
fn kkt_saddle_point_reconstructs() {
    // K = [[H, Aᵀ], [A, 0]], H = diag(2,3,4), A = [[1,0,1],[0,1,1]].
    // Nonsingular indefinite (inertia (3,2,0)); the zero (2,2) block
    // forces 2×2 pivots.
    let n = 5;
    let rows = [0, 1, 2, 3, 3, 3, 4, 4, 4];
    let cols = [0, 1, 2, 0, 2, 3, 1, 2, 4];
    let vals = [2.0, 3.0, 4.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let a_ref = dense_symmetric(n, &rows, &cols, &vals);
    let k = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("triplets");

    let mut solver = Solver::new();
    let status = solver.factor(&k, None);
    assert!(matches!(status, FactorStatus::Success), "got {status:?}");
    assert_reconstructs(&solver, n, &a_ref, 1e-9);
}

#[test]
fn spd_tridiagonal_reconstructs() {
    let n = 40;
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
    let a_ref = dense_symmetric(n, &rows, &cols, &vals);
    let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("triplets");

    let mut solver = Solver::new();
    let status = solver.factor(&a, None);
    assert!(matches!(status, FactorStatus::Success), "got {status:?}");
    assert_reconstructs(&solver, n, &a_ref, 1e-9);
}

#[test]
fn l_is_unit_lower_triangular() {
    let n = 40;
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
    let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("triplets");
    let mut solver = Solver::new();
    assert!(matches!(solver.factor(&a, None), FactorStatus::Success));
    let ex = solver.factors().expect("factor").ldlt_export();

    // perm is a permutation.
    let mut seen = vec![false; n];
    for &p in &ex.perm {
        assert!(p < n && !seen[p], "perm not a permutation");
        seen[p] = true;
    }
    // perm_inv is the inverse.
    for e in 0..n {
        assert_eq!(ex.perm_inv[ex.perm[e]], e);
    }
    // Each column: rows are sorted, all >= col (lower-triangular), and
    // the diagonal entry is present and equal to 1 (unit).
    for col in 0..n {
        let start = ex.l_indptr[col];
        let end = ex.l_indptr[col + 1];
        assert!(end > start, "column {col} has no diagonal entry");
        let mut found_diag = false;
        let mut prev: Option<usize> = None;
        for k in start..end {
            let r = ex.l_indices[k];
            assert!(r >= col, "upper-triangle entry at ({r},{col})");
            if let Some(p) = prev {
                assert!(r > p, "rows not strictly sorted in column {col}");
            }
            prev = Some(r);
            if r == col {
                found_diag = true;
                assert_eq!(ex.l_values[k], 1.0, "non-unit diagonal at {col}");
            }
        }
        assert!(found_diag, "missing unit diagonal in column {col}");
    }
}

#[test]
fn symbolic_accessor_after_factor() {
    let n = 30;
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
    let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("triplets");

    let mut solver = Solver::new();
    assert!(solver.symbolic().is_none(), "no symbolic before factor");
    assert!(matches!(solver.factor(&a, None), FactorStatus::Success));

    let sym = solver.symbolic().expect("symbolic present after factor");
    assert_eq!(sym.n, n);
    // perm is a permutation.
    let mut seen = vec![false; n];
    for &p in &sym.perm {
        assert!(p < n && !seen[p]);
        seen[p] = true;
    }
    // etree parents point strictly upward (forest invariant).
    for (j, &par) in sym.etree.parent.iter().enumerate() {
        if let Some(p) = par {
            assert!(p > j && p < n, "etree parent {p} of {j} out of range");
        }
    }
    // Predicted L nnz bounds the realized nnz of the export.
    let realized = solver
        .factors()
        .expect("factor")
        .ldlt_export()
        .l_values
        .len();
    assert!(
        sym.factor_nnz_estimate >= realized,
        "estimate {} should bound realized {}",
        sym.factor_nnz_estimate,
        realized
    );
}
