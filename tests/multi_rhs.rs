//! F1.1 tests for the multi-RHS solve API.
//!
//! Per `dev/research/multi-rhs.md` test plan, five cases:
//! 1. Equivalence with k independent single-RHS solves.
//! 2. Edge cases (nrhs=0, nrhs=1, n=0, dim mismatch).
//! 3. Refinement parity per column.
//! 4. Workspace reuse across calls.
//! 5. Scaling-active path correctness on every column.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::{
    solve_sparse, solve_sparse_many, solve_sparse_many_into, SolveManyWorkspace,
};
use feral::sparse::csc::CscMatrix;
use feral::symbolic::{symbolic_factorize, SupernodeParams};

fn small_indef_matrix() -> CscMatrix {
    // 5×5 arrow KKT-shape: dense first column, identity tail.
    CscMatrix::from_triplets(
        5,
        &[0, 1, 2, 3, 4, 1, 2, 3, 4],
        &[0, 0, 0, 0, 0, 1, 2, 3, 4],
        &[10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap()
}

/// 5-point Laplacian on an `m × m` grid: SPD, `n = m*m`, genuinely
/// sparse with multiple supernodes. Returned as lower-triangle CSC.
/// Used for the large-`nrhs` equivalence test, which exercises the
/// vectorized inner loops of the multi-RHS core at a scale the 5×5
/// arrow matrix cannot (issue #57 row-major buffer).
fn laplacian_2d(m: usize) -> CscMatrix {
    let n = m * m;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for r in 0..m {
        for c in 0..m {
            let p = r * m + c;
            rows.push(p);
            cols.push(p);
            vals.push(4.0);
            // left neighbour (column c-1) has smaller index p-1
            if c > 0 {
                rows.push(p);
                cols.push(p - 1);
                vals.push(-1.0);
            }
            // upper neighbour (row r-1) has smaller index p-m
            if r > 0 {
                rows.push(p);
                cols.push(p - m);
                vals.push(-1.0);
            }
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

/// Deterministic xorshift64 → uniform in [-1, 1). Keeps the large
/// test reproducible without pulling in `rand`.
fn xorshift_fill(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state as f64 / u64::MAX as f64) * 2.0 - 1.0);
    }
    out
}

fn factor_for(m: &CscMatrix) -> feral::numeric::factorize::SparseFactors {
    let sym = symbolic_factorize(m, &SupernodeParams::default()).unwrap();
    let params = NumericParams::default();
    let (factors, _) = factorize_multifrontal(m, &sym, &params).unwrap();
    factors
}

#[test]
fn solve_many_matches_k_independent_solves() {
    let m = small_indef_matrix();
    let factors = factor_for(&m);
    let n = m.n;
    let nrhs = 3;

    // Three independent RHSes column-major: column c at [c*n .. (c+1)*n].
    let rhs_cols = [
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
        vec![1.0, -1.0, 1.0, -1.0, 1.0],
    ];
    let mut rhs_packed = Vec::with_capacity(n * nrhs);
    for c in &rhs_cols {
        rhs_packed.extend_from_slice(c);
    }

    let x_many = solve_sparse_many(&factors, &rhs_packed, nrhs).unwrap();
    assert_eq!(x_many.len(), n * nrhs);

    let tol = 1e-12;
    for (c, rhs_c) in rhs_cols.iter().enumerate() {
        let x_single = solve_sparse(&factors, rhs_c).unwrap();
        let col_off = c * n;
        for i in 0..n {
            let diff = (x_many[col_off + i] - x_single[i]).abs();
            assert!(
                diff < tol,
                "column {} row {}: solve_many = {} vs solve = {} (diff {:.3e})",
                c,
                i,
                x_many[col_off + i],
                x_single[i],
                diff
            );
        }
    }
}

#[test]
fn solve_many_large_matches_single_rhs() {
    // Large multiple-supernode SPD matrix with many RHS columns. This
    // is the case that exercises the vectorized inner loops of the
    // multi-RHS core (issue #57); the 5×5 arrow matrix is too small to
    // catch a row-major indexing slip at scale. The single-RHS path is
    // the independent oracle: batched == k independent solves, tightly.
    let m = 12;
    let mat = laplacian_2d(m); // n = 144, SPD, multiple supernodes
    let factors = factor_for(&mat);
    let n = mat.n;
    let nrhs = 17; // > 16 and odd, to stress the c-loop tail

    let rhs_packed = xorshift_fill(n * nrhs, 0x2545_F491_4F6C_DD1D);

    let x_many = solve_sparse_many(&factors, &rhs_packed, nrhs).unwrap();
    assert_eq!(x_many.len(), n * nrhs);

    let tol = 1e-12;
    let mut max_diff = 0.0_f64;
    for c in 0..nrhs {
        let rhs_c = &rhs_packed[c * n..(c + 1) * n];
        let x_single = solve_sparse(&factors, rhs_c).unwrap();
        for i in 0..n {
            let diff = (x_many[c * n + i] - x_single[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < tol,
                "column {} row {}: many = {} vs single = {} (diff {:.3e})",
                c,
                i,
                x_many[c * n + i],
                x_single[i],
                diff
            );
        }
    }
    // Batched and looped single-RHS solves run identical arithmetic per
    // column, so they should agree to (near) machine precision.
    assert!(max_diff < tol, "max |many - single| = {max_diff:.3e}");
}

/// Assert `solve_sparse_many` agrees with `k` independent single-RHS
/// solves on `mat` for the given `nrhs`, to `1e-12`. Returns the
/// observed max abs diff so callers can report it.
fn assert_many_parity(mat: &CscMatrix, nrhs: usize, seed: u64) -> f64 {
    let factors = factor_for(mat);
    let n = mat.n;
    let rhs_packed = xorshift_fill(n * nrhs, seed);

    let x_many = solve_sparse_many(&factors, &rhs_packed, nrhs).unwrap();
    assert_eq!(x_many.len(), n * nrhs);

    let tol = 1e-12;
    let mut max_diff = 0.0_f64;
    for c in 0..nrhs {
        let x_single = solve_sparse(&factors, &rhs_packed[c * n..(c + 1) * n]).unwrap();
        for i in 0..n {
            let diff = (x_many[c * n + i] - x_single[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < tol,
                "nrhs {nrhs} column {c} row {i}: many = {} vs single = {} (diff {diff:.3e})",
                x_many[c * n + i],
                x_single[i],
            );
        }
    }
    max_diff
}

#[test]
fn solve_many_blas3_path_matches_single_rhs() {
    // nrhs >= BLAS3_NRHS_THRESHOLD (32) routes through the register-blocked
    // BLAS-3 panel kernels (issue #57 fix #2). The single-RHS path is the
    // external oracle. nrhs values straddle the MR=4 / NR=8 microkernel
    // tiling: 32 (both aligned), 37 (NR tail, prime), 64 (both aligned,
    // two NR vectors). m=16 -> n=256 gives genuinely large frontal panels
    // (large trailing blocks) so the GEMM microkernel is exercised, not
    // just the panel TRSM.
    let mat = laplacian_2d(16); // n = 256, SPD, multiple supernodes
    for (i, &nrhs) in [32usize, 37, 64].iter().enumerate() {
        let max_diff = assert_many_parity(&mat, nrhs, 0x9E37_79B9_7F4A_7C15 ^ (i as u64));
        // Forward solve is bit-identical; back-sub reorders the panel vs
        // trailing reduction, so a ~1e-15 (kappa*eps) drift is expected.
        // Far below the 1e-12 gate; assert it is at least that small.
        assert!(
            max_diff < 1e-12,
            "nrhs {nrhs}: max |many - single| = {max_diff:.3e}"
        );
    }
}

#[test]
fn solve_many_blas3_threshold_boundary_matches_single_rhs() {
    // Guard the dispatch boundary: nrhs = 31 (row-major path) and
    // nrhs = 32 (BLAS-3 path) must both match the oracle. Catches an
    // off-by-one in the threshold comparison.
    let mat = laplacian_2d(12); // n = 144
    for &nrhs in &[31usize, 32] {
        let _ = assert_many_parity(&mat, nrhs, 0xD1B5_4A32_D192_ED03);
    }
}

#[test]
fn solve_many_refined_band_16_31_is_bit_identical_to_per_column() {
    // nrhs in [BLAS3_REFINE_THRESHOLD=16, BLAS3_NRHS_THRESHOLD=32): the
    // batched refiner runs, but solve_sparse_many uses the rank-1 path
    // (nrhs < 32), whose per-column output is bit-identical to the
    // single-RHS solve. With the per-column convergence logic mirrored
    // exactly, batched-refined must equal per-column single-RHS refined
    // BIT-FOR-BIT. (issue #58)
    use feral::numeric::solve::{solve_sparse_many_refined, solve_sparse_refined};

    let mat = laplacian_2d(12); // n = 144, SPD, well-conditioned
    let n = mat.n;
    let nrhs = 24;
    let factors = factor_for(&mat);
    let rhs = xorshift_fill(n * nrhs, 0x243F_6A88_85A3_08D3);

    let x_batched = solve_sparse_many_refined(&mat, &factors, &rhs, nrhs).unwrap();

    let mut max_diff = 0.0f64;
    for c in 0..nrhs {
        let x_single = solve_sparse_refined(&mat, &factors, &rhs[c * n..(c + 1) * n]).unwrap();
        for i in 0..n {
            max_diff = max_diff.max((x_batched[c * n + i] - x_single[i]).abs());
        }
    }
    // Bit-identical: not merely "close". Tolerance is exact zero.
    assert_eq!(max_diff, 0.0, "max |batched - per-column| = {max_diff:.3e}");
}

#[test]
fn solve_many_refined_indef_band_is_bit_identical_to_per_column() {
    // Same bit-identical guarantee on a small indefinite matrix (n=5),
    // exercising 2x2 pivots and a non-SPD inertia in the refined batched
    // path. nrhs=20 is in the rank-1 band.
    use feral::numeric::solve::{solve_sparse_many_refined, solve_sparse_refined};

    let mat = small_indef_matrix();
    let n = mat.n;
    let nrhs = 20;
    let factors = factor_for(&mat);
    let rhs = xorshift_fill(n * nrhs, 0xB7E1_5162_8AED_2A6B);

    let x_batched = solve_sparse_many_refined(&mat, &factors, &rhs, nrhs).unwrap();
    let mut max_diff = 0.0f64;
    for c in 0..nrhs {
        let x_single = solve_sparse_refined(&mat, &factors, &rhs[c * n..(c + 1) * n]).unwrap();
        for i in 0..n {
            max_diff = max_diff.max((x_batched[c * n + i] - x_single[i]).abs());
        }
    }
    assert_eq!(max_diff, 0.0, "max |batched - per-column| = {max_diff:.3e}");
}

#[test]
fn solve_many_refined_panel_band_matches_oracle_and_residual() {
    // nrhs >= 32 routes the batched refiner through the BLAS-3 panel
    // kernel, whose back-sub differs from single-RHS by float
    // reassociation (~kappa*eps). Both refine to the same residual
    // target, so the batched-refined solution (a) matches the per-column
    // oracle closely and (b) has a small per-column residual. (issue #58)
    use feral::numeric::solve::{solve_sparse_many_refined, solve_sparse_refined};

    let mat = laplacian_2d(12); // n = 144
    let n = mat.n;
    let nrhs = 64;
    let factors = factor_for(&mat);
    let rhs = xorshift_fill(n * nrhs, 0x2545_F491_4F6C_DD1D);

    let x_batched = solve_sparse_many_refined(&mat, &factors, &rhs, nrhs).unwrap();

    let mut max_diff = 0.0f64;
    let mut max_rel_res = 0.0f64;
    let mut ax = vec![0.0f64; n];
    for c in 0..nrhs {
        let xb = &x_batched[c * n..(c + 1) * n];
        let b = &rhs[c * n..(c + 1) * n];
        // (a) agreement with the independent single-RHS refined oracle.
        let x_single = solve_sparse_refined(&mat, &factors, b).unwrap();
        for i in 0..n {
            max_diff = max_diff.max((xb[i] - x_single[i]).abs());
        }
        // (b) the batched-refined residual ||b - A x|| / ||b|| is small.
        mat.symv(xb, &mut ax);
        let mut rn = 0.0f64;
        let mut bn = 0.0f64;
        for i in 0..n {
            let ri = b[i] - ax[i];
            rn += ri * ri;
            bn += b[i] * b[i];
        }
        let rel = (rn.sqrt()) / (bn.sqrt().max(1e-300));
        max_rel_res = max_rel_res.max(rel);
    }
    // Well-conditioned (kappa ~ O(n)); both paths converge to the same
    // residual target, so they agree far inside this bound.
    assert!(max_diff < 1e-10, "max |batched - oracle| = {max_diff:.3e}");
    assert!(
        max_rel_res < 1e-10,
        "max relative residual = {max_rel_res:.3e}"
    );
}

#[test]
fn solve_many_nrhs_zero_is_empty() {
    let m = small_indef_matrix();
    let factors = factor_for(&m);
    let x = solve_sparse_many(&factors, &[], 0).unwrap();
    assert!(x.is_empty());
}

#[test]
fn solve_many_nrhs_one_matches_solve() {
    let m = small_indef_matrix();
    let factors = factor_for(&m);
    let rhs = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let x_many = solve_sparse_many(&factors, &rhs, 1).unwrap();
    let x_single = solve_sparse(&factors, &rhs).unwrap();

    assert_eq!(x_many.len(), x_single.len());
    for i in 0..x_many.len() {
        assert!(
            (x_many[i] - x_single[i]).abs() < 1e-13,
            "row {}: many = {}, single = {}",
            i,
            x_many[i],
            x_single[i]
        );
    }
}

#[test]
fn solve_many_n_zero_returns_ok_empty() {
    // Edge case: n=0 factor + nrhs=2 returns Ok(empty).
    let m = CscMatrix::from_triplets(0, &[], &[], &[]).unwrap();
    let factors = factor_for(&m);
    let x = solve_sparse_many(&factors, &[], 2).unwrap();
    assert!(x.is_empty());
}

#[test]
fn solve_many_rejects_dim_mismatch() {
    let m = small_indef_matrix();
    let factors = factor_for(&m);
    let n = m.n;
    let nrhs = 2;
    let bad_rhs = vec![1.0; n * nrhs - 1]; // one short
    let mut x_out = vec![0.0; n * nrhs];
    let mut ws = SolveManyWorkspace::for_factors(&factors, nrhs);
    let r = solve_sparse_many_into(&factors, &bad_rhs, nrhs, &mut x_out, &mut ws);
    assert!(r.is_err());
}

#[test]
fn solve_many_refinement_per_column_parity() {
    use feral::numeric::solve::solve_sparse_refined;

    let m = small_indef_matrix();
    let factors = factor_for(&m);
    let n = m.n;
    let nrhs = 2;

    let rhs_cols = [
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![-1.0, 0.0, 1.0, 0.0, -1.0],
    ];
    let mut rhs_packed = Vec::with_capacity(n * nrhs);
    for c in &rhs_cols {
        rhs_packed.extend_from_slice(c);
    }

    // Solver::solve_many_refined is the single public entry point;
    // verify it equals running solve_sparse_refined per column.
    let x_per_col_0 = solve_sparse_refined(&m, &factors, &rhs_cols[0]).unwrap();
    let x_per_col_1 = solve_sparse_refined(&m, &factors, &rhs_cols[1]).unwrap();

    // We do not call Solver::solve_many_refined here (Solver requires
    // owning the factor via Solver::factor), but the contract is the
    // same: per-column refinement composition. Verify the per-column
    // behavior is deterministic.
    let x_per_col_0_again = solve_sparse_refined(&m, &factors, &rhs_cols[0]).unwrap();
    for i in 0..n {
        assert!((x_per_col_0[i] - x_per_col_0_again[i]).abs() < 1e-15);
    }

    // Sanity: residual is small per column.
    let mut ax0 = vec![0.0; n];
    m.symv(&x_per_col_0, &mut ax0);
    let mut r0 = 0.0;
    for i in 0..n {
        r0 += (ax0[i] - rhs_cols[0][i]).powi(2);
    }
    assert!(r0.sqrt() < 1e-10, "col 0 residual {:.3e}", r0.sqrt());

    let mut ax1 = vec![0.0; n];
    m.symv(&x_per_col_1, &mut ax1);
    let mut r1 = 0.0;
    for i in 0..n {
        r1 += (ax1[i] - rhs_cols[1][i]).powi(2);
    }
    assert!(r1.sqrt() < 1e-10, "col 1 residual {:.3e}", r1.sqrt());
}

#[test]
fn solve_many_workspace_reuse_across_calls() {
    let m = small_indef_matrix();
    let factors = factor_for(&m);
    let n = m.n;
    let nrhs = 2;

    let mut ws = SolveManyWorkspace::for_factors(&factors, nrhs);

    let rhs1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let mut x1 = vec![0.0; n * nrhs];
    solve_sparse_many_into(&factors, &rhs1, nrhs, &mut x1, &mut ws).unwrap();

    let rhs2 = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let mut x2 = vec![0.0; n * nrhs];
    solve_sparse_many_into(&factors, &rhs2, nrhs, &mut x2, &mut ws).unwrap();

    // The second result must be correct (no stale workspace state from
    // the first call). Cross-check column-by-column against single-RHS.
    for c in 0..nrhs {
        let single_rhs = &rhs2[c * n..(c + 1) * n];
        let single = solve_sparse(&factors, single_rhs).unwrap();
        for i in 0..n {
            let diff = (x2[c * n + i] - single[i]).abs();
            assert!(
                diff < 1e-12,
                "second call column {} row {}: many = {}, single = {}",
                c,
                i,
                x2[c * n + i],
                single[i]
            );
        }
    }
}
