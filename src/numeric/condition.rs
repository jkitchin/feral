//! Hager-Higham 1-norm condition number estimator.
//!
//! Given a sparse symmetric indefinite factor of `A`, estimate
//! `kappa_1(A) = ||A||_1 * ||A^{-1}||_1` using Hager's 1984
//! power iteration with Higham's 1988 alternating-sign refinement.
//!
//! Cost: 3-5 solves with the stored factor per call. The estimator
//! is opt-in — callers invoke it explicitly when conditioning data
//! is wanted (e.g., for ripopt's `delta_w` schedule diagnostics).
//!
//! See `dev/research/condition-estimate.md` for the design rationale
//! and validation strategy.
//!
//! # References
//! - Hager, W. W. (1984). "Condition Estimates." SIAM J. Sci. Stat.
//!   Comput. 5(2), 311-316.
//! - Higham, N. J. (1988). "FORTRAN codes for estimating the
//!   one-norm of a real or complex matrix." ACM TOMS 14(4), 381-396.
//! - LAPACK auxiliary routine DLACON.

use super::factorize::SparseFactors;
use super::solve::{solve_sparse_into_ws, SolveWorkspace};
use crate::error::FeralError;
use crate::sparse::csc::CscMatrix;

/// Maximum number of Hager iterations. LAPACK's DLACON also caps at 5.
const MAX_ITER: usize = 5;

/// Compute `||A||_1 = max_j sum_i |A[i,j]|` for a symmetric matrix
/// stored in lower-triangular CSC form.
///
/// The stored entries are `A[i, j]` with `i >= j`. By symmetry each
/// off-diagonal entry contributes to two column sums (column `j` and
/// column `i`), and each diagonal entry contributes to a single column.
pub fn matrix_norm_1(matrix: &CscMatrix) -> f64 {
    if matrix.n == 0 {
        return 0.0;
    }
    let mut col_sums = vec![0.0f64; matrix.n];
    for j in 0..matrix.n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            let i = matrix.row_idx[k];
            let a_abs = matrix.values[k].abs();
            col_sums[j] += a_abs;
            if i != j {
                col_sums[i] += a_abs;
            }
        }
    }
    col_sums.into_iter().fold(0.0f64, f64::max)
}

/// Hager-Higham estimate of `||A^{-1}||_1` given a factor of `A`.
///
/// Uses the symmetric specialization `A^{-T} = A^{-1}`, so each iteration
/// performs one solve (not two as in the general nonsymmetric case).
///
/// Returns `Ok(0.0)` for `n == 0`. Returns `Err(DimensionMismatch)` if
/// the factor's `n` is inconsistent with itself (should not happen in
/// well-formed factors).
pub fn estimate_inverse_norm_1(factors: &SparseFactors) -> Result<f64, FeralError> {
    let n = factors.n;
    if n == 0 {
        return Ok(0.0);
    }

    // N5 (`dev/research/repo-review-2026-06-09.md`): pool one solve
    // workspace and one output buffer across the up-to 2·MAX_ITER + 1
    // internal solves. `solve_sparse` allocates a fresh `SolveWorkspace`
    // (three vecs) plus a result vec per call, and this estimator calls it
    // ~11× — the cited allocation churn. Reuse is safe: `solve_sparse_into_ws`
    // fully overwrites both its output (`sol`) and its scratch on every call
    // (see solve.rs), and `sol` (the "y" output) is fully consumed into `xi`
    // before it is overwritten by the "z" solve. The arithmetic is
    // bit-identical to the old per-call `solve_sparse`.
    let mut ws = SolveWorkspace::for_factors(factors);
    let mut sol = vec![0.0f64; n];
    let mut xi = vec![0.0f64; n];

    // x_0 = (1/n, ..., 1/n)
    let mut x = vec![1.0 / (n as f64); n];
    let mut est_old = 0.0f64;
    let mut est = 0.0f64;
    // Track the previous unit-vector index so we can detect cycles
    // (LAPACK DLACON's standard termination guard for ties).
    let mut prev_j: Option<usize> = None;

    for _iter in 0..MAX_ITER {
        // y = A^{-1} x  (written into `sol`)
        solve_sparse_into_ws(factors, &x, &mut sol, &mut ws)?;
        est = sol.iter().map(|v| v.abs()).sum();

        // Termination: estimate stopped growing.
        if _iter > 0 && est <= est_old {
            est = est_old;
            break;
        }

        // xi = sign(y), with sign(0) = +1 (LAPACK convention).
        for (slot, &v) in xi.iter_mut().zip(sol.iter()) {
            *slot = if v >= 0.0 { 1.0 } else { -1.0 };
        }

        // z = A^{-T} xi = A^{-1} xi (symmetry), reusing `sol` — the "y"
        // values are already folded into `xi` above, so overwriting is safe.
        solve_sparse_into_ws(factors, &xi, &mut sol, &mut ws)?;

        // Termination: ||z||_inf <= z . x  =>  current estimate is
        // a local maximum on the cube {x: ||x||_1 <= 1}.
        let zx: f64 = sol.iter().zip(x.iter()).map(|(zi, xv)| zi * xv).sum();
        let z_inf = sol.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        if z_inf <= zx {
            break;
        }

        // x = e_j with j = argmax |z_i|.
        let mut j = 0usize;
        let mut zmax = 0.0f64;
        for (i, &zi) in sol.iter().enumerate() {
            let zabs = zi.abs();
            if zabs > zmax {
                zmax = zabs;
                j = i;
            }
        }
        // Cycle guard (DLACON convention): if argmax repeats, terminate.
        if Some(j) == prev_j {
            break;
        }
        prev_j = Some(j);

        for slot in x.iter_mut() {
            *slot = 0.0;
        }
        x[j] = 1.0;
        est_old = est;
    }

    // Higham 1988 refinement: try the alternating-sign vector
    //   b_i = (-1)^{i+1} * (1 + (i-1)/(n-1))
    // (1-based, so for 0-based i: sign = (-1)^i, magnitude = 1 + i/(n-1) for n>1).
    // Replace `est` if 2 * ||A^{-1} b||_1 / (3n) is larger.
    let mut b = vec![0.0f64; n];
    if n == 1 {
        b[0] = 1.0;
    } else {
        let denom = (n - 1) as f64;
        for (i, slot) in b.iter_mut().enumerate() {
            let mag = 1.0 + (i as f64) / denom;
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            *slot = sign * mag;
        }
    }
    solve_sparse_into_ws(factors, &b, &mut sol, &mut ws)?;
    let yb_l1: f64 = sol.iter().map(|v| v.abs()).sum();
    let refined = 2.0 * yb_l1 / (3.0 * n as f64);
    if refined > est {
        est = refined;
    }

    Ok(est)
}

/// Estimate `kappa_1(A) = ||A||_1 * ||A^{-1}||_1`.
///
/// Cost: one O(nnz) pass over `matrix` plus 3-5 solves with `factors`.
///
/// Returns `Ok(0.0)` for `n == 0`. The caller is responsible for ensuring
/// `factors` was produced from `matrix` (or a permutation/scaling of it).
pub fn estimate_condition_1norm(
    matrix: &CscMatrix,
    factors: &SparseFactors,
) -> Result<f64, FeralError> {
    if matrix.n != factors.n {
        return Err(FeralError::DimensionMismatch {
            expected: factors.n,
            got: matrix.n,
        });
    }
    if matrix.n == 0 {
        return Ok(0.0);
    }
    let anorm = matrix_norm_1(matrix);
    let inv_norm = estimate_inverse_norm_1(factors)?;
    Ok(anorm * inv_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::factorize::factorize_multifrontal;
    use crate::symbolic::{symbolic_factorize, SupernodeParams};

    fn factor(m: &CscMatrix) -> SparseFactors {
        let sym = symbolic_factorize(m, &SupernodeParams::default()).unwrap();
        let (factors, _) = factorize_multifrontal(
            m,
            &sym,
            &crate::numeric::factorize::NumericParams::default(),
        )
        .unwrap();
        factors
    }

    /// Diagonal symmetric matrix with chosen spectrum.
    fn diag(values: &[f64]) -> CscMatrix {
        let n = values.len();
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        CscMatrix::from_triplets(n, &rows, &cols, values).unwrap()
    }

    /// N5 (`dev/research/repo-review-2026-06-09.md`): `estimate_inverse_norm_1`
    /// drives up to 2·MAX_ITER + 1 internal solves. Pre-fix each went
    /// through `solve_sparse`, which constructs a fresh `SolveWorkspace`
    /// (three vecs) plus a result vec on every call — the cited ~11×
    /// allocation churn. The fix pools one workspace and one output buffer
    /// across all the internal solves.
    ///
    /// `SOLVE_WORKSPACE_BUILDS` counts `SolveWorkspace` constructions
    /// (`#[cfg(test)]` only).
    ///   * Pre-fix: one build per internal `solve_sparse` (≥ 3 here) — RED.
    ///   * Post-fix: exactly one build, reused across every solve — GREEN.
    #[test]
    fn estimate_pools_one_solve_workspace_across_internal_solves() {
        use crate::numeric::solve::{reset_solve_workspace_builds, solve_workspace_builds};

        // A non-trivial diagonal spectrum: the Hager iteration runs its
        // initial solve, at least one sign-vector solve, and the Higham
        // refinement solve, so the pre-fix path constructs several
        // workspaces. Math is unaffected by the pooling.
        let m = diag(&[1.0, 2.0, 4.0, 8.0]);
        let f = factor(&m);

        reset_solve_workspace_builds();
        let est = estimate_inverse_norm_1(&f).expect("estimate");
        let builds = solve_workspace_builds();

        // Sanity: the estimate is still finite and positive (the pooling
        // must not change the result).
        assert!(
            est.is_finite() && est > 0.0,
            "estimate must stay valid: {est}"
        );

        assert_eq!(
            builds, 1,
            "N5: estimate_inverse_norm_1 must construct exactly ONE \
             SolveWorkspace and reuse it across all internal solves; pre-fix \
             it built one per solve_sparse call ({builds} here)",
        );
    }

    #[test]
    fn matrix_norm_1_diagonal() {
        // Diagonal A with entries [1, -3, 2]: ||A||_1 = 3.
        let m = diag(&[1.0, -3.0, 2.0]);
        assert!((matrix_norm_1(&m) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn matrix_norm_1_symmetric_off_diagonal() {
        // 2x2: [[1, 2], [2, 5]] (lower stored as (0,0)=1, (1,0)=2, (1,1)=5).
        // Column 0 sum: |1| + |2| = 3. Column 1 sum: |2| + |5| = 7. Max = 7.
        let m = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[1.0, 2.0, 5.0]).unwrap();
        assert!((matrix_norm_1(&m) - 7.0).abs() < 1e-15);
    }

    #[test]
    fn condition_diagonal_spectrum() {
        // Diagonal A = diag(1, 1e3, 1e6). True kappa_1 = 1e6.
        // Hager on a diagonal converges in 1 iteration to the exact value.
        let m = diag(&[1.0, 1e3, 1e6]);
        let factors = factor(&m);
        let kappa = estimate_condition_1norm(&m, &factors).unwrap();
        // True value is exactly 1e6; estimator should be within 2x.
        assert!(
            (0.5e6..=2.0e6).contains(&kappa),
            "diagonal kappa {} not within 2x of 1e6",
            kappa
        );
    }

    #[test]
    fn condition_well_conditioned_lower_bound() {
        // Identity scaled has kappa = 1. Estimator must return >= 1 - sqrt(eps).
        let m = diag(&[1.0; 5]);
        let factors = factor(&m);
        let kappa = estimate_condition_1norm(&m, &factors).unwrap();
        assert!(kappa >= 1.0 - 1e-8, "kappa {} below 1 for identity", kappa);
        assert!(
            kappa <= 2.0,
            "kappa {} unexpectedly large for identity",
            kappa
        );
    }

    #[test]
    fn condition_n_zero() {
        let m = CscMatrix::from_triplets(0, &[], &[], &[]).unwrap();
        let factors = factor(&m);
        let kappa = estimate_condition_1norm(&m, &factors).unwrap();
        assert_eq!(kappa, 0.0);
    }

    #[test]
    fn condition_dimension_mismatch_rejected() {
        let m1 = diag(&[1.0, 2.0]);
        let m2 = diag(&[1.0, 2.0, 3.0]);
        let factors = factor(&m1);
        let r = estimate_condition_1norm(&m2, &factors);
        assert!(r.is_err());
    }

    /// Hilbert matrix H_n[i,j] = 1/(i+j+1) (0-indexed).
    /// Stored lower triangle.
    fn hilbert(n: usize) -> CscMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            for i in j..n {
                rows.push(i);
                cols.push(j);
                vals.push(1.0 / ((i + j + 1) as f64));
            }
        }
        CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn condition_hilbert_n4() {
        // H_4 has true kappa_1 ~ 2.84e4 (Higham 2002 §28 table).
        // Hager is a lower bound; require >= 2.84e3 (10x tolerance) and
        // <= 1e6 (sanity).
        let m = hilbert(4);
        let factors = factor(&m);
        let kappa = estimate_condition_1norm(&m, &factors).unwrap();
        let true_kappa = 2.84e4;
        assert!(
            kappa >= true_kappa / 10.0,
            "H_4 kappa {} below 10% of true {}",
            kappa,
            true_kappa
        );
        // Lower bound; should not exceed true by much.
        assert!(
            kappa <= 10.0 * true_kappa,
            "H_4 kappa {} above 10x true {}",
            kappa,
            true_kappa
        );
    }

    #[test]
    fn condition_hilbert_n6() {
        // H_6 true kappa_1 ~ 2.91e7 (Higham 2002 §28).
        let m = hilbert(6);
        let factors = factor(&m);
        let kappa = estimate_condition_1norm(&m, &factors).unwrap();
        let true_kappa = 2.91e7;
        assert!(
            kappa >= true_kappa / 10.0,
            "H_6 kappa {} below 10% of true {}",
            kappa,
            true_kappa
        );
        assert!(
            kappa <= 10.0 * true_kappa,
            "H_6 kappa {} above 10x true {}",
            kappa,
            true_kappa
        );
    }
}
