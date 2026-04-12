#![allow(clippy::needless_range_loop)]
use crate::error::FeralError;
use crate::sparse::csc::CscMatrix;
use super::factorize::SparseFactors;

/// Solve A·x = b using the sparse multifrontal factorization.
///
/// Three phases matching the multifrontal factorization:
/// 1. Forward substitution: L-solve through supernodes (postorder)
/// 2. D-block solve: D^{-1} for eliminated pivots at each node
/// 3. Backward substitution: L^T-solve through supernodes (reverse postorder)
pub fn solve_sparse(
    factors: &SparseFactors,
    rhs: &[f64],
) -> Result<Vec<f64>, FeralError> {
    let n = factors.n;
    if rhs.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: rhs.len(),
        });
    }

    if n == 0 {
        return Ok(Vec::new());
    }

    // Permute RHS with AMD ordering: y[new] = b[perm[new]]
    let mut y = vec![0.0; n];
    for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
        y[new_idx] = rhs[old_idx];
    }

    // Phase 1: Forward substitution (postorder)
    for node in &factors.node_factors {
        let ff = &node.frontal_factors;
        let ncol = ff.ncol;
        let nrow = ff.nrow;
        if ncol == 0 { continue; }

        // Gather and apply BK permutation
        let mut w = vec![0.0; nrow];
        for i in 0..nrow {
            w[i] = y[node.row_indices[ff.perm[i]]];
        }

        // L-solve: for each eliminated column j, update rows below
        for j in 0..ncol {
            let w_j = w[j];
            for i in (j + 1)..nrow {
                w[i] -= ff.l[j * nrow + i] * w_j;
            }
        }

        // Undo BK permutation and scatter back
        for i in 0..nrow {
            y[node.row_indices[ff.perm[i]]] = w[i];
        }
    }

    // Phase 2: D-block solve
    for node in &factors.node_factors {
        let ff = &node.frontal_factors;
        let ncol = ff.ncol;
        let nrow = ff.nrow;
        if ncol == 0 { continue; }

        // Gather and apply BK permutation
        let mut w = vec![0.0; nrow];
        for i in 0..nrow {
            w[i] = y[node.row_indices[ff.perm[i]]];
        }

        // D-block solve (first ncol entries only)
        let mut k = 0;
        while k < ncol {
            if k + 1 < ncol && ff.d_subdiag[k] != 0.0 {
                let a = ff.d_diag[k];
                let b = ff.d_subdiag[k];
                let c = ff.d_diag[k + 1];
                let z1 = w[k];
                let z2 = w[k + 1];

                if b.abs() > f64::EPSILON * (a.abs() + c.abs()).max(1.0) {
                    let ak = a / b;
                    let ck = c / b;
                    let denom = 1.0 / (ak * ck - 1.0);
                    let z1k = z1 / b;
                    let z2k = z2 / b;
                    w[k] = (ck * z1k - z2k) * denom;
                    w[k + 1] = (ak * z2k - z1k) * denom;
                } else {
                    let det = a * c - b * b;
                    if det.abs() > 0.0 {
                        w[k] = (c * z1 - b * z2) / det;
                        w[k + 1] = (a * z2 - b * z1) / det;
                    }
                }
                k += 2;
            } else {
                if ff.d_diag[k].abs() > 0.0 {
                    w[k] /= ff.d_diag[k];
                }
                k += 1;
            }
        }

        // Undo BK permutation and scatter back
        for i in 0..nrow {
            y[node.row_indices[ff.perm[i]]] = w[i];
        }
    }

    // Phase 3: Backward substitution (reverse postorder)
    for node in factors.node_factors.iter().rev() {
        let ff = &node.frontal_factors;
        let ncol = ff.ncol;
        let nrow = ff.nrow;
        if ncol == 0 { continue; }

        // Gather and apply BK permutation
        let mut w = vec![0.0; nrow];
        for i in 0..nrow {
            w[i] = y[node.row_indices[ff.perm[i]]];
        }

        // L^T-solve: for each eliminated column j (reverse order)
        for j in (0..ncol).rev() {
            let mut sum = 0.0;
            for i in (j + 1)..nrow {
                sum += ff.l[j * nrow + i] * w[i];
            }
            w[j] -= sum;
        }

        // Undo BK permutation and scatter back
        for i in 0..nrow {
            y[node.row_indices[ff.perm[i]]] = w[i];
        }
    }

    // Unpermute: x[old] = y[new]
    let mut x = vec![0.0; n];
    for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
        x[old_idx] = y[new_idx];
    }

    Ok(x)
}

/// Solve A·x = rhs using the sparse factorization with iterative refinement.
///
/// Mirrors `crate::dense::solve::solve_refined` for the multifrontal path.
/// Per FERAL-PROJECT-SPEC.md §1709, this is the Phase 1b solve convention:
/// because `ZeroPivotAction::ForceAccept` is the default, an unrefined solve
/// can leave a non-trivial residual on near-singular pivots, and refinement
/// recovers machine precision in 0–3 steps for well-conditioned matrices.
///
/// Convergence test: stop when `||δx||₂ / ||x||₂ < ε·√n` or after 3 steps.
pub fn solve_sparse_refined(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
) -> Result<Vec<f64>, FeralError> {
    let n = factors.n;
    if rhs.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: rhs.len(),
        });
    }

    let mut x = solve_sparse(factors, rhs)?;

    let max_steps = 3;
    let n_sqrt = (n as f64).sqrt();
    let threshold = f64::EPSILON * n_sqrt;

    for _ in 0..max_steps {
        // Residual: r = b - A·x
        let mut ax = vec![0.0; n];
        matrix.symv(&x, &mut ax);
        let mut r = vec![0.0; n];
        for i in 0..n {
            r[i] = rhs[i] - ax[i];
        }

        // Correction: δx = A⁻¹·r
        let dx = solve_sparse(factors, &r)?;

        let dx_norm = norm2(&dx);
        let x_norm = norm2(&x);

        for i in 0..n {
            x[i] += dx[i];
        }

        if x_norm > 0.0 {
            if dx_norm / x_norm < threshold {
                break;
            }
        } else if dx_norm < threshold {
            break;
        }
    }

    Ok(x)
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::factor::{BunchKaufmanParams, ZeroPivotAction};
    use crate::numeric::factorize::factorize_multifrontal;
    use crate::sparse::csc::CscMatrix;
    use crate::symbolic::{symbolic_factorize, SupernodeParams};

    fn make_params() -> BunchKaufmanParams {
        BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            ..BunchKaufmanParams::default()
        }
    }

    fn check_solve(m: &CscMatrix, rhs: &[f64], tol: f64) {
        let sym = symbolic_factorize(m, &SupernodeParams::default()).unwrap();
        let params = make_params();
        let (factors, _) = factorize_multifrontal(m, &sym, &params).unwrap();
        let x = solve_sparse(&factors, rhs).unwrap();

        let n = m.n;
        let mut ax = vec![0.0; n];
        m.symv(&x, &mut ax);

        let mut res_sq = 0.0;
        let mut b_sq = 0.0;
        for i in 0..n { res_sq += (ax[i] - rhs[i]).powi(2); b_sq += rhs[i].powi(2); }
        let rel_res = if b_sq > 0.0 { (res_sq / b_sq).sqrt() } else { res_sq.sqrt() };
        assert!(rel_res < tol, "relative residual {:.2e} exceeds tolerance {:.2e}", rel_res, tol);
    }

    #[test]
    fn test_solve_diagonal() {
        let m = CscMatrix::from_triplets(3, &[0,1,2], &[0,1,2], &[2.0,3.0,5.0]).unwrap();
        check_solve(&m, &[4.0, 9.0, 25.0], 1e-14);
    }

    #[test]
    fn test_solve_tridiagonal() {
        let m = CscMatrix::from_triplets(3, &[0,1,1,2,2], &[0,0,1,1,2], &[2.0,-1.0,2.0,-1.0,2.0]).unwrap();
        check_solve(&m, &[1.0, 0.0, 1.0], 1e-13);
    }

    #[test]
    fn test_solve_kkt() {
        let m = CscMatrix::from_triplets(3, &[0,1,2,2,2], &[0,1,0,1,2], &[2.0,3.0,1.0,1.0,-1e-8]).unwrap();
        check_solve(&m, &[1.0, 2.0, 3.0], 1e-6);
    }

    #[test]
    fn test_solve_larger_spd() {
        let n = 5;
        let mut rows = Vec::new(); let mut cols = Vec::new(); let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i); cols.push(i); vals.push(4.0);
            if i+1 < n { rows.push(i+1); cols.push(i); vals.push(-1.0); }
        }
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        check_solve(&m, &(0..n).map(|i| (i+1) as f64).collect::<Vec<_>>(), 1e-13);
    }

    #[test]
    fn test_solve_indefinite() {
        let m = CscMatrix::from_triplets(2, &[0,1,1], &[0,0,1], &[1.0,2.0,1.0]).unwrap();
        check_solve(&m, &[5.0, 4.0], 1e-13);
    }

    #[test]
    fn test_solve_arrow_multi_supernode() {
        let m = CscMatrix::from_triplets(
            5, &[0,1,2,3,4,1,2,3,4], &[0,0,0,0,0,1,2,3,4],
            &[10.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],
        ).unwrap();
        check_solve(&m, &[1.0,2.0,3.0,4.0,5.0], 1e-12);
    }
}
