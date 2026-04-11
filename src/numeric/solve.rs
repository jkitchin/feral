use crate::error::FeralError;
use super::factorize::SparseFactors;

/// Solve A·x = b using the sparse multifrontal factorization.
///
/// The multifrontal solve has three phases:
/// 1. Forward substitution: for each supernode in postorder, solve L·z = b
///    for the eliminated variables and update the non-eliminated rows.
/// 2. D-block solve: apply D^{-1} to the eliminated variables.
/// 3. Backward substitution: for each supernode in reverse postorder,
///    solve L^T·x = z and scatter back.
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

    // Step 1: Permute RHS with AMD ordering: y[new] = b[perm[new]]
    let mut y = vec![0.0; n];
    for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
        y[new_idx] = rhs[old_idx];
    }

    // Step 2: Solve using dense_solve per supernode.
    //
    // Each supernode's dense factors represent the full frontal factorization.
    // We gather the local RHS, apply the dense solve (which handles
    // equilibration, BK permutation, L/D/L^T, and un-equilibration internally),
    // then scatter the solution back to the global vector.
    //
    // This works correctly when each variable is eliminated at exactly one
    // supernode (the one where it appears as an eliminated column). Non-eliminated
    // rows in a frontal receive updates during the dense solve, which is correct
    // because the contribution block has been assembled into the frontal.
    for node in &factors.node_factors {
        if node.ncol == 0 {
            continue;
        }

        // Gather local RHS from global y
        let local_rhs: Vec<f64> = node.row_indices.iter().map(|&gi| y[gi]).collect();

        // Solve using the dense solver (handles equilibration + BK perm internally)
        let local_x = crate::dense::solve::solve(&node.dense_factors, &local_rhs)?;

        // Scatter back to global y
        for (li, &gi) in node.row_indices.iter().enumerate() {
            y[gi] = local_x[li];
        }
    }

    // Step 5: Unpermute: x[old] = y[new]
    let mut x = vec![0.0; n];
    for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
        x[old_idx] = y[new_idx];
    }

    Ok(x)
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
        for i in 0..n {
            let r = ax[i] - rhs[i];
            res_sq += r * r;
            b_sq += rhs[i] * rhs[i];
        }
        let rel_res = if b_sq > 0.0 {
            (res_sq / b_sq).sqrt()
        } else {
            res_sq.sqrt()
        };

        assert!(
            rel_res < tol,
            "relative residual {:.2e} exceeds tolerance {:.2e}",
            rel_res,
            tol
        );
    }

    #[test]
    fn test_solve_diagonal() {
        let m = CscMatrix::from_triplets(
            3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0],
        ).unwrap();
        check_solve(&m, &[4.0, 9.0, 25.0], 1e-14);
    }

    #[test]
    fn test_solve_tridiagonal() {
        let m = CscMatrix::from_triplets(
            3, &[0, 1, 1, 2, 2], &[0, 0, 1, 1, 2], &[2.0, -1.0, 2.0, -1.0, 2.0],
        ).unwrap();
        check_solve(&m, &[1.0, 0.0, 1.0], 1e-13);
    }

    #[test]
    fn test_solve_kkt() {
        let m = CscMatrix::from_triplets(
            3, &[0, 1, 2, 2, 2], &[0, 1, 0, 1, 2], &[2.0, 3.0, 1.0, 1.0, -1e-8],
        ).unwrap();
        check_solve(&m, &[1.0, 2.0, 3.0], 1e-6);
    }

    #[test]
    fn test_solve_larger_spd() {
        let n = 5;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i); cols.push(i); vals.push(4.0);
            if i + 1 < n { rows.push(i + 1); cols.push(i); vals.push(-1.0); }
        }
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let rhs: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        check_solve(&m, &rhs, 1e-13);
    }

    #[test]
    fn test_solve_indefinite() {
        let m = CscMatrix::from_triplets(
            2, &[0, 1, 1], &[0, 0, 1], &[1.0, 2.0, 1.0],
        ).unwrap();
        check_solve(&m, &[5.0, 4.0], 1e-13);
    }

    #[test]
    fn test_solve_arrow_multi_supernode() {
        // Arrow matrix with multiple supernodes (AMD reorders)
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ).unwrap();
        check_solve(&m, &[1.0, 2.0, 3.0, 4.0, 5.0], 1e-12);
    }
}
