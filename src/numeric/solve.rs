use crate::dense::solve::solve as dense_solve;
use crate::error::FeralError;
use super::factorize::SparseFactors;

/// Solve A·x = b using the sparse multifrontal factorization.
///
/// For each supernode, gathers the local RHS, uses the dense solver on the
/// full frontal system, then scatters the eliminated variables back and
/// updates the non-eliminated rows.
///
/// Steps:
/// 1. Permute RHS with AMD ordering
/// 2. For each supernode in postorder: solve the frontal system
/// 3. Unpermute the solution
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

    // Step 2: Use the dense solve on each supernode's full frontal system.
    // For a single-supernode case, this is equivalent to the dense solve.
    // For multi-supernode, we solve each frontal system using the dense
    // factors stored per-node.

    // Approach: use the dense solve function directly on each node's factors.
    // The dense solve handles L, D, L^T, and the internal BK permutation.
    // We just need to gather/scatter the RHS correctly.

    for node in &factors.node_factors {
        if node.ncol == 0 {
            continue;
        }

        // Gather local RHS from global y
        let local_rhs: Vec<f64> = node.row_indices.iter().map(|&gi| y[gi]).collect();

        // Solve using the dense solver
        let local_x = dense_solve(&node.dense_factors, &local_rhs)?;

        // Scatter back to global y
        for (li, &gi) in node.row_indices.iter().enumerate() {
            y[gi] = local_x[li];
        }
    }

    // Step 3: Unpermute: x[old] = y[new]
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
            3,
            &[0, 1, 2],
            &[0, 1, 2],
            &[2.0, 3.0, 5.0],
        )
        .unwrap();
        let rhs = vec![4.0, 9.0, 25.0];
        check_solve(&m, &rhs, 1e-14);
    }

    #[test]
    fn test_solve_tridiagonal() {
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[2.0, -1.0, 2.0, -1.0, 2.0],
        )
        .unwrap();
        let rhs = vec![1.0, 0.0, 1.0];
        check_solve(&m, &rhs, 1e-13);
    }

    #[test]
    fn test_solve_kkt() {
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 2, 2, 2],
            &[0, 1, 0, 1, 2],
            &[2.0, 3.0, 1.0, 1.0, -1e-8],
        )
        .unwrap();
        let rhs = vec![1.0, 2.0, 3.0];
        check_solve(&m, &rhs, 1e-6);
    }

    #[test]
    fn test_solve_larger_spd() {
        let n = 5;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(4.0);
            if i + 1 < n {
                rows.push(i + 1);
                cols.push(i);
                vals.push(-1.0);
            }
        }
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let rhs: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        check_solve(&m, &rhs, 1e-13);
    }

    #[test]
    fn test_solve_indefinite() {
        let m = CscMatrix::from_triplets(
            2,
            &[0, 1, 1],
            &[0, 0, 1],
            &[1.0, 2.0, 1.0],
        )
        .unwrap();
        let rhs = vec![5.0, 4.0];
        check_solve(&m, &rhs, 1e-13);
    }
}
