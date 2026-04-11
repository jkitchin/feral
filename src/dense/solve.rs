use crate::dense::factor::Factors;
use crate::dense::matrix::SymmetricMatrix;
use crate::error::FeralError;

/// Solve A·x = rhs using previously computed factors.
/// Full 7-step sequence per Section 2.9. No iterative refinement.
pub fn solve(factors: &Factors, rhs: &[f64]) -> Result<Vec<f64>, FeralError> {
    let n = factors.n;
    if rhs.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: rhs.len(),
        });
    }

    // Step 1: b̂ = D_eq · b
    let mut b_hat = vec![0.0; n];
    for i in 0..n {
        b_hat[i] = factors.d_eq[i] * rhs[i];
    }

    // Step 2: ŷ = Pᵀ · b̂ (apply BK permutation)
    // perm[i] = j means original row j → pivot position i
    // So y[i] = b_hat[perm[i]]
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = b_hat[factors.perm[i]];
    }

    // Step 3: z = L⁻¹ · ŷ (forward substitution, unit lower triangular)
    let mut z = y;
    forward_substitute(factors, &mut z);

    // Step 4: w = D_bk⁻¹ · z (D-block solve)
    let mut w = z;
    d_block_solve(factors, &mut w);

    // Step 5: v = L⁻ᵀ · w (backward substitution)
    let mut v = w;
    backward_substitute(factors, &mut v);

    // Step 6: x̂ = P · v (undo BK permutation)
    // x_hat[perm[i]] = v[i]
    let mut x_hat = vec![0.0; n];
    for i in 0..n {
        x_hat[factors.perm[i]] = v[i];
    }

    // Step 7: x = D_eq · x̂ (undo equilibration)
    let mut x = x_hat;
    for (xi, &di) in x.iter_mut().zip(factors.d_eq.iter()) {
        *xi *= di;
    }

    Ok(x)
}

/// Solve A·x = rhs with iterative refinement (Section 2.10).
/// Requires the original matrix to compute residuals.
pub fn solve_refined(
    matrix: &SymmetricMatrix,
    factors: &Factors,
    rhs: &[f64],
) -> Result<Vec<f64>, FeralError> {
    let n = factors.n;
    if rhs.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: rhs.len(),
        });
    }

    // Initial solve
    let mut x = solve(factors, rhs)?;

    let max_steps = 3;
    let n_sqrt = (n as f64).sqrt();

    for _ in 0..max_steps {
        // Compute residual: r = b - A·x
        let mut ax = vec![0.0; n];
        matrix.symv(&x, &mut ax);

        let mut r = vec![0.0; n];
        for i in 0..n {
            r[i] = rhs[i] - ax[i];
        }

        // Solve correction: δx = A⁻¹ r
        let dx = solve(factors, &r)?;

        // Check convergence: ||δx||₂ / ||x||₂ < macheps * sqrt(n)
        let dx_norm = norm2(&dx);
        let x_norm = norm2(&x);

        // Update: x = x + δx
        for i in 0..n {
            x[i] += dx[i];
        }

        let threshold = f64::EPSILON * n_sqrt;
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

/// Forward substitution: solve L·z = y where L is unit lower triangular.
fn forward_substitute(factors: &Factors, z: &mut [f64]) {
    let n = factors.n;
    let l = &factors.l;
    for j in 0..n {
        let z_j = z[j];
        for i in (j + 1)..n {
            z[i] -= l[j * n + i] * z_j;
        }
    }
}

/// Backward substitution: solve Lᵀ·v = w where L is unit lower triangular.
fn backward_substitute(factors: &Factors, v: &mut [f64]) {
    let n = factors.n;
    let l = &factors.l;
    for j in (0..n).rev() {
        let mut sum = 0.0;
        for i in (j + 1)..n {
            sum += l[j * n + i] * v[i];
        }
        v[j] -= sum;
    }
}

/// D-block solve: solve D_bk · w = z.
/// Handles both 1×1 and 2×2 blocks using the normalized formulation.
fn d_block_solve(factors: &Factors, w: &mut [f64]) {
    let n = factors.n;
    let mut k = 0;
    while k < n {
        if k + 1 < n && factors.d_subdiag[k] != 0.0 {
            // 2×2 block at (k, k+1)
            let a = factors.d_diag[k];
            let b = factors.d_subdiag[k];
            let c = factors.d_diag[k + 1];

            // Normalized formulation (faer's approach, Section 8.1 of research note)
            let b_inv = 1.0 / b;
            let ak = a * b_inv;
            let ck = c * b_inv;
            let denom = 1.0 / (ak * ck - 1.0);
            let z0k = w[k] * b_inv;
            let z1k = w[k + 1] * b_inv;
            w[k] = (ck * z0k - z1k) * denom;
            w[k + 1] = (ak * z1k - z0k) * denom;
            k += 2;
        } else {
            // 1×1 block
            let d = factors.d_diag[k];
            if d.abs() > f64::EPSILON * 1e-10 {
                w[k] /= d;
            }
            // If d is near-zero (ForceAccept case), leave w[k] as-is
            // (will be corrected by iterative refinement)
            k += 1;
        }
    }
}

/// L2 norm of a vector.
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
