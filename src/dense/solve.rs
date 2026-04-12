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
///
/// **Best-iterate:** tracks the smallest `||r||₂` seen across all
/// refinement steps and returns the corresponding `x`. On rank-deficient
/// matrices where ForceAccept produced a wrong `A⁻¹`, the correction
/// `dx = A⁻¹·r` can amplify error; tracking the best iterate guarantees
/// the returned `x` is no worse than the unrefined `solve()` output.
/// Intermediate steps are still allowed to be non-monotone — extreme
/// scaling cases sometimes need a transient bump before subsequent steps
/// reduce the residual below the unrefined baseline.
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

    // Initial residual
    let mut r = vec![0.0; n];
    let mut ax = vec![0.0; n];
    matrix.symv(&x, &mut ax);
    for i in 0..n {
        r[i] = rhs[i] - ax[i];
    }
    let mut r_norm = norm2(&r);

    // Track the best iterate seen so far
    let mut best_x = x.clone();
    let mut best_r_norm = r_norm;

    let max_steps = 3;
    let n_sqrt = (n as f64).sqrt();
    let threshold = f64::EPSILON * n_sqrt;
    // Bail out if the residual blows up far beyond the best seen
    let divergence_factor = 100.0;

    for _ in 0..max_steps {
        // Solve correction: δx = A⁻¹ r
        let dx = solve(factors, &r)?;

        // Candidate x_new = x + δx
        let mut x_new = x.clone();
        for i in 0..n {
            x_new[i] += dx[i];
        }

        // Candidate residual
        let mut r_new = vec![0.0; n];
        let mut ax_new = vec![0.0; n];
        matrix.symv(&x_new, &mut ax_new);
        for i in 0..n {
            r_new[i] = rhs[i] - ax_new[i];
        }
        let r_new_norm = norm2(&r_new);

        // Track best
        if r_new_norm < best_r_norm {
            best_r_norm = r_new_norm;
            best_x = x_new.clone();
        }

        // Convergence check on δx, before stepping
        let dx_norm = norm2(&dx);
        let x_norm = norm2(&x_new);

        // Step
        x = x_new;
        r = r_new;
        r_norm = r_new_norm;

        if x_norm > 0.0 {
            if dx_norm / x_norm < threshold {
                break;
            }
        } else if dx_norm < threshold {
            break;
        }

        // Diverging hard? Stop trying.
        if r_norm > best_r_norm * divergence_factor {
            break;
        }
    }

    Ok(best_x)
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
///
/// Pivots that were force-accepted as numerically zero during factorization
/// (`|d| <= factors.zero_tol` for 1×1, `|det| <= factors.zero_tol_2x2` for
/// 2×2) are skipped — `w[k]` is left untouched, producing a least-squares-
/// like solution where the corresponding row was rank-deficient. Dividing by
/// such pivots produces catastrophic error; see dev/plans/threshold-mismatch-fix.md.
fn d_block_solve(factors: &Factors, w: &mut [f64]) {
    let n = factors.n;
    let mut k = 0;
    while k < n {
        if k + 1 < n && factors.d_subdiag[k] != 0.0 {
            // 2×2 block at (k, k+1)
            let a = factors.d_diag[k];
            let b = factors.d_subdiag[k];
            let c = factors.d_diag[k + 1];
            let det = a * c - b * b;

            if det.abs() > factors.zero_tol_2x2 {
                // Normalized formulation (faer's approach)
                let b_inv = 1.0 / b;
                let ak = a * b_inv;
                let ck = c * b_inv;
                let denom = 1.0 / (ak * ck - 1.0);
                let z0k = w[k] * b_inv;
                let z1k = w[k + 1] * b_inv;
                w[k] = (ck * z0k - z1k) * denom;
                w[k + 1] = (ak * z1k - z0k) * denom;
            }
            // else: 2×2 block was force-accepted as singular; leave w[k], w[k+1] alone
            k += 2;
        } else {
            // 1×1 block
            let d = factors.d_diag[k];
            if d.abs() > factors.zero_tol {
                w[k] /= d;
            }
            // else: pivot was force-accepted as zero; leave w[k] alone
            k += 1;
        }
    }
}

/// L2 norm of a vector.
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
