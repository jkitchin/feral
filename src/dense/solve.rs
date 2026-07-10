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

    let max_steps = 10;
    let n_sqrt = (n as f64).sqrt();
    let threshold = f64::EPSILON * n_sqrt;
    // Bail out if the residual blows up far beyond the best seen
    let divergence_factor = 100.0;
    let b_norm = norm2(rhs).max(1.0);
    let target_r = threshold * b_norm;

    for _ in 0..max_steps {
        // Already at machine precision? Stop.
        if best_r_norm < target_r {
            break;
        }

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

        // Step
        x = x_new;
        r = r_new;
        r_norm = r_new_norm;

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
/// The skip decision must match the factor side exactly, otherwise a pivot the
/// factorization validly accepted and stored is silently skipped at solve time
/// (wrong solution, no error, no flag).
///
/// **1×1 pivots (issue #116):** skip iff the pivot was *force-zeroed* — the
/// factor sets `d_diag[k]` to exactly `0.0` and clears the L column, the only
/// path that produces a pivot the solve must leave untouched. Every other
/// accepted pivot is *live* (its L column participates in the elimination) and
/// must be divided by, including small-but-nonzero ones from rook rescue, the
/// static/PerturbToEps floor, or the F-01 rank-deficiency band. The previous
/// `|d| > zero_tol` gate skipped those live tiny pivots too, silently dropping
/// an O(1/d) term (`needs_refinement` is set so refinement can recover it).
///
/// **Finding D4 (2×2 pivots):** the gate previously used the *naive*
/// `a*c - b*b` against the *absolute* `zero_tol_2x2 ≈ EPS²`, while the factor
/// side accepts via the *scale-invariant* SSIDS floor (`ssids_det_floor_fail`)
/// — so a well-conditioned block at small absolute scale (true `|det|` below
/// `EPS²`) was accepted by the factor but skipped by the solve. Both sides now
/// call the shared `ssids_det_floor_fail`, so a block the factor inverts the
/// solve inverts. (`zero_tol_2x2` is retained on `Factors` for the legacy
/// `count_2x2_inertia` accounting but no longer gates the solve.)
fn d_block_solve(factors: &Factors, w: &mut [f64]) {
    let n = factors.n;
    let mut k = 0;
    while k < n {
        if k + 1 < n && factors.d_subdiag[k] != 0.0 {
            // 2×2 block at (k, k+1)
            let a = factors.d_diag[k];
            let b = factors.d_subdiag[k];
            let c = factors.d_diag[k + 1];

            if !crate::dense::factor::ssids_det_floor_fail(a, b, c) {
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
            // else: 2×2 block rejected by the shared SSIDS floor (the
            // factor side would not have stored it as invertible); leave
            // w[k], w[k+1] untouched.
            k += 2;
        } else {
            // 1×1 block. Skip iff the pivot was force-zeroed (its L column
            // cleared and `d_diag` set to exactly 0.0 — the only skip outcome
            // the factor produces). A small-but-nonzero *live* pivot (rook
            // rescue, static/PerturbToEps floor, F-01 band) keeps its L column
            // and MUST be divided by — the old `|d| > zero_tol` gate wrongly
            // skipped rook-accepted pivots with `|d| ≤ zero_tol`, silently
            // dropping an O(1/d) term from the solution (issue #116).
            let d = factors.d_diag[k];
            if d != 0.0 {
                w[k] /= d;
            }
            k += 1;
        }
    }
}

/// L2 norm of a vector.
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #116: the D-block solve must divide by a *live* small-but-nonzero
    /// 1×1 pivot (rook rescue / static floor / F-01 band keep the L column and
    /// can leave `d_diag` below `zero_tol`), and skip only a *force-zeroed*
    /// pivot (`d_diag == 0.0` exactly, L cleared). The old `|d| > zero_tol`
    /// gate skipped the live tiny pivot too, silently dropping an O(1/d) term
    /// from the solution.
    #[test]
    fn d_block_solve_divides_live_tiny_pivot_skips_forced_zero() {
        // Position 0: live tiny pivot d = 1e-16 (< zero_tol = 1e-13) → divide.
        // Position 1: force-zeroed d = 0.0 → skip (leave w unchanged).
        let factors = Factors {
            n: 2,
            l: vec![1.0, 0.0, 0.0, 1.0], // d_block_solve ignores L
            d_diag: vec![1e-16, 0.0],
            d_subdiag: vec![0.0, 0.0],
            perm: vec![0, 1],
            perm_inv: vec![0, 1],
            d_eq: vec![1.0, 1.0],
            needs_refinement: true,
            zero_tol: 1e-13,
            zero_tol_2x2: 1e-26,
        };
        let mut w = vec![2.0, 5.0];
        d_block_solve(&factors, &mut w);
        assert_eq!(
            w[0],
            2.0 / 1e-16,
            "live tiny pivot must be divided, not skipped"
        );
        assert_eq!(w[1], 5.0, "force-zeroed pivot (d == 0) must be skipped");
    }
}
