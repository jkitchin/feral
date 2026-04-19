#![allow(clippy::needless_range_loop)]
use super::factorize::SparseFactors;
use crate::error::FeralError;
use crate::scaling::ScalingInfo;
use crate::sparse::csc::CscMatrix;

/// Solve A·x = b using the sparse multifrontal factorization.
///
/// Three phases matching the multifrontal factorization:
/// 1. Forward substitution: L-solve through supernodes (postorder)
/// 2. D-block solve: D^{-1} for eliminated pivots at each node
/// 3. Backward substitution: L^T-solve through supernodes (reverse postorder)
///
/// # MC64 scaling (Phase 2.2.1 Step 7)
///
/// When `factors.scaling_info != ScalingInfo::NotApplied`, the
/// factors represent `M = D · A · D` with `D = diag(factors.scaling)`,
/// not the user's original `A`. To solve `A · x = b` the user actually
/// wants, we bracket the core solve with a symmetric congruence:
///
/// ```text
///     A · x = b
///     (D^-1 · M · D^-1) · x = b
///     M · (D^-1 · x) = D · b        // left-multiply by D
///     M · y          = D · b        // let y = D^-1 · x
///     y = core_solve(D · b)
///     x = D · y                      // recover x
/// ```
///
/// Note the **same** `D` vector is applied on both ends, not its
/// inverse — the `D^-1` cancels out algebraically. Intuition:
/// pre-scaling the RHS by `D` compensates for the pre-scaling that
/// assembly-time baked into the factors, and post-scaling by `D`
/// maps the intermediate `y` back into the user coordinate system.
///
/// When `ScalingInfo::NotApplied`, the scaling vector is all ones
/// and the pre/post-scale passes are skipped as a fast path.
pub fn solve_sparse(factors: &SparseFactors, rhs: &[f64]) -> Result<Vec<f64>, FeralError> {
    let n = factors.n;
    if n == 0 && rhs.is_empty() {
        return Ok(Vec::new());
    }
    let mut x = vec![0.0; n];
    let mut ws = SolveWorkspace::for_factors(factors);
    solve_sparse_into_ws(factors, rhs, &mut x, &mut ws)?;
    Ok(x)
}

/// Workspace holding the per-call scratch buffers used by the sparse
/// solve. Allowing the caller to own this lets us amortize the
/// allocations across many solves — see `solve_sparse_refined`, which
/// performs up to 11 solves per call (1 initial + 10 refinement steps)
/// against the same factors.
struct SolveWorkspace {
    /// Permuted RHS / working solution vector, length `n`.
    y: Vec<f64>,
    /// Per-supernode gather/scatter buffer, length `max_nrow`.
    w: Vec<f64>,
    /// Scaled RHS storage when MC64 scaling is active, length `n`.
    /// Empty when no scaling is applied (the `solve_sparse` fast path).
    scaled_rhs: Vec<f64>,
}

impl SolveWorkspace {
    fn for_factors(factors: &SparseFactors) -> Self {
        let n = factors.n;
        let max_nrow = factors
            .node_factors
            .iter()
            .map(|node| node.frontal_factors.nrow)
            .max()
            .unwrap_or(0);
        let scaled_rhs_len = if matches!(factors.scaling_info, ScalingInfo::NotApplied) {
            0
        } else {
            n
        };
        Self {
            y: vec![0.0; n],
            w: vec![0.0; max_nrow],
            scaled_rhs: vec![0.0; scaled_rhs_len],
        }
    }
}

fn solve_sparse_into_ws(
    factors: &SparseFactors,
    rhs: &[f64],
    x_out: &mut [f64],
    ws: &mut SolveWorkspace,
) -> Result<(), FeralError> {
    let n = factors.n;
    if rhs.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: rhs.len(),
        });
    }
    if x_out.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: x_out.len(),
        });
    }
    if n == 0 {
        return Ok(());
    }

    // Pre-scale the RHS (user-order) in preparation for the core
    // solve. `NotApplied` ⇒ `scaling == [1.0; n]`, so the multiply
    // would be a no-op; skip it for the happy path.
    let needs_scaling = !matches!(factors.scaling_info, ScalingInfo::NotApplied);
    let rhs_for_core: &[f64] = if needs_scaling {
        for i in 0..n {
            ws.scaled_rhs[i] = rhs[i] * factors.scaling[i];
        }
        &ws.scaled_rhs
    } else {
        rhs
    };

    solve_sparse_core_into(factors, rhs_for_core, x_out, &mut ws.y, &mut ws.w);

    // Post-scale the solution with the same vector (not its inverse;
    // see the docstring math above).
    if needs_scaling {
        for i in 0..n {
            x_out[i] *= factors.scaling[i];
        }
    }

    Ok(())
}

/// Core sparse solve: runs forward-sub, D-solve, backward-sub on an
/// RHS that is assumed to already be in the pre-scaled coordinate
/// system of `M = D · A · D`. Callers other than `solve_sparse` (e.g.,
/// the refinement loop's correction solve) go through `solve_sparse`
/// itself so the pre/post-scale wrapping stays in one place.
///
/// `y_buf` (length `n`) and `w_buf` (length `max_nrow`) are caller-
/// owned scratch so refinement can amortize them across iterations.
fn solve_sparse_core_into(
    factors: &SparseFactors,
    rhs: &[f64],
    x_out: &mut [f64],
    y_buf: &mut [f64],
    w_buf: &mut [f64],
) {
    let n = factors.n;
    let y = &mut y_buf[..n];

    // Permute RHS with AMD ordering: y[new] = b[perm[new]]
    for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
        y[new_idx] = rhs[old_idx];
    }

    // Phase 1: Forward substitution (postorder)
    //
    // Phase 2.3 Step 6: iterate over the `nelim` actually-eliminated
    // pivots, not `ncol` (which is the *attempted* count and may be
    // larger when the kernel delayed pivots to an ancestor). `ff.l` is
    // sized `nrow × nelim`, so bounding the outer loop by `ncol` would
    // read past the end of L on any node that delayed columns.
    for node in &factors.node_factors {
        let ff = &node.frontal_factors;
        let nelim = ff.nelim;
        let nrow = ff.nrow;
        if nelim == 0 {
            continue;
        }

        // Gather and apply BK permutation. The gather overwrites every
        // entry in `[0..nrow)`, so no zeroing is needed despite the
        // shared buffer.
        let w = &mut w_buf[..nrow];
        for i in 0..nrow {
            w[i] = y[node.row_indices[ff.perm[i]]];
        }

        // L-solve: for each eliminated column j, update rows below
        for j in 0..nelim {
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
        let nelim = ff.nelim;
        let nrow = ff.nrow;
        if nelim == 0 {
            continue;
        }

        // Gather and apply BK permutation
        let w = &mut w_buf[..nrow];
        for i in 0..nrow {
            w[i] = y[node.row_indices[ff.perm[i]]];
        }

        // D-block solve over the `nelim` eliminated pivots. `d_diag`
        // and `d_subdiag` are sized `nelim`, so bounding by `ncol`
        // would run off the end on any node that delayed columns.
        // Pivots that were force-accepted as zero during factorization
        // are skipped — see dev/plans/threshold-mismatch-fix.md.
        let mut k = 0;
        while k < nelim {
            if k + 1 < nelim && ff.d_subdiag[k] != 0.0 {
                let a = ff.d_diag[k];
                let b = ff.d_subdiag[k];
                let c = ff.d_diag[k + 1];
                let det = a * c - b * b;

                if det.abs() > ff.zero_tol_2x2 {
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
                        w[k] = (c * z1 - b * z2) / det;
                        w[k + 1] = (a * z2 - b * z1) / det;
                    }
                }
                // else: 2×2 block force-accepted as singular; leave w[k], w[k+1]
                k += 2;
            } else {
                if ff.d_diag[k].abs() > ff.zero_tol {
                    w[k] /= ff.d_diag[k];
                }
                // else: pivot force-accepted as zero; leave w[k] alone
                k += 1;
            }
        }

        // Undo BK permutation and scatter back
        for i in 0..nrow {
            y[node.row_indices[ff.perm[i]]] = w[i];
        }
    }

    // Phase 3: Backward substitution (reverse postorder). Bounded by
    // `nelim` for the same reason as the forward sweep: L has `nelim`
    // columns and indexing by `ncol` would walk past the end on nodes
    // that delayed pivots.
    for node in factors.node_factors.iter().rev() {
        let ff = &node.frontal_factors;
        let nelim = ff.nelim;
        let nrow = ff.nrow;
        if nelim == 0 {
            continue;
        }

        // Gather and apply BK permutation
        let w = &mut w_buf[..nrow];
        for i in 0..nrow {
            w[i] = y[node.row_indices[ff.perm[i]]];
        }

        // L^T-solve: for each eliminated column j (reverse order)
        for j in (0..nelim).rev() {
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
    for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
        x_out[old_idx] = y[new_idx];
    }
}

/// Solve A·x = rhs using the sparse factorization with iterative refinement.
///
/// Mirrors `crate::dense::solve::solve_refined` for the multifrontal path.
/// Per FERAL-PROJECT-SPEC.md §1709, this is the Phase 1b solve convention:
/// because `ZeroPivotAction::ForceAccept` is the default, an unrefined solve
/// can leave a non-trivial residual on near-singular pivots, and refinement
/// recovers machine precision in 0–3 steps for well-conditioned matrices.
///
/// **Best-iterate:** tracks the smallest `||r||₂` seen across all
/// refinement steps and returns the corresponding `x`. On rank-deficient
/// matrices where ForceAccept produced a wrong `A⁻¹`, the correction
/// `dx = A⁻¹·r` can amplify error; tracking the best iterate guarantees
/// the returned `x` is no worse than the unrefined `solve_sparse()` output.
///
/// Convergence test: stop when `||r||₂ / ||b||₂ < ε·√n` (we've reached
/// machine precision) or after 10 steps. 10 is MUMPS's ICNTL(10)
/// default; below that some near-rank-deficient KKT matrices
/// (CERI651C/ELS, HAHN1, MEYER3NE) bounce in and out of the machine-
/// precision basin before settling, and the best-iterate tracker below
/// guarantees no regression from the extra steps.
///
/// A prior version of this routine used a `||δx||/||x|| < ε·√n`
/// convergence test, but that fires prematurely on matrices where
/// ForceAccept produced a non-contractive correction — the iterate
/// stops updating (tiny δx) without the residual having actually
/// dropped into the target basin. Residual-based termination is
/// honest about "are we done yet."
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

    let mut ws = SolveWorkspace::for_factors(factors);
    let mut x = vec![0.0; n];
    solve_sparse_into_ws(factors, rhs, &mut x, &mut ws)?;

    // Initial residual: compute A·x directly into r, then negate-add.
    let mut r = vec![0.0; n];
    matrix.symv(&x, &mut r);
    for i in 0..n {
        r[i] = rhs[i] - r[i];
    }
    let mut r_norm = norm2(&r);

    let mut best_x = x.clone();
    let mut best_r_norm = r_norm;
    let mut stagnant_count: usize = 0;
    let mut dx = vec![0.0; n];

    // Phase 2.5 (2026-04-18) tuning: profile_sparse showed refinement
    // was running 10 iterations on most KKT matrices because the
    // `ε·√n` relative target is below double-precision floor noise.
    // The 10x multiplier on top of the bare solve drove the 1.82×
    // SSIDS solve-time gap on the 154k-matrix bench.
    //
    // Strategy: keep `max_steps = 10` for the worst-case ill-conditioned
    // matrices, but exit after `max_stagnant_steps` consecutive steps
    // fail to improve the best residual. A 2-strike rule preserves the
    // bouncing-into-basin behavior on borderline KKT matrices (which a
    // single-strike exit kills) while still capping the easy-case cost.
    // Bench evidence (cap=2 / cap=3 / two-tier / 1-strike / 2-strike)
    // is in `dev/journal/2026-04-18-06.org`.
    let max_steps = 10;
    let max_stagnant_steps = 2;
    let n_sqrt = (n as f64).sqrt();
    let threshold = f64::EPSILON * n_sqrt;
    let divergence_factor = 100.0;
    let b_norm = norm2(rhs);
    // Target is a RELATIVE residual: ||r||/||b|| < ε·√n. When ||b|| = 0
    // the true answer is x = 0 and r = -A·x; we target ||r|| < threshold
    // directly in that case.
    let relative_reached = |r_norm: f64| -> bool {
        if b_norm > 0.0 {
            r_norm < threshold * b_norm
        } else {
            r_norm < threshold
        }
    };

    for _step in 0..max_steps {
        if relative_reached(best_r_norm) {
            break;
        }

        solve_sparse_into_ws(factors, &r, &mut dx, &mut ws)?;
        for i in 0..n {
            x[i] += dx[i];
        }

        // Recompute residual in place: r = b - A·x.
        matrix.symv(&x, &mut r);
        for i in 0..n {
            r[i] = rhs[i] - r[i];
        }
        r_norm = norm2(&r);

        let improved = r_norm < best_r_norm;
        if improved {
            best_r_norm = r_norm;
            best_x.copy_from_slice(&x);
            stagnant_count = 0;
        } else {
            stagnant_count += 1;
        }

        if r_norm > best_r_norm * divergence_factor {
            break;
        }
        // Plateau: `max_stagnant_steps` consecutive non-improving
        // steps means refinement has bottomed out (floor noise or
        // ill-conditioning) — further iterations will not help.
        // A single non-improving step is allowed because some KKT
        // matrices oscillate into a better basin on the next step.
        if stagnant_count >= max_stagnant_steps {
            break;
        }
    }

    Ok(best_x)
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

    fn make_params() -> crate::numeric::factorize::NumericParams {
        crate::numeric::factorize::NumericParams::with_bk(BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            ..BunchKaufmanParams::default()
        })
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
            res_sq += (ax[i] - rhs[i]).powi(2);
            b_sq += rhs[i].powi(2);
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
        let m = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
        check_solve(&m, &[4.0, 9.0, 25.0], 1e-14);
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
        check_solve(&m, &[1.0, 0.0, 1.0], 1e-13);
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
        check_solve(&m, &[1.0, 2.0, 3.0], 1e-6);
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
        check_solve(
            &m,
            &(0..n).map(|i| (i + 1) as f64).collect::<Vec<_>>(),
            1e-13,
        );
    }

    #[test]
    fn test_solve_indefinite() {
        let m = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[1.0, 2.0, 1.0]).unwrap();
        check_solve(&m, &[5.0, 4.0], 1e-13);
    }

    #[test]
    fn test_solve_arrow_multi_supernode() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        check_solve(&m, &[1.0, 2.0, 3.0, 4.0, 5.0], 1e-12);
    }
}
