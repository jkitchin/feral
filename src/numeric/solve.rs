#![allow(clippy::needless_range_loop)]
use super::condition::{estimate_inverse_norm_1, matrix_norm_1};
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

/// Workspace for `solve_sparse_many_into`. Sized for `nrhs` columns
/// at construction time. Reuse across calls with the same `nrhs`
/// avoids reallocation on the IPM hot path.
///
/// See `dev/research/multi-rhs.md` (F1.0) for the layout decisions
/// — y/w/scaled_rhs are all column-major and widened by a factor
/// of `nrhs` relative to the single-RHS `SolveWorkspace`.
pub struct SolveManyWorkspace {
    /// Permuted RHS / working solution vector, length `n * nrhs`,
    /// column-major (column `c` lives at `[c*n .. (c+1)*n]`).
    y: Vec<f64>,
    /// Per-supernode gather/scatter buffer, length `max_nrow * nrhs`,
    /// column-major.
    w: Vec<f64>,
    /// Pre-scaled RHS storage when MC64 scaling is active, length
    /// `n * nrhs`. Empty when no scaling is applied.
    scaled_rhs: Vec<f64>,
    /// `nrhs` baked in at construction time. Re-using the workspace
    /// for a different `nrhs` is a logic error and is checked.
    nrhs: usize,
    /// `n` baked in for the dimension check.
    n: usize,
}

impl SolveManyWorkspace {
    /// Allocate a workspace sized for `nrhs` solves against `factors`.
    pub fn for_factors(factors: &SparseFactors, nrhs: usize) -> Self {
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
            n * nrhs
        };
        Self {
            y: vec![0.0; n * nrhs],
            w: vec![0.0; max_nrow * nrhs],
            scaled_rhs: vec![0.0; scaled_rhs_len],
            nrhs,
            n,
        }
    }
}

/// Solve `A · X = B` for `X`, where `B` and `X` are column-major
/// `n × nrhs` matrices stored as flat slices of length `n * nrhs`.
///
/// Equivalent to `nrhs` independent calls to `solve_sparse`, but
/// shares workspace and the supernodal traversal across columns.
/// At small `nrhs` (1–8) this saves the per-call allocation; at
/// larger `nrhs` the per-supernode kernels can amortize the
/// gather/scatter overhead across columns.
///
/// `nrhs == 0` returns `Ok(Vec::new())`. `nrhs == 1` is a thin
/// wrapper around `solve_sparse_into_ws`.
///
/// See `dev/plans/kkt-feature-gaps.md` F1 for the design and
/// `dev/research/multi-rhs.md` for the layout decisions.
pub fn solve_sparse_many(
    factors: &SparseFactors,
    rhs: &[f64],
    nrhs: usize,
) -> Result<Vec<f64>, FeralError> {
    let n = factors.n;
    if nrhs == 0 {
        return Ok(Vec::new());
    }
    let mut x = vec![0.0; n * nrhs];
    let mut ws = SolveManyWorkspace::for_factors(factors, nrhs);
    solve_sparse_many_into(factors, rhs, nrhs, &mut x, &mut ws)?;
    Ok(x)
}

/// In-place form of `solve_sparse_many` using a caller-owned
/// workspace. The workspace must have been constructed with the
/// same `nrhs` and `factors.n`; otherwise returns
/// `FeralError::DimensionMismatch`.
pub fn solve_sparse_many_into(
    factors: &SparseFactors,
    rhs: &[f64],
    nrhs: usize,
    x_out: &mut [f64],
    ws: &mut SolveManyWorkspace,
) -> Result<(), FeralError> {
    let n = factors.n;
    if nrhs == 0 {
        return Ok(());
    }
    if ws.nrhs != nrhs || ws.n != n {
        return Err(FeralError::DimensionMismatch {
            expected: n * nrhs,
            got: ws.n * ws.nrhs,
        });
    }
    if rhs.len() != n * nrhs {
        return Err(FeralError::DimensionMismatch {
            expected: n * nrhs,
            got: rhs.len(),
        });
    }
    if x_out.len() != n * nrhs {
        return Err(FeralError::DimensionMismatch {
            expected: n * nrhs,
            got: x_out.len(),
        });
    }
    if n == 0 {
        return Ok(());
    }

    // Pre-scale every column by D (MC64 congruence). Skipped when
    // ScalingInfo::NotApplied (the scaling vector is all-ones).
    let needs_scaling = !matches!(factors.scaling_info, ScalingInfo::NotApplied);
    let rhs_for_core: &[f64] = if needs_scaling {
        for c in 0..nrhs {
            let off = c * n;
            for i in 0..n {
                ws.scaled_rhs[off + i] = rhs[off + i] * factors.scaling[i];
            }
        }
        &ws.scaled_rhs
    } else {
        rhs
    };

    solve_sparse_core_many_into(factors, rhs_for_core, nrhs, x_out, &mut ws.y, &mut ws.w);

    // Post-scale every column with the same D vector (see
    // `solve_sparse_into_ws` for the cancellation argument).
    if needs_scaling {
        for c in 0..nrhs {
            let off = c * n;
            for i in 0..n {
                x_out[off + i] *= factors.scaling[i];
            }
        }
    }

    Ok(())
}

/// Multi-RHS core solve: forward-sub, D-solve, backward-sub on
/// `nrhs` columns laid out column-major in `rhs`. Mirrors
/// `solve_sparse_core_into` with the inner update loops widened
/// to `nrhs` columns. The single-RHS path
/// (`solve_sparse_core_into`) is preserved unchanged so the
/// iterative-refinement code path stays on a tested code path.
fn solve_sparse_core_many_into(
    factors: &SparseFactors,
    rhs: &[f64],
    nrhs: usize,
    x_out: &mut [f64],
    y_buf: &mut [f64],
    w_buf: &mut [f64],
) {
    let n = factors.n;
    let y = &mut y_buf[..n * nrhs];

    // Permute every column of the RHS: y[c, new] = b[c, perm[new]]
    for c in 0..nrhs {
        let src_off = c * n;
        let dst_off = c * n;
        for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
            y[dst_off + new_idx] = rhs[src_off + old_idx];
        }
    }

    // Phase 1: Forward substitution (postorder).
    for node in &factors.node_factors {
        let ff = &node.frontal_factors;
        let nelim = ff.nelim;
        let nrow = ff.nrow;
        if nelim == 0 {
            continue;
        }

        let w = &mut w_buf[..nrow * nrhs];
        // Gather every column with the BK permutation applied.
        for c in 0..nrhs {
            let w_col = &mut w[c * nrow..(c + 1) * nrow];
            let y_col = &y[c * n..(c + 1) * n];
            for i in 0..nrow {
                w_col[i] = y_col[node.row_indices[ff.perm[i]]];
            }
        }

        // L-solve: for each eliminated column j, update rows below.
        // Inner loop is length-`nrhs` axpy; compiler auto-vectorizes
        // for small constant or runtime `nrhs`.
        for j in 0..nelim {
            for i in (j + 1)..nrow {
                let l_ij = ff.l[j * nrow + i];
                for c in 0..nrhs {
                    let off = c * nrow;
                    w[off + i] -= l_ij * w[off + j];
                }
            }
        }

        // Scatter back, undoing BK permutation.
        for c in 0..nrhs {
            let w_col = &w[c * nrow..(c + 1) * nrow];
            let y_col = &mut y[c * n..(c + 1) * n];
            for i in 0..nrow {
                y_col[node.row_indices[ff.perm[i]]] = w_col[i];
            }
        }
    }

    // Phase 2: D-block solve. Per-column, since the 2×2 logic
    // depends on values inside the column.
    for node in &factors.node_factors {
        let ff = &node.frontal_factors;
        let nelim = ff.nelim;
        let nrow = ff.nrow;
        if nelim == 0 {
            continue;
        }

        let w = &mut w_buf[..nrow * nrhs];
        for c in 0..nrhs {
            let w_col = &mut w[c * nrow..(c + 1) * nrow];
            let y_col = &y[c * n..(c + 1) * n];
            for i in 0..nrow {
                w_col[i] = y_col[node.row_indices[ff.perm[i]]];
            }
        }

        for c in 0..nrhs {
            let w_col = &mut w[c * nrow..(c + 1) * nrow];
            let mut k = 0;
            while k < nelim {
                if k + 1 < nelim && ff.d_subdiag[k] != 0.0 {
                    let a = ff.d_diag[k];
                    let b = ff.d_subdiag[k];
                    let cc = ff.d_diag[k + 1];
                    let det = a * cc - b * b;

                    if det.abs() > ff.zero_tol_2x2 {
                        let z1 = w_col[k];
                        let z2 = w_col[k + 1];
                        if b.abs() > f64::EPSILON * (a.abs() + cc.abs()).max(1.0) {
                            let ak = a / b;
                            let ck = cc / b;
                            let denom = 1.0 / (ak * ck - 1.0);
                            let z1k = z1 / b;
                            let z2k = z2 / b;
                            w_col[k] = (ck * z1k - z2k) * denom;
                            w_col[k + 1] = (ak * z2k - z1k) * denom;
                        } else {
                            w_col[k] = (cc * z1 - b * z2) / det;
                            w_col[k + 1] = (a * z2 - b * z1) / det;
                        }
                    }
                    // else: 2×2 block force-accepted as singular; leave as-is.
                    k += 2;
                } else {
                    if ff.d_diag[k].abs() > ff.zero_tol {
                        w_col[k] /= ff.d_diag[k];
                    }
                    // else: pivot force-accepted as zero; leave as-is.
                    k += 1;
                }
            }
        }

        for c in 0..nrhs {
            let w_col = &w[c * nrow..(c + 1) * nrow];
            let y_col = &mut y[c * n..(c + 1) * n];
            for i in 0..nrow {
                y_col[node.row_indices[ff.perm[i]]] = w_col[i];
            }
        }
    }

    // Phase 3: Backward substitution (reverse postorder).
    for node in factors.node_factors.iter().rev() {
        let ff = &node.frontal_factors;
        let nelim = ff.nelim;
        let nrow = ff.nrow;
        if nelim == 0 {
            continue;
        }

        let w = &mut w_buf[..nrow * nrhs];
        for c in 0..nrhs {
            let w_col = &mut w[c * nrow..(c + 1) * nrow];
            let y_col = &y[c * n..(c + 1) * n];
            for i in 0..nrow {
                w_col[i] = y_col[node.row_indices[ff.perm[i]]];
            }
        }

        // L^T-solve: per column, dot the trailing entries of column j
        // of L with the trailing entries of `w_col` and subtract.
        for j in (0..nelim).rev() {
            for c in 0..nrhs {
                let off = c * nrow;
                let mut sum = 0.0;
                for i in (j + 1)..nrow {
                    sum += ff.l[j * nrow + i] * w[off + i];
                }
                w[off + j] -= sum;
            }
        }

        for c in 0..nrhs {
            let w_col = &w[c * nrow..(c + 1) * nrow];
            let y_col = &mut y[c * n..(c + 1) * n];
            for i in 0..nrow {
                y_col[node.row_indices[ff.perm[i]]] = w_col[i];
            }
        }
    }

    // Unpermute every column: x[c, old] = y[c, new].
    for c in 0..nrhs {
        let src_off = c * n;
        let dst_off = c * n;
        for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
            x_out[dst_off + old_idx] = y[src_off + new_idx];
        }
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
    let (x, _) = solve_sparse_refined_core(matrix, factors, rhs, false)?;
    Ok(x)
}

/// Per-step diagnostic data emitted by
/// [`solve_sparse_refined_with_diagnostics`].
///
/// Step 0 is the unrefined initial solve; subsequent steps are refinement
/// iterations. The number of steps is bounded by the refinement cap
/// (currently 10 + 1 initial = 11) and may exit early on convergence,
/// divergence, or plateau.
#[derive(Debug, Clone, Copy)]
pub struct RefinementStep {
    /// Step index (0 = unrefined solve, 1.. = refinement iterations).
    pub step: usize,
    /// `||r||_2` where `r = b - A·x` after this step.
    pub residual_2norm: f64,
    /// `||r||_2 / ||b||_2`. Falls back to `residual_2norm` when
    /// `||b|| = 0` (the trivial RHS case).
    pub relative_residual: f64,
    /// Skeel-style forward-error bound estimate
    /// `kappa_1_est * relative_residual` — a conservative upper bound
    /// on the relative forward error `||x - x_true||_∞ / ||x_true||_∞`
    /// for iterative refinement (Skeel 1980; Higham 2002 §15).
    /// Constant `kappa_1_est` is shared across all steps within one
    /// refinement run.
    pub forward_error_bound: f64,
    /// True iff this step strictly improved on the best residual so far.
    pub improved: bool,
}

/// Diagnostic data returned by [`solve_sparse_refined_with_diagnostics`].
///
/// `kappa_1_est` is computed once per refinement run via the Hager–Higham
/// 1-norm power iteration (3–5 extra solves) — it depends only on `A` and
/// its factor, not on the residual or `x`. Per-step `forward_error_bound`
/// values multiply this constant against the trajectory's relative
/// residual.
///
/// This is the F2.3 deliverable from `dev/plans/kkt-feature-gaps.md`:
/// diagnostic emission only, no behavior change. The non-diagnostic
/// [`solve_sparse_refined`] continues to make the identical control-flow
/// choices.
#[derive(Debug, Clone)]
pub struct RefinementDiagnostics {
    /// Exact `||A||_1` (single linear pass over the CSC values).
    pub anorm_1: f64,
    /// Hager–Higham estimate of `||A||_1 · ||A^{-1}||_1`. A statistical
    /// lower bound; see `dev/research/condition-estimate.md`.
    pub kappa_1_est: f64,
    /// Per-step residual / forward-error trajectory. `steps[0]` is the
    /// unrefined solve.
    pub steps: Vec<RefinementStep>,
    /// Index into `steps` whose iterate is returned (best `||r||_2`).
    pub returned_step: usize,
}

/// Iterative refinement with full per-step diagnostics.
///
/// Mirrors [`solve_sparse_refined`] exactly in control flow and returned
/// iterate; additionally returns a [`RefinementDiagnostics`] struct
/// containing `||A||_1`, the Hager–Higham 1-norm κ̂ estimate, and the
/// per-step residual / Skeel forward-error-bound trajectory.
///
/// Cost: one extra `||A||_1` pass plus 3–5 extra sparse solves for the
/// κ̂ estimate, on top of the refinement loop. Intended for
/// observability (ripopt's δ-ladder logging, Skeel-style termination
/// research) — production hot paths should call [`solve_sparse_refined`]
/// instead.
pub fn solve_sparse_refined_with_diagnostics(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
) -> Result<(Vec<f64>, RefinementDiagnostics), FeralError> {
    let (x, diag) = solve_sparse_refined_core(matrix, factors, rhs, true)?;
    // `with_diagnostics = true` always yields `Some`; if it ever doesn't,
    // that's a logic bug — `expect` is fine in test code, but per CLAUDE.md
    // we use Result in src/. Return DimensionMismatch as a defensive
    // signal (can't actually happen with current control flow).
    let diag = diag.ok_or(FeralError::DimensionMismatch {
        expected: 1,
        got: 0,
    })?;
    Ok((x, diag))
}

fn solve_sparse_refined_core(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    with_diagnostics: bool,
) -> Result<(Vec<f64>, Option<RefinementDiagnostics>), FeralError> {
    let n = factors.n;
    if rhs.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: rhs.len(),
        });
    }

    // κ̂ is a property of (A, factor), independent of x and the
    // refinement trajectory. Compute it once up front so per-step
    // diagnostics can derive the Skeel forward-error bound by
    // multiplying with the step's relative residual.
    let (anorm_1, kappa_1_est) = if with_diagnostics && n > 0 {
        let a1 = matrix_norm_1(matrix);
        let inv1 = estimate_inverse_norm_1(factors)?;
        (a1, a1 * inv1)
    } else {
        (0.0, 0.0)
    };

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

    let rel_res = |rn: f64| if b_norm > 0.0 { rn / b_norm } else { rn };

    let mut steps: Vec<RefinementStep> = if with_diagnostics {
        let rr = rel_res(r_norm);
        vec![RefinementStep {
            step: 0,
            residual_2norm: r_norm,
            relative_residual: rr,
            forward_error_bound: kappa_1_est * rr,
            improved: true,
        }]
    } else {
        Vec::new()
    };
    let mut returned_step: usize = 0;

    for step in 1..=max_steps {
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
            if with_diagnostics {
                returned_step = step;
            }
        } else {
            stagnant_count += 1;
        }

        if with_diagnostics {
            let rr = rel_res(r_norm);
            steps.push(RefinementStep {
                step,
                residual_2norm: r_norm,
                relative_residual: rr,
                forward_error_bound: kappa_1_est * rr,
                improved,
            });
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

    let diag = if with_diagnostics {
        Some(RefinementDiagnostics {
            anorm_1,
            kappa_1_est,
            steps,
            returned_step,
        })
    } else {
        None
    };
    Ok((best_x, diag))
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

    // ----- F2.3 RefinementDiagnostics tests -----

    fn factor_well_cond(m: &CscMatrix) -> SparseFactors {
        let sym = symbolic_factorize(m, &SupernodeParams::default()).unwrap();
        let (factors, _) = factorize_multifrontal(
            m,
            &sym,
            &crate::numeric::factorize::NumericParams::default(),
        )
        .unwrap();
        factors
    }

    /// Hilbert matrix H_n[i,j] = 1/(i+j+1), lower-triangular CSC.
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
    fn diagnostics_match_non_diagnostic_solution() {
        // The diagnostic variant must produce the same iterate as the
        // non-diagnostic one — F2.3 mandate is "no behavior change".
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 2, 2, 2],
            &[0, 1, 0, 1, 2],
            &[2.0, 3.0, 1.0, 1.0, -1e-8],
        )
        .unwrap();
        let rhs = [1.0, 2.0, 3.0];
        let factors = factor_well_cond(&m);

        let x_plain = solve_sparse_refined(&m, &factors, &rhs).unwrap();
        let (x_diag, _diag) = solve_sparse_refined_with_diagnostics(&m, &factors, &rhs).unwrap();
        for i in 0..x_plain.len() {
            assert_eq!(
                x_plain[i].to_bits(),
                x_diag[i].to_bits(),
                "iterate mismatch at index {}: {} vs {}",
                i,
                x_plain[i],
                x_diag[i],
            );
        }
    }

    #[test]
    fn diagnostics_populate_well_conditioned() {
        // SPD tridiagonal: refinement should converge in 0-1 steps and
        // kappa_1_est should be modest.
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
        let factors = factor_well_cond(&m);
        let (_, diag) = solve_sparse_refined_with_diagnostics(&m, &factors, &rhs).unwrap();

        assert!(diag.anorm_1 > 0.0, "anorm_1 must be > 0 for nonzero A");
        assert!(
            diag.kappa_1_est >= 1.0 - 1e-8,
            "kappa_1_est {} below 1.0 lower bound",
            diag.kappa_1_est
        );
        assert!(!diag.steps.is_empty(), "diagnostics must contain step 0");
        assert_eq!(diag.steps[0].step, 0);
        // returned_step must index a valid step.
        assert!(diag.returned_step < diag.steps.len());
        // The returned iterate's residual must be the best seen.
        let best = diag
            .steps
            .iter()
            .map(|s| s.residual_2norm)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(diag.steps[diag.returned_step].residual_2norm, best);
    }

    #[test]
    fn diagnostics_kappa_matches_standalone() {
        // The κ̂ embedded in diagnostics must equal what callers would
        // get from calling estimate_condition_1norm() directly on the
        // same (matrix, factor) pair.
        let m = hilbert(6);
        let rhs = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5];
        let factors = factor_well_cond(&m);
        let kappa_standalone =
            crate::numeric::condition::estimate_condition_1norm(&m, &factors).unwrap();
        let (_, diag) = solve_sparse_refined_with_diagnostics(&m, &factors, &rhs).unwrap();
        assert_eq!(
            diag.kappa_1_est.to_bits(),
            kappa_standalone.to_bits(),
            "diag kappa {} != standalone {}",
            diag.kappa_1_est,
            kappa_standalone,
        );
        // Hilbert-6 is ill-conditioned: κ̂ should easily exceed 1e4.
        assert!(
            diag.kappa_1_est > 1.0e4,
            "Hilbert-6 kappa_1_est {} too small",
            diag.kappa_1_est,
        );
    }

    #[test]
    fn diagnostics_forward_error_bound_field() {
        // forward_error_bound[k] = kappa_1_est * relative_residual[k].
        // Verify the identity directly so downstream consumers
        // (ripopt δ-ladder logging) can rely on the derived field.
        let m = hilbert(4);
        let rhs = [1.0, 2.0, 3.0, 4.0];
        let factors = factor_well_cond(&m);
        let (_, diag) = solve_sparse_refined_with_diagnostics(&m, &factors, &rhs).unwrap();
        for s in &diag.steps {
            let expected = diag.kappa_1_est * s.relative_residual;
            let diff = (s.forward_error_bound - expected).abs();
            assert!(
                diff <= 1e-15 * expected.max(1.0),
                "step {} fwd-err {} vs expected {} (diff {})",
                s.step,
                s.forward_error_bound,
                expected,
                diff
            );
            assert!(s.forward_error_bound >= 0.0);
            assert!(s.residual_2norm.is_finite());
        }
    }

    #[test]
    fn diagnostics_n_zero() {
        let m = CscMatrix::from_triplets(0, &[], &[], &[]).unwrap();
        let factors = factor_well_cond(&m);
        let (x, diag) = solve_sparse_refined_with_diagnostics(&m, &factors, &[]).unwrap();
        assert!(x.is_empty());
        // For n=0 we skip the kappa computation; values default to 0.
        assert_eq!(diag.anorm_1, 0.0);
        assert_eq!(diag.kappa_1_est, 0.0);
    }

    #[test]
    fn diagnostics_dim_mismatch_rejected() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 2.0, 3.0]).unwrap();
        let factors = factor_well_cond(&m);
        // Wrong-length RHS must surface as DimensionMismatch.
        let r = solve_sparse_refined_with_diagnostics(&m, &factors, &[1.0, 2.0]);
        assert!(r.is_err());
    }
}
