#![allow(clippy::needless_range_loop)]
use super::condition::{estimate_inverse_norm_1, matrix_norm_1};
use super::factorize::SparseFactors;
use crate::error::FeralError;
use crate::scaling::ScalingInfo;
use crate::sparse::csc::CscMatrix;

/// Multi-RHS dispatch crossover (issue #57 fix #2). At or above this
/// `nrhs`, `solve_sparse_core_many_into` routes the per-supernode
/// forward/back substitution through the register-blocked BLAS-3 panel
/// kernels (`fwd_blas3`/`back_blas3`); below it, the fix-#1 row-major
/// rank-1 kernels run (bit-identical to looping the single-RHS path).
/// 32 keeps the IPM hot path (small `nrhs`) and the small-`nrhs` many
/// path on the proven rank-1 code, and clears the `k ≈ 16` crossover
/// from `dev/research/multi-rhs.md` D3 with margin for the microkernel
/// setup. See `dev/research/issue-57-blas3-panel.md`.
const BLAS3_NRHS_THRESHOLD: usize = 32;

/// Multi-RHS refinement dispatch crossover (issue #58). At or above this
/// `nrhs`, `Solver::solve_many_refined` refines through the batched
/// `solve_sparse_many_refined` (one panel solve per refinement step over
/// the still-active columns) instead of looping the single-RHS refiner
/// per column. Below it, the per-column loop runs — keeping the IPM
/// predictor-corrector (`nrhs = 2`) and other narrow refined solves on
/// the proven, bit-identical path. 16 captures the batched-solve
/// amortization (it begins below the 32 panel-kernel crossover) while
/// staying provably bit-identical to the per-column loop for
/// `16 ≤ nrhs < 32`. See `dev/research/issue-58-batched-refinement.md`.
pub(crate) const BLAS3_REFINE_THRESHOLD: usize = 16;

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

// N5 (`dev/research/repo-review-2026-06-09.md`) reproducing-test
// instrumentation: counts `SolveWorkspace` constructions so a white-box
// test can prove the condition estimator pools one workspace across its
// internal solves instead of building a fresh one per `solve_sparse`
// call. `#[cfg(test)]` only — zero production footprint.
//
// Thread-local, not a global atomic: the cargo test harness runs tests
// concurrently and several `condition` tests call the estimator, so a
// shared counter would race. The estimator's internal solves all run on
// the calling thread, so a per-thread counter measures exactly its own
// workspace builds regardless of what other test threads are doing.
#[cfg(test)]
thread_local! {
    pub(super) static SOLVE_WORKSPACE_BUILDS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// N5: reset the current thread's `SolveWorkspace`-construction counter
/// before a measured region. Test-only.
#[cfg(test)]
pub(super) fn reset_solve_workspace_builds() {
    SOLVE_WORKSPACE_BUILDS.with(|c| c.set(0));
}

/// N5: read the current thread's `SolveWorkspace`-construction counter.
/// Test-only.
#[cfg(test)]
pub(super) fn solve_workspace_builds() -> usize {
    SOLVE_WORKSPACE_BUILDS.with(|c| c.get())
}

/// Workspace holding the per-call scratch buffers used by the sparse
/// solve. Allowing the caller to own this lets us amortize the
/// allocations across many solves — see `solve_sparse_refined`, which
/// performs up to 11 solves per call (1 initial + 10 refinement steps)
/// against the same factors, and `estimate_inverse_norm_1` (N5), which
/// pools one across its ~11 internal Hager-iteration solves.
pub(super) struct SolveWorkspace {
    /// Permuted RHS / working solution vector, length `n`.
    y: Vec<f64>,
    /// Per-supernode gather/scatter buffer, length `max_nrow`.
    w: Vec<f64>,
    /// Scaled RHS storage when MC64 scaling is active, length `n`.
    /// Empty when no scaling is applied (the `solve_sparse` fast path).
    scaled_rhs: Vec<f64>,
}

impl SolveWorkspace {
    pub(super) fn for_factors(factors: &SparseFactors) -> Self {
        #[cfg(test)]
        SOLVE_WORKSPACE_BUILDS.with(|c| c.set(c.get() + 1));
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

pub(super) fn solve_sparse_into_ws(
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

/// Solve the symmetric 2×2 D-block system `[[a,b],[b,c]] · [x0,x1] = [z0,z1]`,
/// returning `Some((x0, x1))`, or `None` when the shared SSIDS determinant
/// floor rejects the block.
///
/// REG-3 (`dev/research/repo-review-2026-06-09-verification.md`): the sparse
/// forward and multi-RHS D-block solves previously gated on the *naive*
/// `det.abs() > zero_tol_2x2` — the absolute floor that finding D4 already
/// replaced on the *dense* solve path (`dense/solve.rs`) with the
/// scale-invariant `ssids_det_floor_fail`. A well-conditioned 2×2 block at
/// small absolute scale (true `|det| < zero_tol_2x2 ≈ EPS²`) is accepted by
/// the factor (which uses the SSIDS floor) but was silently *skipped* by the
/// sparse solve — wrong solution, no error, no flag. Both sparse sites now
/// route through this helper, so a block the factor stores as invertible the
/// solve inverts, and the dense and sparse solve gates agree.
#[inline]
fn solve_2x2_dblock(a: f64, b: f64, c: f64, z0: f64, z1: f64) -> Option<(f64, f64)> {
    if crate::dense::factor::ssids_det_floor_fail(a, b, c) {
        return None;
    }
    // `b != 0` for a stored 2×2 (`d_subdiag != 0`). The normalized form
    // (faer) avoids cancellation; the b-tiny direct branch is retained for
    // bit-parity with the prior sparse kernel on accepted blocks.
    if b.abs() > f64::EPSILON * (a.abs() + c.abs()).max(1.0) {
        let ak = a / b;
        let ck = c / b;
        let denom = 1.0 / (ak * ck - 1.0);
        let z0k = z0 / b;
        let z1k = z1 / b;
        Some(((ck * z0k - z1k) * denom, (ak * z1k - z0k) * denom))
    } else {
        let det = a * c - b * b;
        Some(((c * z0 - b * z1) / det, (a * z1 - b * z0) / det))
    }
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
                // REG-3: gate on the shared scale-invariant SSIDS floor
                // (matching the factor side and the dense solve — finding
                // D4), not the naive absolute `det.abs() > zero_tol_2x2`.
                if let Some((x0, x1)) = solve_2x2_dblock(a, b, c, w[k], w[k + 1]) {
                    w[k] = x0;
                    w[k + 1] = x1;
                }
                // else: 2×2 block rejected by the shared SSIDS floor (the
                // factor side would not have stored it as invertible);
                // leave w[k], w[k + 1] untouched.
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
/// See `dev/research/multi-rhs.md` (F1.0) for the layout decisions and
/// `dev/research/issue-57-blas3-panel.md` for the row-major flip —
/// y/w are row-major, scaled_rhs is column-major (caller layout), all
/// widened by a factor of `nrhs` relative to the single-RHS
/// `SolveWorkspace`.
pub struct SolveManyWorkspace {
    /// Permuted RHS / working solution vector, length `n * nrhs`,
    /// **row-major**: node `k` lives at `[k*nrhs .. (k+1)*nrhs]`. Row-major
    /// so the per-supernode gather/scatter is a contiguous memcpy
    /// (issue #57); the caller-visible `rhs`/`x` stay column-major, with
    /// the transpose absorbed by the one-time entry/exit (un)permute.
    y: Vec<f64>,
    /// Per-supernode gather/scatter buffer, length `max_nrow * nrhs`,
    /// **row-major**: element `(i, c)` lives at `w[i*nrhs + c]`.
    /// Row-major so the per-RHS inner loops in `solve_sparse_core_many_into`
    /// are contiguous (stride-1) and auto-vectorize (issue #57).
    w: Vec<f64>,
    /// Back-substitution dot-product accumulator, length `nrhs`. One
    /// slot per column, reused across pivots and nodes so the inner
    /// `c`-loop stays contiguous without a per-pivot allocation.
    acc: Vec<f64>,
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
            acc: vec![0.0; nrhs],
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
    // N6: `ws.scaled_rhs` is sized from the scaling state of the factors the
    // workspace was built against (`for_factors`): `n * nrhs` when scaling is
    // applied, empty otherwise. A workspace built for unscaled factors reused
    // with scaled factors of the same `(n, nrhs)` shape (or vice versa) would
    // otherwise index `scaled_rhs` out of bounds at the pre-scale step below.
    // Validate it here so the crate returns `Result` rather than panicking.
    let needs_scaling = !matches!(factors.scaling_info, ScalingInfo::NotApplied);
    let expected_scaled_len = if needs_scaling { n * nrhs } else { 0 };
    if ws.scaled_rhs.len() != expected_scaled_len {
        return Err(FeralError::DimensionMismatch {
            expected: expected_scaled_len,
            got: ws.scaled_rhs.len(),
        });
    }
    if n == 0 {
        return Ok(());
    }

    // Pre-scale every column by D (MC64 congruence). Skipped when
    // ScalingInfo::NotApplied (the scaling vector is all-ones).
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

    solve_sparse_core_many_into(
        factors,
        rhs_for_core,
        nrhs,
        x_out,
        &mut ws.y,
        &mut ws.w,
        &mut ws.acc,
    );

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
/// `nrhs` columns. `rhs` and `x_out` are **column-major** `n × nrhs`
/// (the caller-visible contract, matching MUMPS/SSIDS). The internal
/// working buffers `y` and the per-supernode `w` are both **row-major**
/// (`y[node*nrhs + c]`, `w[i*nrhs + c]`) so the per-RHS inner loops are
/// contiguous and auto-vectorize and the per-supernode gather/scatter
/// is a contiguous memcpy (issue #57). The column-major ↔ row-major
/// transpose happens once each, in the entry permute and exit unpermute.
/// The single-RHS path (`solve_sparse_core_into`) is preserved unchanged
/// so the iterative-refinement code path stays on a tested code path.
fn solve_sparse_core_many_into(
    factors: &SparseFactors,
    rhs: &[f64],
    nrhs: usize,
    x_out: &mut [f64],
    y_buf: &mut [f64],
    w_buf: &mut [f64],
    acc_buf: &mut [f64],
) {
    let n = factors.n;
    let y = &mut y_buf[..n * nrhs];

    // Route wide solves through the BLAS-3 panel kernels (issue #57
    // fix #2); narrow solves stay on the bit-identical rank-1 kernels.
    let use_blas3 = nrhs >= BLAS3_NRHS_THRESHOLD;

    // Permute the RHS into the **row-major** working layout
    // `y[new*nrhs + c] = rhs[c, perm[new]]`. The caller's `rhs` stays
    // column-major; this one-time gather is the only stride-`n` read,
    // and it lets every per-supernode gather/scatter below be a
    // contiguous memcpy (issue #57: the stride-`n` transpose in the
    // hot per-supernode loops was the multi-RHS bottleneck, badly so
    // when `n` is a power of two and columns alias in cache).
    for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
        let dst = new_idx * nrhs;
        for c in 0..nrhs {
            y[dst + c] = rhs[c * n + old_idx];
        }
    }

    // Phase 1+2: Forward substitution and D-block solve, fused into a
    // single postorder pass (postorder).
    for node in &factors.node_factors {
        let ff = &node.frontal_factors;
        let nelim = ff.nelim;
        let nrow = ff.nrow;
        if nelim == 0 {
            continue;
        }

        // Gather the supernode's rows from `y` into `w` (both row-major):
        // w[i, :] = y[row_indices[perm[i]], :], a contiguous memcpy.
        let w = &mut w_buf[..nrow * nrhs];
        for i in 0..nrow {
            let src = node.row_indices[ff.perm[i]] * nrhs;
            w[i * nrhs..(i + 1) * nrhs].copy_from_slice(&y[src..src + nrhs]);
        }

        // L-solve. At small `nrhs` the row-major rank-1 cascade runs
        // (bit-identical to looping single-RHS); at `nrhs >=
        // BLAS3_NRHS_THRESHOLD` the register-blocked panel kernel runs
        // (TRSM on L_11 + GEMM on L_21, issue #57 fix #2).
        if use_blas3 {
            fwd_blas3(w, &ff.l, nrow, nelim, nrhs);
        } else {
            fwd_rank1(w, &ff.l, nrow, nelim, nrhs);
        }

        // D-block solve, fused into the forward pass. A node's
        // eliminated rows (0..nelim) are final once its forward-sub
        // completes — ancestors only ever touch its separator rows — so
        // D⁻¹ can be applied here instead of in a second postorder pass,
        // saving one gather/scatter round trip per supernode (issue #57).
        dsolve_node(w, ff, nelim, nrhs);

        // Scatter back into `y` (both row-major), undoing the BK
        // permutation: y[row_indices[perm[i]], :] = w[i, :].
        for i in 0..nrow {
            let dst = node.row_indices[ff.perm[i]] * nrhs;
            y[dst..dst + nrhs].copy_from_slice(&w[i * nrhs..(i + 1) * nrhs]);
        }
    }

    // Phase 3: Backward substitution (reverse postorder).
    let acc = &mut acc_buf[..nrhs];
    for node in factors.node_factors.iter().rev() {
        let ff = &node.frontal_factors;
        let nelim = ff.nelim;
        let nrow = ff.nrow;
        if nelim == 0 {
            continue;
        }

        let w = &mut w_buf[..nrow * nrhs];
        for i in 0..nrow {
            // y is row-major (`y[node*nrhs + c]`), so each supernode row
            // gathers a contiguous run — a memcpy, not a stride-`n` walk.
            let src = node.row_indices[ff.perm[i]] * nrhs;
            w[i * nrhs..(i + 1) * nrhs].copy_from_slice(&y[src..src + nrhs]);
        }

        // L^T-solve (mirror of the forward dispatch).
        if use_blas3 {
            back_blas3(w, &ff.l, nrow, nelim, nrhs, acc);
        } else {
            back_rank1(w, &ff.l, nrow, nelim, nrhs, acc);
        }

        for i in 0..nrow {
            let dst = node.row_indices[ff.perm[i]] * nrhs;
            y[dst..dst + nrhs].copy_from_slice(&w[i * nrhs..(i + 1) * nrhs]);
        }
    }

    // Unpermute from the row-major `y` back to the caller's column-major
    // `x_out`: x[c, old] = y[new*nrhs + c]. One-time scatter (mirror of
    // the entry permute).
    for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
        let src = new_idx * nrhs;
        for c in 0..nrhs {
            x_out[c * n + old_idx] = y[src + c];
        }
    }
}

// === Per-supernode multi-RHS substitution kernels (issue #57) ========
//
// `w` is the row-major per-supernode buffer (`w[i*nrhs + c]`, `nrow`
// rows × `nrhs` columns). `l` is the column-major panel `ff.l`
// (`L[i,j] = l[j*nrow + i]`, unit lower-trapezoidal, `nelim` columns).
// All four kernels operate purely on `w` in place; the caller handles
// gather/scatter and the D-block solve.

/// Forward L-solve, rank-1 cascade (fix #1). For each eliminated column
/// `j`, broadcast `w[j, :]` into every trailing row `i > j`. The inner
/// `c`-loop is a contiguous axpy. Bit-identical to looping single-RHS.
fn fwd_rank1(w: &mut [f64], l: &[f64], nrow: usize, nelim: usize, nrhs: usize) {
    for j in 0..nelim {
        let (head, tail) = w.split_at_mut((j + 1) * nrhs);
        let w_j = &head[j * nrhs..(j + 1) * nrhs];
        for i in (j + 1)..nrow {
            let l_ij = l[j * nrow + i];
            let base = (i - j - 1) * nrhs;
            let w_i = &mut tail[base..base + nrhs];
            for c in 0..nrhs {
                w_i[c] -= l_ij * w_j[c];
            }
        }
    }
}

/// Backward Lᵀ-solve, rank-1 cascade (fix #1). For each column `j`
/// (descending), `acc[c] = sum_{i>j} L[i,j]·w[i,c]`, then `w[j,:] -=
/// acc`. Iterating `i` outer keeps the per-column accumulation order
/// identical to the single-RHS path → bit-identical.
fn back_rank1(w: &mut [f64], l: &[f64], nrow: usize, nelim: usize, nrhs: usize, acc: &mut [f64]) {
    for j in (0..nelim).rev() {
        for s in acc.iter_mut() {
            *s = 0.0;
        }
        for i in (j + 1)..nrow {
            let l_ij = l[j * nrow + i];
            let w_i = &w[i * nrhs..(i + 1) * nrhs];
            for c in 0..nrhs {
                acc[c] += l_ij * w_i[c];
            }
        }
        let w_j = &mut w[j * nrhs..(j + 1) * nrhs];
        for c in 0..nrhs {
            w_j[c] -= acc[c];
        }
    }
}

/// Forward L-solve, BLAS-3 panel form (fix #2): TRSM on the unit-lower
/// triangle `L_11` (panel rows only) followed by a register-blocked
/// GEMM `w_bot -= L_21 @ w_top` on the trailing rows. The TRSM updates
/// rows in increasing `j` and the GEMM seeds its accumulator with the
/// current `w` value and reduces in increasing `j`, so the whole
/// forward solve stays **bit-identical** to the rank-1 cascade.
fn fwd_blas3(w: &mut [f64], l: &[f64], nrow: usize, nelim: usize, nrhs: usize) {
    // TRSM: L_11 (unit lower), update only panel rows i in (j+1)..nelim.
    for j in 0..nelim {
        let (head, tail) = w.split_at_mut((j + 1) * nrhs);
        let w_j = &head[j * nrhs..(j + 1) * nrhs];
        for i in (j + 1)..nelim {
            let l_ij = l[j * nrow + i];
            let base = (i - j - 1) * nrhs;
            let w_i = &mut tail[base..base + nrhs];
            for c in 0..nrhs {
                w_i[c] -= l_ij * w_j[c];
            }
        }
    }
    // GEMM: w_bot -= L_21 @ w_top. L_21[i', j] = l[j*nrow + nelim + i'].
    if nelim < nrow {
        let (top, bot) = w.split_at_mut(nelim * nrhs);
        let a = PanelBlock {
            l,
            base: nelim,
            row_stride: 1,
            col_stride: nrow,
        };
        gemm_panel_minus(bot, &a, top, nrow - nelim, nelim, nrhs);
    }
}

/// Backward Lᵀ-solve, BLAS-3 panel form (fix #2): register-blocked GEMM
/// `w_top -= L_21ᵀ @ w_bot` (trailing contribution to every panel
/// column) followed by the TRSM back-solve of `L_11ᵀ` on the panel
/// rows. The GEMM applies the trailing rows before the panel TRSM,
/// whereas the cascade interleaves them per column, so the result
/// differs from the rank-1 path only by floating-point reassociation
/// (~κ·eps) — well inside the 1e-12 parity tolerance.
fn back_blas3(w: &mut [f64], l: &[f64], nrow: usize, nelim: usize, nrhs: usize, acc: &mut [f64]) {
    // GEMM: w_top -= L_21^T @ w_bot. (L_21^T)[j, i'] = l[j*nrow + nelim + i'].
    if nelim < nrow {
        let (top, bot) = w.split_at_mut(nelim * nrhs);
        let a = PanelBlock {
            l,
            base: nelim,
            row_stride: nrow,
            col_stride: 1,
        };
        gemm_panel_minus(top, &a, bot, nelim, nrow - nelim, nrhs);
    }
    // TRSM: L_11^T, update only panel rows i in (j+1)..nelim.
    for j in (0..nelim).rev() {
        for s in acc.iter_mut() {
            *s = 0.0;
        }
        for i in (j + 1)..nelim {
            let l_ij = l[j * nrow + i];
            let w_i = &w[i * nrhs..(i + 1) * nrhs];
            for c in 0..nrhs {
                acc[c] += l_ij * w_i[c];
            }
        }
        let w_j = &mut w[j * nrhs..(j + 1) * nrhs];
        for c in 0..nrhs {
            w_j[c] -= acc[c];
        }
    }
}

/// D-block solve on the eliminated rows of one supernode, in place on
/// the row-major `w` (`w[k*nrhs + c]`). Applies `D⁻¹` per column: 1×1
/// pivots divide, 2×2 pivots solve the symmetric system. Arithmetic is
/// identical to the single-RHS path (`solve_sparse_core_into`); only the
/// element addresses change to the row-major layout. Force-accepted zero
/// pivots (1×1) and singular 2×2 blocks are left untouched, matching the
/// single-RHS path.
fn dsolve_node(
    w: &mut [f64],
    ff: &crate::dense::factor::FrontalFactors,
    nelim: usize,
    nrhs: usize,
) {
    for c in 0..nrhs {
        let mut k = 0;
        while k < nelim {
            if k + 1 < nelim && ff.d_subdiag[k] != 0.0 {
                let a = ff.d_diag[k];
                let b = ff.d_subdiag[k];
                let cc = ff.d_diag[k + 1];
                // REG-3: shared scale-invariant SSIDS gate (see
                // `solve_2x2_dblock`), not the naive absolute floor.
                if let Some((x0, x1)) =
                    solve_2x2_dblock(a, b, cc, w[k * nrhs + c], w[(k + 1) * nrhs + c])
                {
                    w[k * nrhs + c] = x0;
                    w[(k + 1) * nrhs + c] = x1;
                }
                // else: rejected by the shared SSIDS floor; leave as-is.
                k += 2;
            } else {
                if ff.d_diag[k].abs() > ff.zero_tol {
                    w[k * nrhs + c] /= ff.d_diag[k];
                }
                // else: pivot force-accepted as zero; leave as-is.
                k += 1;
            }
        }
    }
}

/// Column-major sub-block of the panel `ff.l`, viewed as a dense matrix
/// `A` with `A[m, k] = l[base + m*row_stride + k*col_stride]`. Lets one
/// GEMM microkernel serve both the forward (`L_21`) and the backward
/// (`L_21ᵀ`) trailing update by swapping the strides.
struct PanelBlock<'a> {
    l: &'a [f64],
    base: usize,
    row_stride: usize,
    col_stride: usize,
}

/// Register-blocked panel GEMM: `C[m, c] -= sum_k A[m, k] · B[k, c]`,
/// `C` (`m_dim × nrhs`) and `B` (`k_dim × nrhs`) row-major with leading
/// dimension `nrhs`, `A` an `m_dim × k_dim` view into the column-major
/// panel. The MR×NR core holds the output tile in registers and reduces
/// over `k`, seeding the accumulator with the current `C` value so the
/// reduction is a left fold over increasing `k` (bit-identical to the
/// cascade when the reduction axis matches). Tails fall back to a
/// scalar block (same left-fold order).
fn gemm_panel_minus(
    c_rows: &mut [f64],
    a: &PanelBlock,
    b_rows: &[f64],
    m_dim: usize,
    k_dim: usize,
    nrhs: usize,
) {
    const MR: usize = 4;
    const NR: usize = 8;
    let m_main = m_dim - m_dim % MR;
    let c_main = nrhs - nrhs % NR;

    let mut m0 = 0;
    while m0 < m_main {
        // Four contiguous output rows, disjoint so they can be held
        // mutably at once.
        let block = &mut c_rows[m0 * nrhs..(m0 + MR) * nrhs];
        let (r0, rest) = block.split_at_mut(nrhs);
        let (r1, rest) = rest.split_at_mut(nrhs);
        let (r2, r3) = rest.split_at_mut(nrhs);
        let ab0 = a.base + m0 * a.row_stride;
        let ab1 = a.base + (m0 + 1) * a.row_stride;
        let ab2 = a.base + (m0 + 2) * a.row_stride;
        let ab3 = a.base + (m0 + 3) * a.row_stride;

        let mut c0 = 0;
        while c0 < c_main {
            let mut acc0 = [0.0f64; NR];
            let mut acc1 = [0.0f64; NR];
            let mut acc2 = [0.0f64; NR];
            let mut acc3 = [0.0f64; NR];
            acc0.copy_from_slice(&r0[c0..c0 + NR]);
            acc1.copy_from_slice(&r1[c0..c0 + NR]);
            acc2.copy_from_slice(&r2[c0..c0 + NR]);
            acc3.copy_from_slice(&r3[c0..c0 + NR]);
            let mut bb = [0.0f64; NR];
            for k in 0..k_dim {
                bb.copy_from_slice(&b_rows[k * nrhs + c0..k * nrhs + c0 + NR]);
                let kc = k * a.col_stride;
                let a0 = a.l[ab0 + kc];
                let a1 = a.l[ab1 + kc];
                let a2 = a.l[ab2 + kc];
                let a3 = a.l[ab3 + kc];
                for s in 0..NR {
                    let bv = bb[s];
                    acc0[s] -= a0 * bv;
                    acc1[s] -= a1 * bv;
                    acc2[s] -= a2 * bv;
                    acc3[s] -= a3 * bv;
                }
            }
            r0[c0..c0 + NR].copy_from_slice(&acc0);
            r1[c0..c0 + NR].copy_from_slice(&acc1);
            r2[c0..c0 + NR].copy_from_slice(&acc2);
            r3[c0..c0 + NR].copy_from_slice(&acc3);
            c0 += NR;
        }
        m0 += MR;
    }

    // Column tail (nrhs % NR) for the MR-tiled rows.
    gemm_scalar_block(c_rows, a, b_rows, 0, m_main, c_main, nrhs, k_dim, nrhs);
    // Row tail (m_dim % MR), full column range.
    gemm_scalar_block(c_rows, a, b_rows, m_main, m_dim, 0, nrhs, k_dim, nrhs);
}

/// Scalar fallback for the GEMM tails: `C[m, c] -= sum_k A[m, k]·B[k, c]`
/// over `m ∈ [m_lo, m_hi)`, `c ∈ [c_lo, c_hi)`. Accumulates per `(m, c)`
/// in increasing `k` (left fold), matching the core kernel's order.
#[allow(clippy::too_many_arguments)]
fn gemm_scalar_block(
    c_rows: &mut [f64],
    a: &PanelBlock,
    b_rows: &[f64],
    m_lo: usize,
    m_hi: usize,
    c_lo: usize,
    c_hi: usize,
    k_dim: usize,
    nrhs: usize,
) {
    for m in m_lo..m_hi {
        let ab = a.base + m * a.row_stride;
        let row = &mut c_rows[m * nrhs..(m + 1) * nrhs];
        for c in c_lo..c_hi {
            let mut sum = row[c];
            for k in 0..k_dim {
                sum -= a.l[ab + k * a.col_stride] * b_rows[k * nrhs + c];
            }
            row[c] = sum;
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

/// Multi-RHS solve with per-column iterative refinement, batched through
/// the panel kernel (issue #58). The initial and per-step correction
/// solves go through `solve_sparse_many` — one batched solve over the
/// still-active columns — instead of `nrhs` single-RHS solves, so wide
/// refined solves reach the BLAS-3 panel kernel that fix #2 added.
///
/// The per-column convergence logic mirrors `solve_sparse_refined_core`
/// exactly (same `max_steps`, 2-strike plateau, `ε·√n` relative target,
/// 100× divergence guard, and per-column best-iterate). Each step
/// **compacts** the active (un-converged) columns into the batched
/// solve, so the work never exceeds the per-column loop. `rhs` is
/// column-major `n × nrhs`; the column-major best-iterate solution is
/// returned. See `dev/research/issue-58-batched-refinement.md`.
pub fn solve_sparse_many_refined(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    nrhs: usize,
) -> Result<Vec<f64>, FeralError> {
    let n = factors.n;
    if rhs.len() != n * nrhs {
        return Err(FeralError::DimensionMismatch {
            expected: n * nrhs,
            got: rhs.len(),
        });
    }
    if nrhs == 0 || n == 0 {
        return Ok(vec![0.0; n * nrhs]);
    }

    // Same constants as the single-RHS refiner (solve_sparse_refined_core).
    let max_steps = 10;
    let max_stagnant_steps = 2;
    let threshold = f64::EPSILON * (n as f64).sqrt();
    let divergence_factor = 100.0;
    let relative_reached = |r_norm: f64, b_norm: f64| -> bool {
        if b_norm > 0.0 {
            r_norm < threshold * b_norm
        } else {
            r_norm < threshold
        }
    };

    // Initial batched solve.
    let mut x = solve_sparse_many(factors, rhs, nrhs)?;
    let mut best_rn = vec![0.0f64; nrhs];
    let mut bnorm = vec![0.0f64; nrhs];

    // Initial per-column residual r_c = b_c - A·x_c into a small reused
    // scratch; build the active set (columns not yet at the target). The
    // wide per-call buffers (best_x, the residual gather buffer) are NOT
    // allocated yet — the well-conditioned common case, where the direct
    // solve already meets the target for every column, returns below
    // having allocated only `x` and the length-`n` scratch. (Allocating
    // three `n × nrhs` Vecs up front was ~50 µs/RHS of the Python
    // `solve_refined` overhead, issue #58.)
    let mut rc = vec![0.0f64; n];
    let mut active: Vec<usize> = Vec::new();
    for c in 0..nrhs {
        matrix.symv(&x[c * n..(c + 1) * n], &mut rc);
        for i in 0..n {
            rc[i] = rhs[c * n + i] - rc[i];
        }
        bnorm[c] = norm2(&rhs[c * n..(c + 1) * n]);
        best_rn[c] = norm2(&rc);
        if !relative_reached(best_rn[c], bnorm[c]) {
            active.push(c);
        }
    }
    if active.is_empty() {
        return Ok(x);
    }

    // Refinement is needed for at least one column.
    let mut best_x = x.clone();
    let mut stagnant = vec![0usize; nrhs];
    // Gather buffer sized to the (shrinking) active set; the leading
    // `n * active.len()` is used each step.
    let mut r_act = vec![0.0f64; n * active.len()];

    for _step in 1..=max_steps {
        if active.is_empty() {
            break;
        }
        let na = active.len();

        // Residual of each active column → gather buffer, then
        // batched-solve the correction over just the active columns.
        for (k, &c) in active.iter().enumerate() {
            matrix.symv(&x[c * n..(c + 1) * n], &mut r_act[k * n..(k + 1) * n]);
            for i in 0..n {
                r_act[k * n + i] = rhs[c * n + i] - r_act[k * n + i];
            }
        }
        let dx = solve_sparse_many(factors, &r_act[..n * na], na)?;

        let mut still: Vec<usize> = Vec::with_capacity(na);
        for (k, &c) in active.iter().enumerate() {
            // x_c += dx_k
            for i in 0..n {
                x[c * n + i] += dx[k * n + i];
            }
            // Residual of the updated column.
            matrix.symv(&x[c * n..(c + 1) * n], &mut rc);
            for i in 0..n {
                rc[i] = rhs[c * n + i] - rc[i];
            }
            let rn = norm2(&rc);

            if rn < best_rn[c] {
                best_rn[c] = rn;
                best_x[c * n..(c + 1) * n].copy_from_slice(&x[c * n..(c + 1) * n]);
                stagnant[c] = 0;
            } else {
                stagnant[c] += 1;
            }

            // Stop this column on convergence, divergence, or plateau —
            // identical predicates to the single-RHS refiner.
            let done = relative_reached(best_rn[c], bnorm[c])
                || rn > best_rn[c] * divergence_factor
                || stagnant[c] >= max_stagnant_steps;
            if !done {
                still.push(c);
            }
        }
        active = still;
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

    /// N6 (repo-review-2026-06-09.md): `solve_sparse_many_into` validated
    /// `ws.nrhs` / `ws.n` but not whether `ws.scaled_rhs` was sized for the
    /// factors' scaling state. A workspace built for *unscaled* factors
    /// (`scaled_rhs` empty) reused with *scaled* factors of the same
    /// `(n, nrhs)` shape would index the empty `scaled_rhs` out of bounds at
    /// the pre-scale step — a panic in a crate that otherwise returns
    /// `Result`. The validation must surface this as `DimensionMismatch`.
    #[test]
    fn solve_many_into_rejects_scaling_mismatched_workspace() {
        use crate::scaling::ScalingStrategy;

        // SPD tridiagonal; factorizes cleanly under either scaling choice.
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[2.0, -1.0, 2.0, -1.0, 2.0],
        )
        .unwrap();
        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let nrhs = 2;

        // Unscaled factors -> ScalingInfo::NotApplied -> ws.scaled_rhs empty.
        let mut params_unscaled = make_params();
        params_unscaled.scaling = ScalingStrategy::Identity;
        let (factors_unscaled, _) = factorize_multifrontal(&m, &sym, &params_unscaled).unwrap();
        assert!(matches!(
            factors_unscaled.scaling_info,
            ScalingInfo::NotApplied
        ));
        let mut ws = SolveManyWorkspace::for_factors(&factors_unscaled, nrhs);
        assert_eq!(ws.scaled_rhs.len(), 0);

        // Scaled factors of the SAME (n, nrhs) shape. External always reports
        // ScalingInfo::Applied (even all-ones), so needs_scaling is true and
        // the pre-scale step writes into ws.scaled_rhs.
        let mut params_scaled = make_params();
        params_scaled.scaling = ScalingStrategy::External(vec![1.0; m.n]);
        let (factors_scaled, _) = factorize_multifrontal(&m, &sym, &params_scaled).unwrap();
        assert!(!matches!(
            factors_scaled.scaling_info,
            ScalingInfo::NotApplied
        ));

        let rhs = vec![1.0; m.n * nrhs];
        let mut x = vec![0.0; m.n * nrhs];
        // Before the fix this panicked (OOB on the empty scaled_rhs); it must
        // now return DimensionMismatch instead.
        let result = solve_sparse_many_into(&factors_scaled, &rhs, nrhs, &mut x, &mut ws);
        assert!(
            matches!(result, Err(FeralError::DimensionMismatch { .. })),
            "expected DimensionMismatch for a scaling-mismatched workspace, got {result:?}"
        );
    }

    /// REG-3 (`repo-review-2026-06-09-verification.md`): a well-conditioned
    /// 2×2 D-block at small absolute scale that the factor side accepts
    /// (scale-invariant SSIDS floor) must be inverted by the sparse solve,
    /// not skipped by the old naive `det.abs() > zero_tol_2x2` absolute
    /// floor. Mirrors `tests/d4_solve_2x2_gate.rs` on the sparse multi-RHS
    /// D-block path (`dsolve_node`). Oracle: `rhs = D · x_true`
    /// hand-computed (pure linear algebra), independent of the solver.
    /// Pre-fix the block is skipped (w ≈ rhs, off by 16 orders); post-fix
    /// w ≈ x_true.
    #[test]
    fn reg3_sparse_dsolve_small_scale_2x2_inverted() {
        // D = [[1e-16, 1e-17],[1e-17, 1e-16]]: det = 9.9e-33 < zero_tol_2x2
        // (≈4.9e-32) → naive gate skips; ssids_det_floor_fail accepts
        // (max_piv 1e-16, detpiv ≈ 9.9e-17 > cancel_floor 5e-17).
        let (a, b, c) = (1e-16, 1e-17, 1e-16);
        let x_true = [1.0_f64, 1.0_f64];
        let rhs = [a * x_true[0] + b * x_true[1], b * x_true[0] + c * x_true[1]];

        let ff = crate::dense::factor::FrontalFactors {
            nrow: 2,
            ncol: 2,
            nelim: 2,
            l: vec![1.0, 0.0, 0.0, 1.0],
            d_diag: vec![a, c],
            d_subdiag: vec![b, 0.0],
            perm: vec![0, 1],
            perm_inv: vec![0, 1],
            contrib: vec![],
            contrib_dim: 0,
            n_delayed: 0,
            inertia: crate::inertia::Inertia::new(1, 1, 0),
            needs_refinement: false,
            n_rook_rescues: 0,
            n_tiny: 0,
            zero_tol: f64::EPSILON,
            zero_tol_2x2: f64::EPSILON * f64::EPSILON,
        };

        let mut w = vec![rhs[0], rhs[1]]; // nrhs = 1, row-major w[k*nrhs+0]
        dsolve_node(&mut w, &ff, 2, 1);

        assert!(
            (w[0] - x_true[0]).abs() < 1e-6 && (w[1] - x_true[1]).abs() < 1e-6,
            "REG-3: small-scale 2×2 must be inverted by sparse dsolve, not \
             skipped; got w = {w:?}, expected ≈ {x_true:?}"
        );
    }

    /// `solve_2x2_dblock` inverts a small-scale well-conditioned block (the
    /// single source of truth shared by both sparse D-solve sites).
    #[test]
    fn reg3_helper_inverts_small_scale_block() {
        let (a, b, c) = (1e-16, 1e-17, 1e-16);
        let x = [1.0_f64, 1.0_f64];
        let (z0, z1) = (a * x[0] + b * x[1], b * x[0] + c * x[1]);
        let (x0, x1) = solve_2x2_dblock(a, b, c, z0, z1).expect("accepted");
        assert!((x0 - 1.0).abs() < 1e-6 && (x1 - 1.0).abs() < 1e-6);
    }

    /// REG-3 consistency guard: an ill-conditioned block the factor-side
    /// SSIDS floor rejects (detpiv = 0) must be skipped by the sparse solve
    /// too. D = [[2^53+1, 2^53],[2^53, 2^53]] (true det = 2^53, condition
    /// ~2^53). `solve_2x2_dblock` returns None so the caller leaves the RHS
    /// untouched — pins that the fix did not start inverting rejected blocks.
    #[test]
    fn reg3_rejected_block_skipped_by_helper() {
        let p = (1u64 << 53) as f64;
        assert!(solve_2x2_dblock(p + 1.0, p, p, 1.0, 2.0).is_none());
    }
}
