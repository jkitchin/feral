#![allow(clippy::needless_range_loop)]
use super::condition::{estimate_inverse_norm_1, matrix_norm_1};
use super::factorize::{NodeFactors, SparseFactors};
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

/// In-place form of [`solve_sparse`]: writes the solution into `x_out`
/// instead of returning an owned `Vec` (issue #178 item 2).
///
/// Bit-for-bit identical to [`solve_sparse`] on the same factor and
/// right-hand side. Returns [`FeralError::DimensionMismatch`] when
/// `x_out.len() != factors.n` — never panics, and never leaves a partial
/// write behind a length error.
///
/// Aliasing `rhs` and `x_out` is unrepresentable: `&[f64]` and
/// `&mut [f64]` borrowing one allocation cannot coexist, so the compiler
/// rejects the aliased call rather than this function having to.
pub fn solve_sparse_into(
    factors: &SparseFactors,
    rhs: &[f64],
    x_out: &mut [f64],
) -> Result<(), FeralError> {
    let n = factors.n;
    if x_out.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: x_out.len(),
        });
    }
    if n == 0 && rhs.is_empty() {
        return Ok(());
    }
    let mut ws = SolveWorkspace::for_factors(factors);
    solve_sparse_into_ws(factors, rhs, x_out, &mut ws)
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

        // D-block solve, fused into the forward pass (issue #126). A
        // node's eliminated rows (0..nelim) are final once its forward-sub
        // completes — ancestors only ever touch its separator rows — so
        // D⁻¹ can be applied here instead of in a second postorder pass,
        // saving one full gather/scatter sweep per solve. This mirrors the
        // multi-RHS core (`solve_sparse_core_many_into`), which has always
        // fused these. `d_diag`/`d_subdiag` are sized `nelim`; pivots
        // force-accepted as zero are skipped.
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
                // Skip iff force-zeroed (`d_diag == 0.0` exactly, L cleared);
                // divide by any live pivot, including a small-but-nonzero one
                // from rook rescue / static floor / F-01 band (issue #116 — the
                // old `|d| > zero_tol` gate silently dropped rook-accepted
                // sub-`zero_tol` pivots from the solution).
                if ff.d_diag[k] != 0.0 {
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

/// Single-RHS D-block solve on `w[0..nelim]` in place — the 1×1 / 2×2
/// disposition extracted from `solve_sparse_core_into` so the
/// contribution-block core (issue #131 Gap A) applies byte-identical
/// arithmetic. Force-accepted zero pivots and SSIDS-floor-rejected 2×2
/// blocks are left untouched, matching the shared-vector core.
#[inline]
fn dsolve_single(w: &mut [f64], ff: &crate::dense::factor::FrontalFactors, nelim: usize) {
    let mut k = 0;
    while k < nelim {
        if k + 1 < nelim && ff.d_subdiag[k] != 0.0 {
            let a = ff.d_diag[k];
            let b = ff.d_subdiag[k];
            let c = ff.d_diag[k + 1];
            if let Some((x0, x1)) = solve_2x2_dblock(a, b, c, w[k], w[k + 1]) {
                w[k] = x0;
                w[k + 1] = x1;
            }
            k += 2;
        } else {
            if ff.d_diag[k] != 0.0 {
                w[k] /= ff.d_diag[k];
            }
            k += 1;
        }
    }
}

// === Issue #131 Gap A: contribution-block (tree-parallel) sparse solve ==
//
// The shared-global-vector core (`solve_sparse_core_into`) accumulates
// each front's separator updates directly into a global `y`, so sibling
// subtrees read-modify-write the same ancestor entries — not
// tree-parallelisable, and a private-accumulation merge would break bit
// exactness (float non-associativity). The contribution-block core below
// instead assembles each front's RHS from `b` (at its eliminated rows)
// plus its children's contribution blocks, summed in a fixed child order
// — exactly the deterministic reduction the factor's `extend_add` uses.
// The reduction order is independent of thread scheduling, so a serial
// run and a tree-parallel run are byte-identical (the #131 contract).
//
// Forward substitution is leaves-up (a front assembles once all children
// have produced their contribution blocks); backward is root-down and
// keeps the shared-vector arithmetic unchanged (separator rows are
// read-only there, eliminated rows disjoint). Because the forward
// reduction groups contributions by subtree (a sum tree) rather than the
// flat postorder left-fold of `solve_sparse_core_into`, the result
// differs from that path only by floating-point reassociation (~κ·eps):
// a valid solve, offered as an opt-in path; the default `solve_sparse`
// is unchanged.
//
// Issue #177: because the two cores genuinely differ in the last bits,
// *which core runs must be an explicit caller decision* — see
// `SolveCore`. It previously fell out of `CbTaskPlan::worthwhile`, a
// predicate derived from `rayon::current_num_threads()`, so the same
// binary on a 4-core and a 32-core host solved with different
// arithmetic; an IPM host amplified that into different iterate
// trajectories (henon120, issues #177 and #16). The coarsening plan is
// now confined to scheduling: it decides `cb_run_parallel` vs
// `cb_run_serial`, and those are byte-identical.

/// One front's forward contribution: the `nrow - nelim` separator-row
/// values (in frontal separator order) it passes to its parent. Boxed
/// per node in a slot vector consumed leaves-up.
type FwdContrib = Vec<f64>;

/// Build per-node child lists (ascending node index) from the parent
/// table, for a deterministic child-reduction order in the CB forward.
fn build_children(node_parents: &[Option<usize>], n_nodes: usize) -> Vec<Vec<usize>> {
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (c, p) in node_parents.iter().enumerate() {
        if let Some(p) = p {
            if *p < n_nodes {
                children[*p].push(c);
            }
        }
    }
    // node_parents is filled by ascending child index, so each list is
    // already ascending; make it explicit for the determinism contract.
    for cs in &mut children {
        cs.sort_unstable();
    }
    children
}

/// Assemble + eliminate one front for the CB forward pass, writing the
/// eliminated finals into `y` (at the front's own eliminated global rows,
/// disjoint across fronts) and returning its contribution block (or
/// `None` if it has no separator rows). `row_map` is caller-owned scratch
/// (length `n`, all `usize::MAX` on entry and restored on exit); `w` is
/// caller-owned scratch (length ≥ `nrow`). `take_child` yields each
/// child's already-computed contribution in ascending child order.
#[allow(clippy::too_many_arguments)]
fn cb_forward_front(
    node: &NodeFactors,
    children: &[usize],
    y: &mut [f64],
    row_map: &mut [usize],
    w: &mut [f64],
    mut take_child: impl FnMut(usize) -> Option<FwdContrib>,
    child_nodes: &[NodeFactors],
) -> Option<FwdContrib> {
    let ff = &node.frontal_factors;
    let nrow = ff.nrow;
    let nelim = ff.nelim;
    if nrow == 0 {
        return None;
    }

    // global -> local frontal index for this front.
    for i in 0..nrow {
        row_map[node.row_indices[ff.perm[i]]] = i;
    }

    let w = &mut w[..nrow];
    // b at eliminated rows (still intact in `y` — each global row is
    // eliminated at exactly one front, written only here); separators 0.
    for i in 0..nrow {
        w[i] = 0.0;
    }
    for i in 0..nelim {
        w[i] = y[node.row_indices[ff.perm[i]]];
    }

    // Extend-add each child's contribution block in ascending order.
    for &child in children {
        if let Some(cc) = take_child(child) {
            let cff = &child_nodes[child].frontal_factors;
            let cnelim = cff.nelim;
            let cnrow = cff.nrow;
            for i in cnelim..cnrow {
                let g = child_nodes[child].row_indices[cff.perm[i]];
                let local = row_map[g];
                debug_assert!(
                    local != usize::MAX && local < nrow,
                    "CB forward: child separator row {g} not in parent frontal"
                );
                w[local] += cc[i - cnelim];
            }
        }
    }

    // L-solve over all rows (uses pre-D eliminated values), then D-solve
    // on the eliminated rows only — mirrors solve_sparse_core_into.
    for j in 0..nelim {
        let w_j = w[j];
        for i in (j + 1)..nrow {
            w[i] -= ff.l[j * nrow + i] * w_j;
        }
    }
    dsolve_single(w, ff, nelim);

    // Eliminated finals -> y (disjoint rows across fronts).
    for i in 0..nelim {
        y[node.row_indices[ff.perm[i]]] = w[i];
    }

    // Contribution block = separator rows after the L-update.
    let contrib = if nrow > nelim {
        Some(w[nelim..nrow].to_vec())
    } else {
        None
    };

    // Restore row_map.
    for i in 0..nrow {
        row_map[node.row_indices[ff.perm[i]]] = usize::MAX;
    }

    contrib
}

/// Backward substitution for one front, shared-vector arithmetic
/// (unchanged from `solve_sparse_core_into`): gather from `y`, Lᵀ-solve
/// the eliminated rows, scatter the eliminated rows back. Separator rows
/// are read-only here (they hold ancestors' final values, written by the
/// ancestors' backward step which runs first in root-down order), so
/// concurrent fronts touch disjoint `y` entries.
fn cb_backward_front(node: &NodeFactors, y: &mut [f64], w: &mut [f64]) {
    let ff = &node.frontal_factors;
    let nelim = ff.nelim;
    let nrow = ff.nrow;
    if nelim == 0 {
        return;
    }
    let w = &mut w[..nrow];
    for i in 0..nrow {
        w[i] = y[node.row_indices[ff.perm[i]]];
    }
    for j in (0..nelim).rev() {
        let mut sum = 0.0;
        for i in (j + 1)..nrow {
            sum += ff.l[j * nrow + i] * w[i];
        }
        w[j] -= sum;
    }
    // Only the eliminated rows changed; scatter just those (separators are
    // read-only, so this keeps concurrent fronts' writes disjoint).
    for i in 0..nelim {
        y[node.row_indices[ff.perm[i]]] = w[i];
    }
}

/// Serial contribution-block solve core (single RHS). Bit-identical to
/// the parallel CB core; the default `solve_sparse` path is unaffected.
/// Serial contribution-block forward+backward over `y` in place (`y`
/// holds the permuted RHS on entry, the permuted solution on exit),
/// against caller-owned workspace buffers. `contribs` must be all-`None`
/// and `row_map` all-`usize::MAX` on entry (both restored on exit —
/// `row_map` by each front, `contribs` by the parent that drains each
/// child; roots' contributions are left behind, so callers reset
/// `contribs` between solves).
fn cb_run_serial(
    nodes: &[NodeFactors],
    children: &[Vec<usize>],
    y: &mut [f64],
    row_map: &mut [usize],
    w: &mut [f64],
    contribs: &mut [Option<FwdContrib>],
) {
    // Forward: postorder (node_factors order lists children before
    // parents), so every child's contribution is ready at its parent.
    for idx in 0..nodes.len() {
        let contrib = cb_forward_front(
            &nodes[idx],
            &children[idx],
            y,
            row_map,
            w,
            |c| contribs[c].take(),
            nodes,
        );
        contribs[idx] = contrib;
    }
    // Backward: reverse postorder (parents before children).
    for node in nodes.iter().rev() {
        cb_backward_front(node, y, w);
    }
}

/// `*mut f64` wrapper that is `Send`+`Sync` so rayon tasks can write the
/// shared global solution vector `y` at **disjoint** indices. Safety is
/// established by the callers below, not by this type.
#[derive(Clone, Copy)]
struct YPtr(*mut f64);
// SAFETY: `YPtr` only ever hands out writes to disjoint `y` indices
// (each global row is eliminated at exactly one front, and a front writes
// only its own eliminated rows) and reads that are ordered after the
// writing front by the task-dependency graph, so no two threads touch the
// same element concurrently.
unsafe impl Send for YPtr {}
unsafe impl Sync for YPtr {}

/// Per-worker forward scratch: `row_map` (length `n`, all `usize::MAX`
/// between fronts) and `w` (length `max_nrow`).
struct CbScratch {
    row_map: Vec<usize>,
    w: Vec<f64>,
}

/// Parallel contribution-block forward+backward over `y` in place,
/// against caller-owned pooled buffers (`scratch` per worker, `contribs`
/// shared). Byte-identical to `cb_run_serial`: the child-reduction order
/// is fixed (ascending child index) regardless of thread scheduling. The
/// caller resets `contribs` to all-`None` and provides `scratch` with
/// `row_map` all-`usize::MAX`; both are left in that state on exit.
#[allow(clippy::too_many_arguments)]
fn cb_run_parallel(
    nodes: &[NodeFactors],
    children: &[Vec<usize>],
    parents: &[Option<usize>],
    plan: &CbTaskPlan,
    scratch: &[std::sync::Mutex<CbScratch>],
    contribs: &std::sync::Mutex<Vec<Option<FwdContrib>>>,
    y: &mut [f64],
    n: usize,
) {
    use std::sync::atomic::AtomicUsize;
    let num_threads = rayon::current_num_threads().max(1);

    // ---- Forward: task roots processed leaves-up ----
    let pending_fwd: Vec<AtomicUsize> = plan
        .pending_fwd
        .iter()
        .map(|&c| AtomicUsize::new(c))
        .collect();
    {
        let y_ptr = YPtr(y.as_mut_ptr());
        let ctx = CbCtx {
            nodes,
            children,
            parents,
            plan,
            contribs,
            scratch,
            y_ptr,
            n,
            num_threads,
        };
        rayon::scope(|scope| {
            for &t in &plan.fwd_seeds {
                cb_fwd_task(scope, t, &ctx, &pending_fwd);
            }
        });
    }

    // ---- Backward: task roots processed root-down ----
    {
        let y_ptr = YPtr(y.as_mut_ptr());
        let ctx = CbCtx {
            nodes,
            children,
            parents,
            plan,
            contribs,
            scratch,
            y_ptr,
            n,
            num_threads,
        };
        rayon::scope(|scope| {
            for &t in &plan.bwd_seeds {
                cb_bwd_task(scope, t, &ctx);
            }
        });
    }
}

/// Coarsening plan: the top portion of the assembly tree is cut into
/// "task roots"; each task root owns the contiguous postorder block of
/// its light (below-threshold) descendants and runs them serially.
struct CbTaskPlan {
    /// `owned[t]` = node indices assigned to task root `t`, ascending
    /// (postorder). Empty for non-task-root nodes.
    owned: Vec<Vec<usize>>,
    /// Task roots that have no task-root children (forward seeds).
    fwd_seeds: Vec<usize>,
    /// Task roots that have no parent (whole-tree roots; backward seeds).
    bwd_seeds: Vec<usize>,
    /// Direct task-root children of each task root (backward spawns).
    tr_children: Vec<Vec<usize>>,
    /// Forward pending count per task root = number of task-root children.
    pending_fwd: Vec<usize>,
    /// Whether each node is a task root.
    is_task_root: Vec<bool>,
    /// The shape half of the gate — [`cb_gate_shape`] on this plan's
    /// terms. Recorded so `cb_core_profitable_matches_the_plan_gate` can
    /// pin it against the flat reimplementation that decides the *core*
    /// (issue #177); `worthwhile` is this **and** the overhead half.
    #[cfg(test)]
    shape_ok: bool,
    /// Σ `own_cost` over every front — the accumulator both halves of the
    /// gate are computed from. Recorded so the #175 regression tests can
    /// assert against the definition the builder actually used instead of
    /// recomputing `nrow·(nelim+1)` themselves; a hand copy would keep
    /// passing against a stale formula if `own_cost` ever changed.
    #[cfg(test)]
    total: u64,
    /// Whether tree-parallel *execution* is expected to beat running the
    /// same CB core serially: the shape terms of [`cb_gate_shape`] —
    /// at least two independent task roots, enough total work, no single
    /// task root's serial share dominating (Amdahl) — **and** the
    /// per-front overhead term of [`cb_sync_amortized`] (issue #175).
    ///
    /// Issue #177: this gates `cb_run_parallel` vs `cb_run_serial` only —
    /// two byte-identical cores. It must never be used to choose between
    /// the CB core and the shared-vector core, because it is derived from
    /// `rayon::current_num_threads()` and would make the arithmetic a
    /// function of the host's core count.
    worthwhile: bool,
}

/// ~1e6 flops ≈ tens of µs of solve, the floor below which the
/// O(threads·n) scratch alloc + rayon scope is not amortized.
const MIN_TOTAL_COST: u64 = 1_000_000;

/// Amdahl ceiling: if one task root runs this share of the total work by
/// itself, parallel overhead dominates whatever the rest of the tree does
/// (e.g. an arrow front whose huge root front is one serial task).
const MAX_LOCAL_SHARE: f64 = 0.7;

/// Minimum average work per **front**, in `own_cost` units
/// (`nrow·(nelim+1)`), for tree-parallel execution to pay — about an 8×8
/// front (issue #175).
///
/// `cb_run_parallel` pays a fixed synchronization cost *per front*: the
/// shared `contribs` mutex is taken once per child drained and once to
/// store the front's own block, inside the per-front loop. That cost
/// scales with the number of supernodes, while [`MIN_TOTAL_COST`] is a
/// floor on *total* work — so a wide, extremely sparse tree can clear
/// every shape term and still spend the solve synchronizing. That is
/// what #175 reports on the Mittelmann KKT `NARX_CFy`: 45,736
/// supernodes, Lagrangian Hessian nnz 19,851, ~30 cost units per front,
/// 15% of the IPM run and ~3.0M involuntary context switches lost to the
/// tree-parallel solve on a 14-core host.
///
/// 64 is the measured break-even (4-core container, pooled CB core,
/// serial vs tree-parallel at 2/4/8 workers, geometric mean over two
/// runs — `issue175_cb_gate_calibration` reproduces it):
///
/// | work/front | 25 | 28† | 53 | 74 | 103 | 202† | 235 | 305 |
/// |---|---:|---:|---:|---:|---:|---:|---:|---:|
/// | par/ser | 1.08 | 1.02 | 0.98 | 0.91 | 0.81 | 0.73 | 0.75 | 0.63 |
///
/// † **not reachable in production.** The harness force-sets
/// `plan.worthwhile` so it can time both arms on any fixture, and two of
/// its eight fixtures are rejected by an *earlier* term than this one:
/// `narx_w2` (28 units/front, total 958,763) and `poisson_96` (202,
/// total 351,544) both fall under [`MIN_TOTAL_COST`], so `shape_ok` is
/// already false and no per-front floor can change their fate. They are
/// listed because the calibration measured them, not as evidence.
///
/// On the six reachable points — 25, 53, 74, 103, 235, 305 — everything
/// at or below 53 is a wash or a loss and everything at or above 74
/// pays, so the break-even the constant is set from does not depend on
/// the two unreachable columns. See
/// `dev/research/issue-175-cb-solve-gate-overhead.md`.
const MIN_COST_PER_FRONT: u64 = 64;

/// The shape half of the CB gate: at least two independent task roots,
/// enough total work to amortize the `O(threads·n)` scratch setup, and no
/// single task root's serial share dominating the tree (Amdahl).
///
/// Shared by [`CbTaskPlan::worthwhile`] (scheduling) and
/// [`cb_core_profitable`] (core choice, host-independent) so the two
/// cannot drift; the overhead half below is scheduling-only.
#[inline]
fn cb_gate_shape(fwd_seeds: usize, total: u64, max_local: u64) -> bool {
    fwd_seeds >= 2 && total >= MIN_TOTAL_COST && (max_local as f64) < MAX_LOCAL_SHARE * total as f64
}

/// The overhead half of the CB gate (issue #175): is there enough work
/// per front to amortize `cb_run_parallel`'s per-front synchronization?
///
/// Scheduling-only. `cb_core_profitable` must **not** consult this: it
/// chooses between two numerically distinct cores, and the two answers
/// this predicate separates are byte-identical (issue #177).
#[inline]
fn cb_sync_amortized(total: u64, n_nodes: usize) -> bool {
    total >= MIN_COST_PER_FRONT.saturating_mul(n_nodes as u64)
}

/// Reference fan-out for the host-independent coarsening (issue #177):
/// 16 task roots per worker on a canonical 4-worker host. Used only to
/// judge whether the CB core suits a factor at all — a verdict that must
/// not vary with the host — never to schedule real work.
const CB_REFERENCE_FANOUT: u64 = 64;

/// How the coarsening threshold — the subtree cost at or above which a
/// node becomes its own task root — is chosen.
#[derive(Debug, Clone, Copy)]
enum CbThreshold {
    /// Aim for ~16 task roots per worker so the pool stays fed without
    /// per-node overhead; `FERAL_CB_THRESH` overrides. Depends on the
    /// host, so it may drive **scheduling only** (issue #177).
    FromWorkers,
    /// A fixed [`CB_REFERENCE_FANOUT`]-way cut, identical on every host.
    /// The basis for [`cb_core_profitable`].
    Reference,
    /// Explicit, for the tests that sweep the threshold.
    #[cfg(test)]
    Fixed(u64),
}

impl CbThreshold {
    fn resolve(self, total: u64) -> u64 {
        match self {
            CbThreshold::FromWorkers => {
                let num_threads = rayon::current_num_threads().max(1);
                // Parsed through `crate::env` so `1e18` means 1e18 and a
                // typo warns instead of silently restoring the default
                // (issue #176).
                crate::env::u64_var("FERAL_CB_THRESH")
                    .unwrap_or_else(|| (total / (num_threads as u64 * 16)).max(1))
            }
            CbThreshold::Reference => (total / CB_REFERENCE_FANOUT).max(1),
            #[cfg(test)]
            CbThreshold::Fixed(t) => t,
        }
    }
}

impl CbTaskPlan {
    fn build_with_threshold(
        children: &[Vec<usize>],
        parents: &[Option<usize>],
        nodes: &[NodeFactors],
        n_nodes: usize,
        threshold: CbThreshold,
    ) -> Self {
        // Per-front solve cost ~ nrow·(nelim+1) (forward L-solve +
        // backward Lᵀ-solve are both O(nrow·nelim)); saturating.
        let own_cost = |i: usize| -> u64 {
            let ff = &nodes[i].frontal_factors;
            (ff.nrow as u64).saturating_mul(ff.nelim as u64 + 1)
        };
        // Subtree cost bottom-up (node_factors is postorder ⇒ children
        // precede parents).
        let mut subtree_cost = vec![0u64; n_nodes];
        let mut total = 0u64;
        for i in 0..n_nodes {
            let mut c = own_cost(i);
            for &ch in &children[i] {
                c = c.saturating_add(subtree_cost[ch]);
            }
            subtree_cost[i] = c;
            total = total.saturating_add(own_cost(i));
        }
        let thresh: u64 = threshold.resolve(total);

        let is_task_root: Vec<bool> = (0..n_nodes)
            .map(|i| parents[i].is_none() || subtree_cost[i] >= thresh)
            .collect();

        // Owner = nearest task-root ancestor (self if task root). Process
        // descending idx so each parent (higher idx) is set first.
        let mut owner = vec![0usize; n_nodes];
        for i in (0..n_nodes).rev() {
            // A non-task-root always has a parent (roots are forced task
            // roots), whose owner is already computed (higher idx). The
            // `unwrap_or(i)` is an unreachable belt-and-braces fallback
            // that keeps this `unwrap`-free.
            owner[i] = match (is_task_root[i], parents[i]) {
                (true, _) => i,
                (false, Some(p)) => owner[p],
                (false, None) => i,
            };
        }
        let mut owned: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for i in 0..n_nodes {
            owned[owner[i]].push(i);
        }

        // Task-root children (a task root's parent is always a task root,
        // by subtree-cost monotonicity, so these are direct children).
        let mut tr_children: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        let mut pending_fwd = vec![0usize; n_nodes];
        for i in 0..n_nodes {
            if !is_task_root[i] {
                continue;
            }
            for &ch in &children[i] {
                if is_task_root[ch] {
                    tr_children[i].push(ch);
                    pending_fwd[i] += 1;
                }
            }
        }
        let fwd_seeds: Vec<usize> = (0..n_nodes)
            .filter(|&i| is_task_root[i] && pending_fwd[i] == 0)
            .collect();
        let bwd_seeds: Vec<usize> = (0..n_nodes)
            .filter(|&i| is_task_root[i] && parents[i].is_none())
            .collect();

        // Worthwhile gate. `local_cost[t] = subtree_cost[t] − Σ task-root
        // children subtree_cost` is the serial work task root `t` runs
        // itself; if the largest such share is most of the total, Amdahl
        // caps the speedup and the parallel overhead dominates (e.g. an
        // arrow front whose huge root front is one serial task). Also
        // require enough total work to amortize the scratch setup.
        let mut max_local = 0u64;
        for t in 0..n_nodes {
            if !is_task_root[t] {
                continue;
            }
            let mut local = subtree_cost[t];
            for &c in &tr_children[t] {
                local = local.saturating_sub(subtree_cost[c]);
            }
            max_local = max_local.max(local);
        }
        let shape_ok = cb_gate_shape(fwd_seeds.len(), total, max_local);
        // Issue #175: the shape terms alone accept wide, extremely sparse
        // trees whose fronts are too small to pay for the per-front
        // synchronization `cb_run_parallel` does.
        let worthwhile = shape_ok && cb_sync_amortized(total, n_nodes);

        CbTaskPlan {
            owned,
            fwd_seeds,
            bwd_seeds,
            tr_children,
            pending_fwd,
            is_task_root,
            #[cfg(test)]
            shape_ok,
            #[cfg(test)]
            total,
            worthwhile,
        }
    }
}

/// Whether the contribution-block core suits this factor at all — a
/// **host-independent** verdict (issue #177).
///
/// The two solve cores use different, equally valid reassociations, so
/// which one runs is numerically visible and must not be decided by
/// anything the host controls. This builds the coarsening at the fixed
/// [`CB_REFERENCE_FANOUT`] granularity — never from
/// `rayon::current_num_threads()` and never from `FERAL_CB_THRESH` — and
/// applies [`cb_gate_shape`], the shape half of `CbTaskPlan`'s gate.
///
/// Issue #175 added a second, scheduling-only half
/// ([`cb_sync_amortized`]) to `CbTaskPlan::worthwhile`. It is
/// deliberately **not** applied here: it separates two byte-identical
/// executions of one core, whereas this predicate separates two cores
/// with different arithmetic, so folding it in would silently change
/// which reassociation wide-sparse factors solve with.
///
/// Measured on this repo's fixtures (4 workers, refined solve, time
/// relative to the shared-vector core): the CB core costs 1.27–1.86x on
/// the trees this predicate rejects (path-like chains, small grids) and
/// returns 0.72x on the bushy grids it accepts. Rejecting is therefore
/// not merely conservative — it is the faster answer on those trees, and
/// it stays the faster answer on a host with no workers at all.
/// Runs on every refined solve, including the ones it turns down, so it
/// is computed on flat `O(n_nodes)` arrays rather than by building a
/// `CbTaskPlan` — that allocates three `Vec<Vec<usize>>` of length
/// `n_nodes` and cost 1.24-1.29x of a whole chain solve when the answer
/// is "no". `cb_core_profitable_matches_the_plan_gate` pins this against
/// `CbTaskPlan::worthwhile` so the two cannot drift.
fn cb_core_profitable(factors: &SparseFactors) -> bool {
    let nodes = &factors.node_factors;
    let parents = &factors.node_parents;
    let n_nodes = nodes.len();
    if n_nodes == 0 {
        return false;
    }
    let own_cost = |i: usize| -> u64 {
        let ff = &nodes[i].frontal_factors;
        (ff.nrow as u64).saturating_mul(ff.nelim as u64 + 1)
    };
    // Subtree cost bottom-up. `node_factors` is postorder, so a node's
    // own total is complete by the time we fold it into its parent — no
    // child lists needed.
    let mut subtree_cost: Vec<u64> = (0..n_nodes).map(own_cost).collect();
    let mut total = 0u64;
    for i in 0..n_nodes {
        total = total.saturating_add(own_cost(i));
        if let Some(p) = parents[i] {
            subtree_cost[p] = subtree_cost[p].saturating_add(subtree_cost[i]);
        }
    }
    let thresh = CbThreshold::Reference.resolve(total);

    // `local[t]` = the work task root `t` runs itself = its subtree minus
    // the subtrees of its task-root children. A task root's parent is
    // itself a task root by subtree-cost monotonicity, so every task root
    // with a parent is a task-root child of it.
    let is_task_root = |i: usize| parents[i].is_none() || subtree_cost[i] >= thresh;
    let mut local = subtree_cost.clone();
    let mut tr_children = vec![0usize; n_nodes];
    for i in 0..n_nodes {
        if !is_task_root(i) {
            continue;
        }
        if let Some(p) = parents[i] {
            tr_children[p] += 1;
            local[p] = local[p].saturating_sub(subtree_cost[i]);
        }
    }
    let mut fwd_seeds = 0usize;
    let mut max_local = 0u64;
    for t in 0..n_nodes {
        if !is_task_root(t) {
            continue;
        }
        if tr_children[t] == 0 {
            fwd_seeds += 1;
        }
        max_local = max_local.max(local[t]);
    }
    cb_gate_shape(fwd_seeds, total, max_local)
}

/// Shared, read-only context threaded through the CB solve tasks.
#[derive(Clone, Copy)]
struct CbCtx<'a> {
    nodes: &'a [NodeFactors],
    children: &'a [Vec<usize>],
    parents: &'a [Option<usize>],
    plan: &'a CbTaskPlan,
    contribs: &'a std::sync::Mutex<Vec<Option<FwdContrib>>>,
    scratch: &'a [std::sync::Mutex<CbScratch>],
    y_ptr: YPtr,
    n: usize,
    num_threads: usize,
}

/// Pick a per-worker scratch slot: the rayon worker index, or the extra
/// slot `num_threads` for the calling thread (`current_thread_index() ==
/// None`).
#[inline]
fn cb_scratch_slot(num_threads: usize) -> usize {
    rayon::current_thread_index().unwrap_or(num_threads)
}

fn cb_fwd_task<'a>(
    scope: &rayon::Scope<'a>,
    t: usize,
    ctx: &CbCtx<'a>,
    pending_fwd: &'a [std::sync::atomic::AtomicUsize],
) {
    use std::sync::atomic::Ordering;
    let ctx = *ctx;
    scope.spawn(move |s| {
        {
            let slot = cb_scratch_slot(ctx.num_threads);
            let mut sc = ctx.scratch[slot].lock().unwrap_or_else(|p| p.into_inner());
            let CbScratch { row_map, w } = &mut *sc;
            // SAFETY: every front in this task writes only its own
            // eliminated global rows — disjoint across all fronts (each row
            // is eliminated once) — and reads `b` only at those same rows,
            // which no other live task touches. Length `n`; row indices are
            // in `[0, n)`.
            let y: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(ctx.y_ptr.0, ctx.n) };
            // Process this task root's owned subtree serially, leaves-up
            // (owned is ascending = postorder), so every child's
            // contribution — light (in `owned`, already processed) or a
            // task-root child (done via the pending gate) — is ready.
            for &j in &ctx.plan.owned[t] {
                let contrib = cb_forward_front(
                    &ctx.nodes[j],
                    &ctx.children[j],
                    y,
                    row_map,
                    w,
                    |c| ctx.contribs.lock().unwrap_or_else(|p| p.into_inner())[c].take(),
                    ctx.nodes,
                );
                ctx.contribs.lock().unwrap_or_else(|p| p.into_inner())[j] = contrib;
            }
        }
        // Signal the parent task root; spawn it when all its task-root
        // children are done.
        if let Some(p) = ctx.parents[t] {
            debug_assert!(ctx.plan.is_task_root[p]);
            if pending_fwd[p].fetch_sub(1, Ordering::AcqRel) == 1 {
                cb_fwd_task(s, p, &ctx, pending_fwd);
            }
        }
    });
}

fn cb_bwd_task<'a>(scope: &rayon::Scope<'a>, t: usize, ctx: &CbCtx<'a>) {
    let ctx = *ctx;
    scope.spawn(move |s| {
        {
            let slot = cb_scratch_slot(ctx.num_threads);
            let mut sc = ctx.scratch[slot].lock().unwrap_or_else(|p| p.into_inner());
            // SAFETY: every front writes only its own eliminated global
            // rows (disjoint across fronts) and reads separator rows
            // written by ancestor fronts — those in this task's owned set
            // (processed earlier, this task) or in the parent task root
            // (completed before this task was spawned, root-down
            // dependency). Length `n`; row indices in `[0, n)`.
            let y: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(ctx.y_ptr.0, ctx.n) };
            // Root-down within the owned subtree: descending idx = reverse
            // postorder (parents before children).
            for &j in ctx.plan.owned[t].iter().rev() {
                cb_backward_front(&ctx.nodes[j], y, &mut sc.w);
            }
        }
        // This task root is done ⇒ its task-root children may run.
        for &c in &ctx.plan.tr_children[t] {
            cb_bwd_task(s, c, &ctx);
        }
    });
}

/// Pooled workspace for the contribution-block solve (issue #131 Gap A).
/// Caches the RHS-independent state — the assembly-tree child lists and
/// the coarsening plan — plus the per-solve scratch, so a caller doing
/// several solves against one factor (iterative refinement runs up to
/// ~11) pays the `O(threads·n)` scratch and `O(n_nodes)` plan setup once.
/// Build with [`CbSolveWorkspace::for_factors`]; reuse across solves.
pub(crate) struct CbSolveWorkspace {
    n: usize,
    max_nrow: usize,
    scaled: bool,
    children: Vec<Vec<usize>>,
    plan: CbTaskPlan,
    y: Vec<f64>,
    // Serial scratch.
    row_map: Vec<usize>,
    w: Vec<f64>,
    contribs: Vec<Option<FwdContrib>>,
    // Parallel scratch, built lazily on first parallel solve and rebuilt
    // only if the worker count changes.
    par_scratch: Vec<std::sync::Mutex<CbScratch>>,
    par_contribs: std::sync::Mutex<Vec<Option<FwdContrib>>>,
}

impl CbSolveWorkspace {
    pub(crate) fn for_factors(factors: &SparseFactors) -> Self {
        Self::for_factors_with_threshold(factors, CbThreshold::FromWorkers)
    }

    /// `threshold` picks the coarsening granularity. Only the task
    /// decomposition changes; the arithmetic is invariant (issue #177),
    /// which is what `cb_coarsening_threshold_is_arithmetically_inert`
    /// asserts.
    fn for_factors_with_threshold(factors: &SparseFactors, threshold: CbThreshold) -> Self {
        let n = factors.n;
        let nodes = &factors.node_factors;
        let n_nodes = nodes.len();
        let children = build_children(&factors.node_parents, n_nodes);
        let plan = CbTaskPlan::build_with_threshold(
            &children,
            &factors.node_parents,
            nodes,
            n_nodes,
            threshold,
        );
        let max_nrow = nodes
            .iter()
            .map(|nd| nd.frontal_factors.nrow)
            .max()
            .unwrap_or(0);
        let scaled = !matches!(factors.scaling_info, ScalingInfo::NotApplied);
        Self {
            n,
            max_nrow,
            scaled,
            children,
            plan,
            y: vec![0.0; n],
            row_map: vec![usize::MAX; n],
            w: vec![0.0; max_nrow],
            contribs: (0..n_nodes).map(|_| None).collect(),
            par_scratch: Vec::new(),
            par_contribs: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Whether tree-parallel execution is expected to beat running the
    /// same CB core serially on this factor (see `CbTaskPlan::worthwhile`).
    /// A scheduling predicate: both answers produce identical bits, which
    /// is why nothing outside the CB core is allowed to branch on it
    /// (issue #177) — hence test-only visibility.
    #[cfg(test)]
    pub(crate) fn worthwhile(&self) -> bool {
        self.plan.worthwhile
    }

    /// Number of task roots the coarsening produced — the size of the
    /// task decomposition. Reported so the #177 threshold sweep can show
    /// the plan really changed while the output did not.
    #[cfg(test)]
    pub(crate) fn task_root_count(&self) -> usize {
        self.plan.is_task_root.iter().filter(|b| **b).count()
    }

    /// Solve `A x = b` into `x_out` using this workspace. `parallel`
    /// requests tree-parallel execution, honoured only when the plan is
    /// `worthwhile`; otherwise the byte-identical serial core runs — the
    /// result is the same bits either way, so neither the `parallel`
    /// argument nor the worker count is observable in the output. MC64
    /// pre/post scaling is fused into the entry permute and exit unpermute.
    pub(crate) fn solve_into(
        &mut self,
        factors: &SparseFactors,
        rhs: &[f64],
        x_out: &mut [f64],
        parallel: bool,
    ) {
        let n = self.n;
        // Permute the RHS into `y`, fusing MC64 scaling: y[new] =
        // b[old]·D[old] (the same congruence solve_sparse applies).
        if self.scaled {
            for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
                self.y[new_idx] = rhs[old_idx] * factors.scaling[old_idx];
            }
        } else {
            for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
                self.y[new_idx] = rhs[old_idx];
            }
        }

        if parallel && self.plan.worthwhile {
            let num_threads = rayon::current_num_threads().max(1);
            if self.par_scratch.len() != num_threads + 1 {
                let max_nrow = self.max_nrow;
                self.par_scratch = (0..num_threads + 1)
                    .map(|_| {
                        std::sync::Mutex::new(CbScratch {
                            row_map: vec![usize::MAX; n],
                            w: vec![0.0; max_nrow],
                        })
                    })
                    .collect();
            }
            {
                let mut pc = self.par_contribs.lock().unwrap_or_else(|p| p.into_inner());
                pc.clear();
                pc.resize_with(factors.node_factors.len(), || None);
            }
            cb_run_parallel(
                &factors.node_factors,
                &self.children,
                &factors.node_parents,
                &self.plan,
                &self.par_scratch,
                &self.par_contribs,
                &mut self.y,
                n,
            );
        } else {
            // Roots' contributions are never drained, so reset.
            for c in self.contribs.iter_mut() {
                *c = None;
            }
            cb_run_serial(
                &factors.node_factors,
                &self.children,
                &mut self.y,
                &mut self.row_map,
                &mut self.w,
                &mut self.contribs,
            );
        }

        // Unpermute, fusing the MC64 post-scale: x[old] = y[new]·D[old].
        if self.scaled {
            for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
                x_out[old_idx] = self.y[new_idx] * factors.scaling[old_idx];
            }
        } else {
            for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
                x_out[old_idx] = self.y[new_idx];
            }
        }
    }
}

/// Opt-in contribution-block sparse solve (issue #131 Gap A), single RHS.
/// Applies the same MC64 pre/post scaling as `solve_sparse`. `parallel`
/// selects tree-parallel execution (honoured when the tree is
/// `worthwhile`); serial and parallel execution of this core are
/// byte-identical, and so are runs on hosts with different core counts.
///
/// That byte-identity is *within* this core. It is not a claim that this
/// core agrees with [`solve_sparse`]: the two use different, equally
/// valid reassociations (see the module notes above and [`SolveCore`]),
/// and issue #177 records what went wrong when a runtime gate was allowed
/// to pick between them. Allocates a one-shot [`CbSolveWorkspace`];
/// callers doing repeated solves against one factor should build and
/// reuse a workspace instead.
pub fn solve_sparse_cb(
    factors: &SparseFactors,
    rhs: &[f64],
    parallel: bool,
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
    let mut ws = CbSolveWorkspace::for_factors(factors);
    let mut x = vec![0.0f64; n];
    ws.solve_into(factors, rhs, &mut x, parallel);
    Ok(x)
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
                // Skip iff force-zeroed (`d_diag == 0.0` exactly); divide by
                // any live pivot, including rook/static/F-01 sub-`zero_tol`
                // ones (issue #116). Mirrors the single-RHS gate above.
                if ff.d_diag[k] != 0.0 {
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

/// Default cap on iterative-refinement **correction** steps: 10.
///
/// Chosen from FERAL's own corpus, not inherited: below 10, some
/// near-rank-deficient KKT matrices (CERI651C/ELS, HAHN1, MEYER3NE)
/// bounce in and out of the machine-precision basin before settling — see
/// `dev/journal/2026-04-18-06.org`. Issue #178 makes the cap settable
/// per call but explicitly does **not** change this default.
pub const DEFAULT_REFINE_MAX_STEPS: usize = 10;

/// Per-call knobs for the sparse iterative-refinement entry points.
///
/// `Default` reproduces FERAL's historical behavior exactly, so
/// `solve_sparse_refined(a, f, b)` and
/// `solve_sparse_refined_opts(a, f, b, RefineOptions::default())` are
/// bit-for-bit identical.
///
/// # Why a cap exists (issue #178)
///
/// The 10-step budget is right for a caller that solves `Ax = b` and
/// keeps the answer. It is wrong for a caller that is *itself* running
/// iterative refinement over the same system — an interior-point method
/// is exactly that. Ipopt's `PDFullSpaceSolver` computes the residual
/// after each of its own back-solves and decides from it whether to
/// continue; when each of those back-solves is a `solve_refined`, the
/// two loops nest and one augmented-system solve can cost up to
/// `10 × 11 = 110` substitution passes. The outer loop owns the
/// convergence criterion; the inner loop drives a residual nobody
/// consults toward a tolerance nobody set. Measured on a 118 276 × 118 276
/// augmented system, that inner loop was 60 % of back-solve time
/// (147.3 s vs 58.3 s) — see `dev/research/refinement-cap-2026-08-19.md`.
///
/// # Semantics
///
/// `max_steps` counts *corrections*, not total substitution passes: the
/// initial solve always happens, so a call runs at most `1 + max_steps`
/// passes. This matches how [`RefinementDiagnostics`] numbers its steps
/// (`steps[0]` is the unrefined solve), so a run under `max_steps = k`
/// yields at most `k + 1` entries.
///
/// It is a **cap, not a target**. The `ε·√n` relative-residual target,
/// the 100× divergence guard, and the 2-strike plateau exit all keep
/// priority, so raising `max_steps` can never add work to a system that
/// has already converged. And the best-iterate contract is preserved
/// under every value: the returned `x` is the iterate with the smallest
/// `‖r‖₂` seen, which always includes the unrefined solve, so no cap can
/// return an answer worse than [`solve_sparse`]'s.
///
/// `max_steps = 0` is the unrefined solve, bit-for-bit — and costs the
/// same, since the residual matvec is skipped rather than computed and
/// discarded. (Under the diagnostics entry point the matvec still runs:
/// `steps[0]` is the point there.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefineOptions {
    /// Maximum number of correction steps. See the type-level docs.
    pub max_steps: usize,
}

impl Default for RefineOptions {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_REFINE_MAX_STEPS,
        }
    }
}

impl RefineOptions {
    /// Options capping refinement at `max_steps` correction steps.
    ///
    /// `RefineOptions::with_max_steps(1)` is the interior-point host's
    /// case: one correction per call, leaving the convergence decision to
    /// the host's own refinement loop.
    pub fn with_max_steps(max_steps: usize) -> Self {
        Self { max_steps }
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
/// machine precision) or after 10 steps. 10 comes from FERAL's corpus,
/// not from MUMPS (whose `ICNTL(10)` default is `0` — no refinement at
/// all; the comparison harness sets it to `2`): below 10 some
/// near-rank-deficient KKT matrices
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
    solve_sparse_refined_opts(matrix, factors, rhs, RefineOptions::default())
}

/// [`solve_sparse_refined`] with a caller-supplied correction-step cap
/// (issue #178). `RefineOptions::default()` is bit-for-bit identical to
/// [`solve_sparse_refined`]; see [`RefineOptions`] for what the cap does
/// and does not override.
pub fn solve_sparse_refined_opts(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    opts: RefineOptions,
) -> Result<Vec<f64>, FeralError> {
    let mut x = vec![0.0; factors.n];
    solve_sparse_refined_core(
        matrix,
        factors,
        rhs,
        &mut x,
        false,
        SolveCore::SharedVector,
        opts,
    )?;
    Ok(x)
}

/// In-place form of [`solve_sparse_refined_opts`]: writes the
/// best-residual iterate into `x_out` (issue #178 item 2).
///
/// `x_out` *is* the best-iterate storage, so this saves the refiner an
/// `n`-length allocation on top of saving the caller the copy-back.
/// Returns [`FeralError::DimensionMismatch`] when `x_out.len() != factors.n`.
pub fn solve_sparse_refined_into(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    x_out: &mut [f64],
    opts: RefineOptions,
) -> Result<(), FeralError> {
    solve_sparse_refined_core(
        matrix,
        factors,
        rhs,
        x_out,
        false,
        SolveCore::SharedVector,
        opts,
    )?;
    Ok(())
}

/// Iterative refinement through the contribution-block solve core (issue
/// #131 Gap A) for the initial and correction solves, pooling one
/// `CbSolveWorkspace` across them.
///
/// `parallel` requests tree-parallel execution over the current rayon
/// pool; the CB core self-gates per factor (`CbTaskPlan::worthwhile`) and
/// runs single-threaded on path-like / small trees where the scope
/// overhead would not pay. **Neither `parallel` nor the gate's verdict
/// changes a single result bit** — the child-reduction order is fixed at
/// ascending child index in both. Pass `parallel: false` when no worker
/// pool is available and you still need the CB core's arithmetic (issue
/// #154's pool-less fallback, issue #177).
///
/// The refined result is a valid solution equal to
/// `solve_sparse_refined`'s up to floating-point reassociation: the CB
/// forward groups contributions by subtree, the shared-vector core folds
/// them in flat postorder. Choosing between the two cores is therefore a
/// numerically visible decision and belongs to the caller — see
/// [`SolveCore`].
pub fn solve_sparse_refined_cb(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    parallel: bool,
) -> Result<Vec<f64>, FeralError> {
    solve_sparse_refined_cb_opts(matrix, factors, rhs, parallel, RefineOptions::default())
}

/// [`solve_sparse_refined_cb`] with a caller-supplied correction-step cap
/// (issue #178).
pub fn solve_sparse_refined_cb_opts(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    parallel: bool,
    opts: RefineOptions,
) -> Result<Vec<f64>, FeralError> {
    let mut x = vec![0.0; factors.n];
    solve_sparse_refined_core(
        matrix,
        factors,
        rhs,
        &mut x,
        false,
        SolveCore::ContribBlock { parallel },
        opts,
    )?;
    Ok(x)
}

/// Iterative refinement through whichever solve core suits the factor,
/// chosen by [`SolveCore::Auto`] — the CB core on bushy trees where it
/// pays, the shared-vector core elsewhere.
///
/// The choice is a pure function of the factor: the same factor solves
/// with the same arithmetic on a single-core host and a 64-core one, with
/// or without a thread pool, whatever `FERAL_CB_THRESH` says (issue
/// #177). `parallel` requests tree-parallel execution when the CB core
/// was chosen, and cannot move a result bit.
///
/// This is what [`crate::numeric::solver::Solver::solve_refined`] calls.
pub fn solve_sparse_refined_auto(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    parallel: bool,
) -> Result<Vec<f64>, FeralError> {
    solve_sparse_refined_auto_opts(matrix, factors, rhs, parallel, RefineOptions::default())
}

/// [`solve_sparse_refined_auto`] with a caller-supplied correction-step
/// cap (issue #178).
pub fn solve_sparse_refined_auto_opts(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    parallel: bool,
    opts: RefineOptions,
) -> Result<Vec<f64>, FeralError> {
    let mut x = vec![0.0; factors.n];
    solve_sparse_refined_core(
        matrix,
        factors,
        rhs,
        &mut x,
        false,
        SolveCore::Auto { parallel },
        opts,
    )?;
    Ok(x)
}

/// In-place form of [`solve_sparse_refined_auto_opts`] (issue #178 item
/// 2): writes the best-residual iterate into `x_out`.
pub fn solve_sparse_refined_auto_into(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    x_out: &mut [f64],
    parallel: bool,
    opts: RefineOptions,
) -> Result<(), FeralError> {
    solve_sparse_refined_core(
        matrix,
        factors,
        rhs,
        x_out,
        false,
        SolveCore::Auto { parallel },
        opts,
    )?;
    Ok(())
}

/// Iterative refinement with tree-parallel execution requested, and the
/// core chosen from the factor's structure — exactly
/// [`solve_sparse_refined_auto`]`(.., parallel: true)`, which this
/// forwards to.
///
/// # Relationship to the other entry points
///
/// This is the pre-#177 name. It predates the separation of *which core*
/// (a numerical decision) from *how it executes* (a scheduling one), so
/// its name says "parallel" while its behavior is "let the factor pick
/// the core, and run it over the pool if that lands on the CB core".
/// Prefer [`solve_sparse_refined_auto`] in new code, which says that
/// outright, or [`solve_sparse_refined_cb`] when you want the
/// contribution-block core regardless of what the factor's shape
/// suggests.
///
/// # Why it is not the CB core unconditionally
///
/// Through v0.16.0 this function built a `CbSolveWorkspace` and used it
/// only when `worthwhile()` held, falling back to the shared-vector core
/// otherwise. Issue #177 removed that gate because it read
/// `rayon::current_num_threads()` and `FERAL_CB_THRESH`, so the same
/// factor solved with different arithmetic on different hosts.
/// [`SolveCore::Auto`] is the host-independent replacement: it keeps the
/// fallback (from the factor's shape alone) rather than dropping it.
/// Routing this function to the CB core unconditionally instead would
/// have cost 1.86x on the small path-like factors the gate exists to
/// protect (poisson_40, refined solve: 659 µs → 1228 µs) — the
/// alternative issue #177 measured and rejected.
pub fn solve_sparse_refined_parallel(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
) -> Result<Vec<f64>, FeralError> {
    solve_sparse_refined_parallel_opts(matrix, factors, rhs, RefineOptions::default())
}

/// [`solve_sparse_refined_parallel`] with a caller-supplied
/// correction-step cap (issue #178).
pub fn solve_sparse_refined_parallel_opts(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    opts: RefineOptions,
) -> Result<Vec<f64>, FeralError> {
    let mut x = vec![0.0; factors.n];
    solve_sparse_refined_core(
        matrix,
        factors,
        rhs,
        &mut x,
        false,
        SolveCore::Auto { parallel: true },
        opts,
    )?;
    Ok(x)
}

/// In-place form of [`solve_sparse_refined_parallel_opts`].
pub fn solve_sparse_refined_parallel_into(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    x_out: &mut [f64],
    opts: RefineOptions,
) -> Result<(), FeralError> {
    solve_sparse_refined_core(
        matrix,
        factors,
        rhs,
        x_out,
        false,
        SolveCore::Auto { parallel: true },
        opts,
    )?;
    Ok(())
}

/// Per-step diagnostic data emitted by
/// [`solve_sparse_refined_with_diagnostics`].
///
/// Step 0 is the unrefined initial solve; subsequent steps are refinement
/// iterations. The number of steps is bounded by the refinement cap
/// (`RefineOptions::max_steps` + 1 initial) and may exit early on convergence,
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
    solve_sparse_refined_with_diagnostics_opts(matrix, factors, rhs, RefineOptions::default())
}

/// [`solve_sparse_refined_with_diagnostics`] with a caller-supplied
/// correction-step cap (issue #178).
///
/// This is the observable the cap is verified against: a run under
/// `max_steps = k` returns at most `k + 1` entries in
/// [`RefinementDiagnostics::steps`], `steps[0]` being the unrefined
/// solve. Unlike the non-diagnostic entry points, `max_steps = 0` still
/// computes the initial residual — emitting `steps[0]` is the point.
pub fn solve_sparse_refined_with_diagnostics_opts(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    opts: RefineOptions,
) -> Result<(Vec<f64>, RefinementDiagnostics), FeralError> {
    let mut x = vec![0.0; factors.n];
    let diag = solve_sparse_refined_core(
        matrix,
        factors,
        rhs,
        &mut x,
        true,
        SolveCore::SharedVector,
        opts,
    )?; // `with_diagnostics = true` always yields `Some`; if it ever doesn't,
        // that's a logic bug — `expect` is fine in test code, but per CLAUDE.md
        // we use Result in src/. Return DimensionMismatch as a defensive
        // signal (can't actually happen with current control flow).
    let diag = diag.ok_or(FeralError::DimensionMismatch {
        expected: 1,
        got: 0,
    })?;
    Ok((x, diag))
}

/// Which numerical core a refinement run uses. Issue #177: this is the
/// *only* thing that may change a refined solve's arithmetic, and it is
/// always set explicitly by the caller — never inferred from the host's
/// core count, the rayon pool's existence, or `FERAL_CB_THRESH`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveCore {
    /// The shared-global-vector core (`solve_sparse_core_into`): each
    /// front's separator update folded into `y` in flat postorder. The
    /// default, and the only core available without parallelism.
    SharedVector,
    /// The contribution-block core (issue #131 Gap A): each front's RHS
    /// assembled from `b` plus its children's contribution blocks, summed
    /// in ascending child order — a subtree sum tree rather than a flat
    /// fold, hence a different (equally valid) reassociation.
    ///
    /// `parallel` selects tree-parallel execution over the rayon pool;
    /// it is a *scheduling* choice with no effect on the result bits, and
    /// is further gated internally by `CbTaskPlan::worthwhile`.
    ContribBlock {
        /// Execute over the rayon pool rather than in one thread.
        parallel: bool,
    },
    /// Pick between the two cores from the factor's structure alone, via
    /// the host-independent `cb_core_profitable` predicate: the CB core
    /// on bushy trees where it pays, the shared-vector core on path-like
    /// and small trees where it does not.
    ///
    /// This is the mode `Solver` uses. It is deliberately *not* a
    /// function of `use_parallel`, the thread pool, the worker count, or
    /// `FERAL_CB_THRESH` — issue #177: the same factor must solve with
    /// the same arithmetic on every host, and only the schedule may vary.
    /// `parallel` therefore selects execution strategy alone.
    Auto {
        /// Execute over the rayon pool rather than in one thread, if the
        /// factor-derived choice landed on the CB core.
        parallel: bool,
    },
}

/// Shared refinement loop behind every `solve_sparse_refined*` entry
/// point. Writes the best-residual iterate into `x_out` — which doubles
/// as the `best_x` storage, so the refined path holds one working vector
/// and one residual vector, not three.
fn solve_sparse_refined_core(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    x_out: &mut [f64],
    with_diagnostics: bool,
    core: SolveCore,
    opts: RefineOptions,
) -> Result<Option<RefinementDiagnostics>, FeralError> {
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

    // Issue #131 Gap A: when the caller selects the contribution-block
    // core, pool one CB workspace across the initial + capped correction
    // solves (each reuses the plan + scratch). Otherwise use the
    // shared-vector `SolveWorkspace` path, bit-for-bit.
    //
    // Issue #177: the choice is made *here*, from the caller's explicit
    // `core` argument, and nowhere else. It used to be made by
    // `CbSolveWorkspace::worthwhile()`, which is derived from
    // `rayon::current_num_threads()` and `FERAL_CB_THRESH` — so the same
    // build on hosts with different core counts silently ran different
    // arithmetic, and an IPM host amplified the ULP difference into
    // different iterate trajectories (henon120). `worthwhile` now only
    // picks the CB core's *execution* strategy, and both strategies emit
    // identical bits.
    let use_cb = n > 0
        && match core {
            SolveCore::SharedVector => false,
            SolveCore::ContribBlock { .. } => true,
            SolveCore::Auto { .. } => cb_core_profitable(factors),
        };
    let mut cb_ws: Option<CbSolveWorkspace> =
        use_cb.then(|| CbSolveWorkspace::for_factors(factors));
    let mut ws: Option<SolveWorkspace> = if cb_ws.is_none() {
        Some(SolveWorkspace::for_factors(factors))
    } else {
        None
    };
    let cb_parallel = matches!(
        core,
        SolveCore::ContribBlock { parallel: true } | SolveCore::Auto { parallel: true }
    );
    // The unrefined solve goes straight into the caller's buffer: it is
    // the first (and, under `max_steps = 0`, the only) candidate for the
    // best iterate.
    match (cb_ws.as_mut(), ws.as_mut()) {
        (Some(cb), _) => cb.solve_into(factors, rhs, x_out, cb_parallel),
        (None, Some(w)) => solve_sparse_into_ws(factors, rhs, x_out, w)?,
        (None, None) => unreachable!("exactly one solve workspace is built"),
    }

    // Issue #178: `max_steps = 0` must be the same *computation* as
    // `solve_sparse`, not merely the same answer — otherwise a caller
    // that opts out of refinement still pays a matvec and a norm per
    // solve. Return before the residual is formed, and before the
    // working iterate is even allocated. The diagnostics path is the
    // exception: `steps[0]` carries that residual, so it runs the matvec
    // and then simply skips the loop.
    if opts.max_steps == 0 && !with_diagnostics {
        return Ok(None);
    }

    // `x` is the working iterate the corrections accumulate into;
    // `x_out` keeps the best one seen.
    let mut x = x_out.to_vec();

    // Initial residual: compute A·x directly into r, then negate-add.
    let mut r = vec![0.0; n];
    matrix.symv(&x, &mut r);
    for i in 0..n {
        r[i] = rhs[i] - r[i];
    }
    let mut r_norm = norm2(&r);

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
    //
    // Issue #178 made the step budget a per-call parameter
    // (`RefineOptions::max_steps`, default `DEFAULT_REFINE_MAX_STEPS = 10`)
    // so an interior-point host running its own refinement loop over the
    // same system can ask for one correction instead of ten. It is a cap
    // only: every exit below still takes priority over it.
    let max_steps = opts.max_steps;
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

        match (cb_ws.as_mut(), ws.as_mut()) {
            (Some(cb), _) => cb.solve_into(factors, &r, &mut dx, cb_parallel),
            (None, Some(w)) => solve_sparse_into_ws(factors, &r, &mut dx, w)?,
            (None, None) => unreachable!("exactly one solve workspace is built"),
        }
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
            x_out.copy_from_slice(&x);
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
    Ok(diag)
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
    solve_sparse_many_refined_opts(matrix, factors, rhs, nrhs, RefineOptions::default())
}

/// [`solve_sparse_many_refined`] with a caller-supplied correction-step
/// cap (issue #178). The cap applies **per column**, exactly as the
/// uncapped budget does: each column stops at the first of its own
/// convergence, divergence, plateau, or `opts.max_steps`.
pub fn solve_sparse_many_refined_opts(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    nrhs: usize,
    opts: RefineOptions,
) -> Result<Vec<f64>, FeralError> {
    let mut x = vec![0.0; factors.n * nrhs];
    solve_sparse_many_refined_into(matrix, factors, rhs, nrhs, &mut x, opts)?;
    Ok(x)
}

/// In-place form of [`solve_sparse_many_refined_opts`]: writes the
/// column-major best-iterate solution into `x_out` (issue #178 item 2).
///
/// `x_out` doubles as the per-column best-iterate storage, so this saves
/// the refiner an `n × nrhs` allocation as well as the caller's
/// copy-back. Returns [`FeralError::DimensionMismatch`] when `rhs` or
/// `x_out` is not `n * nrhs` long.
pub fn solve_sparse_many_refined_into(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    nrhs: usize,
    x_out: &mut [f64],
    opts: RefineOptions,
) -> Result<(), FeralError> {
    let n = factors.n;
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
    if nrhs == 0 || n == 0 {
        x_out.fill(0.0);
        return Ok(());
    }

    // Same constants as the single-RHS refiner (solve_sparse_refined_core),
    // including the issue #178 per-call step cap.
    let max_steps = opts.max_steps;
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

    // Initial batched solve, written straight into the caller's buffer:
    // it is both the working iterate for the residual sweep below and
    // the per-column best iterate. Bit-for-bit `solve_sparse_many`,
    // which is the same call with a workspace it builds itself.
    let mut ws = SolveManyWorkspace::for_factors(factors, nrhs);
    solve_sparse_many_into(factors, rhs, nrhs, x_out, &mut ws)?;

    // Issue #178: `max_steps = 0` is `solve_sparse_many`, and must cost
    // the same — return before the per-column residual sweep rather than
    // computing residuals nobody will act on.
    if max_steps == 0 {
        return Ok(());
    }

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
        matrix.symv(&x_out[c * n..(c + 1) * n], &mut rc);
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
        return Ok(());
    }

    // Refinement is needed for at least one column. `x_out` holds the
    // initial solve and from here on is the per-column *best* iterate;
    // `x` is the working iterate the corrections accumulate into.
    let mut x = x_out.to_vec();
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
                x_out[c * n..(c + 1) * n].copy_from_slice(&x[c * n..(c + 1) * n]);
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

    Ok(())
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::factor::{BunchKaufmanParams, FrontalFactors, ZeroPivotAction};
    use crate::inertia::Inertia;
    use crate::numeric::factorize::factorize_multifrontal;
    use crate::sparse::csc::CscMatrix;
    use crate::symbolic::{symbolic_factorize, SupernodeParams};

    fn make_params() -> crate::numeric::factorize::NumericParams {
        crate::numeric::factorize::NumericParams::with_bk(BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            ..BunchKaufmanParams::default()
        })
    }

    // ---- Issue #175: the tree-parallel solve gate ---------------------
    // Fixtures and the calibration harness that set `MIN_COST_PER_FRONT`,
    // plus the regression tests that pin what it rejects.
    /// NARX-shaped proxy (issue #175): `blocks` independent dynamic-
    /// system KKT blocks, each a `steps`-long chain of `width`-wide
    /// states with one equality constraint per step and a near-empty
    /// Hessian, joined by one light coupling row. `width` sets the
    /// frontal size, i.e. the work each front does per unit of
    /// synchronization — the axis issue #175 is about.
    fn narx_proxy(blocks: usize, steps: usize, width: usize) -> CscMatrix {
        let per_step = width + 1; // `width` states + one multiplier
        let per_block = per_step * steps;
        let n = blocks * per_block + 1;
        let link = n - 1;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        let mut push = |r: usize, c: usize, v: f64| {
            let (r, c) = if r >= c { (r, c) } else { (c, r) };
            rows.push(r);
            cols.push(c);
            vals.push(v);
        };
        for b in 0..blocks {
            let base = b * per_block;
            for t in 0..steps {
                let x0 = base + t * per_step;
                let y = x0 + width;
                for i in 0..width {
                    for j in 0..=i {
                        push(x0 + i, x0 + j, if i == j { 1e-2 } else { 1e-3 });
                    }
                    push(y, x0 + i, 1.0 + i as f64 * 0.1);
                    if t + 1 < steps {
                        push(y, x0 + per_step + i, -0.9);
                    }
                }
                push(y, y, -1e-8);
            }
            push(link, base, 0.25);
        }
        push(link, link, 1.0);
        CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
    }

    /// A `NodeFactors` carrying only what the coarsening reads —
    /// `frontal_factors.{nrow, nelim}` — so a tree with tens of
    /// thousands of fronts can be built in microseconds, and each front's
    /// shape can be dialled directly to the value under test. The gate is
    /// a pure function of the tree shape and these two numbers
    /// (`own_cost = nrow·(nelim+1)`), so nothing the factorization would
    /// add is read.
    ///
    /// This is a convenience, not a necessity: factoring the real
    /// wide-sparse fixture `narx_proxy(24, 2000, 1)` (47,228 supernodes,
    /// n = 96,001) costs ~0.7 s in a debug build, and
    /// `cb_core_profitable_matches_the_plan_gate` does exactly that. An
    /// earlier version of this comment claimed "minutes", which is wrong
    /// by three orders of magnitude and would have been a reason to keep
    /// a real wide-sparse matrix out of the pinning test.
    fn synthetic_node(nrow: usize, nelim: usize) -> NodeFactors {
        NodeFactors {
            first_col: 0,
            ncol: nelim,
            nelim,
            n_delayed_in: 0,
            nrow,
            row_indices: Vec::new(),
            frontal_factors: FrontalFactors {
                nrow,
                ncol: nelim,
                nelim,
                l: Vec::new(),
                d_diag: Vec::new(),
                d_subdiag: Vec::new(),
                perm: Vec::new(),
                perm_inv: Vec::new(),
                contrib: Vec::new(),
                contrib_dim: nrow - nelim,
                n_delayed: 0,
                inertia: Inertia::new(0, 0, 0),
                needs_refinement: false,
                n_rook_rescues: 0,
                n_tiny: 0,
                zero_tol: 0.0,
                zero_tol_2x2: 0.0,
            },
            inertia: Inertia::new(0, 0, 0),
        }
    }

    /// `branches` independent chains of `per_branch` identical fronts
    /// under one root — postorder, so children precede parents. The
    /// shape of a wide, extremely sparse assembly tree: many independent
    /// subtrees, no single one dominating, every front the same size.
    fn wide_thin_plan(
        branches: usize,
        per_branch: usize,
        nrow: usize,
        nelim: usize,
    ) -> (CbTaskPlan, u64, usize) {
        let n_nodes = branches * per_branch + 1;
        let root = n_nodes - 1;
        let mut parents: Vec<Option<usize>> = vec![None; n_nodes];
        for b in 0..branches {
            let base = b * per_branch;
            for k in 0..per_branch {
                // within a branch, node `base + k` feeds `base + k + 1`
                parents[base + k] = Some(if k + 1 < per_branch {
                    base + k + 1
                } else {
                    root
                });
            }
        }
        let nodes: Vec<NodeFactors> = (0..n_nodes).map(|_| synthetic_node(nrow, nelim)).collect();
        let children = build_children(&parents, n_nodes);
        let plan = CbTaskPlan::build_with_threshold(
            &children,
            &parents,
            &nodes,
            n_nodes,
            CbThreshold::Reference,
        );
        // Read the builder's own accumulator rather than recomputing
        // `n_nodes · nrow · (nelim+1)` here: a hand copy of `own_cost`
        // would keep these guards passing against a stale formula if the
        // per-front cost model ever changed.
        let total = plan.total;
        (plan, total, n_nodes)
    }

    /// Issue #175: a wide, extremely sparse tree — the `NARX_CFy` shape,
    /// 45,736 supernodes at ~30 cost units per front — clears every
    /// *shape* term of the gate (11+ independent seeds, well over
    /// `MIN_TOTAL_COST` of work, no dominant subtree) and still loses
    /// 15% of an IPM run to per-front synchronization when scheduled
    /// over the pool. The overhead term must reject it.
    #[test]
    fn issue175_wide_thin_tree_is_not_scheduled_in_parallel() {
        // 16 branches x 2858 fronts of nrow 6 / nelim 4 = 30 units each.
        let (plan, total, n_nodes) = wide_thin_plan(16, 2858, 6, 4);

        // Non-vacuity: this is exactly the tree the pre-#175 gate liked.
        assert!(
            plan.shape_ok,
            "fixture must pass the shape half (seeds {}, total {total}) — \
             otherwise it is not the overhead term doing the rejecting",
            plan.fwd_seeds.len()
        );
        // ...and it is thin: 30 units per front, under the 64 floor.
        assert!(
            total / (n_nodes as u64) < MIN_COST_PER_FRONT,
            "fixture must be thinner than the floor: {} units/front",
            total / (n_nodes as u64)
        );
        assert!(
            !plan.worthwhile,
            "wide-thin tree ({} units/front, {n_nodes} fronts) was scheduled \
             in parallel — issue #175 regression",
            total / (n_nodes as u64)
        );
    }

    /// The complement of the test above: the same wide tree with fronts
    /// big enough to amortize the synchronization stays parallel. Guards
    /// against #175's fix degenerating into "never schedule in
    /// parallel", which would forfeit the 25-37% tree-parallel win
    /// measured on bushy factors.
    #[test]
    fn issue175_wide_tree_with_real_fronts_still_schedules_in_parallel() {
        // Same 16-way tree, fronts of nrow 16 / nelim 15 = 256 units —
        // the poisson_160 end of the calibration table (235 units/front,
        // par/ser 0.75).
        let (plan, total, n_nodes) = wide_thin_plan(16, 400, 16, 15);
        assert!(
            total / (n_nodes as u64) >= MIN_COST_PER_FRONT,
            "fixture must be fatter than the floor: {} units/front",
            total / (n_nodes as u64)
        );
        assert!(
            plan.worthwhile,
            "bushy tree ({} units/front, {} seeds) must still be scheduled \
             in parallel",
            total / n_nodes as u64,
            plan.fwd_seeds.len()
        );
    }

    /// Issue #175's term is a *scheduling* term: it may not reach the
    /// predicate that picks the numerical core (issue #177), or the same
    /// factor would solve with different arithmetic depending on how
    /// thin its tree is. Pinned on the scalar rule: the shape half is
    /// what `cb_core_profitable` applies, and it ignores front size.
    #[test]
    fn issue175_overhead_term_is_scheduling_only() {
        let (thin, total, n_nodes) = wide_thin_plan(16, 2858, 6, 4);
        assert!(!thin.worthwhile && thin.shape_ok);
        // The core-choice half sees the same tree as profitable, whatever
        // the front size, because both cores are equally available to it.
        let max_local = total / 800;
        assert!(
            cb_gate_shape(thin.fwd_seeds.len(), total, max_local),
            "the shape half must not have picked up the per-front term"
        );
        assert!(!cb_sync_amortized(total, n_nodes));
        // The same total spread over few enough fronts does clear the
        // term, so the rejection above is about work *per front* and not
        // about the fixture being small. `n_nodes / 4` puts it at ~120
        // units/front, comfortably over the floor.
        assert!(cb_sync_amortized(total, n_nodes / 4));
    }

    /// Calibration harness for `MIN_COST_PER_FRONT`, not an assertion.
    /// Run with `cargo test --release -- --ignored --nocapture issue175`.
    /// Prints, per fixture, the gate's terms next to the measured serial
    /// and tree-parallel times of the *pooled* CB core — the choice
    /// `worthwhile` actually makes — over rayon pools of 1/2/4/8
    /// workers. Kept in-tree so the constant can be re-derived on
    /// another host; the numbers this produced are in
    /// `dev/research/issue-175-cb-solve-gate-overhead.md`.
    #[test]
    #[ignore = "calibration harness (issue #175), not an assertion"]
    fn issue175_cb_gate_calibration() {
        let reps: usize = crate::env::usize_var("FERAL_CALIB_REPS").unwrap_or(30);
        let fixtures = [
            ("poisson_96", grid_2d(96)),
            ("poisson_160", grid_2d(160)),
            ("narx_w1", narx_proxy(24, 2000, 1)),
            ("narx_w2", narx_proxy(24, 1300, 2)),
            ("narx_w3", narx_proxy(24, 1000, 3)),
            ("narx_w4", narx_proxy(24, 800, 4)),
            ("narx_w6", narx_proxy(24, 570, 6)),
            ("narx_w8", narx_proxy(24, 440, 8)),
        ];
        println!(
            "{:<15} {:>3} {:>7} {:>8} {:>10} {:>6} {:>6} {:>8} {:>9} {:>9} {:>7}",
            "fixture",
            "w",
            "n",
            "nodes",
            "total",
            "roots",
            "seeds",
            "tot/node",
            "ser_us",
            "par_us",
            "par/ser"
        );
        for (label, m) in fixtures.iter() {
            let sym = symbolic_factorize(m, &SupernodeParams::default()).unwrap();
            let (factors, _) = factorize_multifrontal(m, &sym, &make_params()).unwrap();
            let n = m.n;
            let b: Vec<f64> = (0..n).map(|i| 1.0 + 0.37 * (i % 7) as f64).collect();
            for w in [1usize, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(w)
                    .build()
                    .unwrap();
                pool.install(|| {
                    let mut x = vec![0.0; n];
                    let mut ws = CbSolveWorkspace::for_factors(&factors);
                    let roots = ws.plan.is_task_root.iter().filter(|b| **b).count();
                    let seeds = ws.plan.fwd_seeds.len();
                    let nodes = factors.node_factors.len();
                    let total: u64 = factors
                        .node_factors
                        .iter()
                        .map(|nd| {
                            (nd.frontal_factors.nrow as u64)
                                .saturating_mul(nd.frontal_factors.nelim as u64 + 1)
                        })
                        .sum();
                    let time = |ws: &mut CbSolveWorkspace, par: bool, x: &mut [f64]| -> f64 {
                        ws.plan.worthwhile = par;
                        ws.solve_into(&factors, &b, x, true);
                        let mut best = f64::INFINITY;
                        for _ in 0..reps {
                            let t0 = std::time::Instant::now();
                            ws.solve_into(&factors, &b, x, true);
                            best = best.min(t0.elapsed().as_secs_f64() * 1e6);
                        }
                        best
                    };
                    let ser = time(&mut ws, false, &mut x);
                    let par = time(&mut ws, true, &mut x);
                    let ser = ser.min(time(&mut ws, false, &mut x));
                    let par = par.min(time(&mut ws, true, &mut x));
                    println!(
                        "{:<15} {:>3} {:>7} {:>8} {:>10} {:>6} {:>6} {:>8} {:>9.1} {:>9.1} {:>7.2}",
                        label,
                        w,
                        n,
                        nodes,
                        total,
                        roots,
                        seeds,
                        total / nodes.max(1) as u64,
                        ser,
                        par,
                        par / ser
                    );
                });
            }
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

    /// Issue #126: the single-RHS core now fuses the D-block solve into
    /// the forward pass (previously a separate second postorder sweep),
    /// mirroring the multi-RHS core. The fused result must be bit-for-bit
    /// identical to the multi-RHS core with `nrhs = 1` (which has always
    /// fused), including through 2×2 D-blocks on an indefinite KKT. This
    /// pins the fusion so a regression cannot silently change solutions.
    #[test]
    fn fused_single_rhs_matches_multi_rhs_k1_issue_126() {
        // Indefinite saddle-point KKT with a zero-Hessian variable block,
        // which forces 2×2 D-blocks in Bunch-Kaufman — the fusion's tricky
        // case (2×2 pivot straddling the gather/scatter boundary).
        //
        //   [  1        1   ]
        //   [     1     1   ]
        //   [        1  1   ]
        //   [  1  1  1  0   ]   (dual row: zero diagonal)
        let m = CscMatrix::from_triplets(
            4,
            &[0, 1, 2, 3, 3, 3],
            &[0, 1, 2, 0, 1, 2],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let (factors, _) = factorize_multifrontal(&m, &sym, &make_params()).unwrap();

        for rhs in [
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-5.0, 0.0, 7.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ] {
            let single = solve_sparse(&factors, &rhs).unwrap();
            let many = solve_sparse_many(&factors, &rhs, 1).unwrap();
            assert_eq!(single.len(), many.len());
            for (i, (s, mny)) in single.iter().zip(many.iter()).enumerate() {
                assert_eq!(
                    s.to_bits(),
                    mny.to_bits(),
                    "issue #126: fused single-RHS diverged from multi-RHS(k=1) \
                     at component {i} (rhs={rhs:?}): {s} vs {mny}"
                );
            }
        }
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

    /// 2-D Poisson on a `k x k` grid: a bushy nested-dissection tree.
    fn grid_2d(k: usize) -> CscMatrix {
        let n = k * k;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..k {
            for i in 0..k {
                let p = j * k + i;
                rows.push(p);
                cols.push(p);
                vals.push(4.0);
                if i + 1 < k {
                    rows.push(p + 1);
                    cols.push(p);
                    vals.push(-1.0);
                }
                if j + 1 < k {
                    rows.push(p + k);
                    cols.push(p);
                    vals.push(-1.0);
                }
            }
        }
        CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
    }

    /// A pentadiagonal chain: path-like, so the Amdahl arm rejects it.
    fn pentadiag_chain(n: usize) -> CscMatrix {
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(4.0);
            if i + 1 < n {
                rows.push(i + 1);
                cols.push(i);
                vals.push(-1.0);
            }
            if i + 2 < n {
                rows.push(i + 2);
                cols.push(i);
                vals.push(-0.5);
            }
        }
        CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
    }

    /// Issue #177: the CB coarsening threshold is a **scheduling** knob.
    /// It is derived from `rayon::current_num_threads()` (and overridable
    /// via `FERAL_CB_THRESH`), so if it could move a result bit, the same
    /// binary would solve differently on hosts with different core counts
    /// — which is exactly what the henon120 report caught.
    ///
    /// Sweeps the threshold across its whole useful range, asserts the
    /// task decomposition really changes (otherwise the test is vacuous),
    /// and asserts every solve is byte-identical, under both serial and
    /// tree-parallel execution.
    #[test]
    fn cb_coarsening_threshold_is_arithmetically_inert() {
        // Poisson 2-D 40x40: a bushy nested-dissection tree, so the
        // threshold has many cut points to choose between.
        let k = 40usize;
        let n = k * k;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..k {
            for i in 0..k {
                let p = j * k + i;
                rows.push(p);
                cols.push(p);
                vals.push(4.0);
                if i + 1 < k {
                    rows.push(p + 1);
                    cols.push(p);
                    vals.push(-1.0);
                }
                if j + 1 < k {
                    rows.push(p + k);
                    cols.push(p);
                    vals.push(-1.0);
                }
            }
        }
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let (factors, _) = factorize_multifrontal(&m, &sym, &make_params()).unwrap();
        let b: Vec<f64> = (0..n).map(|i| 1.0 + 0.37 * (i % 7) as f64).collect();

        let mut reference: Option<Vec<f64>> = None;
        let mut root_counts = Vec::new();
        for thresh in [1u64, 8, 64, 512, 4096, 32_768, 1 << 40, u64::MAX] {
            for parallel in [false, true] {
                let mut ws = CbSolveWorkspace::for_factors_with_threshold(
                    &factors,
                    CbThreshold::Fixed(thresh),
                );
                let mut x = vec![0.0; n];
                ws.solve_into(&factors, &b, &mut x, parallel);
                if !parallel {
                    root_counts.push(ws.task_root_count());
                }
                match &reference {
                    None => reference = Some(x),
                    Some(r) => {
                        for i in 0..n {
                            assert_eq!(
                                r[i].to_bits(),
                                x[i].to_bits(),
                                "thresh={thresh} parallel={parallel}: bit {i} moved \
                                 — the coarsening plan is not arithmetically inert"
                            );
                        }
                    }
                }
            }
        }
        // Non-vacuity: the sweep really did produce different task
        // decompositions, from every-node-a-root down to roots-only.
        let lo = root_counts.iter().copied().min().unwrap_or(0);
        let hi = root_counts.iter().copied().max().unwrap_or(0);
        assert!(
            hi > lo,
            "threshold sweep produced a single decomposition ({lo} task roots) — \
             the invariance assertion above proves nothing"
        );
    }

    /// Issue #177: `cb_core_profitable` reimplements `CbTaskPlan`'s gate
    /// on flat arrays, because it runs on every refined solve including
    /// the ones it turns down and the plan's three `Vec<Vec<usize>>`
    /// allocations cost more than a whole chain solve. Two
    /// implementations of one rule can drift, so pin them together
    /// across tree shapes that land on both sides of the gate.
    ///
    /// The rule they share is the *shape* half (`cb_gate_shape`). Issue
    /// #175's per-front overhead term belongs to `worthwhile` alone —
    /// it decides between two byte-identical executions, not between two
    /// cores — so it is `shape_ok`, not `worthwhile`, that is pinned
    /// here.
    ///
    /// `narx_proxy` is load-bearing and must not be dropped from the
    /// fixture list: it is the only shape here where `shape_ok` and
    /// `worthwhile` disagree, so it is the only one that can catch the
    /// #175 term leaking into the core choice. Grids and chains agree on
    /// both halves, and against those alone the leaked-term mutation
    /// passes the whole suite. `saw_split` asserts that such a fixture is
    /// still present rather than trusting the list.
    #[test]
    fn cb_core_profitable_matches_the_plan_gate() {
        let mut checked_true = false;
        let mut checked_false = false;
        let mut saw_split = false;
        for m in [
            grid_2d(8),
            grid_2d(40),
            grid_2d(160),
            pentadiag_chain(64),
            pentadiag_chain(400),
            pentadiag_chain(20_000),
            narx_proxy(24, 2000, 1),
        ] {
            let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
            let (factors, _) = factorize_multifrontal(&m, &sym, &make_params()).unwrap();
            let n_nodes = factors.node_factors.len();
            let children = build_children(&factors.node_parents, n_nodes);
            let plan = CbTaskPlan::build_with_threshold(
                &children,
                &factors.node_parents,
                &factors.node_factors,
                n_nodes,
                CbThreshold::Reference,
            );
            let flat = cb_core_profitable(&factors);
            assert_eq!(
                flat, plan.shape_ok,
                "n={}: flat predicate says {flat}, CbTaskPlan says {}",
                m.n, plan.shape_ok
            );
            checked_true |= flat;
            checked_false |= !flat;
            saw_split |= plan.shape_ok && !plan.worthwhile;
        }
        assert!(
            checked_true && checked_false,
            "fixtures must land on both sides of the gate \
             (saw profitable={checked_true}, rejected={checked_false})"
        );
        assert!(
            saw_split,
            "no fixture separates shape_ok from worthwhile, so this test \
             cannot detect the #175 term leaking into the core choice"
        );
    }

    /// Issue #177: `worthwhile` is a scheduling predicate, so requesting
    /// parallel execution on a factor it rejects must still run the CB
    /// core (serially) — not silently switch to the shared-vector core.
    ///
    /// A short tridiagonal chain is below `MIN_TOTAL_COST` and path-like,
    /// so the gate rejects it; the CB result must nonetheless differ from
    /// `solve_sparse` only as a reassociation, and must equal the CB core
    /// run either way, bit for bit.
    #[test]
    fn cb_core_is_used_even_when_the_parallel_gate_rejects_the_tree() {
        let n = 120usize;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
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
        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let (factors, _) = factorize_multifrontal(&m, &sym, &make_params()).unwrap();
        let b: Vec<f64> = (0..n).map(|i| 1.0 + 0.37 * (i % 7) as f64).collect();

        let ws = CbSolveWorkspace::for_factors(&factors);
        assert!(
            !ws.worthwhile(),
            "fixture must land on the rejected side of the gate"
        );

        let mut x_par = vec![0.0; n];
        let mut x_ser = vec![0.0; n];
        CbSolveWorkspace::for_factors(&factors).solve_into(&factors, &b, &mut x_par, true);
        CbSolveWorkspace::for_factors(&factors).solve_into(&factors, &b, &mut x_ser, false);
        for i in 0..n {
            assert_eq!(
                x_par[i].to_bits(),
                x_ser[i].to_bits(),
                "gate-rejected factor: requesting parallel changed bit {i}"
            );
        }
    }
}
