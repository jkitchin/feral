//! Rook pivoting rescue path (Phase 2.4.3).
//!
//! Textbook rook pivoting (Duff & Reid 1996, `duffreid1996zeros`;
//! Ashcraft, Grimes & Lewis 1998, `ashcraft1998accurate`) widens BK-
//! partial's single-column search to a path of alternating column/row
//! scans, finding a local-max pivot within the trailing submatrix.
//! In FERAL, rook is **not** a top-level strategy — it is spliced
//! into `try_reject_1x1_frontal` (see plan §"Splice point") only
//! after BK-partial has decided a 1×1 pivot and the column-relative
//! threshold test has rejected it. Well-conditioned matrices never
//! enter this path and pay zero rook cost.
//!
//! This file is Step 3 of the plan: the stub exists so tests can
//! reference the module and `rook_rescue` signature. Step 4
//! implements the full alternating-scan algorithm; Step 5 wires the
//! call into `try_reject_1x1_frontal`. Until then, `rook_rescue`
//! returns `None` unconditionally, which at Step 5 will fall through
//! to the existing delay/force-accept branch.
//!
//! See `dev/research/rook-rescue.md` for algorithmic background and
//! `dev/plans/phase-2.4.3-rook-rescue.md` for the implementation order.

use crate::dense::factor::BunchKaufmanParams;

/// Pivot shape chosen by a rook rescue search.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RookKind {
    /// 1×1 pivot at the rook-selected row (via symmetric swap into position k).
    Pivot1x1,
    /// 2×2 block pivot using two rook-selected rows/columns.
    Pivot2x2,
}

/// Outcome of a rook rescue search. Positions are absolute row/column
/// indices in the working array `a` (not relative to the trailing
/// submatrix). The caller applies `row_swaps` and `col_swaps` as
/// symmetric swaps, updates `perm`, and re-enters `scalar_pivot_step`
/// at the same `k` to run the standard 1×1 or 2×2 update on the newly-
/// positioned pivot.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RookPivot {
    pub kind: RookKind,
    /// Symmetric row/column swaps to apply before the rank-1 update.
    /// For a 1×1 pivot only `swaps[0]` is meaningful (swap row/col `.1`
    /// into position `.0 == k`); for a 2×2, `swaps[0]` and `swaps[1]`
    /// bring the pivot block into positions `k` and `k+1`.
    pub swaps: [(usize, usize); 2],
    /// Number of populated entries in `swaps` (1 for 1×1, 2 for 2×2).
    pub n_swaps: usize,
}

/// Rook-pivoting rescue search over the trailing submatrix starting
/// at pivot `k`. Reads `a` in column-major lower-triangle layout;
/// `nrow` is the full frontal height and `ncol` the count of fully-
/// summed columns eligible to host a pivot. Rows `[ncol, nrow)` are
/// ghost rows: they contribute to the column-max computation but
/// cannot host a pivot at this front (plan §"Ghost rows").
///
/// # Step 3 stub
///
/// Returns `None` unconditionally. The splice call site in
/// `try_reject_1x1_frontal` will exist at Step 5; until then this
/// function is not called from production paths. Step 4 replaces
/// the body with the full alternating-scan algorithm from Duff-Reid
/// 1996 with the Ashcraft-Grimes-Lewis 1998 bounded-iteration
/// safeguard (max 8 iterations).
#[allow(dead_code)]
pub(crate) fn rook_rescue(
    _a: &[f64],
    _nrow: usize,
    _ncol: usize,
    _k: usize,
    _params: &BunchKaufmanParams,
) -> Option<RookPivot> {
    None
}
