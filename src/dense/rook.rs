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
//! # Algorithm (symmetric rook, threshold-gated)
//!
//! Initialize `r := k` (current anchor). Let `gamma_r` be the maximum
//! off-diagonal magnitude in symmetric row `r` over `[k, nrow)`, and
//! `s` its argmax. Repeat up to 8 iterations (Ashcraft-Grimes-Lewis
//! 1998 safeguard):
//!
//! 1. If `r` is fully-summed and `|A[r,r]| >= u * gamma_r`, accept 1×1 at `r`.
//! 2. Compute `gamma_s` (symmetric row max for `s`) and its argmax `t`.
//! 3. If `s` is fully-summed and `|A[s,s]| >= u * gamma_s`, accept 1×1 at `s`.
//! 4. If both `r`, `s` are fully-summed and `gamma_s <= gamma_r` — the
//!    off-diagonal `A[s, r]` dominates both rows — try a 2×2 at `{r, s}`.
//!    Accept iff SSIDS scale-invariant det floor + Duff-Reid growth
//!    bound pass (both also used by BK-partial at `factor_frontal`
//!    lines 1409-1439).
//! 5. Advance: `(r, gamma_r, s) := (s, gamma_s, t)` and loop.
//!
//! Returning `None` means rook could not find any pivot passing the
//! `u`-threshold within 8 iterations; the caller falls through to the
//! existing delay / force-accept branch.
//!
//! # Ghost rows
//!
//! Rows `[ncol, nrow)` are ghost rows — they contribute to row-max
//! computations (via `gamma_r`) but cannot host a pivot, since only
//! fully-summed columns `[0, ncol)` are eliminated at this front. The
//! initial `r = k` is always fully-summed. After the advance step,
//! `r` may become ghost; in that case 1×1 / 2×2 accept is skipped but
//! the scan continues to look for a fully-summed partner.
//!
//! # Threshold choice (research §11 Q2)
//!
//! Rook uses `params.pivot_threshold` (`u`), not `alpha`, as the 1×1
//! accept gate. This matches the column-relative threshold applied by
//! `try_reject_1x1_frontal` at the splice site — if rook's candidate
//! passes `u * row_max`, the caller's re-check against `u * col_max`
//! also passes (modulo the distinction that row and column maxes are
//! identical by symmetry).
//!
//! See `dev/research/rook-rescue.md` for algorithmic background and
//! `dev/plans/phase-2.4.3-rook-rescue.md` for the implementation order.

use crate::dense::factor::BunchKaufmanParams;

/// Dead-zero floor for the 2×2 cancellation-aware determinant test.
/// Matches the constant used by `factor_frontal`'s BK-partial 2×2 gate.
const SSIDS_DET_SMALL: f64 = 1e-20;

/// Maximum number of alternating row scans before giving up
/// (Ashcraft-Grimes-Lewis 1998 safeguard).
const MAX_ROOK_ITER: usize = 8;

/// Pivot shape chosen by a rook rescue search.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RookKind {
    /// 1×1 pivot at the rook-selected row (via symmetric swap into position k).
    Pivot1x1,
    /// 2×2 block pivot using two rook-selected rows/columns.
    Pivot2x2,
}

/// Outcome of a rook rescue search. Positions in `swaps` are absolute
/// row/column indices in the working array `a` (not relative to the
/// trailing submatrix). The caller applies `swaps[0..n_swaps]` in order
/// as symmetric row/column swaps via `swap_rows_cols`, updating `perm`,
/// then re-enters the standard 1×1 or 2×2 update at pivot position `k`
/// (or `k+1` for 2×2).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RookPivot {
    pub kind: RookKind,
    /// Symmetric row/column swaps to apply before the update. Each
    /// `(a, b)` swaps rows/cols `a` and `b`. Applied in order; the
    /// caller must account for index drift.
    pub swaps: [(usize, usize); 2],
    /// Number of populated entries in `swaps` (0..=2).
    pub n_swaps: usize,
}

#[allow(dead_code)]
impl RookPivot {
    fn single(k: usize, r: usize) -> Self {
        let (swaps, n_swaps) = if r == k {
            ([(0, 0); 2], 0)
        } else {
            ([(k, r), (0, 0)], 1)
        };
        RookPivot {
            kind: RookKind::Pivot1x1,
            swaps,
            n_swaps,
        }
    }

    fn pair(k: usize, r: usize, s: usize) -> Self {
        // Place the 2×2 block at {k, k+1}. Order: bring `r` to position
        // `k` first (dominant row), then bring `s` to position `k+1`,
        // tracking the swap-induced index drift.
        //
        // Special case: if {r, s} == {k+1, k} (rows are already in the
        // block but reversed), a single (k, k+1) swap fixes both.
        if r == k + 1 && s == k {
            return RookPivot {
                kind: RookKind::Pivot2x2,
                swaps: [(k, k + 1), (0, 0)],
                n_swaps: 1,
            };
        }

        let mut swaps = [(0, 0); 2];
        let mut n = 0;

        if r != k {
            swaps[n] = (k, r);
            n += 1;
        }

        // Track where `s` ended up after the first swap.
        let s_after = if r != k {
            if s == k {
                r
            } else if s == r {
                k
            } else {
                s
            }
        } else {
            s
        };

        if s_after != k + 1 {
            swaps[n] = (k + 1, s_after);
            n += 1;
        }

        RookPivot {
            kind: RookKind::Pivot2x2,
            swaps,
            n_swaps: n,
        }
    }
}

/// Symmetric lookup: return `A[i, j]` from a column-major lower-
/// triangle buffer of leading dimension `nrow`.
#[inline]
fn sym_elem(a: &[f64], nrow: usize, i: usize, j: usize) -> f64 {
    let (row, col) = if i >= j { (i, j) } else { (j, i) };
    a[col * nrow + row]
}

/// Maximum absolute off-diagonal in symmetric row `r`, scanning
/// indices `i ∈ [k, nrow)` with `i != r`. Returns `(max_val, argmax)`.
/// If the row is all-zero off-diagonal, returns `(0.0, r)`.
fn sym_row_argmax(a: &[f64], nrow: usize, k: usize, r: usize) -> (f64, usize) {
    let mut max_val = 0.0f64;
    let mut max_idx = r;
    for i in k..r {
        let v = a[i * nrow + r].abs();
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    for i in (r + 1)..nrow {
        let v = a[r * nrow + i].abs();
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    (max_val, max_idx)
}

/// Test whether a 2×2 pivot block at absolute positions `(p, q)` with
/// `p < q`, both in `[k, ncol)`, would pass both stability gates:
///
/// - SSIDS scale-invariant cancellation-aware determinant floor (ported
///   from `factor_frontal` lines 1409-1439).
/// - Duff-Reid column-max growth bound with threshold `u`.
///
/// Both bounds are evaluated in the current (pre-swap) coordinates.
/// RMAX and TMAX are computed over `i ∈ [k, nrow) \ {p, q}` — i.e.
/// the rows that will become the trailing `k+2..nrow` slab after
/// swapping `(p, q)` into `(k, k+1)`.
fn passes_2x2_gates(a: &[f64], nrow: usize, k: usize, p: usize, q: usize, u: f64) -> bool {
    debug_assert!(p < q);
    let d11 = sym_elem(a, nrow, p, p);
    let d22 = sym_elem(a, nrow, q, q);
    let d21 = sym_elem(a, nrow, q, p);
    let det = d11 * d22 - d21 * d21;

    // SSIDS scale-invariant determinant floor.
    let max_piv = d11.abs().max(d21.abs()).max(d22.abs());
    if max_piv < SSIDS_DET_SMALL {
        return false;
    }
    let det_scale = 1.0 / max_piv;
    let detpiv0 = (d11 * det_scale) * d22;
    let detpiv1 = (d21 * det_scale) * d21;
    let detpiv = detpiv0 - detpiv1;
    let cancel_floor = SSIDS_DET_SMALL
        .max(detpiv0.abs() * 0.5)
        .max(detpiv1.abs() * 0.5);
    if detpiv.abs() < cancel_floor {
        return false;
    }

    // Duff-Reid growth bound. RMAX/TMAX over rows in [k, nrow) excluding
    // the pivot rows {p, q}.
    let mut rmax = 0.0f64;
    let mut tmax = 0.0f64;
    for i in k..nrow {
        if i == p || i == q {
            continue;
        }
        let r_val = sym_elem(a, nrow, i, p).abs();
        if r_val > rmax {
            rmax = r_val;
        }
        let t_val = sym_elem(a, nrow, i, q).abs();
        if t_val > tmax {
            tmax = t_val;
        }
    }

    let amax = d21.abs();
    let absdet = det.abs();
    (d22.abs() * rmax + amax * tmax) * u <= absdet && (d11.abs() * tmax + amax * rmax) * u <= absdet
}

/// Rook-pivoting rescue search over the trailing submatrix starting
/// at pivot `k`. Reads `a` in column-major lower-triangle layout;
/// `nrow` is the full frontal height and `ncol` the count of fully-
/// summed columns eligible to host a pivot. Rows `[ncol, nrow)` are
/// ghost rows: they contribute to row/column-max computation but
/// cannot host a pivot at this front.
///
/// Returns `None` when no candidate in `MAX_ROOK_ITER` iterations
/// passes the `u`-threshold; the caller falls through to the existing
/// delay / force-accept branch. Returns `Some(pivot)` with the swap
/// sequence and pivot kind otherwise.
#[allow(dead_code)]
pub(crate) fn rook_rescue(
    a: &[f64],
    nrow: usize,
    ncol: usize,
    k: usize,
    params: &BunchKaufmanParams,
) -> Option<RookPivot> {
    let u = params.pivot_threshold;
    if u <= 0.0 {
        return None;
    }
    if k >= ncol {
        return None;
    }

    // Initialize anchor at r = k.
    let mut r = k;
    let (mut gamma_r, mut s) = sym_row_argmax(a, nrow, k, r);
    if gamma_r == 0.0 {
        return None;
    }

    for _ in 0..MAX_ROOK_ITER {
        // Step 1: accept 1×1 at r (if fully-summed).
        if r < ncol {
            let arr = a[r * nrow + r].abs();
            if arr >= u * gamma_r {
                return Some(RookPivot::single(k, r));
            }
        }

        // Step 2: compute row-max for s.
        let (gamma_s, t) = sym_row_argmax(a, nrow, k, s);
        if gamma_s == 0.0 {
            return None;
        }

        // Step 3: accept 1×1 at s (if fully-summed).
        if s < ncol {
            let ass = a[s * nrow + s].abs();
            if ass >= u * gamma_s {
                return Some(RookPivot::single(k, s));
            }
        }

        // Step 4: 2×2 termination. When gamma_s <= gamma_r, the
        // off-diagonal A[s, r] dominates both rows, so {r, s} is a
        // locally-maximal 2×2 candidate. Both rows must be fully-summed.
        if r < ncol && s < ncol && gamma_s <= gamma_r {
            let (p, q) = if r < s { (r, s) } else { (s, r) };
            if passes_2x2_gates(a, nrow, k, p, q, u) {
                return Some(RookPivot::pair(k, r, s));
            }
        }

        // Step 5: advance.
        r = s;
        gamma_r = gamma_s;
        s = t;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params_u(u: f64) -> BunchKaufmanParams {
        BunchKaufmanParams {
            pivot_threshold: u,
            ..BunchKaufmanParams::default()
        }
    }

    /// Test 2 matrix from `tests/rook_rescue.rs`: rook walks
    /// `(0,0) -> (0,1) -> (2,1)` and accepts 1×1 at row 2.
    #[test]
    fn rook_accepts_1x1_at_row_2() {
        let n = 4;
        let mut a = vec![0.0f64; n * n];
        a[0] = 0.008;
        a[1] = 1.0; // A[1,0]
        a[n + 1] = 0.5; // A[1,1]
        a[n + 2] = 100.0; // A[2,1]
        a[2 * n + 2] = 500.0; // A[2,2]
        a[3 * n + 3] = 1.0; // A[3,3]

        let params = default_params_u(0.01);
        let pivot = rook_rescue(&a, n, n, 0, &params).expect("rook should find a pivot");
        assert_eq!(pivot.kind, RookKind::Pivot1x1);
        assert_eq!(pivot.n_swaps, 1);
        assert_eq!(pivot.swaps[0], (0, 2));
    }

    /// Test 3 matrix: rook walks `(0,0) -> (0,1) -> (2,1)` and accepts
    /// 2×2 at `{1, 2}`.
    #[test]
    fn rook_accepts_2x2_at_rows_1_2() {
        let n = 5;
        let mut a = vec![0.0f64; n * n];
        a[0] = 0.008;
        a[1] = 1.0;
        a[n + 1] = 0.1;
        a[n + 2] = 1.0e4;
        a[2 * n + 2] = 0.1;
        a[3 * n + 3] = 1.0;
        a[4 * n + 4] = 1.0;

        let params = default_params_u(0.01);
        let pivot = rook_rescue(&a, n, n, 0, &params).expect("rook should find a pivot");
        assert_eq!(pivot.kind, RookKind::Pivot2x2);
        // Scan terminates at (r=1, s=2) when gamma_s == gamma_r == 1e4.
        // `pair(0, 1, 2)` brings row 1 to position 0 via (0, 1), then
        // row 2 (unmoved at position 2) to position 1 via (1, 2).
        assert_eq!(pivot.n_swaps, 2);
        assert_eq!(pivot.swaps[0], (0, 1));
        assert_eq!(pivot.swaps[1], (1, 2));
    }

    /// SPD input: BK-partial accepts 1×1 at k=0 directly. Rook is not
    /// normally called on SPD, but if called, it should return a 1×1
    /// at k (diagonally dominant).
    #[test]
    fn rook_accepts_1x1_at_k_when_diagonally_dominant() {
        let n = 4;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            a[i * n + i] = 10.0 + i as f64;
        }
        a[1] = 1.0;
        a[n + 2] = 0.5;
        a[2 * n + 3] = 0.25;

        let params = default_params_u(0.01);
        let pivot = rook_rescue(&a, n, n, 0, &params).expect("rook finds pivot on SPD");
        assert_eq!(pivot.kind, RookKind::Pivot1x1);
        assert_eq!(pivot.n_swaps, 0);
    }

    /// Zero pivot_threshold: rook declines unconditionally (there's
    /// nothing to rescue because no pivot can fail the column-relative
    /// test).
    #[test]
    fn rook_declines_when_threshold_zero() {
        let n = 3;
        let mut a = vec![0.0f64; n * n];
        a[0] = 1.0;
        a[n + 1] = 1.0;
        a[2 * n + 2] = 1.0;

        let params = default_params_u(0.0);
        assert!(rook_rescue(&a, n, n, 0, &params).is_none());
    }

    /// Ghost-row safeguard: if the trailing submatrix has fully-summed
    /// rows that are unusable (all pivots below threshold) and any
    /// sufficient pivot lives in a ghost row, rook must return `None`.
    #[test]
    fn rook_declines_when_only_ghost_has_good_pivot() {
        // 3×3 with ncol=2. Row 2 is ghost.
        //   A[0,0] = 0.001, A[1,0] = 1
        //   A[1,1] = 0.001, A[2,1] = 0
        //   A[2,2] = 100 (ghost — not a candidate)
        let n = 3;
        let ncol = 2;
        let mut a = vec![0.0f64; n * n];
        a[0] = 0.001;
        a[1] = 1.0;
        a[n + 1] = 0.001;
        a[2 * n + 2] = 100.0;

        let params = default_params_u(0.01);
        // Rook walks r=0 → s=1 → t=0 (both 0.001 diagonals fail u*gamma).
        // No 2×2 termination either (gamma_s = gamma_r trivially, but
        // d11*d22 = 1e-6, d21*d21 = 1 → detpiv = 1e-6/1 - 1 ≈ -1,
        // cancel_floor = |1|/2 = 0.5, |detpiv| = 1 >= 0.5, passes det
        // floor; BUT growth (|d22|*0 + |d21|*0)*u = 0 <= |det|=~1, so
        // 2×2 accepts). Actually this matrix accepts 2×2 at {0,1}.
        // Not a valid ghost-only test — rewrite.
        //
        // Replacement: matrix where rook cycles without hitting a fully-
        // summed candidate. Make the max off-diag of rows 0 and 1
        // point at the ghost row 2, and diagonals 0 and 1 tiny.
        let mut b = vec![0.0f64; n * n];
        b[0] = 0.001;
        b[1] = 0.0; // A[1,0] = 0
        b[n + 1] = 0.001;
        b[n + 2] = 0.0; // A[2,1] = 0
        b[2] = 1.0; // A[2,0] = 1 (strongest in row 0)
        b[2 * n + 2] = 100.0;

        // With these entries: row 0 argmax is row 2 (ghost), row 1 is
        // all zero off-diagonal. From r=0: gamma_r=1, s=2 (ghost);
        // gamma_s = sym_row_max(2) = 1 (from A[0,2]); t=0. 1×1 at
        // r=0? 0.001 < 0.01. No. 1×1 at s=2? skipped (ghost). 2×2 at
        // {0, 2}? skipped (s ghost). Advance: r=2 (ghost), gamma_r=1,
        // s=0. Next iter: 1×1 at r=2 skipped (ghost). gamma_s for s=0
        // = 1. 1×1 at s=0? 0.001 < 0.01. No. 2×2 at {2,0}? skipped.
        // Advance: r=0, s=2. Cycle. Terminates at MAX_ROOK_ITER with
        // None.
        let _ = a; // silence
        let pivot = rook_rescue(&b, n, ncol, 0, &params);
        assert!(
            pivot.is_none(),
            "rook must decline when only ghost rows host sufficient pivots, got {:?}",
            pivot
        );
    }
}
