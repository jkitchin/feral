//! Threshold-Markowitz LU: the pivot order is chosen from the numbers, not ahead
//! of them (issue #167).
//!
//! The shipped sparse path splits into [`SparseLuSymbolic::analyze`] (a static
//! fill-reducing column permutation, AMD on the AᵀA pattern) and
//! [`SparseLu::factor`] (Gilbert–Peierls with partial pivoting inside that
//! order). On real LP bases that split is expensive: measured on 16 discopt
//! simplex bases, the static order reaches geomean fill 3.00x where a Markowitz
//! order reaches 1.11x, and SuperLU/COLAMD — the same algorithm class — is 3.24x,
//! which is why comparing against it could not detect the headroom
//! (`dev/research/lu-fill-markowitz-2026-08-13.md`).
//!
//! This module is the other half: a right-looking factorization that picks each
//! pivot `(i, j)` to minimise the Markowitz count `(r_i - 1)(c_j - 1)` over the
//! *active* submatrix, subject to the relative-magnitude threshold
//! `|a_ij| ≥ u · max_k |a_kj|`. It is not an ordering — the pivot depends on
//! values that only exist part-way through the elimination — so it replaces both
//! phases rather than slotting into either.
//!
//! **The Suhl–Suhl peel (#160) is a special case of this, not a separate step.**
//! A column singleton has Markowitz count 0 and is therefore taken immediately,
//! so a triangularizable basis is triangularized here with no peel code at all.
//! That is why the oracle reaches 1.00x fill on the 93–99% triangularizable
//! bases in the corpus.
//!
//! # Storage
//!
//! The active submatrix is **column-major with values** (`cols[j]`), plus a
//! row-wise *index-only* list (`row_cols[i]`) that is allowed to carry stale and
//! duplicate entries and is deduplicated with a mark array when a pivot row is
//! gathered. The rank-1 update runs column-oriented — for each alive column `j`
//! in the pivot row, `col_j ← col_j − u_j · l` — so finding `u_j` costs a scan of
//! `col_j`, which that column's rebuild pays for anyway. The pivot search reads
//! column values directly, which is what it needs: both `max_k|a_kj|` and the
//! threshold test are per column.
//!
//! `rcnt` / `ccnt` are maintained exactly. They are not a heuristic cache — a
//! drifted count makes every subsequent Markowitz cost a cost of the wrong
//! matrix.
//!
//! # Stability
//!
//! Accepting a non-maximal pivot bounds `max|L| ≤ 1/u` and bounds element growth
//! much more weakly than partial pivoting does. On QPLIB_1157 at `u = 0.1` the
//! oracle measured `max|U|/max|B| = 81.8` and `max|L| = 9.70` against SuperLU's
//! 2.56 and 1.00. That is the trade this path makes; `LuParams::max_growth` and
//! [`SparseLu::should_refactor_growth`] are the existing instruments for it.

use super::sparse_matrix::SparseColMatrix;
use crate::error::FeralError;
use crate::lu::LuSingularAction;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Raw output of the elimination, in **original** row/column indices. The caller
/// remaps to pivot positions and assembles the [`SparseLu`](super::SparseLu).
pub(super) struct MarkowitzFactor {
    /// `perm[k]` = original row of pivot `k`.
    pub perm: Vec<usize>,
    /// `qcol[k]` = original column of pivot `k`.
    pub qcol: Vec<usize>,
    /// `L` in CSC over pivot positions; `l_row_idx` holds **original** rows.
    pub l_col_ptr: Vec<usize>,
    pub l_row_idx: Vec<usize>,
    pub l_val: Vec<f64>,
    /// `U` row per pivot position, entries `(original column, value)`, the
    /// diagonal included but not necessarily first.
    pub u_rows_orig: Vec<Vec<(usize, f64)>>,
    /// `max|A|` of the factored (possibly scaled) matrix.
    pub a_max: f64,
}

/// Right-looking threshold-Markowitz LU of `a`.
///
/// `u_thresh` is the relative pivot threshold (`0 < u ≤ 1`; `1.0` degenerates to
/// partial pivoting within the chosen column). `max_search` is the Suhl cutoff:
/// stop examining candidate columns once this many have been examined and a
/// candidate is in hand. `zero_pivot_tol` is scaled by `max|A|` exactly as the
/// Gilbert–Peierls path scales it.
pub(super) fn markowitz_factor(
    a: &SparseColMatrix,
    u_thresh: f64,
    max_search: usize,
    zero_pivot_tol: f64,
    on_singular: LuSingularAction,
) -> Result<MarkowitzFactor, FeralError> {
    let m = a.m;

    let mut a_max = 0.0_f64;
    for &v in a.values.iter() {
        a_max = a_max.max(v.abs());
    }
    let ztol = zero_pivot_tol * a_max;

    // ---- active submatrix ------------------------------------------------
    let mut cols: Vec<Vec<(usize, f64)>> = Vec::with_capacity(m);
    let mut row_cols: Vec<Vec<usize>> = vec![Vec::new(); m];
    let mut rcnt = vec![0usize; m];
    for j in 0..m {
        let mut col = Vec::with_capacity(a.col_ptr[j + 1] - a.col_ptr[j]);
        for idx in a.col_ptr[j]..a.col_ptr[j + 1] {
            let (i, v) = (a.row_idx[idx], a.values[idx]);
            if v != 0.0 {
                col.push((i, v));
                row_cols[i].push(j);
                rcnt[i] += 1;
            }
        }
        cols.push(col);
    }
    let mut ccnt: Vec<usize> = cols.iter().map(|c| c.len()).collect();
    let mut alive_r = vec![true; m];
    let mut alive_c = vec![true; m];

    // Suhl-Suhl inside the Markowitz loop. A column with one live entry has
    // Markowitz cost 0 and is always taken; so is a row with one live entry
    // whose value passes the threshold. Handling those two off a stack instead
    // of through the count heaps is what makes this competitive on
    // near-triangular LP bases, where 93-99% of the columns peel and the heap
    // traffic would otherwise dominate the whole factorization
    // (`dev/research/lu-fill-markowitz-2026-08-13.md`). It changes cost, not the
    // pivot rule: these are exactly the pivots the general search would pick.
    let mut col_singletons: Vec<usize> = (0..m).filter(|&j| ccnt[j] == 1).collect();
    let mut row_singletons: Vec<usize> = (0..m).filter(|&i| rcnt[i] == 1).collect();

    // Lazy min-heaps over the counts; stale pairs are discarded on pop.
    let mut cheap: BinaryHeap<Reverse<(usize, usize)>> =
        (0..m).map(|j| Reverse((ccnt[j], j))).collect();
    let mut rheap: BinaryHeap<Reverse<(usize, usize)>> =
        (0..m).map(|i| Reverse((rcnt[i], i))).collect();

    // ---- workspaces ------------------------------------------------------
    let mut lscat = vec![0.0_f64; m];
    let mut lmark = vec![false; m];
    let mut seen_mark = vec![false; m];
    let mut seen_list: Vec<usize> = Vec::new();
    let mut row_seen = vec![false; m];
    let mut dirty_rows: Vec<usize> = Vec::new();
    let mut dirty_mark = vec![false; m];
    let mut lvec: Vec<(usize, f64)> = Vec::new();
    let mut uvec: Vec<(usize, f64)> = Vec::new();
    let mut scanned: Vec<usize> = Vec::new();

    // ---- output ----------------------------------------------------------
    let mut perm = Vec::with_capacity(m);
    let mut qcol = Vec::with_capacity(m);
    let mut l_col_ptr = Vec::with_capacity(m + 1);
    let mut l_row_idx: Vec<usize> = Vec::new();
    let mut l_val: Vec<f64> = Vec::new();
    l_col_ptr.push(0);
    let mut u_rows_orig: Vec<Vec<(usize, f64)>> = Vec::with_capacity(m);

    for _step in 0..m {
        // ---- pivot search ------------------------------------------------
        // Singleton fast path first (cost 0, no heap traffic).
        let mut fast: Option<(usize, usize)> = None;
        while let Some(j) = col_singletons.pop() {
            if !alive_c[j] || ccnt[j] != 1 {
                continue;
            }
            let (i, v) = cols[j][0];
            if v.abs() > ztol {
                fast = Some((i, j));
                break;
            }
        }
        if fast.is_none() {
            while let Some(i) = row_singletons.pop() {
                if !alive_r[i] || rcnt[i] != 1 {
                    continue;
                }
                // Find the one live column holding it. `row_cols` carries stale
                // entries, so this is a search, not a lookup.
                let mut found = None;
                for &j in row_cols[i].iter() {
                    if !alive_c[j] {
                        continue;
                    }
                    if let Some(&(_, v)) = cols[j].iter().find(|&&(r, _)| r == i) {
                        found = Some((j, v));
                        break;
                    }
                }
                // A row singleton still has to pass the threshold against its
                // own column — it is a cheap pivot, not a free one.
                if let Some((j, v)) = found {
                    let colmax = cols[j].iter().fold(0.0_f64, |a, &(_, x)| a.max(x.abs()));
                    if v.abs() > ztol && v.abs() >= u_thresh * colmax {
                        fast = Some((i, j));
                        break;
                    }
                }
            }
        }

        // Smallest live row count, a valid lower bound on `r_i` for the
        // remaining columns: no column can beat `(c-1)(minr-1)`.
        let minr = if fast.is_some() {
            1
        } else {
            loop {
                match rheap.peek() {
                    Some(&Reverse((c, i))) => {
                        if alive_r[i] && rcnt[i] == c {
                            break c.max(1);
                        }
                        rheap.pop();
                    }
                    None => break 1,
                }
            }
        };

        let mut best: Option<(usize, f64, usize, usize)> = None; // cost, |v|, i, j
        let mut singular_col: Option<usize> = None;
        scanned.clear();
        let mut examined = 0usize;
        while fast.is_none() {
            let Some(&Reverse((c, j))) = cheap.peek() else {
                break;
            };
            if !alive_c[j] || ccnt[j] != c {
                cheap.pop();
                continue;
            }
            if let Some((bc, _, _, _)) = best {
                if c.saturating_sub(1) * minr.saturating_sub(1) > bc {
                    break;
                }
            }
            cheap.pop();
            scanned.push(j);

            let colmax = cols[j]
                .iter()
                .fold(0.0_f64, |acc, &(_, v)| acc.max(v.abs()));
            if colmax <= ztol {
                // Numerically empty column: not a pivot candidate, and the
                // reason this basis is singular if nothing else is left.
                singular_col = Some(singular_col.map_or(j, |p: usize| p.min(j)));
                continue;
            }
            let bar = u_thresh * colmax;
            for &(i, v) in cols[j].iter() {
                if v.abs() < bar {
                    continue;
                }
                let cost = rcnt[i].saturating_sub(1) * c.saturating_sub(1);
                let better = match best {
                    None => true,
                    Some((bc, bv, _, _)) => cost < bc || (cost == bc && v.abs() > bv),
                };
                if better {
                    best = Some((cost, v.abs(), i, j));
                }
            }
            examined += 1;
            if let Some((0, _, _, _)) = best {
                break;
            }
            if examined >= max_search && best.is_some() {
                break;
            }
        }
        for &j in scanned.iter() {
            if alive_c[j] {
                cheap.push(Reverse((ccnt[j], j)));
            }
        }

        let (pi, pj) = match fast.or(best.map(|(_, _, i, j)| (i, j))) {
            Some((i, j)) => (i, j),
            None => {
                // Nothing acceptable anywhere: the remaining active submatrix has
                // no pivot above `ztol`.
                let col = singular_col
                    .or_else(|| (0..m).find(|&j| alive_c[j]))
                    .unwrap_or(0);
                match on_singular {
                    LuSingularAction::Fail => {
                        return Err(FeralError::SingularBasis { column: col });
                    }
                    LuSingularAction::PerturbToEps { abs_floor } => {
                        // Perturb in place: give the column an entry on some live
                        // row and let the normal path consume it.
                        let row = match (0..m).find(|&i| alive_r[i]) {
                            Some(r) => r,
                            None => return Err(FeralError::SingularBasis { column: col }),
                        };
                        let mag = abs_floor.max(f64::MIN_POSITIVE);
                        match cols[col].iter_mut().find(|(i, _)| *i == row) {
                            Some(e) => e.1 = if e.1 < 0.0 { -mag } else { mag },
                            None => {
                                cols[col].push((row, mag));
                                row_cols[row].push(col);
                                rcnt[row] += 1;
                                ccnt[col] += 1;
                                cheap.push(Reverse((ccnt[col], col)));
                                rheap.push(Reverse((rcnt[row], row)));
                            }
                        }
                        (row, col)
                    }
                }
            }
        };

        let pv = cols[pj]
            .iter()
            .find(|&&(i, _)| i == pi)
            .map(|&(_, v)| v)
            .ok_or(FeralError::SingularBasis { column: pj })?;

        // ---- gather the pivot column (→ L) and pivot row (→ U) ------------
        lvec.clear();
        let inv = 1.0 / pv;
        for &(i, v) in cols[pj].iter() {
            if i != pi {
                lvec.push((i, v * inv));
            }
        }
        uvec.clear();
        for &j in row_cols[pi].iter() {
            if j == pj || !alive_c[j] || row_seen[j] {
                continue;
            }
            row_seen[j] = true;
            if let Some(&(_, v)) = cols[j].iter().find(|&&(i, _)| i == pi) {
                uvec.push((j, v));
            }
        }
        for &j in row_cols[pi].iter() {
            row_seen[j] = false;
        }

        alive_r[pi] = false;
        alive_c[pj] = false;
        perm.push(pi);
        qcol.push(pj);

        // U row: diagonal plus the surviving row entries.
        let mut urow = Vec::with_capacity(1 + uvec.len());
        urow.push((pj, pv));
        urow.extend_from_slice(&uvec);
        u_rows_orig.push(urow);

        // L column: the multipliers, in original rows.
        for &(i, mult) in lvec.iter() {
            l_row_idx.push(i);
            l_val.push(mult);
        }
        l_col_ptr.push(l_row_idx.len());

        // Every live row in the pivot column loses its `pj` entry.
        dirty_rows.clear();
        for &(i, _) in lvec.iter() {
            rcnt[i] -= 1;
            if !dirty_mark[i] {
                dirty_mark[i] = true;
                dirty_rows.push(i);
            }
        }
        cols[pj] = Vec::new();

        // ---- rank-1 update, column-oriented ------------------------------
        // In place: each column is walked once, existing entries updated where
        // the pivot column touches them and swap-removed on exact cancellation,
        // and the rows the walk did not meet appended as fill. Rebuilding into a
        // fresh `Vec` per column instead costs one allocation per (pivot, column)
        // pair, which on a 5150-column basis is millions of them.
        for &(i, lv) in lvec.iter() {
            lscat[i] = lv;
            lmark[i] = true;
        }
        for &(j, uval) in uvec.iter() {
            let col = &mut cols[j];
            let mut k = 0;
            while k < col.len() {
                let (i, v) = col[k];
                if i == pi {
                    col.swap_remove(k); // consumed into the U row
                    continue;
                }
                if uval != 0.0 && lmark[i] {
                    let nv = v - uval * lscat[i];
                    seen_mark[i] = true;
                    seen_list.push(i);
                    if nv == 0.0 {
                        // Exact cancellation: drop it, and keep the counts exact.
                        rcnt[i] -= 1;
                        if !dirty_mark[i] {
                            dirty_mark[i] = true;
                            dirty_rows.push(i);
                        }
                        col.swap_remove(k);
                        continue;
                    }
                    col[k].1 = nv;
                }
                k += 1;
            }
            if uval != 0.0 {
                for &(i, lv) in lvec.iter() {
                    if seen_mark[i] {
                        continue;
                    }
                    let nv = -uval * lv;
                    if nv != 0.0 {
                        col.push((i, nv));
                        rcnt[i] += 1;
                        row_cols[i].push(j);
                        if !dirty_mark[i] {
                            dirty_mark[i] = true;
                            dirty_rows.push(i);
                        }
                    }
                }
            }
            for &i in seen_list.iter() {
                seen_mark[i] = false;
            }
            seen_list.clear();
            ccnt[j] = col.len();
            // Always into the heap as well: the stack is a fast path, not an
            // alternative index. A column that stops being a singleton is
            // dropped from the stack, and if it were not in `cheap` too it would
            // be invisible to the general search and reported singular.
            cheap.push(Reverse((ccnt[j], j)));
            if ccnt[j] == 1 {
                col_singletons.push(j);
            }
        }
        for &(i, _) in lvec.iter() {
            lmark[i] = false;
        }

        for &i in dirty_rows.iter() {
            dirty_mark[i] = false;
            if alive_r[i] {
                if rcnt[i] == 1 {
                    row_singletons.push(i);
                }
                rheap.push(Reverse((rcnt[i], i)));
            }
        }
    }

    Ok(MarkowitzFactor {
        perm,
        qcol,
        l_col_ptr,
        l_row_idx,
        l_val,
        u_rows_orig,
        a_max,
    })
}
