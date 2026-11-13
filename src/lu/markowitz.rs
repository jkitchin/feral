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
//! gathered. Both are packed into one buffer each ([`Lists`]) rather than one
//! `Vec` per column or row (issue #232).
//!
//! The count heaps that drive the general pivot search are built only when the
//! singleton stacks first run dry, from a log of what they would have held, so
//! the pivots are exactly those of eagerly maintained heaps (issue #232). The rank-1 update runs column-oriented — for each alive column `j`
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

/// Variable-length lists, one per column (or row), packed in one buffer.
///
/// Replaces `Vec<Vec<T>>` for the active submatrix (issue #232): on a
/// near-triangular LP basis with m = 17 470 the per-list `Vec`s were ~35k small
/// allocations per factorization, most of them for lists that are consumed at
/// a singleton step and never grow. A list that outgrows its slot moves to the
/// end of the buffer with doubled capacity, so total storage stays within a
/// constant factor of the largest size each list ever reaches; the abandoned
/// slots are not reclaimed, which is fine for a buffer that lives for one
/// factorization.
///
/// `push` and `swap_remove` reproduce `Vec`'s element order exactly, so the
/// pivot search's tie-breaks (which depend on that order) are those of the
/// `Vec<Vec<T>>` it replaced.
struct Lists<T> {
    start: Vec<usize>,
    len: Vec<usize>,
    cap: Vec<usize>,
    data: Vec<T>,
}

impl<T: Copy + Default> Lists<T> {
    /// Empty lists, list `j` with room for `caps[j]` entries before it has to
    /// move. Filled with [`Self::push`].
    fn with_capacities(caps: &[usize]) -> Self {
        let mut start = Vec::with_capacity(caps.len());
        let mut total = 0usize;
        for &c in caps {
            start.push(total);
            total += c;
        }
        Self {
            start,
            len: vec![0; caps.len()],
            cap: caps.to_vec(),
            data: vec![T::default(); total],
        }
    }

    #[inline]
    fn get(&self, j: usize) -> &[T] {
        let s = self.start[j];
        &self.data[s..s + self.len[j]]
    }

    #[inline]
    fn get_mut(&mut self, j: usize) -> &mut [T] {
        let s = self.start[j];
        let l = self.len[j];
        &mut self.data[s..s + l]
    }

    #[inline]
    fn push(&mut self, j: usize, x: T) {
        if self.len[j] == self.cap[j] {
            self.grow(j);
        }
        let at = self.start[j] + self.len[j];
        self.data[at] = x;
        self.len[j] += 1;
    }

    #[cold]
    fn grow(&mut self, j: usize) {
        let (s, l, c) = (self.start[j], self.len[j], self.cap[j]);
        let new_cap = (2 * c).max(4);
        if s + c == self.data.len() {
            // Already the last slot: extend in place.
            self.data.resize(s + new_cap, T::default());
        } else {
            let ns = self.data.len();
            self.data.extend_from_within(s..s + l);
            self.data.resize(ns + new_cap, T::default());
            self.start[j] = ns;
        }
        self.cap[j] = new_cap;
    }

    /// `Vec::swap_remove`: the last element takes position `k`.
    #[inline]
    fn swap_remove(&mut self, j: usize, k: usize) {
        let s = self.start[j];
        let last = s + self.len[j] - 1;
        self.data.swap(s + k, last);
        self.len[j] -= 1;
    }

    #[inline]
    fn clear(&mut self, j: usize) {
        self.len[j] = 0;
    }
}

/// Column-heap entry `(count, column)` as one `u64`: `count << 32 | column`.
/// Ordering equals the tuple's while both are below 2^32, which
/// `markowitz_factor` checks (`count <= m` always).
#[inline]
fn ckey(count: usize, col: usize) -> u64 {
    ((count as u64) << 32) | col as u64
}

#[inline]
fn ckey_count(key: u64) -> usize {
    (key >> 32) as usize
}

#[inline]
fn ckey_col(key: u64) -> usize {
    (key & 0xFFFF_FFFF) as usize
}

/// How many live rows have each count: the smallest live row count in O(1)
/// amortised, without a heap (issue #232).
///
/// The search only needs `min_i r_i` over live rows (as a lower bound), and
/// duplicates cannot change a minimum, so unlike the column heap this one
/// has no hidden state to preserve: it reports exactly what the lazy row heap
/// it replaces reported. Each live row is registered under the count it had
/// when last [`Self::set`]; the caller re-registers every row whose count
/// changed before the next query, and [`Self::remove`]s a row as it is
/// eliminated.
struct RowCounts {
    hist: Vec<usize>,
    reg: Vec<usize>,
    /// No live row has a count below this.
    lo: usize,
}

impl RowCounts {
    fn new(rcnt: &[usize]) -> Self {
        let mut hist = vec![0usize; rcnt.len() + 1];
        for &c in rcnt {
            hist[c] += 1;
        }
        Self {
            hist,
            reg: rcnt.to_vec(),
            lo: 0,
        }
    }

    fn set(&mut self, i: usize, c: usize) {
        self.hist[self.reg[i]] -= 1;
        if c >= self.hist.len() {
            self.hist.resize(c + 1, 0);
        }
        self.hist[c] += 1;
        self.reg[i] = c;
        self.lo = self.lo.min(c);
    }

    fn remove(&mut self, i: usize) {
        self.hist[self.reg[i]] -= 1;
    }

    /// Smallest count among live rows, `None` when no row is live.
    fn min(&mut self) -> Option<usize> {
        while self.lo < self.hist.len() && self.hist[self.lo] == 0 {
            self.lo += 1;
        }
        (self.lo < self.hist.len()).then_some(self.lo)
    }
}

/// Right-looking threshold-Markowitz LU of `a`.
///
/// `u_thresh` is the relative pivot threshold (`0 < u ≤ 1`; `1.0` degenerates to
/// partial pivoting within the chosen column). `max_search` is the Suhl cutoff:
/// stop examining candidate columns once this many heap entries have been
/// examined and a candidate is in hand. `zero_pivot_tol` is scaled by `max|A|`
/// exactly as the Gilbert–Peierls path scales it.
///
/// `row_pool` supplies recycled buffers for the `U` rows (popped, cleared,
/// refilled; an empty pool allocates), and `l_nnz_hint` pre-sizes the `L`
/// arrays. Both only change where memory comes from, never the result; they
/// are how [`SparseLu::refactor_markowitz`](super::SparseLu::refactor_markowitz)
/// avoids one allocation per pivot (issue #232).
#[allow(clippy::too_many_arguments)]
pub(super) fn markowitz_factor(
    a: &SparseColMatrix,
    u_thresh: f64,
    max_search: usize,
    zero_pivot_tol: f64,
    on_singular: LuSingularAction,
    row_pool: &mut Vec<Vec<(usize, f64)>>,
    l_nnz_hint: usize,
) -> Result<MarkowitzFactor, FeralError> {
    let m = a.m;

    let mut a_max = 0.0_f64;
    for &v in a.values.iter() {
        a_max = a_max.max(v.abs());
    }
    let ztol = zero_pivot_tol * a_max;

    // ---- active submatrix ------------------------------------------------
    // Explicit zeros are dropped, so count the stored nonzeros first and size
    // the packed lists exactly.
    let mut ccnt = vec![0usize; m];
    let mut rcnt = vec![0usize; m];
    for (j, cj) in ccnt.iter_mut().enumerate() {
        for idx in a.col_ptr[j]..a.col_ptr[j + 1] {
            if a.values[idx] != 0.0 {
                *cj += 1;
                rcnt[a.row_idx[idx]] += 1;
            }
        }
    }
    let mut cols: Lists<(usize, f64)> = Lists::with_capacities(&ccnt);
    let mut row_cols: Lists<usize> = Lists::with_capacities(&rcnt);
    for j in 0..m {
        for idx in a.col_ptr[j]..a.col_ptr[j + 1] {
            let (i, v) = (a.row_idx[idx], a.values[idx]);
            if v != 0.0 {
                cols.push(j, (i, v));
                row_cols.push(i, j);
            }
        }
    }
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

    // Lazy min-heap over the column counts; stale pairs are discarded on pop.
    // (The row side only needs the smallest live row count: `RowCounts`.)
    //
    // **The heap contents are part of the pivot rule.** The heap can hold
    // several live copies of `(c, j)` (a column re-pushed without its count
    // changing, or a count returning to an earlier value), the search scans
    // and counts each copy against `max_search`, so how far it looks depends
    // on the multiset of entries. pounce's active-set QP uses this
    // factorization as a rank-reveal probe, and a search that examined each
    // column once instead (an earlier revision of #232) changed which rows it
    // declared dependent and made QGFRDXPN cycle to the iteration cap
    // (PR #234). So the multiset is reproduced exactly, not approximated.
    //
    // Nothing pops the heap until the first step with no singleton pivot, so
    // up to that step it is just "initial entries + every push", in
    // order-free form. During the singleton peel those entries go into a
    // plain log (an append, not a sift), and the heap is built from the log by
    // one heapify when the stacks first run dry. A min-heap's sequence of
    // popped values depends only on its multiset (equal entries are equal
    // values), so the search then sees exactly what the eagerly maintained
    // heap would have held. Entries for columns already eliminated are left
    // out of the build: they are discarded on every pop and a dead column
    // never comes back, so dropping them changes nothing but cost. On a
    // near-triangular LP basis that is ~17k entries the eager heap sifted in
    // and then popped one at a time (issue #232).
    let mut heaps_built = false;
    // Entries are `(count, column)` packed as `count << 32 | column`
    // (`ckey`), which orders exactly like the tuple for values below 2^32 and
    // makes each heap comparison and move one `u64` (issue #232).
    if m > u32::MAX as usize {
        return Err(FeralError::InvalidInput(format!(
            "threshold-Markowitz LU supports dimension up to {}, got {m}",
            u32::MAX
        )));
    }
    let mut cheap_log: Vec<u64> = (0..m).map(|j| ckey(ccnt[j], j)).collect();
    let mut row_counts = RowCounts::new(&rcnt);
    let mut cheap: BinaryHeap<Reverse<u64>> = BinaryHeap::new();

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
    // `upos[t]`: position of the pivot-row entry in column `uvec[t].0`, found
    // by the gather. Still valid when that column's update runs: each column
    // is updated once per step and only its own update moves its entries.
    let mut upos: Vec<usize> = Vec::new();
    let mut scanned: Vec<usize> = Vec::new();

    // ---- output ----------------------------------------------------------
    let mut perm = Vec::with_capacity(m);
    let mut qcol = Vec::with_capacity(m);
    let mut l_col_ptr = Vec::with_capacity(m + 1);
    let mut l_row_idx: Vec<usize> = Vec::with_capacity(l_nnz_hint);
    let mut l_val: Vec<f64> = Vec::with_capacity(l_nnz_hint);
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
            let (i, v) = cols.get(j)[0];
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
                for &j in row_cols.get(i).iter() {
                    if !alive_c[j] {
                        continue;
                    }
                    if let Some(&(_, v)) = cols.get(j).iter().find(|&&(r, _)| r == i) {
                        found = Some((j, v));
                        break;
                    }
                }
                // A row singleton still has to pass the threshold against its
                // own column — it is a cheap pivot, not a free one.
                if let Some((j, v)) = found {
                    let colmax = cols
                        .get(j)
                        .iter()
                        .fold(0.0_f64, |a, &(_, x)| a.max(x.abs()));
                    if v.abs() > ztol && v.abs() >= u_thresh * colmax {
                        fast = Some((i, j));
                        break;
                    }
                }
            }
        }

        if fast.is_none() && !heaps_built {
            cheap = cheap_log
                .drain(..)
                .filter(|&k| alive_c[ckey_col(k)])
                .map(Reverse)
                .collect::<Vec<_>>()
                .into();
            heaps_built = true;
        }

        // Smallest live row count, a valid lower bound on `r_i` for the
        // remaining columns: no column can beat `(c-1)(minr-1)`.
        let minr = if fast.is_some() {
            1
        } else {
            row_counts.min().map_or(1, |c| c.max(1))
        };

        let mut best: Option<(usize, f64, usize, usize)> = None; // cost, |v|, i, j
        let mut singular_col: Option<usize> = None;
        scanned.clear();
        let mut examined = 0usize;
        while fast.is_none() {
            let Some(&Reverse(key)) = cheap.peek() else {
                break;
            };
            let (c, j) = (ckey_count(key), ckey_col(key));
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
            // A column with several live copies is scanned once per copy and
            // each scan counts against `max_search`: see the heap comment above
            // for why that is kept.
            scanned.push(j);

            let colmax = cols
                .get(j)
                .iter()
                .fold(0.0_f64, |acc, &(_, v)| acc.max(v.abs()));
            if colmax <= ztol {
                // Numerically empty column: not a pivot candidate, and the
                // reason this basis is singular if nothing else is left.
                singular_col = Some(singular_col.map_or(j, |p: usize| p.min(j)));
                continue;
            }
            let bar = u_thresh * colmax;
            for &(i, v) in cols.get(j).iter() {
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
                cheap.push(Reverse(ckey(ccnt[j], j)));
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
                        match cols.get_mut(col).iter_mut().find(|(i, _)| *i == row) {
                            Some(e) => e.1 = if e.1 < 0.0 { -mag } else { mag },
                            None => {
                                cols.push(col, (row, mag));
                                row_cols.push(row, col);
                                rcnt[row] += 1;
                                ccnt[col] += 1;
                                // Reached only with no fast pivot, so the
                                // heaps are built.
                                cheap.push(Reverse(ckey(ccnt[col], col)));
                                row_counts.set(row, rcnt[row]);
                            }
                        }
                        (row, col)
                    }
                }
            }
        };

        let pv = cols
            .get(pj)
            .iter()
            .find(|&&(i, _)| i == pi)
            .map(|&(_, v)| v)
            .ok_or(FeralError::SingularBasis { column: pj })?;

        // ---- gather the pivot column (→ L) and pivot row (→ U) ------------
        lvec.clear();
        let inv = 1.0 / pv;
        for &(i, v) in cols.get(pj).iter() {
            if i != pi {
                lvec.push((i, v * inv));
            }
        }
        uvec.clear();
        upos.clear();
        for &j in row_cols.get(pi).iter() {
            if j == pj || !alive_c[j] || row_seen[j] {
                continue;
            }
            row_seen[j] = true;
            if let Some(k) = cols.get(j).iter().position(|&(i, _)| i == pi) {
                uvec.push((j, cols.get(j)[k].1));
                upos.push(k);
            }
        }
        for &j in row_cols.get(pi).iter() {
            row_seen[j] = false;
        }

        alive_r[pi] = false;
        row_counts.remove(pi);
        alive_c[pj] = false;
        perm.push(pi);
        qcol.push(pj);

        // U row: diagonal plus the surviving row entries. This buffer becomes
        // the `SparseLu` U-row storage itself (the caller remaps it in place),
        // so it comes from the pool when one is supplied.
        let mut urow = row_pool.pop().unwrap_or_default();
        urow.clear();
        urow.reserve(1 + uvec.len());
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
        cols.clear(pj);
        row_cols.clear(pi);

        // ---- rank-1 update, column-oriented ------------------------------
        // In place: each column is walked once, existing entries updated where
        // the pivot column touches them and swap-removed on exact cancellation,
        // and the rows the walk did not meet appended as fill.
        for &(i, lv) in lvec.iter() {
            lscat[i] = lv;
            lmark[i] = true;
        }
        for (&(j, uval), &pos) in uvec.iter().zip(upos.iter()) {
            if lvec.is_empty() || uval == 0.0 {
                // Nothing to subtract: the walk below would only take out the
                // pivot-row entry (every other entry fails its `lmark` test),
                // and it would do so with this same `swap_remove` at this same
                // position. Every column-singleton pivot lands here, so on a
                // near-triangular basis this skips most column walks
                // (issue #232).
                cols.swap_remove(j, pos);
            } else {
                let mut k = 0;
                while k < cols.len[j] {
                    let (i, v) = cols.get(j)[k];
                    if i == pi {
                        cols.swap_remove(j, k); // consumed into the U row
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
                            cols.swap_remove(j, k);
                            continue;
                        }
                        cols.get_mut(j)[k].1 = nv;
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
                            cols.push(j, (i, nv));
                            rcnt[i] += 1;
                            row_cols.push(i, j);
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
            }
            ccnt[j] = cols.len[j];
            // Into the heap as well (or its log, before it exists): the stack
            // is a fast path, not an alternative index. A column that stops
            // being a singleton is dropped from the stack, and if it were not
            // in `cheap` too it would be invisible to the general search and
            // reported singular.
            if heaps_built {
                cheap.push(Reverse(ckey(ccnt[j], j)));
            } else {
                cheap_log.push(ckey(ccnt[j], j));
            }
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
                row_counts.set(i, rcnt[i]);
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

#[cfg(test)]
mod tests {
    use super::{Lists, RowCounts};

    /// `RowCounts::min` against a brute-force minimum over the live rows,
    /// under random re-registrations and removals.
    #[test]
    fn row_counts_min_matches_brute_force() {
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let n = 50;
        let mut cnt: Vec<usize> = (0..n).map(|_| next(8)).collect();
        let mut live = vec![true; n];
        let mut rc = RowCounts::new(&cnt);
        for step in 0..5_000 {
            let i = next(n);
            if live[i] {
                if next(25) == 0 {
                    live[i] = false;
                    rc.remove(i);
                } else {
                    // Counts can exceed the initial histogram length.
                    cnt[i] = next(n + 20);
                    rc.set(i, cnt[i]);
                }
            }
            let brute = (0..n).filter(|&r| live[r]).map(|r| cnt[r]).min();
            assert_eq!(rc.min(), brute, "step {step}");
        }
    }

    /// `Lists` must behave exactly like `Vec<Vec<T>>` under the operations the
    /// elimination uses — element order included, since the pivot search's
    /// tie-breaks read it. The oracle is `std`'s `Vec`.
    #[test]
    fn lists_match_vec_of_vecs_under_random_ops() {
        let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let n = 37;
        let init: Vec<Vec<(usize, f64)>> = (0..n)
            .map(|j| (0..(j % 5)).map(|k| (k, (j * 10 + k) as f64)).collect())
            .collect();
        let caps: Vec<usize> = init.iter().map(|v| v.len()).collect();
        let mut lists: Lists<(usize, f64)> = Lists::with_capacities(&caps);
        for (j, v) in init.iter().enumerate() {
            for &x in v {
                lists.push(j, x);
            }
        }
        let mut oracle = init;
        for step in 0..20_000 {
            let j = next(n);
            match next(10) {
                0..=5 => {
                    let x = (step, step as f64 * 0.5);
                    lists.push(j, x);
                    oracle[j].push(x);
                }
                6..=8 => {
                    if !oracle[j].is_empty() {
                        let k = next(oracle[j].len());
                        lists.swap_remove(j, k);
                        oracle[j].swap_remove(k);
                    }
                }
                _ => {
                    if next(8) == 0 {
                        lists.clear(j);
                        oracle[j].clear();
                    } else if !oracle[j].is_empty() {
                        let k = next(oracle[j].len());
                        lists.get_mut(j)[k].1 = -1.0;
                        oracle[j][k].1 = -1.0;
                    }
                }
            }
            assert_eq!(lists.get(j), oracle[j].as_slice(), "step {step}, list {j}");
        }
        for (j, v) in oracle.iter().enumerate() {
            assert_eq!(lists.get(j), v.as_slice(), "final, list {j}");
        }
    }
}
