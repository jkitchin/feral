//! Sparse rank-1 column-replacement update — Forrest–Tomlin.
//!
//! Replacing basis slot `leaving_slot` (column position `r`) by a new column
//! folds the spike `ρ = G⁻¹ L⁻¹ P aₙₑw` into `U`'s column `r`, then restores the
//! triangular factor by the **Forrest–Tomlin** scheme: a *symmetric permutation*
//! moves column `r` and row `r` to the bottom of the bump (so the bump's diagonal
//! pivots are the old, nonzero `U` diagonals — dodging the zero-superdiagonal
//! landmine of the column-shift Bartels–Golub), and then the single resulting
//! pivotal row is eliminated by one sparse forward sweep (one row-eta).
//!
//! The permutation is *logical*: an evolving `uperm` (pivot-position ↔ triangular
//! rank) carried by `SparseLu` and applied once per solve. `U`'s stored indices,
//! the base `L`, `P`, `Q`, and all prior etas stay in fixed pivot-position
//! coordinates and are never relabeled. So one update is `O(bump)` for sparse
//! `U` — the cyclic shift of the rank range plus the elimination of one row — and
//! records an `O(bump)` eta, not the `O(bump²)` of a full bump re-triangularization
//! (`dev/research/ft-row-elimination-design-2026-06-21.md`, issue #87).
//!
//! The work is otherwise bump-local: the spike is computed by a Gilbert–Peierls
//! depth-first reach, and the "unchanged on failure" guarantee saves/restores only
//! the changed rows and the bump's `uperm` range (no `O(nnz)` clone of `U`).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::sparse_factor::{FtEta, FtOp, SparseLu};
use super::sparse_symbolic::SparseLuSymbolic;
use crate::error::FeralError;
use crate::lu::sparse_matrix::SparseColMatrix;

impl SparseLu {
    /// Number of rank-1 updates applied since the last factor/refactor.
    #[inline]
    pub fn updates_since_refactor(&self) -> usize {
        self.etas.len()
    }

    /// Replace basis slot `leaving_slot` with the dense entering column `aₙₑw`.
    ///
    /// Convenience wrapper over [`SparseLu::update_sparse`]: it scans `aₙₑw`
    /// once (`O(n)`) for its nonzeros, then delegates. Callers that already hold
    /// the sparse entering column should call `update_sparse` directly to avoid
    /// the scan.
    ///
    /// Returns [`FeralError::NeedsRefactor`] (leaving `self` unchanged) when the
    /// update or growth budget is exceeded, or when the bump has no acceptable
    /// pivot (a singular replacement basis). In every failure mode the update
    /// signals NeedsRefactor rather than SingularBasis — matching the dense
    /// update path ([`super::DenseLu::update`]); the authoritative singularity
    /// verdict comes from a fresh factorization, not the incremental update.
    pub fn update(&mut self, leaving_slot: usize, entering_col: &[f64]) -> Result<(), FeralError> {
        if entering_col.len() != self.m {
            return Err(FeralError::DimensionMismatch {
                expected: self.m,
                got: entering_col.len(),
            });
        }
        let sparse: Vec<(usize, f64)> = entering_col
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0.0)
            .map(|(i, &v)| (i, v))
            .collect();
        self.update_sparse(leaving_slot, &sparse)
    }

    /// Replace basis slot `leaving_slot` with the entering column given by its
    /// nonzeros `(row, value)` (rows need not be sorted; duplicates are summed).
    /// Fully bump-local: cost is `O(bump + nnz(aₙₑw))`, with no `O(n)` term.
    ///
    /// Returns [`FeralError::NeedsRefactor`] on any failure, as for
    /// [`SparseLu::update`].
    pub fn update_sparse(
        &mut self,
        leaving_slot: usize,
        entering: &[(usize, f64)],
    ) -> Result<(), FeralError> {
        let m = self.m;
        if leaving_slot >= m {
            return Err(FeralError::InvalidInput(format!(
                "leaving_slot {} out of range for basis dimension {}",
                leaving_slot, m
            )));
        }
        for &(row, _) in entering.iter() {
            if row >= m {
                return Err(FeralError::InvalidInput(format!(
                    "entering-column row {} out of range for dimension {}",
                    row, m
                )));
            }
        }
        if self.updates_since_refactor() + 1 > self.params.max_updates {
            return Err(FeralError::NeedsRefactor);
        }

        // True per-update build cost (scalar multiply-adds): the spike solve plus
        // the row-elimination scatters. Folded into the public `last_update_work`
        // / `update_work` counters only on commit (issue #89).
        let mut work: usize = 0;

        // --- Sparse spike ρ = G⁻¹ L⁻¹ P (scaled aₙₑw) via Gilbert–Peierls reach ---
        let mut w = std::mem::take(&mut self.ft_work); // dedicated buffer, zero on entry
        let mut touched: Vec<usize> = Vec::new(); // positions made nonzero in w
        self.compute_spike(entering, leaving_slot, &mut w, &mut touched, &mut work);

        let r = self.qcol_inv[leaving_slot];
        let r_rank = self.uperm[r];

        // Spike support (pivot positions) and the bump's bottom rank: the deepest
        // rank at which the spike has an entry. The bump is the rank range
        // `[r_rank, h_rank]`; `h_rank < r_rank` means the new column has nothing at
        // or below its own diagonal in rank order ⇒ singular replacement.
        let mut supp: Vec<usize> = touched.iter().copied().filter(|&k| w[k] != 0.0).collect();
        supp.sort_unstable();
        supp.dedup();
        let h_rank = supp.iter().map(|&p| self.uperm[p]).max();
        let h_rank = match h_rank {
            Some(hr) if hr >= r_rank => hr,
            _ => {
                clear(&mut w, &touched);
                self.ft_work = w;
                // Singular as far as the incremental update can tell; refactor
                // from scratch for the authoritative verdict (see DenseLu::update).
                return Err(FeralError::NeedsRefactor);
            }
        };

        // --- Rows whose U content changes: row `r` (rebuilt by the elimination),
        // the spike support (gains a column-`r` entry), and the old column-`r`
        // holders (lose theirs). The other bump rows are NOT touched — that is the
        // Forrest–Tomlin win. ---
        let mut changed: Vec<usize> = Vec::with_capacity(1 + supp.len() + self.u_above[r].len());
        changed.push(r);
        changed.extend(supp.iter().copied());
        changed.extend(self.u_above[r].iter().copied());
        changed.sort_unstable();
        changed.dedup();

        // Save the changed rows (pooled buffers) for rollback / u_above refresh.
        let mut saved = std::mem::take(&mut self.saved_scratch);
        saved.clear();
        for &i in changed.iter() {
            let mut buf = self.saved_pool.pop().unwrap_or_default();
            buf.clear();
            buf.extend_from_slice(&self.u_rows[i]);
            saved.push((i, buf));
        }
        // Save the bump's `uperm` range so the cyclic shift can be rolled back.
        let saved_uperm_inv: Vec<usize> = self.uperm_inv[r_rank..=h_rank].to_vec();

        // Replace column `r` of U with the spike in every row but `r` (row `r`'s
        // column-`r` value is the spike diagonal `w[r]`, handled by the elimination).
        self.set_column_r(r, &w, &supp);
        // Symmetric cyclic shift: move rank `r_rank` to `h_rank` (column `r` and
        // row `r` to the bottom of the bump). Pure index bookkeeping, O(bump).
        self.shift_uperm(r, r_rank, h_rank);

        // Eliminate the single pivotal row `r` (now at rank `h_rank`).
        let result = self.eliminate_pivot_row(r, h_rank, w[r], &mut work);

        clear(&mut w, &touched);
        self.ft_work = w;

        match result {
            Ok(ops) => {
                // Element-growth high-water over the changed rows (L5): only row
                // `r` and the spike rows changed values, so this stays O(changed).
                let mut changed_max = 0.0_f64;
                for &i in changed.iter() {
                    for &(_, v) in self.u_rows[i].iter() {
                        changed_max = changed_max.max(v.abs());
                    }
                }
                let growth = self.growth.max(changed_max / self.u_max0);
                if growth > self.params.max_growth {
                    self.rollback(saved, &saved_uperm_inv, r_rank);
                    return Err(FeralError::NeedsRefactor);
                }
                // Commit: refresh the `u_above` column index *incrementally*. Only
                // two things changed structurally (issue #89): `set_column_r`
                // rewrote column `r` (its holders are now exactly the spike
                // support), and `eliminate_pivot_row` rebuilt row `r`. Every other
                // "changed" row only gained or lost its single column-`r` entry —
                // already captured by column `r`'s holder list — so its membership
                // in *other* columns is untouched and must NOT be re-indexed. The
                // old code re-indexed every changed row wholesale, an
                // `O(bump · rowlen · shift)` (≈ `O(m³)` on a dense bump) churn that
                // dwarfed the elimination's `O(factor_nnz)` arithmetic.
                //
                // (a) Column `r`'s holders = spike support minus `r`. `supp` is
                //     already sorted+deduped and `p != r` preserves order, so the
                //     list stays sorted.
                self.u_above[r].clear();
                self.u_above[r].extend(supp.iter().copied().filter(|&p| p != r));
                // (b) Row `r` changed its column set: drop `r` from its old
                //     columns' holder lists and add it to the new ones. (Both skip
                //     column `r` itself — that is the diagonal, not a `u_above`
                //     entry — so this never touches the list rebuilt in (a).)
                //     `saved` is a local moved out of `self`, so borrowing the
                //     snapshot while mutating `self` is sound (no clone needed).
                let old_row_r: &[(usize, f64)] = saved
                    .iter()
                    .find(|(i, _)| *i == r)
                    .map(|(_, b)| b.as_slice())
                    .unwrap_or(&[]);
                self.unindex_above(r, old_row_r);
                let new_row_r = std::mem::take(&mut self.u_rows[r]);
                self.index_above(r, &new_row_r);
                self.u_rows[r] = new_row_r;
                for (_, buf) in saved.drain(..) {
                    self.saved_pool.push(buf);
                }
                self.saved_scratch = saved;
                self.etas.push(FtEta { ops });
                self.growth = growth;
                self.last_update_work = work;
                self.update_work_total += work;
                #[cfg(feature = "lu-ft-invariant-check")]
                self.debug_check_invariants();
                Ok(())
            }
            Err(e) => {
                self.rollback(saved, &saved_uperm_inv, r_rank);
                Err(e)
            }
        }
    }

    /// Restore `u_rows` (from the saved snapshots) and the bump's `uperm` range
    /// after a failed update — leaving `self` exactly as it was. `u_above` was not
    /// yet modified (it is refreshed only on commit), so it needs no restore.
    fn rollback(
        &mut self,
        mut saved: Vec<(usize, Vec<(usize, f64)>)>,
        saved_uperm_inv: &[usize],
        r_rank: usize,
    ) {
        for (i, row) in saved.drain(..) {
            self.u_rows[i] = row;
        }
        self.saved_scratch = saved;
        for (off, &pos) in saved_uperm_inv.iter().enumerate() {
            let rank = r_rank + off;
            self.uperm_inv[rank] = pos;
            self.uperm[pos] = rank;
        }
    }

    /// Discard all pending updates and re-factor from scratch on `a`, reusing
    /// the column ordering in `symbolic`.
    pub fn refactor(
        &mut self,
        a: &SparseColMatrix,
        symbolic: &SparseLuSymbolic,
    ) -> Result<(), FeralError> {
        *self = SparseLu::factor(a, symbolic, self.params.clone())?;
        Ok(())
    }

    /// Compute the spike `ρ = G⁻¹ L⁻¹ P (D_row Π aₙₑw D_col[slot])` into the dense
    /// work vector `w`, recording every touched position in `touched`. Seeds
    /// directly from the sparse entering column (no `O(n)` scan), then uses a
    /// Gilbert–Peierls depth-first reach so only the reachable `L`-columns are
    /// visited, and finally replays the FT etas forward.
    fn compute_spike(
        &mut self,
        entering: &[(usize, f64)],
        leaving_slot: usize,
        w: &mut [f64],
        touched: &mut Vec<usize>,
        work: &mut usize,
    ) {
        let dcol = self.scale.d_col[leaving_slot];
        let mut mark = std::mem::take(&mut self.scratch_mark);
        let mut stack: Vec<usize> = Vec::new();

        // Scatter the scaled entering column into w (pivot-position space) and
        // seed the reach. An original-row entry `o` scales to scaled row
        // `i = rperm_inv[o]` (factor `d_row[i]·dcol`) and lands at pivot position
        // `perm_inv[i]`. Duplicates accumulate (`+=`).
        for &(o, val) in entering.iter() {
            let i = self.scale_rperm_inv[o];
            let v = self.scale.d_row[i] * val * dcol;
            if v == 0.0 {
                continue;
            }
            let k = self.perm_inv[i];
            w[k] += v;
            if !mark[k] {
                mark[k] = true;
                touched.push(k);
                stack.push(k);
            }
        }

        // Depth-first reach over the graph of L (column k -> its rows).
        let mut reach: Vec<usize> = touched.clone();
        while let Some(k) = stack.pop() {
            let (lo, hi) = (self.l_col_ptr[k], self.l_col_ptr[k + 1]);
            for idx in lo..hi {
                let i = self.l_row_idx[idx];
                if !mark[i] {
                    mark[i] = true;
                    touched.push(i);
                    reach.push(i);
                    stack.push(i);
                }
            }
        }
        reach.sort_unstable(); // ascending = valid topological order for L

        // Sparse forward solve L y = w over the reach.
        for &k in reach.iter() {
            let yk = w[k];
            if yk == 0.0 {
                continue;
            }
            let (lo, hi) = (self.l_col_ptr[k], self.l_col_ptr[k + 1]);
            *work += hi - lo;
            for idx in lo..hi {
                w[self.l_row_idx[idx]] -= self.l_val[idx] * yk;
            }
        }

        // Replay the FT etas forward (G⁻¹), tracking newly touched positions.
        for eta in self.etas.iter() {
            *work += eta.ops.len();
            for op in eta.ops.iter() {
                match *op {
                    FtOp::Axpy { target, src, mult } => {
                        w[target] -= mult * w[src];
                        if w[target] != 0.0 && !mark[target] {
                            mark[target] = true;
                            touched.push(target);
                        }
                    }
                }
            }
        }

        // Clear the marker for the touched set and hand the buffer back.
        for &k in touched.iter() {
            mark[k] = false;
        }
        self.scratch_mark = mark;
    }

    /// Replace column `r` of `U` with the spike (`w` at positions `supp`) in every
    /// row except `r`. Old column-`r` entries (located via the `u_above` index,
    /// read only — refreshed wholesale on commit) are removed; the new spike
    /// entries are inserted into the off-diagonal part of each support row. Row
    /// `r`'s own column-`r` value is the spike diagonal `w[r]`, consumed directly
    /// by [`Self::eliminate_pivot_row`], so row `r` is not touched here.
    fn set_column_r(&mut self, r: usize, w: &[f64], supp: &[usize]) {
        let old_holders = self.u_above[r].clone();
        for &i in old_holders.iter() {
            remove_offdiag(&mut self.u_rows[i], r);
        }
        for &p in supp.iter() {
            if p != r {
                insert_offdiag(&mut self.u_rows[p], r, w[p]);
            }
        }
    }

    /// Symmetric cyclic shift of the rank range `[r_rank, h_rank]`: move the
    /// leaving column's position `r` from rank `r_rank` to rank `h_rank`, sliding
    /// every other position in the range down one rank. Pure `uperm`/`uperm_inv`
    /// bookkeeping (`O(bump)`); `U`'s stored entries and the etas are untouched.
    fn shift_uperm(&mut self, r: usize, r_rank: usize, h_rank: usize) {
        for rank in r_rank..h_rank {
            let pos = self.uperm_inv[rank + 1];
            self.uperm_inv[rank] = pos;
            self.uperm[pos] = rank;
        }
        self.uperm_inv[h_rank] = r;
        self.uperm[r] = h_rank;
    }

    /// Forrest–Tomlin elimination of the single pivotal row `r` (now at rank
    /// `h_rank`, the bottom of the bump after [`Self::shift_uperm`]). Its
    /// sub-diagonal entries — columns whose new rank is `< h_rank` — are cleared by
    /// a sparse forward sweep against the (unmodified) upper-triangular bump rows,
    /// in increasing rank order. Each eliminated column contributes one
    /// `FtOp::Axpy{target: r, ..}` to the returned eta. Only row `r` is rewritten;
    /// `diag0 = w[r]` seeds its column-`r` value.
    ///
    /// Returns [`FeralError::NeedsRefactor`] if the resulting diagonal pivot
    /// vanishes (singular replacement). `touched` accumulates the dense-scatter
    /// positions so the caller clears them.
    fn eliminate_pivot_row(
        &mut self,
        r: usize,
        h_rank: usize,
        diag0: f64,
        work: &mut usize,
    ) -> Result<Vec<FtOp>, FeralError> {
        let ztol = self.params.zero_pivot_tol * self.u_max0;
        let mut ops: Vec<FtOp> = Vec::new();

        let mut rw = std::mem::take(&mut self.ft_rw); // dense scatter, zero on entry
        let mut rw_touched = std::mem::take(&mut self.targets_scratch);
        rw_touched.clear();
        let mut queued = std::mem::take(&mut self.scratch_mark); // all-false on entry
        let mut heap: BinaryHeap<Reverse<usize>> = BinaryHeap::new();

        // Seed: row r off-diagonals (column `r` excluded — its value is `diag0`),
        // pushing the sub-diagonal columns onto the heap.
        let row_r = std::mem::take(&mut self.u_rows[r]);
        *work += row_r.len();
        for &(c, v) in row_r.iter() {
            if c == r {
                continue; // old diagonal discarded; replaced by the spike value
            }
            scatter_into(
                &mut rw,
                &mut rw_touched,
                &mut queued,
                &mut heap,
                c,
                v,
                &self.uperm,
                h_rank,
            );
        }
        // Column r is the bump diagonal (rank h_rank): never sub-diagonal, so it is
        // not pushed to the heap; just record its starting value.
        if rw[r] == 0.0 {
            rw_touched.push(r);
        }
        rw[r] += diag0;
        self.row_pool.push(row_r); // recycle the old row's buffer

        // Sweep sub-diagonal columns of row r in increasing rank order.
        while let Some(Reverse(rank)) = heap.pop() {
            let c = self.uperm_inv[rank];
            queued[c] = false;
            let vrc = rw[c];
            if vrc == 0.0 {
                continue;
            }
            let prow = &self.u_rows[c];
            let &(dc, piv) = match prow.first() {
                Some(p) => p,
                None => {
                    self.restore_elim_pools(rw, rw_touched, queued);
                    return Err(FeralError::NeedsRefactor);
                }
            };
            if dc != c || piv == 0.0 || !piv.is_finite() {
                self.restore_elim_pools(rw, rw_touched, queued);
                return Err(FeralError::NeedsRefactor);
            }
            let mult = vrc / piv;
            ops.push(FtOp::Axpy {
                target: r,
                src: c,
                mult,
            });
            // row_r -= mult · row_c. Clear column `c` *exactly* (it equals
            // `vrc - mult·piv = 0` mathematically, but the floating-point residual
            // would otherwise re-cross zero and re-enqueue `c` — an infinite loop);
            // skip the pivot's own diagonal in the scatter accordingly. Fill at
            // strictly higher ranks may enqueue further sub-diagonal columns.
            rw[c] = 0.0;
            // One scatter per off-diagonal of the eliminated row — the
            // fill-proportional term that makes the update O(factor_nnz) on a
            // dense bump (issue #89).
            *work += self.u_rows[c].len();
            for &(cc, v) in self.u_rows[c].iter() {
                if cc == c {
                    continue;
                }
                scatter_into(
                    &mut rw,
                    &mut rw_touched,
                    &mut queued,
                    &mut heap,
                    cc,
                    -mult * v,
                    &self.uperm,
                    h_rank,
                );
            }
        }

        // Diagonal pivot check, then gather the rebuilt row r (diagonal first).
        let diag = rw[r];
        if diag.abs() <= ztol || !diag.is_finite() {
            self.restore_elim_pools(rw, rw_touched, queued);
            return Err(FeralError::NeedsRefactor);
        }
        let mut new_row = self.row_pool.pop().unwrap_or_default();
        new_row.clear();
        new_row.push((r, diag));
        let mut offdiag: Vec<(usize, f64)> = rw_touched
            .iter()
            .filter(|&&c| c != r && rw[c] != 0.0)
            .map(|&c| (c, rw[c]))
            .collect();
        offdiag.sort_unstable_by_key(|&(c, _)| c);
        // `rw_touched` may list a column more than once (its scatter value crossed
        // zero and back), so drop duplicate columns — each carries the same final
        // `rw[c]`. Leaving them would put duplicate entries in the row and corrupt
        // `U` / `u_above`.
        offdiag.dedup_by_key(|&mut (c, _)| c);
        new_row.extend_from_slice(&offdiag);
        self.u_rows[r] = new_row;

        // Clear the dense scatter and hand the pools back.
        self.restore_elim_pools(rw, rw_touched, queued);
        Ok(ops)
    }

    /// Clear the dense scatter `rw` and the `queued` marker over their touched
    /// positions (so both reach all-zero / all-false for the next update, on every
    /// exit path — including a mid-sweep error where the heap still held columns)
    /// and return the row-elimination churn buffers to their `SparseLu` pools.
    fn restore_elim_pools(
        &mut self,
        mut rw: Vec<f64>,
        mut rw_touched: Vec<usize>,
        mut queued: Vec<bool>,
    ) {
        for &c in rw_touched.iter() {
            rw[c] = 0.0;
            queued[c] = false;
        }
        rw_touched.clear();
        self.ft_rw = rw;
        self.targets_scratch = rw_touched;
        self.scratch_mark = queued;
    }

    /// Structural self-check (opt-in via the `lu-ft-invariant-check` feature, off
    /// by default because it allocates `O(m)` per update): `uperm` bijection,
    /// diagonal-first rows, upper-triangular-in-`uperm` order, and `u_above`
    /// matching `U`'s off-diagonal pattern exactly. Catches FT-update bookkeeping
    /// drift at its source.
    #[cfg(feature = "lu-ft-invariant-check")]
    fn debug_check_invariants(&self) {
        let m = self.m;
        for k in 0..m {
            debug_assert_eq!(self.uperm[self.uperm_inv[k]], k, "uperm not a bijection");
        }
        // Rebuild the expected off-diagonal column index from U.
        let mut expect: Vec<Vec<usize>> = vec![Vec::new(); m];
        for (i, row) in self.u_rows.iter().enumerate() {
            debug_assert!(!row.is_empty(), "row {i} empty (no diagonal)");
            debug_assert_eq!(row[0].0, i, "row {i} diagonal not stored first");
            for &(c, _) in row[1..].iter() {
                debug_assert!(
                    self.uperm[c] > self.uperm[i],
                    "row {i} (rank {}) off-diagonal column {c} (rank {}) not upper in uperm",
                    self.uperm[i],
                    self.uperm[c]
                );
                expect[c].push(i);
            }
        }
        for (c, (exp, idx)) in expect.iter_mut().zip(self.u_above.iter()).enumerate() {
            exp.sort_unstable();
            let mut got = idx.clone();
            got.sort_unstable();
            debug_assert_eq!(
                &got, exp,
                "u_above[{c}] mismatch (duplicate or stale entry)"
            );
        }
    }

    /// Remove row `i`'s off-diagonal entries from the `u_above` column index
    /// (using its pre-update content `old_row`).
    fn unindex_above(&mut self, i: usize, old_row: &[(usize, f64)]) {
        for &(c, _) in old_row.iter() {
            if c != i {
                if let Ok(pos) = self.u_above[c].binary_search(&i) {
                    self.u_above[c].remove(pos);
                }
            }
        }
    }

    /// Add row `i`'s off-diagonal entries to the `u_above` column index.
    fn index_above(&mut self, i: usize, new_row: &[(usize, f64)]) {
        for &(c, _) in new_row.iter() {
            if c != i {
                if let Err(pos) = self.u_above[c].binary_search(&i) {
                    self.u_above[c].insert(pos, i);
                }
            }
        }
    }
}

fn clear(w: &mut [f64], touched: &[usize]) {
    for &k in touched.iter() {
        w[k] = 0.0;
    }
}

/// Add `v` to the dense scatter `rw[c]` of the pivotal row, recording `c` as
/// touched on its first nonzero and enqueuing it (by triangular rank) when it is
/// a not-yet-queued sub-diagonal column (`uperm[c] < h_rank`). `queued` dedups the
/// heap; a column is re-enqueueable only after it is popped (rank order guarantees
/// fill lands at strictly higher ranks, so no column is processed twice).
#[allow(clippy::too_many_arguments)]
fn scatter_into(
    rw: &mut [f64],
    rw_touched: &mut Vec<usize>,
    queued: &mut [bool],
    heap: &mut BinaryHeap<Reverse<usize>>,
    c: usize,
    v: f64,
    uperm: &[usize],
    h_rank: usize,
) {
    if rw[c] == 0.0 {
        rw_touched.push(c);
    }
    rw[c] += v;
    if uperm[c] < h_rank && !queued[c] && rw[c] != 0.0 {
        queued[c] = true;
        heap.push(Reverse(uperm[c]));
    }
}

/// Look up the value at column `c` in a row stored diagonal-first with a
/// column-sorted off-diagonal tail (`row[0]` is the diagonal; `row[1..]` is
/// sorted ascending by column). After a Forrest–Tomlin symmetric permutation the
/// diagonal column need not be the row's minimum column, so the diagonal is
/// checked separately from the binary-searched tail.
#[cfg(test)]
fn get_col(row: &[(usize, f64)], c: usize) -> Option<f64> {
    if let Some(&(dc, dv)) = row.first() {
        if dc == c {
            return Some(dv);
        }
    }
    if row.len() <= 1 {
        return None;
    }
    row[1..]
        .binary_search_by_key(&c, |&(col, _)| col)
        .ok()
        .map(|pos| row[1 + pos].1)
}

/// Remove off-diagonal column `c` from a diagonal-first row, if present. Never
/// removes the diagonal (`row[0]`).
fn remove_offdiag(row: &mut Vec<(usize, f64)>, c: usize) {
    if row.len() <= 1 {
        return;
    }
    if let Ok(pos) = row[1..].binary_search_by_key(&c, |&(col, _)| col) {
        row.remove(1 + pos);
    }
}

/// Insert/replace off-diagonal column `c` = `v` in a diagonal-first row (the tail
/// `row[1..]` stays column-sorted); remove it if `v == 0`. `c` must not be the
/// row's own diagonal column.
fn insert_offdiag(row: &mut Vec<(usize, f64)>, c: usize, v: f64) {
    // Empty row (no diagonal) should not occur for a valid pivot position, but be
    // defensive: fall back to a plain insert.
    if row.is_empty() {
        if v != 0.0 {
            row.push((c, v));
        }
        return;
    }
    match row[1..].binary_search_by_key(&c, |&(col, _)| col) {
        Ok(pos) => {
            if v != 0.0 {
                row[1 + pos].1 = v;
            } else {
                row.remove(1 + pos);
            }
        }
        Err(pos) => {
            if v != 0.0 {
                row.insert(1 + pos, (c, v));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::lu::sparse_factor::SparseLu;
    use crate::lu::sparse_matrix::SparseColMatrix;
    use crate::lu::sparse_symbolic::SparseLuSymbolic;
    use crate::lu::LuParams;

    /// L5 (dev/research/repo-review-2026-06-09.md): the sparse growth monitor
    /// recorded only the largest single elimination multiplier, so compounded
    /// element growth across a chain of updates went unmonitored. After the fix
    /// `growth` is the ‖U‖∞ high-water ratio (max|U| over update history ÷
    /// max|U| at factor), which compounds. Oracle is the independent
    /// recomputation of that ratio from the committed `U` (via `u_dense`), not
    /// the monitor's own bookkeeping; pre-fix `growth` is the max single
    /// multiplier and does not match.
    #[test]
    fn growth_monitor_tracks_compounded_element_growth() {
        let cols = vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0, 5.0],
        ];
        let m = 4;
        let params = LuParams {
            max_updates: 20,
            max_growth: 1e12,
            ..LuParams::default()
        };
        let mut lu = SparseLu::factor_dense_columns(m, &cols, params).expect("factor");

        let umax = |lu: &SparseLu| {
            let mut mx = 0.0_f64;
            for i in 0..m {
                for j in 0..m {
                    mx = mx.max(lu.u_dense(i, j).abs());
                }
            }
            mx
        };
        let u_max0 = umax(&lu);
        let mut hw = 1.0_f64;

        // Replace the last slot each time: no bump forms (the spike lands in the
        // last column), every update commits, and max|U| in that column grows.
        let updates = [
            (3usize, vec![0.0, 0.0, 1.0, 20.0]),
            (3usize, vec![0.0, 0.0, 1.0, 60.0]),
            (3usize, vec![0.0, 0.0, 1.0, 180.0]),
        ];
        for (i, (slot, col)) in updates.iter().enumerate() {
            lu.update(*slot, col)
                .unwrap_or_else(|e| panic!("update {i} should commit: {e:?}"));
            hw = hw.max(umax(&lu) / u_max0);
            assert!(
                (lu.growth - hw).abs() <= 1e-9 * hw,
                "growth monitor {} must equal element-growth high-water {}",
                lu.growth,
                hw
            );
        }
        assert!(hw > 1.0, "test must exercise genuine element growth");
    }

    /// Issue #93: the public `growth`/`u_max0` getters must expose exactly the
    /// internal fields. Fresh factor reports `growth == 1.0`; after a committed
    /// update the getter tracks the internal high-water field.
    #[test]
    fn growth_getters_expose_internal_fields() {
        let cols = vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0, 5.0],
        ];
        let m = 4;
        let params = LuParams {
            max_updates: 20,
            max_growth: 1e12,
            ..LuParams::default()
        };
        let mut lu = SparseLu::factor_dense_columns(m, &cols, params).expect("factor");

        assert_eq!(lu.growth(), 1.0, "fresh factor growth is 1.0");
        assert_eq!(lu.growth(), lu.growth, "getter mirrors internal field");
        assert_eq!(lu.u_max0(), lu.u_max0, "getter mirrors internal field");
        assert!(
            lu.u_max0() > 0.0,
            "reference max|U| is floored away from zero"
        );

        lu.update(3, &[0.0, 0.0, 1.0, 60.0])
            .expect("update commits");
        assert_eq!(
            lu.growth(),
            lu.growth,
            "getter mirrors internal after update"
        );
        assert_eq!(lu.u_max0(), lu.u_max0, "u_max0 unchanged by update");
    }

    /// After a chain of wide-bump dense-column updates, `U` must stay upper
    /// triangular *in `uperm` order* (every off-diagonal entry's column outranks
    /// its row) and diagonal-first — the structural invariant the row-ordered
    /// solves and the FT elimination both rely on. Tridiagonal base ⇒ dense spike
    /// ⇒ full-width bump (the issue #87 regime).
    #[test]
    fn uperm_triangular_invariant_holds_after_wide_bump_chain() {
        let m = 12;
        let mut cols: Vec<Vec<f64>> = vec![vec![0.0; m]; m];
        for (j, col) in cols.iter_mut().enumerate() {
            col[j] = 4.0;
            if j > 0 {
                col[j - 1] = -1.0;
            }
            if j + 1 < m {
                col[j + 1] = -1.0;
            }
        }
        let params = LuParams {
            max_updates: 1000,
            max_growth: 1e30,
            ..LuParams::default()
        };
        let mut lu = SparseLu::factor_dense_columns(m, &cols, params).expect("factor");

        for s in 0..15usize {
            let slot = s % 4;
            let mut col = vec![0.0; m];
            for (i, ci) in col.iter_mut().enumerate() {
                *ci = 0.5 + ((i * 5 + s * 3) % 7) as f64 * 0.25;
            }
            col[slot] = 40.0 + s as f64;
            if lu.update(slot, &col).is_err() {
                continue; // a legitimately singular/over-budget step; skip
            }
            // Invariant 1: diagonal-first.
            for (i, row) in lu.u_rows.iter().enumerate() {
                assert_eq!(row[0].0, i, "row {i} diagonal must be stored first");
            }
            // Invariant 2: upper-triangular in `uperm` order.
            for (i, row) in lu.u_rows.iter().enumerate() {
                for &(c, _) in row[1..].iter() {
                    assert!(
                        lu.uperm[c] > lu.uperm[i],
                        "update {s}: off-diagonal U[{i},{c}] has rank {} <= row rank {}",
                        lu.uperm[c],
                        lu.uperm[i]
                    );
                }
            }
            // Invariant 3: uperm and uperm_inv are mutual inverses.
            for k in 0..m {
                assert_eq!(lu.uperm[lu.uperm_inv[k]], k);
            }
        }
    }

    /// `last_update_work`/`update_work` accounting (issue #89): zero after factor,
    /// strictly positive and equal to `last_update_work` after the first commit,
    /// accumulating across updates, and reset by `refactor`. Oracle is the
    /// definitional bookkeeping (no external solver needed): the cumulative
    /// counter is the running sum of the per-update counter.
    #[test]
    fn update_work_accumulates_and_resets() {
        let cols = vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0, 5.0],
        ];
        let m = 4;
        let params = LuParams {
            max_updates: 20,
            max_growth: 1e12,
            ..LuParams::default()
        };
        let mut lu = SparseLu::factor_dense_columns(m, &cols, params).expect("factor");
        assert_eq!(lu.last_update_work(), 0, "no work before any update");
        assert_eq!(lu.update_work(), 0);
        assert!(!lu.should_refactor(), "fresh factor never needs refactor");

        let mut running = 0usize;
        for (slot, col) in [
            (1usize, vec![0.5, 6.0, 0.5, 0.0]),
            (2usize, vec![0.0, 0.5, 4.0, 0.5]),
        ] {
            lu.update(slot, &col).expect("update commits");
            let w = lu.last_update_work();
            assert!(w > 0, "a committed update must record positive work");
            running += w;
            assert_eq!(lu.update_work(), running, "cumulative = running sum");
        }

        // Rebuild from the *current* basis columns; refactor must zero the work.
        let a = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
        let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
        lu.refactor(&a, &sym).expect("refactor");
        assert_eq!(lu.last_update_work(), 0, "refactor resets per-update work");
        assert_eq!(lu.update_work(), 0, "refactor resets cumulative work");
    }

    /// The new counter must reflect the *fill-proportional build* cost that
    /// `eta_ops` (a solve-replay op count) cannot. External oracle (FT structure,
    /// not the counter's own bookkeeping): each eliminated sub-diagonal column `c`
    /// of the pivotal row contributes `nnz(U row c) >= 1` to the build work but
    /// exactly 1 to the eta op-count, so `last_update_work >= last_eta_ops` always
    /// — and on a dense bump (tridiagonal base ⇒ dense spike, the issue #87/#89
    /// regime) the U rows are long, so the gap is wide. A counter that merely
    /// re-counted eta ops could not satisfy the strict inequality.
    #[test]
    fn update_work_exceeds_eta_ops_on_dense_bump() {
        let m = 12;
        let mut cols: Vec<Vec<f64>> = vec![vec![0.0; m]; m];
        for (j, col) in cols.iter_mut().enumerate() {
            col[j] = 4.0;
            if j > 0 {
                col[j - 1] = -1.0;
            }
            if j + 1 < m {
                col[j + 1] = -1.0;
            }
        }
        let params = LuParams {
            max_updates: 1000,
            max_growth: 1e30,
            ..LuParams::default()
        };
        let mut lu = SparseLu::factor_dense_columns(m, &cols, params).expect("factor");

        let mut saw_wide_bump = false;
        for s in 0..15usize {
            let slot = s % 4;
            let mut col = vec![0.0; m];
            for (i, ci) in col.iter_mut().enumerate() {
                *ci = 0.5 + ((i * 5 + s * 3) % 7) as f64 * 0.25;
            }
            col[slot] = 40.0 + s as f64;
            if lu.update(slot, &col).is_err() {
                continue;
            }
            // General invariant: build work never undercounts the eta op-count.
            assert!(
                lu.last_update_work() >= lu.last_eta_ops(),
                "update {s}: build work {} must be >= eta ops {}",
                lu.last_update_work(),
                lu.last_eta_ops()
            );
            // On a genuine wide bump the gap is large — the density factor the old
            // `eta_ops` signal misses.
            if lu.last_eta_ops() >= 3 {
                saw_wide_bump = true;
                assert!(
                    lu.last_update_work() >= 2 * lu.last_eta_ops(),
                    "update {s}: dense bump build work {} should dwarf eta ops {}",
                    lu.last_update_work(),
                    lu.last_eta_ops()
                );
            }
        }
        assert!(
            saw_wide_bump,
            "test must exercise at least one wide (dense) bump"
        );
    }

    /// The diagonal-first row helpers must round-trip a value at the diagonal
    /// column and at an off-diagonal column whose index is *below* the diagonal's
    /// (the post-permutation case the binary-searched tail must handle).
    #[test]
    fn diagonal_first_row_helpers() {
        // Row for position 5: diagonal at column 5, off-diagonals at 2 and 8.
        let mut row = vec![(5usize, 3.0)];
        super::insert_offdiag(&mut row, 8, 7.0);
        super::insert_offdiag(&mut row, 2, -1.0);
        assert_eq!(super::get_col(&row, 5), Some(3.0)); // diagonal
        assert_eq!(super::get_col(&row, 2), Some(-1.0)); // below diagonal column
        assert_eq!(super::get_col(&row, 8), Some(7.0));
        assert_eq!(super::get_col(&row, 4), None);
        assert_eq!(row[0], (5, 3.0), "diagonal stays first");
        super::remove_offdiag(&mut row, 2);
        assert_eq!(super::get_col(&row, 2), None);
        assert_eq!(row[0], (5, 3.0), "diagonal still first after remove");
    }
}
