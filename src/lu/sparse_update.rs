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
//! FT never re-selects a pivot, and its fixed order can cancel the final bump
//! diagonal to exactly `0.0` on a **nonsingular** basis when an intermediate
//! grows past `|true pivot|/ε` (issue #112). Two defenses:
//!
//! - **Compensated accumulation (always on):** the working-row scatter is a
//!   Neumaier (Kahan–Babuška) compensated sum ([`two_sum_add`]), so the bits a
//!   plain sum would absorb survive to the final diagonal check. This is the
//!   fix for the issue #112 exact-`0.0` class — re-ordering pivots cannot be,
//!   because any interchange order's working row is exactly proportional to
//!   the fixed order's (`dev/research/issue-112-bg-update.md` §UPDATE).
//! - **Bartels–Golub interchanges (opt-in, [`super::LuParams::update_pivot_search`]):**
//!   whenever the working row's entry strictly dominates the retained
//!   diagonal, the two rows swap contents (recorded as an `FtOp::Swap` in the
//!   eta, so `uperm`, `L`, `P`, `Q`, and prior etas still never change),
//!   bounding every multiplier by 1 — the classic BG growth guarantee, which
//!   keeps the factor balanced across long update chains.
//!
//! The work is otherwise bump-local: the spike is computed by a Gilbert–Peierls
//! depth-first reach, and the "unchanged on failure" guarantee saves/restores only
//! the changed rows and the bump's `uperm` range (no `O(nnz)` clone of `U`).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::sparse_factor::{FtEta, FtOp, SparseLu};
use super::sparse_symbolic::SparseLuSymbolic;
use super::RefactorCause;
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
        for &(row, val) in entering.iter() {
            if row >= m {
                return Err(FeralError::InvalidInput(format!(
                    "entering-column row {} out of range for dimension {}",
                    row, m
                )));
            }
            // Reject non-finite entries up front, matching the factor path
            // (`sparse_matrix.rs`): a NaN passes the `v != 0.0` scan in
            // `update` and would otherwise be committed into `U` — where the
            // growth scan's `f64::max` ignores it and the solves guard only
            // the diagonal — yielding a silent NaN solution (issue #114).
            if !val.is_finite() {
                return Err(FeralError::InvalidInput(
                    "LU entering column contains non-finite entries".to_string(),
                ));
            }
        }
        if self.updates_since_refactor() + 1 > self.params.max_updates {
            self.last_refactor =
                Some((RefactorCause::UpdateBudget, self.params.max_updates as f64));
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
                self.last_refactor = Some((RefactorCause::Singular, 0.0));
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

        // Eliminate the single pivotal row `r` (now at rank `h_rank`). With
        // `update_pivot_search` off (the default) this is the plain fixed-order
        // Forrest–Tomlin sweep, byte-identical to the pre-#112 behavior; with
        // it on, the sweep performs Bartels–Golub row interchanges wherever the
        // working row strictly dominates the retained diagonal. A retry-only
        // rescue was tried and rejected: any interchange order's working row is
        // exactly proportional to the fixed order's, so a pivot the FT sweep
        // has already cancelled to zero is unrecoverable by re-ordering
        // (dev/tried-and-rejected.md 2026-07-10); robustness must be always-on
        // (bounded multipliers from the first update), not curative.
        let diag0 = w[r];
        let allow_swaps = self.params.update_pivot_search;
        let result = self.eliminate_pivot_row(
            r,
            h_rank,
            diag0,
            &mut work,
            allow_swaps,
            &mut saved,
            &changed,
        );

        clear(&mut w, &touched);
        self.ft_work = w;

        match result {
            Ok((ops, swapped)) => {
                // Rows the pivot search displaced were rewritten wholesale:
                // include them in the growth scan (duplicates with `changed`
                // are harmless for a max) and rebuild the column index below.
                changed.extend(swapped.iter().copied());
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
                    self.last_refactor = Some((RefactorCause::Growth, growth));
                    return Err(FeralError::NeedsRefactor);
                }
                if swapped.is_empty() {
                    // Commit: refresh the `u_above` column index *incrementally*.
                    // Only two things changed structurally (issue #89):
                    // `set_column_r` rewrote column `r` (its holders are now
                    // exactly the spike support), and `eliminate_pivot_row`
                    // rebuilt row `r`. Every other "changed" row only gained or
                    // lost its single column-`r` entry — already captured by
                    // column `r`'s holder list — so its membership in *other*
                    // columns is untouched and must NOT be re-indexed. The old
                    // code re-indexed every changed row wholesale, an
                    // `O(bump · rowlen · shift)` (≈ `O(m³)` on a dense bump)
                    // churn that dwarfed the elimination's `O(factor_nnz)`
                    // arithmetic.
                    //
                    // (a) Column `r`'s holders = spike support minus `r`. `supp`
                    //     is already sorted+deduped and `p != r` preserves order,
                    //     so the list stays sorted.
                    self.u_above[r].clear();
                    self.u_above[r].extend(supp.iter().copied().filter(|&p| p != r));
                    // (b) Row `r` changed its column set: drop `r` from its old
                    //     columns' holder lists and add it to the new ones. (Both
                    //     skip column `r` itself — that is the diagonal, not a
                    //     `u_above` entry — so this never touches the list
                    //     rebuilt in (a).) `saved` is a local moved out of
                    //     `self`, so borrowing the snapshot while mutating `self`
                    //     is sound (no clone needed).
                    let old_row_r: &[(usize, f64)] = saved
                        .iter()
                        .find(|(i, _)| *i == r)
                        .map(|(_, b)| b.as_slice())
                        .unwrap_or(&[]);
                    self.unindex_above(r, old_row_r);
                    let new_row_r = std::mem::take(&mut self.u_rows[r]);
                    self.index_above(r, &new_row_r);
                    self.u_rows[r] = new_row_r;
                } else {
                    // The pivot search rewrote the displaced rows' column sets
                    // wholesale (each took the working row's full support), so
                    // the incremental refresh above would leave stale holders.
                    // Rebuild the index from `U` — O(nnz(U)); never taken on
                    // the default (plain-FT) path.
                    self.rebuild_u_above();
                }
                for (_, buf) in saved.drain(..) {
                    self.saved_pool.push(buf);
                }
                self.saved_scratch = saved;
                self.etas.push(FtEta { ops });
                self.growth = growth;
                self.last_update_work = work;
                self.update_work_total += work;
                self.pivot_search_swaps += swapped.len();
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
                    FtOp::Swap { a, b } => {
                        w.swap(a, b);
                        if !mark[a] {
                            mark[a] = true;
                            touched.push(a);
                        }
                        if !mark[b] {
                            mark[b] = true;
                            touched.push(b);
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

    /// Elimination of the single pivotal row `r` (now at rank `h_rank`, the
    /// bottom of the bump after [`Self::shift_uperm`]). The working row's
    /// sub-diagonal entries — columns whose new rank is `< h_rank` — are cleared
    /// by a sparse forward sweep against the upper-triangular bump rows, in
    /// increasing rank order. `diag0 = w[r]` seeds its column-`r` value.
    ///
    /// With `allow_swaps == false` this is the plain **Forrest–Tomlin** sweep:
    /// the pivot at each step is the retained diagonal `U[c,c]`, only row `r`
    /// is rewritten, and each step contributes one `FtOp::Axpy{target: r, ..}`
    /// to the eta.
    ///
    /// With `allow_swaps == true` ([`super::LuParams::update_pivot_search`],
    /// issue #112) each step performs a **Bartels–Golub row interchange**
    /// whenever the working row's entry strictly dominates the retained
    /// diagonal (`|U[c,c]| < |rw[c]|`): the working row is installed as the
    /// new `u_rows[c]` (its column-`c` entry is the new rank-`k` diagonal,
    /// every other entry has rank `> k`, so triangularity under the
    /// *unchanged* `uperm` is preserved), the displaced row becomes the new
    /// working row, and the eta gains `FtOp::Swap{a: c, b: r}` + the
    /// elimination `Axpy`. Multipliers then satisfy `|mult| ≤ 1` — the classic
    /// BG growth bound. Each displaced row is snapshotted into `saved` for
    /// rollback (unless its position is already in the sorted pre-update
    /// snapshot list `presaved`); the rewritten positions are returned so the
    /// caller can extend its growth scan and rebuild the column index.
    ///
    /// In both modes the working-row scatter is **Neumaier-compensated**
    /// ([`two_sum_add`]): the fixed pivot order can drive an intermediate past
    /// `|true pivot|/ε` and a plain sum then cancels the final bump diagonal
    /// to exactly `0.0` on a nonsingular basis — the issue #112 failure. The
    /// compensation retains those bits, so the final diagonal check sees the
    /// true value.
    ///
    /// Returns [`FeralError::NeedsRefactor`] if the resulting diagonal pivot
    /// vanishes (singular replacement) or a retained pivot row is structurally
    /// broken.
    #[allow(clippy::too_many_arguments)]
    fn eliminate_pivot_row(
        &mut self,
        r: usize,
        h_rank: usize,
        diag0: f64,
        work: &mut usize,
        allow_swaps: bool,
        saved: &mut Vec<(usize, Vec<(usize, f64)>)>,
        presaved: &[usize],
    ) -> Result<(Vec<FtOp>, Vec<usize>), FeralError> {
        // Anchor the bump-pivot floor to the matrix magnitude `a_max` (as the
        // factor does), not `u_max0`: on high-growth bases `u_max0 ≫ a_max`
        // would reject healthy `O(a_max)` pivots and livelock the caller's
        // update→refactor→retry loop (issue #118).
        let ztol = self.params.zero_pivot_tol * self.a_max;
        let mut ops: Vec<FtOp> = Vec::new();
        let mut swapped: Vec<usize> = Vec::new();

        let mut rw = std::mem::take(&mut self.ft_rw); // dense scatter, zero on entry
        let mut comp = std::mem::take(&mut self.ft_rw_comp); // its compensation terms
        let mut rw_touched = std::mem::take(&mut self.targets_scratch);
        rw_touched.clear();
        let mut queued = std::mem::take(&mut self.scratch_mark); // all-false on entry
        let mut touch_mark = std::mem::take(&mut self.ft_touch_mark); // all-false on entry
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
                &mut comp,
                &mut rw_touched,
                &mut queued,
                &mut touch_mark,
                &mut heap,
                c,
                v,
                &self.uperm,
                h_rank,
            );
        }
        // Column r is the bump diagonal (rank h_rank): never sub-diagonal, so it is
        // not pushed to the heap; just record its starting value.
        if !touch_mark[r] {
            touch_mark[r] = true;
            rw_touched.push(r);
        }
        two_sum_add(&mut rw, &mut comp, r, diag0);
        self.row_pool.push(row_r); // recycle the old row's buffer

        // Sweep sub-diagonal columns of the working row in increasing rank order.
        while let Some(Reverse(rank)) = heap.pop() {
            let c = self.uperm_inv[rank];
            queued[c] = false;
            let vrc = rw[c] + comp[c];
            if vrc == 0.0 {
                continue;
            }
            let &(dc, piv) = match self.u_rows[c].first() {
                Some(p) => p,
                None => {
                    self.restore_elim_pools(rw, comp, rw_touched, queued, touch_mark);
                    // No diagonal in the pivot column ⇒ effectively a zero pivot.
                    self.last_refactor = Some((RefactorCause::TinyPivot, 0.0));
                    return Err(FeralError::NeedsRefactor);
                }
            };
            if dc != c || !piv.is_finite() {
                self.restore_elim_pools(rw, comp, rw_touched, queued, touch_mark);
                self.last_refactor = Some((RefactorCause::TinyPivot, piv.abs()));
                return Err(FeralError::NeedsRefactor);
            }
            if allow_swaps && piv.abs() < vrc.abs() {
                // Bartels–Golub interchange: the working row strictly dominates
                // the retained diagonal, so it becomes the pivot row for rank
                // `rank`. Gather it out of the scatter as the new `u_rows[c]`:
                // diagonal `(c, vrc)` first, then the column-sorted tail —
                // every other entry has rank > `rank` (ranks below were
                // eliminated exactly), so triangularity under `uperm` holds.
                // `rw_touched` is duplicate-free (`touch_mark`), so the gather
                // is exact.
                let mut new_c = self.row_pool.pop().unwrap_or_default();
                new_c.clear();
                new_c.push((c, vrc));
                for &cc in rw_touched.iter() {
                    let val = rw[cc] + comp[cc];
                    if cc != c && val != 0.0 {
                        new_c.push((cc, val));
                    }
                }
                new_c[1..].sort_unstable_by_key(|&(cc, _)| cc);
                *work += new_c.len();

                let mult = piv / vrc; // |mult| < 1 by the swap condition
                ops.push(FtOp::Swap { a: c, b: r });
                ops.push(FtOp::Axpy {
                    target: r,
                    src: c,
                    mult,
                });

                // New working row = displaced_row_c − mult · W, formed in place
                // in the scatter: scale W's support by −mult, clear column `c`
                // exactly (`piv − mult·vrc` is 0 mathematically; the FP residual
                // would re-enqueue `c` forever — the FT bring-up landmine), then
                // scatter the displaced row, skipping its consumed diagonal. A
                // scaled-to-zero entry (only possible when `piv == 0`) stays
                // queued and is skipped on pop.
                for &(cc, _) in new_c[1..].iter() {
                    rw[cc] *= -mult;
                    comp[cc] *= -mult;
                }
                rw[c] = 0.0;
                comp[c] = 0.0;
                let old_c = std::mem::replace(&mut self.u_rows[c], new_c);
                *work += old_c.len();
                for &(cc, v) in old_c.iter() {
                    if cc == c {
                        continue;
                    }
                    scatter_into(
                        &mut rw,
                        &mut comp,
                        &mut rw_touched,
                        &mut queued,
                        &mut touch_mark,
                        &mut heap,
                        cc,
                        v,
                        &self.uperm,
                        h_rank,
                    );
                }

                // Snapshot the displaced row for rollback unless this position
                // was already saved at update start (each rank pops once, so a
                // position is displaced at most once; a later duplicate would
                // shadow the true pre-update snapshot in `rollback`).
                if presaved.binary_search(&c).is_err() {
                    saved.push((c, old_c));
                } else {
                    self.saved_pool.push(old_c);
                }
                swapped.push(c);
                continue;
            }
            if piv == 0.0 {
                self.restore_elim_pools(rw, comp, rw_touched, queued, touch_mark);
                self.last_refactor = Some((RefactorCause::TinyPivot, 0.0));
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
            comp[c] = 0.0;
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
                    &mut comp,
                    &mut rw_touched,
                    &mut queued,
                    &mut touch_mark,
                    &mut heap,
                    cc,
                    -mult * v,
                    &self.uperm,
                    h_rank,
                );
            }
        }

        // Diagonal pivot check, then gather the rebuilt row r (diagonal first).
        // Reads collapse the compensated pair `(rw, comp)` with one rounding.
        let diag = rw[r] + comp[r];
        if diag.abs() <= ztol || !diag.is_finite() {
            self.restore_elim_pools(rw, comp, rw_touched, queued, touch_mark);
            self.last_refactor = Some((RefactorCause::TinyPivot, diag.abs()));
            return Err(FeralError::NeedsRefactor);
        }
        let mut new_row = self.row_pool.pop().unwrap_or_default();
        new_row.clear();
        new_row.push((r, diag));
        let mut offdiag: Vec<(usize, f64)> = rw_touched
            .iter()
            .map(|&c| (c, rw[c] + comp[c]))
            .filter(|&(c, v)| c != r && v != 0.0)
            .collect();
        offdiag.sort_unstable_by_key(|&(c, _)| c);
        new_row.extend_from_slice(&offdiag);
        self.u_rows[r] = new_row;

        // Clear the dense scatter and hand the pools back.
        self.restore_elim_pools(rw, comp, rw_touched, queued, touch_mark);
        Ok((ops, swapped))
    }

    /// Clear the dense scatter `rw` and the `queued`/`touch_mark` markers over
    /// their touched positions (so all reach all-zero / all-false for the next
    /// update, on every exit path — including a mid-sweep error where the heap
    /// still held columns) and return the row-elimination churn buffers to
    /// their `SparseLu` pools.
    fn restore_elim_pools(
        &mut self,
        mut rw: Vec<f64>,
        mut comp: Vec<f64>,
        mut rw_touched: Vec<usize>,
        mut queued: Vec<bool>,
        mut touch_mark: Vec<bool>,
    ) {
        for &c in rw_touched.iter() {
            rw[c] = 0.0;
            comp[c] = 0.0;
            queued[c] = false;
            touch_mark[c] = false;
        }
        rw_touched.clear();
        self.ft_rw = rw;
        self.ft_rw_comp = comp;
        self.targets_scratch = rw_touched;
        self.scratch_mark = queued;
        self.ft_touch_mark = touch_mark;
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

    /// Rebuild the `u_above` column index from `U` wholesale — used when a
    /// pivot-searching rescue rewrote displaced rows' column sets, where the
    /// incremental refresh would leave stale holders. Rows are walked in
    /// ascending position, so each holder list comes out sorted (the
    /// `unindex_above`/`index_above` binary-search invariant).
    fn rebuild_u_above(&mut self) {
        let mut above = std::mem::take(&mut self.u_above);
        for col in above.iter_mut() {
            col.clear();
        }
        for (i, row) in self.u_rows.iter().enumerate() {
            for &(c, _) in row.iter() {
                if c != i {
                    above[c].push(i);
                }
            }
        }
        self.u_above = above;
    }
}

fn clear(w: &mut [f64], touched: &[usize]) {
    for &k in touched.iter() {
        w[k] = 0.0;
    }
}

/// Neumaier (Kahan–Babuška) compensated add into the working-row scatter:
/// `(rw[c], comp[c]) += v`, with the rounding error of the add captured in
/// `comp[c]` via a branch-free two-sum. The compensated value is
/// `rw[c] + comp[c]`. This is what defeats the issue #112 cancellation: when
/// an intermediate sum grows past `|true value|/ε`, a plain `+=` rounds the
/// true value's bits away irrecoverably (the classic Kahan form fails here
/// too — its `y = v − c` pre-subtraction re-absorbs the compensation into the
/// next large addend), while the Neumaier form keeps them in `comp[c]` until
/// the final read.
#[inline]
fn two_sum_add(rw: &mut [f64], comp: &mut [f64], c: usize, v: f64) {
    let s = rw[c];
    let t = s + v;
    let z = t - s;
    comp[c] += (s - (t - z)) + (v - z);
    rw[c] = t;
}

/// Add `v` to the compensated dense scatter `(rw, comp)[c]` of the pivotal
/// row, recording `c` as touched on first contact (`touch_mark` keeps
/// `rw_touched` duplicate-free — the pivot-searching swap gathers the working
/// row straight from the scatter) and enqueuing it (by triangular rank) when
/// it is a not-yet-queued sub-diagonal column (`uperm[c] < h_rank`). `queued`
/// dedups the heap; a column is re-enqueueable only after it is popped (rank
/// order guarantees fill lands at strictly higher ranks, so no column is
/// processed twice).
#[allow(clippy::too_many_arguments)]
fn scatter_into(
    rw: &mut [f64],
    comp: &mut [f64],
    rw_touched: &mut Vec<usize>,
    queued: &mut [bool],
    touch_mark: &mut [bool],
    heap: &mut BinaryHeap<Reverse<usize>>,
    c: usize,
    v: f64,
    uperm: &[usize],
    h_rank: usize,
) {
    if !touch_mark[c] {
        touch_mark[c] = true;
        rw_touched.push(c);
    }
    two_sum_add(rw, comp, c, v);
    if uperm[c] < h_rank && !queued[c] && rw[c] + comp[c] != 0.0 {
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

    use crate::error::FeralError;
    use crate::lu::RefactorCause;

    /// Issue #95: `last_refactor()` is `None` on a fresh factor and each
    /// `NeedsRefactor` return records the cause + magnitude. Update-count trip
    /// (`max_updates = 1`): the second update fails as `UpdateBudget`, magnitude =
    /// the cap that was hit; a successful update leaves the accessor `None`.
    #[test]
    fn last_refactor_reports_update_budget() {
        let cols = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let params = LuParams {
            max_updates: 1,
            ..LuParams::default()
        };
        let mut lu = SparseLu::factor_dense_columns(2, &cols, params).expect("factor");
        assert_eq!(lu.last_refactor(), None, "fresh factor: no cause");

        lu.update(0, &[1.0, 1.0])
            .expect("first update within budget");
        assert_eq!(lu.last_refactor(), None, "successful update leaves it None");

        let err = lu.update(0, &[1.0, 2.0]);
        assert!(matches!(err, Err(FeralError::NeedsRefactor)));
        assert_eq!(
            lu.last_refactor(),
            Some((RefactorCause::UpdateBudget, 1.0)),
            "second update trips the count cap (max_updates = 1)"
        );
    }

    /// Issue #95: replacing a slot with the **zero** column gives an empty spike
    /// support (nothing at or below its own diagonal in rank order), which the
    /// sparse path detects as `Singular` (magnitude `0.0`) before any elimination.
    #[test]
    fn last_refactor_reports_singular_on_zero_column() {
        let cols = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut lu = SparseLu::factor_dense_columns(2, &cols, LuParams::default()).expect("factor");
        let err = lu.update(0, &[0.0, 0.0]); // zero entering column ⇒ dependent
        assert!(matches!(err, Err(FeralError::NeedsRefactor)));
        assert_eq!(lu.last_refactor(), Some((RefactorCause::Singular, 0.0)));
    }

    /// Issue #95: a growth trip records `Growth` with the ratio that exceeded the
    /// cap. Same compounding last-slot updates as the growth-monitor test but with
    /// a small `max_growth` so an early update trips.
    #[test]
    fn last_refactor_reports_growth() {
        let cols = vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0, 5.0],
        ];
        let params = LuParams {
            max_updates: 20,
            max_growth: 5.0,
            ..LuParams::default()
        };
        let mut lu = SparseLu::factor_dense_columns(4, &cols, params).expect("factor");
        let updates = [
            vec![0.0, 0.0, 1.0, 20.0],
            vec![0.0, 0.0, 1.0, 60.0],
            vec![0.0, 0.0, 1.0, 180.0],
        ];
        let mut tripped = None;
        for col in updates.iter() {
            if let Err(FeralError::NeedsRefactor) = lu.update(3, col) {
                tripped = lu.last_refactor();
                break;
            }
        }
        let (cause, mag) = tripped.expect("some update trips the growth cap");
        assert_eq!(cause, RefactorCause::Growth);
        assert!(mag > 5.0, "growth magnitude {mag} exceeds max_growth = 5.0");
    }

    /// Issue #95: growth-aware recommendation. `should_refactor_growth()` fires
    /// once the growth high-water reaches `sqrt(max_growth)`, pre-empting the trip.
    /// With `max_growth = 100`, the recommendation must fire while updates are
    /// still committing (growth in `[10, 100)`), before any `NeedsRefactor`.
    #[test]
    fn should_refactor_growth_preempts_trip() {
        let cols = vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0],
            vec![0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0, 5.0],
        ];
        let params = LuParams {
            max_updates: 20,
            max_growth: 100.0, // sqrt = 10: recommend once growth >= 10
            ..LuParams::default()
        };
        let mut lu = SparseLu::factor_dense_columns(4, &cols, params).expect("factor");
        assert!(!lu.should_refactor_growth(), "fresh: growth == 1");

        // First commit raises max|U| to ~20 (u_max0 ≈ 5) ⇒ growth ≈ 3.6 < 10.
        lu.update(3, &[0.0, 0.0, 1.0, 20.0]).expect("commit 1");
        // Second commit raises it to ~60 ⇒ growth ≈ 11 ≥ 10: the pre-emptive
        // recommendation fires while the update itself still committed cleanly.
        lu.update(3, &[0.0, 0.0, 1.0, 60.0]).expect("commit 2");
        assert!(
            lu.should_refactor_growth(),
            "growth {} should reach sqrt(max_growth) = 10 and recommend refactor",
            lu.growth()
        );
        assert!(
            lu.last_refactor().is_none(),
            "no trip yet, only a recommendation"
        );
    }
}
