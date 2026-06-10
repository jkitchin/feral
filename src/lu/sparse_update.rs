//! Sparse rank-1 column-replacement update — Forrest–Tomlin / Bartels–Golub–Reid.
//!
//! Replacing basis slot `leaving_slot` (column position `r`) by a new column
//! folds the spike `ρ = G⁻¹ L⁻¹ P aₙₑw` into `U`'s column `r`, then
//! re-triangularizes the resulting *bump* by sparse Gaussian elimination with
//! partial pivoting (partial pivoting is what makes this correct on a sparse
//! `U`: a zero diagonal pivot is replaced by a nonzero sub-diagonal spike entry
//! via a row interchange; the swap goes into the eta so the base `L` is never
//! permuted).
//!
//! The work is **bump-local**: the spike is computed by a Gilbert–Peierls
//! depth-first reach (only the reachable `L`-columns are touched, not all `n`),
//! column `r` is located via the `u_above` index (no full-row scan), and the
//! "unchanged on failure" guarantee is provided by saving/restoring only the
//! changed rows (no `O(nnz)` clone of `U`). Apart from one `O(n)` read of the
//! dense entering column, the cost is `O(bump)`.

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
    /// update or growth budget is exceeded, and [`FeralError::SingularBasis`]
    /// when the bump has no acceptable pivot (the new basis is singular).
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
    /// Returns [`FeralError::NeedsRefactor`] / [`FeralError::SingularBasis`] as
    /// for [`SparseLu::update`].
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

        // --- Sparse spike ρ = G⁻¹ L⁻¹ P (scaled aₙₑw) via Gilbert–Peierls reach ---
        let mut w = std::mem::take(&mut self.ft_work); // dedicated buffer, zero on entry
        let mut touched: Vec<usize> = Vec::new(); // positions made nonzero in w
        self.compute_spike(entering, leaving_slot, &mut w, &mut touched);

        let r = self.qcol_inv[leaving_slot];
        let mut supp: Vec<usize> = touched.iter().copied().filter(|&k| w[k] != 0.0).collect();
        supp.sort_unstable();
        supp.dedup();
        let h = match supp.last().copied() {
            Some(h) if h >= r => h,
            _ => {
                clear(&mut w, &touched);
                self.ft_work = w;
                return Err(FeralError::SingularBasis { column: r });
            }
        };

        // --- Rows whose U content will change (all <= h): bump + spike support
        // + old column-r entries (above-diagonal rows from u_above, plus r). ---
        let mut changed: Vec<usize> = (r..=h).collect();
        changed.extend(supp.iter().copied());
        changed.extend(self.u_above[r].iter().copied());
        changed.push(r);
        changed.sort_unstable();
        changed.dedup();

        // Save the changed rows so a mid-elimination failure can roll back.
        let saved: Vec<(usize, Vec<(usize, f64)>)> = changed
            .iter()
            .map(|&i| (i, self.u_rows[i].clone()))
            .collect();

        // Overwrite column r of U with the spike ρ (= w).
        self.set_column_r(r, &w, &supp);

        // Re-triangularize the bump [r, h] with partial pivoting.
        let result = self.eliminate_bump(r, h);

        clear(&mut w, &touched);
        self.ft_work = w;

        match result {
            Ok(ops) => {
                // L5 (dev/research/repo-review-2026-06-09.md): monitor element
                // growth, not the largest single multiplier. Every U entry that
                // changed this update lives in a `changed` row, so the high-water
                // `max|U|/u_max0` is maintained by scanning only those rows —
                // O(changed), keeping the update cheap — and it compounds across
                // a chain of updates (a per-multiplier max did not).
                let mut changed_max = 0.0_f64;
                for &i in changed.iter() {
                    for &(_, v) in self.u_rows[i].iter() {
                        changed_max = changed_max.max(v.abs());
                    }
                }
                let growth = self.growth.max(changed_max / self.u_max0);
                if growth > self.params.max_growth {
                    // Over budget: roll back exactly like the elimination-error
                    // path (u_above was not yet modified) and force a refactor.
                    for (i, row) in saved {
                        self.u_rows[i] = row;
                    }
                    return Err(FeralError::NeedsRefactor);
                }
                // Commit: refresh the u_above index for every changed row.
                for (i, old_row) in saved.iter() {
                    self.unindex_above(*i, old_row);
                    let new_row = std::mem::take(&mut self.u_rows[*i]);
                    self.index_above(*i, &new_row);
                    self.u_rows[*i] = new_row;
                }
                self.etas.push(FtEta { ops });
                self.growth = growth;
                Ok(())
            }
            Err(e) => {
                // Roll back the changed rows (u_above was not yet modified).
                for (i, row) in saved {
                    self.u_rows[i] = row;
                }
                Err(e)
            }
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
            for idx in lo..hi {
                w[self.l_row_idx[idx]] -= self.l_val[idx] * yk;
            }
        }

        // Replay the FT etas forward (G⁻¹), tracking newly touched positions.
        for eta in self.etas.iter() {
            for op in eta.ops.iter() {
                match *op {
                    FtOp::Swap(a, b) => {
                        w.swap(a, b);
                        for x in [a, b] {
                            if w[x] != 0.0 && !mark[x] {
                                mark[x] = true;
                                touched.push(x);
                            }
                        }
                    }
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

    /// Overwrite column `r` of `U` with the spike (`w` at positions `supp`),
    /// removing any old column-`r` entries. Old above-diagonal rows are located
    /// via `u_above[r]` (read only — `u_above` is refreshed wholesale in the
    /// commit phase via unindex/reindex of the changed rows).
    fn set_column_r(&mut self, r: usize, w: &[f64], supp: &[usize]) {
        let old_above = self.u_above[r].clone();
        for &i in old_above.iter() {
            remove_col(&mut self.u_rows[i], r);
        }
        remove_col(&mut self.u_rows[r], r); // old diagonal
        for &i in supp.iter() {
            insert_or_set(&mut self.u_rows[i], r, w[i]);
        }
    }

    /// Eliminate the bump `[r, h]` of `U` with partial pivoting, returning the
    /// recorded eta ops and the updated growth monitor. Operates in place on
    /// `self.u_rows`; the caller is responsible for rollback on `Err`.
    fn eliminate_bump(&mut self, r: usize, h: usize) -> Result<Vec<FtOp>, FeralError> {
        // L6 (dev/research/repo-review-2026-06-09.md): the bump-pivot zero test
        // is relative to the basis magnitude, not an absolute 1e-13. `u_max0`
        // (max|U| at the last factor, floored away from zero for L5) is the scale
        // reference, consistent with the dense update and the factor paths.
        let ztol = self.params.zero_pivot_tol * self.u_max0;
        let mut ops: Vec<FtOp> = Vec::new();

        for k in r..=h {
            // Partial pivot among rows [k, h] with a column-k entry.
            let mut pivot_row = k;
            let mut pivot_abs = 0.0_f64;
            for i in k..=h {
                if let Some(v) = get_col(&self.u_rows[i], k) {
                    if v.abs() > pivot_abs {
                        pivot_abs = v.abs();
                        pivot_row = i;
                    }
                }
            }
            if pivot_abs <= ztol {
                return Err(FeralError::SingularBasis { column: k });
            }
            if pivot_row != k {
                self.u_rows.swap(k, pivot_row);
                ops.push(FtOp::Swap(k, pivot_row));
            }
            let pivot_data = self.u_rows[k].clone();
            let pivot = get_col(&pivot_data, k).unwrap_or(0.0);

            for i in k + 1..=h {
                if let Some(vik) = get_col(&self.u_rows[i], k) {
                    let mult = vik / pivot;
                    self.u_rows[i] = row_sub(&self.u_rows[i], &pivot_data, mult, k);
                    ops.push(FtOp::Axpy {
                        target: i,
                        src: k,
                        mult,
                    });
                }
            }
        }
        Ok(ops)
    }

    /// Remove row `i`'s strict-upper entries from the `u_above` column index
    /// (using its pre-update content `old_row`).
    fn unindex_above(&mut self, i: usize, old_row: &[(usize, f64)]) {
        for &(c, _) in old_row.iter() {
            if c > i {
                if let Ok(pos) = self.u_above[c].binary_search(&i) {
                    self.u_above[c].remove(pos);
                }
            }
        }
    }

    /// Add row `i`'s strict-upper entries to the `u_above` column index.
    fn index_above(&mut self, i: usize, new_row: &[(usize, f64)]) {
        for &(c, _) in new_row.iter() {
            if c > i {
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

/// Look up the value at column `c` in a column-sorted sparse row.
fn get_col(row: &[(usize, f64)], c: usize) -> Option<f64> {
    row.binary_search_by_key(&c, |&(col, _)| col)
        .ok()
        .map(|pos| row[pos].1)
}

/// Remove column `c` from a column-sorted sparse row, if present.
fn remove_col(row: &mut Vec<(usize, f64)>, c: usize) {
    if let Ok(pos) = row.binary_search_by_key(&c, |&(col, _)| col) {
        row.remove(pos);
    }
}

/// Set column `c` of a column-sorted sparse row to `v` (insert/replace; remove
/// if `v == 0`).
fn insert_or_set(row: &mut Vec<(usize, f64)>, c: usize, v: f64) {
    match row.binary_search_by_key(&c, |&(col, _)| col) {
        Ok(pos) => {
            if v != 0.0 {
                row[pos].1 = v;
            } else {
                row.remove(pos);
            }
        }
        Err(pos) => {
            if v != 0.0 {
                row.insert(pos, (c, v));
            }
        }
    }
}

/// `dst − mult·src` over two column-sorted sparse rows, dropping the eliminated
/// column `drop_col` and any exact zeros. Result stays column-sorted.
fn row_sub(
    dst: &[(usize, f64)],
    src: &[(usize, f64)],
    mult: f64,
    drop_col: usize,
) -> Vec<(usize, f64)> {
    let mut out = Vec::with_capacity(dst.len() + src.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < dst.len() && j < src.len() {
        let (dc, dv) = dst[i];
        let (sc, sv) = src[j];
        if dc < sc {
            if dc != drop_col {
                out.push((dc, dv));
            }
            i += 1;
        } else if dc > sc {
            let v = -mult * sv;
            if sc != drop_col && v != 0.0 {
                out.push((sc, v));
            }
            j += 1;
        } else {
            let v = dv - mult * sv;
            if dc != drop_col && v != 0.0 {
                out.push((dc, v));
            }
            i += 1;
            j += 1;
        }
    }
    while i < dst.len() {
        let (dc, dv) = dst[i];
        if dc != drop_col {
            out.push((dc, dv));
        }
        i += 1;
    }
    while j < src.len() {
        let (sc, sv) = src[j];
        let v = -mult * sv;
        if sc != drop_col && v != 0.0 {
            out.push((sc, v));
        }
        j += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::lu::sparse_factor::SparseLu;
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
}
