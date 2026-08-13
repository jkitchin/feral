//! Sparse-in / sparse-out `ftran` / `btran` (issue #161B, second half).
//!
//! [`SparseLu::ftran`] and [`SparseLu::btran`] take a dense `&mut [f64]`. That
//! alone forces `Omega(m)` per solve — the right-hand side must be read and the
//! solution written — no matter how little of the factor the answer depends on.
//! Making the triangular sweeps reach-limited took a solve from `O(nnz(factor))`
//! down to `O(m + reach work)`, and then stopped there: a phase probe on a
//! 3.4-position reach at `m = 4000` still measured 22 us, spread evenly across
//! ~6 linear passes with no single removable term
//! (`dev/research/hyper-sparse-solves-2026-08-13.md`).
//!
//! The floor is the signature, so this module changes the signature.
//! [`SparseLu::ftran_sparse`] and [`SparseLu::btran_sparse`] take the
//! right-hand side as `(index, value)` pairs and append the solution's nonzeros
//! to a caller-owned buffer. Nothing in the path is proportional to `m`:
//!
//! | step | cost |
//! |---|---|
//! | scatter the rhs (with row scaling) into pivot-position space | `O(nnz(rhs))` |
//! | `L`-solve over the Gilbert–Peierls reach | `O(reach work)` |
//! | replay the Forrest–Tomlin etas | `O(eta ops)` |
//! | `U`-solve over the reach | `O(reach work)` |
//! | gather the solution (with column scaling) | `O(nnz(x) log nnz(x))` |
//!
//! The eta term is proportional to the update chain, not to `m`, and is what
//! `compute_spike` in the Forrest–Tomlin update already pays.
//!
//! ## The zeroed-accumulator contract
//!
//! [`HyperWork::w`] is a dense length-`m` accumulator that is **all zero
//! between calls**, and every position a solve writes is recorded in
//! `pattern` so it can be zeroed again in `O(touched)` rather than `O(m)`. A
//! solve that failed to restore that invariant would not fail loudly — it would
//! silently seed the *next* solve with stale values. So the reset runs on every
//! return path, including the error paths out of the `U` solves, and
//! `tests/lu_sparse_rhs.rs` interleaves failing solves with succeeding ones to
//! prove it.
//!
//! This is the same convention as `ft_work`/`scratch_mark` in
//! [`super::sparse_update`], which is also where the reach idiom (mark array,
//! explicit stack, `sort_unstable`, clear over the touched list) comes from.
//! The two are deliberately parallel but not shared: `compute_spike` fuses its
//! scatter with the update's work accounting and the leaving column's scale
//! factor. `sparse_spike_matches_dense_ftran_partial` in the test file pins
//! this module's `L`-solve-plus-etas against the dense `ftran_partial` so the
//! two cannot drift apart unnoticed.

use super::sparse_factor::{FtOp, SparseLu};
use crate::error::FeralError;

/// Workspace for the sparse-in / sparse-out solves.
///
/// Allocated lazily on the first sparse solve — a caller that only ever uses
/// the dense entry points never pays for it.
#[derive(Debug, Clone, Default)]
pub(super) struct HyperWork {
    /// Dense accumulator, **all zero between calls**. See the module docs.
    w: Vec<f64>,
    /// Membership marker for `pattern`, **all false between calls**.
    mark: Vec<bool>,
    /// Every position touched so far this solve. A superset of the numerical
    /// nonzero pattern (an entry can cancel to exactly zero after being
    /// recorded), which is what makes it safe to use as the reset list.
    pattern: Vec<usize>,
    /// Depth-first stack, reused across the three reaches of one solve.
    stack: Vec<usize>,
    /// `pattern` sorted into the topological order of the sweep about to run.
    order: Vec<usize>,
    /// Scalar work of the last solve: positions swept plus factor entries
    /// traversed. The scalability witness — see
    /// [`SparseLu::last_sparse_solve_work`].
    work: usize,
}

impl HyperWork {
    fn is_sized(&self, m: usize) -> bool {
        self.w.len() == m && self.mark.len() == m
    }

    fn new(m: usize) -> Self {
        HyperWork {
            w: vec![0.0; m],
            mark: vec![false; m],
            pattern: Vec::new(),
            stack: Vec::new(),
            order: Vec::new(),
            work: 0,
        }
    }

    /// Record `k` as touched. Returns `true` the first time, so a depth-first
    /// walk can use it as its "discovered" test.
    #[inline]
    fn touch(&mut self, k: usize) -> bool {
        if self.mark[k] {
            return false;
        }
        self.mark[k] = true;
        self.pattern.push(k);
        true
    }

    /// Restore the all-zero / all-false invariant in `O(touched)`. Must run on
    /// every exit path — a missed reset silently corrupts the *next* solve
    /// rather than this one.
    fn reset(&mut self) {
        for &k in self.pattern.iter() {
            self.w[k] = 0.0;
            self.mark[k] = false;
        }
        self.pattern.clear();
        self.stack.clear();
        self.order.clear();
    }

    /// Seed the depth-first stack with everything touched so far, then load
    /// `order` with the same set for the caller to sort.
    fn begin_reach(&mut self) {
        self.stack.clear();
        self.stack.extend(self.pattern.iter().copied());
    }

    fn finish_reach(&mut self) {
        self.order.clear();
        self.order.extend_from_slice(&self.pattern);
        self.work += self.order.len();
    }
}

impl SparseLu {
    /// Solve `B x = a` with both sides sparse: `rhs` is `(row, value)` pairs,
    /// and the nonzeros of `x` are written to `out` as `(row, value)` pairs
    /// sorted by index (issue #161B).
    ///
    /// This is the work-proportional entry point. Unlike [`SparseLu::ftran`] it
    /// has **no term proportional to `m`** — the cost is the reach of `rhs`'s
    /// pattern through the factor plus the Forrest–Tomlin eta chain. On a
    /// simplex `ftran` against a near-unit-vector column, that is the difference
    /// between "touch the whole basis" and "touch the handful of positions the
    /// answer depends on".
    ///
    /// `out` is cleared first and is only written on success. Repeated indices
    /// in `rhs` accumulate. Explicit zeros in `rhs`, and entries of `x` that
    /// cancel to exactly zero, are not emitted.
    ///
    /// Errors with [`FeralError::DimensionMismatch`] on an out-of-range index
    /// and [`FeralError::SingularBasis`] on a degenerate `U` diagonal that the
    /// solution depends on — the same narrowed guard the reach-limited dense
    /// route uses, documented in `dev/decisions.md` (2026-08-13).
    ///
    /// When the solution is *dense* this is slower than [`SparseLu::ftran`]:
    /// it sorts a length-`m` reach where the dense sweep just walks the factor
    /// in order. Route on what the caller knows about its right-hand sides.
    pub fn ftran_sparse(
        &mut self,
        rhs: &[(usize, f64)],
        out: &mut Vec<(usize, f64)>,
    ) -> Result<(), FeralError> {
        self.prepare_sparse_solve()?;
        let mut hw = std::mem::take(&mut self.hyper);
        hw.work = 0;
        let res = self.ftran_sparse_inner(rhs, &mut hw, out);
        hw.reset();
        self.hyper = hw;
        if res.is_err() {
            out.clear();
        }
        res
    }

    /// Solve `Bᵀ x = a` with both sides sparse. The transpose twin of
    /// [`SparseLu::ftran_sparse`]; `rhs` is indexed by *column* of `B` and the
    /// result by *row*, matching [`SparseLu::btran`].
    pub fn btran_sparse(
        &mut self,
        rhs: &[(usize, f64)],
        out: &mut Vec<(usize, f64)>,
    ) -> Result<(), FeralError> {
        self.prepare_sparse_solve()?;
        let mut hw = std::mem::take(&mut self.hyper);
        hw.work = 0;
        let res = self.btran_sparse_inner(rhs, &mut hw, out);
        hw.reset();
        self.hyper = hw;
        if res.is_err() {
            out.clear();
        }
        res
    }

    /// Scalar work of the most recent sparse solve: positions swept plus factor
    /// entries traversed.
    ///
    /// This exists to be *asserted on*. "Work-proportional" is a claim about
    /// asymptotics, and a wall-clock benchmark cannot pin it — a regression
    /// that reintroduced an `O(m)` term would show up as a constant-factor
    /// slowdown, indistinguishable from noise on a small `m`. Holding the local
    /// structure fixed and growing `m`, this counter must stay flat;
    /// `tests/lu_sparse_rhs.rs` asserts exactly that across an 8x range. It is
    /// the same kind of witness as [`Self::reach_visits`] for the factor.
    pub fn last_sparse_solve_work(&self) -> usize {
        self.hyper.work
    }

    /// Size the sparse-solve workspace and make sure the `L` row index the
    /// `Lᵀ` reach walks exists. Both are lazy: the sparse entry points work
    /// regardless of [`LuParams::hyper_sparse_max_density`], which governs only
    /// whether the *dense* entry points take their reach-limited route.
    fn prepare_sparse_solve(&mut self) -> Result<(), FeralError> {
        let m = self.m;
        if !self.hyper.is_sized(m) {
            self.hyper = HyperWork::new(m);
        }
        self.ensure_l_row_index();
        Ok(())
    }

    fn ftran_sparse_inner(
        &self,
        rhs: &[(usize, f64)],
        hw: &mut HyperWork,
        out: &mut Vec<(usize, f64)>,
    ) -> Result<(), FeralError> {
        out.clear();
        // Scatter into pivot-position space, applying the row scaling on the
        // way. `ftran` computes `bt[i] = d_row[i]·rhs[rperm[i]]` and then
        // `out[k] = bt[perm[k]]`; composed, original row `o` lands at position
        // `perm_inv[scale_rperm_inv[o]]`. Identical to the mapping
        // `compute_spike` uses to seed an entering column.
        for &(o, v) in rhs.iter() {
            if o >= self.m {
                return Err(FeralError::DimensionMismatch {
                    expected: self.m,
                    got: o + 1,
                });
            }
            if v == 0.0 {
                continue;
            }
            let i = self.scale_rperm_inv[o];
            let k = self.perm_inv[i];
            hw.touch(k);
            hw.w[k] += self.scale.d_row[i] * v;
        }

        self.l_solve_sparse(hw);
        self.etas_forward_sparse(hw);
        self.u_solve_sparse(hw)?;

        // Gather, applying the column scaling: position `k` is original column
        // `qcol[k]`, and `ftran` finishes with `x[j] = d_col[j]·bt[j]`.
        for &k in hw.pattern.iter() {
            let v = hw.w[k];
            if v == 0.0 {
                continue;
            }
            let j = self.qcol[k];
            out.push((j, self.scale.d_col[j] * v));
        }
        out.sort_unstable_by_key(|&(j, _)| j);
        Ok(())
    }

    fn btran_sparse_inner(
        &self,
        rhs: &[(usize, f64)],
        hw: &mut HyperWork,
        out: &mut Vec<(usize, f64)>,
    ) -> Result<(), FeralError> {
        out.clear();
        // `btran` computes `bt[j] = d_col[j]·rhs[j]` (no permutation) and then
        // `s[k] = bt[qcol[k]]`, so original column `j` lands at position
        // `qcol_inv[j]`.
        for &(j, v) in rhs.iter() {
            if j >= self.m {
                return Err(FeralError::DimensionMismatch {
                    expected: self.m,
                    got: j + 1,
                });
            }
            if v == 0.0 {
                continue;
            }
            let k = self.qcol_inv[j];
            hw.touch(k);
            hw.w[k] += self.scale.d_col[j] * v;
        }

        self.ut_solve_sparse(hw)?;
        self.etas_transpose_sparse(hw);
        self.lt_solve_sparse(hw);

        // Position `k` is scaled row `perm[k]`, which is original row
        // `rperm[perm[k]]`; `btran` finishes with `x[rperm[i]] = d_row[i]·bt[i]`.
        for &k in hw.pattern.iter() {
            let v = hw.w[k];
            if v == 0.0 {
                continue;
            }
            let i = self.perm[k];
            out.push((self.scale.rperm[i], self.scale.d_row[i] * v));
        }
        out.sort_unstable_by_key(|&(i, _)| i);
        Ok(())
    }

    /// Sparse forward solve `L y = w` over the Gilbert–Peierls reach of the
    /// current pattern. `L` is strictly lower triangular in fixed pivot-position
    /// coordinates, so ascending position is a valid topological order.
    fn l_solve_sparse(&self, hw: &mut HyperWork) {
        hw.begin_reach();
        while let Some(k) = hw.stack.pop() {
            for idx in self.l_col_ptr[k]..self.l_col_ptr[k + 1] {
                let i = self.l_row_idx[idx];
                if hw.touch(i) {
                    hw.stack.push(i);
                }
            }
        }
        hw.finish_reach();
        hw.order.sort_unstable();
        let HyperWork { w, order, work, .. } = hw;
        for &k in order.iter() {
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
    }

    /// Sparse back solve `Lᵀ v = w` over the reach in `Lᵀ`'s predecessor graph
    /// (row `i` feeds every column it appears in). Descending position is the
    /// topological order.
    fn lt_solve_sparse(&self, hw: &mut HyperWork) {
        hw.begin_reach();
        while let Some(i) = hw.stack.pop() {
            for idx in self.l_row_ptr[i]..self.l_row_ptr[i + 1] {
                let k = self.l_row_cols[idx];
                if hw.touch(k) {
                    hw.stack.push(k);
                }
            }
        }
        hw.finish_reach();
        hw.order.sort_unstable_by_key(|&k| std::cmp::Reverse(k));
        let HyperWork { w, order, work, .. } = hw;
        for &k in order.iter() {
            let (lo, hi) = (self.l_col_ptr[k], self.l_col_ptr[k + 1]);
            *work += hi - lo;
            let mut acc = w[k];
            for idx in lo..hi {
                acc -= self.l_val[idx] * w[self.l_row_idx[idx]];
            }
            w[k] = acc;
        }
    }

    /// Sparse back solve `U x = w` over the reach in `U`'s predecessor graph
    /// (`u_above[c]` holds the rows with an entry in column `c`). Decreasing
    /// triangular *rank* — not position index — is the topological order, since
    /// `U` is upper triangular in `uperm` order once an update has permuted it.
    fn u_solve_sparse(&self, hw: &mut HyperWork) -> Result<(), FeralError> {
        hw.begin_reach();
        while let Some(c) = hw.stack.pop() {
            for &k in self.u_above[c].iter() {
                if hw.touch(k) {
                    hw.stack.push(k);
                }
            }
        }
        hw.finish_reach();
        let uperm = &self.uperm;
        hw.order.sort_unstable_by(|&a, &b| uperm[b].cmp(&uperm[a]));
        let HyperWork { w, order, work, .. } = hw;
        for &k in order.iter() {
            let row = &self.u_rows[k];
            let &(dc, d) = row.first().ok_or(FeralError::SingularBasis { column: k })?;
            if dc != k || d == 0.0 || !d.is_finite() {
                return Err(FeralError::SingularBasis { column: k });
            }
            *work += row.len();
            let mut acc = w[k];
            for &(c, v) in row[1..].iter() {
                acc -= v * w[c];
            }
            w[k] = acc / d;
        }
        Ok(())
    }

    /// Sparse forward solve `Uᵀ z = w` over the reach in `Uᵀ`'s predecessor
    /// graph (row `i` feeds every column it holds). Increasing triangular rank.
    fn ut_solve_sparse(&self, hw: &mut HyperWork) -> Result<(), FeralError> {
        hw.begin_reach();
        while let Some(i) = hw.stack.pop() {
            for &(c, _) in self.u_rows[i].iter().skip(1) {
                if hw.touch(c) {
                    hw.stack.push(c);
                }
            }
        }
        hw.finish_reach();
        let uperm = &self.uperm;
        hw.order.sort_unstable_by(|&a, &b| uperm[a].cmp(&uperm[b]));
        let HyperWork { w, order, work, .. } = hw;
        for &i in order.iter() {
            let row = &self.u_rows[i];
            let &(dc, d) = row.first().ok_or(FeralError::SingularBasis { column: i })?;
            if dc != i || d == 0.0 || !d.is_finite() {
                return Err(FeralError::SingularBasis { column: i });
            }
            let si = w[i] / d;
            w[i] = si;
            if si == 0.0 {
                continue;
            }
            *work += row.len();
            for &(c, v) in row[1..].iter() {
                w[c] -= v * si;
            }
        }
        Ok(())
    }

    /// Replay the Forrest–Tomlin etas forward, extending the pattern with
    /// whatever they make nonzero.
    ///
    /// Every op is walked — the `w[src] == 0.0` shortcut is deliberately *not*
    /// taken. It would change nothing asymptotically (this is `O(eta ops)`
    /// either way) but it would diverge from [`FtEta::apply_forward`] on a
    /// non-finite multiplier, where `mult * 0.0` is `NaN` rather than `0.0`.
    /// Bit-for-bit agreement with the dense path is worth more here than a
    /// branch that saves nothing.
    fn etas_forward_sparse(&self, hw: &mut HyperWork) {
        for eta in self.etas.iter() {
            hw.work += eta.ops.len();
            for op in eta.ops.iter() {
                match *op {
                    FtOp::Axpy { target, src, mult } => {
                        hw.w[target] -= mult * hw.w[src];
                        if hw.w[target] != 0.0 {
                            hw.touch(target);
                        }
                    }
                    FtOp::Swap { a, b } => {
                        hw.w.swap(a, b);
                        hw.touch(a);
                        hw.touch(b);
                    }
                }
            }
        }
    }

    /// Replay the etas transposed, in reverse — the mirror of
    /// [`SparseLu::etas_forward_sparse`], matching [`FtEta::apply_transpose`].
    fn etas_transpose_sparse(&self, hw: &mut HyperWork) {
        for eta in self.etas.iter().rev() {
            hw.work += eta.ops.len();
            for op in eta.ops.iter().rev() {
                match *op {
                    FtOp::Axpy { target, src, mult } => {
                        hw.w[src] -= mult * hw.w[target];
                        if hw.w[src] != 0.0 {
                            hw.touch(src);
                        }
                    }
                    FtOp::Swap { a, b } => {
                        hw.w.swap(a, b);
                        hw.touch(a);
                        hw.touch(b);
                    }
                }
            }
        }
    }
}
