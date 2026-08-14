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
//! ...*while the answer stays sparse*. Reach-building and sorting are only
//! worth it below a solution density of about 10%, measured under a real dual
//! simplex, so past `LuParams::sparse_rhs_max_density · m` the reach is
//! abandoned mid-walk and the sweep runs over the whole basis in natural order
//! — the dense entry points' work, reached through the sparse signature (issue
//! #164). The kernels are the same either way; only what lands in
//! `HyperWork::order` differs.
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
//! factor, so extracting a common helper would have meant reshaping a delicate,
//! well-tested path for the benefit of a new one.
//!
//! That duplication is a real maintenance hazard and is guarded end to end
//! rather than at the seam: every test in `tests/lu_sparse_rhs.rs` compares this
//! module against the dense entry point on the same factor, and
//! `sparse_solves_track_forrest_tomlin_updates` does so across an update chain —
//! which is the case where the `L`-solve and the eta replay actually have to
//! agree with `spike_space`. A divergence shows up there as a numeric
//! disagreement, not as a silent drift.

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
    /// Pattern size past which this solve abandons the reach and sweeps the
    /// whole basis instead: `LuParams::sparse_rhs_max_density · m`, refreshed
    /// on every entry. `m` disables the fallback (a pattern can never exceed
    /// `m`), which is how a cap of `1.0` reads.
    cap: usize,
    /// Has *this* solve switched to whole-basis sweeps? Cleared by `reset()`.
    dense: bool,
    /// Cumulative count of solves that did — the non-vacuity witness behind
    /// [`SparseLu::sparse_rhs_fallbacks`].
    fallbacks: usize,
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
            cap: usize::MAX,
            dense: false,
            fallbacks: 0,
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
        self.dense = false;
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

    /// Has the pattern outgrown the fallback cap? Monotone within a solve —
    /// `pattern` accumulates across the three sweeps and is never shortened —
    /// so once this is true the rest of the solve is dense too, and each
    /// remaining reach costs one `pop` rather than a re-walk.
    #[inline]
    fn over_cap(&self) -> bool {
        self.pattern.len() > self.cap
    }

    /// Switch this solve to whole-basis sweeps: mark every position and clear
    /// `order` for the caller to fill with its own natural topological order.
    ///
    /// Marking all of `mark`/`pattern` is what keeps the zeroed-accumulator
    /// contract intact. The sweep about to run writes across the whole
    /// accumulator, so the reset list has to cover the whole accumulator; a
    /// nonzero left outside `pattern` would not corrupt this solve but the
    /// *next* one. That makes `reset()` `O(m)` here, which is the right price
    /// in a regime where the answer itself is `Theta(m)`.
    fn go_dense(&mut self, m: usize) {
        if !self.dense {
            self.dense = true;
            self.fallbacks += 1;
        }
        if self.pattern.len() < m {
            self.mark.fill(true);
            self.pattern.clear();
            self.pattern.extend(0..m);
        }
        self.order.clear();
        self.work += m;
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
    /// route uses, documented in `dev/decisions.md` (2026-08-13). Its `column`
    /// is the **original basis column**, matching the factor path, not the
    /// internal pivot position.
    ///
    /// # This is not bit-identical to [`SparseLu::ftran`]
    ///
    /// It sums the same terms in a different order, so results agree to
    /// rounding (the in-tree differential tests use `< 1e-9`) rather than
    /// exactly — unlike the reach-limited route behind the dense entry points,
    /// which *is* pinned with `assert_eq!` on real bases. Migrating a solver
    /// from `ftran` to `ftran_sparse` is therefore a rounding-trajectory change,
    /// and on an ill-conditioned problem that is not free: issue #163 is a case
    /// where a change of that kind cost a downstream simplex its dual bound. If
    /// you need the speedup without the trajectory change, the dense entry
    /// points already carry the reach-limited route.
    ///
    /// # A dense answer falls back to a whole-basis sweep
    ///
    /// When the solution is dense, the reach machinery costs more than it saves:
    /// it walks and sorts a length-`m` reach where the dense sweep just walks
    /// the factor as stored. Measured across 14 QPLIB relaxations driven by a
    /// dual simplex, against the dense entry points, by mean density of the
    /// solution:
    ///
    /// | solution density | speedup (geomean) |
    /// |---|---|
    /// | under 10% | **1.167x** (n = 10, best 1.359x) |
    /// | 10% and over | **0.837x** (n = 4, worst 0.711x) |
    ///
    /// log-log `r(density, speedup) = -0.944`. So once the pattern outgrows
    /// [`LuParams::sparse_rhs_max_density`](super::LuParams::sparse_rhs_max_density)
    /// · `m` (0.10 by default, the same cap and the same measurement as the
    /// dense route's) this abandons the reach mid-walk and sweeps the whole
    /// basis in natural order instead. The signature does not change: the
    /// right-hand side is still read sparsely and only the nonzeros of `x` are
    /// emitted, still sorted.
    ///
    /// The fallback is internal rather than the caller's to route because **the
    /// caller that needs it cannot supply it**: a dual simplex does not know the
    /// density of `B⁻¹A_q` until the solve that produces it has run. Set the cap
    /// to `1.0` if you know your right-hand sides are sparse and want the solves
    /// strictly work-proportional; [`SparseLu::sparse_rhs_fallbacks`] reports how
    /// often it fired.
    ///
    /// One consequence worth knowing: on the fallback the `U` diagonal guard is
    /// evaluated on every row, exactly as on [`SparseLu::ftran`], rather than
    /// narrowed to the rows the solution depends on. A degenerate pivot the
    /// answer does not depend on can therefore surface as
    /// [`FeralError::SingularBasis`] on a dense right-hand side and not on a
    /// sparse one — the same width the dense entry points have always had.
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

    /// Sparse solves that abandoned the reach and swept the whole basis because
    /// the pattern outgrew
    /// [`LuParams::sparse_rhs_max_density`](super::LuParams::sparse_rhs_max_density)
    /// (issue #164).
    ///
    /// Like [`Self::hyper_sparse_sweeps`] on the dense route, this exists
    /// because the fallback is *silent* — it returns the same answer either way,
    /// so a differential test that does not assert on this counter can pass
    /// having never run the path it means to test. Cumulative over the life of
    /// the factor; reset when the workspace is resized.
    pub fn sparse_rhs_fallbacks(&self) -> usize {
        self.hyper.fallbacks
    }

    /// Size the sparse-solve workspace, refresh the fallback cap, and make sure
    /// the `L` row index the `Lᵀ` reach walks exists.
    ///
    /// The workspace and the index are lazy, and the sparse entry points work
    /// regardless of [`LuParams::hyper_sparse_max_density`], which governs only
    /// whether the *dense* entry points take their reach-limited route. The cap
    /// they *do* read is
    /// [`LuParams::sparse_rhs_max_density`](super::LuParams::sparse_rhs_max_density)
    /// (issue #164), which is a separate knob.
    fn prepare_sparse_solve(&mut self) -> Result<(), FeralError> {
        let m = self.m;
        if !self.hyper.is_sized(m) {
            self.hyper = HyperWork::new(m);
        }
        // Read the cap on every entry rather than caching it at factor time:
        // `params` is owned by the factor and a caller may swap it between
        // solves. `d = 1.0` gives `cap = m`, which no pattern can exceed.
        let d = self.params.sparse_rhs_max_density.clamp(0.0, 1.0);
        self.hyper.cap = (m as f64 * d) as usize;
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
        if hw.dense {
            // The whole-basis fallback fired, so walk output indices instead:
            // that emits in sorted order for free, where gathering by position
            // would then have to sort `Theta(m)` pairs — the cost the fallback
            // exists to avoid, reintroduced at the last step.
            for j in 0..self.m {
                let v = hw.w[self.qcol_inv[j]];
                if v == 0.0 {
                    continue;
                }
                out.push((j, self.scale.d_col[j] * v));
            }
            return Ok(());
        }
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
        if hw.dense {
            // Whole-basis fallback: walk original rows, inverting the same
            // mapping the scatter uses. Sorted for free — see `ftran_sparse`.
            for o in 0..self.m {
                let i = self.scale_rperm_inv[o];
                let v = hw.w[self.perm_inv[i]];
                if v == 0.0 {
                    continue;
                }
                out.push((o, self.scale.d_row[i] * v));
            }
            return Ok(());
        }
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

    /// Reach of the current pattern in `L`'s successor graph.
    ///
    /// `false` means the pattern outgrew [`HyperWork::cap`] and the caller must
    /// sweep the whole basis instead (issue #164). The walk is abandoned the
    /// moment that happens rather than completed and thrown away — the same
    /// early-abort `ReachWork::push` does on the dense route — so an over-cap
    /// solve pays a bounded fraction of the DFS, not all of it.
    fn l_reach_sparse(&self, hw: &mut HyperWork) -> bool {
        if hw.over_cap() {
            return false;
        }
        hw.begin_reach();
        while let Some(k) = hw.stack.pop() {
            for idx in self.l_col_ptr[k]..self.l_col_ptr[k + 1] {
                let i = self.l_row_idx[idx];
                if hw.touch(i) {
                    hw.stack.push(i);
                }
            }
            if hw.over_cap() {
                return false;
            }
        }
        hw.finish_reach();
        true
    }

    /// Reach in `Lᵀ`'s predecessor graph (row `i` feeds every column it appears
    /// in). `false` when over cap; see [`SparseLu::l_reach_sparse`].
    fn lt_reach_sparse(&self, hw: &mut HyperWork) -> bool {
        if hw.over_cap() {
            return false;
        }
        hw.begin_reach();
        while let Some(i) = hw.stack.pop() {
            for idx in self.l_row_ptr[i]..self.l_row_ptr[i + 1] {
                let k = self.l_row_cols[idx];
                if hw.touch(k) {
                    hw.stack.push(k);
                }
            }
            if hw.over_cap() {
                return false;
            }
        }
        hw.finish_reach();
        true
    }

    /// Reach in `U`'s predecessor graph (`u_above[c]` holds the rows with an
    /// entry in column `c`). `false` when over cap; see [`SparseLu::l_reach_sparse`].
    fn u_reach_sparse(&self, hw: &mut HyperWork) -> bool {
        if hw.over_cap() {
            return false;
        }
        hw.begin_reach();
        while let Some(c) = hw.stack.pop() {
            for &k in self.u_above[c].iter() {
                if hw.touch(k) {
                    hw.stack.push(k);
                }
            }
            if hw.over_cap() {
                return false;
            }
        }
        hw.finish_reach();
        true
    }

    /// Reach in `Uᵀ`'s predecessor graph (row `i` feeds every column it holds).
    /// `false` when over cap; see [`SparseLu::l_reach_sparse`].
    fn ut_reach_sparse(&self, hw: &mut HyperWork) -> bool {
        if hw.over_cap() {
            return false;
        }
        hw.begin_reach();
        while let Some(i) = hw.stack.pop() {
            for &(c, _) in self.u_rows[i].iter().skip(1) {
                if hw.touch(c) {
                    hw.stack.push(c);
                }
            }
            if hw.over_cap() {
                return false;
            }
        }
        hw.finish_reach();
        true
    }

    /// Sparse forward solve `L y = w` over the Gilbert–Peierls reach of the
    /// current pattern. `L` is strictly lower triangular in fixed pivot-position
    /// coordinates, so ascending position is a valid topological order.
    fn l_solve_sparse(&self, hw: &mut HyperWork) {
        if self.l_reach_sparse(hw) {
            hw.order.sort_unstable();
        } else {
            hw.go_dense(self.m);
            hw.order.extend(0..self.m);
        }
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
        if self.lt_reach_sparse(hw) {
            hw.order.sort_unstable_by_key(|&k| std::cmp::Reverse(k));
        } else {
            hw.go_dense(self.m);
            hw.order.extend((0..self.m).rev());
        }
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
        if self.u_reach_sparse(hw) {
            let uperm = &self.uperm;
            hw.order.sort_unstable_by(|&a, &b| uperm[b].cmp(&uperm[a]));
        } else {
            hw.go_dense(self.m);
            hw.order
                .extend((0..self.m).rev().map(|rank| self.uperm_inv[rank]));
        }
        let HyperWork { w, order, work, .. } = hw;
        for &k in order.iter() {
            let row = &self.u_rows[k];
            let &(dc, d) = row.first().ok_or(FeralError::SingularBasis {
                column: self.qcol[k],
            })?;
            if dc != k || d == 0.0 || !d.is_finite() {
                return Err(FeralError::SingularBasis {
                    column: self.qcol[k],
                });
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
        if self.ut_reach_sparse(hw) {
            let uperm = &self.uperm;
            hw.order.sort_unstable_by(|&a, &b| uperm[a].cmp(&uperm[b]));
        } else {
            hw.go_dense(self.m);
            hw.order
                .extend((0..self.m).map(|rank| self.uperm_inv[rank]));
        }
        let HyperWork { w, order, work, .. } = hw;
        for &i in order.iter() {
            // Hoisted above the `u_rows[i]` access for the same reason the dense
            // `ut_solve` hoists it: one heap allocation per row, so touching a
            // row that scatters nothing still costs a cache miss — `m` of them
            // once the whole-basis fallback fires. A zero scatters nothing, and
            // `si = 0/d` would write back the zero already there.
            if w[i] == 0.0 {
                continue;
            }
            let row = &self.u_rows[i];
            let &(dc, d) = row.first().ok_or(FeralError::SingularBasis {
                column: self.qcol[i],
            })?;
            if dc != i || d == 0.0 || !d.is_finite() {
                return Err(FeralError::SingularBasis {
                    column: self.qcol[i],
                });
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
