//! Sparse `ftran` / `btran` triangular solves and iterative refinement.
//!
//! The factorization is `P A Q = L G U`, where `L` is the base lower factor,
//! `U` the (Forrest–Tomlin-updated) upper factor, and `G = E₁⁻¹…Eₜ⁻¹` the
//! product of the update eliminations (see [`super::sparse_update`]). So
//! `ftran` (`B⁻¹a`) applies `L⁻¹`, then `G⁻¹ = Eₜ…E₁` (the etas forward), then
//! `U⁻¹`, then `Q`; `btran` does the transpose in reverse. `ftran_partial`
//! returns the spike `G⁻¹L⁻¹Pa` (the column the update inserts into `U`).

use super::sparse_factor::SparseLu;
use super::sparse_matrix::SparseColMatrix;
use crate::error::FeralError;

/// Scratch for the reach-limited ("hyper-sparse") triangular sweeps
/// (issue #161B). Owned by [`SparseLu`] and taken out with `mem::take` for the
/// duration of a solve, so the sweep can borrow the factor immutably while
/// mutating the workspace.
///
/// `mark` is all-false between calls and is cleared over `list` — never over
/// `0..m` — so a solve that touches `r` positions pays `O(r)`, not `O(m)`, for
/// its bookkeeping. This is the same convention as `scratch_mark` in the
/// Forrest–Tomlin update. Empty (`mark.len() != m`) when the route is disabled.
#[derive(Debug, Clone, Default)]
pub(super) struct ReachWork {
    mark: Vec<bool>,
    stack: Vec<usize>,
    /// The reached positions; after a successful reach, in topological order
    /// for the sweep that asked for it.
    list: Vec<usize>,
    /// How many sweeps the reach-limited route has actually served since the
    /// factor, and how many positions they swept in total.
    ///
    /// The route is a silent fallback, so a benchmark or differential test that
    /// does not assert on `sweeps` can pass vacuously having never run the code
    /// it means to test. Both are observable through
    /// [`SparseLu::hyper_sparse_sweeps`] and [`SparseLu::hyper_sparse_nodes`].
    sweeps: usize,
    nodes: usize,
}

impl ReachWork {
    /// Workspace for an `m`-sided factor. `disabled()` when the route is off.
    pub(super) fn new(m: usize) -> Self {
        ReachWork {
            mark: vec![false; m],
            stack: Vec::new(),
            list: Vec::new(),
            sweeps: 0,
            nodes: 0,
        }
    }

    /// The zero-allocation workspace used when the route is disabled.
    pub(super) fn disabled() -> Self {
        ReachWork::default()
    }

    /// Length of the membership marker — `m` when the route is live, `0` when
    /// it was never built.
    #[inline]
    fn mark_len(&self) -> usize {
        self.mark.len()
    }

    /// Abandon a partial reach, restoring the all-false `mark` invariant.
    fn abandon(&mut self) {
        for &k in self.list.iter() {
            self.mark[k] = false;
        }
        self.list.clear();
        self.stack.clear();
    }

    /// Seed the reach from the nonzeros of `s`. Returns `false` (workspace
    /// restored) if the sources alone already exceed `cap`.
    ///
    /// This is not a second heuristic layered on `cap`: a reach always contains
    /// its own sources, so an rhs denser than `cap` could only abort later
    /// anyway. Testing it here just moves the inevitable abort earlier, before
    /// any edge is walked.
    fn seed(&mut self, s: &[f64], cap: usize) -> bool {
        debug_assert!(self.list.is_empty() && self.stack.is_empty());
        for (k, &sk) in s.iter().enumerate() {
            if sk == 0.0 {
                continue;
            }
            if self.list.len() >= cap {
                self.abandon();
                return false;
            }
            self.mark[k] = true;
            self.list.push(k);
            self.stack.push(k);
        }
        true
    }

    /// Discover `k` as reachable. Returns `false` if that would exceed `cap`.
    #[inline]
    fn push(&mut self, k: usize, cap: usize) -> bool {
        if self.mark[k] {
            return true;
        }
        if self.list.len() >= cap {
            return false;
        }
        self.mark[k] = true;
        self.list.push(k);
        self.stack.push(k);
        true
    }

    /// Close a completed reach: the marks are dead once no more edges will be
    /// walked, and clearing them here means every later return path — including
    /// an error out of the numeric sweep — leaves the workspace clean.
    fn close(&mut self) {
        for &k in self.list.iter() {
            self.mark[k] = false;
        }
        self.stack.clear();
    }

    /// Count of reach-limited sweeps this workspace has served.
    #[inline]
    pub(super) fn sweeps(&self) -> usize {
        self.sweeps
    }

    /// Total positions swept by those sweeps.
    #[inline]
    pub(super) fn nodes(&self) -> usize {
        self.nodes
    }
}

// Test-only: counts heap (re)allocations of the pooled scaled-solve / refine
// scratch buffers on the calling thread. Proves the buffers reach steady state
// with zero per-call allocation (L3, dev/research/repo-review-2026-06-09.md).
// Thread-local, not a global atomic, because the cargo harness runs solve tests
// concurrently and a shared atomic would race across sibling tests.
#[cfg(test)]
thread_local! {
    pub(super) static SOLVE_SCRATCH_ALLOCS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn reset_solve_scratch_allocs() {
    SOLVE_SCRATCH_ALLOCS.with(|c| c.set(0));
}

#[cfg(test)]
pub(super) fn solve_scratch_allocs() -> usize {
    SOLVE_SCRATCH_ALLOCS.with(|c| c.get())
}

/// Take a pooled buffer out of `pool`, sized to `m` and zeroed. Counts one
/// (re)allocation (test builds) only when the pooled buffer was not already
/// length `m` — a pre-sized pool reaches steady state at zero. The caller MUST
/// restore the buffer to `pool` after use: `mem::take` leaves `pool` empty, so
/// failing to restore turns the next call into a fresh allocation.
#[inline]
fn take_zeroed(pool: &mut Vec<f64>, m: usize) -> Vec<f64> {
    let mut b = std::mem::take(pool);
    if b.len() != m {
        #[cfg(test)]
        SOLVE_SCRATCH_ALLOCS.with(|c| c.set(c.get() + 1));
        b.clear();
        b.resize(m, 0.0);
    } else {
        for x in b.iter_mut() {
            *x = 0.0;
        }
    }
    b
}

impl SparseLu {
    /// Solve `B x = a`, overwriting `rhs` with `x` (scaling applied around the
    /// core solve on `Ã = D_row Π B D_col`).
    pub fn ftran(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let m = self.m;
        check_len(rhs.len(), m)?;
        if self.scale.is_identity() {
            return self.ftran_core(rhs);
        }
        let mut bt = take_zeroed(&mut self.scratch_b, m);
        for (i, bi) in bt.iter_mut().enumerate() {
            *bi = self.scale.d_row[i] * rhs[self.scale.rperm[i]];
        }
        // Restore the pooled buffer on every path; only write `rhs` on success.
        let res = self.ftran_core(&mut bt);
        if res.is_ok() {
            for (j, rj) in rhs.iter_mut().enumerate() {
                *rj = self.scale.d_col[j] * bt[j];
            }
        }
        self.scratch_b = bt;
        res
    }

    /// Solve `Bᵀ x = a`, overwriting `rhs` with `x`.
    pub fn btran(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let m = self.m;
        check_len(rhs.len(), m)?;
        if self.scale.is_identity() {
            return self.btran_core(rhs);
        }
        let mut bt = take_zeroed(&mut self.scratch_b, m);
        for (j, bj) in bt.iter_mut().enumerate() {
            *bj = self.scale.d_col[j] * rhs[j];
        }
        // Restore the pooled buffer on every path; only write `rhs` on success.
        let res = self.btran_core(&mut bt);
        if res.is_ok() {
            for (i, &yi) in bt.iter().enumerate() {
                rhs[self.scale.rperm[i]] = self.scale.d_row[i] * yi;
            }
        }
        self.scratch_b = bt;
        res
    }

    /// Core `ftran` on the (scaled) factored matrix, ignoring outer scaling.
    pub(super) fn ftran_core(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let mut s = std::mem::take(&mut self.scratch);
        let mut rw = std::mem::take(&mut self.reach);
        let res = self.solve_colspace(rhs, &mut s, &mut rw);
        if res.is_ok() {
            for (k, &wk) in s.iter().enumerate() {
                rhs[self.qcol[k]] = wk;
            }
        }
        self.reach = rw;
        self.scratch = s;
        res
    }

    /// Core `btran` on the (scaled) factored matrix, ignoring outer scaling.
    /// `Bᵀ⁻¹ = P⁻¹ Lᵀ⁻¹ Gᵀ⁻¹ Uᵀ⁻¹ Q⁻¹`: gather Q, `Uᵀ`-solve, apply the etas
    /// transposed in reverse, `Lᵀ`-solve, scatter P.
    pub(super) fn btran_core(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let mut s = std::mem::take(&mut self.scratch);
        let mut rw = std::mem::take(&mut self.reach);
        for (k, sk) in s.iter_mut().enumerate() {
            *sk = rhs[self.qcol[k]];
        }
        let res = self.ut_solve(&mut s);
        if res.is_ok() {
            for eta in self.etas.iter().rev() {
                eta.apply_transpose(&mut s);
            }
            self.lt_solve(&mut s, &mut rw);
            for (k, &vk) in s.iter().enumerate() {
                rhs[self.perm[k]] = vk;
            }
        }
        self.reach = rw;
        self.scratch = s;
        res
    }

    /// Compute the spike `G⁻¹ L⁻¹ P a` (the `ftran` result in `U`-column space,
    /// before the `U`-solve and `Q` scatter), overwriting `rhs`. This is the
    /// column that the Forrest–Tomlin update inserts into `U`.
    pub fn ftran_partial(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let m = self.m;
        check_len(rhs.len(), m)?;
        let mut s = std::mem::take(&mut self.scratch);
        self.spike_space(rhs, &mut s);
        rhs.copy_from_slice(&s);
        self.scratch = s;
        Ok(())
    }

    /// `ftran` with iterative refinement against the original basis `b`.
    pub fn ftran_refined(
        &mut self,
        b: &SparseColMatrix,
        rhs: &mut [f64],
    ) -> Result<(), FeralError> {
        check_len(rhs.len(), self.m)?;
        let mut a = take_zeroed(&mut self.scratch_d, self.m);
        a.copy_from_slice(rhs);
        let res = match self.ftran(rhs) {
            Ok(()) => self.refine(b, &a, rhs, false),
            Err(e) => Err(e),
        };
        self.scratch_d = a;
        res
    }

    /// `btran` with iterative refinement against the original basis `b`.
    pub fn btran_refined(
        &mut self,
        b: &SparseColMatrix,
        rhs: &mut [f64],
    ) -> Result<(), FeralError> {
        check_len(rhs.len(), self.m)?;
        let mut a = take_zeroed(&mut self.scratch_d, self.m);
        a.copy_from_slice(rhs);
        let res = match self.btran(rhs) {
            Ok(()) => self.refine(b, &a, rhs, true),
            Err(e) => Err(e),
        };
        self.scratch_d = a;
        res
    }

    /// Spike `G⁻¹ L⁻¹ P · rhs` (apply P, `L`-solve, replay the FT etas forward),
    /// without the `U`-solve. Used to form the column the update inserts into U.
    pub(super) fn spike_space(&self, rhs: &[f64], out: &mut [f64]) {
        for (k, ok) in out.iter_mut().enumerate() {
            *ok = rhs[self.perm[k]];
        }
        self.lsolve(out);
        for eta in self.etas.iter() {
            eta.apply_forward(out);
        }
    }

    /// Solve into column-position space: `out = U⁻¹ G⁻¹ L⁻¹ P · rhs` (the
    /// `ftran` result before the final `Q` scatter).
    pub(super) fn solve_colspace(
        &self,
        rhs: &[f64],
        out: &mut [f64],
        rw: &mut ReachWork,
    ) -> Result<(), FeralError> {
        self.spike_space(rhs, out);
        self.usolve(out, rw)
    }

    /// Node budget for the reach-limited route, or `None` when it is disabled
    /// (`hyper_sparse_max_density == 0`) or its workspace was never built —
    /// which is the same condition, since the workspace is only allocated when
    /// the route is on. A cap of `0` (a small `m` against a small density) is
    /// legal and simply routes every nonempty rhs to the dense sweep.
    #[inline]
    fn reach_cap(&self, rw: &ReachWork) -> Option<usize> {
        let d = self.params.hyper_sparse_max_density;
        // The liveness test reads `rw`, not `self.reach`: the solve entry points
        // `mem::take` the workspace out of the factor for the duration of the
        // call, so `self.reach` is the empty `Default` right now and testing it
        // here would disable the route on every solve.
        if d <= 0.0 || rw.mark_len() != self.m {
            return None;
        }
        Some((self.m as f64 * d) as usize)
    }

    /// Forward solve `L y = s` (unit lower), in place.
    fn lsolve(&self, s: &mut [f64]) {
        for k in 0..self.m {
            let sk = s[k];
            if sk == 0.0 {
                continue;
            }
            let (lo, hi) = (self.l_col_ptr[k], self.l_col_ptr[k + 1]);
            for idx in lo..hi {
                s[self.l_row_idx[idx]] -= self.l_val[idx] * sk;
            }
        }
    }

    /// Back solve `U w = s` (upper; per-row storage, diagonal first), in place.
    ///
    /// Errors with [`FeralError::SingularBasis`] if a row's stored diagonal is
    /// absent, zero, non-finite, or stored out of diagonal-first order: after a
    /// Forrest–Tomlin update the diagonal of `u_rows[k]` is the bump pivot, and
    /// dividing by an exact zero would otherwise emit a silent `±Inf`. (A fresh
    /// factor floors pivots to `±ztol`, so the zero guard only bites the updated
    /// path.) The `dc != k` check is an always-on hardening of the diagonal-first
    /// invariant (L10): without it, a violated invariant would make release-mode
    /// solves silently take an off-diagonal as the pivot.
    ///
    /// Takes the reach-limited route when [`LuParams::hyper_sparse_max_density`]
    /// admits it (issue #161B). On that route the diagonal guard is evaluated on
    /// the rows the solution depends on rather than on all `m`; see
    /// [`SparseLu::usolve_over`].
    fn usolve(&self, s: &mut [f64], rw: &mut ReachWork) -> Result<(), FeralError> {
        if let Some(cap) = self.reach_cap(rw) {
            if self.u_reach(s, rw, cap) {
                let res = self.usolve_over(s, rw.list.iter().copied());
                rw.list.clear();
                return res;
            }
        }
        // Dense back-substitution in `uperm` order: process positions by
        // *decreasing* triangular rank. Each row's pivot is its diagonal entry
        // (column == its own position, stored first); its off-diagonal columns
        // all have strictly greater rank, so they are already solved. At
        // identity `uperm` (`uperm_inv[rank] == rank`) this is the plain
        // reverse-position sweep.
        self.usolve_over(s, (0..self.m).rev().map(|rank| self.uperm_inv[rank]))
    }

    /// The `U` back substitution over `order`, which must list the positions to
    /// solve in decreasing triangular rank.
    ///
    /// Positions absent from `order` are left untouched. On the reach-limited
    /// route that is exactly right: a position outside the reach has `s[k] == 0`
    /// and no reached predecessor, so back substitution would assign it
    /// `0 / U[k,k] == 0`, which is what leaving it alone already gives. It does
    /// mean the diagonal guard below is not evaluated for those rows — a
    /// deliberate narrowing recorded in
    /// `dev/research/hyper-sparse-solves-2026-08-13.md`. Degeneracy the caller
    /// did not ask about is caught at factor and update time, against the pivot
    /// tolerance, rather than incidentally during an unrelated solve.
    fn usolve_over(
        &self,
        s: &mut [f64],
        order: impl Iterator<Item = usize>,
    ) -> Result<(), FeralError> {
        for k in order {
            let row = &self.u_rows[k];
            let &(dc, d) = row.first().ok_or(FeralError::SingularBasis { column: k })?;
            if dc != k || d == 0.0 || !d.is_finite() {
                return Err(FeralError::SingularBasis { column: k });
            }
            let mut acc = s[k];
            for &(c, v) in row[1..].iter() {
                acc -= v * s[c];
            }
            s[k] = acc / d;
        }
        Ok(())
    }

    /// Reach of `s`'s nonzeros in `U`'s predecessor graph, into `rw.list` in
    /// decreasing triangular rank. `false` (workspace clean) if it exceeds `cap`.
    ///
    /// `w[k]` can be nonzero only if `s[k] != 0` or `U[k,c] != 0` for some
    /// already-nonzero `w[c]`, so the edges out of `c` are the off-diagonal
    /// holders of column `c` — exactly `u_above[c]`, which the Forrest–Tomlin
    /// update already builds and maintains. Every such edge runs from higher to
    /// lower triangular rank (`U` is upper triangular *in `uperm` order*), so
    /// sorting by decreasing `uperm` is a valid topological order. It must be
    /// the rank and not the position index: after an update a column can hold
    /// entries at positions whose index is below it but whose rank is above.
    fn u_reach(&self, s: &[f64], rw: &mut ReachWork, cap: usize) -> bool {
        if !rw.seed(s, cap) {
            return false;
        }
        while let Some(c) = rw.stack.pop() {
            for &k in self.u_above[c].iter() {
                if !rw.push(k, cap) {
                    rw.abandon();
                    return false;
                }
            }
        }
        rw.close();
        let uperm = &self.uperm;
        rw.list.sort_unstable_by(|&a, &b| uperm[b].cmp(&uperm[a]));
        rw.sweeps += 1;
        rw.nodes += rw.list.len();
        true
    }

    /// Forward solve `Uᵀ z = s` (`Uᵀ` lower; scatter form on per-row U).
    ///
    /// Errors with [`FeralError::SingularBasis`] on an absent/zero/non-finite
    /// or out-of-order stored diagonal, for the same reason as
    /// [`SparseLu::usolve`].
    ///
    /// The `s[i] == 0.0` test is hoisted *above* the `u_rows[i]` access, which
    /// is what makes this kernel work-proportional in cache traffic and not just
    /// in flops: `u_rows` is one heap allocation per row, so reading
    /// `row.first()` on a row that contributes nothing still costs a cache miss,
    /// `m` times per solve (issue #161B). A zero entry scatters nothing, so
    /// skipping it changes no arithmetic — only, as on the reach-limited `usolve`
    /// route, which rows the diagonal guard sees. `lsolve` has always hoisted
    /// its zero test the same way.
    fn ut_solve(&self, s: &mut [f64]) -> Result<(), FeralError> {
        // Forward solve in `uperm` order: process positions by *increasing*
        // triangular rank (the transpose of `usolve`). At identity `uperm` this
        // is the plain ascending-position sweep.
        for rank in 0..self.m {
            let i = self.uperm_inv[rank];
            if s[i] == 0.0 {
                continue;
            }
            let row = &self.u_rows[i];
            let &(dc, d) = row.first().ok_or(FeralError::SingularBasis { column: i })?;
            if dc != i || d == 0.0 || !d.is_finite() {
                return Err(FeralError::SingularBasis { column: i });
            }
            let si = s[i] / d;
            s[i] = si;
            if si == 0.0 {
                continue;
            }
            for &(c, v) in row[1..].iter() {
                s[c] -= v * si;
            }
        }
        Ok(())
    }

    /// Back solve `Lᵀ v = s` (`Lᵀ` unit upper), in place. Reach-limited when
    /// [`LuParams::hyper_sparse_max_density`] admits it (issue #161B).
    fn lt_solve(&self, s: &mut [f64], rw: &mut ReachWork) {
        if let Some(cap) = self.reach_cap(rw) {
            if self.lt_reach(s, rw, cap) {
                for &k in rw.list.iter() {
                    let mut acc = s[k];
                    let (lo, hi) = (self.l_col_ptr[k], self.l_col_ptr[k + 1]);
                    for idx in lo..hi {
                        acc -= self.l_val[idx] * s[self.l_row_idx[idx]];
                    }
                    s[k] = acc;
                }
                rw.list.clear();
                return;
            }
        }
        for k in (0..self.m).rev() {
            let mut acc = s[k];
            let (lo, hi) = (self.l_col_ptr[k], self.l_col_ptr[k + 1]);
            for idx in lo..hi {
                acc -= self.l_val[idx] * s[self.l_row_idx[idx]];
            }
            s[k] = acc;
        }
    }

    /// Reach of `s`'s nonzeros in `Lᵀ`'s predecessor graph, into `rw.list` in
    /// decreasing pivot position. `false` (workspace clean) if it exceeds `cap`,
    /// or if the `L` row index was not built (the route is off).
    ///
    /// `v[k]` can be nonzero only if `s[k] != 0` or `L[i,k] != 0` for some
    /// already-nonzero `v[i]`, so the edges out of `i` are the columns in which
    /// row `i` appears — the *row*-wise structure of `L`, which `L`'s CSC
    /// storage does not give directly and which
    /// `build_l_row_index` therefore materializes at factor time.
    /// `L` is strictly lower triangular in fixed pivot-position coordinates and
    /// the Forrest–Tomlin update never touches it, so every edge runs from a
    /// higher to a lower position for the whole life of the factor, and sorting
    /// by decreasing position is a valid topological order.
    fn lt_reach(&self, s: &[f64], rw: &mut ReachWork, cap: usize) -> bool {
        if self.l_row_ptr.len() != self.m + 1 {
            return false;
        }
        if !rw.seed(s, cap) {
            return false;
        }
        while let Some(i) = rw.stack.pop() {
            for idx in self.l_row_ptr[i]..self.l_row_ptr[i + 1] {
                if !rw.push(self.l_row_cols[idx], cap) {
                    rw.abandon();
                    return false;
                }
            }
        }
        rw.close();
        rw.list.sort_unstable_by_key(|&k| std::cmp::Reverse(k));
        rw.sweeps += 1;
        rw.nodes += rw.list.len();
        true
    }

    fn refine(
        &mut self,
        b: &SparseColMatrix,
        a: &[f64],
        x: &mut [f64],
        transpose: bool,
    ) -> Result<(), FeralError> {
        let steps = self.params.refine_steps;
        let tol = self.params.refine_tol;
        if steps == 0 {
            return Ok(());
        }
        let anorm = inf_norm(a);
        if anorm == 0.0 {
            return Ok(());
        }
        let mut r = take_zeroed(&mut self.scratch_c, self.m);
        let mut result = Ok(());
        for _ in 0..steps {
            if transpose {
                b.matvec_transpose(x, &mut r);
            } else {
                b.matvec(x, &mut r);
            }
            for (ri, &ai) in r.iter_mut().zip(a.iter()) {
                *ri = ai - *ri;
            }
            if inf_norm(&r) / anorm < tol {
                break;
            }
            // Restore the pooled residual buffer on every path before returning.
            let step = if transpose {
                self.btran(&mut r)
            } else {
                self.ftran(&mut r)
            };
            if let Err(e) = step {
                result = Err(e);
                break;
            }
            for (xi, &dxi) in x.iter_mut().zip(r.iter()) {
                *xi += dxi;
            }
        }
        self.scratch_c = r;
        result
    }
}

fn check_len(got: usize, expected: usize) -> Result<(), FeralError> {
    if got != expected {
        Err(FeralError::DimensionMismatch { expected, got })
    } else {
        Ok(())
    }
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lu::{LuParams, LuScaling};

    /// L3 (dev/research/repo-review-2026-06-09.md): the sparse twin of the dense
    /// pooling guard. With scaling enabled the `ftran`/`btran` wrappers and the
    /// refine loop must reuse pooled struct buffers, not allocate a fresh
    /// `vec![0.0; m]` per call. `SOLVE_SCRATCH_ALLOCS` counts a (re)allocation
    /// only when a pooled buffer is taken at the wrong length, so steady-state
    /// must be exactly zero.
    #[test]
    fn scaled_solves_and_refine_reuse_pooled_scratch() {
        let cols = vec![
            vec![10.0, 1.0, 0.0],
            vec![1.0, 8.0, 2.0],
            vec![0.0, 1.0, 5.0],
        ];
        let m = 3;
        let params = LuParams {
            scaling: LuScaling::InfNorm,
            refine_steps: 2,
            ..LuParams::default()
        };
        let mut lu = SparseLu::factor_dense_columns(m, &cols, params).expect("factor");
        assert!(
            !lu.scale.is_identity(),
            "InfNorm scaling should be non-identity for this matrix"
        );
        let b = SparseColMatrix::from_dense_columns(m, &cols).expect("sparse matrix");

        reset_solve_scratch_allocs();
        for _ in 0..5 {
            let mut x = vec![1.0, 2.0, 3.0];
            lu.ftran(&mut x).expect("ftran");
            assert!(x.iter().all(|v| v.is_finite()));
            let mut y = vec![3.0, 2.0, 1.0];
            lu.btran(&mut y).expect("btran");
            assert!(y.iter().all(|v| v.is_finite()));
        }
        let mut xr = vec![1.0, 1.0, 1.0];
        lu.ftran_refined(&b, &mut xr).expect("ftran_refined");
        let mut yr = vec![1.0, 1.0, 1.0];
        lu.btran_refined(&b, &mut yr).expect("btran_refined");

        assert_eq!(
            solve_scratch_allocs(),
            0,
            "scaled ftran/btran + refine must reuse pooled buffers, not \
             allocate per call (L3)"
        );

        // Correctness: the pooling must not change the math — B x = a.
        let a = vec![2.0, -1.0, 4.0];
        let mut x = a.clone();
        lu.ftran(&mut x).expect("ftran");
        let mut bx = vec![0.0; m];
        b.matvec(&x, &mut bx);
        for (bxi, ai) in bx.iter().zip(a.iter()) {
            assert!((bxi - ai).abs() < 1e-9, "B x != a: {bxi} vs {ai}");
        }
    }

    /// L6 (dev/research/repo-review-2026-06-09.md): the sparse twin of the dense
    /// tiny-basis test. `diag(1e-14)` is perfectly conditioned (cond₂ = 1, exact
    /// inverse `diag(1e14)`) but every pivot 1e-14 ≤ the absolute `zero_pivot_tol`
    /// (1e-13), so the sparse factor declared `SingularBasis { column: 0 }`. With
    /// the relative tolerance `zero_pivot_tol · max|A|` it factors. Oracle: the
    /// hand-computed exact solution of `B x = b`. Pre-fix this `expect` panics.
    #[test]
    fn factor_tiny_well_conditioned_basis_not_singular() {
        let s = 1e-14;
        let cols = vec![vec![s, 0.0], vec![0.0, s]];
        let mut lu =
            SparseLu::factor_dense_columns(2, &cols, LuParams::default()).expect("tiny basis");
        // B = s·I, b = s·[1, 2]  =>  x = [1, 2] exactly.
        let mut rhs = vec![s, 2.0 * s];
        lu.ftran(&mut rhs).expect("ftran");
        assert!((rhs[0] - 1.0).abs() < 1e-6, "x0 = {}", rhs[0]);
        assert!((rhs[1] - 2.0).abs() < 1e-6, "x1 = {}", rhs[1]);
    }

    /// L2 (dev/research/repo-review-2026-06-09.md): the sparse LU must honor
    /// `pivot_threshold` (threshold partial pivoting), matching the dense path
    /// and the documented contract at `lu/mod.rs:67-69`. Before the fix `utol`
    /// was discarded (`let _ = utol`) and the sparse path always took the
    /// strict-max-magnitude row, so a sub-1.0 threshold silently changed
    /// nothing.
    ///
    /// Matrix A = [[1,0],[2,1]] (col0 = [1,2], col1 = [0,1]), natural column
    /// order: column 0 has diagonal `w[0] = 1` and off-diagonal `w[1] = 2`.
    /// With u = 1.0 (strict) `|2| > |1|`, so the pivot is row 1 and perm[0] = 1.
    /// With u = 0.5 (threshold) `|w[0]| = 1 >= 0.5*2 = 1.0`, so the diagonal is
    /// within threshold and the structure-preserving row is taken: perm[0] = 0.
    /// Both factorizations must still solve A x = b exactly. External oracle:
    /// the hand-computed solution of A x = [1, 4] is x = [1, 2]. The pivot-row
    /// divergence is the behavioral witness; pre-fix both perms are [1, 0].
    /// The diagonal-preference rule matches CSparse `cs_lu`
    /// (Davis, Direct Methods for Sparse Linear Systems, section 6.3).
    #[test]
    fn sparse_lu_honors_pivot_threshold() {
        use crate::lu::SparseLuSymbolic;
        let cols = vec![vec![1.0, 2.0], vec![0.0, 1.0]];
        let a = SparseColMatrix::from_dense_columns(2, &cols).expect("matrix");
        let symbolic = SparseLuSymbolic::natural(2);

        // Strict partial pivoting (u = 1.0): the max-magnitude row wins.
        let strict = SparseLu::factor(
            &a,
            &symbolic,
            LuParams {
                pivot_threshold: 1.0,
                ..LuParams::default()
            },
        )
        .expect("strict factor");
        assert_eq!(strict.perm()[0], 1, "u=1.0 must pivot the larger row 1");

        // Threshold partial pivoting (u = 0.5): the diagonal is within
        // threshold, so it is preferred over the larger off-diagonal entry.
        let mut relaxed = SparseLu::factor(
            &a,
            &symbolic,
            LuParams {
                pivot_threshold: 0.5,
                ..LuParams::default()
            },
        )
        .expect("relaxed factor");
        assert_eq!(
            relaxed.perm()[0],
            0,
            "u=0.5 must prefer the within-threshold diagonal row 0 (L2)"
        );

        // Both must solve correctly. Oracle: A x = [1, 4]  =>  x = [1, 2].
        let mut strict = strict;
        let mut rhs = vec![1.0, 4.0];
        strict.ftran(&mut rhs).expect("strict ftran");
        assert!(
            (rhs[0] - 1.0).abs() < 1e-12 && (rhs[1] - 2.0).abs() < 1e-12,
            "strict solve {rhs:?}"
        );
        let mut rhs = vec![1.0, 4.0];
        relaxed.ftran(&mut rhs).expect("relaxed ftran");
        assert!(
            (rhs[0] - 1.0).abs() < 1e-12 && (rhs[1] - 2.0).abs() < 1e-12,
            "relaxed solve {rhs:?}"
        );
    }

    /// L2 follow-up (repo-review-2026-06-09-verification.md, residual #2): the
    /// sparse diagonal-preference rule must also require the diagonal to clear
    /// the singularity floor `ztol`, matching the dense path's `&& diag > ztol`
    /// conjunct (`dense_factor.rs`). Without that conjunct a `pivot_threshold`
    /// at or below `zero_pivot_tol` lets a sub-`ztol` diagonal be *preferred*
    /// over a sound max-magnitude row (it is "within threshold": `1e-14 >=
    /// u·amax` for `u = 1e-14`), then the old silent `±ztol` clamp perturbed it
    /// to `±ztol` without consulting `on_singular` — a sub-tolerance pivot
    /// perturbation under `Fail` and a drift from the dense path, which routes
    /// the same matrix through its max-magnitude row.
    ///
    /// Matrix A = [[1e-14, 1.0], [1.0, 1.0]] is well conditioned (det = 1e-14 −
    /// 1 ≈ −1), but the (0,0) diagonal `1e-14` is two orders below
    /// `ztol = zero_pivot_tol·max|A| = 1e-13`. The sound pivot is the
    /// max-magnitude row 1, NOT the sub-floor diagonal. External oracle: the
    /// dense LU on the same matrix (which pivots row 1) and the hand-computed
    /// solution of A x = [2, 3], x ≈ [1, 2]. Pre-fix the sparse path pivots
    /// row 0 with a silently clamped pivot (`perm()[0] == 0`); post-fix it
    /// matches the dense path (`perm()[0] == 1`).
    #[test]
    fn sparse_diagonal_preference_respects_zero_pivot_floor() {
        use super::super::DenseLu;
        use crate::lu::SparseLuSymbolic;

        let cols = vec![vec![1e-14, 1.0], vec![1.0, 1.0]];
        // pivot_threshold == zero_pivot_tol/amax-scale region: a *valid*
        // (in (0, 1]) but tiny threshold that makes the sub-floor diagonal
        // "within threshold". The conjunct, not range validation, must catch it.
        let params = LuParams {
            pivot_threshold: 1e-14,
            ..LuParams::default()
        };

        let a = SparseColMatrix::from_dense_columns(2, &cols).expect("matrix");
        let symbolic = SparseLuSymbolic::natural(2);
        let mut sparse = SparseLu::factor(&a, &symbolic, params.clone()).expect("sparse factor");
        assert_eq!(
            sparse.perm()[0],
            1,
            "a sub-ztol diagonal must not be preferred over the max-magnitude row (L2)"
        );

        // Dense parity: the dense LU pivots the same row on the same matrix.
        let dense = DenseLu::factor(&cols, 2, params).expect("dense factor");
        assert_eq!(
            sparse.perm()[0],
            dense.perm()[0],
            "dense/sparse diagonal-preference parity"
        );

        // And the solve is correct. Oracle: A x = [2, 3]  =>  x ≈ [1, 2].
        let mut rhs = vec![2.0, 3.0];
        sparse.ftran(&mut rhs).expect("sparse ftran");
        assert!(
            (rhs[0] - 1.0).abs() < 1e-9 && (rhs[1] - 2.0).abs() < 1e-9,
            "sparse solve {rhs:?}"
        );
    }

    /// `pivot_threshold` outside `(0, 1]` (the documented `u` range at
    /// `lu/mod.rs`) is now rejected with `InvalidInput` on both factor paths,
    /// rather than silently producing a degenerate pivot rule. `0.0` would
    /// disable pivoting entirely (always prefer the diagonal); `> 1.0` is
    /// meaningless; `NaN` poisons every comparison. Oracle: the documented
    /// contract `u ∈ (0, 1]`.
    #[test]
    fn pivot_threshold_out_of_range_is_rejected() {
        use super::super::DenseLu;
        use crate::lu::SparseLuSymbolic;

        let cols = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let a = SparseColMatrix::from_dense_columns(2, &cols).expect("matrix");
        let symbolic = SparseLuSymbolic::natural(2);

        for bad in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            let params = LuParams {
                pivot_threshold: bad,
                ..LuParams::default()
            };
            assert!(
                matches!(
                    SparseLu::factor(&a, &symbolic, params.clone()),
                    Err(FeralError::InvalidInput(_))
                ),
                "sparse factor must reject pivot_threshold = {bad}"
            );
            assert!(
                matches!(
                    DenseLu::factor(&cols, 2, params),
                    Err(FeralError::InvalidInput(_))
                ),
                "dense factor must reject pivot_threshold = {bad}"
            );
        }

        // A valid boundary value (u = 1.0, strict partial pivoting) still works.
        let ok = LuParams {
            pivot_threshold: 1.0,
            ..LuParams::default()
        };
        assert!(SparseLu::factor(&a, &symbolic, ok.clone()).is_ok());
        assert!(DenseLu::factor(&cols, 2, ok).is_ok());
    }

    /// A zero `U` diagonal (as a degenerate post-update bump pivot could leave)
    /// must surface as `SingularBasis`, not a silent `±Inf` out of the divide.
    #[test]
    fn zero_u_diagonal_errors_instead_of_inf() {
        let cols = vec![vec![2.0, 0.0], vec![1.0, 3.0]]; // nonsingular 2x2
        let mut lu = SparseLu::factor_dense_columns(2, &cols, LuParams::default()).expect("factor");
        // Sanity: a clean solve has no NaN/Inf.
        let mut rhs = vec![1.0, 1.0];
        lu.ftran(&mut rhs).expect("clean ftran");
        assert!(rhs.iter().all(|x| x.is_finite()));

        // Corrupt the stored diagonal of pivot position 1 to an exact zero.
        lu.u_rows[1][0].1 = 0.0;

        let mut bad = vec![1.0, 1.0];
        assert!(matches!(
            lu.ftran(&mut bad),
            Err(FeralError::SingularBasis { column: 1 })
        ));
        let mut bad_t = vec![1.0, 1.0];
        assert!(matches!(
            lu.btran(&mut bad_t),
            Err(FeralError::SingularBasis { column: 1 })
        ));
    }

    /// L10 (dev/research/repo-review-2026-06-09.md): the diagonal-first
    /// `u_rows` invariant was enforced only by a `debug_assert_eq!`, compiled
    /// out in release. A violated invariant would make release-mode `usolve` /
    /// `ut_solve` silently take an off-diagonal entry as the pivot (and treat
    /// the real diagonal as a regular `row[1..]` term) — a silent wrong solve.
    /// The guard is now always-on: a U row whose first entry is not its
    /// diagonal surfaces as `SingularBasis` in every build mode, matching the
    /// absent/zero/non-finite diagonal guard. Pre-fix this test panics on the
    /// `debug_assert_eq!` (a debug build) rather than returning a clean `Err`.
    #[test]
    fn misplaced_u_diagonal_errors_instead_of_silent_wrong_pivot() {
        let cols = vec![vec![2.0, 0.0], vec![1.0, 3.0]]; // nonsingular 2x2
        let mut lu = SparseLu::factor_dense_columns(2, &cols, LuParams::default()).expect("factor");
        // u_rows[0] stores its diagonal (column 0) first, then an off-diagonal.
        assert!(
            lu.u_rows[0].len() >= 2,
            "test needs an off-diagonal to misplace the diagonal behind"
        );
        assert_eq!(lu.u_rows[0][0].0, 0, "diagonal of row 0 must start first");
        // Corrupt the invariant: move the diagonal off the front of row 0.
        lu.u_rows[0].swap(0, 1);

        // usolve (via ftran) must reject the misplaced diagonal, not divide by
        // the off-diagonal value as the pivot.
        let mut bad = vec![1.0, 1.0];
        assert!(
            matches!(
                lu.ftran(&mut bad),
                Err(FeralError::SingularBasis { column: 0 })
            ),
            "usolve must reject a misplaced diagonal (L10)"
        );

        // ut_solve (via btran) on a fresh, equally-corrupted factor.
        let mut lu_t =
            SparseLu::factor_dense_columns(2, &cols, LuParams::default()).expect("factor");
        lu_t.u_rows[0].swap(0, 1);
        let mut bad_t = vec![1.0, 1.0];
        assert!(
            matches!(
                lu_t.btran(&mut bad_t),
                Err(FeralError::SingularBasis { column: 0 })
            ),
            "ut_solve must reject a misplaced diagonal (L10)"
        );
    }
}
