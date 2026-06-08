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

impl SparseLu {
    /// Solve `B x = a`, overwriting `rhs` with `x` (scaling applied around the
    /// core solve on `Ã = D_row Π B D_col`).
    pub fn ftran(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let m = self.m;
        check_len(rhs.len(), m)?;
        if self.scale.is_identity() {
            return self.ftran_core(rhs);
        }
        let mut bt = vec![0.0; m];
        for (i, bi) in bt.iter_mut().enumerate() {
            *bi = self.scale.d_row[i] * rhs[self.scale.rperm[i]];
        }
        self.ftran_core(&mut bt)?;
        for (j, rj) in rhs.iter_mut().enumerate() {
            *rj = self.scale.d_col[j] * bt[j];
        }
        Ok(())
    }

    /// Solve `Bᵀ x = a`, overwriting `rhs` with `x`.
    pub fn btran(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let m = self.m;
        check_len(rhs.len(), m)?;
        if self.scale.is_identity() {
            return self.btran_core(rhs);
        }
        let mut bt = vec![0.0; m];
        for (j, bj) in bt.iter_mut().enumerate() {
            *bj = self.scale.d_col[j] * rhs[j];
        }
        self.btran_core(&mut bt)?;
        for (i, &yi) in bt.iter().enumerate() {
            rhs[self.scale.rperm[i]] = self.scale.d_row[i] * yi;
        }
        Ok(())
    }

    /// Core `ftran` on the (scaled) factored matrix, ignoring outer scaling.
    pub(super) fn ftran_core(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let mut s = std::mem::take(&mut self.scratch);
        self.solve_colspace(rhs, &mut s);
        for (k, &wk) in s.iter().enumerate() {
            rhs[self.qcol[k]] = wk;
        }
        self.scratch = s;
        Ok(())
    }

    /// Core `btran` on the (scaled) factored matrix, ignoring outer scaling.
    /// `Bᵀ⁻¹ = P⁻¹ Lᵀ⁻¹ Gᵀ⁻¹ Uᵀ⁻¹ Q⁻¹`: gather Q, `Uᵀ`-solve, apply the etas
    /// transposed in reverse, `Lᵀ`-solve, scatter P.
    pub(super) fn btran_core(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let mut s = std::mem::take(&mut self.scratch);
        for (k, sk) in s.iter_mut().enumerate() {
            *sk = rhs[self.qcol[k]];
        }
        self.ut_solve(&mut s);
        for eta in self.etas.iter().rev() {
            eta.apply_transpose(&mut s);
        }
        self.lt_solve(&mut s);
        for (k, &vk) in s.iter().enumerate() {
            rhs[self.perm[k]] = vk;
        }
        self.scratch = s;
        Ok(())
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
        let a = rhs.to_vec();
        self.ftran(rhs)?;
        self.refine(b, &a, rhs, false)
    }

    /// `btran` with iterative refinement against the original basis `b`.
    pub fn btran_refined(
        &mut self,
        b: &SparseColMatrix,
        rhs: &mut [f64],
    ) -> Result<(), FeralError> {
        check_len(rhs.len(), self.m)?;
        let a = rhs.to_vec();
        self.btran(rhs)?;
        self.refine(b, &a, rhs, true)
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
    pub(super) fn solve_colspace(&self, rhs: &[f64], out: &mut [f64]) {
        self.spike_space(rhs, out);
        self.usolve(out);
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
    fn usolve(&self, s: &mut [f64]) {
        for k in (0..self.m).rev() {
            let row = &self.u_rows[k];
            let mut acc = s[k];
            // Skip the diagonal (row[0], column == k); the rest are columns > k.
            for &(c, v) in row[1..].iter() {
                acc -= v * s[c];
            }
            s[k] = acc / row[0].1;
        }
    }

    /// Forward solve `Uᵀ z = s` (`Uᵀ` lower; scatter form on per-row U).
    fn ut_solve(&self, s: &mut [f64]) {
        for i in 0..self.m {
            let row = &self.u_rows[i];
            let si = s[i] / row[0].1;
            s[i] = si;
            if si == 0.0 {
                continue;
            }
            for &(c, v) in row[1..].iter() {
                s[c] -= v * si;
            }
        }
    }

    /// Back solve `Lᵀ v = s` (`Lᵀ` unit upper), in place.
    fn lt_solve(&self, s: &mut [f64]) {
        for k in (0..self.m).rev() {
            let mut acc = s[k];
            let (lo, hi) = (self.l_col_ptr[k], self.l_col_ptr[k + 1]);
            for idx in lo..hi {
                acc -= self.l_val[idx] * s[self.l_row_idx[idx]];
            }
            s[k] = acc;
        }
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
        let mut r = vec![0.0; self.m];
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
            if transpose {
                self.btran(&mut r)?;
            } else {
                self.ftran(&mut r)?;
            }
            for (xi, &dxi) in x.iter_mut().zip(r.iter()) {
                *xi += dxi;
            }
        }
        Ok(())
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
