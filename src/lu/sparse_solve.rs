//! Sparse `ftran` / `btran` triangular solves and iterative refinement.
//!
//! Mirrors the dense path with `P A Q = L U F₁…Fₜ`, where the `Fᵢ` are the
//! product-form `U`-updates (see [`super::sparse_update`]). `ftran` solves
//! `B x = a`, `btran` solves `Bᵀ x = a`, and `ftran_partial` returns the spike
//! `L⁻¹ P a`. Triangular solves operate directly on the CSC factors (no
//! explicit transpose), and the eta updates are applied after the `U`-solve in
//! `ftran` (transposed, in reverse, before the `Uᵀ`-solve in `btran`).

use super::sparse_factor::SparseLu;
use super::sparse_matrix::SparseColMatrix;
use crate::error::FeralError;

impl SparseLu {
    /// Solve `B x = a`, overwriting `rhs` with `x`.
    pub fn ftran(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let m = self.m;
        check_len(rhs.len(), m)?;
        let mut s = std::mem::take(&mut self.scratch);
        self.solve_colspace(rhs, &mut s);
        for (k, &wk) in s.iter().enumerate() {
            rhs[self.qcol[k]] = wk;
        }
        self.scratch = s;
        Ok(())
    }

    /// Solve `Bᵀ x = a`, overwriting `rhs` with `x`.
    pub fn btran(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let m = self.m;
        check_len(rhs.len(), m)?;
        let mut s = std::mem::take(&mut self.scratch);
        for (k, sk) in s.iter_mut().enumerate() {
            *sk = rhs[self.qcol[k]];
        }
        self.apply_etas_transpose_reverse(&mut s);
        self.ut_solve(&mut s);
        self.lt_solve(&mut s);
        for (k, &vk) in s.iter().enumerate() {
            rhs[self.perm[k]] = vk;
        }
        self.scratch = s;
        Ok(())
    }

    /// Compute the spike `L⁻¹ P a` (base factor only, no eta/`U`), overwriting
    /// `rhs`. Exposed for inspection; the update computes its own transformed
    /// spike internally.
    pub fn ftran_partial(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let m = self.m;
        check_len(rhs.len(), m)?;
        let mut s = std::mem::take(&mut self.scratch);
        for (k, sk) in s.iter_mut().enumerate() {
            *sk = rhs[self.perm[k]];
        }
        self.lsolve(&mut s);
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

    /// Solve into column-position space: `out = F⁻¹ U⁻¹ L⁻¹ P · rhs` (the
    /// `ftran` result before the final `Q` scatter). Shared by `ftran` and the
    /// rank-1 update (which reads `out[qcol_inv[slot]]` as the update pivot).
    pub(super) fn solve_colspace(&self, rhs: &[f64], out: &mut [f64]) {
        for (k, ok) in out.iter_mut().enumerate() {
            *ok = rhs[self.perm[k]];
        }
        self.lsolve(out);
        self.usolve(out);
        self.apply_etas_forward(out);
    }

    /// Apply all product-form updates `F₁⁻¹ … Fₜ⁻¹` (chronological order).
    fn apply_etas_forward(&self, s: &mut [f64]) {
        for eta in self.etas.iter() {
            let q = eta.q;
            let tq = eta.tau[q];
            let zq = s[q] / tq;
            for (j, &tj) in eta.tau.iter().enumerate() {
                if j != q {
                    s[j] -= tj * zq;
                }
            }
            s[q] = zq;
        }
    }

    /// Apply `(F₁⁻¹ … Fₜ⁻¹)ᵀ` in reverse: each `(Fᵢ⁻¹)ᵀ` changes only `s[q]`.
    fn apply_etas_transpose_reverse(&self, s: &mut [f64]) {
        for eta in self.etas.iter().rev() {
            let q = eta.q;
            let tq = eta.tau[q];
            let mut dot = 0.0;
            for (j, &tj) in eta.tau.iter().enumerate() {
                if j != q {
                    dot += tj * s[j];
                }
            }
            s[q] = (s[q] - dot) / tq;
        }
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

    /// Back solve `U w = s` (upper), in place.
    fn usolve(&self, s: &mut [f64]) {
        for k in (0..self.m).rev() {
            let sk = s[k] / self.u_diag[k];
            s[k] = sk;
            if sk == 0.0 {
                continue;
            }
            let (lo, hi) = (self.u_col_ptr[k], self.u_col_ptr[k + 1]);
            for idx in lo..hi {
                s[self.u_row_idx[idx]] -= self.u_val[idx] * sk;
            }
        }
    }

    /// Forward solve `Uᵀ z = s` (`Uᵀ` lower), in place.
    fn ut_solve(&self, s: &mut [f64]) {
        for k in 0..self.m {
            let mut acc = s[k];
            let (lo, hi) = (self.u_col_ptr[k], self.u_col_ptr[k + 1]);
            for idx in lo..hi {
                acc -= self.u_val[idx] * s[self.u_row_idx[idx]];
            }
            s[k] = acc / self.u_diag[k];
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
