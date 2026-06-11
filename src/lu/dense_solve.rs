//! Dense `ftran` / `btran` triangular solves and iterative refinement.
//!
//! With `P B Q = L U`, `ftran` solves `B x = a` as `x = Q U⁻¹ L⁻¹ P a`, and
//! `btran` solves `Bᵀ x = a` via `Bᵀ = Q Uᵀ Lᵀ P` with the transposed triangles
//! in reverse order.
//!
//! `ftran_partial` returns the spike `L⁻¹ P a` (no `U`-solve, no column
//! permutation), the input the rank-1 update consumes.

use super::dense_factor::DenseLu;
use super::dense_matrix::GeneralMatrix;
use crate::error::FeralError;

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

impl DenseLu {
    /// Solve `B x = a`, overwriting `rhs` (= `a`) with `x`. Applies scaling
    /// around the core solve on the scaled matrix `Ã = D_row Π B D_col`:
    /// `Ã x̃ = D_row Π a`, then `x = D_col x̃`.
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

    /// Solve `Bᵀ x = a`, overwriting `rhs` (= `a`) with `x`. With scaling:
    /// `Ãᵀ ỹ = D_col a`, then `x[rperm[i]] = D_row[i] ỹ[i]`.
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
        let m = self.m;
        check_len(rhs.len(), m)?;
        let mut s = std::mem::take(&mut self.scratch_a);
        for (k, sk) in s.iter_mut().enumerate() {
            *sk = rhs[self.perm[k]];
        }
        lsolve(&self.l, m, &mut s);
        // Restore the scratch buffer on every path (it was `mem::take`n, so an
        // early `?` would otherwise leave `self.scratch_a` empty and panic the
        // next call). Only write `rhs` on success.
        let res = usolve(&self.u, m, &mut s);
        if res.is_ok() {
            for (k, &wk) in s.iter().enumerate() {
                rhs[self.qcol[k]] = wk;
            }
        }
        self.scratch_a = s;
        res
    }

    /// Core `btran` on the (scaled) factored matrix, ignoring outer scaling.
    pub(super) fn btran_core(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let m = self.m;
        check_len(rhs.len(), m)?;
        let mut s = std::mem::take(&mut self.scratch_a);
        for (k, sk) in s.iter_mut().enumerate() {
            *sk = rhs[self.qcol[k]];
        }
        // Restore scratch on every path; only finish + write `rhs` on success.
        let res = ut_solve(&self.u, m, &mut s); // Uᵀ z = g
        if res.is_ok() {
            lt_solve(&self.l, m, &mut s); // Lᵀ v = z
            for (k, &vk) in s.iter().enumerate() {
                rhs[self.perm[k]] = vk;
            }
        }
        self.scratch_a = s;
        res
    }

    /// Compute the spike `L⁻¹ P a`, overwriting `rhs` (= `a`). This is `ftran`
    /// stopped before the `U`-solve and column permutation; it is the input to
    /// [`DenseLu::update`](crate::lu::DenseLu::update).
    pub fn ftran_partial(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        let m = self.m;
        check_len(rhs.len(), m)?;
        let mut s = std::mem::take(&mut self.scratch_a);
        for (k, sk) in s.iter_mut().enumerate() {
            *sk = rhs[self.perm[k]];
        }
        lsolve(&self.l, m, &mut s);
        rhs.copy_from_slice(&s);
        self.scratch_a = s;
        Ok(())
    }

    /// `ftran` with iterative refinement against the original basis `b`.
    /// Loops `r = a − Bx; Bδ = r; x += δ` up to `params.refine_steps`,
    /// stopping when `‖r‖ / ‖a‖ < params.refine_tol`.
    pub fn ftran_refined(&mut self, b: &GeneralMatrix, rhs: &mut [f64]) -> Result<(), FeralError> {
        let m = self.m;
        check_len(rhs.len(), m)?;
        let mut a = take_zeroed(&mut self.scratch_d, m);
        a.copy_from_slice(rhs);
        let res = match self.ftran(rhs) {
            Ok(()) => refine(self, b, &a, rhs, false),
            Err(e) => Err(e),
        };
        self.scratch_d = a;
        res
    }

    /// `btran` with iterative refinement against the original basis `b`.
    pub fn btran_refined(&mut self, b: &GeneralMatrix, rhs: &mut [f64]) -> Result<(), FeralError> {
        let m = self.m;
        check_len(rhs.len(), m)?;
        let mut a = take_zeroed(&mut self.scratch_d, m);
        a.copy_from_slice(rhs);
        let res = match self.btran(rhs) {
            Ok(()) => refine(self, b, &a, rhs, true),
            Err(e) => Err(e),
        };
        self.scratch_d = a;
        res
    }
}

fn check_len(got: usize, expected: usize) -> Result<(), FeralError> {
    if got != expected {
        Err(FeralError::DimensionMismatch { expected, got })
    } else {
        Ok(())
    }
}

/// Forward solve `L y = s` (unit lower triangular), in place.
fn lsolve(l: &[f64], m: usize, s: &mut [f64]) {
    for k in 0..m {
        let mut acc = s[k];
        for i in 0..k {
            acc -= l[k + i * m] * s[i];
        }
        s[k] = acc;
    }
}

/// Back solve `U w = s` (upper triangular), in place.
///
/// Errors with [`FeralError::SingularBasis`] if a diagonal is zero or
/// non-finite: a degenerate post-update bump pivot can leave `u[k,k] == 0`,
/// and dividing by it would otherwise emit a silent `±Inf`/`NaN`. Mirrors the
/// sparse path (`sparse_solve.rs`). L1, dev/research/repo-review-2026-06-09.md.
fn usolve(u: &[f64], m: usize, s: &mut [f64]) -> Result<(), FeralError> {
    for k in (0..m).rev() {
        let d = u[k + k * m];
        if d == 0.0 || !d.is_finite() {
            return Err(FeralError::SingularBasis { column: k });
        }
        let mut acc = s[k];
        for i in k + 1..m {
            acc -= u[k + i * m] * s[i];
        }
        s[k] = acc / d;
    }
    Ok(())
}

/// Forward solve `Uᵀ z = s` (`Uᵀ` is lower triangular), in place.
///
/// Errors with [`FeralError::SingularBasis`] on a zero/non-finite diagonal,
/// for the same reason as [`usolve`].
fn ut_solve(u: &[f64], m: usize, s: &mut [f64]) -> Result<(), FeralError> {
    for k in 0..m {
        let d = u[k + k * m];
        if d == 0.0 || !d.is_finite() {
            return Err(FeralError::SingularBasis { column: k });
        }
        let mut acc = s[k];
        for i in 0..k {
            acc -= u[i + k * m] * s[i]; // Uᵀ[k,i] = U[i,k]
        }
        s[k] = acc / d;
    }
    Ok(())
}

/// Back solve `Lᵀ v = s` (`Lᵀ` is unit upper triangular), in place.
fn lt_solve(l: &[f64], m: usize, s: &mut [f64]) {
    for k in (0..m).rev() {
        let mut acc = s[k];
        for i in k + 1..m {
            acc -= l[i + k * m] * s[i]; // Lᵀ[k,i] = L[i,k]
        }
        s[k] = acc;
    }
}

/// Shared refinement loop. `transpose = true` refines a `btran` solve.
fn refine(
    lu: &mut DenseLu,
    b: &GeneralMatrix,
    a: &[f64],
    x: &mut [f64],
    transpose: bool,
) -> Result<(), FeralError> {
    let m = lu.m;
    let steps = lu.params.refine_steps;
    let tol = lu.params.refine_tol;
    if steps == 0 {
        return Ok(());
    }
    let anorm = inf_norm(a);
    if anorm == 0.0 {
        return Ok(());
    }
    let mut r = take_zeroed(&mut lu.scratch_c, m);
    let mut result = Ok(());
    for _ in 0..steps {
        // r = a − (B or Bᵀ) x
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
            lu.btran(&mut r)
        } else {
            lu.ftran(&mut r)
        };
        if let Err(e) = step {
            result = Err(e);
            break;
        }
        for (xi, &dxi) in x.iter_mut().zip(r.iter()) {
            *xi += dxi;
        }
    }
    lu.scratch_c = r;
    result
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lu::{LuParams, LuScaling};

    /// L3 (dev/research/repo-review-2026-06-09.md): with scaling enabled the
    /// `ftran`/`btran` wrappers and the refine loop must reuse pooled struct
    /// buffers, not allocate a fresh `vec![0.0; m]` per call. The struct doc
    /// claims "no per-call allocation in solves"; this guards that claim.
    /// `SOLVE_SCRATCH_ALLOCS` counts a (re)allocation only when a pooled buffer
    /// is taken at the wrong length — i.e. it was never sized, or a caller
    /// forgot to restore it — so steady-state must be exactly zero.
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
        let mut lu = DenseLu::factor(&cols, m, params).expect("factor");
        assert!(
            !lu.scale.is_identity(),
            "InfNorm scaling should be non-identity for this matrix"
        );
        let b = GeneralMatrix::from_columns(m, &cols).expect("general matrix");

        reset_solve_scratch_allocs();
        for _ in 0..5 {
            let mut x = vec![1.0, 2.0, 3.0];
            lu.ftran(&mut x).expect("ftran");
            assert!(x.iter().all(|v| v.is_finite()));
            let mut y = vec![3.0, 2.0, 1.0];
            lu.btran(&mut y).expect("btran");
            assert!(y.iter().all(|v| v.is_finite()));
        }
        // Refined solves exercise the residual (`scratch_c`) and the RHS
        // snapshot (`scratch_d`) pools as well.
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

    /// L1 (dev/research/repo-review-2026-06-09.md): a zero `U` diagonal (as a
    /// degenerate post-update bump pivot could leave) must surface as
    /// `SingularBasis`, not a silent `±Inf` out of the dense back-solve divide.
    /// Mirrors the sparse regression `zero_u_diagonal_errors_instead_of_inf`.
    #[test]
    fn dense_zero_u_diagonal_errors_instead_of_inf() {
        let cols = vec![vec![2.0, 0.0], vec![1.0, 3.0]]; // nonsingular 2x2
        let mut lu = DenseLu::factor(&cols, 2, LuParams::default()).expect("factor");

        // Sanity: a clean solve has no NaN/Inf.
        let mut rhs = vec![1.0, 1.0];
        lu.ftran(&mut rhs).expect("clean ftran");
        assert!(rhs.iter().all(|x| x.is_finite()));

        // Corrupt the stored diagonal of pivot position 1 (row 1, col 1 of the
        // column-major 2×2 `U`, index `1 + 1*2 = 3`) to an exact zero.
        lu.u[3] = 0.0;

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
}
