//! Dense unsymmetric LU factorization with threshold partial pivoting.
//!
//! Maintains the invariant `P B Q = L G U`, where `P` is the row permutation
//! from partial pivoting, `Q` is a column permutation (identity after a fresh
//! factorization; a rank-1 update permutes columns), `L` is unit lower
//! triangular, `U` is upper triangular, and `G = E₁⁻¹…Eₜ⁻¹` is the product of
//! the rank-1 updates' Hessenberg-reduction eliminations. `G` is identity after
//! a fresh factorization (`etas` empty), so all one-shot factor/solve behavior
//! is unchanged. `L` and `U` are stored as separate dense `m`×`m` column-major
//! buffers.
//!
//! The update ([`super::dense_update`]) records its eliminations as `etas`
//! rather than folding them into `L`: a Bartels–Golub row interchange (needed
//! to dodge the zero-superdiagonal landmine and bound element growth) is an
//! *unsymmetric* row permutation of `U` that cannot be absorbed into an
//! explicit unit-lower-triangular `L`, so — exactly as on the sparse path
//! ([`super::sparse_update`]) — the base `L` stays fixed and the solves replay
//! the etas between the `L`-solve and the `U`-solve.
//!
//! The factorization kernel itself works in a packed buffer (the classic
//! right-looking LAPACK layout) and is split into `L`/`U` at the end.

use super::scaling::{compute_lu_scale, LuScale};
use super::sparse_matrix::SparseColMatrix;
use super::{LuParams, LuScaling, LuSingularAction, RefactorCause};
use crate::error::FeralError;

/// One elementary operation of a dense rank-1 update's Hessenberg reduction,
/// recorded so it can be replayed on a solve vector (and transposed for btran).
/// Mirrors the sparse [`super::sparse_factor::FtOp`]; the base `L`, `P`, and the
/// prior etas are never relabeled.
#[derive(Debug, Clone, Copy)]
pub(super) enum DenseFtOp {
    /// `s[target] -= mult · s[src]` (elimination of one Hessenberg subdiagonal).
    Axpy {
        target: usize,
        src: usize,
        mult: f64,
    },
    /// Rows `a` and `b` exchanged (a Bartels–Golub interchange); the matching
    /// solve-vector entries swap with them. A transposition is its own
    /// transpose, so btran replays the same swap in the reversed op walk.
    Swap { a: usize, b: usize },
}

impl DenseFtOp {
    /// Apply forward (`s ← Eᵢ s`): the row op exactly as performed on `U`.
    #[inline]
    pub(super) fn apply_forward(&self, s: &mut [f64]) {
        match *self {
            DenseFtOp::Axpy { target, src, mult } => s[target] -= mult * s[src],
            DenseFtOp::Swap { a, b } => s.swap(a, b),
        }
    }

    /// Apply transpose (`s ← Eᵢᵀ s`): axpy flips target/src; swap is unchanged.
    #[inline]
    pub(super) fn apply_transpose(&self, s: &mut [f64]) {
        match *self {
            DenseFtOp::Axpy { target, src, mult } => s[src] -= mult * s[target],
            DenseFtOp::Swap { a, b } => s.swap(a, b),
        }
    }
}

/// Dense LU factorization of a square basis, with rank-1 update support.
#[derive(Debug, Clone)]
pub struct DenseLu {
    pub(super) m: usize,
    /// Unit lower triangular `L`, column-major `m*m` (explicit unit diagonal).
    pub(super) l: Vec<f64>,
    /// Upper triangular `U`, column-major `m*m` (strict lower part is zero).
    pub(super) u: Vec<f64>,
    /// Row permutation: `perm[k]` is the original row now in pivot row `k`
    /// (so `(P a)[k] = a[perm[k]]`).
    pub(super) perm: Vec<usize>,
    /// Inverse of `perm`: `perm_inv[orig_row] = pivot_position`.
    pub(super) perm_inv: Vec<usize>,
    /// Column permutation: `qcol[k]` is the basis slot in column position `k`
    /// (so the factorization's column `k` is basis column `qcol[k]`).
    pub(super) qcol: Vec<usize>,
    /// Inverse of `qcol`: `qcol_inv[slot] = column_position`.
    pub(super) qcol_inv: Vec<usize>,
    /// Rank-1 update eliminations since the last factor/refactor (the `G`
    /// factor of `P B Q = L G U`). Empty after a fresh factor; the solves
    /// replay them between the `L`-solve and the `U`-solve so the base `L`
    /// never needs the unsymmetric relabeling a row interchange would force.
    pub(super) etas: Vec<DenseFtOp>,
    /// Updates applied since the last `factor`/`refactor`.
    pub(super) updates_since_refactor: usize,
    /// Running growth monitor; tripping `params.max_growth` forces a refactor.
    /// Element-growth (‖U‖∞) high-water ratio: the largest `max|U|` seen across
    /// all updates divided by [`Self::u_max0`]. Unlike a max-single-multiplier
    /// monitor this compounds across a chain of updates (L5).
    pub(super) growth: f64,
    /// `max|U|` immediately after the last factor/refactor — the denominator of
    /// the element-growth monitor. Floored away from zero.
    pub(super) u_max0: f64,
    /// `max|A|` of the (scaled) factored matrix — the singularity-tolerance
    /// reference the factor uses. The update anchors its bump-pivot ztol here,
    /// not to `u_max0`, so healthy `O(a_max)` pivots on high-growth bases are
    /// not spuriously rejected (issue #118).
    pub(super) a_max: f64,
    /// Cause + magnitude of the most recent [`Self::update`] that returned
    /// [`FeralError::NeedsRefactor`]. `None` after a fresh factor/refactor;
    /// untouched by a successful update (read only after an `Err`). See issue #95.
    pub(super) last_refactor: Option<(RefactorCause, f64)>,
    pub(super) params: LuParams,
    /// Two-sided scaling of the factored matrix (identity when unscaled).
    pub(super) scale: LuScale,
    /// Reusable length-`m` scratch buffer (no per-call allocation in solves).
    pub(super) scratch_a: Vec<f64>,
    /// Pooled length-`m` buffer for the scaled `ftran`/`btran` wrappers' inner
    /// RHS (`bt`); distinct from `scratch_a`, which the core solve dirties (L3).
    pub(super) scratch_b: Vec<f64>,
    /// Pooled length-`m` residual buffer for iterative refinement (`r`);
    /// distinct from `scratch_a`/`scratch_b`, which the inner solve uses (L3).
    pub(super) scratch_c: Vec<f64>,
    /// Pooled length-`m` buffer holding the refinement's original-RHS snapshot
    /// (`a`); live across the whole refine loop, so it cannot reuse the others.
    pub(super) scratch_d: Vec<f64>,
}

impl DenseLu {
    /// Factor the `m`×`m` basis given by its `m` columns (`cols[j]` is column
    /// `j`, length `m`). Computes `P B = L U` with threshold partial pivoting.
    pub fn factor(cols: &[Vec<f64>], m: usize, params: LuParams) -> Result<Self, FeralError> {
        params.validate()?;
        let (scale, scaled) = compute_scale(cols, m, params.scaling)?;
        let factor_cols: &[Vec<f64>] = scaled.as_deref().unwrap_or(cols);
        let mut packed = vec![0.0; m * m];
        copy_columns_into(&mut packed, factor_cols, m)?;
        // `max|A|` before `factorize_packed` overwrites `packed` in place.
        let a_max = packed.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
        let mut perm: Vec<usize> = (0..m).collect();
        factorize_packed(&mut packed, &mut perm, m, &params)?;
        let (l, u) = split_packed(&packed, m);
        let u_max0 = umax(&u);
        let mut perm_inv = vec![0usize; m];
        for (k, &p) in perm.iter().enumerate() {
            perm_inv[p] = k;
        }
        Ok(DenseLu {
            m,
            l,
            u,
            perm,
            perm_inv,
            qcol: (0..m).collect(),
            qcol_inv: (0..m).collect(),
            etas: Vec::new(),
            updates_since_refactor: 0,
            growth: 1.0,
            u_max0,
            a_max,
            last_refactor: None,
            params,
            scale,
            scratch_a: vec![0.0; m],
            scratch_b: vec![0.0; m],
            scratch_c: vec![0.0; m],
            scratch_d: vec![0.0; m],
        })
    }

    /// Discard all pending updates and re-factor from scratch on fresh columns.
    pub fn refactor(&mut self, cols: &[Vec<f64>]) -> Result<(), FeralError> {
        self.params.validate()?;
        let m = self.m;
        let (scale, scaled) = compute_scale(cols, m, self.params.scaling)?;
        self.scale = scale;
        let factor_cols: &[Vec<f64>] = scaled.as_deref().unwrap_or(cols);
        let mut packed = vec![0.0; m * m];
        copy_columns_into(&mut packed, factor_cols, m)?;
        self.a_max = packed.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
        for (k, p) in self.perm.iter_mut().enumerate() {
            *p = k;
        }
        factorize_packed(&mut packed, &mut self.perm, m, &self.params)?;
        let (l, u) = split_packed(&packed, m);
        self.u_max0 = umax(&u);
        self.l = l;
        self.u = u;
        for (k, &p) in self.perm.iter().enumerate() {
            self.perm_inv[p] = k;
        }
        for (k, (q, qi)) in self
            .qcol
            .iter_mut()
            .zip(self.qcol_inv.iter_mut())
            .enumerate()
        {
            *q = k;
            *qi = k;
        }
        self.etas.clear();
        self.updates_since_refactor = 0;
        self.growth = 1.0;
        self.last_refactor = None;
        Ok(())
    }

    /// Basis dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.m
    }

    /// Number of rank-1 updates applied since the last factor/refactor.
    #[inline]
    pub fn updates_since_refactor(&self) -> usize {
        self.updates_since_refactor
    }

    /// Current ‖U‖∞ element-growth high-water ratio since the last factorize:
    /// the largest `max|U|` seen across all updates divided by [`Self::u_max0`].
    /// A continuous conditioning signal (`1.0` on a fresh factor); tripping
    /// `params.max_growth` is what forces [`DenseLu::update`] to return
    /// `NeedsRefactor`. Exposing it lets a caller pre-empt a growth trip (issue #95).
    #[inline]
    pub fn growth(&self) -> f64 {
        self.growth
    }

    /// Reference `max|U|` captured immediately after the last factor/refactor —
    /// the denominator of [`Self::growth`]. Floored away from zero.
    #[inline]
    pub fn u_max0(&self) -> f64 {
        self.u_max0
    }

    /// Cause + magnitude of the most recent [`Self::update`] that returned
    /// [`FeralError::NeedsRefactor`], or `None` if no update has failed since the
    /// last factor/refactor. `update()` itself still returns the payload-free
    /// `Err(NeedsRefactor)`; this accessor carries the *why* (issue #95).
    ///
    /// The `f64` is a cause-specific magnitude: the growth ratio ([`RefactorCause::Growth`]),
    /// the update count that hit the cap ([`RefactorCause::UpdateBudget`]), or the
    /// offending `|pivot|` ([`RefactorCause::TinyPivot`]). The dense path never
    /// reports [`RefactorCause::Singular`] — a dependent replacement surfaces as
    /// `TinyPivot` (the final `U` diagonal collapses to ~0).
    #[inline]
    pub fn last_refactor(&self) -> Option<(RefactorCause, f64)> {
        self.last_refactor
    }

    /// Advisory (cost-based): has the update chain since the last factor/refactor
    /// cost about as much as a fresh factorization? A dense update is `O(m²)`
    /// (Hessenberg elimination plus the `O(m²)` factor clones) and a fresh factor
    /// is `O(m³)`, so ~`m` updates cost about one refactor. True once
    /// [`Self::updates_since_refactor`] `>= m`. Purely advisory: it does **not**
    /// change [`Self::update`]'s behaviour. The dense analogue of
    /// [`SparseLu::should_refactor`](super::SparseLu::should_refactor) (issue #95).
    #[inline]
    pub fn should_refactor(&self) -> bool {
        self.updates_since_refactor >= self.m
    }

    /// Advisory (growth-aware): is the element-growth high-water close enough to
    /// [`LuParams::max_growth`] that the next update is at risk of tripping it?
    /// True once [`Self::growth`] reaches the geometric midpoint (log-space)
    /// between `1` and `max_growth`, i.e. `growth >= sqrt(max_growth)` (only when
    /// `max_growth` is finite and `> 1`). Lets a caller refactor pre-emptively
    /// instead of discovering the trip on the update that fails (issue #95).
    #[inline]
    pub fn should_refactor_growth(&self) -> bool {
        let cap = self.params.max_growth;
        cap.is_finite() && cap > 1.0 && self.growth >= cap.sqrt()
    }

    /// The row permutation `perm` (`perm[k]` = original row in pivot row `k`).
    #[inline]
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// The column permutation `qcol` (`qcol[k]` = basis slot in column `k`).
    #[inline]
    pub fn qcol(&self) -> &[usize] {
        &self.qcol
    }

    /// `L[i,j]`: unit lower triangular.
    #[inline]
    pub fn l(&self, i: usize, j: usize) -> f64 {
        self.l[i + j * self.m]
    }

    /// `U[i,j]`: upper triangular.
    #[inline]
    pub fn u(&self, i: usize, j: usize) -> f64 {
        self.u[i + j * self.m]
    }
}

/// `(scale, optional scaled columns to factor)`.
type ScaleResult = Result<(LuScale, Option<Vec<Vec<f64>>>), FeralError>;

/// Compute the scaling for `cols` under `strategy`, returning the scale and
/// (when non-identity) the scaled columns to factor.
fn compute_scale(cols: &[Vec<f64>], m: usize, strategy: LuScaling) -> ScaleResult {
    if strategy == LuScaling::None {
        return Ok((LuScale::identity(m), None));
    }
    let b = SparseColMatrix::from_dense_columns(m, cols)?;
    let scale = compute_lu_scale(&b, strategy)?;
    let scaled = scale.scaled_dense_columns(&b);
    Ok((scale, Some(scaled)))
}

/// Copy `m` columns (each length `m`) into a packed column-major buffer.
fn copy_columns_into(buf: &mut [f64], cols: &[Vec<f64>], m: usize) -> Result<(), FeralError> {
    if cols.len() != m {
        return Err(FeralError::DimensionMismatch {
            expected: m,
            got: cols.len(),
        });
    }
    for (j, col) in cols.iter().enumerate() {
        if col.len() != m {
            return Err(FeralError::DimensionMismatch {
                expected: m,
                got: col.len(),
            });
        }
        buf[j * m..j * m + m].copy_from_slice(col);
        if col.iter().any(|x| !x.is_finite()) {
            return Err(FeralError::InvalidInput(
                "LU basis column contains non-finite entries".to_string(),
            ));
        }
    }
    Ok(())
}

/// Split a packed LU buffer into explicit `L` (unit lower) and `U` (upper).
fn split_packed(packed: &[f64], m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut l = vec![0.0; m * m];
    let mut u = vec![0.0; m * m];
    for j in 0..m {
        for i in 0..m {
            let v = packed[i + j * m];
            if i > j {
                l[i + j * m] = v;
            } else {
                u[i + j * m] = v;
            }
        }
        l[j + j * m] = 1.0;
    }
    (l, u)
}

/// `max|U|` over the packed column-major buffer, floored away from zero so it
/// is a safe denominator for the element-growth monitor (L5).
#[inline]
fn umax(u: &[f64]) -> f64 {
    u.iter()
        .fold(0.0_f64, |a, &x| a.max(x.abs()))
        .max(f64::MIN_POSITIVE)
}

/// Right-looking outer-product LU with threshold partial pivoting, in place on
/// the packed buffer. `perm` starts as identity and accumulates row swaps.
pub(super) fn factorize_packed(
    packed: &mut [f64],
    perm: &mut [usize],
    m: usize,
    params: &LuParams,
) -> Result<(), FeralError> {
    let u = params.pivot_threshold;
    // L6 (dev/research/repo-review-2026-06-09.md): scale the zero-pivot tolerance
    // by the matrix magnitude. An absolute `zero_pivot_tol` declared a uniformly
    // small but perfectly conditioned basis (e.g. `diag(1e-14)`) singular and
    // gave a large-magnitude basis effectively no singularity detection. The
    // relative tolerance `zero_pivot_tol · max|A|` tracks the basis scale; for a
    // zero matrix `a_max == 0`, so only an exact-zero pivot is rejected — still
    // correct, since the zero matrix is singular.
    let a_max = packed.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
    let ztol = params.zero_pivot_tol * a_max;
    for k in 0..m {
        // Pivot search: max-magnitude entry in column k over rows k..m.
        let mut amax = 0.0_f64;
        let mut argmax = k;
        for i in k..m {
            let v = packed[i + k * m].abs();
            if v > amax {
                amax = v;
                argmax = i;
            }
        }
        // Threshold partial pivoting: keep the diagonal if it is within
        // threshold of the column max (and nonzero), else take the row max.
        let diag = packed[k + k * m].abs();
        let pivot_row = if diag >= u * amax && diag > ztol {
            k
        } else {
            argmax
        };

        if pivot_row != k {
            swap_rows(packed, k, pivot_row, m);
            perm.swap(k, pivot_row);
        }

        // Singularity / perturbation.
        let mut piv = packed[k + k * m];
        if piv.abs() <= ztol {
            match params.on_singular {
                LuSingularAction::Fail => {
                    return Err(FeralError::SingularBasis { column: k });
                }
                LuSingularAction::PerturbToEps { abs_floor } => {
                    let s = if piv < 0.0 { -1.0 } else { 1.0 };
                    piv = s * abs_floor.max(piv.abs());
                    packed[k + k * m] = piv;
                }
            }
        }

        // Multipliers L(k+1:, k) = A(k+1:, k) / piv.
        let inv = 1.0 / piv;
        for i in k + 1..m {
            packed[i + k * m] *= inv;
        }
        // Rank-1 trailing update A(k+1:, k+1:) -= L(k+1:, k) · U(k, k+1:).
        for j in k + 1..m {
            let ukj = packed[k + j * m];
            if ukj != 0.0 {
                for i in k + 1..m {
                    packed[i + j * m] -= packed[i + k * m] * ukj;
                }
            }
        }
    }
    Ok(())
}

/// Swap rows `a` and `b` across all `m` columns of the packed buffer.
fn swap_rows(buf: &mut [f64], a: usize, b: usize, m: usize) {
    for c in 0..m {
        buf.swap(a + c * m, b + c * m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Columns (column-major) from rows-of-the-matrix, for readable tests.
    fn cols_from_rows(rows: &[&[f64]]) -> (Vec<Vec<f64>>, usize) {
        let m = rows.len();
        let mut cols = vec![vec![0.0; m]; m];
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                cols[j][i] = v;
            }
        }
        (cols, m)
    }

    /// max_{i,j} | (P B)[i,j] - (L U)[i,j] |, the reconstruction residual.
    /// (Q = I after a fresh factorization.)
    fn reconstruction_residual(lu: &DenseLu, rows: &[&[f64]]) -> f64 {
        let m = lu.m;
        let mut worst = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                let pb = rows[lu.perm[i]][j];
                let mut prod = 0.0;
                for k in 0..m {
                    prod += lu.l(i, k) * lu.u(k, j);
                }
                worst = worst.max((pb - prod).abs());
            }
        }
        worst
    }

    #[test]
    fn factor_2x2_swap_exact() {
        // B = [[0,2],[3,4]]; partial pivoting swaps rows: P=[1,0], L=I,
        // U=[[3,4],[0,2]]. Hand-computed.
        let (cols, m) = cols_from_rows(&[&[0.0, 2.0], &[3.0, 4.0]]);
        let lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
        assert_eq!(lu.perm, vec![1, 0]);
        assert!((lu.u(0, 0) - 3.0).abs() < 1e-14);
        assert!((lu.u(0, 1) - 4.0).abs() < 1e-14);
        assert!((lu.u(1, 1) - 2.0).abs() < 1e-14);
        assert!((lu.l(1, 0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn factor_3x3_reconstruction() {
        let rows: [&[f64]; 3] = [&[2.0, 1.0, 1.0], &[4.0, 3.0, 3.0], &[8.0, 7.0, 9.0]];
        let (cols, m) = cols_from_rows(&rows);
        let lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
        assert!(reconstruction_residual(&lu, &rows) < 1e-12);
    }

    #[test]
    fn factor_random_reconstruction() {
        let m = 7;
        let mut cols = vec![vec![0.0; m]; m];
        let mut state = 0x1234_5678_u64;
        for j in 0..m {
            for i in 0..m {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
                cols[j][i] = r;
            }
            cols[j][j] += 5.0;
        }
        let rows_owned: Vec<Vec<f64>> = (0..m)
            .map(|i| (0..m).map(|j| cols[j][i]).collect())
            .collect();
        let rows: Vec<&[f64]> = rows_owned.iter().map(|r| r.as_slice()).collect();
        let lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
        assert!(reconstruction_residual(&lu, &rows) < 1e-10);
    }

    #[test]
    fn factor_singular_repeated_columns_fails() {
        let (cols, m) = cols_from_rows(&[&[1.0, 1.0], &[2.0, 2.0]]);
        let err = DenseLu::factor(&cols, m, LuParams::default());
        assert!(matches!(err, Err(FeralError::SingularBasis { .. })));
    }

    #[test]
    fn factor_singular_perturb_succeeds() {
        let (cols, m) = cols_from_rows(&[&[1.0, 1.0], &[2.0, 2.0]]);
        let params = LuParams {
            on_singular: LuSingularAction::PerturbToEps { abs_floor: 1e-10 },
            ..LuParams::default()
        };
        let lu = DenseLu::factor(&cols, m, params).expect("perturbed factor");
        assert!(lu.u(1, 1).abs() >= 1e-10);
    }

    /// L6 (dev/research/repo-review-2026-06-09.md): the zero-pivot test compared
    /// the pivot against the absolute `zero_pivot_tol` (1e-13). `diag(1e-14)` is
    /// perfectly conditioned — cond₂ = 1, exact inverse `diag(1e14)` — yet every
    /// pivot 1e-14 ≤ 1e-13, so it was declared `SingularBasis { column: 0 }` even
    /// though it is trivially invertible. The fix scales the tolerance by the
    /// matrix magnitude (`zero_pivot_tol · max|A|`), so a uniformly small but
    /// well-conditioned basis factors. Oracle: the hand-computed exact solution
    /// of `B x = b`. Pre-fix this `expect` panics on `SingularBasis`.
    #[test]
    fn factor_tiny_well_conditioned_basis_not_singular() {
        let s = 1e-14;
        let (cols, m) = cols_from_rows(&[&[s, 0.0], &[0.0, s]]);
        let mut lu = DenseLu::factor(&cols, m, LuParams::default())
            .expect("tiny but well-conditioned basis must factor");
        // B = s·I, b = s·[1, 2]  =>  x = [1, 2] exactly.
        let mut rhs = vec![s, 2.0 * s];
        lu.ftran(&mut rhs).expect("ftran");
        assert!((rhs[0] - 1.0).abs() < 1e-6, "x0 = {}", rhs[0]);
        assert!((rhs[1] - 2.0).abs() < 1e-6, "x1 = {}", rhs[1]);
    }

    #[test]
    fn refactor_resets_state() {
        let (cols, m) = cols_from_rows(&[&[2.0, 1.0], &[1.0, 3.0]]);
        let mut lu = DenseLu::factor(&cols, m, LuParams::default()).expect("factor");
        let (cols2, _) = cols_from_rows(&[&[4.0, 0.0], &[1.0, 5.0]]);
        lu.refactor(&cols2).expect("refactor");
        assert_eq!(lu.updates_since_refactor(), 0);
        let rows2: [&[f64]; 2] = [&[4.0, 0.0], &[1.0, 5.0]];
        assert!(reconstruction_residual(&lu, &rows2) < 1e-12);
    }
}
