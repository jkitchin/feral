//! Symbolic analysis for the sparse LU: the fill-reducing column ordering `Q`.
//!
//! **Since 0.16.0 nothing in this module is on the default factorization path.**
//! [`super::LuParams::pivoting`] defaults to [`super::LuPivoting::Markowitz`],
//! which chooses its column order *during* the numeric factorization and reads
//! the symbolic it is handed only to check its dimension. Everything below —
//! both orderings and the tradeoff between them — applies to
//! [`super::SparseLu::factor`] only under
//! `LuParams { pivoting: LuPivoting::GilbertPeierls, ..Default::default() }`
//! (issue #171). [`super::SparseLu::used_markowitz`] reports which rule ran.
//!
//! Two orderings, and the caller picks:
//!
//! - [`SparseLuSymbolic::analyze`] — the **default of the two**: feral's in-tree AMD
//!   (`feral_amd`) on the whole basis's `AᵀA` (column-intersection) pattern, a
//!   stand-in for COLAMD that needs no new ordering algorithm.
//! - [`SparseLuSymbolic::analyze_triangularized`] — **opt-in**: first peel
//!   column and row singletons to fixpoint ([`super::sparse_triangular`]), whose
//!   pivots are structurally forced and whose blocks are upper triangular, then
//!   AMD over the residual bump only. Running AMD on the bump is cheaper (AMD is
//!   superlinear in the matrix it is handed, and a simplex basis is typically
//!   85–100% peelable), and the peel leaves the bump *contiguous*, which is what
//!   [`super::LuParams::dense_bump_max_dim`] needs.
//!
//! The peel was briefly the default and was reverted to opt-in by issue #163.
//! It is the **faster** of the two — 4.2–9.8x on this step, which a simplex pays
//! on every refactorization — and also a rounding-trajectory change that cost one
//! ill-conditioned LP its dual bound. Neither ordering dominates; see
//! [`SparseLuSymbolic::analyze`] for the measurements on both sides.
//!
//! Both are reachable as one call — [`SparseLuSymbolic::analyze_with`] taking
//! [`LuOrderingParams`] — so a caller can carry the choice in its own config and
//! A/B it on its own instances, which is the only way to settle a tradeoff this
//! instance-dependent (issue #165). The two constructors remain as names for the
//! two settings.
//!
//! The resulting permutation is the reusable symbolic handle: across numerically
//! different but structurally identical bases, only the numeric factor is
//! recomputed. Every stage above reads only the pattern, so that contract holds
//! for both orderings.

use super::sparse_matrix::SparseColMatrix;
use super::sparse_triangular::triangularize;
use crate::error::FeralError;

/// Which fill-reducing column ordering [`SparseLuSymbolic::analyze_with`]
/// computes (issue #165).
///
/// The ordering is chosen at *symbolic* time, so it cannot live in
/// [`super::LuParams`], which is consumed at factor time. It gets its own
/// struct rather than a second constructor name because the right setting
/// varies by instance — it is worth 1.306x geomean across 14 QPLIB relaxations
/// under a dual simplex and 0.389x on one of them — and a choice that has to be
/// measured per caller has to be reachable as a value, not only as a call site.
///
/// [`Default`] is whole-basis AMD, matching [`SparseLuSymbolic::analyze`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LuOrderingParams {
    /// Peel the triangular border (Suhl–Suhl) before ordering, and hand AMD only
    /// the residual bump.
    ///
    /// Much cheaper — 4.2–9.8x on this step across the in-tree fixtures, which a
    /// simplex pays on *every* refactorization — and the precondition for
    /// [`super::LuParams::dense_bump_max_dim`]. It is also a different rounding
    /// trajectory, which cost an ill-conditioned LP its dual bound (issue #163)
    /// and one QPLIB relaxation a 2.6x slowdown. Neither setting dominates; see
    /// [`SparseLuSymbolic::analyze`] for the measurements on both sides.
    pub triangularize: bool,
}

/// Reusable symbolic factorization: the column permutation `Q`.
#[derive(Debug, Clone)]
pub struct SparseLuSymbolic {
    /// Dimension.
    pub m: usize,
    /// Column order: factorization column position `k` is original column
    /// `qcol[k]`.
    pub qcol: Vec<usize>,
    /// Inverse: `qcol_inv[original_col] = column_position`.
    pub qcol_inv: Vec<usize>,
    /// First pivot position of the residual bump after triangularization.
    /// Positions `< bump_lo` are column singletons; positions `>= bump_hi` are
    /// row singletons. Both peeled blocks factor exactly, with an empty `L`
    /// column and no arithmetic.
    pub bump_lo: usize,
    /// One past the last pivot position of the residual bump.
    pub bump_hi: usize,
    /// Whether triangularization actually ran, i.e. whether `bump_lo`/`bump_hi`
    /// are a *measured* peel rather than the "no structure known" default.
    ///
    /// [`Self::analyze_triangularized`] sets this; [`Self::analyze`],
    /// [`Self::natural`], [`Self::with_order`] and [`Self::analyze_amd_only`]
    /// do not — they claim `(0, m)` because they never looked, not because they
    /// looked and found the whole basis irreducible. The two cases are
    /// indistinguishable from the indices alone, and they warrant opposite
    /// answers from
    /// [`LuParams::dense_bump_max_dim`](super::LuParams::dense_bump_max_dim):
    /// an `analyze_triangularized` basis that peels to nothing really is a dense
    /// block worth the dense kernel, while an unpeeled `natural` ordering of a
    /// large sparse basis is the pathological case that route must never take.
    pub triangularized: bool,
}

impl SparseLuSymbolic {
    /// AMD over the whole basis. The default ordering of the two this module
    /// offers, and the one to use unless you are opting into the dense-bump
    /// route.
    ///
    /// # Since 0.16.0 this is not what `SparseLu::factor` does by default
    ///
    /// [`super::LuParams::pivoting`] defaults to
    /// [`super::LuPivoting::Markowitz`], which picks each pivot column during
    /// the factorization and never consults the ordering computed here. Under
    /// the shipped defaults, calling `analyze` and passing the result to
    /// [`super::SparseLu::factor`] costs the AMD run and changes nothing about
    /// the factor produced. To get this ordering, ask for it:
    /// `LuParams { pivoting: LuPivoting::GilbertPeierls, ..Default::default() }`.
    /// [`super::SparseLu::used_markowitz`] reports which rule actually ran, so a
    /// caller need not infer it.
    ///
    /// The rest of this comment weighs whole-basis AMD against the peel. That
    /// comparison is unchanged and still decides
    /// [`Self::analyze`]-vs-[`Self::analyze_triangularized`] — but it is a
    /// comparison *within* the Gilbert-Peierls route, not a description of what
    /// feral does out of the box. On the 16-basis LP corpus Markowitz beat
    /// whole-basis AMD on fill by 2.77x -> 1.06x geomean, which is why the
    /// default moved (issue #171, `CHANGELOG.md` 0.16.0).
    ///
    /// # Why this does not triangularize
    ///
    /// feral 0.16.0-dev briefly made this a Suhl–Suhl peel plus AMD over the
    /// residual bump. That ordering was reverted to opt-in
    /// ([`Self::analyze_triangularized`]) after issue #163: on an
    /// ill-conditioned LP it turned a solve that certified `Optimal` into
    /// `Numerical` — a *lost dual bound*, with `dense_bump_max_dim` at its
    /// default of `0`, i.e. with the route the peel exists to enable switched
    /// off.
    ///
    /// **The peel is not the less accurate ordering, and that is not why it was
    /// reverted.** Every basis that LP's simplex handed feral was dumped and
    /// re-factored both ways. Backward error is ~1e-16 under both orderings on
    /// all of them; forward error against a known solution reaches 2.6e-11 (the
    /// basis really is ill-conditioned) and the peel is **never the worse of
    /// the two** — its ratio to whole-basis AMD runs 0.0x–1.0x across all 30
    /// bases of the failing run. What differs is the *trajectory*: at that
    /// forward error the two orderings' solves disagree in exactly the bits the
    /// simplex's ratio test reads, the two runs diverge onto different pivot
    /// sequences, and this LP is conditioned badly enough that one path
    /// certifies and the other trips the caller's numerical guard.
    ///
    /// # What this default costs, and why it is still the default
    ///
    /// **This ordering is substantially slower than the peel, and the cost is
    /// paid on every refactorization.** A simplex re-runs `analyze` each time it
    /// refactorizes, so the symbolic cost is multiplied by the refactorization
    /// count rather than amortized across it. On the in-tree fixtures
    /// (`tests/data/lu_bases`, 20 reps, release, `examples/basis_refactor.rs`):
    ///
    /// | basis | `analyze` | `analyze_triangularized` | symbolic | total |
    /// |---|---|---|---|---|
    /// | QPLIB_1157 (m=3937) | 21.28 ms | 2.17 ms | **9.8x** | 1.03x |
    /// | QPLIB_3852 (m=1760) | 0.86 ms | 0.13 ms | **6.6x** | 2.73x |
    /// | bchoco06 (m=833) | 0.54 ms | 0.13 ms | **4.2x** | 2.26x |
    ///
    /// Measured end to end by the maintainer across 14 QPLIB relaxations under a
    /// dual simplex, switching only the constructor is **1.306x geomean** (max
    /// 1.674x) — the largest single effect on the PR that made this change.
    ///
    /// **An earlier version of this comment claimed the peel had "no standalone
    /// payoff", from the numeric-factorization column alone (97.45 vs 101.40 ms
    /// on QPLIB_1157). That was wrong**: it ignored the symbolic column, where
    /// the peel wins by 4–10x, and generalized a 1.03x total from the one
    /// fixture where the peel's total advantage is smallest. The evidence was
    /// already in this repository — `CHANGELOG.md` records #160 cutting the
    /// ordering from 9.837 ms to 0.851 ms on a real basis.
    ///
    /// So this is a **genuine tradeoff, not a free revert**, and the peel is
    /// opt-in because the caller has to make it — not because it costs nothing
    /// to give up. Against the speedup: the peel is a different rounding
    /// trajectory, it lost an ill-conditioned LP its dual bound (above), and on
    /// QPLIB_2055 it is **0.389x** — a 2.6x *slowdown* — with the objective
    /// moving in the 9th significant figure, so it changed that pivot path too.
    /// Whole-basis AMD is the default because it is the trajectory that was in
    /// place when the downstream suite was green, not because it is faster or
    /// more accurate. It is neither, reliably.
    ///
    /// Take [`Self::analyze_triangularized`] if you can afford to qualify your
    /// own instances against it — and take it together with
    /// [`LuParams::dense_bump_max_dim`], which needs it and is worth a further
    /// 4.28x on the numeric side.
    /// Equivalent to [`Self::analyze_with`] at [`LuOrderingParams::default`].
    pub fn analyze(a: &SparseColMatrix) -> Result<Self, FeralError> {
        Self::analyze_with(a, LuOrderingParams::default())
    }

    /// Analyze under an explicit [`LuOrderingParams`] (issue #165).
    ///
    /// The parameter form of [`Self::analyze`] / [`Self::analyze_triangularized`],
    /// for callers that want the ordering to be a setting they can flip and
    /// measure rather than a constructor they have to choose at the call site.
    /// Both of those are thin wrappers over this.
    ///
    /// Since 0.16.0 this setting is only observable under
    /// `LuParams { pivoting: LuPivoting::GilbertPeierls, .. }`; the default
    /// [`super::LuPivoting::Markowitz`] orders columns itself. See
    /// [`Self::analyze`].
    pub fn analyze_with(a: &SparseColMatrix, params: LuOrderingParams) -> Result<Self, FeralError> {
        if params.triangularize {
            Self::analyze_triangularized(a)
        } else {
            Self::analyze_amd_only(a)
        }
    }

    /// Triangularize (Suhl–Suhl peel), then order the residual bump with AMD.
    ///
    /// **Usually much faster than [`Self::analyze`]**, because AMD is
    /// superlinear in the matrix it is handed and this hands it only the bump:
    /// 4.2–9.8x on the symbolic step across the in-tree fixtures, 1.03–2.73x on
    /// symbolic + numeric together, and 1.306x geomean end to end across 14
    /// QPLIB relaxations under a dual simplex. A simplex pays the symbolic cost
    /// on every refactorization, so that is not amortized away. See
    /// [`Self::analyze`] for the table.
    ///
    /// It is also the ordering [`LuParams::dense_bump_max_dim`] requires: it
    /// makes the bump a contiguous block the dense kernel can be handed, worth a
    /// further 4.28x on the numeric side. Take both.
    ///
    /// **Why it is nonetheless not the default.** It is a different rounding
    /// trajectory, and that is not free even though it is not less *accurate*
    /// (see [`Self::analyze`] for the error measurements). On an ill-conditioned
    /// LP it changed a downstream simplex's pivot choices and lost that solve's
    /// dual bound (issue #163), and on QPLIB_2055 it is **0.389x** — a 2.6x
    /// slowdown — because the path it took was longer. Use it when you can
    /// qualify your own instances against it; that is a real prerequisite, not
    /// boilerplate.
    ///
    /// Since 0.16.0 this ordering reaches the factor only under
    /// `LuParams { pivoting: LuPivoting::GilbertPeierls, .. }` — the default
    /// [`super::LuPivoting::Markowitz`] ignores the symbolic, which also makes
    /// [`super::LuParams::dense_bump_max_dim`] unreachable without that pin.
    /// See [`Self::analyze`].
    ///
    /// Equivalent to [`Self::analyze_with`] at
    /// `LuOrderingParams { triangularize: true }`.
    pub fn analyze_triangularized(a: &SparseColMatrix) -> Result<Self, FeralError> {
        let m = a.m;
        if m == 0 {
            return Ok(Self::empty());
        }
        let t = triangularize(a);
        let bump = t.bump_hi - t.bump_lo;

        let mut qcol = t.cols;
        if bump > 1 {
            // Order only the bump. `sub` is the bump in local 0..bump
            // coordinates; AMD's permutation is mapped back to original columns.
            let sub = submatrix(a, &qcol[t.bump_lo..t.bump_hi], &t.bump_rows);
            let perm = amd_permutation(&sub)?;
            let local: Vec<usize> = qcol[t.bump_lo..t.bump_hi].to_vec();
            for (n, &p) in perm.iter().enumerate() {
                qcol[t.bump_lo + n] = local[p];
            }
        }

        Ok(Self::from_qcol(m, qcol, t.bump_lo, t.bump_hi, true))
    }

    /// AMD over the whole basis, with no triangularization. Identical to
    /// [`Self::analyze`]; retained as an explicit name for benchmark arms that
    /// compare the two orderings side by side. Such an arm must pin
    /// `LuParams { pivoting: LuPivoting::GilbertPeierls, .. }` since 0.16.0, or
    /// both arms factor the same Markowitz ordering and the comparison is
    /// vacuous — see [`Self::analyze`].
    pub fn analyze_amd_only(a: &SparseColMatrix) -> Result<Self, FeralError> {
        let m = a.m;
        if m == 0 {
            return Ok(Self::empty());
        }
        let qcol = amd_permutation(a)?;
        Ok(Self::from_qcol(m, qcol, 0, m, false))
    }

    /// Identity column ordering (natural order) — for testing and as a fallback.
    pub fn natural(m: usize) -> Self {
        Self::from_qcol(m, (0..m).collect(), 0, m, false)
    }

    /// Build a handle from an explicit column order, claiming no triangular
    /// structure (the whole matrix is treated as bump). For callers supplying
    /// their own ordering; `qcol` must be a permutation of `0..m`.
    pub fn with_order(m: usize, qcol: Vec<usize>) -> Result<Self, FeralError> {
        if qcol.len() != m {
            return Err(FeralError::DimensionMismatch {
                expected: m,
                got: qcol.len(),
            });
        }
        let mut seen = vec![false; m];
        for &q in qcol.iter() {
            if q >= m || seen[q] {
                return Err(FeralError::InvalidInput(
                    "column order is not a permutation".to_string(),
                ));
            }
            seen[q] = true;
        }
        Ok(Self::from_qcol(m, qcol, 0, m, false))
    }

    fn empty() -> Self {
        SparseLuSymbolic {
            m: 0,
            qcol: Vec::new(),
            qcol_inv: Vec::new(),
            bump_lo: 0,
            bump_hi: 0,
            triangularized: true,
        }
    }

    fn from_qcol(
        m: usize,
        qcol: Vec<usize>,
        bump_lo: usize,
        bump_hi: usize,
        triangularized: bool,
    ) -> Self {
        let mut qcol_inv = vec![0usize; m];
        for (k, &q) in qcol.iter().enumerate() {
            qcol_inv[q] = k;
        }
        SparseLuSymbolic {
            m,
            qcol,
            qcol_inv,
            bump_lo,
            bump_hi,
            triangularized,
        }
    }
}

/// AMD on the `AᵀA` (column-intersection) pattern of `a`.
fn amd_permutation(a: &SparseColMatrix) -> Result<Vec<usize>, FeralError> {
    let m = a.m;
    let pat = a.ata_pattern();
    let col_ptr: Vec<i32> = pat
        .col_ptr
        .iter()
        .map(|&x| i32::try_from(x))
        .collect::<Result<_, _>>()
        .map_err(|_| FeralError::InvalidInput("AᵀA pattern index overflow".to_string()))?;
    let row_idx: Vec<i32> = pat
        .row_idx
        .iter()
        .map(|&x| i32::try_from(x))
        .collect::<Result<_, _>>()
        .map_err(|_| FeralError::InvalidInput("AᵀA pattern index overflow".to_string()))?;
    let cpat = feral_ordering_core::CscPattern::new(m, &col_ptr, &row_idx)
        .ok_or_else(|| FeralError::InvalidInput("malformed AᵀA pattern".to_string()))?;
    let perm = feral_amd::amd_order(&cpat)
        .map_err(|e| FeralError::InvalidInput(format!("AMD ordering failed: {:?}", e)))?;
    Ok(perm.iter().map(|&x| x as usize).collect())
}

/// Extract the square submatrix `a[rows, cols]` in local `0..cols.len()`
/// coordinates. `rows` must be ascending; `cols` may be in any order.
fn submatrix(a: &SparseColMatrix, cols: &[usize], rows: &[usize]) -> SparseColMatrix {
    let b = cols.len();
    let mut local_row = vec![usize::MAX; a.m];
    for (n, &i) in rows.iter().enumerate() {
        local_row[i] = n;
    }
    let mut col_ptr = Vec::with_capacity(b + 1);
    let mut row_idx: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    col_ptr.push(0);
    for &j in cols.iter() {
        let start = row_idx.len();
        for idx in a.col_ptr[j]..a.col_ptr[j + 1] {
            let li = local_row[a.row_idx[idx]];
            if li != usize::MAX {
                row_idx.push(li);
                values.push(a.values[idx]);
            }
        }
        // `rows` is ascending and the source column is row-ascending, so the
        // emitted slice is already ascending — `SparseColMatrix`'s invariant.
        debug_assert!(row_idx[start..].windows(2).all(|w| w[0] < w[1]));
        col_ptr.push(row_idx.len());
    }
    SparseColMatrix {
        m: b,
        col_ptr,
        row_idx,
        values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(m: usize, entries: &[(usize, usize, f64)]) -> SparseColMatrix {
        let mut trip = entries.to_vec();
        trip.sort_by_key(|&(i, j, _)| (j, i));
        let mut col_ptr = vec![0usize; m + 1];
        for &(_, j, _) in &trip {
            col_ptr[j + 1] += 1;
        }
        for j in 0..m {
            col_ptr[j + 1] += col_ptr[j];
        }
        SparseColMatrix {
            m,
            col_ptr,
            row_idx: trip.iter().map(|&(i, _, _)| i).collect(),
            values: trip.iter().map(|&(_, _, v)| v).collect(),
        }
    }

    #[test]
    fn qcol_is_a_permutation_and_inverse_agrees() {
        let a = mat(
            5,
            &[
                (0, 0, 2.0),
                (1, 0, 1.0),
                (1, 1, 3.0),
                (2, 2, 1.0),
                (2, 3, 4.0),
                (3, 2, 5.0),
                (3, 3, 1.0),
                (4, 4, 7.0),
                (4, 2, 2.0),
            ],
        );
        let s = SparseLuSymbolic::analyze(&a).expect("analyze");
        assert_eq!(s.m, 5);
        let mut seen = [false; 5];
        for &j in s.qcol.iter() {
            assert!(!seen[j]);
            seen[j] = true;
        }
        assert!(seen.iter().all(|&x| x));
        for (k, &q) in s.qcol.iter().enumerate() {
            assert_eq!(s.qcol_inv[q], k);
        }
        assert!(s.bump_lo <= s.bump_hi && s.bump_hi <= 5);
    }

    #[test]
    fn triangular_basis_has_an_empty_bump() {
        let m = 8;
        let e: Vec<(usize, usize, f64)> = (0..m)
            .flat_map(|j| (j..m).map(move |i| (i, j, if i == j { 2.0 } else { 1.0 })))
            .collect();
        let a = mat(m, &e);
        // `analyze_triangularized`, not `analyze`: since issue #163 only the
        // opt-in constructor peels.
        let s = SparseLuSymbolic::analyze_triangularized(&a).expect("analyze");
        assert_eq!(
            s.bump_hi - s.bump_lo,
            0,
            "triangular basis must leave no bump"
        );
    }

    #[test]
    fn amd_only_marks_the_whole_matrix_as_bump() {
        let m = 6;
        let e: Vec<(usize, usize, f64)> = (0..m)
            .flat_map(|j| (j..m).map(move |i| (i, j, if i == j { 2.0 } else { 1.0 })))
            .collect();
        let a = mat(m, &e);
        let s = SparseLuSymbolic::analyze_amd_only(&a).expect("analyze");
        assert_eq!((s.bump_lo, s.bump_hi), (0, m));
    }

    #[test]
    fn submatrix_extracts_the_bump() {
        let a = mat(
            4,
            &[
                (0, 0, 1.0),
                (1, 1, 2.0),
                (2, 1, 3.0),
                (1, 2, 4.0),
                (2, 2, 5.0),
                (3, 3, 6.0),
            ],
        );
        let s = submatrix(&a, &[1, 2], &[1, 2]);
        assert_eq!(s.m, 2);
        assert_eq!(s.col_ptr, vec![0, 2, 4]);
        assert_eq!(s.row_idx, vec![0, 1, 0, 1]);
        assert_eq!(s.values, vec![2.0, 3.0, 4.0, 5.0]);
    }
}
