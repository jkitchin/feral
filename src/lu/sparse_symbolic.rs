//! Symbolic analysis for the sparse LU: the fill-reducing column ordering `Q`.
//!
//! Two stages, in the order every production LP INVERT runs them:
//!
//! 1. **Triangularization** ([`super::sparse_triangular`]) — peel column and row
//!    singletons to fixpoint. Their pivots are structurally forced and their
//!    blocks are upper triangular, so they need no ordering at all.
//! 2. **Fill-reducing ordering of the residual bump** — feral's in-tree AMD
//!    (`feral_amd`) on the bump's `AᵀA` (column-intersection) pattern, a
//!    stand-in for COLAMD that needs no new ordering algorithm.
//!
//! Running AMD on the bump instead of the whole basis is where the win is: AMD
//! is superlinear in the matrix it is handed, and a simplex basis is typically
//! 85–100% peelable (see [`super::sparse_triangular`]).
//!
//! The resulting permutation is the reusable symbolic handle: across numerically
//! different but structurally identical bases, only the numeric factor is
//! recomputed. Both stages read only the pattern, so that contract is unchanged.

use super::sparse_matrix::SparseColMatrix;
use super::sparse_triangular::triangularize;
use crate::error::FeralError;

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
    /// [`Self::analyze`] sets this; [`Self::natural`], [`Self::with_order`] and
    /// [`Self::analyze_amd_only`] do not — they claim `(0, m)` because they
    /// never looked, not because they looked and found the whole basis
    /// irreducible. The two cases are indistinguishable from the indices alone,
    /// and they warrant opposite answers from
    /// [`LuParams::dense_bump_max_dim`](super::LuParams::dense_bump_max_dim):
    /// an `analyze`d basis that peels to nothing really is a dense block worth
    /// the dense kernel, while an unpeeled `natural` ordering of a large sparse
    /// basis is the pathological case that route must never take.
    pub triangularized: bool,
}

impl SparseLuSymbolic {
    /// Triangularize, then order the residual bump with AMD.
    pub fn analyze(a: &SparseColMatrix) -> Result<Self, FeralError> {
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

    /// AMD over the whole basis, with no triangularization — feral's pre-0.16
    /// behavior. Retained so the two orderings can be compared directly in
    /// benchmarks and differential tests; [`Self::analyze`] is the one to use.
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
        let s = SparseLuSymbolic::analyze(&a).expect("analyze");
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
