//! Input sparsity-pattern type.
//!
//! The crate intentionally takes slices rather than a library-
//! specific matrix type, so any caller can adapt without a type
//! dependency.

/// Borrowed symmetric sparsity pattern in CSC form.
///
/// The pattern must be **full-symmetric** — both the upper and lower
/// halves are present. Row indices within each column must be
/// sorted in ascending order.
///
/// Invariants (checked by [`CscPattern::new`]):
/// - `col_ptr.len() == n + 1`
/// - `col_ptr[0] == 0`, `col_ptr` is non-decreasing
/// - `row_idx.len() == col_ptr[n]`
/// - every row index is `< n`
///
/// Structural symmetry is not checked in release builds; it is
/// debug-asserted at the entry point of the algorithm.
#[derive(Debug, Clone, Copy)]
pub struct CscPattern<'a> {
    /// Matrix dimension.
    pub n: usize,
    /// Column pointers. Length `n + 1`.
    pub col_ptr: &'a [usize],
    /// Row indices. Length `col_ptr[n]`.
    pub row_idx: &'a [usize],
}

impl<'a> CscPattern<'a> {
    /// Construct a validated pattern.
    ///
    /// Returns `None` if the structural invariants above are
    /// violated. Does not check symmetry.
    pub fn new(n: usize, col_ptr: &'a [usize], row_idx: &'a [usize]) -> Option<Self> {
        if col_ptr.len() != n + 1 {
            return None;
        }
        if col_ptr[0] != 0 {
            return None;
        }
        let nnz = *col_ptr.last()?;
        if row_idx.len() != nnz {
            return None;
        }
        for w in col_ptr.windows(2) {
            if w[1] < w[0] {
                return None;
            }
        }
        for &r in row_idx {
            if r >= n {
                return None;
            }
        }
        Some(Self {
            n,
            col_ptr,
            row_idx,
        })
    }

    /// Number of stored nonzeros.
    pub fn nnz(&self) -> usize {
        self.row_idx.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pattern_ok() {
        let cp = [0];
        let ri: [usize; 0] = [];
        let p = CscPattern::new(0, &cp, &ri).expect("n=0 pattern");
        assert_eq!(p.n, 0);
        assert_eq!(p.nnz(), 0);
    }

    #[test]
    fn diagonal_2x2_ok() {
        let cp = [0, 1, 2];
        let ri = [0, 1];
        let p = CscPattern::new(2, &cp, &ri).unwrap();
        assert_eq!(p.nnz(), 2);
    }

    #[test]
    fn rejects_bad_col_ptr_length() {
        let cp = [0, 1]; // should be length n+1 = 3 for n=2
        let ri = [0];
        assert!(CscPattern::new(2, &cp, &ri).is_none());
    }

    #[test]
    fn rejects_oob_row_index() {
        let cp = [0, 1];
        let ri = [5]; // n=1, so row index 5 is out of bounds
        assert!(CscPattern::new(1, &cp, &ri).is_none());
    }

    #[test]
    fn rejects_nonzero_first_col_ptr() {
        let cp = [1, 2];
        let ri = [0];
        assert!(CscPattern::new(1, &cp, &ri).is_none());
    }

    #[test]
    fn rejects_nonmonotone_col_ptr() {
        let cp = [0, 2, 1];
        let ri = [0, 0];
        assert!(CscPattern::new(2, &cp, &ri).is_none());
    }

    #[test]
    fn rejects_row_idx_length_mismatch() {
        let cp = [0, 1, 2];
        let ri = [0]; // should be length 2
        assert!(CscPattern::new(2, &cp, &ri).is_none());
    }
}
