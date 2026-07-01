//! One-norm condition estimate (`κ₁ ≈ ‖B‖₁·‖B⁻¹‖₁`) for the unsymmetric LU
//! basis (issue #94).
//!
//! The crate's Hager–Higham 1-norm estimator lives in
//! [`crate::numeric::condition`], where the shared driver
//! [`hager_higham_inverse_norm_1`](crate::numeric::condition::hager_higham_inverse_norm_1)
//! runs the power iteration through a
//! [`HagerHighamOperator`](crate::numeric::condition::HagerHighamOperator).
//! The symmetric LDLᵀ path exploits `B⁻ᵀ = B⁻¹` (one solve per iteration); the
//! LU basis is unsymmetric, so each iteration needs one `ftran` (`B⁻¹`) and one
//! `btran` (`B⁻ᵀ`) — the two adapters below wire those solves into the driver.
//!
//! `‖B‖₁` is the maximum absolute column sum of the *supplied* basis `b` (the
//! factor does not retain it, mirroring `ftran_refined`).

use super::dense_factor::DenseLu;
use super::dense_matrix::GeneralMatrix;
use super::sparse_factor::SparseLu;
use super::sparse_matrix::SparseColMatrix;
use crate::error::FeralError;
use crate::numeric::condition::{hager_higham_inverse_norm_1, HagerHighamOperator};

/// Driver adapter over `SparseLu`: `apply_inverse` is `ftran` (`B⁻¹`),
/// `apply_inverse_transpose` is `btran` (`B⁻ᵀ`).
struct SparseLuHager<'a> {
    lu: &'a mut SparseLu,
}

impl HagerHighamOperator for SparseLuHager<'_> {
    fn dim(&self) -> usize {
        self.lu.dim()
    }
    fn apply_inverse(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        self.lu.ftran(rhs)
    }
    fn apply_inverse_transpose(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        self.lu.btran(rhs)
    }
}

/// Driver adapter over `DenseLu`.
struct DenseLuHager<'a> {
    lu: &'a mut DenseLu,
}

impl HagerHighamOperator for DenseLuHager<'_> {
    fn dim(&self) -> usize {
        self.lu.dim()
    }
    fn apply_inverse(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        self.lu.ftran(rhs)
    }
    fn apply_inverse_transpose(&mut self, rhs: &mut [f64]) -> Result<(), FeralError> {
        self.lu.btran(rhs)
    }
}

impl SparseLu {
    /// One-norm condition estimate `κ₁ ≈ ‖B‖₁·‖B⁻¹‖₁` via Hager–Higham, using
    /// this factor's `ftran`/`btran` for the `‖B⁻¹‖₁` solves.
    ///
    /// `b` is the original basis this factor was produced from (the factor does
    /// not retain it, mirroring [`SparseLu::ftran_refined`]); its maximum
    /// absolute column sum supplies `‖B‖₁`. The estimate is a lower bound on the
    /// true `κ₁` (Hager returns a lower bound on `‖B⁻¹‖₁`).
    ///
    /// Cost: `‖B‖₁` is one O(nnz) pass; the inverse-norm estimate is ≤ `2·5+1`
    /// ftran/btran solves against the stored factor. Returns
    /// `Err(DimensionMismatch)` if `b.m ≠ self.m`, and propagates a solve error
    /// (e.g. `SingularBasis`) from the underlying ftran/btran.
    pub fn condition_estimate_1(&mut self, b: &SparseColMatrix) -> Result<f64, FeralError> {
        if b.m != self.dim() {
            return Err(FeralError::DimensionMismatch {
                expected: self.dim(),
                got: b.m,
            });
        }
        if self.dim() == 0 {
            return Ok(0.0);
        }
        let bnorm = b.one_norm();
        let mut op = SparseLuHager { lu: self };
        let inv_norm = hager_higham_inverse_norm_1(&mut op)?;
        Ok(bnorm * inv_norm)
    }
}

impl DenseLu {
    /// One-norm condition estimate `κ₁ ≈ ‖B‖₁·‖B⁻¹‖₁` via Hager–Higham, using
    /// this factor's `ftran`/`btran` for the `‖B⁻¹‖₁` solves.
    ///
    /// See [`SparseLu::condition_estimate_1`] for the contract; `b` is the
    /// original dense basis (supplies `‖B‖₁`), and the estimate is a lower bound
    /// on the true `κ₁`.
    pub fn condition_estimate_1(&mut self, b: &GeneralMatrix) -> Result<f64, FeralError> {
        if b.m != self.dim() {
            return Err(FeralError::DimensionMismatch {
                expected: self.dim(),
                got: b.m,
            });
        }
        if self.dim() == 0 {
            return Ok(0.0);
        }
        let bnorm = b.one_norm();
        let mut op = DenseLuHager { lu: self };
        let inv_norm = hager_higham_inverse_norm_1(&mut op)?;
        Ok(bnorm * inv_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lu::LuParams;

    // ---- `one_norm` (max absolute column sum). Oracle: hand computation. ----

    #[test]
    fn sparse_one_norm_hand() {
        // B = [[1, 2], [3, 4]]  (col0 = [1,3], col1 = [2,4]).
        // Column sums: |1|+|3| = 4, |2|+|4| = 6.  ‖B‖₁ = 6.
        let cols = vec![vec![1.0, 3.0], vec![2.0, 4.0]];
        let b = SparseColMatrix::from_dense_columns(2, &cols).expect("matrix");
        assert!((b.one_norm() - 6.0).abs() < 1e-15);
    }

    #[test]
    fn dense_one_norm_uses_absolute_value() {
        // B = [[-1, 0], [0, -5]].  Column sums: 1 and 5.  ‖B‖₁ = 5.
        let cols = vec![vec![-1.0, 0.0], vec![0.0, -5.0]];
        let b = GeneralMatrix::from_columns(2, &cols).expect("matrix");
        assert!((b.one_norm() - 5.0).abs() < 1e-15);
    }

    // ---- Condition estimate. Oracle: hand-computed κ₁ = ‖B‖₁·‖B⁻¹‖₁. ----

    /// Identity `I₅`: `B⁻¹ = I`, so κ₁ = 1 exactly. The estimator is a lower
    /// bound, so require `≥ 1 − √eps`; a sane upper cap guards against blow-up.
    #[test]
    fn sparse_condition_identity_is_one() {
        let m = 5;
        let cols: Vec<Vec<f64>> = (0..m)
            .map(|j| (0..m).map(|i| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let b = SparseColMatrix::from_dense_columns(m, &cols).expect("matrix");
        let mut lu = SparseLu::factor_dense_columns(m, &cols, LuParams::default()).expect("factor");
        let kappa = lu.condition_estimate_1(&b).expect("estimate");
        assert!(kappa >= 1.0 - 1e-8, "identity κ {kappa} below 1");
        assert!(kappa <= 2.0, "identity κ {kappa} unexpectedly large");
    }

    /// Diagonal `diag(1, 1e3, 1e6)`: `‖B‖₁ = 1e6`, `B⁻¹ = diag(1, 1e-3, 1e-6)`
    /// so `‖B⁻¹‖₁ = 1`, giving true κ₁ = 1e6. Hager on a diagonal is exact, so
    /// require the estimate within 2× (matching the symmetric diagonal test).
    #[test]
    fn sparse_condition_diagonal_spectrum() {
        let cols = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1e3, 0.0],
            vec![0.0, 0.0, 1e6],
        ];
        let b = SparseColMatrix::from_dense_columns(3, &cols).expect("matrix");
        let mut lu = SparseLu::factor_dense_columns(3, &cols, LuParams::default()).expect("factor");
        let kappa = lu.condition_estimate_1(&b).expect("estimate");
        assert!(
            (0.5e6..=2.0e6).contains(&kappa),
            "diagonal κ {kappa} not within 2× of 1e6"
        );
    }

    /// `B = [[1, 2], [3, 4]]`: `det = −2`, `B⁻¹ = [[−2, 1], [1.5, −0.5]]`,
    /// `‖B‖₁ = 6`, `‖B⁻¹‖₁ = 3.5`, so true κ₁ = 21 (hand). The estimate is a
    /// lower bound on `‖B⁻¹‖₁`, so require `κ_est ∈ [κ_true/2, κ_true·(1+ε)]`.
    #[test]
    fn sparse_condition_2x2_hand_oracle() {
        let cols = vec![vec![1.0, 3.0], vec![2.0, 4.0]];
        let b = SparseColMatrix::from_dense_columns(2, &cols).expect("matrix");
        let mut lu = SparseLu::factor_dense_columns(2, &cols, LuParams::default()).expect("factor");
        let kappa = lu.condition_estimate_1(&b).expect("estimate");
        assert!(
            (10.5..=21.0 * (1.0 + 1e-6)).contains(&kappa),
            "2×2 κ {kappa} outside [10.5, 21] (true 21)"
        );
    }

    /// Same `B = [[1, 2], [3, 4]]` through the dense path, hand oracle κ₁ = 21.
    #[test]
    fn dense_condition_2x2_hand_oracle() {
        let cols = vec![vec![1.0, 3.0], vec![2.0, 4.0]];
        let b = GeneralMatrix::from_columns(2, &cols).expect("matrix");
        let mut lu = DenseLu::factor(&cols, 2, LuParams::default()).expect("factor");
        let kappa = lu.condition_estimate_1(&b).expect("estimate");
        assert!(
            (10.5..=21.0 * (1.0 + 1e-6)).contains(&kappa),
            "dense 2×2 κ {kappa} outside [10.5, 21] (true 21)"
        );
    }

    /// The dense and sparse paths estimate the same κ on the same well-
    /// conditioned basis (both exercise ftran/btran; the estimate must not
    /// depend on the storage path beyond floating-point rounding).
    #[test]
    fn dense_sparse_parity_on_same_basis() {
        let cols = vec![
            vec![10.0, 1.0, 0.0],
            vec![1.0, 8.0, 2.0],
            vec![0.0, 1.0, 5.0],
        ];
        let sb = SparseColMatrix::from_dense_columns(3, &cols).expect("sparse matrix");
        let db = GeneralMatrix::from_columns(3, &cols).expect("dense matrix");
        let mut slu =
            SparseLu::factor_dense_columns(3, &cols, LuParams::default()).expect("sparse factor");
        let mut dlu = DenseLu::factor(&cols, 3, LuParams::default()).expect("dense factor");
        let sk = slu.condition_estimate_1(&sb).expect("sparse estimate");
        let dk = dlu.condition_estimate_1(&db).expect("dense estimate");
        let rel = (sk - dk).abs() / sk.max(dk).max(1.0);
        assert!(
            rel < 1e-6,
            "dense/sparse κ parity: sparse {sk}, dense {dk} (rel {rel})"
        );
    }

    /// `b.m ≠ self.dim()` is rejected before any solve, on both paths.
    #[test]
    fn condition_dimension_mismatch_rejected() {
        let cols2 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cols3 = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let sb3 = SparseColMatrix::from_dense_columns(3, &cols3).expect("matrix");
        let mut slu =
            SparseLu::factor_dense_columns(2, &cols2, LuParams::default()).expect("factor");
        assert!(matches!(
            slu.condition_estimate_1(&sb3),
            Err(FeralError::DimensionMismatch { .. })
        ));

        let db3 = GeneralMatrix::from_columns(3, &cols3).expect("matrix");
        let mut dlu = DenseLu::factor(&cols2, 2, LuParams::default()).expect("factor");
        assert!(matches!(
            dlu.condition_estimate_1(&db3),
            Err(FeralError::DimensionMismatch { .. })
        ));
    }
}
