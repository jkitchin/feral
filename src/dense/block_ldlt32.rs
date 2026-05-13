//! Block-32 register-resident LDLᵀ kernel (Phase 2.4.3, issue #9).
//!
//! See `dev/plans/phase-2.4.3-block32-kernel.md` and
//! `dev/research/block32-register-resident-kernel.md`.
//!
//! **Status: Step 1 scaffolding.** `factor_block32` currently delegates
//! to `factor_frontal` so the dispatch and test harness can land
//! independently of the SIMD body. The real `update_1x1_block32` /
//! `update_2x2_block32` and the monomorphized BLOCK_SIZE=32 driver
//! arrive in Steps 2–4 of the plan.
//!
//! ## Bit-parity contract
//!
//! At every step of this plan the kernel must produce
//! `f64::to_bits()`-equal `(L, D, perm, subdiag, contrib)` to
//! `factor_frontal` on the same input. The scalar oracle is
//! `factor_frontal` (not `factor_frontal_blocked`) because the
//! unblocked scalar loop's per-element rounding chain
//! (`axpy_minus_unroll4_nofma` applied eagerly to the ground-truth
//! trailing state) is exactly what the eager block-32 update
//! reproduces.
//!
//! ## Rounding discipline
//!
//! Inherits the 2026-04-14 decision (`dev/decisions.md:464`): every
//! SIMD trailing-update lane uses separate `mul_f64s` + `sub_f64s`
//! instead of `mul_add_f64s`. No FMA anywhere in this module.

use crate::dense::factor::{factor_frontal, BunchKaufmanParams, FrontalFactors};
use crate::dense::matrix::SymmetricMatrix;
use crate::error::FeralError;

/// Hard-coded block size for this kernel.
///
/// The kernel is monomorphized at BS=32 because that is the dominant
/// front size on KKT chain matrices (see
/// `dev/research/ssids-small-frontal-speed.md` §0). Other block sizes
/// take the existing `factor_frontal_blocked` path.
#[allow(dead_code)] // Wired into factor.rs dispatch in Step 2 of the plan.
pub const BLOCK_SIZE: usize = 32;

/// Factor a 32×32 symmetric matrix using the register-resident
/// block-32 kernel.
///
/// Equivalent in semantics to `factor_frontal(matrix, 32, may_delay,
/// params)`, but the trailing update is intended (Step 3 onward) to
/// run in one pulp dispatch per pivot rather than per-column axpy.
///
/// **Step 1 stub:** delegates to `factor_frontal`. The bit-parity
/// tests in this module pass trivially under the stub; they become
/// load-bearing in Step 2 when the kernel body diverges from the
/// scalar oracle's call path.
#[allow(dead_code)] // Wired into factor.rs dispatch in Step 2 of the plan.
pub(crate) fn factor_block32(
    matrix: &SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
) -> Result<FrontalFactors, FeralError> {
    if matrix.n != BLOCK_SIZE {
        return Err(FeralError::InvalidInput(format!(
            "factor_block32: matrix size {} != BLOCK_SIZE {}",
            matrix.n, BLOCK_SIZE
        )));
    }
    // Step 1: delegate. The dispatch site in factor_frontal_blocked
    // does not yet route here, so this branch is currently exercised
    // only by the unit tests below.
    factor_frontal(matrix, ncol, may_delay, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 32×32 lower-triangular `SymmetricMatrix` from a
    /// row-major dense slice. Only the lower triangle is read; the
    /// upper is ignored by every consumer.
    fn from_lower(rows: &[[f64; BLOCK_SIZE]; BLOCK_SIZE]) -> SymmetricMatrix {
        let mut data = vec![0.0f64; BLOCK_SIZE * BLOCK_SIZE];
        for j in 0..BLOCK_SIZE {
            for i in j..BLOCK_SIZE {
                data[j * BLOCK_SIZE + i] = rows[i][j];
            }
        }
        SymmetricMatrix {
            n: BLOCK_SIZE,
            data,
        }
    }

    /// Construct a 32×32 indefinite symmetric matrix with a fixed
    /// seed. Diagonal entries are pseudo-random in `[-1, 1)`, off-
    /// diagonal in `[-0.5, 0.5)`. Diagonally non-dominant so the BK
    /// pivot rules genuinely fire 1×1 / swap-1×1 / 2×2 branches.
    fn seeded_indefinite_32x32() -> SymmetricMatrix {
        // Splitmix64 — deterministic, no external crate, identical
        // output across architectures.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || -> f64 {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            // Map to [0, 1) by taking the high 53 bits.
            ((z >> 11) as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
        };
        let mut rows = [[0.0f64; BLOCK_SIZE]; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            for j in 0..=i {
                if i == j {
                    rows[i][j] = 2.0 * next() - 1.0;
                } else {
                    rows[i][j] = next() - 0.5;
                }
            }
        }
        from_lower(&rows)
    }

    fn assert_factors_bit_equal(actual: &FrontalFactors, expected: &FrontalFactors) {
        assert_eq!(actual.nrow, expected.nrow, "nrow");
        assert_eq!(actual.ncol, expected.ncol, "ncol");
        assert_eq!(actual.nelim, expected.nelim, "nelim");
        assert_eq!(actual.n_delayed, expected.n_delayed, "n_delayed");
        assert_eq!(actual.inertia, expected.inertia, "inertia");
        assert_eq!(actual.perm, expected.perm, "perm");
        assert_eq!(actual.perm_inv, expected.perm_inv, "perm_inv");

        assert_eq!(actual.l.len(), expected.l.len(), "L length");
        for k in 0..actual.l.len() {
            assert_eq!(
                actual.l[k].to_bits(),
                expected.l[k].to_bits(),
                "L[{k}] mismatch: actual={} expected={}",
                actual.l[k],
                expected.l[k]
            );
        }

        assert_eq!(actual.d_diag.len(), expected.d_diag.len(), "d_diag length");
        for k in 0..actual.d_diag.len() {
            assert_eq!(
                actual.d_diag[k].to_bits(),
                expected.d_diag[k].to_bits(),
                "d_diag[{k}] mismatch"
            );
        }

        assert_eq!(
            actual.d_subdiag.len(),
            expected.d_subdiag.len(),
            "d_subdiag length"
        );
        for k in 0..actual.d_subdiag.len() {
            assert_eq!(
                actual.d_subdiag[k].to_bits(),
                expected.d_subdiag[k].to_bits(),
                "d_subdiag[{k}] mismatch"
            );
        }

        assert_eq!(
            actual.contrib.len(),
            expected.contrib.len(),
            "contrib length"
        );
        for k in 0..actual.contrib.len() {
            assert_eq!(
                actual.contrib[k].to_bits(),
                expected.contrib[k].to_bits(),
                "contrib[{k}] mismatch"
            );
        }
    }

    /// Construct two independent copies of the same lower triangle so
    /// that scalar and block-32 paths each get their own scratch.
    fn dup_lower(src: &SymmetricMatrix) -> (SymmetricMatrix, SymmetricMatrix) {
        let a = SymmetricMatrix {
            n: src.n,
            data: src.data.clone(),
        };
        let b = SymmetricMatrix {
            n: src.n,
            data: src.data.clone(),
        };
        (a, b)
    }

    #[test]
    fn factor_block32_rejects_wrong_size() {
        let m = SymmetricMatrix::zeros(16);
        let params = BunchKaufmanParams::default();
        let res = factor_block32(&m, 16, false, &params);
        assert!(res.is_err());
    }

    /// Smoke test: on a diagonal SPD matrix, the block-32 path and
    /// the scalar oracle agree bit-for-bit. Under the Step 1 stub
    /// this is tautological; the same assertion is the load-bearing
    /// regression test once the kernel body lands in Step 2+.
    #[test]
    fn factor_block32_diagonal_spd_matches_scalar() {
        let mut rows = [[0.0f64; BLOCK_SIZE]; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            rows[i][i] = (i as f64) + 1.0;
        }
        let src = from_lower(&rows);
        let (a, b) = dup_lower(&src);
        let params = BunchKaufmanParams::default();
        let scalar = factor_frontal(&a, BLOCK_SIZE, false, &params).expect("scalar");
        let block = factor_block32(&b, BLOCK_SIZE, false, &params).expect("block32");
        assert_factors_bit_equal(&block, &scalar);
    }

    /// Bit-parity on a seeded random indefinite 32×32 matrix. This is
    /// the harness Step 2/3/4 commits will re-run against the real
    /// kernel body. Under the Step 1 stub it passes trivially.
    #[test]
    fn factor_block32_seeded_indefinite_matches_scalar() {
        let src = seeded_indefinite_32x32();
        let (a, b) = dup_lower(&src);
        let params = BunchKaufmanParams::default();
        let scalar = factor_frontal(&a, BLOCK_SIZE, false, &params).expect("scalar");
        let block = factor_block32(&b, BLOCK_SIZE, false, &params).expect("block32");
        assert_factors_bit_equal(&block, &scalar);
    }
}
