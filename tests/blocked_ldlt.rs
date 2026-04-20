//! RED tests for Phase 2.4.1b — blocked dense LDLᵀ via faer-style
//! peek-ahead panel. Spec in `dev/plans/phase-2.4.1-blocked-ldlt.md`
//! §Test plan (six items).
//!
//! State on RED commit: `factor_frontal_blocked` is a stub that
//! returns `FeralError::InvalidInput("…not yet implemented")`. The
//! tests here compile against the stub API and fail at runtime in
//! exactly the places the GREEN commit is expected to fix. The
//! scalar-path oracles (`factor_frontal`) are unaffected — they
//! continue to pass — so the RED commit is safe to land.
//!
//! Test map (vs plan §Test plan):
//!   test_spd_scalar_blocked_parity_size_sweep       -> §1
//!   test_indefinite_bk77_parity                     -> §2
//!   test_frontal_ncol_lt_nrow_parity                -> §3
//!   test_2x2_at_block_boundary                      -> §4
//!   test_rejection_fallback                         -> §5
//!   test_kkt_regression_spot_checks                 -> §6
//!
//! Parity ORACLE is exact byte-identity of `(L, D_diag, D_subdiag,
//! perm, inertia, contrib, nelim, n_delayed, needs_refinement)`.

use feral::dense::factor::{factor_frontal, factor_frontal_blocked, FrontalFactors};
use feral::{BunchKaufmanParams, SymmetricMatrix, ZeroPivotAction};

/// Deterministic pseudo-random f64 in (-1, 1). Matches the style used
/// by `tests/dense_fast_path.rs`.
fn rng_scalar(state: &mut u64, idx: usize) -> f64 {
    *state = state
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(idx as u64 + 1);
    let u = (*state >> 32) as u32 as f64;
    (u / (u32::MAX as f64)) * 2.0 - 1.0
}

/// Random SPD matrix A = U + U^T + n*I where U is lower-triangular. The
/// `+ n*I` shift guarantees strict diagonal dominance and therefore
/// SPD; BK picks 1×1 pivots throughout, which stresses the panel
/// 1×1-only fast path.
fn random_spd(n: usize, seed: u64) -> SymmetricMatrix {
    let mut state = seed;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            let v = rng_scalar(&mut state, i * n + j);
            data[j * n + i] = v;
        }
    }
    // Force SPD: diagonal += n, and scale off-diagonals down to keep
    // the shift effective.
    for j in 0..n {
        data[j * n + j] = data[j * n + j].abs() + (n as f64) + 1.0;
    }
    SymmetricMatrix { n, data }
}

fn assert_frontals_byte_identical(scalar: &FrontalFactors, blocked: &FrontalFactors, tag: &str) {
    assert_eq!(scalar.nrow, blocked.nrow, "{} nrow", tag);
    assert_eq!(scalar.ncol, blocked.ncol, "{} ncol", tag);
    assert_eq!(scalar.nelim, blocked.nelim, "{} nelim", tag);
    assert_eq!(scalar.n_delayed, blocked.n_delayed, "{} n_delayed", tag);
    assert_eq!(
        scalar.contrib_dim, blocked.contrib_dim,
        "{} contrib_dim",
        tag
    );
    assert_eq!(scalar.inertia, blocked.inertia, "{} inertia", tag);
    assert_eq!(
        scalar.needs_refinement, blocked.needs_refinement,
        "{} needs_refinement",
        tag
    );
    assert_eq!(scalar.perm, blocked.perm, "{} perm", tag);
    assert_eq!(scalar.perm_inv, blocked.perm_inv, "{} perm_inv", tag);
    assert_eq!(
        scalar.l.len(),
        blocked.l.len(),
        "{} l.len (nrow*nelim)",
        tag
    );
    for (i, (a, b)) in scalar.l.iter().zip(blocked.l.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{} l[{}] scalar={} blocked={}",
            tag,
            i,
            a,
            b
        );
    }
    for (i, (a, b)) in scalar.d_diag.iter().zip(blocked.d_diag.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{} d_diag[{}]", tag, i);
    }
    for (i, (a, b)) in scalar
        .d_subdiag
        .iter()
        .zip(blocked.d_subdiag.iter())
        .enumerate()
    {
        assert_eq!(a.to_bits(), b.to_bits(), "{} d_subdiag[{}]", tag, i);
    }
    for (i, (a, b)) in scalar
        .contrib
        .iter()
        .zip(blocked.contrib.iter())
        .enumerate()
    {
        assert_eq!(a.to_bits(), b.to_bits(), "{} contrib[{}]", tag, i);
    }
}

fn default_params() -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    }
}

/// Plan §1 — SPD size sweep covering scalar fallback (32, 64), the
/// block-boundary 1-column leftover (65, 129), a clean 2-panel case
/// (128), and sizes well past the boundary (100, 200, 256, 300).
#[test]
fn test_spd_scalar_blocked_parity_size_sweep() {
    let params = default_params();
    for &n in &[32usize, 64, 65, 100, 128, 129, 200, 256, 300] {
        let mat = random_spd(n, 0xABCD_1234_0000 ^ n as u64);
        let scalar = factor_frontal(&mat, n, false, &params).unwrap();
        let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
        assert_frontals_byte_identical(&scalar, &blocked, &format!("spd n={}", n));
    }
}

/// Plan §2 — symmetric indefinite from the Bunch-Kaufman 1977 paper.
/// BK's Example 1 (`dev/research/dense-ldlt.md`):
///   [ 1   1   0 ]
///   [ 1  1.5  1 ]
///   [ 0   1   1 ]
/// which produces a 2×2 pivot at k=0 in the scalar kernel. The
/// blocked kernel must produce the same L, D, perm, inertia (1+, 1-,
/// 1+ = 2+, 1−, 0 zero) byte-for-byte. To exercise sizes past the
/// block boundary, we also test a 70×70 shifted-indefinite matrix.
#[test]
fn test_indefinite_bk77_parity() {
    let params = default_params();

    // BK77 Example 1 (3×3). Small — exercises scalar fallback only,
    // but serves as a low-risk sanity check on the API surface.
    let mat = SymmetricMatrix {
        n: 3,
        data: vec![1.0, 1.0, 0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 1.0],
    };
    let scalar = factor_frontal(&mat, 3, false, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, 3, false, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "bk77_3x3");

    // Indefinite 70×70: random symmetric with small diagonal so 2×2
    // pivots are likely. 70 > 64 so this crosses the panel boundary.
    let n = 70;
    let mut state = 0xFACE_1977u64;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            data[j * n + i] = rng_scalar(&mut state, i * n + j);
        }
        // Small diagonal keeps BK in 2×2 territory periodically.
        data[j * n + j] *= 0.05;
    }
    let mat = SymmetricMatrix { n, data };
    let scalar = factor_frontal(&mat, n, false, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "indef_70");
}

/// Plan §3 — frontal `ncol < nrow`: the blocked kernel eliminates only
/// the first `ncol` columns and the contribution block must match
/// scalar byte-for-byte. Uses `nrow=80, ncol=48` so the panel stops
/// before the first block boundary and has to finalize partial state.
#[test]
fn test_frontal_ncol_lt_nrow_parity() {
    let params = default_params();
    let nrow = 80;
    let ncol = 48;
    let mat = random_spd(nrow, 0x1234_FFFF);
    let scalar = factor_frontal(&mat, ncol, false, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, ncol, false, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "ncol_lt_nrow");
}

/// Plan §4 — 2×2 BK pivot lands at `k = block_size - 1 = 63`. We
/// construct a matrix whose diagonals are large everywhere except at
/// `{63, 64}`, where a small-diagonal / large-off-diagonal 2×2 block
/// is forced. The blocked kernel must extend its panel through k=64
/// (returning `n_elim = bs - 1` on the first panel iteration) and
/// re-enter for the remainder. Parity is byte-identical to scalar.
#[test]
fn test_2x2_at_block_boundary() {
    let params = default_params();
    let n = 128;
    let mut data = vec![0.0f64; n * n];
    // Strong diagonal everywhere.
    for j in 0..n {
        data[j * n + j] = 1.0 + j as f64 * 0.001;
    }
    // Weak off-diagonal noise (keeps everything else in 1×1 land).
    let mut state = 0xBD22_BD22u64;
    for j in 0..n {
        for i in (j + 1)..n {
            data[j * n + i] = 1e-6 * rng_scalar(&mut state, i * n + j);
        }
    }
    // Boundary 2×2 trigger at {63, 64}: zero the diagonals and put a
    // large cross term so BK is forced into a 2×2 pivot.
    data[63 * n + 63] = 0.0;
    data[64 * n + 64] = 0.0;
    data[63 * n + 64] = 1.0;
    let mat = SymmetricMatrix { n, data };
    let scalar = factor_frontal(&mat, n, false, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "2x2_boundary");
}

/// Plan §5 — forced rejection at `k = block_size/2 = 32`. With
/// `pivot_threshold = 0.01` a column whose max off-diagonal exceeds
/// `100 × |diag|` is rejected via the column-relative threshold. We
/// construct that shape at column 32 to force the panel to return
/// early and have the caller finish the step in scalar mode before
/// re-entering the panel path.
#[test]
fn test_rejection_fallback() {
    let params = default_params();
    let n = 128;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        data[j * n + j] = 1.0 + j as f64 * 0.001;
    }
    // Column 32 gets a dominant off-diagonal entry at row 50 —
    // outside the 2×2 boundary and strong enough to force
    // rejection.
    data[32 * n + 50] = 1000.0;
    // Ensure row 50 is not itself a good pivot candidate when swapped
    // in, by leaving `data[50*n + 50]` untouched at its small value.
    let mat = SymmetricMatrix { n, data };
    let scalar = factor_frontal(&mat, n, false, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "rejection_fallback");
}

/// Plan §6 — KKT regression spot-checks. We synthesize two tiny KKT
/// blocks styled after the triage canaries (`ERRINBAR`, `ACOPP30`)
/// and verify scalar/blocked byte-parity. These are not the literal
/// matrices — the reference residuals live in the KKT corpus — but
/// they exercise the same structural shape: dense arrow-KKT with a
/// small saddle-point block at the corner.
#[test]
fn test_kkt_regression_spot_checks() {
    let params = default_params();

    // ERRINBAR-style: SPD-dominant top block + two equality rows.
    {
        let n = 96;
        let mut data = vec![0.0f64; n * n];
        for j in 0..n - 2 {
            data[j * n + j] = 2.0 + 0.01 * j as f64;
        }
        // Small bandwidth off-diagonals in the SPD top block.
        for j in 0..n - 3 {
            data[j * n + (j + 1)] = 0.3;
            data[j * n + (j + 2)] = 0.1;
        }
        // Two zero-diagonal equality rows at the bottom:
        data[(n - 2) * n + (n - 2)] = 0.0;
        data[(n - 1) * n + (n - 1)] = 0.0;
        // Arrow coupling into the equality rows:
        for j in 0..(n - 2) {
            data[j * n + (n - 2)] = 0.1;
            data[j * n + (n - 1)] = 0.05;
        }
        data[(n - 2) * n + (n - 1)] = 0.01;
        let mat = SymmetricMatrix { n, data };
        let scalar = factor_frontal(&mat, n, false, &params).unwrap();
        let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
        assert_frontals_byte_identical(&scalar, &blocked, "kkt_errinbar_like");
    }

    // ACOPP30-style: SPD block of size 150 with a small saddle-point
    // structure at the end. 150 > 128 so this spans multiple panels.
    {
        let n = 150;
        let mut data = vec![0.0f64; n * n];
        let mut state = 0xAC00_7530u64;
        for j in 0..n - 4 {
            data[j * n + j] = 3.0 + 0.005 * j as f64;
        }
        for j in 0..n - 4 {
            for i in (j + 1)..(j + 4).min(n - 4) {
                data[j * n + i] = 0.1 * rng_scalar(&mut state, i * n + j);
            }
        }
        // Saddle-point tail:
        for k in 0..4 {
            data[(n - 4 + k) * n + (n - 4 + k)] = 0.0;
            for j in 0..(n - 4) {
                data[j * n + (n - 4 + k)] = 0.05 * rng_scalar(&mut state, k * n + j);
            }
        }
        let mat = SymmetricMatrix { n, data };
        let scalar = factor_frontal(&mat, n, false, &params).unwrap();
        let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
        assert_frontals_byte_identical(&scalar, &blocked, "kkt_acopp30_like");
    }
}
