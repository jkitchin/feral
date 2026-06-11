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

use feral::dense::factor::{
    factor_frontal, factor_frontal_blocked, panel_diag, FrontalFactors, PANEL_DIAG_ENABLED,
};
use feral::{BunchKaufmanParams, SymmetricMatrix, ZeroPivotAction};
use std::sync::atomic::Ordering;

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
    assert_eq!(
        scalar.n_rook_rescues, blocked.n_rook_rescues,
        "{} n_rook_rescues",
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

/// Phase 2.4.1b Step 5 — `may_delay == true` parity on SPD across the
/// same size sweep. SPD → no rejection → panel runs to completion; the
/// blocked and scalar paths should produce byte-identical output.
#[test]
fn test_may_delay_spd_parity_size_sweep() {
    let params = default_params();
    for &n in &[32usize, 64, 100, 128, 200, 300] {
        let mat = random_spd(n, 0xDEAD_BEEF ^ n as u64);
        let scalar = factor_frontal(&mat, n, true, &params).unwrap();
        let blocked = factor_frontal_blocked(&mat, n, true, &params).unwrap();
        assert_frontals_byte_identical(&scalar, &blocked, &format!("may_delay spd n={}", n));
    }
}

/// Step 5 — `may_delay == true` with `ncol < nrow` (non-root supernode
/// case). The blocked path must handle partial elimination with
/// delayed-pivot semantics.
#[test]
fn test_may_delay_ncol_lt_nrow_parity() {
    let params = default_params();
    let nrow = 200;
    let ncol = 128;
    let mat = random_spd(nrow, 0x5555_3333);
    let scalar = factor_frontal(&mat, ncol, true, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, ncol, true, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "may_delay ncol_lt_nrow");
}

/// W-1 (`dev/plans/dense-kernel-speedup.md`): the panel gate was
/// lowered from `ncol > block_size` (default 64) to `ncol >= 8`,
/// engaging the deferred-Schur path on the 32-col CHAINWOO root
/// supernode and on every front in the 8..=32 size band. The scalar
/// `factor_frontal` predates this change, so byte-identity against
/// it satisfies the bit-parity contract per CLAUDE.md.
///
/// SPD sweep covering the new band, including sizes that exercise:
///  - exact panel-cap fits (`ncol == 8, 16, 32`)
///  - cap-not-multiple-of-cap leftovers (`ncol == 12, 24`)
///  - small `ncol < nrow` fronts where the contribution block is
///    populated by the deferred Schur (the new common case for the
///    multifrontal driver).
#[test]
fn test_w1_spd_panel_band_parity() {
    let params = default_params();
    for &n in &[8usize, 12, 16, 24, 32] {
        let mat = random_spd(n, 0xA1B2_C3D4 ^ n as u64);
        let scalar = factor_frontal(&mat, n, false, &params).unwrap();
        let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
        assert_frontals_byte_identical(&scalar, &blocked, &format!("w1_spd ncol=nrow={}", n));
    }
}

/// W-1: small `ncol < nrow` shapes. `nrow` chosen larger than ncol so
/// the deferred Schur fires on a non-trivial trailing block. The
/// CHAINWOO root supernode is `nrow=1984, ncol=32`; we use scaled-down
/// analogs (`nrow=4*ncol`) to keep the test fast.
#[test]
fn test_w1_spd_panel_band_partial_parity() {
    let params = default_params();
    for &ncol in &[8usize, 12, 16, 24, 32] {
        let nrow = 4 * ncol;
        let mat = random_spd(nrow, 0x5151_AAAA ^ ncol as u64);
        let scalar = factor_frontal(&mat, ncol, false, &params).unwrap();
        let blocked = factor_frontal_blocked(&mat, ncol, false, &params).unwrap();
        assert_frontals_byte_identical(
            &scalar,
            &blocked,
            &format!("w1_spd nrow={}, ncol={}", nrow, ncol),
        );
    }
}

/// W-1: `may_delay == true` parity in the new panel band. SPD ⇒ no
/// rejection ⇒ panel runs to completion, and the deferred-Schur path
/// must continue to byte-match scalar.
#[test]
fn test_w1_may_delay_panel_band_parity() {
    let params = default_params();
    for &ncol in &[8usize, 12, 16, 24, 32] {
        let nrow = 4 * ncol;
        let mat = random_spd(nrow, 0xC0DE_1234 ^ ncol as u64);
        let scalar = factor_frontal(&mat, ncol, true, &params).unwrap();
        let blocked = factor_frontal_blocked(&mat, ncol, true, &params).unwrap();
        assert_frontals_byte_identical(
            &scalar,
            &blocked,
            &format!("w1_may_delay nrow={}, ncol={}", nrow, ncol),
        );
    }
}

/// W-2 2×2 inline (no-swap fast path) — a panel that contains two
/// no-swap 2×2 pivots at panel-internal positions, plus surrounding
/// 1×1 pivots. The blocked panel must accept the 2×2's inline (no
/// `ScalarFallback` bail-out) and produce byte-identical output to
/// scalar `factor_frontal`. See `dev/plans/dense-kernel-blas3.md` §3.
///
/// Construction: ncol=nrow=64 (single panel pass). Strong diagonal
/// at all rows except {16, 17} and {40, 41}, where the diagonals
/// are zeroed and a large cross term `a[k, k+1]` forces a no-swap
/// 2×2 pivot. Other off-diagonals are tiny noise so the column
/// argmax for k=16, k=40 is at row k+1 (no swap required).
#[test]
fn test_2x2_inside_panel() {
    let params = default_params();
    let n = 64;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        data[j * n + j] = 1.0 + j as f64 * 0.001;
    }
    let mut state = 0x2222_BD22u64;
    for j in 0..n {
        for i in (j + 1)..n {
            data[j * n + i] = 1e-6 * rng_scalar(&mut state, i * n + j);
        }
    }
    // Two no-swap 2×2 triggers inside the same panel (panel cap = 64
    // by default). Zero diagonals at the pair, large cross term so
    // BK picks 2×2 with r == k+1 (no swap).
    for &k in &[16usize, 40] {
        data[k * n + k] = 0.0;
        data[(k + 1) * n + (k + 1)] = 0.0;
        data[k * n + (k + 1)] = 1.0;
    }
    let mat = SymmetricMatrix { n, data };
    let scalar = factor_frontal(&mat, n, false, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "2x2_inside_panel");
}

/// W-2 2×2 inline — multifrontal-style fixture: `nrow > ncol` so the
/// panel's deferred-Schur step writes into the contribution block,
/// AND a no-swap 2×2 fires inside the panel. Mirrors the path the
/// real KKT matrices (SWOPF, HIMMELBJ) hit through
/// `factorize_multifrontal`.
#[test]
fn test_2x2_inside_panel_ncol_lt_nrow() {
    let params = default_params();
    let nrow = 80;
    let ncol = 48;
    let mut data = vec![0.0f64; nrow * nrow];
    for j in 0..nrow {
        data[j * nrow + j] = 1.0 + j as f64 * 0.001;
    }
    let mut state = 0x7777_BD22u64;
    for j in 0..nrow {
        for i in (j + 1)..nrow {
            data[j * nrow + i] = 1e-6 * rng_scalar(&mut state, i * nrow + j);
        }
    }
    let k = 16;
    data[k * nrow + k] = 0.0;
    data[(k + 1) * nrow + (k + 1)] = 0.0;
    data[k * nrow + (k + 1)] = 1.0;
    let mat = SymmetricMatrix { n: nrow, data };
    let scalar = factor_frontal(&mat, ncol, false, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, ncol, false, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "2x2_inside_panel_ncol_lt_nrow");
}

/// W-2 2×2 inline under `may_delay = true`. The multifrontal driver
/// passes may_delay=true for non-root supernodes; the inline path
/// must remain bit-exact under the SSIDS-style break-on-first-failure
/// semantics.
#[test]
fn test_2x2_inside_panel_may_delay() {
    let params = default_params();
    let nrow = 80;
    let ncol = 48;
    let mut data = vec![0.0f64; nrow * nrow];
    for j in 0..nrow {
        data[j * nrow + j] = 1.0 + j as f64 * 0.001;
    }
    let mut state = 0xA1A1_BD22u64;
    for j in 0..nrow {
        for i in (j + 1)..nrow {
            data[j * nrow + i] = 1e-6 * rng_scalar(&mut state, i * nrow + j);
        }
    }
    let k = 20;
    data[k * nrow + k] = 0.0;
    data[(k + 1) * nrow + (k + 1)] = 0.0;
    data[k * nrow + (k + 1)] = 1.0;
    let mat = SymmetricMatrix { n: nrow, data };
    let scalar = factor_frontal(&mat, ncol, true, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, ncol, true, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "2x2_inside_panel_may_delay");
}

/// W-2 2×2 inline — mixed-pivot panel pattern. Forces `{1, 1, 2, 1,
/// 2, 1}` over the first 8 pivots (one panel pass), with surrounding
/// 1×1's filling out the front. Tests the deferred Schur fallback
/// that walks pivot-pair-or-singleton outer with `axpy2` for pairs
/// and `axpy` for singletons. ncol=nrow=24 keeps the test fast while
/// staying above `PANEL_MIN_NCOL=8`.
#[test]
fn test_mixed_pivots_in_panel() {
    let params = default_params();
    let n = 24;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        data[j * n + j] = 1.0 + j as f64 * 0.001;
    }
    let mut state = 0x4242_BD22u64;
    for j in 0..n {
        for i in (j + 1)..n {
            data[j * n + i] = 1e-6 * rng_scalar(&mut state, i * n + j);
        }
    }
    // Pattern producing 2×2's at pivot positions (2, 3) and (4, 5),
    // with 1×1's everywhere else.
    for &k in &[2usize, 4] {
        data[k * n + k] = 0.0;
        data[(k + 1) * n + (k + 1)] = 0.0;
        data[k * n + (k + 1)] = 1.0;
    }
    let mat = SymmetricMatrix { n, data };
    let scalar = factor_frontal(&mat, n, false, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "mixed_pivots_in_panel");
}

/// Phase A2 (`dev/plans/dense-kernel-w2-2x2-swap.md`) — swap-required
/// 2×2 inside a panel. Constructs a fixture where BK at column `k=8`
/// triggers a 2×2 pivot whose argmax row `r=13` is NOT `k+1`. Scalar
/// applies `swap_rows_cols(a, n, 9, 13, perm)` and proceeds; the
/// blocked panel currently bails to scalar via
/// `PanelStatus::ScalarFallback` (Phase A excluded swap-2×2). After
/// Phase A2 the panel handles the swap inline. Either way, the
/// outputs must be byte-identical with scalar `factor_frontal`.
#[test]
fn test_swap_2x2_inside_panel_bare() {
    // Use block_size=8 so the panel boundary lands at column k=8 —
    // making the swap-2×2 the FIRST pivot of the second panel
    // (c == 0). The Phase A2 c==0 inline path catches this case;
    // mid-panel (c > 0) swap-2×2 still bails to scalar.
    let params = BunchKaufmanParams {
        block_size: 8,
        fma: false,
        ..default_params()
    };
    let n = 24;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        data[j * n + j] = 1.0 + j as f64 * 0.001;
    }
    let mut state = 0x5A2A_BD22u64;
    for j in 0..n {
        for i in (j + 1)..n {
            data[j * n + i] = 1e-6 * rng_scalar(&mut state, i * n + j);
        }
    }
    // Force swap-required 2×2 at column k=8.
    //   - akk = a[8,8] = 0 (forces 2×2 trigger)
    //   - gamma0 found at row r=13 (NOT k+1=9), so symmetric swap of
    //     rows/cols (9, 13) is required before the 2×2 can be taken
    //     with consecutive columns.
    //   - a[13,13] = 0.5 keeps swap-1×1 reject (arr < alpha * gamma_r)
    //   - akk = 0 keeps LAPACK-1×1 reject (akk * gamma_r < alpha * gamma0^2)
    //   - growth + det floor pass because gamma0 dominates.
    let k = 8;
    data[k * n + k] = 0.0;
    data[k * n + 13] = 5.0;
    data[k * n + (k + 1)] = 1e-3; // dwarfed by gamma0
    data[13 * n + 13] = 0.5;
    let mat = SymmetricMatrix { n, data };
    let scalar = factor_frontal(&mat, n, false, &params).unwrap();
    PANEL_DIAG_ENABLED.store(true, Ordering::Relaxed);
    panel_diag::reset();
    let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
    let snap = panel_diag::snapshot();
    PANEL_DIAG_ENABLED.store(false, Ordering::Relaxed);
    let swap_ok = snap
        .iter()
        .find(|(k, _)| *k == "inline_2x2_swap_ok")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    assert!(
        swap_ok >= 1,
        "swap_2x2_inside_panel_bare: expected inline_2x2_swap_ok>=1, got {} (panel still bailing). \
         Snapshot: {:?}",
        swap_ok,
        snap
    );
    assert_frontals_byte_identical(&scalar, &blocked, "swap_2x2_inside_panel_bare");
}

/// Phase A2 — chain of two swap-required 2×2 pivots inside the same
/// panel. After the first swap-2×2 commits at panel positions (0, 1)
/// (covering original cols `k=4` and `k=5` with a `9 ↔ 13`-style row
/// swap), the next pivot at panel position 2 (col `k=6`) also triggers
/// a swap-required 2×2. Tests that `peek_ahead_replay` correctly
/// brings BOTH target columns into scalar state at every panel
/// iteration, and that the perm tracks two consecutive swaps.
#[test]
fn test_swap_2x2_chain() {
    let params = default_params();
    let n = 32;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        data[j * n + j] = 1.0 + j as f64 * 0.001;
    }
    let mut state = 0x5A2A_C4A1u64;
    for j in 0..n {
        for i in (j + 1)..n {
            data[j * n + i] = 1e-6 * rng_scalar(&mut state, i * n + j);
        }
    }
    // First swap-required 2×2 at k=4: zero diag, gamma0 at row 11.
    {
        let k = 4;
        data[k * n + k] = 0.0;
        data[k * n + 11] = 5.0;
        data[k * n + (k + 1)] = 1e-3;
        data[11 * n + 11] = 0.5;
    }
    // Second swap-required 2×2 at k=6 (after the first 2×2 has
    // committed at panel positions 0,1): zero diag, gamma0 at row 17.
    // Row 17 is well clear of the prior swap target so the rows the
    // second swap references are independent.
    {
        let k = 6;
        data[k * n + k] = 0.0;
        data[k * n + 17] = 5.0;
        data[k * n + (k + 1)] = 1e-3;
        data[17 * n + 17] = 0.5;
    }
    let mat = SymmetricMatrix { n, data };
    let scalar = factor_frontal(&mat, n, false, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "swap_2x2_chain");
}

/// Phase A2 — swap-required 2×2 immediately followed by a clean 1×1
/// inside the same panel. After the 2×2 commits at panel positions
/// (0, 1) with a row swap, the next pivot at panel position 2 must
/// see the swapped state and pick a normal 1×1. Exercises the
/// `peek_ahead_column` correctness for the column AFTER a committed
/// 2×2-with-swap.
#[test]
fn test_swap_2x2_then_1x1() {
    let params = default_params();
    let n = 24;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        data[j * n + j] = 1.0 + j as f64 * 0.001;
    }
    let mut state = 0x5A2A_F00Du64;
    for j in 0..n {
        for i in (j + 1)..n {
            data[j * n + i] = 1e-6 * rng_scalar(&mut state, i * n + j);
        }
    }
    // Swap-required 2×2 at k=4: zero diag, gamma0 at row 9 (not k+1=5).
    let k = 4;
    data[k * n + k] = 0.0;
    data[k * n + 9] = 5.0;
    data[k * n + (k + 1)] = 1e-3;
    data[9 * n + 9] = 0.5;
    // Columns 6..n keep their strong diagonals (set in the loop above)
    // and tiny noise off-diagonals. After the swap at (5, 9), col 6
    // and beyond should pivot cleanly as 1×1.
    let mat = SymmetricMatrix { n, data };
    let scalar = factor_frontal(&mat, n, false, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "swap_2x2_then_1x1");
}

/// Step 5 — forced rejection under `may_delay == true` triggers the
/// SSIDS "break on first failure" path. The blocked panel must stop
/// cleanly at the delayed column, apply the deferred Schur to trailing
/// columns, and hand back `nelim < ncol` with the remaining columns
/// reported as `n_delayed`.
#[test]
fn test_may_delay_rejection_parity() {
    let params = default_params();
    let n = 128;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        data[j * n + j] = 1.0 + j as f64 * 0.001;
    }
    // Dominant off-diagonal at column 32, row 50 — same shape as
    // test_rejection_fallback but under may_delay=true. With
    // pivot_threshold=0.01 this column is rejected; under may_delay
    // that produces a Delayed outcome, so scalar breaks at k=32 with
    // nelim=32 and n_delayed=n-32. The blocked path must produce the
    // same (L, D, perm, inertia, contrib, nelim, n_delayed) byte-for
    // byte.
    data[32 * n + 50] = 1000.0;
    let mat = SymmetricMatrix { n, data };
    let scalar = factor_frontal(&mat, n, true, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, n, true, &params).unwrap();
    assert_frontals_byte_identical(&scalar, &blocked, "may_delay rejection");
}

/// Issue #36 — regression guard for the `MAX_N_ELIM` cap in
/// `apply_blocked_schur_panel`. Before #36 the cap was 64, matching
/// the default `block_size`, but any caller that raised `block_size`
/// past 64 hit a release-build panic ("range end index 128 out of
/// range for slice of length 64") in the deferred-Schur path.
///
/// This test exercises the previously-panicking code path by setting
/// `block_size = 128` on an SPD problem large enough that the panel
/// fully populates (`nrow > block_size`, `ncol > block_size`). It is
/// a *liveness* test (no panic) rather than a parity test: the
/// scalar kernel ignores `block_size`, so byte-identity here would
/// only re-prove the existing parity contract on a different size.
#[test]
fn test_issue36_block_size_128_no_panic() {
    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        block_size: 128,
        ..BunchKaufmanParams::default()
    };
    let n = 200;
    let mat = random_spd(n, 0x3600_0036);
    let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();
    assert_eq!(blocked.nelim, n, "issue36 nelim should be full n");
    assert_eq!(blocked.inertia.zero, 0, "issue36 SPD has no zero pivots");
}

/// Finding D2 (`dev/research/repo-review-2026-06-09.md`): the panel's
/// inline 2×2 accept path skipped the MA57 static-pivot perturbation
/// (`perturb_2x2_to_floor`) that `scalar_pivot_step` applies before the
/// growth/det gates, so with `static_pivot_floor > 0` a sub-floor 2×2
/// accepted inline diverged from the scalar path in `d_diag`,
/// `needs_refinement`, and possibly inertia — violating the module's
/// documented panel/scalar bit-parity contract.
///
/// Construction: an isolated antidiagonal 2×2 block
///   A[0,0] = A[1,1] = 0, A[1,0] = δ  (eigenvalues ±δ)
/// decoupled from a strictly diagonally-dominant SPD trailing block
/// (columns 2..n). BK selects a 2×2 at column 0 (akk = 0 < α·γ₀ with
/// γ₀ = δ); `gamma_r` picks up the block off-diagonal δ via
/// `symmetric_row_offdiag_max`'s left-of-diagonal sweep, so the no-swap
/// inline path is reached (arr = 0 < α·δ). The isolated block makes
/// rmax = tmax = 0, so the Duff-Reid growth bound passes; the SSIDS det
/// floor passes too (|detpiv| = δ ≥ cancel_floor = δ/2). With
/// δ = 1e-3 < static_pivot_floor = 1e-1 the scalar path perturbs the
/// block to the floor (and sets `needs_refinement`); pre-fix the panel
/// accepted it unperturbed.
///
/// Oracle: the scalar `factor_frontal` path, which already perturbs.
/// Pre-fix this fails on `d_diag[0]` (0.0 vs the perturbed floor) and
/// `needs_refinement` (false vs true); post-fix it is byte-identical.
/// `n=80` crosses the 64-column block boundary so the blocked kernel
/// genuinely uses the panel for column 0.
#[test]
fn test_d2_panel_inline_2x2_static_pivot_floor_parity() {
    let n = 80usize;
    let delta = 1e-3f64;
    let floor = 1e-1f64;
    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        static_pivot_floor: floor,
        ..BunchKaufmanParams::default()
    };

    // Lower-triangle, column-major: data[j*n + i] = A[i, j] for i >= j.
    let mut data = vec![0.0f64; n * n];
    // Isolated antidiagonal 2×2 at (0,1): eigenvalues ±δ, both |·| < floor.
    data[1] = delta; // A[1,0] = δ; A[0,0] = A[1,1] = 0 already.
                     // Strictly diagonally-dominant SPD trailing block on columns 2..n,
                     // fully decoupled from columns 0 and 1 (A[i,0] = A[i,1] = 0, i ≥ 2).
    let mut state = 0xD2_0000_1234u64;
    for j in 2..n {
        for i in (j + 1)..n {
            data[j * n + i] = rng_scalar(&mut state, i * n + j) * 0.01;
        }
        data[j * n + j] = (n as f64) + 1.0;
    }
    let mat = SymmetricMatrix { n, data };

    let scalar = factor_frontal(&mat, n, false, &params).unwrap();
    let blocked = factor_frontal_blocked(&mat, n, false, &params).unwrap();

    // Sanity: the construction must actually exercise the inline 2×2 with
    // a perturbed sub-floor block on the scalar (oracle) side, otherwise
    // the test proves nothing.
    assert_eq!(scalar.nelim, n, "scalar should eliminate all columns");
    assert!(
        scalar.needs_refinement,
        "scalar oracle must have perturbed the sub-floor 2×2 \
         (needs_refinement); construction no longer triggers D2"
    );
    // The perturbation lifts the small *eigenvalue* to the floor by
    // shifting both diagonals by τ = floor − δ, so the diagonal entry
    // lands at floor − δ (≈ 0.099), comfortably above the unperturbed
    // 0.0 the panel produced pre-fix.
    assert!(
        scalar.d_diag[0].abs() >= floor / 2.0,
        "scalar oracle d_diag[0]={} should be lifted well above 0 \
         (floor {}); construction no longer triggers D2",
        scalar.d_diag[0],
        floor
    );

    assert_frontals_byte_identical(&scalar, &blocked, "d2_static_pivot_floor");
}
