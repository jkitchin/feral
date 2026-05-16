//! Issue #10 Phase 1: MAXFROMM TPP acceleration parity tests.
//!
//! `TppMethod::Maxfromm` reuses the column-AMAX captured as a byproduct
//! of the previous 1×1 rank-1 trailing update. The acceptance predicate
//! is bit-identical to `Plain` (`|d| >= alpha * gamma0` plus the
//! column-relative threshold check), so `(L, D, perm, inertia, contrib)`
//! must be byte-identical between the two configs on every front shape
//! and every numerical regime. Parity = correctness; perf is gated by
//! Phase 2 corpus validation in a separate diag binary.

use feral::dense::factor::{factor_frontal, factor_frontal_blocked, FrontalFactors, TppMethod};
use feral::{BunchKaufmanParams, SymmetricMatrix};

fn rng_scalar(state: &mut u64, idx: usize) -> f64 {
    *state = state
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(idx as u64 + 1);
    let u = (*state >> 32) as u32 as f64;
    (u / (u32::MAX as f64)) * 2.0 - 1.0
}

fn random_spd(n: usize, seed: u64) -> SymmetricMatrix {
    let mut state = seed;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            data[j * n + i] = rng_scalar(&mut state, i * n + j);
        }
    }
    for j in 0..n {
        data[j * n + j] = data[j * n + j].abs() + (n as f64) + 1.0;
    }
    SymmetricMatrix { n, data }
}

fn random_indefinite(n: usize, seed: u64) -> SymmetricMatrix {
    let mut state = seed;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            data[j * n + i] = rng_scalar(&mut state, i * n + j);
        }
        // Small diagonal forces BK into the 2×2 branch periodically,
        // which clears the MAXFROMM cache and forces a full re-scan
        // on the next pivot. Exercises the cache-invalidation path.
        data[j * n + j] *= 0.05;
    }
    SymmetricMatrix { n, data }
}

fn assert_byte_identical(plain: &FrontalFactors, mfm: &FrontalFactors, tag: &str) {
    assert_eq!(plain.nrow, mfm.nrow, "{} nrow", tag);
    assert_eq!(plain.ncol, mfm.ncol, "{} ncol", tag);
    assert_eq!(plain.nelim, mfm.nelim, "{} nelim", tag);
    assert_eq!(plain.n_delayed, mfm.n_delayed, "{} n_delayed", tag);
    assert_eq!(plain.contrib_dim, mfm.contrib_dim, "{} contrib_dim", tag);
    assert_eq!(plain.inertia, mfm.inertia, "{} inertia", tag);
    assert_eq!(
        plain.needs_refinement, mfm.needs_refinement,
        "{} needs_refinement",
        tag
    );
    assert_eq!(plain.perm, mfm.perm, "{} perm", tag);
    assert_eq!(plain.d_diag, mfm.d_diag, "{} d_diag", tag);
    assert_eq!(plain.d_subdiag, mfm.d_subdiag, "{} d_subdiag", tag);
    assert_eq!(plain.l, mfm.l, "{} l", tag);
    assert_eq!(plain.contrib, mfm.contrib, "{} contrib", tag);
}

fn params_plain() -> BunchKaufmanParams {
    BunchKaufmanParams {
        tpp_method: TppMethod::Plain,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    }
}

fn params_maxfromm() -> BunchKaufmanParams {
    BunchKaufmanParams {
        tpp_method: TppMethod::Maxfromm,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    }
}

/// SPD across size sweep through the scalar `factor_frontal` path. Every
/// pivot is 1×1 with the diagonal dominating, so MAXFROMM short-circuits
/// on every iteration — maximum cache use, maximum chance to expose a
/// divergence.
#[test]
fn maxfromm_spd_size_sweep_scalar() {
    let plain = params_plain();
    let mfm = params_maxfromm();
    for &n in &[3usize, 5, 8, 16, 31, 33, 64, 100, 200] {
        let mat = random_spd(n, 0xDEAD_F00D ^ n as u64);
        let a = factor_frontal(&mat, n, false, &plain).unwrap();
        let b = factor_frontal(&mat, n, false, &mfm).unwrap();
        assert_byte_identical(&a, &b, &format!("spd n={n}"));
    }
}

/// Indefinite stress: 2×2 pivots clear the MAXFROMM cache mid-loop, so
/// the post-2×2 pivot must fall back to a full AMAX scan and still produce
/// the byte-identical factorization.
#[test]
fn maxfromm_indefinite_size_sweep_scalar() {
    let plain = params_plain();
    let mfm = params_maxfromm();
    for &n in &[5usize, 16, 33, 50, 70, 100] {
        let mat = random_indefinite(n, 0xBEEF_CAFE ^ n as u64);
        let a = factor_frontal(&mat, n, false, &plain).unwrap();
        let b = factor_frontal(&mat, n, false, &mfm).unwrap();
        assert_byte_identical(&a, &b, &format!("indef n={n}"));
    }
}

/// Blocked path — `factor_frontal_blocked` uses `scalar_pivot_step` in
/// the scalar tail and as the post-panel fallback. Both call sites must
/// thread the MAXFROMM cache correctly. Size 100 exceeds the 64-block
/// boundary and forces a partial last panel.
#[test]
fn maxfromm_spd_blocked_parity() {
    let plain = params_plain();
    let mfm = params_maxfromm();
    for &n in &[32usize, 64, 65, 100, 200] {
        let mat = random_spd(n, 0xC0DE_BABE ^ n as u64);
        let a = factor_frontal_blocked(&mat, n, false, &plain).unwrap();
        let b = factor_frontal_blocked(&mat, n, false, &mfm).unwrap();
        assert_byte_identical(&a, &b, &format!("blocked spd n={n}"));
    }
}

/// Indefinite blocked: hits the post-panel scalar fallback because 2×2
/// pivots force a partial panel; exercises MAXFROMM through the
/// `ScalarFallback` branch.
#[test]
fn maxfromm_indefinite_blocked_parity() {
    let plain = params_plain();
    let mfm = params_maxfromm();
    for &n in &[33usize, 70, 100] {
        let mat = random_indefinite(n, 0x7777_AAAA ^ n as u64);
        let a = factor_frontal_blocked(&mat, n, false, &plain).unwrap();
        let b = factor_frontal_blocked(&mat, n, false, &mfm).unwrap();
        assert_byte_identical(&a, &b, &format!("blocked indef n={n}"));
    }
}

/// Frontal `ncol < nrow` — the MAXFROMM scan ranges across rows
/// `[k+1..nrow]` (not `..ncol`), so it must include the contribution-only
/// rows in the column-max. Failure mode if the scan is wrong: gamma0 is
/// understated, the BK test trivially passes, and a pivot that should be
/// rejected gets through (factor diverges from Plain).
#[test]
fn maxfromm_ncol_lt_nrow() {
    let plain = params_plain();
    let mfm = params_maxfromm();
    for &(nrow, ncol) in &[(80usize, 48), (60, 30), (50, 25)] {
        let mat = random_spd(nrow, 0x4567_89AB ^ nrow as u64);
        let a = factor_frontal(&mat, ncol, false, &plain).unwrap();
        let b = factor_frontal(&mat, ncol, false, &mfm).unwrap();
        assert_byte_identical(&a, &b, &format!("ncol<nrow {nrow}x{ncol}"));
    }
}
