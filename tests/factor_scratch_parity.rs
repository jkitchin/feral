//! Issue #13 Phase A — bit-exact parity between
//! `factor_frontal_blocked_in_place` (owning wrapper) and
//! `factor_frontal_blocked_in_place_with_scratch` (caller-supplied
//! `FactorScratch`).
//!
//! Three cases per test matrix:
//!   (a) wrapper                          — fresh internal scratch
//!   (b) `_with_scratch` + fresh scratch  — first call on a default scratch
//!   (c) `_with_scratch` + warm scratch   — scratch reused from a prior call
//!       on a DIFFERENT-sized matrix (exercises clear/resize across nrow,bs)
//!
//! All three must produce byte-identical `(L, D_diag, D_subdiag, perm,
//! perm_inv, contrib, inertia, n_delayed, needs_refinement, n_rook_rescues)`.

use feral::dense::factor::{
    factor_frontal_blocked_in_place, factor_frontal_blocked_in_place_with_scratch, FactorScratch,
    FrontalFactors,
};
use feral::{BunchKaufmanParams, SymmetricMatrix, ZeroPivotAction};

fn rng(state: &mut u64, idx: usize) -> f64 {
    *state = state
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(idx as u64 + 1);
    let u = (*state >> 32) as u32 as f64;
    (u / (u32::MAX as f64)) * 2.0 - 1.0
}

fn random_indefinite(n: usize, seed: u64) -> SymmetricMatrix {
    let mut state = seed;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            data[j * n + i] = rng(&mut state, i * n + j);
        }
    }
    // Bias the diagonal to keep BK pivots well-conditioned but allow
    // some 2×2 candidates (no SPD shift).
    for j in 0..n {
        let d = &mut data[j * n + j];
        if d.abs() < 0.2 {
            *d = if *d >= 0.0 { 0.5 } else { -0.5 };
        }
    }
    SymmetricMatrix { n, data }
}

fn params() -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    }
}

fn assert_byte_identical(a: &FrontalFactors, b: &FrontalFactors, tag: &str) {
    assert_eq!(a.nrow, b.nrow, "{} nrow", tag);
    assert_eq!(a.ncol, b.ncol, "{} ncol", tag);
    assert_eq!(a.nelim, b.nelim, "{} nelim", tag);
    assert_eq!(a.n_delayed, b.n_delayed, "{} n_delayed", tag);
    assert_eq!(a.contrib_dim, b.contrib_dim, "{} contrib_dim", tag);
    assert_eq!(a.inertia, b.inertia, "{} inertia", tag);
    assert_eq!(
        a.needs_refinement, b.needs_refinement,
        "{} needs_refinement",
        tag
    );
    assert_eq!(a.n_rook_rescues, b.n_rook_rescues, "{} n_rook_rescues", tag);
    assert_eq!(a.perm, b.perm, "{} perm", tag);
    assert_eq!(a.perm_inv, b.perm_inv, "{} perm_inv", tag);
    assert_eq!(a.l.len(), b.l.len(), "{} l.len", tag);
    for (i, (x, y)) in a.l.iter().zip(b.l.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{} l[{}] {} vs {}", tag, i, x, y);
    }
    for (i, (x, y)) in a.d_diag.iter().zip(b.d_diag.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{} d_diag[{}]", tag, i);
    }
    for (i, (x, y)) in a.d_subdiag.iter().zip(b.d_subdiag.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{} d_subdiag[{}]", tag, i);
    }
    for (i, (x, y)) in a.contrib.iter().zip(b.contrib.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{} contrib[{}]", tag, i);
    }
}

fn one_case(n: usize, ncol: usize, seed: u64, warm: &mut FactorScratch, tag: &str) {
    let p = params();

    // (a) owning wrapper — fresh matrix from seed (the kernel mutates
    // `data` in place, so each call needs its own copy).
    let mut sym_a = random_indefinite(n, seed);
    let ff_wrapper = factor_frontal_blocked_in_place(&mut sym_a, ncol, true, &p).unwrap();

    // (b) _with_scratch + fresh scratch
    let mut sym_b = random_indefinite(n, seed);
    let mut fresh = FactorScratch::new();
    let ff_fresh =
        factor_frontal_blocked_in_place_with_scratch(&mut sym_b, ncol, true, &p, &mut fresh)
            .unwrap();
    assert_byte_identical(&ff_wrapper, &ff_fresh, &format!("{} fresh", tag));

    // (c) _with_scratch + warm scratch (already used by caller on a
    //     possibly-different-size matrix). The kernel must clear+resize
    //     and produce byte-identical output.
    let mut sym_c = random_indefinite(n, seed);
    let ff_warm =
        factor_frontal_blocked_in_place_with_scratch(&mut sym_c, ncol, true, &p, warm).unwrap();
    assert_byte_identical(&ff_wrapper, &ff_warm, &format!("{} warm", tag));

    // (d) Phase C: pre-seed the contrib_pool with a Vec of non-target
    //     size and garbage values. The kernel must clear+resize and
    //     produce byte-identical output. This exercises the pool-hot
    //     path: the slot is occupied, existing capacity != cdim*cdim,
    //     residual bytes must not leak into the new contrib.
    let mut sym_d = random_indefinite(n, seed);
    let mut pooled = FactorScratch::new();
    pooled.contrib_pool = Some(vec![-3.14e7; (n + 5) * (n + 5)]);
    let ff_pool =
        factor_frontal_blocked_in_place_with_scratch(&mut sym_d, ncol, true, &p, &mut pooled)
            .unwrap();
    assert_byte_identical(&ff_wrapper, &ff_pool, &format!("{} pool-hot", tag));
}

#[test]
fn factor_scratch_parity_size_sweep() {
    // Pre-warm the scratch with a 19×19 call to ensure subsequent cases
    // exercise the "scratch holds prior capacity, possibly different size"
    // path. 19 is chosen deliberately to be unequal to any later n.
    let mut warm = FactorScratch::new();
    {
        let p = params();
        let mut sym = random_indefinite(19, 0xA5A5_5A5A_2026_0512_u64);
        let _ = factor_frontal_blocked_in_place_with_scratch(&mut sym, 19, true, &p, &mut warm)
            .unwrap();
    }

    // Range of (n, ncol) chosen to cover:
    //   - scalar-tail path (ncol < PANEL_MIN_NCOL = 8)
    //   - panel path single-iteration (ncol == bs)
    //   - panel path multi-iteration (ncol > bs)
    //   - ncol < nrow (frontal partial elimination — the multifrontal case)
    let cases: &[(usize, usize, u64)] = &[
        (4, 4, 0x0001_2026_0512_AAAA),
        (8, 8, 0x0002_2026_0512_BBBB),
        (16, 12, 0x0003_2026_0512_CCCC),
        (32, 32, 0x0004_2026_0512_DDDD),
        (64, 48, 0x0005_2026_0512_EEEE),
        (96, 64, 0x0006_2026_0512_FFFF),
        (128, 96, 0x0007_2026_0512_1111),
    ];

    for &(n, ncol, seed) in cases {
        one_case(n, ncol, seed, &mut warm, &format!("n={}, ncol={}", n, ncol));
    }
}

#[test]
fn factor_scratch_parity_repeated_calls_same_scratch() {
    // The same warm scratch is reused across many calls of varying size.
    // Every call must remain byte-identical to a fresh-scratch invocation.
    let mut warm = FactorScratch::new();
    let sizes: &[(usize, usize, u64)] = &[
        (16, 16, 0xBEEF_FACE_2026_0512_u64),
        (32, 24, 0xDEAD_BABE_2026_0512_u64),
        (8, 8, 0xCAFE_F00D_2026_0512_u64),
        (64, 64, 0x1234_5678_2026_0512_u64),
        (4, 4, 0x9A77_E11E_2026_0512_u64),
        (96, 80, 0xABCD_EF01_2026_0512_u64),
    ];
    for &(n, ncol, seed) in sizes {
        one_case(
            n,
            ncol,
            seed,
            &mut warm,
            &format!("repeat n={}, ncol={}", n, ncol),
        );
    }
}
