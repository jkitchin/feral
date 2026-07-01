//! Issue #99, Lever 3: opt-in per-front FMA size gate.
//!
//! `BunchKaufmanParams::fma_min_front_area = Some(t)` routes a front to
//! the FMA trailing-Schur kernels when `nrow * ncol >= t`, even when the
//! global `fma` flag is off — while leaving smaller fronts on the
//! bit-exact `*_nofma` path. These tests pin that behavior:
//!
//!  1. gate-armed + large front  ==  explicit `fma = true`   (bit-for-bit)
//!  2. gate-armed + small front  ==  default `fma = false`   (bit-for-bit)
//!  3. gate does not change inertia on a well-conditioned front.
//!
//! The oracle for (1) and (2) is another feral factorization with the
//! effective flag set explicitly — a self-consistency contract, not an
//! external numeric oracle, so no tolerance is involved: the assertion is
//! byte equality (`f64::to_bits`).

use feral::dense::factor::factor_frontal_blocked;
use feral::{BunchKaufmanParams, SymmetricMatrix};

/// Deterministic SplitMix64 → [-1, 1). Reproducible fixture, no `rand`.
fn mix(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// Symmetric indefinite front (alternating-sign dominant diagonal, small
/// off-diagonal noise) that factors through the all-1×1-pivot fast path.
fn build_front(n: usize) -> SymmetricMatrix {
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            data[j * n + i] = if i == j {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sign * (n as f64) + mix(&mut s)
            } else {
                0.30 * mix(&mut s)
            };
        }
    }
    SymmetricMatrix { n, data }
}

fn factor(
    matrix: &SymmetricMatrix,
    params: &BunchKaufmanParams,
) -> feral::dense::factor::FrontalFactors {
    // Root front: every column fully summed, no delayed pivots.
    factor_frontal_blocked(matrix, matrix.n, false, params).expect("factor failed")
}

fn assert_bits_eq(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: element {i} differs ({x} vs {y})"
        );
    }
}

fn assert_factors_bit_identical(
    x: &feral::dense::factor::FrontalFactors,
    y: &feral::dense::factor::FrontalFactors,
    what: &str,
) {
    assert_bits_eq(&x.l, &y.l, &format!("{what}: l"));
    assert_bits_eq(&x.d_diag, &y.d_diag, &format!("{what}: d_diag"));
    assert_bits_eq(&x.d_subdiag, &y.d_subdiag, &format!("{what}: d_subdiag"));
    assert_bits_eq(&x.contrib, &y.contrib, &format!("{what}: contrib"));
    assert_eq!(x.perm, y.perm, "{what}: perm");
    assert_eq!(x.inertia, y.inertia, "{what}: inertia");
}

#[test]
fn gate_on_large_front_matches_explicit_fma() {
    let n = 96;
    let matrix = build_front(n);

    // Gate armed with a threshold this front clears (n*n = 9216 >= 1).
    let gated = factor(
        &matrix,
        &BunchKaufmanParams {
            fma: false,
            fma_min_front_area: Some(1),
            ..BunchKaufmanParams::default()
        },
    );
    // Oracle: the same front factored with FMA turned on globally.
    let explicit_fma = factor(
        &matrix,
        &BunchKaufmanParams {
            fma: true,
            ..BunchKaufmanParams::default()
        },
    );
    assert_factors_bit_identical(&gated, &explicit_fma, "gate-on-large vs explicit-fma");
}

#[test]
fn gate_below_threshold_matches_nofma_default() {
    let n = 96;
    let matrix = build_front(n);

    // Gate armed but threshold unreachable → must stay on nofma.
    let gated_but_small = factor(
        &matrix,
        &BunchKaufmanParams {
            fma: false,
            fma_min_front_area: Some(usize::MAX),
            ..BunchKaufmanParams::default()
        },
    );
    let nofma_default = factor(&matrix, &BunchKaufmanParams::default());
    assert_factors_bit_identical(
        &gated_but_small,
        &nofma_default,
        "gate-below-threshold vs nofma-default",
    );
}

#[test]
fn gate_preserves_inertia() {
    let n = 96;
    let matrix = build_front(n);
    let nofma = factor(&matrix, &BunchKaufmanParams::default());
    let gated = factor(
        &matrix,
        &BunchKaufmanParams {
            fma: false,
            fma_min_front_area: Some(1),
            ..BunchKaufmanParams::default()
        },
    );
    assert_eq!(
        nofma.inertia, gated.inertia,
        "FMA size gate must not change inertia on a well-conditioned front"
    );
    // Sanity: this fixture is genuinely indefinite (both signs present).
    assert!(nofma.inertia.positive > 0 && nofma.inertia.negative > 0);
}

/// The threshold is on `nrow * ncol`, so a boundary value must gate
/// exactly: area == t fires (>=), area == t+1'th smaller front does not.
#[test]
fn gate_threshold_is_area_nrow_times_ncol() {
    let n = 96;
    let matrix = build_front(n);
    let area = n * n; // ncol == nrow == n for a root front.

    // Exactly at the boundary: `>=` means it fires → matches fma=true.
    let at_boundary = factor(
        &matrix,
        &BunchKaufmanParams {
            fma_min_front_area: Some(area),
            ..BunchKaufmanParams::default()
        },
    );
    let explicit_fma = factor(
        &matrix,
        &BunchKaufmanParams {
            fma: true,
            ..BunchKaufmanParams::default()
        },
    );
    assert_factors_bit_identical(&at_boundary, &explicit_fma, "area==t boundary fires");

    // One above the area: does not fire → matches nofma.
    let just_above = factor(
        &matrix,
        &BunchKaufmanParams {
            fma_min_front_area: Some(area + 1),
            ..BunchKaufmanParams::default()
        },
    );
    let nofma_default = factor(&matrix, &BunchKaufmanParams::default());
    assert_factors_bit_identical(&just_above, &nofma_default, "area<t does not fire");
}
