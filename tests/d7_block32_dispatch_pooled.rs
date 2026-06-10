//! D7 (dev/research/repo-review-2026-06-09.md): the 32×32 fully-summed
//! front dispatch inside `factor_frontal_blocked_in_place_with_scratch`
//! routed through `factor_block32`, which delegates to the **public**
//! `factor_frontal` — re-running `validate()` (full NaN scan), allocating
//! an n×n working copy, and building a throwaway `FactorScratch`. That
//! bypasses the caller's pooled scratch which the enclosing W-3a in-place
//! path (issue #13) exists precisely to reuse, on the self-described
//! dominant 32×32 KKT front size.
//!
//! Reproduction signal: the in-place pooled path resizes `scratch.subdiag`
//! to `nrow` (`factor.rs:1593-1595`); the `factor_frontal` detour builds a
//! private `FactorScratch` internally and leaves the caller's scratch
//! pristine. So after a 32×32 dispatch with a fresh `FactorScratch`,
//! `scratch.subdiag.len()` is `nrow` iff the pooled path was used
//! (post-fix) and `0` iff the dispatch took the allocating detour
//! (pre-fix). The companion test pins bit-parity with the `factor_frontal`
//! oracle so the overhead removal cannot change numerics.

use feral::dense::factor::{
    factor_frontal, factor_frontal_blocked_in_place_with_scratch, FactorScratch, FrontalFactors,
};
use feral::{BunchKaufmanParams, SymmetricMatrix};

const BS: usize = 32;

/// Seeded indefinite 32×32 (splitmix64, deterministic, no rng/Date).
/// Diagonally non-dominant so BK genuinely exercises 1×1 / swap / 2×2.
fn seeded_indefinite_32() -> SymmetricMatrix {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
    };
    let mut data = vec![0.0f64; BS * BS];
    for j in 0..BS {
        for i in j..BS {
            data[j * BS + i] = if i == j {
                2.0 * next() - 1.0
            } else {
                next() - 0.5
            };
        }
    }
    SymmetricMatrix { n: BS, data }
}

fn assert_bits_equal(actual: &FrontalFactors, expected: &FrontalFactors) {
    assert_eq!(actual.nelim, expected.nelim, "nelim");
    assert_eq!(actual.n_delayed, expected.n_delayed, "n_delayed");
    assert_eq!(actual.inertia, expected.inertia, "inertia");
    assert_eq!(actual.perm, expected.perm, "perm");
    assert_eq!(actual.perm_inv, expected.perm_inv, "perm_inv");
    let pairs: [(&[f64], &[f64], &str); 4] = [
        (&actual.l, &expected.l, "l"),
        (&actual.d_diag, &expected.d_diag, "d_diag"),
        (&actual.d_subdiag, &expected.d_subdiag, "d_subdiag"),
        (&actual.contrib, &expected.contrib, "contrib"),
    ];
    for (a, e, tag) in pairs {
        assert_eq!(a.len(), e.len(), "{tag} length");
        for k in 0..a.len() {
            assert_eq!(a[k].to_bits(), e[k].to_bits(), "{tag}[{k}] mismatch");
        }
    }
}

/// RED on pre-fix code: the 32×32 dispatch must factor in place through
/// the caller's pooled scratch, not take the `factor_frontal` detour that
/// allocates a throwaway one.
#[test]
fn d7_block32_dispatch_uses_pooled_scratch() {
    let src = seeded_indefinite_32();
    let mut mat = SymmetricMatrix {
        n: BS,
        data: src.data.clone(),
    };
    let params = BunchKaufmanParams::default();
    let mut scratch = FactorScratch::new();
    assert_eq!(scratch.subdiag.len(), 0, "fresh scratch starts pristine");

    let _f =
        factor_frontal_blocked_in_place_with_scratch(&mut mat, BS, false, &params, &mut scratch)
            .expect("32x32 blocked factor");

    assert_eq!(
        scratch.subdiag.len(),
        BS,
        "the 32x32 dispatch must factor in place through the caller's \
         pooled scratch (subdiag resized to nrow); subdiag.len() == 0 means \
         it took the factor_frontal detour that validates, copies the front, \
         and builds a throwaway FactorScratch (D7)"
    );
}

/// Guard: routing the 32×32 dispatch through the in-place pooled path is
/// bit-identical to the documented `factor_frontal` oracle.
#[test]
fn d7_block32_dispatch_bit_identical_to_factor_frontal() {
    let src = seeded_indefinite_32();
    let params = BunchKaufmanParams::default();

    // Oracle: public `factor_frontal` on an untouched copy.
    let oracle_mat = SymmetricMatrix {
        n: BS,
        data: src.data.clone(),
    };
    let oracle = factor_frontal(&oracle_mat, BS, false, &params).expect("oracle");

    // Dispatch path (treats its own copy's data as scratch).
    let mut mat = SymmetricMatrix {
        n: BS,
        data: src.data.clone(),
    };
    let mut scratch = FactorScratch::new();
    let got =
        factor_frontal_blocked_in_place_with_scratch(&mut mat, BS, false, &params, &mut scratch)
            .expect("dispatch");

    assert_bits_equal(&got, &oracle);
}
