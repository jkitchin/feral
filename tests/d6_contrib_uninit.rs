//! D6 (dev/research/repo-review-2026-06-09.md): the contribution-block
//! extraction in `factor_frontal_in_place_with_scratch_impl` and
//! `factor_frontal_blocked_in_place_with_scratch` did
//! `contrib.reserve(cdim²)` then `unsafe { contrib.set_len(cdim²) }` and
//! materialized a `&mut [f64]` over the not-yet-initialized tail. Every
//! cell is written before read, so it produces correct results, but
//! calling `Vec::set_len` to expose elements that have not been
//! initialized violates the documented safety contract — exactly the kind
//! of thing Miri's uninitialized-memory tracking is meant to catch.
//!
//! This test drives both extraction sites with `ncol < nrow`, which forces
//! a non-empty contribution block (`cdim = nrow - nelim > 0`), and then
//! reads back **every** cell of `contrib`. Under a plain `cargo test` it is
//! a cheap finiteness regression guard; under
//! `cargo +nightly miri test --test d6_contrib_uninit` it is the actual
//! reproducing oracle — Miri reports UB at the extraction site if the
//! buffer's bytes are observed while still carved out as uninitialized.

use feral::dense::factor::{factor_frontal, factor_frontal_blocked};
use feral::{BunchKaufmanParams, SymmetricMatrix, ZeroPivotAction};

fn params() -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    }
}

/// Small SPD matrix: lower triangle of A = M·Mᵀ + n·I style diagonal
/// dominance so the leading `ncol` columns eliminate cleanly (no delay),
/// leaving a fully-populated `(nrow-ncol)×(nrow-ncol)` contribution block.
fn spd(n: usize) -> SymmetricMatrix {
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            // deterministic, no rng/Date: a smooth off-diagonal pattern
            let off = ((i + 1) as f64 * 0.1) - ((j + 1) as f64 * 0.07);
            data[j * n + i] = if i == j {
                (n as f64) + 2.0 + (i as f64)
            } else {
                off * 0.25
            };
        }
    }
    SymmetricMatrix { n, data }
}

/// Sum every cell of the contribution block. The point is the *read*: it
/// forces the full `cdim²` region — the memory exposed by `set_len` — to be
/// observed, so Miri evaluates every byte the extraction left in place.
fn sum_contrib(contrib: &[f64]) -> f64 {
    let mut s = 0.0;
    for &v in contrib {
        assert!(v.is_finite(), "contribution cell must be finite, got {v}");
        s += v;
    }
    s
}

#[test]
fn d6_scalar_contrib_block_fully_initialized() {
    let nrow = 5;
    let ncol = 2; // < nrow -> non-empty contribution block
    let mat = spd(nrow);
    let f = factor_frontal(&mat, ncol, false, &params()).expect("scalar factor");
    let cdim = f.contrib_dim;
    assert!(cdim > 0, "expected a non-empty contribution block");
    assert_eq!(f.contrib.len(), cdim * cdim, "contrib length == cdim²");
    // Read every cell — this is what Miri inspects.
    let s = sum_contrib(&f.contrib);
    assert!(s.is_finite());
}

#[test]
fn d6_blocked_contrib_block_fully_initialized() {
    let nrow = 5;
    let ncol = 2;
    let mat = spd(nrow);
    let f = factor_frontal_blocked(&mat, ncol, false, &params()).expect("blocked factor");
    let cdim = f.contrib_dim;
    assert!(cdim > 0, "expected a non-empty contribution block");
    assert_eq!(f.contrib.len(), cdim * cdim, "contrib length == cdim²");
    let s = sum_contrib(&f.contrib);
    assert!(s.is_finite());
}

/// Larger front so the blocked path crosses its panel boundary and the
/// contribution block spans multiple panels — exercises the same
/// extraction over a bigger `cdim²` region.
#[test]
fn d6_blocked_contrib_block_multi_panel() {
    let nrow = 12;
    let ncol = 8;
    let mat = spd(nrow);
    let f = factor_frontal_blocked(&mat, ncol, false, &params()).expect("blocked factor");
    let cdim = f.contrib_dim;
    assert!(cdim > 0, "expected a non-empty contribution block");
    assert_eq!(f.contrib.len(), cdim * cdim);
    let _ = sum_contrib(&f.contrib);
}
