//! Phase 2.4.3 tests for rook pivoting as a rescue path.
//!
//! These tests pin down the contract of `FrontalFactors::n_rook_rescues`
//! and the rook-rescue splice in `try_reject_1x1_frontal`. At Step 3
//! of the plan only Test 1 is expected to pass (the stubbed
//! `rook_rescue` returns `None`, so no rescue ever fires and the
//! telemetry field stays at 0). Tests 2 and 3 are written against the
//! **post-splice** contract and are `#[ignore]`d with a reason pointing
//! at Step 5 of `dev/plans/phase-2.4.3-rook-rescue.md`; they flip to
//! GREEN when the splice lands and the kernel is implemented.
//!
//! Hand-traced matrices
//! --------------------
//!
//! Test 2 (1×1 rescue): a 4×4 block-structured matrix where BK-partial's
//! LAPACK-extension case (`a_kk * gamma_r >= alpha * gamma_0^2`)
//! accepts the (0,0) diagonal at `d = 0.008` with `col_max = 1`, so the
//! column-relative threshold `0.01 * col_max = 0.01` rejects `|d|`.
//! Rook's alternating scan walks `(0,0) -> (0,1) -> row 2`, where
//! `|A[2,2]| = 500 >> alpha * gamma_row(row 2) = 64`, so Step 6 of the
//! rook algorithm accepts row 2 as a 1×1 via swap.
//!
//! Test 3 (2×2 rescue): a 5×5 extension of the same pattern with a
//! `1e4` off-diagonal block at {1,2}. Rook walks
//! `(0,0) -> (0,1) -> (2,1)` and Step 7 accepts the (2,1) off-diagonal
//! as a 2×2 block. The trailing 2×2 at rows {3,4} is trivially SPD.
//!
//! Both matrices have been traced by hand following the algorithm in
//! `dev/research/rook-rescue.md` §3. The LDLᵀ reconstruction under
//! `L·D·Lᵀ ≈ A` is the numerical acceptance criterion; inertia is
//! checked against a hand-computed sign pattern.

#![allow(clippy::erasing_op, clippy::identity_op)]
use feral::dense::factor::{factor_frontal, factor_frontal_blocked};
use feral::{BunchKaufmanParams, SymmetricMatrix, ZeroPivotAction};

/// Params that exercise the rook-rescue path: `pivot_threshold = 0.01`
/// (the MUMPS/SSIDS default that requires rescue or delay on small
/// pivots) with `ForceAccept` as the on-zero-pivot fallback so we can
/// detect the "rook failed, fell through to force-accept" case via
/// `needs_refinement == true`.
fn rook_params() -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    }
}

fn spd_shifted(n: usize) -> SymmetricMatrix {
    // Construct a diagonally-dominant random-ish symmetric matrix by
    // filling the lower triangle with a deterministic pattern and
    // shifting the diagonal by (n+1). Matches the pattern used in
    // tests/blocked_ldlt.rs so Test 1 stays consistent with the
    // existing SPD-identity size sweep.
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            let v = ((i * 7 + j * 13) % 11) as f64 - 5.0;
            data[j * n + i] = v;
        }
    }
    for j in 0..n {
        data[j * n + j] = data[j * n + j].abs() + (n as f64) + 1.0;
    }
    SymmetricMatrix { n, data }
}

/// Reconstruct A from (L, D, perm) and return max|A_rec - A_orig|.
/// Used by Tests 2 and 3 as the numerical acceptance criterion.
fn reconstruct_residual(ff: &feral::dense::factor::FrontalFactors, orig_lower: &[f64]) -> f64 {
    let n = ff.nrow;
    let nelim = ff.nelim;
    debug_assert_eq!(nelim, n, "reconstruction assumes full elimination");

    // Rebuild D (block diagonal with 2×2 blocks where subdiag != 0).
    let mut d = vec![0.0f64; n * n];
    let mut j = 0;
    while j < n {
        d[j * n + j] = ff.d_diag[j];
        if j + 1 < n && ff.d_subdiag[j] != 0.0 {
            d[j * n + (j + 1)] = ff.d_subdiag[j];
            d[(j + 1) * n + j] = ff.d_subdiag[j];
            d[(j + 1) * n + (j + 1)] = ff.d_diag[j + 1];
            j += 2;
        } else {
            j += 1;
        }
    }

    // A_perm = L · D · Lᵀ (column-major).
    let mut ld = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..n {
                s += ff.l[p * n + i] * d[j * n + p];
            }
            ld[j * n + i] = s;
        }
    }
    let mut a_perm = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..n {
                s += ld[p * n + i] * ff.l[p * n + j];
            }
            a_perm[j * n + i] = s;
        }
    }

    // Apply inverse permutation: A_perm[perm_inv[i], perm_inv[j]] == A[i, j].
    // Compare lower triangle of recovered A against the original lower triangle.
    let mut max_err = 0.0f64;
    for j in 0..n {
        for i in j..n {
            let ri = ff.perm_inv[i];
            let rj = ff.perm_inv[j];
            let (pi, pj) = if ri >= rj { (ri, rj) } else { (rj, ri) };
            let recovered = a_perm[pj * n + pi];
            let original = orig_lower[j * n + i];
            let diff = (recovered - original).abs();
            if diff > max_err {
                max_err = diff;
            }
        }
    }
    max_err
}

/// Test 1 — SPD identity. SPD matrices of the full size sweep never
/// trigger the column-relative threshold, so the rook-rescue path is
/// never entered. `n_rook_rescues` must be 0 after factorization.
///
/// This is the "zero cost on easy matrices" gate from the plan. Will
/// continue to pass as-is after Step 5 because no rescue fires on SPD.
#[test]
fn test_rook_identity_on_spd() {
    let params = rook_params();
    for n in [4usize, 8, 16, 32, 64] {
        let matrix = spd_shifted(n);

        let scalar = factor_frontal(&matrix, n, false, &params).expect("scalar frontal");
        assert_eq!(
            scalar.n_rook_rescues, 0,
            "SPD n={} must not invoke rook rescue (scalar path)",
            n
        );

        let blocked = factor_frontal_blocked(&matrix, n, false, &params).expect("blocked frontal");
        assert_eq!(
            blocked.n_rook_rescues, 0,
            "SPD n={} must not invoke rook rescue (blocked path)",
            n
        );
    }
}

/// Test 2 — rook rescues a BK-rejected 1×1 via row swap.
///
/// Hand-trace (see module-level doc):
///   BK at k=0: `a_kk = 0.008`, `gamma_0 = 1`, `gamma_r = 100` in row 1;
///   LAPACK-extension case accepts 1×1 at k=0. Column-relative test
///   rejects `0.008 < 0.01 * col_max = 0.01`.
///   Rook walks `(0,0) -> (0,1) -> row 2` and accepts `|A[2,2]| = 500`
///   as a 1×1 (Step 6 of the rook algorithm).
#[test]
fn test_rook_rescues_delayed_1x1() {
    // Matrix A (symmetric, lower triangle stored column-major):
    //   [ 0.008   1      0      0   ]
    //   [ 1       0.5    100    0   ]
    //   [ 0       100    500    0   ]
    //   [ 0       0      0      1   ]
    let n = 4;
    let mut data = vec![0.0f64; n * n];
    data[0 * n + 0] = 0.008;
    data[0 * n + 1] = 1.0;
    data[1 * n + 1] = 0.5;
    data[1 * n + 2] = 100.0;
    data[2 * n + 2] = 500.0;
    data[3 * n + 3] = 1.0;
    let matrix = SymmetricMatrix { n, data };
    let orig_lower = matrix.data.clone();

    let params = rook_params();
    let ff = factor_frontal(&matrix, n, false, &params).expect("factor must complete");

    assert!(
        ff.n_rook_rescues >= 1,
        "rook rescue must fire at least once (got n_rook_rescues = {})",
        ff.n_rook_rescues
    );
    assert!(
        !ff.needs_refinement,
        "rook rescue must succeed without force-accept fallback \
         (needs_refinement = true means force-accept fired)"
    );
    assert_eq!(ff.nelim, n, "full elimination expected after rescue");

    let err = reconstruct_residual(&ff, &orig_lower);
    assert!(
        err < 1e-9,
        "LDLᵀ reconstruction residual {} exceeds 1e-9",
        err
    );

    // Inertia is pivot-order invariant (Sylvester's law), so rook
    // rescue must produce the same (pos, neg, zero) counts that any
    // correct LDLᵀ gives. Hand-computation on block {0,1,2}:
    //   det = 0.008*(0.5*500 - 100*100) - 1*(1*500 - 100*0)
    //       = 0.008*(-9750) - 500 = -578 < 0
    //   trace = 0.008 + 0.5 + 500 = 500.508 > 0
    // Negative det with positive trace on a 3×3 implies exactly one
    // negative eigenvalue (odd number < 3). Block {3} contributes
    // (1, 0, 0). Total: (3, 1, 0). Verified against current BK-partial
    // + force-accept run, which produces the same counts (force-accept
    // happens to pick correct signs on this matrix).
    assert_eq!(ff.inertia.positive, 3, "expected 3 positive pivots");
    assert_eq!(ff.inertia.negative, 1, "expected 1 negative pivot");
    assert_eq!(ff.inertia.zero, 0, "expected 0 zero pivots");
}

/// Test 3 — rook rescues a BK-rejected 1×1 via a 2×2 block pivot.
///
/// Hand-trace (see module-level doc):
///   Same column-0 rejection as Test 2. Rook walks `(0,0) -> (0,1) ->
///   (2,1)` and accepts (2,1) as a 2×2 block via Step 7 of the
///   algorithm (`|A[2,1]| = 1e4 >= alpha * gamma_row = 6400`).
///   The trailing 2×2 at rows {3,4} is SPD and factors trivially.
#[test]
fn test_rook_rescues_delayed_2x2() {
    // Matrix A (5×5 symmetric, lower triangle):
    //   [ 0.008   1      0      0    0  ]
    //   [ 1       0.1    1e4    0    0  ]
    //   [ 0       1e4    0.1    0    0  ]
    //   [ 0       0      0      1    0  ]
    //   [ 0       0      0      0    1  ]
    let n = 5;
    let mut data = vec![0.0f64; n * n];
    data[0 * n + 0] = 0.008;
    data[0 * n + 1] = 1.0;
    data[1 * n + 1] = 0.1;
    data[1 * n + 2] = 1.0e4;
    data[2 * n + 2] = 0.1;
    data[3 * n + 3] = 1.0;
    data[4 * n + 4] = 1.0;
    let matrix = SymmetricMatrix { n, data };
    let orig_lower = matrix.data.clone();

    let params = rook_params();
    let ff = factor_frontal(&matrix, n, false, &params).expect("factor must complete");

    assert!(
        ff.n_rook_rescues >= 1,
        "rook rescue must fire at least once (got n_rook_rescues = {})",
        ff.n_rook_rescues
    );
    assert!(
        !ff.needs_refinement,
        "rook 2×2 rescue must succeed without force-accept fallback"
    );
    assert_eq!(ff.nelim, n, "full elimination expected after rescue");

    let err = reconstruct_residual(&ff, &orig_lower);
    assert!(
        err < 1e-6,
        "LDLᵀ reconstruction residual {} exceeds 1e-6 \
         (larger tolerance reflects the 1e4 entry)",
        err
    );

    // Inertia (Sylvester-invariant). Block {0,1,2}:
    //   det = 0.008*(0.01 - 1e8) - 1*(0.1 - 0) ≈ -8e5 < 0
    //   trace = 0.208 > 0
    // Block {1,2} eigenvalues are ±1e4 (from the dominant 1e4 off-
    // diagonal); the third eigenvalue sits near +trace ≈ +0.2 > 0.
    // So the 3×3 has inertia (2, 1, 0). Blocks {3} and {4} each add
    // (1, 0, 0). Total: (4, 1, 0). Verified against the current
    // BK-partial + force-accept run on this matrix.
    assert_eq!(ff.inertia.positive, 4, "expected 4 positive pivots");
    assert_eq!(ff.inertia.negative, 1, "expected 1 negative pivot");
    assert_eq!(ff.inertia.zero, 0, "expected 0 zero pivots");
}

// ---------------------------------------------------------------------------
// Test 4 — Sylvester inertia preservation under rook rescue (Phase 2.4.3
// plan §"Test 4").
//
// Property: Sylvester's law of inertia says the inertia of a symmetric
// matrix is invariant under congruence P·A·Pᵀ, regardless of the pivot
// sequence. Rook rescue changes the pivot sequence on ill-conditioned
// matrices but must not change the inertia. We verify this as a
// property test: for each random indefinite matrix, factor with a
// permissive threshold (u = 1e-14, never rejects) and a heavy threshold
// (u = 0.1, forces many rescues), then compare inertia.
//
// Secondary checks:
//   - LDLᵀ reconstruction holds for the heavy-threshold factor
//     (rook rescue must produce correct L and D, not just correct signs).
//   - `needs_refinement == false` across the panel (rook should rescue
//     cleanly without falling through to force-accept).
//   - At least one rescue fires across the whole panel (otherwise we
//     are not exercising the rescue path at all).
// ---------------------------------------------------------------------------

/// Xorshift64 PRNG, identical to the one in tests/property_tests.rs.
/// Inlined here so the test file stays self-contained.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let t = (self.next_u64() as f64) / (u64::MAX as f64);
        lo + t * (hi - lo)
    }
}

/// Random symmetric indefinite matrix with a pathological small-diagonal
/// pattern designed to exercise BK-partial's column-relative threshold
/// rejection path. Construction:
///
///   1. Generate lower-triangular M with entries in [−5, 5].
///   2. A = M·Mᵀ.
///   3. Shift diagonal by −mean(diag(A)) to produce indefinite inertia.
///   4. Scale ~1/3 of diagonals by 1e-3 — their magnitude now falls below
///      `pivot_threshold * col_max` for u ≥ 0.01, forcing the BK kernel
///      to reject or delay those pivots. Off-diagonals stay untouched,
///      so the column-max is preserved and rook can find a valid pivot
///      by walking the row.
///
/// The resulting matrices are genuinely indefinite (step 3) and force
/// the rescue path (step 4), so inertia matches between the reference
/// and rescue factorizations, and `n_rook_rescues > 0` across the panel.
fn random_indefinite(n: usize, rng: &mut Rng) -> SymmetricMatrix {
    let mut m = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            m[j * n + i] = rng.uniform(-5.0, 5.0);
        }
    }
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            let mut s = 0.0;
            for k in 0..n {
                s += m[k * n + i] * m[k * n + j];
            }
            data[j * n + i] = s + if i == j { 0.01 } else { 0.0 };
        }
    }
    let mean_diag: f64 = (0..n).map(|i| data[i * n + i]).sum::<f64>() / (n as f64);
    for i in 0..n {
        data[i * n + i] -= mean_diag;
    }
    // Scale ~1/3 of diagonals down by 1e-3 to force BK rejection on
    // those pivots. Seed-deterministic via the rng.
    for i in 0..n {
        if rng.uniform(0.0, 1.0) < 0.33 {
            data[i * n + i] *= 1e-3;
        }
    }
    SymmetricMatrix { n, data }
}

/// Frobenius norm of a SymmetricMatrix's lower triangle stored at `data`.
/// Used only to scale the reconstruction tolerance.
fn fro_norm(data: &[f64], n: usize) -> f64 {
    let mut s = 0.0f64;
    for j in 0..n {
        s += data[j * n + j] * data[j * n + j];
        for i in (j + 1)..n {
            let v = data[j * n + i];
            s += 2.0 * v * v;
        }
    }
    s.sqrt()
}

#[test]
fn test_rook_preserves_inertia_random_indefinite() {
    let mut rng = Rng::new(0xC0FF_EE42);

    // Reference params: permissive threshold; the column-relative test
    // is `(u * col_max).max(zero_tol)` so u ≈ 0 degenerates to zero_tol.
    // Well-conditioned pivots clear that trivially and the rook path is
    // never entered — inertia here is the BK-partial reference.
    let ref_params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.0,
        ..BunchKaufmanParams::default()
    };
    // Rescue params: heavy threshold forces many BK rejections, which
    // drives the rook rescue path.
    let rescue_params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.1,
        ..BunchKaufmanParams::default()
    };

    let sizes = [6usize, 10, 16, 24, 32];
    let trials = 5;
    let mut total_rescues = 0usize;

    for &n in &sizes {
        for trial in 0..trials {
            let matrix = random_indefinite(n, &mut rng);
            let orig_lower = matrix.data.clone();
            let scale = fro_norm(&orig_lower, n).max(1.0);

            let ff_ref = factor_frontal(&matrix, n, false, &ref_params)
                .unwrap_or_else(|e| panic!("ref n={} trial={}: factor failed: {}", n, trial, e));
            let ff_rook = factor_frontal(&matrix, n, false, &rescue_params)
                .unwrap_or_else(|e| panic!("rook n={} trial={}: factor failed: {}", n, trial, e));

            assert_eq!(
                ff_ref.nelim, n,
                "ref n={} trial={}: expected full elimination",
                n, trial
            );
            assert_eq!(
                ff_rook.nelim, n,
                "rook n={} trial={}: expected full elimination (rescue should fire in place of delay)",
                n, trial
            );

            // Sylvester: inertia must match regardless of pivot sequence.
            assert_eq!(
                ff_rook.inertia, ff_ref.inertia,
                "n={} trial={}: inertia mismatch — rook={}, ref={}",
                n, trial, ff_rook.inertia, ff_ref.inertia
            );

            // Rook rescue must not fall through to force-accept on well-
            // formed random matrices; force-accept means rook gave up.
            assert!(
                !ff_rook.needs_refinement,
                "n={} trial={}: rook rescue fell through to force-accept \
                 (needs_refinement = true)",
                n, trial
            );

            // L·D·Lᵀ reconstruction under the heavy-threshold factor.
            // Tolerance scales with matrix norm; 1e-10 × ||A||_F is well
            // inside double-precision BK accuracy for n ≤ 32.
            let err = reconstruct_residual(&ff_rook, &orig_lower);
            assert!(
                err <= 1e-10 * scale,
                "n={} trial={}: reconstruction residual {:.3e} exceeds 1e-10 * ||A||_F = {:.3e}",
                n,
                trial,
                err,
                1e-10 * scale
            );

            total_rescues += ff_rook.n_rook_rescues;
        }
    }

    // Cross-panel exercise gate: if rook never fires across the whole
    // sweep we have not tested the feature. Heavy threshold on random
    // indefinite matrices should trigger many rescues; assert at least
    // one to keep the test meaningful if the matrix distribution
    // changes.
    assert!(
        total_rescues > 0,
        "rook rescue never fired across {} trials × {} sizes — test is not \
         exercising the rescue path",
        trials,
        sizes.len()
    );
}
