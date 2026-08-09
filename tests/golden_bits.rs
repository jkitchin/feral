//! Golden bit-pattern test for the dense blocked LDLᵀ (session
//! 2026-08-09, Stage 1 of the kernel-SIMD pass).
//!
//! The byte-exactness contract for the default (nofma) kernels is
//! *cross-platform*: the per-element fold is `mul → sub` (two roundings)
//! at every lane width, so scalar, NEON f64x2, AVX2 f64x4, and AVX-512
//! f64x8 all produce identical bits. The in-repo parity suites pin the
//! kernels against each other *on the machine running the tests*; this
//! test pins them against **hardcoded** u64 digests, so a divergence
//! between architectures fails loudly the first time the suite runs on
//! the other machine (e.g. the aarch64 M-series corpus host).
//!
//! The FMA (opt-in) path is also covered: `f64::mul_add`, libm `fma`,
//! x86 `vfmadd`, and NEON `vfmaq_f64` are all exactly-rounded fused
//! multiply-adds, so its digests are cross-platform too — just different
//! from the nofma digests (one rounding vs two).
//!
//! If a digest here ever changes, that is a change to factorization
//! results, not a refactor — treat it as a correctness event.

use feral::dense::factor::factor_frontal_blocked;
use feral::{BunchKaufmanParams, SymmetricMatrix};

/// SplitMix64, the same generator the dense-front benches use.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Indefinite front with a sign-alternating dominant diagonal (mostly
/// 1×1 pivots) — the `bench_dense_front` fixture shape.
fn build_indefinite_front(n: usize, seed: u64) -> SymmetricMatrix {
    let mut rng = Rng(seed);
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            data[j * n + i] = if i == j {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sign * (n as f64) + rng.next_f64()
            } else {
                0.30 * rng.next_f64()
            };
        }
    }
    SymmetricMatrix { n, data }
}

/// A front with a *weak* diagonal so Bunch-Kaufman takes 2×2 pivots and
/// the mixed 1×1/2×2 packed streams are exercised.
fn build_2x2_heavy_front(n: usize, seed: u64) -> SymmetricMatrix {
    let mut rng = Rng(seed);
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            data[j * n + i] = if i == j {
                0.01 * rng.next_f64()
            } else {
                rng.next_f64()
            };
        }
    }
    SymmetricMatrix { n, data }
}

/// FNV-1a over the bit patterns of a f64 slice.
fn digest(vals: &[f64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in vals {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

fn factor_digest(matrix: &SymmetricMatrix, fma: bool) -> (u64, u64, u64, (usize, usize, usize)) {
    let params = BunchKaufmanParams {
        fma,
        ..BunchKaufmanParams::default()
    };
    let ff = factor_frontal_blocked(matrix, matrix.n, false, &params).expect("factor failed");
    let inertia = (ff.inertia.positive, ff.inertia.negative, ff.inertia.zero);
    (
        digest(&ff.l),
        digest(&ff.d_diag),
        digest(&ff.d_subdiag),
        inertia,
    )
}

#[test]
fn golden_bits_nofma_indefinite_256() {
    let m = build_indefinite_front(256, 0x1234_5678_9ABC_DEF0);
    let (l, d, sd, inertia) = factor_digest(&m, false);
    assert_eq!(inertia, (128, 128, 0), "inertia drifted");
    assert_eq!(
        (l, d, sd),
        (
            GOLDEN_NOFMA_IND[0],
            GOLDEN_NOFMA_IND[1],
            GOLDEN_NOFMA_IND[2]
        ),
        "nofma indefinite-256 factor bits drifted from the golden digests"
    );
}

#[test]
fn golden_bits_nofma_2x2_heavy_96() {
    let m = build_2x2_heavy_front(96, 0x5EED_CAFE_F00D_D00D);
    let (l, d, sd, _inertia) = factor_digest(&m, false);
    assert_eq!(
        (l, d, sd),
        (
            GOLDEN_NOFMA_2X2[0],
            GOLDEN_NOFMA_2X2[1],
            GOLDEN_NOFMA_2X2[2]
        ),
        "nofma 2x2-heavy-96 factor bits drifted from the golden digests"
    );
}

#[test]
fn golden_bits_fma_indefinite_256() {
    let m = build_indefinite_front(256, 0x1234_5678_9ABC_DEF0);
    let (l, d, sd, inertia) = factor_digest(&m, true);
    assert_eq!(inertia, (128, 128, 0), "inertia drifted (fma)");
    assert_eq!(
        (l, d, sd),
        (GOLDEN_FMA_IND[0], GOLDEN_FMA_IND[1], GOLDEN_FMA_IND[2]),
        "fma indefinite-256 factor bits drifted from the golden digests"
    );
}

// Golden digests, recorded on x86_64 (AVX2 via pulp V3 dispatch) at
// session 2026-08-09-01. The nofma digests must reproduce on every
// platform; the fma digests must reproduce on every platform whose
// fused multiply-add is exactly rounded (all supported ones).
// Cross-checked identical on this host across all three kernel paths:
// default (pulp SIMD), FERAL_PACKED_SIMD=0 (scalar tile loop), and
// FERAL_PACKED_SCHUR=0 (strided per-column kernels).
const GOLDEN_NOFMA_IND: [u64; 3] = [0x7073f0edec336dbe, 0x2f1419d50d39b641, 0x28c31cf8df2ec325];
const GOLDEN_NOFMA_2X2: [u64; 3] = [0x3701e106f647e9f1, 0xb36ca5290912f46b, 0x2f16488fbf9b6fd7];
const GOLDEN_FMA_IND: [u64; 3] = [0x8ac5b55facb8b9dd, 0x2f1419d50d39b641, 0x28c31cf8df2ec325];

/// Prints the digests so they can be hardcoded above. Run with
/// `cargo test --release --test golden_bits -- --ignored --nocapture`.
#[test]
#[ignore]
fn record_golden() {
    let m = build_indefinite_front(256, 0x1234_5678_9ABC_DEF0);
    let (l, d, sd, i) = factor_digest(&m, false);
    println!("GOLDEN_NOFMA_IND: [{l:#018x}, {d:#018x}, {sd:#018x}]  inertia={i:?}");
    let m2 = build_2x2_heavy_front(96, 0x5EED_CAFE_F00D_D00D);
    let (l, d, sd, i) = factor_digest(&m2, false);
    println!("GOLDEN_NOFMA_2X2: [{l:#018x}, {d:#018x}, {sd:#018x}]  inertia={i:?}");
    let (l, d, sd, i) = factor_digest(&m, true);
    println!("GOLDEN_FMA_IND:   [{l:#018x}, {d:#018x}, {sd:#018x}]  inertia={i:?}");
}
