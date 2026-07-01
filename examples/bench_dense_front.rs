//! Single-front dense LDLᵀ micro-benchmark.
//!
//! Issue #99 tracks closing the dense-front throughput gap to faer on
//! large conic-KKT roots (the qap15 root is a 2955×2955 indefinite
//! front, 42% of the sequential factor loop). The qap15 fixture and its
//! harnesses referenced by the issue are not present on this branch, so
//! this example provides a *self-contained, reproducible* stand-in: it
//! builds a synthetic indefinite front of a chosen size and factors it
//! through the real blocked-panel path (`factor_frontal_blocked`),
//! exercising exactly the trailing-Schur kernel (W-2 rank-`n_elim`
//! accumulator + intra-front parallelism + optional FMA) that the issue
//! identifies as the bottleneck.
//!
//! It measures the two per-core / parallel levers directly:
//!   * nofma vs FMA kernel throughput (issue Lever 3)
//!   * serial vs intra-front-parallel trailing update (issue Lever 1)
//!
//! and asserts the inertia is identical across all four variants — the
//! correctness gate that any throughput lever must preserve.
//!
//! Usage:
//!   cargo run --release --example bench_dense_front [-- N REPS]
//! Defaults: N = 2955 (the qap15 root size), REPS = 5.

use feral::dense::factor::factor_frontal_blocked;
use feral::{BunchKaufmanParams, Inertia, SymmetricMatrix};
use std::time::Instant;

/// Deterministic SplitMix64-style PRNG so the fixture is reproducible
/// without a `rand` dependency or the forbidden `Math.random`/`Date`.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Map to [-1, 1).
        (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Build a synthetic symmetric *indefinite* front that factors mostly
/// through 1×1 pivots (so it hits the W-2 all-1×1 fast path the issue
/// profiles). Diagonal carries alternating-sign dominant entries;
/// off-diagonals are small deterministic noise. Column-major lower
/// triangle, matching `SymmetricMatrix`'s storage.
fn build_front(n: usize) -> SymmetricMatrix {
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);
    let mut data = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            let v = if i == j {
                // Strong, sign-alternating diagonal → indefinite,
                // 1×1-pivot dominated, well away from singular.
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sign * (n as f64) + rng.next_f64()
            } else {
                0.30 * rng.next_f64()
            };
            data[j * n + i] = v;
        }
    }
    SymmetricMatrix { n, data }
}

fn time_variant(
    matrix: &SymmetricMatrix,
    fma: bool,
    intrafront: bool,
    reps: usize,
) -> (f64, Inertia) {
    let params = BunchKaufmanParams {
        fma,
        intrafront_parallel: intrafront,
        ..BunchKaufmanParams::default()
    };
    let n = matrix.n;
    let mut best = f64::INFINITY;
    let mut inertia = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    for _ in 0..reps {
        let t = Instant::now();
        // Root front: every column is fully summed (ncol == nrow),
        // no delayed pivots.
        let ff = factor_frontal_blocked(matrix, n, false, &params).expect("factor failed");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        inertia = ff.inertia;
        best = best.min(ms);
    }
    (best, inertia)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2955);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    let threads = rayon::current_num_threads();
    println!("bench_dense_front: n={n} reps={reps} rayon_threads={threads}");
    let gflop = (n as f64).powi(3) / 3.0 / 1e9; // ≈ LDLᵀ flop count

    let matrix = build_front(n);

    let (t_nofma_ser, i0) = time_variant(&matrix, false, false, reps);
    let (t_nofma_par, i1) = time_variant(&matrix, false, true, reps);
    let (t_fma_ser, i2) = time_variant(&matrix, true, false, reps);
    let (t_fma_par, i3) = time_variant(&matrix, true, true, reps);

    let report = |label: &str, ms: f64| {
        println!(
            "  {label:<22} {ms:8.2} ms   {:6.2} GFLOP/s   {:5.2}× vs nofma-serial",
            gflop / (ms / 1e3),
            t_nofma_ser / ms
        );
    };
    println!("inertia (must match): {i0:?}");
    report("nofma  serial", t_nofma_ser);
    report("nofma  intrafront", t_nofma_par);
    report("fma    serial", t_fma_ser);
    report("fma    intrafront", t_fma_par);

    // Correctness gate: every throughput lever must preserve inertia.
    assert_eq!(i0, i1, "intrafront changed inertia (nofma)");
    assert_eq!(i0, i2, "fma changed inertia");
    assert_eq!(i0, i3, "fma+intrafront changed inertia");
    println!("inertia identical across all four variants ✓");
}
