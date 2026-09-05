//! Issue #153 — what is feral's dense frontal-factor throughput, and how
//! far is it from what the machine can do?
//!
//! The #200 measurements say feral runs 1.5-3.2x fewer flops per second
//! than MA57 at every front size, on 0.44-1.20x the work. MA57's speed
//! comes from OpenBLAS: its trailing updates are `dgemm`. feral is pure
//! Rust by constraint (CLAUDE.md), so the honest question for #153 is
//! not "why are we slower than MA57" but "how much of the machine does
//! our kernel get, and how much does a tuned `dgemm` get on the same
//! shapes on the same core?"
//!
//! This measures the first half: `factor_frontal` on a dense, diagonally
//! dominant `n x n` front with every column eliminated, at the front
//! sizes the corpus actually produces. Flops are counted with the same
//! `sum_{k<ncol}(nrow-k)^2` multiply-add model used by
//! `diag_200_work_vs_ma57`, so "MFlop/s" here means millions of
//! multiply-adds per second and is directly comparable to that binary's
//! column. The `dgemm` side is measured separately by
//! `dev/scripts/dgemm_peak.f` and `dev/scripts/dsytrf_peak.f` against the
//! same OpenBLAS MA57 links to; see `dev/scripts/README-blas-reference.md`.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin diag_153_dense_peak
//!     [-- --reps N --sizes 32,64,128,256,512]
use feral::dense::factor::factor_frontal;
use feral::{BunchKaufmanParams, SymmetricMatrix};
use std::time::Instant;

/// Deterministic xorshift so runs are comparable.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // Uniform in [-1, 1).
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// A symmetric front with a strong diagonal, so the factorization takes
/// the ordinary 1x1-pivot path and measures kernel throughput rather
/// than pivot-search pathology.
fn make_front(n: usize, seed: u64) -> SymmetricMatrix {
    let mut rng = Rng(seed);
    let mut m = SymmetricMatrix::zeros(n);
    for j in 0..n {
        for i in j..n {
            let v = if i == j {
                n as f64 + 1.0
            } else {
                rng.next_f64()
            };
            m.set(i, j, v);
        }
    }
    m
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut reps = 0usize;
    let mut sizes: Vec<usize> = vec![16, 32, 48, 64, 96, 128, 192, 256, 384, 512];
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--reps" => {
                if let Some(v) = it.next() {
                    reps = v.parse()?;
                }
            }
            "--sizes" => {
                if let Some(v) = it.next() {
                    sizes = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                }
            }
            _ => {}
        }
    }

    let params = BunchKaufmanParams::default();
    println!(
        "{:>6}{:>14}{:>12}{:>12}{:>12}",
        "n", "macs", "min_us", "MMac/s", "% of n=512"
    );
    let mut peak = 0.0f64;
    let mut rows: Vec<(usize, f64)> = Vec::new();
    for &n in &sizes {
        let front = make_front(n, 0x243F_6A88_85A3_08D3);
        // sum_{k<n} (n-k)^2 multiply-adds.
        let macs: f64 = (0..n).map(|k| ((n - k) * (n - k)) as f64).sum();
        // Scale rep count so every size does comparable total work.
        let r = if reps > 0 {
            reps
        } else {
            (2.0e8 / macs).ceil().clamp(5.0, 20000.0) as usize
        };
        // Warm.
        for _ in 0..3 {
            let _ = factor_frontal(&front, n, false, &params)?;
        }
        let mut best = f64::INFINITY;
        for _ in 0..r {
            let t0 = Instant::now();
            let f = factor_frontal(&front, n, false, &params)?;
            let us = t0.elapsed().as_nanos() as f64 / 1000.0;
            std::hint::black_box(&f);
            best = best.min(us);
        }
        let rate = macs / best; // MMac/s == macs per microsecond
        peak = peak.max(rate);
        rows.push((n, rate));
        println!("{:>6}{:>14.0}{:>12.2}{:>12.0}", n, macs, best, rate);
    }
    println!("\npeak = {peak:.0} MMac/s over the swept sizes");
    if let Some(&(_, top)) = rows.last() {
        println!("largest size = {top:.0} MMac/s");
    }
    Ok(())
}
