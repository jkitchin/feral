//! Issue #10/#33 follow-up: measure the per-call cost of
//! `axpy_minus_unroll4_nofma` at the small lengths actually hit by
//! 1D-banded Mittelmann supernodes (3..32). The hypothesis from
//! `dev/research/issue-10-maxfromm-phase2-corpus.md` is that on narrow
//! supernodes (avg ncol ≈ 6), the pulp dispatch overhead dominates the
//! arithmetic, and a plain scalar loop would be faster.
//!
//! Compares three implementations at length sweep [3, 4, 5, 6, 8, 10, 16, 32, 64, 128]:
//!   1. `pulp`     — current `axpy_minus_unroll4_nofma` (SIMD + dispatch)
//!   2. `scalar`   — `for (d, s) in dst.iter_mut().zip(src) { *d -= alpha * *s; }`
//!   3. `unroll4`  — manual 4-way unroll without pulp dispatch
//!
//! Reports ns/call (min-of-N) so the cache-warm steady-state cost is
//! visible. Each measurement runs the kernel ~10M times across a
//! ping-pong of two buffers to keep them L1-resident.
//!
//! Usage: `cargo run --release --bin bench_axpy_small`

use feral::dense::schur_kernel::axpy_minus_unroll4_nofma;
use std::hint::black_box;
use std::time::Instant;

const ITERS_PER_LEN: usize = 50_000_000;
const REPEAT: usize = 5;

#[inline(never)]
fn run_pulp(dst: &mut [f64], src: &[f64], alpha: f64) {
    axpy_minus_unroll4_nofma(dst, src, alpha);
}

#[inline(never)]
fn run_scalar(dst: &mut [f64], src: &[f64], alpha: f64) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d -= alpha * *s;
    }
}

#[inline(never)]
fn run_unroll4(dst: &mut [f64], src: &[f64], alpha: f64) {
    let n = dst.len();
    let chunks = n / 4;
    let rem_start = chunks * 4;
    for c in 0..chunks {
        let b = c * 4;
        let d0 = dst[b] - alpha * src[b];
        let d1 = dst[b + 1] - alpha * src[b + 1];
        let d2 = dst[b + 2] - alpha * src[b + 2];
        let d3 = dst[b + 3] - alpha * src[b + 3];
        dst[b] = d0;
        dst[b + 1] = d1;
        dst[b + 2] = d2;
        dst[b + 3] = d3;
    }
    for i in rem_start..n {
        dst[i] -= alpha * src[i];
    }
}

fn bench<F: FnMut(&mut [f64], &[f64], f64)>(name: &str, len: usize, mut f: F) -> u128 {
    let mut a = vec![1.0f64; len];
    let src = vec![0.5f64; len];
    let alpha = 1e-10;
    f(&mut a, &src, alpha);
    let mut best = u128::MAX;
    for _ in 0..REPEAT {
        let t = Instant::now();
        for _ in 0..ITERS_PER_LEN {
            f(black_box(&mut a), black_box(&src), black_box(alpha));
        }
        let ns = t.elapsed().as_nanos();
        if ns < best {
            best = ns;
        }
    }
    let _ = name;
    best / (ITERS_PER_LEN as u128)
}

fn main() {
    let lens = [3usize, 4, 5, 6, 8, 10, 16, 32, 64, 128];
    println!("=== axpy_minus small-length microbench (min-of-{REPEAT}, {ITERS_PER_LEN} iters/measure) ===");
    println!(
        "{:>4}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "len", "pulp_ns", "scalar_ns", "unroll4_ns", "pulp/scl", "pulp/u4"
    );
    for &len in &lens {
        let p = bench("pulp", len, run_pulp) as f64;
        let s = bench("scalar", len, run_scalar) as f64;
        let u = bench("unroll4", len, run_unroll4) as f64;
        let p_s = if s > 0.0 { p / s } else { f64::NAN };
        let p_u = if u > 0.0 { p / u } else { f64::NAN };
        println!(
            "{:>4}  {:>10.2}  {:>10.2}  {:>10.2}  {:>9.2}x  {:>9.2}x",
            len, p, s, u, p_s, p_u
        );
    }
}
