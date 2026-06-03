//! Lever-B prototype for the dense-kernel research note
//! (`dev/research/dense-kernel-vesuvio-tail.md` §4 step 3).
//!
//! Question: on a 1024×1024 dense lower-triangular trailing block,
//! is a deferred panel-of-`bs` rank-`bs` triangular update (BLAS-3
//! shape) ≥ 3× faster than `bs` sequential rank-1 updates (current
//! BLAS-2 shape)?
//!
//! Decision rule: prototype wins ≥ 3× → commit to the full 2.4.1b
//! plan; otherwise drop lever B and pursue lever A only (which the
//! step-1 microbench data already shows is exhausted on aarch64,
//! so a "no" answer here narrows the next-session menu sharply).
//!
//! Notes:
//! - This is a *throwaway* prototype: no pivot rejection support,
//!   no 2×2 panel boundary, no `may_delay` interaction, no inertia
//!   tracking. It only measures the kernel-throughput question.
//! - Both implementations write to the same buffer layout
//!   (column-major, lower triangle stored). The reference rank-1
//!   path uses the same `axpy_minus_unroll4_nofma` kernel as
//!   `do_1x1_update` so the comparison is apples-to-apples.
//! - The "panel" path is the simplest possible BLAS-3 reformulation:
//!   a hand-written triple loop with the inner length matching the
//!   panel width `bs`. Faer's register-tiled `Ukr<MR, NR, T>` would
//!   beat this; we want to know whether even the naive deferred
//!   form wins, before committing to the tiled effort.
//!
//! Usage: `cargo run --release --bin blas3_prototype`.
//!
//! Self-contained — no test corpus needed.

use feral::dense::schur_kernel::axpy_minus_unroll4_nofma;
use std::time::Instant;

const N: usize = 1024;
const BS: usize = 64;

/// Fill an N×N column-major lower-triangular block with deterministic
/// nonzero values. Upper triangle is left as garbage; both kernels
/// only touch i ≥ j.
fn make_lower(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            // Some nonzero, well-conditioned-ish, deterministic.
            a[j * n + i] = 1.0 + 0.1 * ((i + 7 * j) as f64).sin();
        }
        // Bump the diagonal so 1.0/d is well-defined.
        a[j * n + j] += n as f64;
    }
    a
}

/// L (panel of `bs` columns starting at column `0` in `a`, length-N
/// columns) — same scratch layout as the panel path uses for
/// "L-side" of the deferred update.
fn make_panel_l(rng_seed: u64, n: usize, bs: usize) -> Vec<f64> {
    let mut x = rng_seed;
    let mut l = vec![0.0; n * bs];
    for k in 0..bs {
        for i in 0..n {
            // xorshift-ish
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let v = (x as i64 as f64) / (i64::MAX as f64);
            l[k * n + i] = 0.01 * v;
        }
    }
    l
}

/// Reference path: apply `bs` sequential rank-1 updates to the
/// trailing block `a[bs..n, bs..n]`. Mirrors `do_1x1_update`'s inner
/// loop structure (NEON unroll4_nofma kernel).
fn rank1_cascade(a: &mut [f64], n: usize, l: &[f64], bs: usize) {
    // Pretend each panel column k has already been "scaled by D"
    // (we just use l[k*n..k*n+n] directly as the L column).
    for k in 0..bs {
        for j in (k + 1)..n {
            let l_jk = l[k * n + j];
            let alpha = l_jk; // already scaled
            let (before, rest) = a.split_at_mut(j * n);
            let _ = before; // unused; we read from `l`, not `a`, in the prototype
            let src = &l[k * n + j..k * n + n];
            let dst = &mut rest[j..n];
            axpy_minus_unroll4_nofma(dst, src, alpha);
        }
    }
}

/// Deferred path: one rank-`bs` triangular update applied at the end.
/// Computes `a[i, j] -= sum_{k=0..bs} L[i, k] * L[j, k]` for j ≥ bs,
/// i ≥ j. This is the simplest BLAS-3 reformulation — no register
/// tiling, no SIMD beyond what rustc autovectorizes.
fn rank_bs_update(a: &mut [f64], n: usize, l: &[f64], bs: usize) {
    for j in bs..n {
        for i in j..n {
            let mut acc = 0.0;
            for k in 0..bs {
                acc += l[k * n + i] * l[k * n + j];
            }
            a[j * n + i] -= acc;
        }
    }
}

/// Deferred path with a tiny manual register tile: process `i` in
/// chunks of 4 with 4 independent accumulators, dot-product over `k`
/// in the inner loop. Closer to what a real lever-B implementation
/// would do; still no SIMD intrinsics, just ILP.
fn rank_bs_update_tiled(a: &mut [f64], n: usize, l: &[f64], bs: usize) {
    for j in bs..n {
        let l_j = l; // borrow whole panel; column k starts at k*n
        let mut i = j;
        while i + 4 <= n {
            let mut acc0 = 0.0;
            let mut acc1 = 0.0;
            let mut acc2 = 0.0;
            let mut acc3 = 0.0;
            for k in 0..bs {
                let l_jk = l_j[k * n + j];
                acc0 += l_j[k * n + i] * l_jk;
                acc1 += l_j[k * n + i + 1] * l_jk;
                acc2 += l_j[k * n + i + 2] * l_jk;
                acc3 += l_j[k * n + i + 3] * l_jk;
            }
            a[j * n + i] -= acc0;
            a[j * n + i + 1] -= acc1;
            a[j * n + i + 2] -= acc2;
            a[j * n + i + 3] -= acc3;
            i += 4;
        }
        // Tail.
        while i < n {
            let mut acc = 0.0;
            for k in 0..bs {
                acc += l_j[k * n + i] * l_j[k * n + j];
            }
            a[j * n + i] -= acc;
            i += 1;
        }
    }
}

fn checksum(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |s, x| s + x * x)
}

fn time_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    let elapsed_us = t.elapsed().as_micros() as f64;
    elapsed_us / 1000.0 / reps as f64
}

fn main() {
    println!("Lever-B prototype: 1024×1024 trailing block, panel bs=64");
    println!("------------------------------------------------------------");

    let n = N;
    let bs = BS;
    let l = make_panel_l(0x9E37_79B9_7F4A_7C15, n, bs);

    // Warm + sanity check: both paths produce identical results.
    {
        let mut a1 = make_lower(n);
        let mut a2 = make_lower(n);
        let mut a3 = make_lower(n);
        rank1_cascade(&mut a1, n, &l, bs);
        rank_bs_update(&mut a2, n, &l, bs);
        rank_bs_update_tiled(&mut a3, n, &l, bs);

        // Compare a1 vs a2 vs a3 on the lower triangle of the
        // trailing block only.
        let mut max_diff_12 = 0.0_f64;
        let mut max_diff_13 = 0.0_f64;
        for j in bs..n {
            for i in j..n {
                max_diff_12 = max_diff_12.max((a1[j * n + i] - a2[j * n + i]).abs());
                max_diff_13 = max_diff_13.max((a1[j * n + i] - a3[j * n + i]).abs());
            }
        }
        println!(
            "sanity: |rank1 − rank_bs|_max = {:.3e}  |rank1 − tiled|_max = {:.3e}",
            max_diff_12, max_diff_13,
        );
        println!("        checksum = {:.6e}", checksum(&a1));
        println!();
    }

    // Time each path. Use the same fresh `a` for each rep so cache
    // state is comparable.
    let reps = 10;

    let ms_rank1 = time_ms(reps, || {
        let mut a = make_lower(n);
        rank1_cascade(&mut a, n, &l, bs);
    });
    let ms_rank_bs = time_ms(reps, || {
        let mut a = make_lower(n);
        rank_bs_update(&mut a, n, &l, bs);
    });
    let ms_tiled = time_ms(reps, || {
        let mut a = make_lower(n);
        rank_bs_update_tiled(&mut a, n, &l, bs);
    });
    let ms_setup = time_ms(reps, || {
        let _ = make_lower(n);
    });

    // Subtract the make_lower setup cost so we measure only the kernel.
    let pure_rank1 = ms_rank1 - ms_setup;
    let pure_rank_bs = ms_rank_bs - ms_setup;
    let pure_tiled = ms_tiled - ms_setup;

    // FLOP estimates.
    // rank1_cascade: bs panels × Σ_{j=k+1}^{n-1} (n - j) FMAs
    //              = bs × (n-1) × n / 2 / bs  (approx)  ≈ n^2 × bs / 2 FMAs
    // rank_bs_update: Σ_{j=bs}^{n-1} (n - j) × bs FMAs
    //              = bs × (n - bs) × (n - bs + 1) / 2  FMAs
    // Both ≈ n^2 × bs / 2 (within 5% for n=1024, bs=64). 1 FMA = 2 FLOPs.
    let flops = (n as f64) * (n as f64) * (bs as f64) / 2.0 * 2.0;
    let gflops_rank1 = flops / (pure_rank1 * 1e6);
    let gflops_rank_bs = flops / (pure_rank_bs * 1e6);
    let gflops_tiled = flops / (pure_tiled * 1e6);

    println!(
        "rank-1 cascade ({} updates):     {:>7.3} ms  ({:.1} GFLOP/s)",
        bs, pure_rank1, gflops_rank1,
    );
    println!(
        "rank-{} update (naive triple):  {:>7.3} ms  ({:.1} GFLOP/s)",
        bs, pure_rank_bs, gflops_rank_bs,
    );
    println!(
        "rank-{} update (4-tile ILP):    {:>7.3} ms  ({:.1} GFLOP/s)",
        bs, pure_tiled, gflops_tiled,
    );
    println!();
    println!(
        "speedup naive   / rank-1: {:.2}×",
        pure_rank1 / pure_rank_bs.max(1e-9),
    );
    println!(
        "speedup tiled   / rank-1: {:.2}×",
        pure_rank1 / pure_tiled.max(1e-9),
    );
    println!();
    println!("Decision rule: tiled ≥ 3× → commit to 2.4.1b plan.");
}
