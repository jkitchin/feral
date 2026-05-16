//! Issue #35 probe — direct A/B of FMA vs no-FMA Schur-panel kernels.
//!
//! The issue #14 wide-supernode probe surfaced a regression: setting
//! `BunchKaufmanParams::fma = true` made `factor_frontal_blocked`
//! 0.65–0.93x as fast as the default `fma = false` on M-series
//! (aarch64). That probe times the full factorisation, so kernel time
//! is mixed with dispatch overhead, panel admin, and pivot selection.
//!
//! This probe drops a layer: it calls
//! `schur_panel_minus_{fma,nofma}_strided_quad` directly on identical
//! inputs and times only the kernel body. It is the smallest binary
//! that can answer the issue #35 decision tree:
//!
//!   - aarch64-only regression  → gate `fma=true` behind `cfg(x86_64)`
//!   - regression on both       → remove the FMA path
//!   - x86 wins, aarch64 loses  → document per-arch asymmetry
//!
//! Methodology:
//!   - Synthesise a `src_block` of shape (n_elim × col_stride) and a
//!     destination quad-column slice (lengths nrow, nrow-1, nrow-2,
//!     nrow-3) per the kernel's split contract (matches the call site
//!     in `apply_blocked_schur_panel`, factor.rs:2445-2462).
//!   - For each rep: reset the four dst columns to a deterministic
//!     state, then time one kernel call.
//!   - Report median / min ns and effective GFLOPS over the FMA path
//!     and the nofma path, plus their speedup ratio.
//!   - Three shape buckets exercise the regimes the issue #14 probe
//!     used (wide trailing rows, square-ish, narrow panel).
//!
//! Run:
//!     cargo run --release --bin probe_fma_kernel
//!
//! Optional env knobs:
//!     PROBE_REPS=N       (default 21; odd → exact median)
//!     PROBE_WARMUP=N     (default 5)

use feral::dense::schur_kernel;
use std::time::Instant;

const DEFAULT_REPS: usize = 21;
const DEFAULT_WARMUP: usize = 5;

/// Workload shape for one probe row.
#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    nrow: usize,
    n_elim: usize,
}

const SHAPES: &[Shape] = &[
    // Wide trailing rows over a 433-pivot panel — snode 3593 from
    // MBndryCntrl_3D_27 (issue #14, the original regression locus).
    Shape {
        name: "wide_2829x433",
        nrow: 2829,
        n_elim: 64,
    },
    // Square-ish root-supernode regime — snode 3607.
    Shape {
        name: "square_1928",
        nrow: 1928,
        n_elim: 64,
    },
    // Narrow panel over moderate trailing rows.
    Shape {
        name: "narrow_512x32",
        nrow: 512,
        n_elim: 32,
    },
];

/// Deterministic f64 in (-0.5, 0.5).
fn det_f64(seed: u64) -> f64 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    x ^= x >> 33;
    let u = (x >> 32) as u32 as f64;
    (u / (u32::MAX as f64)) - 0.5
}

/// Build the inputs `apply_blocked_schur_panel` would feed to the
/// strided_quad kernel for `j = n_elim` (i.e. the panel-update site
/// immediately after the n_elim columns are factored).
///
/// `src_block` layout matches the dense column-major frontal carved
/// out by `a.split_at_mut(j * nrow)` in factor.rs:2445 — so it has
/// `j * nrow` elements, with the n_elim pivot columns occupying
/// columns [src_first_col, src_first_col + n_elim).
struct Inputs {
    src_block: Vec<f64>,
    src_first_col: usize,
    col_stride: usize,
    src_row_offset: usize,
    dst0_template: Vec<f64>,
    dst1_template: Vec<f64>,
    dst2_template: Vec<f64>,
    dst3_template: Vec<f64>,
    alphas0: Vec<f64>,
    alphas1: Vec<f64>,
    alphas2: Vec<f64>,
    alphas3: Vec<f64>,
}

fn build_inputs(shape: Shape) -> Inputs {
    let nrow = shape.nrow;
    let n_elim = shape.n_elim;
    let col_stride = nrow;
    // j = n_elim (first trailing column after the pivot panel).
    let j = n_elim;
    let src_first_col = 0;
    let src_row_offset = j;

    let src_block = (0..(j * col_stride))
        .map(|i| det_f64(i as u64 ^ 0xA11A))
        .collect();

    let dst0_template: Vec<f64> = (0..(nrow - j))
        .map(|i| det_f64(i as u64 ^ 0xB22B))
        .collect();
    let dst1_template: Vec<f64> = (0..(nrow - j - 1))
        .map(|i| det_f64(i as u64 ^ 0xC33C))
        .collect();
    let dst2_template: Vec<f64> = (0..(nrow - j - 2))
        .map(|i| det_f64(i as u64 ^ 0xD44D))
        .collect();
    let dst3_template: Vec<f64> = (0..(nrow - j - 3))
        .map(|i| det_f64(i as u64 ^ 0xE55E))
        .collect();

    let alphas0: Vec<f64> = (0..n_elim).map(|i| det_f64(i as u64 ^ 0xF11F)).collect();
    let alphas1: Vec<f64> = (0..n_elim).map(|i| det_f64(i as u64 ^ 0xF22F)).collect();
    let alphas2: Vec<f64> = (0..n_elim).map(|i| det_f64(i as u64 ^ 0xF33F)).collect();
    let alphas3: Vec<f64> = (0..n_elim).map(|i| det_f64(i as u64 ^ 0xF44F)).collect();

    Inputs {
        src_block,
        src_first_col,
        col_stride,
        src_row_offset,
        dst0_template,
        dst1_template,
        dst2_template,
        dst3_template,
        alphas0,
        alphas1,
        alphas2,
        alphas3,
    }
}

/// FLOPs per call: each of the four destination columns receives
/// `len_dst × n_elim` multiply-adds (2 FLOPs each).
fn flops_per_call(shape: Shape) -> f64 {
    let j = shape.n_elim;
    let nrow = shape.nrow;
    let len0 = nrow - j;
    let len1 = nrow - j - 1;
    let len2 = nrow - j - 2;
    let len3 = nrow - j - 3;
    let elements = (len0 + len1 + len2 + len3) * shape.n_elim;
    2.0 * elements as f64
}

fn time_one_call<F: FnMut()>(mut f: F) -> u128 {
    let t = Instant::now();
    f();
    t.elapsed().as_nanos()
}

fn median(mut xs: Vec<u128>) -> u128 {
    xs.sort_unstable();
    xs[xs.len() / 2]
}

fn min(xs: &[u128]) -> u128 {
    *xs.iter().min().unwrap_or(&0)
}

fn bench_one(shape: Shape, reps: usize, warmup: usize) {
    let inp = build_inputs(shape);
    let flops = flops_per_call(shape);

    // Mutable destination buffers reset per rep from templates so each
    // call sees identical input. Allocated once outside the timer.
    let mut dst0 = inp.dst0_template.clone();
    let mut dst1 = inp.dst1_template.clone();
    let mut dst2 = inp.dst2_template.clone();
    let mut dst3 = inp.dst3_template.clone();

    // --- FMA path ---
    for _ in 0..warmup {
        dst0.copy_from_slice(&inp.dst0_template);
        dst1.copy_from_slice(&inp.dst1_template);
        dst2.copy_from_slice(&inp.dst2_template);
        dst3.copy_from_slice(&inp.dst3_template);
        schur_kernel::schur_panel_minus_fma_strided_quad(
            &mut dst0,
            &mut dst1,
            &mut dst2,
            &mut dst3,
            &inp.src_block,
            inp.src_first_col,
            shape.n_elim,
            inp.col_stride,
            inp.src_row_offset,
            &inp.alphas0,
            &inp.alphas1,
            &inp.alphas2,
            &inp.alphas3,
        );
    }
    let mut fma_ns: Vec<u128> = Vec::with_capacity(reps);
    for _ in 0..reps {
        dst0.copy_from_slice(&inp.dst0_template);
        dst1.copy_from_slice(&inp.dst1_template);
        dst2.copy_from_slice(&inp.dst2_template);
        dst3.copy_from_slice(&inp.dst3_template);
        let ns = time_one_call(|| {
            schur_kernel::schur_panel_minus_fma_strided_quad(
                &mut dst0,
                &mut dst1,
                &mut dst2,
                &mut dst3,
                &inp.src_block,
                inp.src_first_col,
                shape.n_elim,
                inp.col_stride,
                inp.src_row_offset,
                &inp.alphas0,
                &inp.alphas1,
                &inp.alphas2,
                &inp.alphas3,
            );
        });
        fma_ns.push(ns);
    }

    // --- nofma path ---
    for _ in 0..warmup {
        dst0.copy_from_slice(&inp.dst0_template);
        dst1.copy_from_slice(&inp.dst1_template);
        dst2.copy_from_slice(&inp.dst2_template);
        dst3.copy_from_slice(&inp.dst3_template);
        schur_kernel::schur_panel_minus_nofma_strided_quad(
            &mut dst0,
            &mut dst1,
            &mut dst2,
            &mut dst3,
            &inp.src_block,
            inp.src_first_col,
            shape.n_elim,
            inp.col_stride,
            inp.src_row_offset,
            &inp.alphas0,
            &inp.alphas1,
            &inp.alphas2,
            &inp.alphas3,
        );
    }
    let mut nofma_ns: Vec<u128> = Vec::with_capacity(reps);
    for _ in 0..reps {
        dst0.copy_from_slice(&inp.dst0_template);
        dst1.copy_from_slice(&inp.dst1_template);
        dst2.copy_from_slice(&inp.dst2_template);
        dst3.copy_from_slice(&inp.dst3_template);
        let ns = time_one_call(|| {
            schur_kernel::schur_panel_minus_nofma_strided_quad(
                &mut dst0,
                &mut dst1,
                &mut dst2,
                &mut dst3,
                &inp.src_block,
                inp.src_first_col,
                shape.n_elim,
                inp.col_stride,
                inp.src_row_offset,
                &inp.alphas0,
                &inp.alphas1,
                &inp.alphas2,
                &inp.alphas3,
            );
        });
        nofma_ns.push(ns);
    }

    let fma_med = median(fma_ns.clone());
    let fma_min = min(&fma_ns);
    let nofma_med = median(nofma_ns.clone());
    let nofma_min = min(&nofma_ns);

    let fma_gflops_med = flops / (fma_med as f64);
    let nofma_gflops_med = flops / (nofma_med as f64);

    let speedup_med = nofma_med as f64 / fma_med as f64;
    let speedup_min = nofma_min as f64 / fma_min as f64;

    println!(
        "{:<18}  nrow={:<5}  n_elim={:<3}  fma med/min ={:>9.1}/{:>9.1} us  nofma med/min ={:>9.1}/{:>9.1} us  \
         fma GF={:>5.2}  nofma GF={:>5.2}  fma/nofma speedup med={:.3} min={:.3}",
        shape.name,
        shape.nrow,
        shape.n_elim,
        fma_med as f64 / 1000.0,
        fma_min as f64 / 1000.0,
        nofma_med as f64 / 1000.0,
        nofma_min as f64 / 1000.0,
        fma_gflops_med,
        nofma_gflops_med,
        speedup_med,
        speedup_min,
    );
}

fn main() {
    let reps: usize = std::env::var("PROBE_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REPS);
    let warmup: usize = std::env::var("PROBE_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_WARMUP);

    println!(
        "probe_fma_kernel: reps={}, warmup={}, target_arch={}",
        reps,
        warmup,
        std::env::consts::ARCH
    );
    println!("speedup = nofma_ns / fma_ns; >1 means FMA wins, <1 means FMA is a regression");
    println!();

    for &shape in SHAPES {
        bench_one(shape, reps, warmup);
    }
}
