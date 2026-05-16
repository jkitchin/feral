//! Issue #14 probe — wide / near-square supernode kernel throughput.
//!
//! Measures the dense BK panel kernel on two representative shapes from
//! the `MBndryCntrl_3D_27` n=31104 NLP profile:
//!
//! - snode 3607 (1928 × 1928) — root supernode, full-square.
//! - snode 3593 (2829 × 433) — wide trailing rows over a 433-pivot panel.
//!
//! For each shape we synthesise an SPD-ish frontal (deterministic) and a
//! KKT-style indefinite frontal (forces 2×2 pivots / panel fallbacks),
//! then time `factor_frontal_blocked_in_place` with `panel_diag` enabled.
//! Reports:
//!
//! - Median / min wall time over `N_REPS` reps.
//! - Effective GFLOPS using the textbook LDLᵀ work model
//!   `work = nelim·ncol·(nrow − nelim/2) − nelim²·ncol/2`
//!   (panel factorisation + Schur update against trailing rows).
//! - Panel-dispatch attribution from `panel_diag` (PIVOTS_INLINE vs
//!   PIVOTS_SCALAR, panel_full / panel_partial / panel_delayed).
//! - SIMD-body vs scalar-tail element counts, derived analytically from
//!   the trailing-block geometry the kernel walks and the platform
//!   F64_LANES constant (NEON = 2, AVX2 = 4, AVX-512 = 8).
//!
//! This is a measurement-only binary in support of issue #14's
//! "concrete next-step probe" — it does NOT modify the kernel.
//!
//! Run:
//!     cargo run --release --bin probe_wide_supernode
//!
//! Optional env knobs (mostly for ad-hoc sweeps):
//!     PROBE_REPS=N         (default 5)
//!     PROBE_BLOCK_SIZE=N   (default 64; matches BunchKaufmanParams default)
//!     PROBE_FMA=1          (default 0; toggle FMA path)
//!     PROBE_SHAPES=full    (default "full,wide"; subset to time)
//!
//! References:
//! - Issue #14 problem statement.
//! - `dev/research/feral-kernel-profile-chainwoo.md` — methodology
//!   precedent (synthetic frontal at observed shape, micro-timed under
//!   `factor_frontal_with_profile`).

use feral::dense::factor::{
    factor_frontal, factor_frontal_blocked_in_place, panel_diag, FactorScratch, PANEL_DIAG_ENABLED,
};
use feral::dense::matrix::SymmetricMatrix;
use feral::BunchKaufmanParams;
use std::time::Instant;

/// Best-effort guess at the SIMD f64 lane count on the running target.
/// Used purely for reporting / FLOP attribution; the kernel itself
/// dispatches through `pulp::Arch::new()` which picks the right lane
/// count regardless.
fn detected_f64_lanes() -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON ASIMD has 128-bit vector regs = 2 × f64.
        return 2;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return 8;
        }
        if std::is_x86_feature_detected!("avx2") {
            return 4;
        }
        if std::is_x86_feature_detected!("sse2") {
            return 2;
        }
        return 1;
    }
    #[allow(unreachable_code)]
    {
        1
    }
}

/// Deterministic SPD-ish symmetric frontal. Diagonally dominant so
/// every pivot accepts as a 1×1 and the panel path runs end-to-end
/// without delaying.
fn make_spd_frontal(n: usize) -> SymmetricMatrix {
    let mut data = vec![0.0; n * n];
    for j in 0..n {
        data[j * n + j] = 4.0 + 0.13 * j as f64;
        for i in (j + 1)..n {
            let x = ((i.wrapping_mul(31) ^ j.wrapping_mul(7)) % 17) as f64 / 17.0;
            data[j * n + i] = 0.05 * (x - 0.5);
        }
    }
    SymmetricMatrix { n, data }
}

/// KKT-style saddle-point frontal: top half PD with mild coupling, bottom
/// half near-zero diagonal with strong cross-block coupling. Exercises
/// the BK 2×2 pivot path that fires on real `MBndryCntrl_3D_27` fronts.
fn make_kkt_frontal(n: usize) -> SymmetricMatrix {
    let mut data = vec![0.0; n * n];
    let n1 = n / 2;
    for j in 0..n1 {
        data[j * n + j] = 1.0 + 0.05 * j as f64;
        for i in (j + 1)..n1 {
            let x = ((i.wrapping_mul(31) ^ j.wrapping_mul(7)) % 13) as f64 / 13.0;
            data[j * n + i] = 0.02 * (x - 0.5);
        }
        for i in n1..n {
            data[j * n + i] = if (i + j) % 3 == 0 { 1.0 } else { 0.0 };
        }
    }
    for j in n1..n {
        data[j * n + j] = -1e-8;
    }
    SymmetricMatrix { n, data }
}

/// LDLᵀ work model (FLOPs) for a frontal of `nrow` rows, `ncol`
/// fully-summed columns, all accepted as 1×1 pivots.
///
/// - Pivot scaling: `(nrow - k - 1)` divides per pivot k.
/// - Schur update at pivot k: `(nrow - k - 1)·(ncol - k - 1)` for the
///   panel-internal slice plus `(nrow - k - 1)·(nrow - ncol)` for the
///   trailing-row update; the kernel issues `2 × mul-add` per element.
///
/// Sum over k = 0..ncol gives:
///     work = ncol·(nrow − 1) − ncol·(ncol − 1)/2     [scaling+axpy]
///          + 2·∑_{k=0}^{ncol-1} (nrow − k − 1)·(nrow − k − 1)
///                                                    [Schur muladds]
///
/// We report this as a single "flop_count" — matches the cost model
/// used by issue #14's back-of-envelope (~250 MFLOP for 2829×433).
fn ldlt_flop_count(nrow: usize, ncol: usize) -> u64 {
    let n = nrow as i128;
    let c = ncol as i128;
    // Scaling axpy: per pivot k, (n-k-1) divides.
    let scaling: i128 = (0..c).map(|k| n - k - 1).sum();
    // Schur: per pivot k, 2 muladds on (n-k-1)² entries (the trailing
    // square below pivot k). For k < ncol, this is the union of the
    // panel-internal slice and the trailing-row slice.
    let schur: i128 = (0..c).map(|k| 2 * (n - k - 1) * (n - k - 1)).sum();
    let total = scaling + schur;
    debug_assert!(total >= 0);
    total as u64
}

/// Analytic SIMD-body vs scalar-tail decomposition for the quad-panel
/// Schur kernel applied to a frontal of `nrow` rows, `ncol` fully-summed
/// columns, panel block size `bs`, with NEON f64 lane count `lanes`.
///
/// The quad kernel processes trailing columns four at a time:
///   for j in (k+n_elim..nrow).step_by(4):
///       dst0 = a[k=j..nrow]            len0 = nrow - j
///       dst1 = a[j+1..nrow]            len1 = nrow - j - 1
///       dst2 = a[j+2..nrow]            len2 = nrow - j - 2
///       dst3 = a[j+3..nrow]            len3 = nrow - j - 3
/// The first 3 rows are scalar-capped; the bulk (rows j+3..nrow) of all
/// four dsts has length `nrow - j - 3` and goes through the SIMD body
/// (lanes elements/iter) with a tail of `(nrow - j - 3) % lanes` lane
/// elements processed via partial_load/store.
///
/// We sum across all panels (bs at a time) and all quad starts; the
/// dual/single fall-through for the trailing 2–3 columns is small
/// (≤ 3 columns per panel) so we approximate it as SIMD too.
fn simd_elements_quad(nrow: usize, ncol: usize, bs: usize, lanes: usize) -> (u64, u64) {
    let mut simd = 0u64;
    let mut tail = 0u64;
    let mut k = 0;
    while k < ncol {
        let panel = bs.min(ncol - k);
        // Deferred Schur runs over j ∈ [k+panel, nrow).
        let j_start = k + panel;
        let mut j = j_start;
        while j + 3 < nrow {
            // Bulk per dst (4 dsts): nrow - j - 3 elements each.
            let bulk = (nrow - j - 3) as u64;
            for _ in 0..4 {
                let simd_elems = bulk - (bulk % lanes as u64);
                let tail_elems = bulk % lanes as u64;
                // Each element is processed `panel` times (n_elim rank-1
                // contributions accumulated).
                simd += simd_elems * panel as u64;
                tail += tail_elems * panel as u64;
            }
            j += 4;
        }
        // Trailing dual / single columns at end of trailing range.
        while j < nrow {
            let bulk = (nrow - j) as u64;
            let simd_elems = bulk - (bulk % lanes as u64);
            let tail_elems = bulk % lanes as u64;
            simd += simd_elems * panel as u64;
            tail += tail_elems * panel as u64;
            j += 1;
        }
        k += panel;
    }
    (simd, tail)
}

#[derive(Default, Clone, Debug)]
struct PanelDiagSnapshot {
    panel_full: u64,
    panel_partial: u64,
    panel_delayed: u64,
    pivots_inline: u64,
    pivots_scalar: u64,
    scalar_tail_steps: u64,
}

fn snapshot_diag() -> PanelDiagSnapshot {
    let snap = panel_diag::snapshot();
    let mut out = PanelDiagSnapshot::default();
    for (k, v) in snap.iter() {
        match *k {
            "panel_full" => out.panel_full = *v,
            "panel_partial" => out.panel_partial = *v,
            "panel_delayed" => out.panel_delayed = *v,
            "pivots_inline" => out.pivots_inline = *v,
            "pivots_scalar" => out.pivots_scalar = *v,
            "scalar_tail_steps" => out.scalar_tail_steps = *v,
            _ => {}
        }
    }
    out
}

struct ShapeResult {
    label: String,
    nrow: usize,
    ncol: usize,
    fma: bool,
    block_size: usize,
    times_ns: Vec<u128>,
    diag: PanelDiagSnapshot,
    flop_count: u64,
    simd_elements: u64,
    scalar_tail_elements: u64,
    nelim: usize,
    inertia_pos: usize,
    inertia_neg: usize,
    inertia_zero: usize,
}

#[allow(clippy::too_many_arguments)]
fn time_blocked(
    label: &str,
    nrow: usize,
    ncol: usize,
    mat_factory: impl Fn() -> SymmetricMatrix,
    fma: bool,
    block_size: usize,
    reps: usize,
    lanes: usize,
) -> ShapeResult {
    let bk = BunchKaufmanParams {
        block_size,
        fma,
        ..BunchKaufmanParams::default()
    };

    // Warm-up — exercise allocator paths so the first measured rep
    // doesn't pay malloc costs the steady-state path doesn't see.
    {
        let mut mat = mat_factory();
        let mut scratch = FactorScratch::new();
        let _ = factor_frontal_blocked_in_place(&mut mat, ncol, false, &bk)
            .expect("warmup factor_blocked");
        let mut mat2 = mat_factory();
        let _ = feral::dense::factor::factor_frontal_blocked_in_place_with_scratch(
            &mut mat2,
            ncol,
            false,
            &bk,
            &mut scratch,
        )
        .expect("warmup with scratch");
    }

    let mut times_ns = Vec::with_capacity(reps);
    let mut last_factors = None;
    PANEL_DIAG_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    panel_diag::reset();
    let mut scratch = FactorScratch::new();
    for _ in 0..reps {
        let mut mat = mat_factory();
        let t0 = Instant::now();
        let factors = feral::dense::factor::factor_frontal_blocked_in_place_with_scratch(
            &mut mat,
            ncol,
            false,
            &bk,
            &mut scratch,
        )
        .expect("blocked factor");
        let dt = t0.elapsed().as_nanos();
        times_ns.push(dt);
        last_factors = Some(factors);
    }
    let diag = snapshot_diag();
    PANEL_DIAG_ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);

    let flop_count = ldlt_flop_count(nrow, ncol);
    let (simd_elements, scalar_tail_elements) = simd_elements_quad(nrow, ncol, block_size, lanes);

    let f = last_factors.expect("at least one rep");
    ShapeResult {
        label: label.to_string(),
        nrow,
        ncol,
        fma,
        block_size,
        times_ns,
        diag,
        flop_count,
        simd_elements,
        scalar_tail_elements,
        nelim: f.nelim,
        inertia_pos: f.inertia.positive,
        inertia_neg: f.inertia.negative,
        inertia_zero: f.inertia.zero,
    }
}

fn time_scalar_reference(
    _nrow: usize,
    ncol: usize,
    mat_factory: impl Fn() -> SymmetricMatrix,
    reps: usize,
) -> Vec<u128> {
    let bk = BunchKaufmanParams::default();
    let mut times_ns = Vec::with_capacity(reps);
    // Warm-up.
    {
        let mat = mat_factory();
        let _ = factor_frontal(&mat, ncol, false, &bk).expect("warmup scalar");
    }
    for _ in 0..reps {
        let mat = mat_factory();
        let t0 = Instant::now();
        let _ = factor_frontal(&mat, ncol, false, &bk).expect("scalar factor");
        let dt = t0.elapsed().as_nanos();
        times_ns.push(dt);
    }
    times_ns
}

fn median(v: &[u128]) -> u128 {
    let mut x: Vec<u128> = v.to_vec();
    x.sort_unstable();
    x[x.len() / 2]
}

fn min(v: &[u128]) -> u128 {
    *v.iter().min().unwrap_or(&0)
}

fn print_result(label_prefix: &str, r: &ShapeResult) {
    let med = median(&r.times_ns);
    let lo = min(&r.times_ns);
    let med_ms = med as f64 / 1.0e6;
    let lo_ms = lo as f64 / 1.0e6;
    let gflops_med = r.flop_count as f64 / med as f64; // ns → GFLOPS = flops / ns
    let gflops_lo = r.flop_count as f64 / lo as f64;
    let total_simd_ops = r.simd_elements * 2; // mul-add → 2 FLOPs/element
    let total_scalar_ops = r.scalar_tail_elements * 2;
    let simd_frac =
        total_simd_ops as f64 / (total_simd_ops + total_scalar_ops).max(1) as f64 * 100.0;
    println!(
        "\n[{label_prefix}] {} nrow={} ncol={} bs={} fma={}",
        r.label, r.nrow, r.ncol, r.block_size, r.fma
    );
    println!(
        "  time   : med {:>8.2} ms   min {:>8.2} ms   (reps={})",
        med_ms,
        lo_ms,
        r.times_ns.len()
    );
    println!(
        "  flops  : {:>12} ({:.2} GFLOP)",
        r.flop_count,
        r.flop_count as f64 / 1.0e9
    );
    println!(
        "  rate   : med {:>6.2} GFLOPS   peak {:>6.2} GFLOPS",
        gflops_med, gflops_lo
    );
    println!(
        "  simd   : {:>12} SIMD elements   {:>10} scalar-tail elements   ({:.3}% SIMD by FLOP)",
        r.simd_elements, r.scalar_tail_elements, simd_frac
    );
    println!(
        "  panels : full={}  partial={}  delayed={}",
        r.diag.panel_full / r.times_ns.len() as u64,
        r.diag.panel_partial / r.times_ns.len() as u64,
        r.diag.panel_delayed / r.times_ns.len() as u64,
    );
    println!(
        "  pivots : inline={}  scalar={}  scalar_tail_steps={}",
        r.diag.pivots_inline / r.times_ns.len() as u64,
        r.diag.pivots_scalar / r.times_ns.len() as u64,
        r.diag.scalar_tail_steps / r.times_ns.len() as u64,
    );
    println!(
        "  nelim  : {} / ncol={}    inertia (+,−,0) = ({},{},{})",
        r.nelim, r.ncol, r.inertia_pos, r.inertia_neg, r.inertia_zero
    );
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().as_deref() {
        Some("1") | Some("true") | Some("yes") => true,
        Some("0") | Some("false") | Some("no") => false,
        _ => default,
    }
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn main() {
    let reps = env_usize("PROBE_REPS", 5);
    let block_size = env_usize("PROBE_BLOCK_SIZE", 64);
    let fma = env_bool("PROBE_FMA", false);
    let shapes = env_string("PROBE_SHAPES", "full,wide");

    let lanes = detected_f64_lanes();
    println!("probe_wide_supernode (issue #14)");
    println!(
        "  target: {} (F64_LANES = {}, used for SIMD-attribution only)",
        std::env::consts::ARCH,
        lanes
    );
    println!(
        "  knobs:  reps={} block_size={} fma={} shapes={}",
        reps, block_size, fma, shapes
    );
    println!("  shapes:");
    println!("    snode 3607 ~ (1928 × 1928)  — root supernode, full-square");
    println!("    snode 3593 ~ (2829 ×  433)  — wide trailing rows, narrow panel");

    let want_full = shapes.split(',').any(|s| s.trim() == "full");
    let want_wide = shapes.split(',').any(|s| s.trim() == "wide");

    if want_full {
        let nrow = 1928;
        let ncol = 1928;
        println!("\n==== Shape A: snode 3607 ({} × {}) ====", nrow, ncol);
        // Scalar reference is too slow at 1928×1928 (issue #14 reports
        // ~152 ms for the panel path; the scalar would be hours). Time
        // a single rep at lower precision so we still have a baseline.
        if env_bool("PROBE_SCALAR_FULL", false) {
            println!("  PROBE_SCALAR_FULL=1 — running 1 scalar reference rep (slow)");
            let scalar = time_scalar_reference(nrow, ncol, || make_spd_frontal(nrow), 1);
            println!(
                "  scalar_ref: {:.2} ms (1 rep, factor_frontal)",
                scalar[0] as f64 / 1.0e6
            );
        } else {
            println!("  scalar_ref: skipped (PROBE_SCALAR_FULL=1 to enable; >> 1 s)");
        }
        let r_spd = time_blocked(
            "SPD",
            nrow,
            ncol,
            || make_spd_frontal(nrow),
            fma,
            block_size,
            reps,
            lanes,
        );
        print_result("A.spd", &r_spd);
        let r_kkt = time_blocked(
            "KKT",
            nrow,
            ncol,
            || make_kkt_frontal(nrow),
            fma,
            block_size,
            reps,
            lanes,
        );
        print_result("A.kkt", &r_kkt);
    }

    if want_wide {
        let nrow = 2829;
        let ncol = 433;
        println!("\n==== Shape B: snode 3593 ({} × {}) ====", nrow, ncol);
        // Wide / narrow-panel — scalar reference is feasible here.
        let scalar = time_scalar_reference(nrow, ncol, || make_spd_frontal(nrow), reps.min(3));
        println!(
            "  scalar_ref (SPD): med {:.2} ms   min {:.2} ms   (reps={})",
            median(&scalar) as f64 / 1.0e6,
            min(&scalar) as f64 / 1.0e6,
            scalar.len()
        );
        let r_spd = time_blocked(
            "SPD",
            nrow,
            ncol,
            || make_spd_frontal(nrow),
            fma,
            block_size,
            reps,
            lanes,
        );
        print_result("B.spd", &r_spd);
        let r_kkt = time_blocked(
            "KKT",
            nrow,
            ncol,
            || make_kkt_frontal(nrow),
            fma,
            block_size,
            reps,
            lanes,
        );
        print_result("B.kkt", &r_kkt);
    }

    // FMA delta — quick A/B if the user did not already set PROBE_FMA.
    if !fma && want_wide {
        println!("\n==== FMA A/B (shape B SPD, single rep each, indicative only) ====");
        let r_no = time_blocked(
            "no-fma",
            2829,
            433,
            || make_spd_frontal(2829),
            false,
            block_size,
            3,
            lanes,
        );
        let r_yes = time_blocked(
            "fma",
            2829,
            433,
            || make_spd_frontal(2829),
            true,
            block_size,
            3,
            lanes,
        );
        let med_no = median(&r_no.times_ns) as f64 / 1.0e6;
        let med_yes = median(&r_yes.times_ns) as f64 / 1.0e6;
        let gfl_no = r_no.flop_count as f64 / median(&r_no.times_ns) as f64;
        let gfl_yes = r_yes.flop_count as f64 / median(&r_yes.times_ns) as f64;
        println!(
            "  no-fma: {:.2} ms ({:.2} GFLOPS)   fma: {:.2} ms ({:.2} GFLOPS)   speedup: {:.2}x",
            med_no,
            gfl_no,
            med_yes,
            gfl_yes,
            med_no / med_yes.max(1e-9)
        );
    }

    println!("\nDone.");
}
