//! Phase B-1.5 panel/scalar attribution probe
//! (`dev/research/dense-kernel-attribution-2026-04-28.md`).
//!
//! For each matrix in a representative mix, factors the matrix once
//! with `feral::dense::factor::PANEL_DIAG_ENABLED = true` and prints
//! the panel counter snapshot. This answers two questions:
//!
//!   1. What fraction of pivots go through the panel inline path vs
//!      the scalar fallback path?
//!   2. Among scalar-fallback bails from the panel, what is the
//!      dominant trigger reason (swap-2x2, swap-1x1-wins, growth/det
//!      reject, etc.)?
//!
//! Used to decide whether the next dense-kernel lever should be:
//!   a) widening `apply_blocked_schur_panel` from NR=2 to NR=4, or
//!   b) extending Phase A 2x2 to handle the swap-required case, or
//!   c) lifting the panel size for near-root supernodes.
//!
//! Run via: `cargo run --release --bin probe_panel_attribution`

use feral::dense::factor::{panel_diag, PANEL_DIAG_ENABLED};
use feral::numeric::factorize::factorize_multifrontal;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, ZeroPivotAction};
use std::path::PathBuf;
use std::sync::atomic::Ordering;

const MATRICES: &[(&str, &str)] = &[
    ("HS118_0000", "data/matrices/kkt/HS118/HS118_0000.mtx"),
    ("BATCH_0000", "data/matrices/kkt/BATCH/BATCH_0000.mtx"),
    ("AVION2_0000", "data/matrices/kkt/AVION2/AVION2_0000.mtx"),
    ("HAHN1_0000", "data/matrices/kkt/HAHN1/HAHN1_0000.mtx"),
    ("VESUVIO_0000", "data/matrices/kkt/VESUVIO/VESUVIO_0000.mtx"),
    ("VESUVIA_0000", "data/matrices/kkt/VESUVIA/VESUVIA_0000.mtx"),
    (
        "CRESC132_0000",
        "data/matrices/kkt/CRESC132/CRESC132_0000.mtx",
    ),
    (
        "CRESC100_0000",
        "data/matrices/kkt/CRESC100/CRESC100_0000.mtx",
    ),
    ("ACOPR30_0000", "data/matrices/kkt/ACOPR30/ACOPR30_0000.mtx"),
    (
        "CHAINWOO_0000",
        "data/matrices/kkt/CHAINWOO/CHAINWOO_0000.mtx",
    ),
];

fn fmt_pct(num: u64, denom: u64) -> String {
    if denom == 0 {
        "  -  ".to_string()
    } else {
        format!("{:5.1}%", 100.0 * num as f64 / denom as f64)
    }
}

fn print_row(label: &str, counts: &[(&str, u64)], total_pivots: u64) {
    let pi = counts
        .iter()
        .find(|(k, _)| *k == "pivots_inline")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    let ps = counts
        .iter()
        .find(|(k, _)| *k == "pivots_scalar")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    let pf = counts
        .iter()
        .find(|(k, _)| *k == "panel_full")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    let pp = counts
        .iter()
        .find(|(k, _)| *k == "panel_partial")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    let pd = counts
        .iter()
        .find(|(k, _)| *k == "panel_delayed")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    let st = counts
        .iter()
        .find(|(k, _)| *k == "scalar_tail_steps")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    let f_swap = counts
        .iter()
        .find(|(k, _)| *k == "fallback_2x2_need_swap_or_bound")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    let f_w11 = counts
        .iter()
        .find(|(k, _)| *k == "fallback_2x2_swap_1x1_wins")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    let f_lpk = counts
        .iter()
        .find(|(k, _)| *k == "fallback_2x2_lapack_1x1_wins")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    let f_gd = counts
        .iter()
        .find(|(k, _)| *k == "fallback_2x2_growth_or_det")
        .map(|(_, v)| *v)
        .unwrap_or(0);

    let f_total = f_swap + f_w11 + f_lpk + f_gd;
    println!(
        "{label:<14} pivots: in={pi:6}({inline_pct}) scal={ps:5}({scal_pct}) tail={st:4}  panels: full={pf:4} part={pp:4} dly={pd:3}  bails(2x2): swap={f_swap}({swap_p}) w11={f_w11}({w11_p}) lpk={f_lpk}({lpk_p}) g/d={f_gd}({gd_p})",
        label = label,
        pi = pi,
        inline_pct = fmt_pct(pi, total_pivots),
        ps = ps,
        scal_pct = fmt_pct(ps, total_pivots),
        st = st,
        pf = pf,
        pp = pp,
        pd = pd,
        f_swap = f_swap,
        swap_p = fmt_pct(f_swap, f_total),
        f_w11 = f_w11,
        w11_p = fmt_pct(f_w11, f_total),
        f_lpk = f_lpk,
        lpk_p = fmt_pct(f_lpk, f_total),
        f_gd = f_gd,
        gd_p = fmt_pct(f_gd, f_total),
    );
}

fn main() {
    let snode_params = SupernodeParams::default();
    let bk = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    let factor_params = feral::numeric::factorize::NumericParams::with_bk(bk);

    PANEL_DIAG_ENABLED.store(true, Ordering::Relaxed);

    let mut aggregate: Vec<(String, u64)> = Vec::new();
    for (label, _) in panel_diag::snapshot() {
        aggregate.push((label.to_string(), 0));
    }

    println!(
        "Phase B-1.5 panel attribution\n\
         Counts are per-matrix (one factor per matrix). Inline = pivots committed inside\n\
         the deferred-Schur panel. Scalar = scalar_pivot_step calls (post-fallback or\n\
         scalar tail). Bails are reasons lblt_panel_frontal returned ScalarFallback*.\n"
    );

    for (name, path) in MATRICES {
        let p = PathBuf::from(path);
        let mtx = match read_mtx(&p) {
            Ok(m) => m,
            Err(e) => {
                println!("SKIP {}: read {}", name, e);
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                println!("SKIP {}: csc {}", name, e);
                continue;
            }
        };
        let sym = match symbolic_factorize(&csc, &snode_params) {
            Ok(s) => s,
            Err(e) => {
                println!("SKIP {}: symbolic {}", name, e);
                continue;
            }
        };

        panel_diag::reset();
        match factorize_multifrontal(&csc, &sym, &factor_params) {
            Ok(_) => {}
            Err(e) => {
                println!("SKIP {}: factor {}", name, e);
                continue;
            }
        }
        let snap = panel_diag::snapshot();
        let snap_ref: Vec<(&str, u64)> = snap.iter().map(|(k, v)| (*k, *v)).collect();
        let pi = snap_ref
            .iter()
            .find(|(k, _)| *k == "pivots_inline")
            .map(|(_, v)| *v)
            .unwrap_or(0);
        let ps = snap_ref
            .iter()
            .find(|(k, _)| *k == "pivots_scalar")
            .map(|(_, v)| *v)
            .unwrap_or(0);
        let total = pi + ps;
        print_row(name, &snap_ref, total);

        for (i, (_, v)) in snap.iter().enumerate() {
            aggregate[i].1 += v;
        }
    }

    println!();
    let agg_ref: Vec<(&str, u64)> = aggregate.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let pi = agg_ref
        .iter()
        .find(|(k, _)| *k == "pivots_inline")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    let ps = agg_ref
        .iter()
        .find(|(k, _)| *k == "pivots_scalar")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    print_row("AGGREGATE", &agg_ref, pi + ps);
}
