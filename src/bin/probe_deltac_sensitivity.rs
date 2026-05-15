//! Probe: how sensitive are feral's Auto routing heuristics and
//! end-to-end timing to the IPM constraint regularization δ_c that
//! the corpus matrices were dumped with?
//!
//! ## Why this exists
//!
//! The KKT corpus under `data/matrices/kkt/` is a set of pre-
//! regularized snapshots from inside an IPM iteration: every dual
//! diagonal carries an explicit `-δ_c` (e.g. `-1e-8` on VESUVIO).
//! feral's adaptive scaling routing (`pick_scaling_strategy`) and
//! ordering preprocess routing (`pick_ordering_preprocess`) were
//! tuned against that distribution. If a downstream consumer
//! (POUNCE, etc.) picks a different δ_c — say 1e-4 because of a
//! larger `mu_init` — those routing decisions might drift, and
//! perf could regress in a way the corpus cannot show.
//!
//! ## What the probe does
//!
//! For each matrix:
//!   1. Detect the **dual reg block** as the longest contiguous tail
//!      run of identical small-magnitude diagonal entries
//!      (|x| ≤ `DUAL_DETECT_MAG`). If no such block is found, the
//!      matrix is reported as "no dual reg detected" and skipped.
//!   2. For each `mult ∈ {1e-4, 1e-2, 1, 1e2, 1e4}`, build a
//!      perturbed CSC where every detected dual-block diagonal is
//!      multiplied by `mult` (sign preserved).
//!   3. Record `pick_scaling_strategy`, `pick_ordering_preprocess`,
//!      and 5-run-median symbolic + numeric wall, plus the residual
//!      norm of `solve_sparse_refined`.
//!
//! Output is a per-matrix table; rows that share the same routing
//! across all multipliers are robust, rows that flip routing or
//! materially shift timing are sensitive.
//!
//! ## Limitations
//!
//! * Detection is value-pattern heuristic, not structural. Matrices
//!   without a clean dual reg block (HS118-class small NLPs whose
//!   slack-block diagonals are barrier perturbations of varying
//!   magnitudes) are skipped — they do not have a single δ_c to
//!   sweep.
//! * Perturbation only touches the dual diagonal; it does not
//!   re-derive the H + Σ block, so very large multipliers may
//!   produce matrices that no real IPM would emit.
//! * 5-run median is enough to flag big shifts but not enough to
//!   distinguish 5% wall changes from noise.

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::factorize_multifrontal;
use feral::scaling::{pick_scaling_strategy, ScalingStrategy};
use feral::symbolic::SupernodeParams;
use feral::symbolic::{pick_ordering_preprocess, symbolic_factorize, OrderingPreprocess};
use feral::{
    read_mtx, solve_sparse_refined, BunchKaufmanParams, CscMatrix, NumericParams, ZeroPivotAction,
};

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
        "MUONSINE_0000",
        "data/matrices/kkt/MUONSINE/MUONSINE_0000.mtx",
    ),
    ("KIRBY2_0007", "data/matrices/kkt/KIRBY2/KIRBY2_0007.mtx"),
    (
        "BENNETT5_0000",
        "data/matrices/kkt/BENNETT5/BENNETT5_0000.mtx",
    ),
    ("MSS1_0009", "data/matrices/kkt/MSS1/MSS1_0009.mtx"),
];

const MULTIPLIERS: &[f64] = &[1e-4, 1e-2, 1.0, 1e2, 1e4];
const DUAL_DETECT_MAG: f64 = 1e-2;
/// Two diagonal values are considered "the same" if their relative
/// difference is below this threshold. Chosen permissively so that
/// reg blocks dumped at single-precision-rounded values still cluster.
const DUAL_DETECT_REL_TOL: f64 = 1e-4;
/// Minimum tail-run length to call something a dual-reg block. Below
/// this we cannot tell δ_c from coincidental small diagonals.
const MIN_TAIL_RUN: usize = 4;
const N_RUNS: usize = 5;

#[derive(Clone, Copy, Debug)]
struct DualBlock {
    /// Lowest column index (0-based) of the detected dual block.
    start: usize,
    /// One past the highest column index of the detected dual block.
    end: usize,
    /// The common δ_c value. Sign preserved.
    delta_c: f64,
}

fn diag_value(m: &CscMatrix, j: usize) -> Option<(usize, f64)> {
    for k in m.col_ptr[j]..m.col_ptr[j + 1] {
        if m.row_idx[k] == j {
            return Some((k, m.values[k]));
        }
    }
    None
}

fn detect_dual_block(m: &CscMatrix) -> Option<DualBlock> {
    let n = m.n;
    if n < MIN_TAIL_RUN {
        return None;
    }
    // Walk from the end; stop when the diagonal value either is
    // missing, exceeds DUAL_DETECT_MAG, or differs from the running
    // anchor by more than DUAL_DETECT_REL_TOL relative.
    let (_, anchor) = diag_value(m, n - 1)?;
    if anchor.abs() > DUAL_DETECT_MAG || anchor == 0.0 {
        return None;
    }
    let mut start = n - 1;
    while start > 0 {
        let j = start - 1;
        let (_, v) = match diag_value(m, j) {
            Some(p) => p,
            None => break,
        };
        if v.abs() > DUAL_DETECT_MAG {
            break;
        }
        let rel = (v - anchor).abs() / anchor.abs().max(f64::MIN_POSITIVE);
        if rel > DUAL_DETECT_REL_TOL {
            break;
        }
        start -= 1;
    }
    let end = n;
    if end - start < MIN_TAIL_RUN {
        return None;
    }
    Some(DualBlock {
        start,
        end,
        delta_c: anchor,
    })
}

fn perturb(m: &CscMatrix, block: DualBlock, mult: f64) -> CscMatrix {
    let mut out = m.clone();
    for j in block.start..block.end {
        if let Some((k, v)) = diag_value(&out, j) {
            out.values[k] = v * mult;
        }
    }
    out
}

fn ldlt_params() -> NumericParams {
    NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        },
        scaling: ScalingStrategy::Auto,
        small_leaf: Default::default(),
        profiler: None,
        parallel_telemetry: None,
        fma: false,
        allow_delayed_pivots: true,
        cascade_break_ratio: None,
        cascade_break_eps: None,
        min_parallel_flops: None,
    }
}

#[derive(Default, Debug)]
struct RunResult {
    sym_us: u64,
    num_us: u64,
    residual: f64,
    inertia_neg: usize,
    failed: bool,
}

fn one_run(csc: &CscMatrix) -> RunResult {
    let mut r = RunResult::default();
    let snode = SupernodeParams::default();
    let params = ldlt_params();

    let t = Instant::now();
    let sym = match symbolic_factorize(csc, &snode) {
        Ok(s) => s,
        Err(_) => {
            r.failed = true;
            return r;
        }
    };
    r.sym_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    let (factors, inertia) = match factorize_multifrontal(csc, &sym, &params) {
        Ok(p) => p,
        Err(_) => {
            r.failed = true;
            return r;
        }
    };
    r.num_us = t.elapsed().as_micros() as u64;
    r.inertia_neg = inertia.negative;

    let n = csc.n;
    let rhs: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.0).collect();
    let x = match solve_sparse_refined(csc, &factors, &rhs) {
        Ok(x) => x,
        Err(_) => {
            r.failed = true;
            return r;
        }
    };
    // Compute ||A·x − b||∞.
    let mut ax = vec![0.0_f64; n];
    for j in 0..n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = csc.values[k];
            ax[i] += v * x[j];
            if i != j {
                ax[j] += v * x[i];
            }
        }
    }
    let mut max_res = 0.0_f64;
    for i in 0..n {
        let r = (ax[i] - rhs[i]).abs();
        if r > max_res {
            max_res = r;
        }
    }
    r.residual = max_res;
    r
}

fn median_u64(xs: &mut [u64]) -> u64 {
    xs.sort_unstable();
    xs[xs.len() / 2]
}

fn run_with_warmup(csc: &CscMatrix) -> RunResult {
    let _ = one_run(csc); // warm-up, discarded
    let mut sym = Vec::with_capacity(N_RUNS);
    let mut num = Vec::with_capacity(N_RUNS);
    let mut last = RunResult::default();
    for _ in 0..N_RUNS {
        last = one_run(csc);
        if last.failed {
            return last;
        }
        sym.push(last.sym_us);
        num.push(last.num_us);
    }
    RunResult {
        sym_us: median_u64(&mut sym),
        num_us: median_u64(&mut num),
        residual: last.residual,
        inertia_neg: last.inertia_neg,
        failed: false,
    }
}

fn route_label(s: &ScalingStrategy) -> &'static str {
    match s {
        ScalingStrategy::Auto => "Auto",
        ScalingStrategy::Mc64Symmetric => "MC64",
        ScalingStrategy::InfNorm => "InfN",
        ScalingStrategy::Identity => "Idnt",
        ScalingStrategy::External(_) => "Extn",
    }
}

fn preprocess_label(p: &OrderingPreprocess) -> &'static str {
    match p {
        OrderingPreprocess::None => "None",
        OrderingPreprocess::LdltCompress => "Comp",
        OrderingPreprocess::Auto => "Auto",
    }
}

fn main() {
    println!(
        "{:<16} {:>6} {:>10} {:>5} | {:>9} {:>4} {:>4} {:>8} {:>8} {:>10} {:>5}",
        "matrix",
        "n",
        "delta_c",
        "block",
        "mult",
        "scal",
        "prep",
        "sym_us",
        "num_us",
        "residual",
        "neg",
    );
    println!(
        "{:-<16} {:->6} {:->10} {:->5}-+-{:->9} {:->4} {:->4} {:->8} {:->8} {:->10} {:->5}",
        "", "", "", "", "", "", "", "", "", "", "",
    );

    let mut total = 0usize;
    let mut detected = 0usize;
    let mut routing_flips = 0usize;

    for &(label, path) in MATRICES {
        if !Path::new(path).exists() {
            eprintln!("SKIP missing: {}", path);
            continue;
        }
        let mtx = match read_mtx(Path::new(path)) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("SKIP {}: read {}", label, e);
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP {}: csc {}", label, e);
                continue;
            }
        };
        total += 1;
        let n = csc.n;

        let block = match detect_dual_block(&csc) {
            Some(b) => b,
            None => {
                println!(
                    "{:<16} {:>6} {:>10} {:>5} | (no dual reg block detected — single-δ_c not applicable)",
                    label, n, "—", "—",
                );
                continue;
            }
        };
        detected += 1;

        let baseline_route_scal = pick_scaling_strategy(&csc);
        let baseline_route_prep = pick_ordering_preprocess(&csc);
        let mut prev_scal = baseline_route_scal.clone();
        let mut prev_prep = baseline_route_prep;
        let mut local_flip = false;

        for (idx, &mult) in MULTIPLIERS.iter().enumerate() {
            let pert = perturb(&csc, block, mult);
            let scal = pick_scaling_strategy(&pert);
            let prep = pick_ordering_preprocess(&pert);
            if scal != prev_scal || prep != prev_prep {
                local_flip = true;
            }
            prev_scal = scal.clone();
            prev_prep = prep;
            let r = run_with_warmup(&pert);
            let block_lbl = if idx == 0 {
                format!("{}", block.end - block.start)
            } else {
                String::new()
            };
            let dc_lbl = if idx == 0 {
                format!("{:.1e}", block.delta_c)
            } else {
                String::new()
            };
            let mtx_lbl = if idx == 0 {
                label.to_string()
            } else {
                String::new()
            };
            let n_lbl = if idx == 0 {
                format!("{}", n)
            } else {
                String::new()
            };
            if r.failed {
                println!(
                    "{:<16} {:>6} {:>10} {:>5} | {:>9.0e} {:>4} {:>4} {:>8} {:>8} {:>10} {:>5}",
                    mtx_lbl,
                    n_lbl,
                    dc_lbl,
                    block_lbl,
                    mult,
                    route_label(&scal),
                    preprocess_label(&prep),
                    "FAIL",
                    "FAIL",
                    "—",
                    "—",
                );
            } else {
                println!(
                    "{:<16} {:>6} {:>10} {:>5} | {:>9.0e} {:>4} {:>4} {:>8} {:>8} {:>10.1e} {:>5}",
                    mtx_lbl,
                    n_lbl,
                    dc_lbl,
                    block_lbl,
                    mult,
                    route_label(&scal),
                    preprocess_label(&prep),
                    r.sym_us,
                    r.num_us,
                    r.residual,
                    r.inertia_neg,
                );
            }
        }
        if local_flip {
            routing_flips += 1;
        }
    }

    println!();
    println!("matrices probed:                {}", total);
    println!("matrices with dual-reg block:   {}", detected);
    println!("matrices that flipped routing:  {}", routing_flips);
    println!();
    println!("legend:");
    println!("  scal: Auto-resolved scaling — MC64 / InfN / Idnt");
    println!("  prep: Auto-resolved preprocess — Comp = LdltCompress, None = passthrough");
    println!("  sym_us / num_us: 5-run median wall (us) over symbolic + numeric");
    println!("  residual: ||A·x − b||∞ from solve_sparse_refined");
    println!("  neg: count of negative pivots (inertia change is a red flag)");
}
