//! Per-supernode profiler dump on a single matrix.
//!
//! Usage:
//!   cargo run --release --bin diag_qcqp_profile -- <path-to.mtx>

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use feral::numeric::factorize::{factorize_multifrontal, NumericParams, Profiler};
use feral::scaling::ScalingStrategy;
use feral::symbolic::{
    symbolic_factorize_with_method, AmalgamationStrategy, OrderingMethod, OrderingPreprocess,
    SupernodeParams,
};
use feral::{read_mtx, BunchKaufmanParams};

const NCOL_BUCKETS: &[(&str, usize, usize)] = &[
    ("ncol=1", 1, 1),
    ("ncol=2", 2, 2),
    ("ncol=3-4", 3, 4),
    ("ncol=5-8", 5, 8),
    ("ncol=9-16", 9, 16),
    ("ncol=17-32", 17, 32),
    ("ncol=33-64", 33, 64),
    ("ncol=65-128", 65, 128),
    ("ncol>128", 129, usize::MAX),
];

fn run(
    label: &str,
    csc: &feral::sparse::csc::CscMatrix,
    method: OrderingMethod,
    nemin: usize,
    amalg: AmalgamationStrategy,
    preproc: OrderingPreprocess,
) {
    let snode = SupernodeParams {
        nemin,
        preprocess: preproc,
        amalgamation_strategy: amalg,
        ..SupernodeParams::default()
    };
    let sym = match symbolic_factorize_with_method(csc, &snode, method) {
        Ok(s) => s,
        Err(e) => {
            println!("\n=== {} === SYM_ERR {}", label, e);
            return;
        }
    };
    let prof = Arc::new(Mutex::new(Profiler::new()));
    let np = NumericParams {
        bk: BunchKaufmanParams::default(),
        scaling: ScalingStrategy::Auto,
        profiler: Some(Arc::clone(&prof)),
        ..NumericParams::default()
    };
    let (factors, _) = match factorize_multifrontal(csc, &sym, &np) {
        Ok(p) => p,
        Err(e) => {
            println!("\n=== {} === NUM_ERR {}", label, e);
            return;
        }
    };
    let prof = prof.lock().unwrap();
    let report = prof.report();

    println!("\n=== {} ===", label);
    println!(
        "n_supernodes: {}, factor_nnz_L: {}, total_us: {} (loop {}, prologue {}, epilogue {})",
        report.n_supernodes,
        factors.factor_nnz(),
        report.total_us,
        report.loop_us,
        report.prologue_us,
        report.epilogue_us,
    );
    println!("overhead = {:.1}% of total", report.overhead_pct);

    // Per-nrow buckets (built-in)
    println!("\nby nrow (front size):");
    println!(
        "  {:<10} {:>8} {:>10} {:>8} {:>8}",
        "range", "count", "sum_us", "pct", "avg_us"
    );
    for b in &report.buckets {
        println!(
            "  {:<10} {:>8} {:>10} {:>7.1}% {:>8.1}",
            b.range, b.count, b.sum_us, b.pct_of_total, b.avg_us
        );
    }

    // Per-ncol buckets (custom)
    let mut ncol_counts = vec![0usize; NCOL_BUCKETS.len()];
    let mut ncol_us = vec![0u64; NCOL_BUCKETS.len()];
    let mut sum_us_all = 0u64;
    for t in prof.timings() {
        sum_us_all += t.us;
        for (i, &(_, lo, hi)) in NCOL_BUCKETS.iter().enumerate() {
            if t.ncol >= lo && t.ncol <= hi {
                ncol_counts[i] += 1;
                ncol_us[i] += t.us;
                break;
            }
        }
    }
    println!("\nby ncol (eliminated columns per supernode):");
    println!(
        "  {:<14} {:>8} {:>10} {:>8} {:>8}",
        "range", "count", "sum_us", "pct", "avg_us"
    );
    for (i, &(label, _, _)) in NCOL_BUCKETS.iter().enumerate() {
        let pct = if sum_us_all > 0 {
            (ncol_us[i] as f64) * 100.0 / (sum_us_all as f64)
        } else {
            0.0
        };
        let avg = if ncol_counts[i] > 0 {
            (ncol_us[i] as f64) / (ncol_counts[i] as f64)
        } else {
            0.0
        };
        println!(
            "  {:<14} {:>8} {:>10} {:>7.1}% {:>8.1}",
            label, ncol_counts[i], ncol_us[i], pct, avg
        );
    }

    // Top 5 hottest supernodes
    let mut sorted: Vec<_> = prof.timings().iter().collect();
    sorted.sort_by_key(|t| std::cmp::Reverse(t.us));
    println!("\ntop 5 hottest supernodes:");
    println!(
        "  {:>5} {:>6} {:>5} {:>5} {:>10}",
        "rank", "snode", "nrow", "ncol", "us"
    );
    for (rank, t) in sorted.iter().take(5).enumerate() {
        println!(
            "  {:>5} {:>6} {:>5} {:>5} {:>10}",
            rank, t.snode_idx, t.nrow, t.ncol, t.us
        );
    }

    // Cumulative-by-ncol: how much wall time comes from ncol<=K?
    let mut sorted_us_by_ncol: Vec<(usize, u64)> =
        prof.timings().iter().map(|t| (t.ncol, t.us)).collect();
    sorted_us_by_ncol.sort_by_key(|&(c, _)| c);
    let mut cum_count = 0usize;
    let mut cum_us = 0u64;
    let n_total = sorted_us_by_ncol.len();
    let total_us: u64 = sorted_us_by_ncol.iter().map(|&(_, u)| u).sum();
    println!("\ncumulative by ncol threshold:");
    println!(
        "  {:<12} {:>10} {:>10} {:>10} {:>10}",
        "ncol<=", "count", "%count", "sum_us", "%us"
    );
    let mut last_emitted = 0usize;
    for (i, &(c, u)) in sorted_us_by_ncol.iter().enumerate() {
        cum_count += 1;
        cum_us += u;
        let next_c = sorted_us_by_ncol.get(i + 1).map(|&(c2, _)| c2).unwrap_or(0);
        if c != next_c && (c <= 8 || c % 8 == 0 || i == n_total - 1) && c != last_emitted {
            println!(
                "  {:<12} {:>10} {:>9.1}% {:>10} {:>9.1}%",
                c,
                cum_count,
                (cum_count as f64) * 100.0 / (n_total as f64),
                cum_us,
                if total_us > 0 {
                    (cum_us as f64) * 100.0 / (total_us as f64)
                } else {
                    0.0
                },
            );
            last_emitted = c;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = if args.is_empty() {
        PathBuf::from("data/matrices/kkt-mittelmann/qcqp1500-1c/qcqp1500-1c_0000.mtx")
    } else {
        PathBuf::from(&args[0])
    };
    let mtx = read_mtx(&path).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    println!(
        "matrix: {}\n  n={} stored_nnz={}",
        path.display(),
        csc.n,
        csc.row_idx.len()
    );

    // Default config (what diag_mittelmann uses).
    run(
        "default(MetisND, nemin=32, Renumber, Compress)",
        &csc,
        OrderingMethod::MetisND,
        32,
        AmalgamationStrategy::Renumber,
        OrderingPreprocess::LdltCompress,
    );

    // Best from prior knob sweep on this matrix.
    run(
        "Amf, nemin=32, Auto, Auto",
        &csc,
        OrderingMethod::Amf,
        32,
        AmalgamationStrategy::Auto,
        OrderingPreprocess::Auto,
    );

    // Probe what no-compress does to the supernode shape.
    run(
        "MetisND, nemin=8, Renumber, no-compress",
        &csc,
        OrderingMethod::MetisND,
        8,
        AmalgamationStrategy::Renumber,
        OrderingPreprocess::None,
    );
}
