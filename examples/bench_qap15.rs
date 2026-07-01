//! Ad-hoc benchmark for the qap15 conic KKT (issue #91).
//!
//! Loads a symmetric MatrixMarket KKT and factors it under several
//! Solver configurations, reporting wall time and fill. Not a
//! committed fixture/test — a measurement harness for the issue #91
//! diagnosis (delayed-pivot blowup on a quasi-definite conic KKT).
//!
//! Usage: cargo run --release --example bench_qap15 -- <path.mtx>

use feral::symbolic::{
    pick_ordering_preprocess, symbolic_factorize_with_method, total_factor_nnz,
    AmalgamationStrategy, OrderingMethod, OrderingPreprocess, SupernodeParams,
};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, NumericParams, Solver};
use std::path::Path;

fn np_with_block_size(bs: usize) -> NumericParams {
    NumericParams {
        bk: BunchKaufmanParams {
            block_size: bs,
            ..BunchKaufmanParams::default()
        },
        ..NumericParams::default()
    }
}
use std::time::Instant;

/// Symbolic-only fill report: isolates AMD ordering quality (simplicial
/// `col_counts` fill, comparable to faer's `len_val`) from supernode
/// amalgamation padding (`factor_nnz_estimate`).
fn symbolic_report(csc: &CscMatrix) {
    println!("--- symbolic fill (no numeric factor) ---");
    for (mname, method) in [
        ("AMD", OrderingMethod::Amd),
        ("AMF", OrderingMethod::Amf),
        ("MetisND", OrderingMethod::MetisND),
    ] {
        for (sname, sn) in [
            ("default(Auto-prep)", SupernodeParams::default()),
            (
                "preprocess=None",
                SupernodeParams {
                    preprocess: OrderingPreprocess::None,
                    ..SupernodeParams::default()
                },
            ),
            (
                "nemin=1,None,Adjac",
                SupernodeParams {
                    nemin: 1,
                    preprocess: OrderingPreprocess::None,
                    amalgamation_strategy: AmalgamationStrategy::Adjacency,
                    ..SupernodeParams::default()
                },
            ),
        ] {
            match symbolic_factorize_with_method(csc, &sn, method) {
                Ok(sym) => {
                    let simplicial = total_factor_nnz(&sym.col_counts);
                    println!(
                        "  {mname:<8} {sname:<20} simplicial_nnz_L={simplicial:>10}  \
                         amalgamated_est={:>10}  n_supernodes={}",
                        sym.factor_nnz_estimate,
                        sym.supernodes.len()
                    );
                }
                Err(e) => println!("  {mname:<8} {sname:<20} ERR {e:?}"),
            }
        }
    }
    println!("  (faer AMD len_val reference = 13,370,955)");

    // None vs LdltCompress on the default ordering, to size the fill effect
    // of preprocessing (issue #91 race criterion).
    println!("--- preprocess None vs LdltCompress (default method) ---");
    println!("  predicate fires: {:?}", pick_ordering_preprocess(csc));
    for (pname, pp) in [
        ("None", OrderingPreprocess::None),
        ("LdltCompress", OrderingPreprocess::LdltCompress),
    ] {
        let sn = SupernodeParams {
            preprocess: pp,
            ..SupernodeParams::default()
        };
        match symbolic_factorize_with_method(csc, &sn, OrderingMethod::Amd) {
            Ok(sym) => println!(
                "  {pname:<14} simplicial_nnz_L={:>10}  est={:>10}",
                total_factor_nnz(&sym.col_counts),
                sym.factor_nnz_estimate
            ),
            Err(e) => println!("  {pname:<14} ERR {e:?}"),
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn run(name: &str, csc: &CscMatrix, build: impl Fn() -> Solver, reps: usize) {
    // Warm one factor to populate symbolic cache the way an IPM would
    // (symbolic is amortized across iterations; we want the per-factor
    // numeric cost, but report both first-call and steady-state).
    let mut first_ms = f64::NAN;
    let mut times = Vec::new();
    let mut nnz_l = 0usize;
    let mut fill = 0.0;
    let mut inertia = (0usize, 0usize, 0usize);
    let mut min_piv = 0.0;
    let mut max_piv = 0.0;
    let mut status = String::new();
    let mut solver = build();
    for r in 0..reps {
        let t = Instant::now();
        let st = solver.factor(csc, None);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if r == 0 {
            first_ms = ms;
            status = format!("{st:?}");
            if let Some(s) = solver.last_factor_stats() {
                nnz_l = s.nnz_l;
                fill = s.fill_ratio;
                inertia = (s.inertia.positive, s.inertia.negative, s.inertia.zero);
                min_piv = s.min_abs_pivot;
                max_piv = s.max_abs_pivot;
            }
        } else {
            times.push(ms);
        }
    }
    let steady = if times.is_empty() {
        first_ms
    } else {
        median(times)
    };
    println!(
        "{name:<24} first={first_ms:>9.1}ms steady={steady:>9.1}ms  \
         nnz_L={nnz_l:>10} fill={fill:>6.1}x  \
         inertia=(+{},-{},0:{})  |piv|=[{:.2e},{:.2e}]  {status}",
        inertia.0, inertia.1, inertia.2, min_piv, max_piv
    );
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| panic!("usage: bench_qap15 <path.mtx>"));
    let mtx = read_mtx(Path::new(&path)).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    println!(
        "matrix: n={} nnz={}  (faer AMD len_val reference = 13,370,955)",
        csc.n,
        csc.row_idx.len()
    );
    if std::env::var("SYMBOLIC").is_ok() {
        symbolic_report(&csc);
        return;
    }

    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    // Comma-separated config selector via CONFIGS env (default: all).
    let want = std::env::var("CONFIGS").unwrap_or_else(|_| "default,sqd,sqdfma,static,fma".into());
    let want: Vec<&str> = want.split(',').collect();
    let has = |k: &str| want.contains(&k);

    if has("default") {
        run("default (BK+delayed)", &csc, Solver::new, reps);
    }
    if has("sqd") {
        run("sqd_mode", &csc, || Solver::new().with_sqd_mode(true), reps);
    }
    if has("sqdfma") {
        run(
            "sqd_mode+fma",
            &csc,
            || Solver::new().with_sqd_mode(true).with_fma(true),
            reps,
        );
    }
    if has("static") {
        run(
            "static_pivot(1e-8)",
            &csc,
            || Solver::new().with_static_pivot_threshold(1e-8),
            reps,
        );
    }
    if has("fma") {
        run("default+fma", &csc, || Solver::new().with_fma(true), reps);
    }
    if has("noprep") {
        run(
            "preprocess=None",
            &csc,
            || {
                Solver::with_params(
                    NumericParams::default(),
                    SupernodeParams {
                        preprocess: OrderingPreprocess::None,
                        ..SupernodeParams::default()
                    },
                )
            },
            reps,
        );
    }
    if has("noprepfma") {
        run(
            "preprocess=None+fma",
            &csc,
            || {
                Solver::with_params(
                    NumericParams::default(),
                    SupernodeParams {
                        preprocess: OrderingPreprocess::None,
                        ..SupernodeParams::default()
                    },
                )
                .with_fma(true)
            },
            reps,
        );
    }
    for bs in [96usize, 128, 160, 192] {
        if has(&format!("bs{bs}")) {
            run(
                &format!("block_size={bs}"),
                &csc,
                || Solver::with_params(np_with_block_size(bs), SupernodeParams::default()),
                reps,
            );
        }
    }
    if has("nemin1") {
        run(
            "nemin=1 (no-amalg)",
            &csc,
            || {
                Solver::with_params(
                    NumericParams::default(),
                    SupernodeParams {
                        nemin: 1,
                        ..SupernodeParams::default()
                    },
                )
            },
            reps,
        );
    }
    if has("nemin1fma") {
        run(
            "nemin=1+fma",
            &csc,
            || {
                Solver::with_params(
                    NumericParams::default(),
                    SupernodeParams {
                        nemin: 1,
                        ..SupernodeParams::default()
                    },
                )
                .with_fma(true)
            },
            reps,
        );
    }
}
