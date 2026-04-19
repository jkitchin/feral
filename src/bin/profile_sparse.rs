//! Per-stage sparse-pipeline profiler.
//!
//! Picks a handful of representative KKT matrices and reports
//! per-call timings broken into:
//!   - symbolic factorize
//!   - numeric factorize (multifrontal)
//!   - one bare solve (`solve_sparse`)
//!   - full refined solve (`solve_sparse_refined`)
//!   - implied refinement overhead (`refined - bare`)
//!
//! Each stage is run `iters` times and the minimum is reported (best
//! representative of cache-warm cost). The SSIDS oracle's `solve_us`
//! sidecar value is also printed so we can see the gap matrix-by-
//! matrix instead of as a single corpus-wide geomean.
//!
//! Usage: `cargo run --release --bin profile_sparse`

use feral::numeric::factorize::factorize_multifrontal;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, solve_sparse, solve_sparse_refined, BunchKaufmanParams, ZeroPivotAction};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct OracleSolveUs {
    ssids: Option<u128>,
    mumps: Option<u128>,
}

fn read_oracle_solve(matrix_path: &Path) -> OracleSolveUs {
    fn extract(path: PathBuf) -> Option<u128> {
        let s = std::fs::read_to_string(&path).ok()?;
        // Cheap parse: find `"solve_us":` and the integer that follows.
        let key = "\"solve_us\":";
        let i = s.find(key)? + key.len();
        let rest = &s[i..];
        let end = rest
            .find(|c: char| !c.is_ascii_digit() && !c.is_whitespace())
            .unwrap_or(rest.len());
        rest[..end].trim().parse::<u128>().ok()
    }
    OracleSolveUs {
        ssids: extract(matrix_path.with_extension("ssids.json")),
        mumps: extract(matrix_path.with_extension("mumps.json")),
    }
}

fn time_min<F: FnMut()>(iters: usize, mut f: F) -> u128 {
    let mut best = u128::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        let us = t.elapsed().as_nanos();
        if us < best {
            best = us;
        }
    }
    best
}

fn run_one(family: &str, sample: &str, base: &str) {
    let matrix_path = PathBuf::from(format!(
        "data/matrices/kkt/{}/{}{}.mtx",
        family, family, sample
    ));
    let _ = base;
    let oracle = read_oracle_solve(&matrix_path);
    let mtx = match read_mtx(&matrix_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP {}/{}: {}", family, sample, e);
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP {}/{}: csc: {}", family, sample, e);
            return;
        }
    };
    let n = csc.n;
    let nnz = csc.row_idx.len();
    let snode_params = SupernodeParams::default();
    let factor_params = feral::numeric::factorize::NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    });

    // Stage timings. For symbolic + factor, use few iters since they
    // mutate global allocator state and we mostly care about ratios.
    let sym_iters = 5;
    let fac_iters = 5;
    let solve_iters = 200;

    let sym_ns = time_min(sym_iters, || {
        let _ = symbolic_factorize(&csc, &snode_params);
    });
    let sym = match symbolic_factorize(&csc, &snode_params) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP {}: sym: {}", family, e);
            return;
        }
    };
    let fac_ns = time_min(fac_iters, || {
        let _ = factorize_multifrontal(&csc, &sym, &factor_params);
    });
    let factors = match factorize_multifrontal(&csc, &sym, &factor_params) {
        Ok((f, _)) => f,
        Err(e) => {
            eprintln!("SKIP {}: fac: {}", family, e);
            return;
        }
    };
    // RHS = ones for a deterministic, well-scaled load.
    let rhs = vec![1.0_f64; n];

    let bare_ns = time_min(solve_iters, || {
        let _ = solve_sparse(&factors, &rhs);
    });
    let refined_ns = time_min(solve_iters, || {
        let _ = solve_sparse_refined(&csc, &factors, &rhs);
    });

    let to_us = |ns: u128| (ns as f64) / 1000.0;
    let ratio_ssids = match oracle.ssids {
        Some(s) if s > 0 => format!("{:>6.2}", to_us(refined_ns) / s as f64),
        _ => "    --".to_string(),
    };
    let ratio_mumps = match oracle.mumps {
        Some(s) if s > 0 => format!("{:>6.2}", to_us(refined_ns) / s as f64),
        _ => "    --".to_string(),
    };
    let oracle_ssids = oracle
        .ssids
        .map(|u| format!("{:>6}", u))
        .unwrap_or_else(|| "    --".to_string());
    let oracle_mumps = oracle
        .mumps
        .map(|u| format!("{:>6}", u))
        .unwrap_or_else(|| "    --".to_string());

    let refine_overhead = refined_ns.saturating_sub(bare_ns);
    let refine_ratio = if bare_ns > 0 {
        refined_ns as f64 / bare_ns as f64
    } else {
        0.0
    };

    println!(
        "{:<14} {:>6} {:>7}   {:>9.1} {:>9.1} {:>9.1} {:>9.1}   {:>5.1}x  {:>9.1}    {} {}    {} {}",
        format!("{}{}", family, sample),
        n,
        nnz,
        to_us(sym_ns),
        to_us(fac_ns),
        to_us(bare_ns),
        to_us(refined_ns),
        refine_ratio,
        to_us(refine_overhead),
        oracle_ssids,
        ratio_ssids,
        oracle_mumps,
        ratio_mumps,
    );
}

fn main() {
    println!(
        "{:<14} {:>6} {:>7}   {:>9} {:>9} {:>9} {:>9}   {:>6}  {:>9}    ssids rs/ss    mumps rs/mu",
        "matrix", "n", "nnz", "sym(us)", "fac(us)", "solve", "refined", "ref/s", "ovhd(us)",
    );
    println!("{}", "-".repeat(135));
    // Span sizes from tiny (where SSIDS reports 1-2us) up to the
    // factor outliers. One sample per family is enough for a hotspot
    // sketch; the per-family geomean in `bench.rs` already covered
    // the breadth.
    let cases: &[(&str, &str)] = &[
        ("HS118", "_0000"),
        ("ALLINITC", "_0000"),
        ("MCONCON", "_0000"),
        ("BATCH", "_0000"),
        ("HAHN1", "_0000"),
        ("AVION2", "_0000"),
        ("VESUVIO", "_0000"),
        ("CRESC132", "_0000"),
    ];
    for (family, sample) in cases {
        run_one(family, sample, "");
    }
}
