//! Issue #8 regression bench.
//!
//! Targets the two matrix families that motivated issue #8:
//!   - `pinene_3200_*`  (n≈128k, nnz≈733k)
//!   - `marine_1600_*`  (n≈77k,  nnz≈414k)
//!
//! Under the legacy defaults (cascade_break_ratio = None) several
//! iterates in each family triggered a delayed-pivot cascade and
//! factored in 60–105 s. With the auto-armed bounded cascade-break
//! (`cascade_break_ratio = Some(0.5)`, `cascade_break_eps = Some(1e-10)`,
//! landed in commits 7998386 / b998e36 / 672ab7a / c92cafe) the same
//! iterates factor in well under one second with exact inertia.
//!
//! This bench loads every available `<family>_NNNN.mtx` under
//! `data/matrices/kkt-mittelmann/<family>/`, runs the production
//! solver path, and reports factor wall, inertia (vs the sidecar
//! oracle), residual after refined solve, and total delays-in.
//! Per-iterate guard: factor < 5 s. Aggregate guard: total < 30 s.
//! Both are an order of magnitude above the post-fix numbers, so
//! a future regression that re-introduces cascade behaviour will
//! trip the guard without flagging on normal noise.
//!
//! Usage:
//!     cargo run --release --bin bench_issue8
//!     cargo run --release --bin bench_issue8 -- --no-guard   # report only
//!
//! Env knobs (optional):
//!     FERAL_ISSUE8_PER_GUARD_S    per-iterate guard in seconds (default 5.0)
//!     FERAL_ISSUE8_TOTAL_GUARD_S  aggregate guard in seconds   (default 30.0)

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{
    factorize_multifrontal_parallel_with_workspace, FactorWorkspace, NumericParams,
};
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, CscMatrix, Inertia};

const FAMILIES: &[&str] = &["pinene_3200", "marine_1600"];

fn rel_residual_2norm(csc: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = csc.n;
    let mut r: Vec<f64> = b.iter().map(|v| -v).collect();
    for j in 0..n {
        for p in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[p];
            let a = csc.values[p];
            r[i] += a * x[j];
            if i != j {
                r[j] += a * x[i];
            }
        }
    }
    let rn: f64 = r.iter().map(|v| v * v).sum();
    let bn: f64 = b.iter().map(|v| v * v).sum();
    if bn == 0.0 {
        0.0
    } else {
        (rn / bn).sqrt()
    }
}

struct IterateResult {
    tag: String,
    factor_s: f64,
    solve_s: f64,
    delay_in: usize,
    inertia_ok: bool,
    rel_res: Option<f64>,
}

fn run_one(tag: &str, family: &str) -> Option<IterateResult> {
    let base = format!("data/matrices/kkt-mittelmann/{family}/{tag}");
    let mtx_path = format!("{base}.mtx");
    let json_path = format!("{base}.json");

    let mtx = read_mtx(Path::new(&mtx_path)).ok()?;
    let csc = mtx.to_csc().ok()?;
    let sidecar = read_sidecar(Path::new(&json_path)).ok()?;
    let rhs = sidecar.finite_rhs()?;
    let oracle = Inertia::new(
        sidecar.inertia.positive,
        sidecar.inertia.negative,
        sidecar.inertia.zero,
    );

    let snode = SupernodeParams::default();
    let sym = symbolic_factorize(&csc, &snode).ok()?;
    let params = NumericParams::default();
    let mut ws = FactorWorkspace::new();

    let t0 = Instant::now();
    let (factors, inertia) =
        factorize_multifrontal_parallel_with_workspace(&csc, &sym, &params, &mut ws).ok()?;
    let factor_s = t0.elapsed().as_secs_f64();

    let delay_in: usize = factors.node_factors.iter().map(|nf| nf.n_delayed_in).sum();
    let inertia_ok = inertia == oracle;

    let t0 = Instant::now();
    let solve = solve_sparse_refined(&csc, &factors, &rhs);
    let solve_s = t0.elapsed().as_secs_f64();
    let rel_res = solve.ok().map(|x| rel_residual_2norm(&csc, &x, &rhs));

    Some(IterateResult {
        tag: tag.to_string(),
        factor_s,
        solve_s,
        delay_in,
        inertia_ok,
        rel_res,
    })
}

fn discover_iterates(family: &str) -> Vec<String> {
    let dir = format!("data/matrices/kkt-mittelmann/{family}");
    let mut tags: Vec<String> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) != Some("mtx") {
                    return None;
                }
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    tags.sort();
    tags
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let enforce_guard = !args.iter().any(|a| a == "--no-guard");

    let per_guard_s = env_f64("FERAL_ISSUE8_PER_GUARD_S", 5.0);
    let total_guard_s = env_f64("FERAL_ISSUE8_TOTAL_GUARD_S", 30.0);

    println!("Issue #8 regression bench");
    println!("  families:           {:?}", FAMILIES);
    println!("  per-iterate guard:  {:.1} s", per_guard_s);
    println!("  aggregate guard:    {:.1} s", total_guard_s);
    println!(
        "  guard enforcement:  {}",
        if enforce_guard {
            "on (nonzero exit on breach)"
        } else {
            "off (--no-guard)"
        }
    );

    let mut all: Vec<IterateResult> = Vec::new();
    let mut missing_data = true;

    for family in FAMILIES {
        let tags = discover_iterates(family);
        if tags.is_empty() {
            println!("\n[{family}]  no matrices found under data/matrices/kkt-mittelmann/{family}");
            continue;
        }
        missing_data = false;
        println!("\n[{family}]  {} iterates", tags.len());
        println!(
            "  {:<24} {:>10} {:>10} {:>9}  {:<11}  {:>10}",
            "iterate", "factor(s)", "solve(s)", "delay_in", "inertia", "rel_res"
        );
        for tag in &tags {
            match run_one(tag, family) {
                Some(r) => {
                    println!(
                        "  {:<24} {:>10.3} {:>10.3} {:>9}  {:<11}  {}",
                        r.tag,
                        r.factor_s,
                        r.solve_s,
                        r.delay_in,
                        if r.inertia_ok { "ok" } else { "MISMATCH" },
                        r.rel_res
                            .map(|v| format!("{:>10.3e}", v))
                            .unwrap_or_else(|| "       n/a".to_string()),
                    );
                    all.push(r);
                }
                None => {
                    println!("  {:<24}  load/factor/solve failed", tag);
                }
            }
        }
    }

    if missing_data {
        println!(
            "\nNo matrices found under data/matrices/kkt-mittelmann/. Run\n\
             scripts/harvest-mittelmann-kkt.sh to populate the corpus, or\n\
             skip this bench in environments without the dataset."
        );
        return;
    }

    let total_factor: f64 = all.iter().map(|r| r.factor_s).sum();
    let max_factor = all
        .iter()
        .map(|r| r.factor_s)
        .fold(0.0_f64, |a, b| a.max(b));
    let n_inertia_ok = all.iter().filter(|r| r.inertia_ok).count();
    let worst_res = all
        .iter()
        .filter_map(|r| r.rel_res)
        .fold(0.0_f64, |a, b| a.max(b));

    println!("\n=== Aggregate ===");
    println!("  iterates run:           {}", all.len());
    println!("  inertia exact:          {} / {}", n_inertia_ok, all.len());
    println!("  total factor wall:      {:>7.3} s", total_factor);
    println!("  max single-iter factor: {:>7.3} s", max_factor);
    println!("  worst rel_res:          {:>10.3e}", worst_res);

    let mut breaches: Vec<String> = Vec::new();
    for r in &all {
        if r.factor_s > per_guard_s {
            breaches.push(format!(
                "{}: factor {:.3}s exceeds per-iterate guard {:.1}s",
                r.tag, r.factor_s, per_guard_s
            ));
        }
        if !r.inertia_ok {
            breaches.push(format!("{}: inertia MISMATCH vs sidecar oracle", r.tag));
        }
    }
    if total_factor > total_guard_s {
        breaches.push(format!(
            "aggregate factor {:.3}s exceeds total guard {:.1}s",
            total_factor, total_guard_s
        ));
    }

    if breaches.is_empty() {
        println!("\nGUARDS: PASS");
    } else {
        println!("\nGUARDS: BREACH ({} item(s))", breaches.len());
        for b in &breaches {
            println!("  - {}", b);
        }
        if enforce_guard {
            std::process::exit(1);
        }
    }
}
