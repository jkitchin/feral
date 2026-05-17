//! One-off probe: does the issue #38 MC64-cache-invalidation fix change
//! pinene_3200 behaviour? Session-30 hypothesised that pinene_3200 (the
//! #37 reproducer) might also hit the cache-staleness path. Run the 10
//! pinene_3200_NNNN.mtx dumps through ONE warm `Solver::new()` (default
//! config, CB=off, scaling=Auto) under the fix and print per-call factor
//! time + inertia vs the .json MUMPS oracle.
//!
//! Usage:
//!     cargo run --release --bin probe_pinene_issue38_fix
//!
//! Not a regression test (corpus is gitignored). Delete after the
//! probe data lands in `dev/research/mc64-cache-staleness-2026-05-16.md`.

use std::path::Path;
use std::time::Instant;

use feral::numeric::solver::{FactorStatus, Solver};
use feral::{read_mtx, read_sidecar, Inertia};

fn main() {
    let dir = "data/matrices/kkt-mittelmann/pinene_3200";
    if !Path::new(dir).exists() {
        eprintln!("SKIP: {dir} not present");
        std::process::exit(2);
    }

    let mut solver = Solver::new();

    println!(
        "{:>4}  {:>10}  {:>10}  {:>10}  {:>10}  status",
        "iter", "factor_s", "pos", "neg", "zero"
    );

    for i in 0..10 {
        let mtx_path = format!("{dir}/pinene_3200_{i:04}.mtx");
        let json_path = format!("{dir}/pinene_3200_{i:04}.json");
        let Ok(mtx) = read_mtx(Path::new(&mtx_path)) else {
            eprintln!("SKIP: cannot read {mtx_path}");
            continue;
        };
        let Ok(csc) = mtx.to_csc() else {
            eprintln!("SKIP: cannot to_csc {mtx_path}");
            continue;
        };
        let Ok(sidecar) = read_sidecar(Path::new(&json_path)) else {
            eprintln!("SKIP: cannot read {json_path}");
            continue;
        };

        let expected = Inertia {
            positive: sidecar.inertia.positive,
            negative: sidecar.inertia.negative,
            zero: sidecar.inertia.zero,
        };

        let t0 = Instant::now();
        let status = solver.factor(&csc, Some(expected.clone()));
        let elapsed = t0.elapsed().as_secs_f64();

        let inertia = solver.inertia().cloned();
        let (p, n, z) = match &inertia {
            Some(i) => (i.positive as i64, i.negative as i64, i.zero as i64),
            None => (-1, -1, -1),
        };
        let label = match status {
            FactorStatus::Success => "OK".to_string(),
            FactorStatus::WrongInertia { actual, expected } => format!(
                "WrongInertia(got {}/{}/{}, want {}/{}/{})",
                actual.positive,
                actual.negative,
                actual.zero,
                expected.positive,
                expected.negative,
                expected.zero
            ),
            FactorStatus::Singular => "Singular".to_string(),
            FactorStatus::FatalError(e) => format!("Fatal: {e:?}"),
        };

        println!("{i:>4}  {elapsed:>10.3}  {p:>10}  {n:>10}  {z:>10}  {label}");
    }
}
