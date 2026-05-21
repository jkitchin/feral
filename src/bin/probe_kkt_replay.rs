//! Per-problem warm-replay probe for the Mittelmann KKT dump corpus.
//!
//! Generalises `probe_pinene_issue38_fix.rs` to any problem under
//! `data/matrices/kkt-mittelmann/<problem>/`. Loads the
//! `<problem>_NNNN.mtx` sequence in order, feeds them through ONE warm
//! `Solver::new()`, and prints per-call factor time, inertia (vs the
//! `<problem>_NNNN.json` MUMPS oracle), and the status.
//!
//! Investigation tool for the IPOPT-level MA57-vs-FERAL comparison:
//! when feral times out at the IPOPT level we want to know whether
//! the cost is in the KKT factor or in the IPM trajectory.
//!
//! Usage:
//!     cargo run --release --bin probe_kkt_replay -- <problem>
//!     FRESH=1 cargo run --release --bin probe_kkt_replay -- <problem>
//!     CB=1   cargo run --release --bin probe_kkt_replay -- <problem>
//!     PAR=0  cargo run --release --bin probe_kkt_replay -- <problem>

use std::env;
use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus, Solver};
use feral::scaling::ScalingStrategy;
use feral::symbolic::supernode::SupernodeParams;
use feral::{read_mtx, read_sidecar, Inertia};

fn build_solver() -> Solver {
    let mut np = NumericParams::default();
    match env::var("SCALING").as_deref() {
        Ok("identity") => np.scaling = ScalingStrategy::Identity,
        Ok("infnorm") => np.scaling = ScalingStrategy::InfNorm,
        Ok("mc64") => np.scaling = ScalingStrategy::Mc64Symmetric,
        _ => {}
    }
    let mut s = Solver::with_params(np, SupernodeParams::default());
    if env::var("CB").is_ok() {
        s = s.with_cascade_break(0.5).with_cascade_break_eps(1e-10);
    }
    if matches!(env::var("PAR").as_deref(), Ok("0") | Ok("off")) {
        s = s.with_parallel(false);
    }
    if env::var("SQD").is_ok() {
        s = s.with_sqd_mode(true);
    }
    if let Ok(s_) = env::var("AUTO_CB") {
        let beta: f64 = s_.parse().unwrap_or(0.05);
        s = s.with_auto_cascade_break(beta);
    }
    // Track B2: value-bounded MC64 scaling cache. On by default;
    // `MC64_CACHE=0` forces a fresh Hungarian on every factor so the
    // probe can diff cache-on vs cache-off inertia/time.
    if matches!(env::var("MC64_CACHE").as_deref(), Ok("0") | Ok("off")) {
        s = s.with_mc64_cache(false);
    }
    s
}

fn main() {
    let problem = env::args()
        .nth(1)
        .expect("usage: probe_kkt_replay <problem>");
    let dir = format!("data/matrices/kkt-mittelmann/{problem}");
    if !Path::new(&dir).exists() {
        eprintln!("SKIP: {dir} not present");
        std::process::exit(2);
    }

    let fresh = env::var("FRESH").is_ok();
    let max_iter: usize = env::var("MAX_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    if fresh {
        eprintln!("[probe] FRESH solver every iteration");
    }
    if env::var("CB").is_ok() {
        eprintln!("[probe] cascade_break ON");
    }
    if matches!(env::var("PAR").as_deref(), Ok("0") | Ok("off")) {
        eprintln!("[probe] parallel OFF");
    }
    if env::var("SQD").is_ok() {
        eprintln!("[probe] SQD mode ON (diagonal pivots only)");
    }
    if let Ok(s_) = env::var("AUTO_CB") {
        let beta: f64 = s_.parse().unwrap_or(0.05);
        eprintln!("[probe] AUTO_CB ON (β={beta})");
    }
    if matches!(env::var("MC64_CACHE").as_deref(), Ok("0") | Ok("off")) {
        eprintln!("[probe] MC64 scaling cache OFF");
    }

    println!(
        "{:>4}  {:>10}  {:>10}  {:>10}  {:>10}  status",
        "iter", "factor_s", "pos", "neg", "zero"
    );

    let mut solver = build_solver();
    let mut total = 0.0_f64;
    for i in 0..max_iter {
        let mtx_path = format!("{dir}/{problem}_{i:04}.mtx");
        let json_path = format!("{dir}/{problem}_{i:04}.json");
        if !Path::new(&mtx_path).exists() {
            break;
        }
        let Ok(mtx) = read_mtx(Path::new(&mtx_path)) else {
            eprintln!("SKIP: cannot read {mtx_path}");
            continue;
        };
        let Ok(csc) = mtx.to_csc() else {
            eprintln!("SKIP: cannot to_csc {mtx_path}");
            continue;
        };
        let expected = read_sidecar(Path::new(&json_path)).ok().map(|s| Inertia {
            positive: s.inertia.positive,
            negative: s.inertia.negative,
            zero: s.inertia.zero,
        });

        if fresh {
            solver = build_solver();
        }
        let t0 = Instant::now();
        let status = solver.factor(&csc, expected.clone());
        let elapsed = t0.elapsed().as_secs_f64();
        total += elapsed;

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
    println!("---");
    println!("total feral factor time: {total:.3} s");
    if !fresh {
        println!("mc64 scaling-cache hits: {}", solver.mc64_cache_hit_count());
    }
}
