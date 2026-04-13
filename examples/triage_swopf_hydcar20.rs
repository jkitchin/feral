//! Phase 2.2.3 follow-up: disentangle the SWOPF / HYDCAR20 failure.
//!
//! triage_plateau showed that these matrices behave identically
//! under nemin=32 and nemin=10000, so amalgamation is NOT the
//! cause. ~34% (SWOPF) and ~77% (HYDCAR20) of pivots get
//! ForceAccept'd as zero under the current default configuration.
//! This binary runs each matrix under the 4 cross-configurations
//! of MC64 scaling on/off and column-relative pivot threshold
//! on/off, and for each reports inertia, zero-pivot count, and
//! residual. The goal is to isolate which feature is breaking
//! these matrices: MC64, threshold, both, or neither (matrix is
//! fundamentally BK-hostile).
//!
//! Run with:  cargo run --release --example triage_swopf_hydcar20

use std::path::Path;

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse_refined;
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};

fn params(threshold: f64) -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: threshold,
        ..BunchKaufmanParams::default()
    }
}

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        rs += (ax[i] - b[i]).powi(2);
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

fn read_mumps_oracle(path: &Path) -> Option<(Inertia, f64)> {
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let inertia = data.get("inertia")?;
    let pos = inertia.get("positive")?.as_u64()? as usize;
    let neg = inertia.get("negative")?.as_u64()? as usize;
    let zero = inertia.get("zero")?.as_u64()? as usize;
    let residual = data.get("residual_2norm_relative")?.as_f64()?;
    Some((Inertia::new(pos, neg, zero), residual))
}

fn run(stem: &str) {
    let base = format!("data/matrices/kkt/{}", stem);
    let mtx_path = format!("{}.mtx", base);
    let json_path = format!("{}.json", base);
    let mumps_path = format!("{}.mumps.json", base);

    let mtx = read_mtx(Path::new(&mtx_path)).expect("mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let sc = read_sidecar(Path::new(&json_path)).expect("sidecar");
    let rhs = sc.finite_rhs().expect("rhs");
    let mumps = read_mumps_oracle(Path::new(&mumps_path));

    println!("\n=== {} ===", stem);
    println!("  n={}  nnz={}", csc.n, csc.values.len());
    if let Some((i, r)) = &mumps {
        println!("  MUMPS oracle: inertia={}  residual={:.3e}", i, r);
    }

    let configs = [
        ("Identity / threshold=0.0  ", ScalingStrategy::Identity, 0.0),
        (
            "Identity / threshold=0.01 ",
            ScalingStrategy::Identity,
            0.01,
        ),
        (
            "Mc64 / threshold=0.0      ",
            ScalingStrategy::Mc64Symmetric,
            0.0,
        ),
        (
            "Mc64 / threshold=0.01     ",
            ScalingStrategy::Mc64Symmetric,
            0.01,
        ),
    ];

    for (label, strategy, threshold) in configs {
        let snp = SupernodeParams {
            scaling_strategy: strategy,
            ..Default::default()
        };
        let sym = symbolic_factorize(&csc, &snp).expect("symbolic");
        let (fac, inertia) =
            factorize_multifrontal(&csc, &sym, &params(threshold)).expect("factor");
        let x = solve_sparse_refined(&csc, &fac, &rhs).expect("solve");
        let res = rel_residual(&csc, &x, &rhs);
        let zero_frac = inertia.zero as f64 / csc.n as f64;
        let inertia_ok = mumps.as_ref().map(|(i, _)| &inertia == i).unwrap_or(false);
        println!(
            "  {}  inertia={}  zeros={}/{} ({:.0}%)  residual={:.3e}  match={}",
            label,
            inertia,
            inertia.zero,
            csc.n,
            100.0 * zero_frac,
            res,
            if inertia_ok { "YES" } else { "no" },
        );
    }
}

fn main() {
    // Phase 2.2.2 pivot_threshold has nowhere to delay rejected pivots
    // to (Phase 2.3 delayed pivoting doesn't exist yet), so every
    // rejected pivot hits ForceAccept and becomes a structural zero.
    // Sweep the 4-matrix Phase 2.2.3 regression panel plus SWOPF /
    // HYDCAR20 across {Identity, Mc64} x {0.0, 0.01} and check
    // whether threshold=0.0 regresses the matrices we care about.
    run("SWOPF/SWOPF_0000");
    run("HYDCAR20/HYDCAR20_0000");
    run("ACOPP30/ACOPP30_0000");
    run("CHWIRUT1/CHWIRUT1_0000");
    run("CRESC100/CRESC100_0000");
    run("CRESC132/CRESC132_0000");
    // A few more from the parity frontier to widen the picture.
    run("HATFLDG/HATFLDG_0005");
    run("METHANL8/METHANL8_0000");
    run("HATFLDBNE/HATFLDBNE_1586");
}
