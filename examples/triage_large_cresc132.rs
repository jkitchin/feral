//! Phase 2.1.2 — sanity check: does feral's sparse path run on n > 500?
//!
//! CRESC132_0000 is a 5314×5314 KKT matrix from the CUTEst CRESC132
//! problem. Phase 1's bench filter silently skipped it. This triage
//! loads it and runs feral's sparse multifrontal pipeline end-to-end,
//! comparing against the existing MUMPS and SSIDS sidecars.
//!
//! Outcomes:
//!   - factors + solves + matches consensus  → sparse path scales, proceed
//!   - factors but wrong inertia              → correctness bug at scale
//!   - times out / crashes                    → latent scaling issue
//!
//! Run with: `cargo run --release --example triage_large_cresc132`

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse_refined;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        let r = ax[i] - b[i];
        rs += r * r;
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

fn read_oracle(path: &Path) -> Option<(Inertia, f64)> {
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let inertia = data.get("inertia")?;
    let pos = inertia.get("positive")?.as_u64()? as usize;
    let neg = inertia.get("negative")?.as_u64()? as usize;
    let zero = inertia.get("zero")?.as_u64()? as usize;
    let residual = data.get("residual_2norm_relative")?.as_f64()?;
    Some((
        Inertia {
            positive: pos,
            negative: neg,
            zero,
        },
        residual,
    ))
}

fn main() {
    // Phase 2.1.2 sanity check — feral on n > 500 matrices.
    // Pass a matrix stem as the first arg, or default to a panel of large
    // representatives from the existing corpus.
    let default_stems = [
        "CHWIRUT1/CHWIRUT1_0000", // n=645,  smallest large case
        "HAHN1/HAHN1_0000",       // n=715
        "GAUSS2/GAUSS2_0000",     // n=758
        "CRESC100/CRESC100_0000", // n=806
        "MUONSINE/MUONSINE_0000", // n=1537
        "VESUVIO/VESUVIO_0000",   // n=3083
        "CRESC132/CRESC132_0000", // n=5314, largest
    ];
    let args: Vec<String> = std::env::args().collect();
    let stems: Vec<String> = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        default_stems.iter().map(|s| s.to_string()).collect()
    };

    for stem in &stems {
        triage_one(stem);
        println!();
    }
}

fn triage_one(stem: &str) {
    let base = format!("data/matrices/kkt/{}", stem);
    let mtx_path = format!("{}.mtx", base);
    let json_path = format!("{}.json", base);
    let mumps_path = format!("{}.mumps.json", base);
    let ssids_path = format!("{}.ssids.json", base);

    let mtx_path = Path::new(&mtx_path);
    let json_path = Path::new(&json_path);
    let mumps_path = Path::new(&mumps_path);
    let ssids_path = Path::new(&ssids_path);

    if !mtx_path.exists() {
        println!("=== {} ===", stem);
        println!("  SKIP: {} not found", mtx_path.display());
        return;
    }
    println!("=== {} ===", stem);

    let t_read = Instant::now();
    let mtx = read_mtx(mtx_path).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    println!(
        "read mtx: n={} nnz={} in {:.1} ms",
        csc.n,
        csc.values.len(),
        t_read.elapsed().as_secs_f64() * 1000.0
    );

    let sc = read_sidecar(json_path).expect("read sidecar");
    let rhs = sc.finite_rhs().expect("finite rhs");
    let expected_rmumps = Inertia {
        positive: sc.inertia.positive,
        negative: sc.inertia.negative,
        zero: sc.inertia.zero,
    };

    let oracle_mumps = read_oracle(mumps_path);
    let oracle_ssids = read_oracle(ssids_path);

    println!();
    println!("Oracles:");
    println!("  rmumps (sidecar) inertia = {}", expected_rmumps);
    if let Some((i, r)) = &oracle_mumps {
        println!("  canonical MUMPS inertia  = {}, residual = {:.3e}", i, r);
    } else {
        println!("  canonical MUMPS          = NOT AVAILABLE");
    }
    if let Some((i, r)) = &oracle_ssids {
        println!("  canonical SSIDS inertia  = {}, residual = {:.3e}", i, r);
    } else {
        println!("  canonical SSIDS          = NOT AVAILABLE");
    }

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };

    println!();
    println!("--- symbolic_factorize ---");
    let t_sym = Instant::now();
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("symbolic");
    let sym_ms = t_sym.elapsed().as_secs_f64() * 1000.0;
    println!(
        "symbolic: {:.1} ms, {} supernodes, factor_nnz_estimate = {}",
        sym_ms,
        sym.supernodes.len(),
        sym.factor_nnz_estimate
    );

    println!();
    println!("--- factorize_multifrontal ---");
    let t_fac = Instant::now();
    let (factors, inertia) = factorize_multifrontal(&csc, &sym, &params).expect("sparse factor");
    let fac_ms = t_fac.elapsed().as_secs_f64() * 1000.0;
    println!("factor: {:.1} ms", fac_ms);
    println!("feral sparse inertia = {}", inertia);
    if let Some((i, _)) = &oracle_mumps {
        println!(
            "  vs MUMPS:  {}",
            if &inertia == i { "MATCH" } else { "MISMATCH" }
        );
    }
    if let Some((i, _)) = &oracle_ssids {
        println!(
            "  vs SSIDS:  {}",
            if &inertia == i { "MATCH" } else { "MISMATCH" }
        );
    }

    println!();
    println!("--- solve_sparse_refined ---");
    let t_sol = Instant::now();
    let x = solve_sparse_refined(&csc, &factors, &rhs).expect("solve");
    let sol_ms = t_sol.elapsed().as_secs_f64() * 1000.0;
    let res = rel_residual(&csc, &x, &rhs);
    println!("solve:  {:.1} ms", sol_ms);
    println!("feral residual: {:.3e}", res);
    if let Some((_, mr)) = &oracle_mumps {
        println!(
            "  vs MUMPS residual: {:.3e}  (ratio feral/MUMPS = {:.1}×)",
            mr,
            res / mr
        );
    }
    if let Some((_, sr)) = &oracle_ssids {
        println!(
            "  vs SSIDS residual: {:.3e}  (ratio feral/SSIDS = {:.1}×)",
            sr,
            res / sr
        );
    }

    println!(
        "  feral: sym {:.1}ms, fac {:.1}ms, sol {:.1}ms, res {:.3e}",
        sym_ms, fac_ms, sol_ms, res
    );
}
