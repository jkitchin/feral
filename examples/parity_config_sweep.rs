//! Run the committed parity panel under 4 configurations and count
//! passes. Used to decide whether Phase 2.2.1 (MC64) and Phase 2.2.2
//! (pivot_threshold=0.01) should remain defaults post-Phase-2.2.3.
//!
//! Run with:  cargo run --release --example parity_config_sweep

use std::path::{Path, PathBuf};

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse_refined;
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};

const K_RESIDUAL: f64 = 10.0;
const ABS_FLOOR: f64 = 1.0e-14;

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

fn read_oracle(path: &Path) -> Option<(Inertia, f64)> {
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let inertia = data.get("inertia")?;
    let pos = inertia.get("positive")?.as_u64()? as usize;
    let neg = inertia.get("negative")?.as_u64()? as usize;
    let zero = inertia.get("zero")?.as_u64()? as usize;
    let residual = data.get("residual_2norm_relative")?.as_f64()?;
    Some((Inertia::new(pos, neg, zero), residual))
}

struct PanelEntry {
    name: String,
    family: String,
    mtx: PathBuf,
    sidecar: PathBuf,
    mumps_json: PathBuf,
}

fn load_panel() -> Vec<PanelEntry> {
    let mut out = Vec::new();
    let root = Path::new("tests/data/parity");
    let Ok(subdirs) = std::fs::read_dir(root) else {
        return out;
    };
    for sub in subdirs.filter_map(|e| e.ok()) {
        let sp = sub.path();
        if !sp.is_dir() {
            continue;
        }
        let fam = sp.file_name().unwrap().to_string_lossy().into_owned();
        let Ok(files) = std::fs::read_dir(&sp) else {
            continue;
        };
        for f in files.filter_map(|e| e.ok()) {
            let fp = f.path();
            if fp.extension().is_some_and(|e| e == "mtx") {
                let stem = fp.file_stem().unwrap().to_string_lossy().into_owned();
                let sidecar = fp.with_extension("json");
                let mumps_json = fp.with_extension("mumps.json");
                if sidecar.exists() && mumps_json.exists() {
                    out.push(PanelEntry {
                        name: stem,
                        family: fam.clone(),
                        mtx: fp.clone(),
                        sidecar,
                        mumps_json,
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn run_matrix(
    entry: &PanelEntry,
    strategy: ScalingStrategy,
    threshold: f64,
) -> Option<(Inertia, Inertia, f64, f64, bool)> {
    let mtx = read_mtx(&entry.mtx).ok()?;
    let csc = mtx.to_csc().ok()?;
    let sc = read_sidecar(&entry.sidecar).ok()?;
    let rhs = sc.finite_rhs()?;
    let (mumps_inertia, mumps_residual) = read_oracle(&entry.mumps_json)?;

    let snp = SupernodeParams {
        scaling_strategy: strategy,
        ..Default::default()
    };
    let sym = symbolic_factorize(&csc, &snp).ok()?;
    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: threshold,
        ..BunchKaufmanParams::default()
    };
    let (fac, feral_inertia) = factorize_multifrontal(&csc, &sym, &params).ok()?;
    let x = solve_sparse_refined(&csc, &fac, &rhs).ok()?;
    let feral_res = rel_residual(&csc, &x, &rhs);

    let inertia_ok = feral_inertia == mumps_inertia;
    let target = (K_RESIDUAL * mumps_residual).max(ABS_FLOOR);
    let res_ok = feral_res <= target;
    Some((
        feral_inertia,
        mumps_inertia,
        feral_res,
        mumps_residual,
        inertia_ok && res_ok,
    ))
}

fn main() {
    let panel = load_panel();
    println!("Loaded {} parity matrices\n", panel.len());

    let configs: &[(&str, ScalingStrategy, f64)] = &[
        ("Id / 0.0 ", ScalingStrategy::Identity, 0.0),
        ("Id / 0.01", ScalingStrategy::Identity, 0.01),
        ("Mc64 / 0.0 ", ScalingStrategy::Mc64Symmetric, 0.0),
        (
            "Mc64 / 0.01 (current default)",
            ScalingStrategy::Mc64Symmetric,
            0.01,
        ),
    ];

    print!("{:<32}", "Matrix");
    for (label, _, _) in configs {
        print!(" {:>32}", label);
    }
    println!();

    let mut pass_count = [0usize; 4];
    let mut regressions: Vec<Vec<String>> = vec![Vec::new(); 4];

    for entry in &panel {
        print!("{:<32}", entry.name);
        let mut row_status = [false; 4];
        for (ci, (_, strat, thr)) in configs.iter().enumerate() {
            let result = run_matrix(entry, strat.clone(), *thr);
            let status = match &result {
                Some((_, _, _, _, true)) => "PASS".to_string(),
                Some((fi, mi, fr, mr, false)) => {
                    if fi != mi {
                        format!("I:{}v{}", fi.zero, mi.zero)
                    } else {
                        format!("R:{:.0e}/{:.0e}", fr, mr)
                    }
                }
                None => "ERR".into(),
            };
            row_status[ci] = matches!(result, Some((_, _, _, _, true)));
            if row_status[ci] {
                pass_count[ci] += 1;
            }
            print!(" {:>32}", status);
        }
        println!();

        // Track regressions relative to config 3 (the current default).
        for ci in 0..4 {
            if row_status[3] && !row_status[ci] {
                regressions[ci].push(entry.name.clone());
            }
        }
    }

    println!("\nPass counts:");
    for (ci, (label, _, _)) in configs.iter().enumerate() {
        println!("  {:<32} {:>3} / {}", label, pass_count[ci], panel.len());
    }

    println!("\nRegressions vs Mc64/0.01 (current default):");
    for (ci, (label, _, _)) in configs.iter().enumerate() {
        if ci == 3 {
            continue;
        }
        print!("  {}: ", label);
        if regressions[ci].is_empty() {
            println!("none");
        } else {
            println!("{:?}", regressions[ci]);
        }
    }
}
