//! Does the badly-scaled-RHS componentwise defect reproduce on the
//! *tracked* parity corpus? (The large fixtures are gitignored, so a CI
//! regression test has to live on these.)
//!
//!   cargo run -p feral-diagnostics --bin probe_omega_parity_corpus --release

use feral::numeric::solve::{RefineOptions, StopCriterion};
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::Path;

fn omega(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0f64; n];
    a.symv(x, &mut ax);
    let r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
    let mut d = vec![0.0f64; n];
    a.abs_symv(x, &mut d);
    let safe1 = ((n + 1) as f64) * f64::MIN_POSITIVE;
    let safe2 = safe1 / f64::EPSILON;
    let mut om = 0.0f64;
    for i in 0..n {
        let den = d[i] + b[i].abs();
        if den == 0.0 {
            continue;
        }
        let v = if den > safe2 {
            r[i].abs() / den
        } else {
            (r[i].abs() + safe1) / (den + safe1)
        };
        if v > om {
            om = v;
        }
    }
    om
}

fn hard_rhs(a: &CscMatrix) -> Vec<f64> {
    let n = a.n;
    let v: Vec<f64> = (0..n)
        .map(|i| {
            let mag = 10f64.powi(((i % 13) as i32) - 6);
            if i % 2 == 0 {
                mag
            } else {
                -mag
            }
        })
        .collect();
    let mut b = vec![0.0f64; n];
    a.symv(&v, &mut b);
    b
}

fn main() {
    let root = Path::new("tests/data/parity");
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(fams) = std::fs::read_dir(root) {
        for fam in fams.flatten() {
            if let Ok(files) = std::fs::read_dir(fam.path()) {
                for f in files.flatten() {
                    let p = f.path();
                    if p.extension().map(|e| e == "mtx").unwrap_or(false) {
                        paths.push(p);
                    }
                }
            }
        }
    }
    for p in [
        Path::new("tests/data/large/sawpath_kkt.mtx"),
        Path::new("tests/data/large/twirism1_kkt.mtx"),
    ] {
        if p.exists() {
            paths.push(p.to_path_buf());
        }
    }
    paths.sort();
    let target = f64::EPSILON.sqrt();
    println!(
        "{:28} {:>7} {:>11} {:>4} {:>11} {:>4} {:>11}",
        "matrix", "n", "eps_om", "it", "def_om", "it", "unref_om"
    );
    let mut worst: Vec<(f64, String)> = Vec::new();
    for p in &paths {
        let Ok(mtx) = read_mtx(p) else { continue };
        let Ok(csc) = mtx.to_csc() else { continue };
        let n = csc.n;
        let b = hard_rhs(&csc);
        let mut s = Solver::new();
        if !matches!(
            s.factor(&csc, None),
            FactorStatus::Success | FactorStatus::WrongInertia { .. }
        ) {
            continue;
        }
        let name = p
            .file_stem()
            .map(|x| x.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut xe = vec![0.0f64; n];
        let Ok(oe) = s.solve_refined_into(
            &csc,
            &b,
            &mut xe,
            RefineOptions::default().and_stop(StopCriterion::EpsSqrtN),
        ) else {
            continue;
        };
        let om_e = omega(&csc, &xe, &b);

        let mut xd = vec![0.0f64; n];
        let Ok(od) = s.solve_refined_into(&csc, &b, &mut xd, RefineOptions::default()) else {
            continue;
        };
        let om_d = omega(&csc, &xd, &b);

        let mut xu = vec![0.0f64; n];
        let Ok(_) =
            s.solve_refined_into(&csc, &b, &mut xu, RefineOptions::default().and_max_steps(0))
        else {
            continue;
        };
        let om_u = omega(&csc, &xu, &b);

        let flag = if om_e > target && om_d <= target {
            "  <== FIXED"
        } else if om_d > target {
            "  !! still bad"
        } else {
            ""
        };
        println!(
            "{name:28} {n:>7} {om_e:>11.3e} {:>4} {om_d:>11.3e} {:>4} {om_u:>11.3e}{flag}",
            oe.steps, od.steps
        );
        if om_e > target {
            worst.push((om_e / target, name));
        }
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "\nEpsSqrtN leaves omega > sqrt(eps) on {} of {} matrices",
        worst.len(),
        paths.len()
    );
    for (r, n) in worst.iter().take(12) {
        println!("  {n:28} omega/sqrt(eps) = {r:.3e}");
    }
}
