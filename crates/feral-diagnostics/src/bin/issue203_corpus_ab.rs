//! Issue #203 — corpus A/B between two orderings, on real KKT matrices.
//!
//! Built to settle whether `Auto`'s preference for `Amf` over `Amd` is
//! right. Six single iterates suggested it is not (geomean 1.17x for
//! `Amd`), which is nowhere near the evidence bar `#67`/`#73` set for
//! the routing they decided — so this walks a whole corpus instead.
//!
//! Per matrix: factor once per arm to warm the symbolic cache, then
//! time `PAIRS` alternating pairs and take the minimum per arm, which
//! is the steady-state per-iterate cost an interior-point host pays.
//! Arms must agree on inertia and both must return `rel_res <= 1e-8`,
//! or the matrix is excluded from the statistics and reported as a gate
//! failure — a matrix where the two orderings disagree cannot carry a
//! timing comparison.
//!
//! Reports per-family geomean, win/loss counts, and an exact two-sided
//! sign test over matrices.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin issue203_corpus_ab \
//!       -- CORPUS_DIR [ARM_A] [ARM_B] [PAIRS] [MAX_PER_FAMILY]
//!
//! Arms: amd | amf | metis | auto. Default `amf` vs `amd`.

use feral::symbolic::OrderingMethod;
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn arm(name: &str) -> OrderingMethod {
    match name {
        "amd" => OrderingMethod::Amd,
        "amf" => OrderingMethod::Amf,
        "metis" => OrderingMethod::MetisND,
        _ => OrderingMethod::Auto,
    }
}

fn matvec(k: &CscMatrix, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; k.n];
    for j in 0..k.n {
        for p in k.col_ptr[j]..k.col_ptr[j + 1] {
            let i = k.row_idx[p];
            let a = k.values[p];
            y[i] += a * x[j];
            if i != j {
                y[j] += a * x[i];
            }
        }
    }
    y
}

fn rel_res(k: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let ax = matvec(k, x);
    let rn: f64 = ax.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum();
    let bn: f64 = b.iter().map(|v| v * v).sum();
    if bn == 0.0 {
        0.0
    } else {
        (rn / bn).sqrt()
    }
}

/// `(steady_us, inertia, rel_res)` for one arm, or `None` on failure.
fn run(
    m: &CscMatrix,
    b: &[f64],
    method: OrderingMethod,
    pairs: usize,
) -> Option<(u128, (usize, usize, usize), f64)> {
    let mut s = Solver::new().with_ordering(method);
    if !matches!(s.factor(m, None), FactorStatus::Success) {
        return None;
    }
    let mut best = u128::MAX;
    for _ in 0..pairs {
        let t = Instant::now();
        if !matches!(s.factor(m, None), FactorStatus::Success) {
            return None;
        }
        best = best.min(t.elapsed().as_micros());
    }
    let inr = s.inertia()?.clone();
    let x = s.solve(b).ok()?;
    Some((
        best,
        (inr.positive, inr.negative, inr.zero),
        rel_res(m, &x, b),
    ))
}

fn sign_test(wins: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let k = wins.min(n - wins);
    let mut tail = 0.0f64;
    for i in 0..=k {
        let mut c = 1.0f64;
        for j in 0..i {
            c *= (n - j) as f64 / (j + 1) as f64;
        }
        tail += c;
    }
    (2.0 * tail / 2f64.powi(n as i32)).min(1.0)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let Some(root) = a.first() else {
        eprintln!("usage: issue203_corpus_ab CORPUS_DIR [ARM_A] [ARM_B] [PAIRS] [MAX_PER_FAMILY]");
        std::process::exit(2);
    };
    let na = a.get(1).map(|s| s.as_str()).unwrap_or("amf");
    let nb = a.get(2).map(|s| s.as_str()).unwrap_or("amd");
    let pairs: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let cap: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(12);

    // <root>/<family>/*.mtx, capped per family and evenly spaced so the
    // sample is not all early (easy) iterates.
    let mut families: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(Path::new(root)) else {
        eprintln!("cannot read {root}");
        std::process::exit(1);
    };
    for e in rd.filter_map(|e| e.ok()) {
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        let fam = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let mut ms: Vec<PathBuf> = std::fs::read_dir(&p)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mtx"))
            .collect();
        ms.sort();
        if ms.len() > cap {
            let step = ms.len() as f64 / cap as f64;
            ms = (0..cap)
                .map(|i| ms[(i as f64 * step) as usize].clone())
                .collect();
        }
        if !ms.is_empty() {
            families.insert(fam, ms);
        }
    }

    println!("corpus {root}   arms: {na} (A) vs {nb} (B)   pairs={pairs} cap={cap}/family");
    println!("ratio = A_us / B_us;  > 1 means B ({nb}) is faster\n");
    println!(
        "{:<12} {:>4} {:>7} {:>10} {:>10} {:>8} {:>8}",
        "family", "n", "mats", "geomean", "worst", "B wins", "p"
    );

    let mut all: Vec<f64> = Vec::new();
    let mut all_wins = 0usize;
    let mut gates: Vec<String> = Vec::new();

    for (fam, mats) in &families {
        let mut ratios: Vec<f64> = Vec::new();
        let mut wins = 0usize;
        let mut dim = 0usize;
        for path in mats {
            let Ok(mtx) = read_mtx(path) else { continue };
            let Ok(m) = mtx.to_csc() else { continue };
            dim = m.n;
            let x_true: Vec<f64> = (0..m.n).map(|i| 1.0 + i as f64 / m.n as f64).collect();
            let b = matvec(&m, &x_true);
            let (ra, rb) = (run(&m, &b, arm(na), pairs), run(&m, &b, arm(nb), pairs));
            let (Some(ra), Some(rb)) = (ra, rb) else {
                gates.push(format!("{}: an arm failed", path.display()));
                continue;
            };
            if ra.1 != rb.1 {
                gates.push(format!(
                    "{}: inertia {:?} vs {:?}",
                    path.display(),
                    ra.1,
                    rb.1
                ));
                continue;
            }
            if ra.2 > 1e-8 || rb.2 > 1e-8 {
                gates.push(format!(
                    "{}: rel_res {:.2e} / {:.2e}",
                    path.display(),
                    ra.2,
                    rb.2
                ));
                continue;
            }
            let r = ra.0 as f64 / rb.0.max(1) as f64;
            ratios.push(r);
            if r > 1.0 {
                wins += 1;
            }
        }
        if ratios.is_empty() {
            continue;
        }
        let gm = (ratios.iter().map(|v| v.ln()).sum::<f64>() / ratios.len() as f64).exp();
        let worst = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "{:<12} {:>4} {:>7} {:>10.3} {:>10.3} {:>8} {:>8.4}",
            fam,
            dim,
            ratios.len(),
            gm,
            worst,
            format!("{}/{}", wins, ratios.len()),
            sign_test(wins, ratios.len())
        );
        all_wins += wins;
        all.extend(ratios);
    }

    if !all.is_empty() {
        let gm = (all.iter().map(|v| v.ln()).sum::<f64>() / all.len() as f64).exp();
        let worst = all.iter().cloned().fold(f64::INFINITY, f64::min);
        let best = all.iter().cloned().fold(0.0f64, f64::max);
        println!(
            "\n{:<12} {:>4} {:>7} {:>10.3} {:>10.3} {:>8} {:>8.4}",
            "ALL",
            "",
            all.len(),
            gm,
            worst,
            format!("{}/{}", all_wins, all.len()),
            sign_test(all_wins, all.len())
        );
        println!("best case for {nb}: {best:.3}x");
    }
    if !gates.is_empty() {
        println!(
            "\ngate failures ({}), excluded from the statistics:",
            gates.len()
        );
        for g in gates.iter().take(12) {
            println!("  {g}");
        }
    }
}
