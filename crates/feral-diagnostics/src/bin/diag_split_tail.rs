//! Split the factor time on long-tail matrices into its parts to see
//! where feral's 10x vs MUMPS goes: is it symbolic? numeric? or both?
//!
//! For each matrix, report medians over N reps of:
//!   sym_us  — symbolic_factorize (ordering + etree + col_counts + supernodes)
//!   num_us  — factorize_multifrontal numeric phase only
//!   tot_us  — sum, comparable to MUMPS factor_us
//!   mumps_us — the canonical MUMPS timing from the *.mumps.json sidecar

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams};

fn median(v: &mut [u128]) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn read_mumps_us(mtx: &Path) -> Option<u64> {
    let sidecar = mtx.with_extension("mumps.json");
    let text = std::fs::read_to_string(&sidecar).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("factor_us")?.as_u64()
}

fn run(path: &Path, reps: usize) {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<?>")
        .to_string();
    let Ok(mtx) = read_mtx(path) else {
        println!("  {:26} SKIP", name);
        return;
    };
    let Ok(csc) = mtx.to_csc() else {
        println!("  {:26} SKIP", name);
        return;
    };

    let snode = SupernodeParams::default();
    let bk = BunchKaufmanParams::default();
    let np = NumericParams::with_bk(bk);

    let mut sym_us = Vec::with_capacity(reps);
    let mut num_us = Vec::with_capacity(reps);

    for _ in 0..reps {
        let t = Instant::now();
        let sym = match symbolic_factorize(&csc, &snode) {
            Ok(s) => s,
            Err(_) => return,
        };
        sym_us.push(t.elapsed().as_micros());

        let t = Instant::now();
        let _ = match factorize_multifrontal(&csc, &sym, &np) {
            Ok(r) => r,
            Err(_) => return,
        };
        num_us.push(t.elapsed().as_micros());
    }

    let sm = median(&mut sym_us);
    let nm = median(&mut num_us);
    let tot = sm + nm;
    let mumps = read_mumps_us(path).unwrap_or(0);
    let ratio = if mumps > 0 {
        tot as f64 / mumps as f64
    } else {
        f64::NAN
    };
    let pct_sym = 100.0 * sm as f64 / tot.max(1) as f64;

    println!(
        "  {:26} n={:>5} | sym={:>5} num={:>5} tot={:>5} μs | mumps={:>4} μs | ratio={:5.2}x | sym%={:4.1}%",
        name, csc.n, sm, nm, tot, mumps, ratio, pct_sym,
    );
}

fn main() {
    println!("=== sym vs num split on long-tail matrices ===");
    println!();

    let targets = [
        ("data/matrices/kkt/CRESC100/CRESC100_0000.mtx", 30),
        ("data/matrices/kkt/ACOPR30/ACOPR30_0185.mtx", 30),
        ("data/matrices/kkt/ACOPR30/ACOPR30_0079.mtx", 30),
        ("data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx", 30),
        ("data/matrices/kkt/HAIFAM/HAIFAM_0082.mtx", 20),
        ("data/matrices/kkt/HAHN1/HAHN1_0049.mtx", 20),
        ("data/matrices/kkt/GAUSS2/GAUSS2_0029.mtx", 20),
        // smaller reference: one from the bulk (should have very small
        // absolute times and similar proportion).
        ("data/matrices/kkt/HS118/HS118_0001.mtx", 50),
        ("data/matrices/kkt/HS92/HS92_0001.mtx", 100),
    ];
    for (p, r) in targets {
        run(Path::new(p), r);
    }
}
