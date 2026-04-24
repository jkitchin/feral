//! Check fill-in on the long-tail matrices to see whether slow factor
//! times correlate with high fill ratios, and whether alternate
//! orderings would reduce the fill.
//!
//! Reports per-matrix:
//!   n, nnz_A, factor_nnz under each ordering (AMD default / METIS / SCOTCH),
//!   and fill ratio = factor_nnz / n.

use std::path::Path;

use feral::read_mtx;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};

fn run(path: &Path) {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<?>")
        .to_string();
    let Ok(mtx) = read_mtx(path) else {
        println!("  {:28} SKIP (read_mtx)", name);
        return;
    };
    let Ok(csc) = mtx.to_csc() else {
        println!("  {:28} SKIP (to_csc)", name);
        return;
    };
    let n = csc.n;
    let nnz = csc.values.len();

    let params = SupernodeParams::default();
    let methods = [
        ("amd", OrderingMethod::Amd),
        ("metis", OrderingMethod::MetisND),
        ("scotch", OrderingMethod::ScotchND),
    ];

    let mut fnnz: Vec<(String, usize)> = Vec::with_capacity(3);
    for (tag, m) in methods.iter() {
        match symbolic_factorize_with_method(&csc, &params, *m) {
            Ok(sym) => {
                let raw = sym.col_counts.iter().sum::<usize>();
                fnnz.push((tag.to_string(), raw));
            }
            Err(_) => fnnz.push((tag.to_string(), usize::MAX)),
        }
    }

    let tri = (n as f64) * (n as f64 + 1.0) / 2.0;
    let amd = fnnz[0].1 as f64;
    let met = fnnz[1].1 as f64;
    let sco = fnnz[2].1 as f64;
    let dens_amd = 100.0 * amd / tri;
    println!(
        "  {:28} n={:>6} nnz_A={:>7} | amd={:>9} met={:>9} sco={:>9} | dense%={:5.1} amd/n={:5.1} amd/met={:4.2}",
        name,
        n,
        nnz,
        fnnz[0].1,
        fnnz[1].1,
        fnnz[2].1,
        dens_amd,
        amd / n as f64,
        if met > 0.0 { amd / met } else { f64::NAN },
    );
    let _ = sco;
}

fn main() {
    println!("=== fill-in on long-tail matrices ===");
    println!("Columns: factor_nnz (L+D total) under each ordering; amd/n = avg fill per column;");
    println!("amd/met > 1 means AMD fills more than METIS.");
    println!();

    // Top-10 from the 2026-04-23 bench (sparse column).
    let targets = [
        "data/matrices/kkt/CRESC100/CRESC100_0000.mtx",
        "data/matrices/kkt/ACOPR30/ACOPR30_0185.mtx",
        "data/matrices/kkt/ACOPR30/ACOPR30_0079.mtx",
        "data/matrices/kkt/ACOPR30/ACOPR30_0078.mtx",
        "data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx",
        "data/matrices/kkt/ACOPR30/ACOPR30_0080.mtx",
        "data/matrices/kkt/ACOPR30/ACOPR30_0200.mtx",
        "data/matrices/kkt/ACOPR30/ACOPR30_0039.mtx",
        "data/matrices/kkt/ACOPR30/ACOPR30_0199.mtx",
        "data/matrices/kkt/HAIFAM/HAIFAM_0082.mtx",
        // Reference: tail dense-kernel leaders (not in sparse top-10 but
        // known hard matrices for sanity check).
        "data/matrices/kkt/HAHN1/HAHN1_0049.mtx",
        "data/matrices/kkt/HAHN1/HAHN1_0193.mtx",
        "data/matrices/kkt/CRESC100/CRESC100_0029.mtx",
        "data/matrices/kkt/GAUSS2/GAUSS2_0029.mtx",
        "data/matrices/kkt/GAUSS2/GAUSS2_0035.mtx",
    ];

    for t in targets {
        run(Path::new(t));
    }
}
