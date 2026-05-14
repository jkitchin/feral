//! Per-family max-ncol probe: prints the maximum supernode `ncol`
//! across all supernodes of representative iterates for each family.
//! Used to calibrate the symbolic-arm threshold N_HEAVY for the
//! cascade-break gate (issue #15 disposition).

use std::path::Path;

use feral::read_mtx;
use feral::symbolic::{symbolic_factorize, SupernodeParams};

fn probe(label: &str, mtx_path: &str) {
    let path = Path::new(mtx_path);
    let mtx = match read_mtx(path) {
        Ok(m) => m,
        Err(e) => {
            println!("[{label}] read_mtx failed: {e:?}");
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            println!("[{label}] to_csc failed: {e:?}");
            return;
        }
    };
    let sym = match symbolic_factorize(&csc, &SupernodeParams::default()) {
        Ok(s) => s,
        Err(e) => {
            println!("[{label}] symbolic failed: {e:?}");
            return;
        }
    };
    let mut sizes: Vec<usize> = sym.supernodes.iter().map(|s| s.ncol).collect();
    sizes.sort_unstable();
    let n = sizes.len();
    let max = *sizes.last().unwrap_or(&0);
    let p99 = sizes
        .get(((n as f64 * 0.99) as usize).min(n.saturating_sub(1)))
        .copied()
        .unwrap_or(0);
    let p95 = sizes
        .get(((n as f64 * 0.95) as usize).min(n.saturating_sub(1)))
        .copied()
        .unwrap_or(0);
    let p50 = sizes.get(n / 2).copied().unwrap_or(0);
    println!(
        "[{label:<30}] n={:>6} n_snodes={:>6} max_ncol={:>6} p99={:>5} p95={:>5} p50={:>4}",
        csc.n, n, max, p99, p95, p50
    );
}

fn main() {
    for i in [0, 5, 15, 29] {
        probe(
            &format!("qcqp1000-1nc_{:04}", i),
            &format!(
                "data/matrices/kkt-mittelmann/qcqp1000-1nc/qcqp1000-1nc_{:04}.mtx",
                i
            ),
        );
    }
    for i in [0, 5, 9, 17] {
        probe(
            &format!("marine_1600_{:04}", i),
            &format!(
                "data/matrices/kkt-mittelmann/marine_1600/marine_1600_{:04}.mtx",
                i
            ),
        );
    }
    for i in [0, 5, 9] {
        probe(
            &format!("pinene_3200_{:04}", i),
            &format!(
                "data/matrices/kkt-mittelmann/pinene_3200/pinene_3200_{:04}.mtx",
                i
            ),
        );
    }
    for i in [0, 100, 329] {
        probe(
            &format!("MSS1_{:04}", i),
            &format!("data/matrices/kkt/MSS1/MSS1_{:04}.mtx", i),
        );
    }
}
