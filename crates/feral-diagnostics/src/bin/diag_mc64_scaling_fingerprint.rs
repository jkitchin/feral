//! diag_mc64_scaling_fingerprint — bit-exact fingerprint of the MC64
//! scaling vector, for use as a same-output oracle across refactors of
//! the Hungarian matching.
//!
//! The MC64 matching is only observable through the scaling vector it
//! produces (`SparseFactors::scaling`). Any change to the matching --
//! including a different choice among equal-cost augmenting paths --
//! moves at least one entry of that vector. Hashing the raw IEEE bits
//! of every entry therefore detects *any* deviation, not just one
//! large enough to move a residual.
//!
//! Usage: diag_mc64_scaling_fingerprint <a.mtx> <b.mtx> ...

use feral::read_mtx;
use feral::Solver;
use std::path::Path;

/// FNV-1a over the raw bits, so that -0.0 vs 0.0 and NaN payloads are
/// distinguishable. A checksum, not a hash for security.
fn fingerprint(v: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: diag_mc64_scaling_fingerprint <a.mtx> ...");
        std::process::exit(2);
    }
    println!(
        "{:<28}{:>10}{:>22}{:>14}",
        "matrix", "n", "scaling_fnv1a", "status"
    );
    for a in &args {
        let csc = match read_mtx(Path::new(a)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {a}: {e:?}");
                continue;
            }
        };
        let name = Path::new(a)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| a.clone());
        let mut solver = Solver::new();
        let status = solver.factor(&csc, None);
        let (fp, n) = match solver.factors() {
            Some(f) => (fingerprint(&f.scaling), f.scaling.len()),
            None => (0, 0),
        };
        println!("{name:<28}{n:>10}{fp:>22x}{:>14}", format!("{status:?}"));
    }
    Ok(())
}
