//! Bitwise digest of a factorization, for A/B-ing a change that claims
//! to be bit-exact.
//!
//! The in-tree parity tests (`parallel_parity_*`, `cb_parity_*`) compare
//! two paths *within one build*, so a change that shifts both identically
//! passes them. This prints a digest of the factor's raw bits so the same
//! matrix can be compared across two builds.
use feral::{read_mtx, NumericParams, Solver};

/// FNV-1a over the raw bit patterns, so `-0.0` and `+0.0` differ and
/// any NaN payload change is visible.
fn digest(vals: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in vals {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for p in std::env::args().skip(1) {
        // Keep going on a matrix that will not load: this tool exists to
        // compare a whole corpus across two builds, and aborting the loop
        // on the first bad file silently shrinks the comparison.
        let csc = match read_mtx(std::path::Path::new(&p)).and_then(|m| m.to_csc()) {
            Ok(c) => c,
            Err(e) => {
                println!("{p} LOAD-ERROR {e:?}");
                continue;
            }
        };
        let label = std::path::Path::new(&p)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let mut s =
            Solver::with_params(NumericParams::default(), Default::default()).with_parallel(false);
        let status = s.factor(&csc, None);
        match s.factors() {
            Some(f) => {
                // `ldlt_export` reassembles the supernodal factors into a
                // single global L (CSC) and D in factorization order, so
                // the digest is over a canonical layout rather than over
                // whatever per-front padding the driver happened to use.
                let e = f.ldlt_export();
                println!(
                    "{label} status={status:?} L={:016x} Ddiag={:016x} Dsub={:016x} nnz_L={}",
                    digest(&e.l_values),
                    digest(&e.d_diag),
                    digest(&e.d_subdiag),
                    e.l_values.len()
                );
            }
            None => println!("{label} status={status:?} NO FACTORS"),
        }
    }
    Ok(())
}
