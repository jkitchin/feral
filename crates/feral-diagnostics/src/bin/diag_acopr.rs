//! Repro: factor N random KKT matrices in sequence with the parallel
//! driver, each compared against sequential, to track down where the
//! corpus audit's mismatches originate.

use feral::numeric::factorize::{
    factorize_multifrontal, factorize_multifrontal_supernodal_parallel, NumericParams,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};
use std::path::{Path, PathBuf};

fn collect_matrices(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_matrices(&p, out);
        } else if p.extension().map(|e| e == "mtx").unwrap_or(false) {
            out.push(p);
        }
    }
}

fn factor_pair(
    csc: &CscMatrix,
    sym: &feral::symbolic::SymbolicFactorization,
    params: &NumericParams,
) -> Option<(Inertia, Inertia)> {
    let seq = factorize_multifrontal(csc, sym, params).ok()?;
    let par = factorize_multifrontal_supernodal_parallel(csc, sym, params).ok()?;
    Some((seq.1, par.1))
}

fn main() {
    let mut matrices = Vec::new();
    collect_matrices(Path::new("data/matrices/kkt"), &mut matrices);
    matrices.sort();
    eprintln!("collected {} matrices", matrices.len());

    let params = NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    });
    let sp = SupernodeParams::default();

    let mut mismatch_count = 0usize;
    let mut run_count = 0usize;
    let mut first_mismatch: Option<(PathBuf, Inertia, Inertia)> = None;

    for path in matrices.iter() {
        let mtx = match read_mtx(path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(_) => continue,
        };
        let sym = match symbolic_factorize(&csc, &sp) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if feral::numeric::factorize::should_use_dense_fast_path(csc.n, csc.row_idx.len()) {
            continue;
        }
        let (seq_i, par_i) = match factor_pair(&csc, &sym, &params) {
            Some(v) => v,
            None => continue,
        };
        run_count += 1;
        if seq_i.positive != par_i.positive
            || seq_i.negative != par_i.negative
            || seq_i.zero != par_i.zero
        {
            mismatch_count += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((path.clone(), seq_i.clone(), par_i.clone()));
                eprintln!(
                    "FIRST MISMATCH at run {} ({}): seq={:?} par={:?}",
                    run_count,
                    path.display(),
                    seq_i,
                    par_i
                );

                // Re-run just this matrix in isolation (10 times) to see
                // if the mismatch reproduces.
                eprintln!("  replaying this matrix 10x in isolation:");
                for i in 0..10 {
                    let (s, p) = factor_pair(&csc, &sym, &params).expect("replay");
                    eprintln!(
                        "    iter {}: seq={:?} par={:?} match={}",
                        i,
                        s,
                        p,
                        s.positive == p.positive && s.negative == p.negative && s.zero == p.zero
                    );
                }
            }
            if mismatch_count >= 5 {
                break;
            }
        }
    }
    eprintln!(
        "final: {} runs, {} mismatches (stopped early)",
        run_count, mismatch_count
    );
}
