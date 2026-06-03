//! Does the parallel driver give bit-identical results on
//! back-to-back runs of the same matrix? If YES: parallel is
//! deterministic per-call but diverges from sequential on some
//! runs. If NO: parallel itself is nondeterministic.

use feral::numeric::factorize::{
    factorize_multifrontal, factorize_multifrontal_supernodal_parallel, NumericParams,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, Inertia, ZeroPivotAction};
use std::path::{Path, PathBuf};

fn collect(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().map(|e| e == "mtx").unwrap_or(false) {
            out.push(p);
        }
    }
}

fn factor_hash(factors: &feral::numeric::factorize::SparseFactors) -> (u64, Inertia) {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let mut total = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    for n in factors.node_factors.iter() {
        total.positive += n.inertia.positive;
        total.negative += n.inertia.negative;
        total.zero += n.inertia.zero;
        // Hash the bit-level L/D data.
        for v in n.frontal_factors.l.iter() {
            v.to_bits().hash(&mut h);
        }
        for v in n.frontal_factors.d_diag.iter() {
            v.to_bits().hash(&mut h);
        }
        for v in n.frontal_factors.d_subdiag.iter() {
            v.to_bits().hash(&mut h);
        }
        n.inertia.positive.hash(&mut h);
        n.inertia.negative.hash(&mut h);
        n.inertia.zero.hash(&mut h);
    }
    (h.finish(), total)
}

fn main() {
    let mut matrices = Vec::new();
    collect(Path::new("data/matrices/kkt"), &mut matrices);
    matrices.sort();
    eprintln!("collected {} matrices", matrices.len());

    let params = NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    });
    let sp = SupernodeParams::default();

    let mut par_nondet = 0usize;
    let mut par_vs_seq_mismatch = 0usize;
    let mut runs = 0usize;

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
        // Sequential reference.
        let seq = match factorize_multifrontal(&csc, &sym, &params) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let (seq_hash, seq_inertia) = factor_hash(&seq.0);

        // Two back-to-back parallel runs.
        let par_a = match factorize_multifrontal_supernodal_parallel(&csc, &sym, &params) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let par_b = match factorize_multifrontal_supernodal_parallel(&csc, &sym, &params) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let (par_a_hash, par_a_inertia) = factor_hash(&par_a.0);
        let (par_b_hash, par_b_inertia) = factor_hash(&par_b.0);
        runs += 1;

        let par_matches_itself = par_a_hash == par_b_hash
            && par_a_inertia.positive == par_b_inertia.positive
            && par_a_inertia.negative == par_b_inertia.negative
            && par_a_inertia.zero == par_b_inertia.zero;
        if !par_matches_itself {
            par_nondet += 1;
            if par_nondet <= 3 {
                eprintln!(
                    "PAR NONDET at run {} ({}): par_a={:?} par_b={:?} (hashes differ: {})",
                    runs,
                    path.display(),
                    par_a_inertia,
                    par_b_inertia,
                    par_a_hash != par_b_hash
                );
            }
        }
        let par_matches_seq = par_a_hash == seq_hash
            && par_a_inertia.positive == seq_inertia.positive
            && par_a_inertia.negative == seq_inertia.negative
            && par_a_inertia.zero == seq_inertia.zero;
        if !par_matches_seq {
            par_vs_seq_mismatch += 1;
        }
        if par_nondet >= 5 || par_vs_seq_mismatch >= 5 {
            break;
        }
    }
    eprintln!(
        "final: {} runs, par-vs-par-nondet {}, par-vs-seq-mismatch {}",
        runs, par_nondet, par_vs_seq_mismatch
    );
}
