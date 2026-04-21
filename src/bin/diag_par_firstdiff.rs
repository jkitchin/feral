//! For the first mismatching matrix between two back-to-back parallel
//! runs, find the first supernode whose frontal_factors differ.

use feral::numeric::factorize::{factorize_multifrontal_supernodal_parallel, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, ZeroPivotAction};
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

fn main() {
    let mut matrices = Vec::new();
    collect(Path::new("data/matrices/kkt"), &mut matrices);
    matrices.sort();

    let params = NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    });
    let sp = SupernodeParams::default();

    for (run, path) in matrices.iter().enumerate() {
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
        let par_a = match factorize_multifrontal_supernodal_parallel(&csc, &sym, &params) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let par_b = match factorize_multifrontal_supernodal_parallel(&csc, &sym, &params) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let na = &par_a.0.node_factors;
        let nb = &par_b.0.node_factors;
        if na.len() != nb.len() {
            eprintln!(
                "{}: DIFFERENT NODE COUNT {} vs {}",
                path.display(),
                na.len(),
                nb.len()
            );
            return;
        }
        for (i, (a, b)) in na.iter().zip(nb.iter()).enumerate() {
            let fa = &a.frontal_factors;
            let fb = &b.frontal_factors;
            let row_indices_eq = a.row_indices == b.row_indices;
            let l_eq = fa.l.len() == fb.l.len()
                && fa
                    .l
                    .iter()
                    .zip(fb.l.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits());
            let dd_eq = fa.d_diag.len() == fb.d_diag.len()
                && fa
                    .d_diag
                    .iter()
                    .zip(fb.d_diag.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits());
            let ds_eq = fa.d_subdiag.len() == fb.d_subdiag.len()
                && fa
                    .d_subdiag
                    .iter()
                    .zip(fb.d_subdiag.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits());
            let contrib_eq = fa.contrib.len() == fb.contrib.len()
                && fa
                    .contrib
                    .iter()
                    .zip(fb.contrib.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits());
            let perm_eq = fa.perm == fb.perm;
            let inertia_eq = a.inertia.positive == b.inertia.positive
                && a.inertia.negative == b.inertia.negative
                && a.inertia.zero == b.inertia.zero;
            let nd_eq = fa.n_delayed == fb.n_delayed;
            let nelim_eq = fa.nelim == fb.nelim;
            let ncol_eq = a.ncol == b.ncol;
            let ndi_eq = a.n_delayed_in == b.n_delayed_in;
            if !(row_indices_eq
                && l_eq
                && dd_eq
                && ds_eq
                && contrib_eq
                && perm_eq
                && inertia_eq
                && nd_eq
                && nelim_eq
                && ncol_eq
                && ndi_eq)
            {
                eprintln!(
                    "run {} ({}): first-diff at snode {} (of {})",
                    run,
                    path.display(),
                    i,
                    na.len()
                );
                eprintln!("  row_indices_eq={} l_eq={} dd_eq={} ds_eq={} contrib_eq={} perm_eq={} inertia_eq={} nd_eq={} nelim_eq={} ncol_eq={} ndi_eq={}",
                          row_indices_eq, l_eq, dd_eq, ds_eq, contrib_eq, perm_eq, inertia_eq, nd_eq, nelim_eq, ncol_eq, ndi_eq);
                eprintln!(
                    "  first_col={} ncol={} nelim={} n_delayed_in={}",
                    a.first_col, a.ncol, a.nelim, a.n_delayed_in
                );
                eprintln!(
                    "  fa.nrow={} fa.ncol={} fa.nelim={} fa.contrib_dim={} fa.n_delayed={}",
                    fa.nrow, fa.ncol, fa.nelim, fa.contrib_dim, fa.n_delayed
                );
                eprintln!(
                    "  fb.nrow={} fb.ncol={} fb.nelim={} fb.contrib_dim={} fb.n_delayed={}",
                    fb.nrow, fb.ncol, fb.nelim, fb.contrib_dim, fb.n_delayed
                );
                eprintln!("  snode children: {:?}", sym.supernodes[i].children);
                eprintln!("  inertia a={:?} b={:?}", a.inertia, b.inertia);
                return;
            }
        }
    }
    eprintln!("no divergence found");
}
