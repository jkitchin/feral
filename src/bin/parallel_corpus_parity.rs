//! Full-corpus bit-exact parity audit for the Phase 2.5.2 parallel
//! multifrontal driver.
//!
//! Walks every `*.mtx` file under `data/matrices/kkt/`, runs the
//! sequential and parallel paths, and asserts bit-equal
//! `SparseFactors` (scaling vector, every node's L/D/contrib buffers,
//! inertia, row indices, permutations). Reports a summary plus any
//! mismatches.
//!
//! Usage: `cargo run --release --bin parallel_corpus_parity`.
//!
//! Exits non-zero on any parity failure so this can be used as a
//! guardrail. Intended as the Step E evidence for
//! `dev/plans/phase-2.5.2-rayon-assembly-tree.md`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use feral::numeric::factorize::{
    factorize_multifrontal, factorize_multifrontal_supernodal_parallel, NodeFactors, NumericParams,
    SparseFactors,
};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};

fn load_csc(path: &Path) -> Result<CscMatrix, String> {
    let mtx = read_mtx(path).map_err(|e| format!("read_mtx: {}", e))?;
    mtx.to_csc().map_err(|e| format!("to_csc: {}", e))
}

fn default_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    })
}

fn bits_eq(a: &[f64], b: &[f64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn inertia_eq(a: &Inertia, b: &Inertia) -> bool {
    a.positive == b.positive && a.negative == b.negative && a.zero == b.zero
}

fn node_eq(a: &NodeFactors, b: &NodeFactors) -> bool {
    a.first_col == b.first_col
        && a.ncol == b.ncol
        && a.nelim == b.nelim
        && a.n_delayed_in == b.n_delayed_in
        && a.nrow == b.nrow
        && a.row_indices == b.row_indices
        && inertia_eq(&a.inertia, &b.inertia)
        && a.frontal_factors.nrow == b.frontal_factors.nrow
        && a.frontal_factors.ncol == b.frontal_factors.ncol
        && a.frontal_factors.nelim == b.frontal_factors.nelim
        && a.frontal_factors.contrib_dim == b.frontal_factors.contrib_dim
        && a.frontal_factors.n_delayed == b.frontal_factors.n_delayed
        && a.frontal_factors.perm == b.frontal_factors.perm
        && a.frontal_factors.perm_inv == b.frontal_factors.perm_inv
        && bits_eq(&a.frontal_factors.l, &b.frontal_factors.l)
        && bits_eq(&a.frontal_factors.d_diag, &b.frontal_factors.d_diag)
        && bits_eq(&a.frontal_factors.d_subdiag, &b.frontal_factors.d_subdiag)
        && bits_eq(&a.frontal_factors.contrib, &b.frontal_factors.contrib)
        && a.frontal_factors.needs_refinement == b.frontal_factors.needs_refinement
}

fn factors_equal(a: &SparseFactors, b: &SparseFactors) -> bool {
    a.n == b.n
        && a.perm == b.perm
        && a.perm_inv == b.perm_inv
        && a.needs_refinement == b.needs_refinement
        && bits_eq(&a.scaling, &b.scaling)
        && a.node_factors.len() == b.node_factors.len()
        && a.node_factors
            .iter()
            .zip(b.node_factors.iter())
            .all(|(na, nb)| node_eq(na, nb))
}

fn collect_matrices(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_dir(root, &mut out);
    out.sort();
    out
}

fn walk_dir(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_dir(&path, out);
        } else if path.extension().map(|e| e == "mtx").unwrap_or(false) {
            out.push(path);
        }
    }
}

fn main() -> ExitCode {
    let root = Path::new("data/matrices/kkt");
    if !root.exists() {
        eprintln!("corpus root {:?} not found; run from project root", root);
        return ExitCode::from(2);
    }

    let matrices = collect_matrices(root);
    eprintln!("found {} matrices", matrices.len());

    let mut ok = 0usize;
    let mut mismatches: Vec<String> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut skipped = 0usize;

    let params = default_params();
    let snode_params = SupernodeParams::default();

    for (idx, path) in matrices.iter().enumerate() {
        if idx % 1000 == 0 && idx > 0 {
            eprintln!(
                "  progress: {}/{} ok={} skipped={} mismatches={} errors={}",
                idx,
                matrices.len(),
                ok,
                skipped,
                mismatches.len(),
                errors.len()
            );
        }
        let csc = match load_csc(path) {
            Ok(c) => c,
            Err(e) => {
                errors.push(format!("{}: load: {}", path.display(), e));
                continue;
            }
        };
        let sym = match symbolic_factorize(&csc, &snode_params) {
            Ok(s) => s,
            Err(e) => {
                errors.push(format!("{}: symbolic: {}", path.display(), e));
                continue;
            }
        };
        // Skip matrices that hit the dense fast-path — the parallel
        // driver is a multifrontal-only replacement; the gated
        // entry point (`factorize_multifrontal_parallel_with_workspace`)
        // correctly routes them to the dense path, but this audit
        // compares the multifrontal paths directly.
        if feral::numeric::factorize::should_use_dense_fast_path(csc.n, csc.row_idx.len()) {
            skipped += 1;
            continue;
        }
        let seq = match factorize_multifrontal(&csc, &sym, &params) {
            Ok(r) => r,
            Err(e) => {
                errors.push(format!("{}: sequential factor: {}", path.display(), e));
                continue;
            }
        };
        let par = match factorize_multifrontal_supernodal_parallel(&csc, &sym, &params) {
            Ok(r) => r,
            Err(e) => {
                errors.push(format!("{}: parallel factor: {}", path.display(), e));
                continue;
            }
        };
        if !inertia_eq(&seq.1, &par.1) || !factors_equal(&seq.0, &par.0) {
            mismatches.push(format!(
                "{}: inertia_seq={:?} inertia_par={:?} factors_bits_eq={}",
                path.display(),
                seq.1,
                par.1,
                factors_equal(&seq.0, &par.0)
            ));
            continue;
        }
        ok += 1;
    }

    println!("corpus parallel parity audit:");
    println!("  total     : {}", matrices.len());
    println!("  ok        : {}", ok);
    println!("  skipped   : {} (dense fast-path)", skipped);
    println!("  mismatches: {}", mismatches.len());
    println!("  errors    : {}", errors.len());

    if !mismatches.is_empty() {
        println!("\nfirst 20 mismatches:");
        for m in mismatches.iter().take(20) {
            println!("  {}", m);
        }
    }
    if !errors.is_empty() {
        println!("\nfirst 20 errors:");
        for e in errors.iter().take(20) {
            println!("  {}", e);
        }
    }

    if mismatches.is_empty() && errors.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
