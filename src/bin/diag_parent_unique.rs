//! Verify that every supernode is listed as a child of at most
//! one other supernode (proper elimination-tree invariant).

use feral::read_mtx;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
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
    eprintln!("collected {}", matrices.len());
    let sp = SupernodeParams::default();
    let mut bad = 0usize;
    for path in matrices.iter().take(5000) {
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
        let n_snodes = sym.supernodes.len();
        let mut parent_count = vec![0u32; n_snodes];
        for s in &sym.supernodes {
            for &c in &s.children {
                if c < n_snodes {
                    parent_count[c] += 1;
                }
            }
        }
        for (i, &k) in parent_count.iter().enumerate() {
            if k > 1 {
                bad += 1;
                if bad <= 5 {
                    eprintln!(
                        "MULTI-PARENT: {} snode {} has {} parents",
                        path.display(),
                        i,
                        k
                    );
                }
            }
        }
    }
    eprintln!("bad: {}", bad);
}
