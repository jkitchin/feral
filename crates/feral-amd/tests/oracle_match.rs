//! T4 oracle-match: compare `feral-amd` output against the pinned
//! SuiteSparse AMD fixtures under `tests/data/amd_oracle/`.
//!
//! Slice A (commits 1-8) lacks mass elimination and supervariable
//! detection, so the permutation on dense fixtures legitimately
//! diverges from SuiteSparse. We assert:
//!
//! - bijection on `0..n` for every fixture;
//! - exact perm match for trivially-ordered fixtures (`diag_4`);
//! - `n_dense_deferred` matches the oracle on every fixture;
//! - dense-deferred vars land at the tail of the perm (`arrow_200`).

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use feral_amd::{amd_order_with_stats, CscPattern};

// ---- pattern generators (mirror the oracle harness) --------------

fn csc_from_triples(n: usize, triples: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
    let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &(i, j) in triples {
        set.insert((i, j));
        set.insert((j, i));
    }
    let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(r, c) in &set {
        cols[c].push(r);
    }
    for col in &mut cols {
        col.sort();
    }
    let mut col_ptr: Vec<usize> = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    let mut row_idx: Vec<usize> = Vec::new();
    for col in &cols {
        for &r in col {
            row_idx.push(r);
        }
        col_ptr.push(row_idx.len());
    }
    (col_ptr, row_idx)
}

fn arrow(n: usize) -> (Vec<usize>, Vec<usize>) {
    let mut t = Vec::new();
    for i in 0..n {
        t.push((i, i));
    }
    for i in 1..n {
        t.push((0, i));
    }
    csc_from_triples(n, &t)
}

fn band(n: usize, b: usize) -> (Vec<usize>, Vec<usize>) {
    let mut t = Vec::new();
    for i in 0..n {
        t.push((i, i));
        for k in 1..=b {
            if i + k < n {
                t.push((i, i + k));
            }
        }
    }
    csc_from_triples(n, &t)
}

fn grid_2d(m: usize, n: usize) -> (Vec<usize>, Vec<usize>) {
    let idx = |r: usize, c: usize| r * n + c;
    let total = m * n;
    let mut t = Vec::new();
    for r in 0..m {
        for c in 0..n {
            let k = idx(r, c);
            t.push((k, k));
            if r + 1 < m {
                t.push((k, idx(r + 1, c)));
            }
            if c + 1 < n {
                t.push((k, idx(r, c + 1)));
            }
        }
    }
    csc_from_triples(total, &t)
}

// ---- oracle parser -----------------------------------------------

struct Oracle {
    n: usize,
    n_dense: u32,
    perm: Vec<usize>,
}

fn parse_oracle(path: &Path) -> Oracle {
    let text = fs::read_to_string(path).expect("read oracle fixture");
    let mut map: HashMap<String, String> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        map.insert(k.trim().to_string(), v.trim().to_string());
    }
    let n: usize = map["n"].parse().expect("parse n");
    let n_dense: u32 = map["n_dense"].parse().expect("parse n_dense");
    let perm: Vec<usize> = map["perm"]
        .split_whitespace()
        .map(|s| s.parse().expect("parse perm entry"))
        .collect();
    assert_eq!(perm.len(), n, "oracle perm length mismatch in {:?}", path);
    Oracle { n, n_dense, perm }
}

fn oracle_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/amd_oracle")
        .join(format!("{}.txt", name))
}

fn assert_bijection(perm: &[usize], n: usize) {
    assert_eq!(perm.len(), n);
    let mut seen = vec![false; n];
    for &p in perm {
        assert!(p < n, "perm entry {} out of range", p);
        assert!(!seen[p], "perm entry {} repeated", p);
        seen[p] = true;
    }
}

fn run_fixture(name: &str, cp: &[usize], ri: &[usize]) -> (Vec<usize>, u32) {
    let oracle = parse_oracle(&oracle_path(name));
    let pattern = CscPattern::new(oracle.n, cp, ri).expect("valid CSC");
    let (perm, stats) = amd_order_with_stats(&pattern).expect("amd_order");
    assert_bijection(&perm, oracle.n);
    assert_eq!(
        stats.n_dense_deferred, oracle.n_dense,
        "{}: n_dense_deferred mismatch (got {}, oracle {})",
        name, stats.n_dense_deferred, oracle.n_dense
    );
    (perm, oracle.n_dense)
}

// ---- tests -------------------------------------------------------

#[test]
fn oracle_diag_4() {
    let (cp, ri) = band(4, 0);
    let oracle = parse_oracle(&oracle_path("diag_4"));
    let pattern = CscPattern::new(oracle.n, &cp, &ri).unwrap();
    let (perm, _stats) = amd_order_with_stats(&pattern).unwrap();
    assert_eq!(perm, oracle.perm, "diag_4 is trivially ordered");
}

#[test]
fn oracle_tridiag_10() {
    let (cp, ri) = band(10, 1);
    run_fixture("tridiag_10", &cp, &ri);
}

#[test]
fn oracle_arrow_5() {
    let (cp, ri) = arrow(5);
    run_fixture("arrow_5", &cp, &ri);
}

#[test]
fn oracle_arrow_200_hub_last() {
    let (cp, ri) = arrow(200);
    let (perm, n_dense) = run_fixture("arrow_200", &cp, &ri);
    assert_eq!(n_dense, 1, "arrow_200 has one dense hub");
    assert_eq!(
        perm[199], 0,
        "arrow_200 hub (var 0) must be dense-deferred to the tail"
    );
}

#[test]
fn oracle_band_20_3() {
    let (cp, ri) = band(20, 3);
    run_fixture("band_20_3", &cp, &ri);
}

#[test]
fn oracle_grid_7x7() {
    let (cp, ri) = grid_2d(7, 7);
    run_fixture("grid_7x7", &cp, &ri);
}

#[test]
fn oracle_amd_demo_24() {
    let (cp, ri) = grid_2d(6, 4);
    run_fixture("amd_demo_24", &cp, &ri);
}
