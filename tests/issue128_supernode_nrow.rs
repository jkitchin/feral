//! Issue #128 item E: `Supernode.nrow` must equal the true frontal-matrix
//! height for **every** supernode, including merged ones.
//!
//! Before the fix, `find_supernodes` set `nrow = col_counts[first_col].max(ncol)`.
//! That is exact for a fundamental supernode (its columns are nested, so the
//! first column's pattern contains every later one's) but wrong after a
//! size-based amalgamation: the merged group's first column is the *child's*,
//! and its pattern misses the rows only the parent contributes. Measured
//! undercounts before the fix ran to 40% of summed `nrow` (a 40x40 grid
//! Laplacian at `nemin = 32`: 1844 vs 3083).
//!
//! `nrow` is not used by the numeric kernels — `build_row_indices` recomputes
//! the truth — but it *is* what `contrib_size()`, `peak_contrib_bytes`,
//! `estimate_assembly_flops` (hence the `PAR_MIN_FLOPS` parallel dispatch),
//! and the profiler's front-size buckets read.
//!
//! Oracle: `SymbolicFactorization::static_rows(i)`, the issue #125 static
//! frontal row layout. That is an independent computation — a seen-deduped
//! union of own-column pattern reach and children's separators over the
//! permuted pattern — and `tests/static_assembly_maps.rs` separately pins it
//! both against a from-scratch `BTreeSet` recompute and against the numeric
//! `build_row_indices` output byte-for-byte. So this file checks the
//! amalgamation bookkeeping against a source that does not share its code.

use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::CscMatrix;

fn from_triplets(n: usize, mut ent: Vec<(usize, usize, f64)>) -> CscMatrix {
    ent.retain(|&(r, c, _)| r >= c);
    ent.sort_by_key(|&(r, c, _)| (c, r));
    ent.dedup_by_key(|&mut (r, c, _)| (r, c));
    let rows: Vec<usize> = ent.iter().map(|e| e.0).collect();
    let cols: Vec<usize> = ent.iter().map(|e| e.1).collect();
    let vals: Vec<f64> = ent.iter().map(|e| e.2).collect();
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("fixture")
}

/// 5-point Laplacian on an `nx x ny` grid — a nested-dissection-friendly
/// pattern with wide merged fronts.
fn grid_laplacian(nx: usize, ny: usize) -> CscMatrix {
    let mut e = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            let k = j * nx + i;
            e.push((k, k, 4.0));
            if i + 1 < nx {
                e.push((k + 1, k, -1.0));
            }
            if j + 1 < ny {
                e.push((k + nx, k, -1.0));
            }
        }
    }
    from_triplets(nx * ny, e)
}

/// Arrow matrix: every variable parented by the last. The archetype where
/// only one child is column-adjacent to its parent.
fn arrow(n: usize) -> CscMatrix {
    let mut e: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, i, 2.0 + i as f64)).collect();
    for i in 0..n - 1 {
        e.push((n - 1, i, 1.0));
    }
    from_triplets(n, e)
}

/// Chain-structured saddle-point KKT (the clnlbeam / steering family shape):
/// short two-row links, which is where amalgamation merges most aggressively.
fn chain_kkt(nblocks: usize) -> CscMatrix {
    let mut e = Vec::new();
    for b in 0..nblocks {
        let o = 3 * b;
        e.push((o, o, 1.0));
        e.push((o + 1, o + 1, 1.0));
        e.push((o + 2, o + 2, -1.0));
        e.push((o + 1, o, 0.5));
        e.push((o + 2, o, 1.0));
        e.push((o + 2, o + 1, 1.0));
        if b + 1 < nblocks {
            e.push((o + 3, o + 2, 1.0));
            e.push((o + 4, o + 2, 1.0));
        }
    }
    from_triplets(3 * nblocks, e)
}

fn banded(n: usize, bw: usize) -> CscMatrix {
    let mut e = Vec::new();
    for i in 0..n {
        e.push((i, i, 10.0));
        for d in 1..=bw {
            if i + d < n {
                e.push((i + d, i, -1.0));
            }
        }
    }
    from_triplets(n, e)
}

/// Disjoint blocks — exercises a forest rather than a single tree.
fn block_diagonal(nblocks: usize, bs: usize) -> CscMatrix {
    let mut e = Vec::new();
    for b in 0..nblocks {
        let o = b * bs;
        for i in 0..bs {
            e.push((o + i, o + i, 5.0 + i as f64));
            for j in 0..i {
                e.push((o + i, o + j, 1.0));
            }
        }
    }
    from_triplets(nblocks * bs, e)
}

fn fixtures() -> Vec<(String, CscMatrix)> {
    let mut v: Vec<(String, CscMatrix)> = Vec::new();
    for &(nx, ny) in &[(8usize, 8usize), (16, 16), (20, 31), (40, 40)] {
        v.push((format!("grid{nx}x{ny}"), grid_laplacian(nx, ny)));
    }
    for &n in &[16usize, 200] {
        v.push((format!("arrow{n}"), arrow(n)));
    }
    for &nb in &[10usize, 100, 400] {
        v.push((format!("chainkkt{}", nb * 3), chain_kkt(nb)));
    }
    for &(n, bw) in &[(64usize, 3usize), (500, 5), (800, 12)] {
        v.push((format!("banded{n}b{bw}"), banded(n, bw)));
    }
    for &(nb, bs) in &[(8usize, 6usize), (40, 11)] {
        v.push((format!("blockdiag{nb}x{bs}"), block_diagonal(nb, bs)));
    }
    v
}

/// Every supernode's `nrow` equals its true front height, across orderings,
/// `nemin` values (so both the merged and unmerged regimes are covered), and
/// both amalgamation strategies.
#[test]
fn supernode_nrow_equals_true_front_height() {
    let methods = [
        OrderingMethod::Amd,
        OrderingMethod::Auto,
        OrderingMethod::Amf,
    ];
    let mut checked = 0usize;
    let mut merged_seen = 0usize;

    for (name, a) in fixtures() {
        for method in methods.iter() {
            // nemin=1 disables amalgamation (the regime the old code got
            // right); the rest exercise increasingly aggressive merging.
            for nemin in [1usize, 2, 4, 8, 16, 32, 64] {
                let params = SupernodeParams {
                    nemin,
                    ..SupernodeParams::default()
                };
                let sym = match symbolic_factorize_with_method(&a, &params, method.clone()) {
                    Ok(s) => s,
                    // Not every ordering backend is compiled in; skip cleanly.
                    Err(_) => continue,
                };
                for (i, s) in sym.supernodes.iter().enumerate() {
                    let truth = sym.static_rows(i).len();
                    if truth == 0 {
                        continue; // maps not populated for this construction
                    }
                    assert_eq!(
                        s.nrow,
                        truth,
                        "{name} ({method:?}, nemin={nemin}) supernode {i} \
                         [first_col={}, ncol={}]: nrow={} but the true front \
                         height is {truth} (issue #128 item E)",
                        s.first_col,
                        s.ncol(),
                        s.nrow,
                    );
                    // A merged supernode is one wider than its first column's
                    // fundamental block; count them so the assertion above is
                    // known to have covered the regime that was broken.
                    if s.nrow > s.ncol() && s.ncol() > 1 {
                        merged_seen += 1;
                    }
                    checked += 1;
                }
            }
        }
    }

    assert!(
        checked > 5_000,
        "expected broad coverage, checked {checked}"
    );
    assert!(
        merged_seen > 500,
        "the merged-supernode regime is what issue #128 item E broke; only \
         {merged_seen} multi-column fronts were exercised"
    );
}

/// `nrow >= ncol` — the frontal matrix is never wider than it is tall. The
/// old code enforced this with an explicit `.max(ncol)` clamp; the union rule
/// should satisfy it structurally.
#[test]
fn supernode_nrow_never_below_ncol() {
    for (name, a) in fixtures() {
        for nemin in [1usize, 8, 32] {
            let params = SupernodeParams {
                nemin,
                ..SupernodeParams::default()
            };
            let sym = symbolic_factorize_with_method(&a, &params, OrderingMethod::Amd).unwrap();
            for (i, s) in sym.supernodes.iter().enumerate() {
                assert!(
                    s.nrow >= s.ncol(),
                    "{name} (nemin={nemin}) supernode {i}: nrow={} < ncol={}",
                    s.nrow,
                    s.ncol()
                );
                assert_eq!(
                    s.row_indices.len(),
                    s.nrow,
                    "{name} (nemin={nemin}) supernode {i}: row_indices length \
                     must track nrow (the F3.2b split asserts this)"
                );
            }
        }
    }
}
