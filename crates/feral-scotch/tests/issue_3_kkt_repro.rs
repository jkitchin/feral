//! Reproducer for issue #3: ScotchND silently falls back to AMD on
//! KKT-shaped matrices.
//!
//! Builds a small PoissonControl KKT *pattern* (n_kkt = 3K²), runs
//! `feral_scotch::scotch_order_full` and `feral_amd::amd_order` on the
//! same full-symmetric pattern, and asserts what each branch produced.
//! The interesting assertion is `n_amd_leaf_calls` in `ScotchStats`:
//! if the entire ND recursion bottoms out in AMD leaves (or even one
//! top-level degenerate-bisection fallback), the resulting permutation
//! is byte-equal to AMD's, while `resolved_method` (in the caller)
//! still says `ScotchND`.

use feral_amd::amd_order;
use feral_ordering_core::CscPattern;
use feral_scotch::{scotch_order_full, ScotchOptions};

use std::collections::BTreeSet;

fn poisson_kkt_pattern(k: usize) -> (Vec<i32>, Vec<i32>, usize) {
    // Build the full-symmetric pattern (with diagonal) of the Poisson
    // optimal-control KKT system from `src/bin/diag_poisson_kkt.rs`.
    // Only the pattern matters for the ordering call.
    let m = k * k;
    let n = 3 * m;
    let mut s: BTreeSet<(i32, i32)> = BTreeSet::new();

    for i in 0..n {
        s.insert((i as i32, i as i32));
    }

    for i in 0..k {
        for j in 0..k {
            let c = i * k + j;
            let con_row = (2 * m + c) as i32;
            // 5-point stencil couples constraint row to u block
            let center = c as i32;
            s.insert((con_row, center));
            s.insert((center, con_row));
            if i > 0 {
                let nbr = ((i - 1) * k + j) as i32;
                s.insert((con_row, nbr));
                s.insert((nbr, con_row));
            }
            if i + 1 < k {
                let nbr = ((i + 1) * k + j) as i32;
                s.insert((con_row, nbr));
                s.insert((nbr, con_row));
            }
            if j > 0 {
                let nbr = (i * k + (j - 1)) as i32;
                s.insert((con_row, nbr));
                s.insert((nbr, con_row));
            }
            if j + 1 < k {
                let nbr = (i * k + (j + 1)) as i32;
                s.insert((con_row, nbr));
                s.insert((nbr, con_row));
            }
            // f coupling: con_row <-> (m + c)
            let f = (m + c) as i32;
            s.insert((con_row, f));
            s.insert((f, con_row));
        }
    }

    let mut col_ptr: Vec<i32> = vec![0];
    let mut row_idx: Vec<i32> = Vec::new();
    let mut by_col: Vec<Vec<i32>> = vec![Vec::new(); n];
    for (r, c) in s {
        by_col[c as usize].push(r);
    }
    for col in &mut by_col {
        col.sort();
    }
    for col in &by_col {
        for &r in col {
            row_idx.push(r);
        }
        col_ptr.push(row_idx.len() as i32);
    }
    (col_ptr, row_idx, n)
}

#[test]
fn issue_3_scotch_bisection_degenerates_on_kkt() {
    // Locks in the bisection-degenerate symptom on a KKT pattern.
    // Independent of how the upper layer surfaces the fact, this
    // asserts that scotch_nd never produces a separator on this
    // shape — which is *why* the upper-layer fallback fires.
    let k = 20; // n_kkt = 1200; nnz_per_row ~ 4.9 on full-symmetric
    let (col_ptr, row_idx, n) = poisson_kkt_pattern(k);
    let pat = CscPattern::new(n, &col_ptr, &row_idx).expect("pattern valid");

    let amd_perm = amd_order(&pat).expect("amd ok");
    let (scotch_perm, _ostats, sstats) =
        scotch_order_full(&pat, &ScotchOptions::default()).expect("scotch ok");

    eprintln!("issue #3 reproducer: n={}, scotch stats={:?}", n, sstats);

    // Bisection produced no separator → recursion bottomed out in
    // amd_leaf for the entire graph → permutation matches AMD's.
    assert_eq!(
        sstats.n_separator_vertices, 0,
        "ScotchND was expected to degenerate (no separator) on KKT; \
         if this fires, the SCOTCH driver has improved and the \
         visibility-fix rationale should be re-checked"
    );
    assert!(
        sstats.n_amd_leaf_calls >= 1,
        "expected at least one amd_leaf fallback on a degenerate-bisection KKT"
    );
    assert_eq!(
        amd_perm, scotch_perm,
        "if bisection produced no separator, the perm must equal AMD's exactly"
    );
}
