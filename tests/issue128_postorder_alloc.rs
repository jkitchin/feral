//! Issue #128 item D: allocation probe for the elimination-tree postorder.
//!
//! All three postorder variants used to call `EliminationTree::children()`,
//! which builds `n` separate `Vec`s, and then materialized a freshly
//! cloned-and-sorted child `Vec` per node on top of that. The traversals are
//! linear in *work* (the S1 fix, `dev/research/repo-review-2026-06-09.md`,
//! pinned by `test_postorder_star_sort_work_is_linear`) but were still
//! `2n+` *allocations* per call, and the symbolic pipeline can run two
//! postorders per factorization.
//!
//! They now share a CSR-style child arena — two `Vec`s total, each node's
//! slice ordered once in place, and a `(node, cursor)` DFS stack — so the
//! allocation count is O(1) in `n` rather than O(n).
//!
//! This probe measures allocations attributable to *only* the postorder call
//! (etree construction happens outside the snapshot window). The traversal's
//! own output (`order` + `inv`, two length-`n` `Vec`s) is unavoidable, so the
//! floor is a small constant, not zero.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

use feral::ordering::elimination_tree::EliminationTree;
use feral::ordering::postorder::{biased_postorder, postorder, schur_constrained_postorder};
use feral::CscMatrix;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);

struct Counting;

// SAFETY: every method forwards verbatim to the `System` allocator and
// returns its pointer unchanged; the only added work is incrementing relaxed
// atomics. No allocator invariant is altered.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Relaxed) {
            ALLOCS.fetch_add(1, Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Relaxed) {
            ALLOCS.fetch_add(1, Relaxed);
        }
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn measure(f: impl FnOnce()) -> u64 {
    let before = ALLOCS.load(Relaxed);
    ARMED.store(true, Relaxed);
    f();
    ARMED.store(false, Relaxed);
    ALLOCS.load(Relaxed) - before
}

/// Star etree: one root with `n - 1` children. The arrow / bordered-KKT
/// shape, and the worst case for per-node child-list churn.
fn star(n: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for i in 0..n {
        rows.push(i);
        cols.push(i);
    }
    for i in 0..n - 1 {
        rows.push(n - 1);
        cols.push(i);
    }
    let vals = vec![1.0; rows.len()];
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("star fixture")
}

/// Chain etree: `0 -> 1 -> ... -> n-1`, one child per node.
fn chain(n: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        if i + 1 < n {
            rows.push(i + 1);
            cols.push(i);
        }
    }
    let vals = vec![1.0; rows.len()];
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("chain fixture")
}

/// Allocation count must not scale with `n`. Doubling `n` doubles the *work*
/// but must leave the allocation count essentially flat — the signature of
/// the arena replacing the per-node `Vec`s.
#[test]
fn postorder_allocations_do_not_scale_with_n() {
    for (name, build) in [
        ("star", star as fn(usize) -> CscMatrix),
        ("chain", chain as fn(usize) -> CscMatrix),
    ] {
        let mut counts = Vec::new();
        for &n in &[2_000usize, 4_000, 8_000] {
            let m = build(n);
            let pat = m.symmetric_pattern();
            let etree = EliminationTree::from_pattern(&pat);

            let plain = measure(|| {
                std::hint::black_box(postorder(&etree));
            });
            let bias = vec![false; n];
            let biased = measure(|| {
                std::hint::black_box(biased_postorder(&etree, &bias));
            });
            let is_schur = vec![false; n];
            let schur = measure(|| {
                std::hint::black_box(schur_constrained_postorder(&etree, &is_schur));
            });

            eprintln!("{name} n={n}: postorder={plain} biased={biased} schur={schur} allocs");
            counts.push((n, plain, biased, schur));
        }

        // Before item D each variant allocated one `Vec` per node (the
        // `children()` rows) plus one per stack push, i.e. >= 2n. Anything
        // remotely proportional to n fails this bound at n = 8000.
        for &(n, plain, biased, schur) in &counts {
            for (variant, c) in [("postorder", plain), ("biased", biased), ("schur", schur)] {
                assert!(
                    c < 64,
                    "{name} n={n} {variant}: {c} allocations — expected O(1); \
                     the per-node child Vecs are back (issue #128 item D)"
                );
            }
        }

        // Flatness: 4x the nodes must not mean materially more allocations.
        let (_, p0, b0, s0) = counts[0];
        let (_, p2, b2, s2) = counts[2];
        assert!(
            p2 <= p0 + 4 && b2 <= b0 + 4 && s2 <= s0 + 4,
            "{name}: allocation count grew with n ({p0}->{p2}, {b0}->{b2}, {s0}->{s2})"
        );
    }
}
