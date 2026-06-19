//! Allocation probe + regression guard for the Forrest–Tomlin update chain on
//! the real `casctanks` trace (discopt#229), the wide-bump workload v0.11.1
//! optimized.
//!
//! A counting `#[global_allocator]` wraps `System` so we can snapshot
//! alloc / realloc / byte counts in a window around *only the update loop* — the
//! per-segment factorization is excluded. It began as the Phase-0 gate
//! instrument (`dev/research/lu-update-alloc-pooling-2026-06-19.md`): baseline
//! was ~1804 allocs/update against an 85.8 µs/update budget. After pooling the
//! bump-loop buffers and the `saved` snapshot, it is ~82 allocs/update.
//!
//! It now also *guards* that gain: the bump elimination must not re-introduce
//! per-pivot / per-axpy / per-changed-row allocation (which would push the count
//! back into the hundreds–thousands). The bound below is generous — the residual
//! ~82 is the irreducible floor (the retained `ops`→`etas` growth plus the
//! handful of O(1) spike buffers left unpooled), not zero.
//!
//! Default fixture: the in-tree reduced `casctanks.txt` (3 widest-bump segments,
//! 144 updates). Point `FERAL_LU_TRACE` at the full extracted trace for the
//! complete 36-segment / 1702-update measurement.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{LuParams, SparseLu, SparseLuSymbolic};

// --- counting allocator -------------------------------------------------

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static REALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

struct Counting;

// SAFETY: every method forwards verbatim to the `System` allocator and returns
// its pointer unchanged; the only added work is incrementing relaxed atomic
// counters. No allocator invariant is altered (same size/align contract, same
// pointer provenance), so wrapping `System` this way is sound.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        REALLOCS.fetch_add(1, Relaxed);
        if new_size > layout.size() {
            BYTES.fetch_add((new_size - layout.size()) as u64, Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[derive(Clone, Copy)]
struct Snap {
    allocs: u64,
    reallocs: u64,
    bytes: u64,
}

fn snap() -> Snap {
    Snap {
        allocs: ALLOCS.load(Relaxed),
        reallocs: REALLOCS.load(Relaxed),
        bytes: BYTES.load(Relaxed),
    }
}

// --- trace parsing (mirrors lu_update_casctanks.rs) ---------------------

type SparseCol = Vec<(usize, f64)>;

struct Segment {
    basis: Vec<SparseCol>,
    updates: Vec<(usize, SparseCol)>,
}

fn parse_sparse_col(fields: &[&str]) -> SparseCol {
    let mut col: SparseCol = fields
        .iter()
        .filter_map(|tok| {
            let (r, v) = tok.split_once(':')?;
            Some((r.parse::<usize>().ok()?, v.parse::<f64>().ok()?))
        })
        .collect();
    col.sort_by_key(|&(r, _)| r);
    col
}

fn parse_trace(text: &str) -> (usize, Vec<Segment>) {
    let mut m = 0usize;
    let mut segments: Vec<Segment> = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("M") => m = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            Some("REFACTOR") => segments.push(Segment {
                basis: Vec::new(),
                updates: Vec::new(),
            }),
            Some("BCOL") => {
                let rest: Vec<&str> = it.collect();
                let col = parse_sparse_col(&rest[2.min(rest.len())..]);
                segments
                    .last_mut()
                    .expect("BCOL before REFACTOR")
                    .basis
                    .push(col);
            }
            Some("UPDATE") => {
                let rest: Vec<&str> = it.collect();
                let slot: usize = rest[0].parse().expect("update slot");
                let col = parse_sparse_col(&rest[2.min(rest.len())..]);
                segments
                    .last_mut()
                    .expect("UPDATE before REFACTOR")
                    .updates
                    .push((slot, col));
            }
            _ => {}
        }
    }
    (m, segments)
}

fn to_dense(col: &SparseCol, m: usize) -> Vec<f64> {
    let mut d = vec![0.0; m];
    for &(r, v) in col {
        d[r] = v;
    }
    d
}

fn replay_params() -> LuParams {
    LuParams {
        max_updates: 1_000_000,
        max_growth: 1e30,
        ..LuParams::default()
    }
}

fn fixture_text() -> Option<String> {
    let path = std::env::var("FERAL_LU_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/lu_trace/casctanks.txt")
        });
    std::fs::read_to_string(&path).ok()
}

/// Measure allocations attributable ONLY to the update loop (each segment's
/// factorization happens outside the snapshot window). Prints a per-update
/// breakdown and guards against regressing the pooling: per-update allocations
/// must stay far below the ~1804 pre-pooling baseline.
#[test]
fn casctanks_update_chain_alloc_probe() {
    let Some(text) = fixture_text() else {
        eprintln!("skipping alloc probe: trace fixture not readable");
        return;
    };
    let (m, segments) = parse_trace(&text);
    assert!(m > 0 && !segments.is_empty(), "empty/invalid trace");

    // Pre-densify entering columns OUTSIDE the measured window.
    type Prepared = (SparseLu, Vec<(usize, Vec<f64>)>);
    let prepared: Vec<Prepared> = segments
        .iter()
        .map(|seg| {
            let a = SparseColMatrix::from_sparse_columns(m, &seg.basis).expect("matrix");
            let sym = SparseLuSymbolic::analyze(&a).expect("analyze");
            let lu = SparseLu::factor(&a, &sym, replay_params()).expect("factor");
            let ups: Vec<(usize, Vec<f64>)> = seg
                .updates
                .iter()
                .map(|(slot, col)| (*slot, to_dense(col, m)))
                .collect();
            (lu, ups)
        })
        .collect();

    let total_updates: usize = prepared.iter().map(|(_, u)| u.len()).sum();
    assert!(total_updates > 0, "trace exercised no updates");

    // Window: snapshot, apply every segment's update chain, snapshot.
    let before = snap();
    let mut applied = 0usize;
    let mut errs = 0usize;
    for (mut lu, ups) in prepared {
        for (slot, dcol) in &ups {
            match lu.update(*slot, dcol) {
                Ok(()) => applied += 1,
                Err(_) => errs += 1,
            }
        }
        std::hint::black_box(&lu);
    }
    let after = snap();

    let da = after.allocs - before.allocs;
    let dr = after.reallocs - before.reallocs;
    let db = after.bytes - before.bytes;
    let n = total_updates as f64;

    eprintln!(
        "\n=== casctanks FT-update alloc probe ===\n\
         m={m}  segments={}  updates_applied={applied}  needs_refactor={errs}\n\
         total: allocs={da}  reallocs={dr}  bytes={db}\n\
         per-update: allocs={:.1}  reallocs={:.1}  bytes={:.0}\n\
         (at ~60 ns/alloc this is ~{:.1} us/update of allocator time)\n",
        segments.len(),
        da as f64 / n,
        dr as f64 / n,
        db as f64 / n,
        (da + dr) as f64 / n * 60.0 / 1000.0,
    );

    assert!(applied > 0, "no updates applied");

    // Regression guard. Measured ~82 allocs/update after pooling (down from
    // ~1804); the bound is generous so it tolerates fixture/allocator variation
    // while still failing loudly if a future change re-introduces the per-pivot
    // `pivot_data` clone, per-axpy `row_sub` allocation, or per-changed-row
    // `saved` clone (each of which would push the count back into the hundreds).
    let allocs_per_update = da as f64 / n;
    assert!(
        allocs_per_update < 250.0,
        "per-update allocations regressed to {allocs_per_update:.1} (was ~82 after \
         pooling, ~1804 before); the FT-update buffer pools may have been broken"
    );
}
