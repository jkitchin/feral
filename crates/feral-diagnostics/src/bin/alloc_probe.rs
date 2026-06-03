//! Per-call allocation counter for `factorize_multifrontal`.
//!
//! Installs a custom `#[global_allocator]` that wraps `System` and
//! increments atomic counters (alloc / realloc / dealloc counts + total
//! bytes requested) when a gate flag is on. We drive the gate around
//! one `factorize_multifrontal(matrix, &symbolic, &params)` call so the
//! measured counters reflect allocations inside the numeric phase only.
//!
//! Purpose: confirm or refute the Lever D.1 alloc-churn hypothesis from
//! `dev/research/sparse-tail-perf-2026-04-19.md` §6 before we design a
//! `FactorWorkspace` API. If AVION2_0000 / BATCH_0000 / LAKES_1199 show
//! hundreds of allocations per call we have a target; if they show a
//! handful we need to pivot (D.3/D.4).

use feral::numeric::factorize::factorize_multifrontal;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, NumericParams, ZeroPivotAction};
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static REALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAlloc;

// SAFETY: `CountingAlloc` forwards every alloc / dealloc / realloc call
// to `std::alloc::System`, which is itself a valid `GlobalAlloc`. The
// atomic counters use `Ordering::Relaxed` because we only need eventual
// consistency for a post-hoc read, not cross-thread synchronisation.
// The wrapped allocator is the safety backbone; we add no new invariants.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            REALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            if new_size > layout.size() {
                ALLOC_BYTES.fetch_add((new_size - layout.size()) as u64, Ordering::Relaxed);
            }
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

struct Counts {
    allocs: u64,
    reallocs: u64,
    deallocs: u64,
    bytes: u64,
    ns: u128,
}

fn reset() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    REALLOC_COUNT.store(0, Ordering::Relaxed);
    DEALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
}

fn snapshot(ns: u128) -> Counts {
    Counts {
        allocs: ALLOC_COUNT.load(Ordering::Relaxed),
        reallocs: REALLOC_COUNT.load(Ordering::Relaxed),
        deallocs: DEALLOC_COUNT.load(Ordering::Relaxed),
        bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        ns,
    }
}

fn run_one(family: &str, sample: &str) {
    let path = PathBuf::from(format!(
        "data/matrices/kkt/{}/{}{}.mtx",
        family, family, sample
    ));
    let mtx = match read_mtx(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP {}{}: {}", family, sample, e);
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP {}{}: csc: {}", family, sample, e);
            return;
        }
    };
    let n = csc.n;
    let nnz = csc.row_idx.len();
    let snode_params = SupernodeParams::default();
    let factor_params = NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    });
    let sym = match symbolic_factorize(&csc, &snode_params) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP {}{}: sym: {}", family, sample, e);
            return;
        }
    };
    let n_snodes = sym.supernodes.len();

    // 100 iterations so the best-of-iters duration has a meaningful
    // noise floor. Alloc counts are deterministic per call, so min==max
    // across iters is still the expected invariant (verified below).
    let iters = 100;
    let mut best_ns = u128::MAX;
    let mut best: Option<Counts> = None;
    let mut max_allocs: u64 = 0;
    let mut min_allocs: u64 = u64::MAX;
    for _ in 0..iters {
        reset();
        COUNTING.store(true, Ordering::Relaxed);
        let t = Instant::now();
        let _factors = factorize_multifrontal(&csc, &sym, &factor_params);
        let ns = t.elapsed().as_nanos();
        COUNTING.store(false, Ordering::Relaxed);
        let snap = snapshot(ns);
        max_allocs = max_allocs.max(snap.allocs);
        min_allocs = min_allocs.min(snap.allocs);
        if ns < best_ns {
            best_ns = ns;
            best = Some(snap);
        }
    }
    let Some(b) = best else {
        eprintln!("SKIP {}{}: no iter succeeded", family, sample);
        return;
    };
    let per_alloc_ns = if b.allocs > 0 {
        b.ns as f64 / b.allocs as f64
    } else {
        0.0
    };
    println!(
        "{:<14} {:>5} {:>7} {:>5} {:>9} {:>8} {:>8} {:>10} {:>10.1} {:>7.1} {:>7}-{:<7}",
        format!("{}{}", family, sample),
        n,
        nnz,
        n_snodes,
        b.allocs,
        b.reallocs,
        b.deallocs,
        b.bytes,
        b.ns as f64 / 1000.0,
        per_alloc_ns,
        min_allocs,
        max_allocs,
    );
}

fn main() {
    println!(
        "{:<14} {:>5} {:>7} {:>5} {:>9} {:>8} {:>8} {:>10} {:>10} {:>7} {:>15}",
        "matrix",
        "n",
        "nnz",
        "snod",
        "allocs",
        "reallocs",
        "deallocs",
        "bytes",
        "fac(us)",
        "ns/alo",
        "min-max allocs",
    );
    println!("{}", "-".repeat(125));
    let cases: &[(&str, &str)] = &[
        // D.1 primary targets: families where geomean factor/MUMPS > 1.
        ("AVION2", "_0000"),
        ("AVION2", "_0500"),
        ("BATCH", "_0000"),
        ("BATCH", "_0500"),
        // Top-10 absolute outliers.
        ("LAKES", "_1199"),
        ("TRO3X3", "_0013"),
        // Tiny-matrix baseline (Class A, n <= 9) for per-call overhead.
        ("HAHN1LS", "_0429"),
        ("FBRAIN3LS", "_0003"),
        // A larger matrix that we win on, as a sanity reference.
        ("HAHN1", "_0000"),
        ("VESUVIO", "_0000"),
    ];
    for (f, s) in cases {
        run_one(f, s);
    }
}
