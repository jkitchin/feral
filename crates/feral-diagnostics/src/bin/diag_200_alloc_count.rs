//! Issue #200 — count heap allocations per frontal matrix.
//!
//! The work-volume comparison (`diag_200_work_vs_ma57`) shows feral does
//! the same or fewer flops than MA57 on every matrix yet is slower
//! wherever fronts are small, so the gap is a fixed per-front cost, not
//! extra arithmetic. This counts one candidate for that cost directly: a
//! wrapping allocator tallies allocations and bytes across one warm
//! `factor()`, reported per front.
use feral::symbolic::{supernode::SupernodeParams, symbolic_factorize};
use feral::{read_mtx, NumericParams, Solver};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

static N_ALLOC: AtomicU64 = AtomicU64::new(0);
static N_BYTES: AtomicU64 = AtomicU64::new(0);
static ON: AtomicU64 = AtomicU64::new(0);

struct Counting;

// SAFETY: every method forwards verbatim to `System`, which upholds the
// `GlobalAlloc` contract; the counters are plain atomics that never touch
// the returned pointers or the allocation state.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ON.load(Relaxed) != 0 {
            N_ALLOC.fetch_add(1, Relaxed);
            N_BYTES.fetch_add(l.size() as u64, Relaxed);
        }
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if ON.load(Relaxed) != 0 {
            N_ALLOC.fetch_add(1, Relaxed);
            N_BYTES.fetch_add(n as u64, Relaxed);
        }
        System.realloc(p, l, n)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{:<22}{:>8}{:>10}{:>12}{:>10}{:>12}",
        "matrix", "fronts", "allocs", "alloc/front", "MB", "bytes/front"
    );
    for p in std::env::args().skip(1) {
        let csc = read_mtx(std::path::Path::new(&p)).and_then(|m| m.to_csc())?;
        let sp = SupernodeParams::default();
        let sym = symbolic_factorize(&csc, &sp)?;
        let nf = sym.supernodes.len() as u64;
        let mut solver = Solver::with_params(NumericParams::default(), sp);
        for _ in 0..3 {
            let _ = solver.factor(&csc, None);
        }
        N_ALLOC.store(0, Relaxed);
        N_BYTES.store(0, Relaxed);
        ON.store(1, Relaxed);
        let _ = solver.factor(&csc, None);
        ON.store(0, Relaxed);
        let (a, b) = (N_ALLOC.load(Relaxed), N_BYTES.load(Relaxed));
        let name = std::path::Path::new(&p)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        println!(
            "{:<22}{:>8}{:>10}{:>12.2}{:>10.1}{:>12.0}",
            name,
            nf,
            a,
            a as f64 / nf as f64,
            b as f64 / 1e6,
            b as f64 / nf as f64
        );
    }
    Ok(())
}
