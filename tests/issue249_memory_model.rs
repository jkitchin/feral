//! Issue #249: `WorkEstimate::peak_bytes` (sequential) and
//! `WorkEstimate::peak_bytes_parallel` must bound what `Solver::factor`
//! really allocates, and the stored-factor terms must equal what the
//! numeric phase really keeps.
//!
//! The oracle is the process's own heap: a counting global allocator
//! (this test binary's only allocator) records the high-water of live
//! heap bytes during one `Solver::factor`. It is external to the model —
//! the model is computed from the analysis alone, before any numeric
//! allocation. Everything runs in a single `#[test]` so no other test
//! thread allocates while a peak is being measured.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use feral::symbolic::OrderingMethod;
use feral::{CscMatrix, FactorStatus, Solver};

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn add(sz: usize) {
    let now = LIVE.fetch_add(sz, Ordering::Relaxed) + sz;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's layout
// unchanged; the counters are bookkeeping only and never touch memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // SAFETY: forwarded with the caller's valid layout.
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        // SAFETY: forwarded with the caller's valid layout.
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        // SAFETY: `p` came from `System` with layout `l`.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        // SAFETY: forwarded unchanged; `p` came from `System` with `l`.
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new >= l.size() {
                add(new - l.size());
            } else {
                LIVE.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// 5-point Laplacian on a `k × k` grid, lower triangle (SPD).
fn grid(k: usize) -> CscMatrix {
    let n = k * k;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        r.push(j);
        c.push(j);
        v.push(4.0);
        if j % k + 1 < k {
            r.push(j + 1);
            c.push(j);
            v.push(-1.0);
        }
        if j + k < n {
            r.push(j + k);
            c.push(j);
            v.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &r, &c, &v).expect("grid")
}

/// IPM-style KKT `[[I, Aᵀ], [A, -1e-8 I]]` with `A` the horizontal-edge
/// difference operator of a `k × k` grid (each row `x_i - x_{i+1}`),
/// lower triangle. Quasidefinite: inertia `(k², rows, 0)`.
fn kkt(k: usize) -> (CscMatrix, usize, usize) {
    let nv = k * k;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..nv {
        r.push(j);
        c.push(j);
        v.push(1.0);
    }
    let mut m = 0;
    for j in 0..nv {
        if j % k + 1 < k {
            let row = nv + m;
            r.extend([row, row]);
            c.extend([j, j + 1]);
            v.extend([1.0, -1.0]);
            // Couple each constraint to the variable one grid row below,
            // so the graph is two-dimensional and the tree is wide.
            if j + k < nv {
                r.push(row);
                c.push(j + k);
                v.push(0.5);
            }
            m += 1;
        }
    }
    for i in 0..m {
        r.push(nv + i);
        c.push(nv + i);
        v.push(-1e-8);
    }
    let n = nv + m;
    (CscMatrix::from_triplets(n, &r, &c, &v).expect("kkt"), nv, m)
}

/// Factor once with a fresh `Solver`, returning (measured peak heap bytes
/// above the pre-factor level, the solver).
fn measure(m: &CscMatrix, method: OrderingMethod, parallel: bool) -> (usize, Solver) {
    let mut s = Solver::new().with_ordering(method).with_parallel(parallel);
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let st = s.factor(m, None);
    assert!(matches!(st, FactorStatus::Success), "factor: {st:?}");
    (PEAK.load(Ordering::Relaxed) - base, s)
}

#[test]
fn predicted_peak_bounds_the_measured_heap_and_stored_terms_are_exact() {
    let (kkt_m, nv, nc) = kkt(70);
    let cases: Vec<(&str, CscMatrix, OrderingMethod)> = vec![
        ("grid120/amd", grid(120), OrderingMethod::Amd),
        ("grid120/metis", grid(120), OrderingMethod::MetisND),
        ("kkt70/amf", kkt_m, OrderingMethod::Amf),
    ];
    let threads = rayon::current_num_threads();
    for (name, m, method) in &cases {
        // Sequential driver.
        let (measured, s) = measure(m, method.clone(), false);
        let e = s.work_estimate().expect("estimate");
        let f = s.factors().expect("factors");
        if name.starts_with("kkt") {
            let inertia = s.inertia().expect("inertia");
            assert_eq!((inertia.positive, inertia.negative), (nv, nc), "{name}");
        }
        // Stored terms against the factor the numeric phase kept. Any
        // delayed pivot moves columns between fronts and is outside the
        // model, so only check exactness when there were none.
        let delayed: usize = f.node_factors.iter().map(|nf| nf.n_delayed_in).sum();
        if delayed == 0 {
            let values: usize = f
                .node_factors
                .iter()
                .map(|nf| nf.frontal_factors.l.len() * 8)
                .sum();
            assert_eq!(e.factor_stored_bytes, values, "{name}: stored values");
            let index: usize = f
                .node_factors
                .iter()
                .map(|nf| {
                    let ff = &nf.frontal_factors;
                    (nf.row_indices.len() + ff.perm.len() + ff.perm_inv.len()) * 8
                        + (ff.d_diag.len() + ff.d_subdiag.len()) * 8
                        + std::mem::size_of::<feral::numeric::factorize::NodeFactors>()
                })
                .sum();
            assert_eq!(e.factor_index_bytes, index, "{name}: per-node index data");
        }
        assert!(
            e.peak_bytes >= measured,
            "{name}: sequential peak_bytes {} below measured {}",
            e.peak_bytes,
            measured
        );
        eprintln!("{name}: seq measured {measured} predicted {} old {} ratio {:.2} old-ratio {:.2} delayed {delayed}", e.peak_bytes, e.factor_bytes + e.peak_contrib_bytes, e.peak_bytes as f64 / measured as f64, (e.factor_bytes + e.peak_contrib_bytes) as f64 / measured as f64);
        // And not so loose it is useless: within 2x of the real peak.
        assert!(
            e.peak_bytes <= 2 * measured,
            "{name}: sequential peak_bytes {} over 2x measured {}",
            e.peak_bytes,
            measured
        );

        // Parallel driver, the Solver's own pool size.
        let (measured_par, s) = measure(m, method.clone(), true);
        let e = s.work_estimate().expect("estimate");
        let predicted = e.peak_bytes_parallel(threads);
        assert!(
            predicted >= measured_par,
            "{name}: peak_bytes_parallel({threads}) {predicted} below measured {measured_par} \
             (frontier {:?})",
            e.parallel_frontier_bytes
        );
        eprintln!(
            "{name}: par measured {measured_par} predicted {predicted} ratio {:.2} frontier {:?}",
            predicted as f64 / measured_par as f64,
            e.parallel_frontier_bytes
        );
        assert!(predicted >= e.peak_bytes);
    }
}
