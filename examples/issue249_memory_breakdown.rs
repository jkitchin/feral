//! Issue #249: break down the gap between `WorkEstimate::peak_bytes` and
//! the real high-water mark of `Solver::factor`.
//!
//! A counting global allocator tracks live and peak heap bytes. The probe
//! reads the matrix, resets the peak, runs one `Solver::factor`, and then
//! walks the retained structures (input, symbolic analysis, factors) by
//! `capacity()` so each term of the high-water can be named.
//!
//! Usage:
//!   cargo run --release --example issue249_memory_breakdown -- \
//!       <matrix.mtx> [amf|amd|metis|auto] [seq|par] [predict]
//!
//! `predict` prints the model from the analysis alone and skips the
//! numeric phase.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use feral::io::mtx::read_mtx;
use feral::symbolic::OrderingMethod;
use feral::{FactorStatus, Solver};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static GPEAK: AtomicUsize = AtomicUsize::new(0);
static N_ALLOC: AtomicUsize = AtomicUsize::new(0);

fn bump(sz: usize) {
    let now = LIVE.fetch_add(sz, Ordering::Relaxed) + sz;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's layout
// unchanged; the counters are side bookkeeping and never touch memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // SAFETY: forwarded with the caller's (valid) layout.
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            bump(l.size());
            N_ALLOC.fetch_add(1, Ordering::Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        // SAFETY: forwarded with the caller's (valid) layout.
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            bump(l.size());
            N_ALLOC.fetch_add(1, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        // SAFETY: `p` was allocated by `System` with layout `l`.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        // SAFETY: forwarded unchanged; `p` came from `System` with `l`.
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            // Count a moving realloc's transient double occupancy as the
            // allocator does not tell us whether it moved; charge the
            // growth only (a lower bound on the transient).
            if new >= l.size() {
                bump(new - l.size());
            } else {
                LIVE.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn gb(b: usize) -> f64 {
    b as f64 / 1e9
}

fn vm_hwm() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM"))
                .map(|l| l.to_string())
        })
        .unwrap_or_default()
}

fn vbytes<T>(v: &Vec<T>) -> usize {
    v.capacity() * std::mem::size_of::<T>()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("primal2.mtx");
    let ord = args.get(2).map(String::as_str).unwrap_or("amf");
    let par = args.get(3).map(String::as_str).unwrap_or("seq") == "par";
    let method = match ord {
        "amd" => OrderingMethod::Amd,
        "metis" => OrderingMethod::MetisND,
        "auto" => OrderingMethod::Auto,
        _ => OrderingMethod::Amf,
    };

    let a = {
        let m = read_mtx(Path::new(path)).expect("read");
        m.to_csc().expect("csc")
    };
    let input = vbytes(&a.col_ptr) + vbytes(&a.row_idx) + vbytes(&a.values);
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    N_ALLOC.store(0, Ordering::Relaxed);
    println!(
        "matrix n={} nnz_stored={} input CSC {:.3} GB (live before factor {:.3} GB)",
        a.n,
        a.row_idx.len(),
        gb(input),
        gb(base)
    );

    if args.get(4).map(String::as_str) == Some("predict") {
        // Analysis only: print the model without the numeric phase.
        let sym = feral::symbolic::symbolic_factorize_with_method(
            &a,
            &feral::symbolic::SupernodeParams::default(),
            method,
        )
        .expect("symbolic");
        let w = sym.work_estimate();
        let t = std::thread::available_parallelism().map_or(1, |n| n.get());
        println!(
            "old {:.3} | peak_bytes {:.3} | peak_bytes_parallel({t}) {:.3} GB | frontier {:?}",
            gb(w.factor_bytes + w.peak_contrib_bytes),
            gb(w.peak_bytes),
            gb(w.peak_bytes_parallel(t)),
            w.parallel_frontier_bytes.map(gb)
        );
        println!(
            "  stored {:.3} index {:.3} analysis {:.3} workspace {:.3} shared {:.3} worker {:.3} \
             running[0..5] {:?}",
            gb(w.factor_stored_bytes),
            gb(w.factor_index_bytes),
            gb(w.analysis_bytes),
            gb(w.workspace_bytes),
            gb(w.parallel_shared_bytes),
            gb(w.parallel_worker_bytes),
            w.parallel_running_bytes
                .iter()
                .take(5)
                .map(|&b| gb(b))
                .collect::<Vec<_>>()
        );
        return;
    }
    let mut s = Solver::new().with_ordering(method).with_parallel(par);
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let d2 = done.clone();
    let sampler = std::thread::spawn(move || {
        let t0 = Instant::now();
        let mut last = 0usize;
        while !d2.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(250));
            let pk = PEAK.swap(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
            if pk > last + 100_000_000 || pk + 100_000_000 < last {
                eprintln!(
                    "  t={:6.1}s window-peak {:.3} GB",
                    t0.elapsed().as_secs_f64(),
                    gb(pk)
                );
                last = pk;
            }
            GPEAK.fetch_max(pk, Ordering::Relaxed);
        }
    });
    let t = Instant::now();
    let st = s.factor(&a, None);
    done.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    GPEAK.fetch_max(PEAK.load(Ordering::Relaxed), Ordering::Relaxed);
    let secs = t.elapsed().as_secs_f64();
    let ok = matches!(st, FactorStatus::Success);
    println!("factor: {:?}-ish ok={} in {:.1} s", ord, ok, secs);
    let peak = GPEAK.load(Ordering::Relaxed);
    let live_after = LIVE.load(Ordering::Relaxed);

    let w = s.work_estimate().expect("work estimate");
    println!(
        "WorkEstimate: factor_bytes {:.3} GB  peak_contrib {:.3} GB  peak_bytes {:.3} GB  \
         n_snodes {}  max_front_rows {}",
        gb(w.factor_bytes),
        gb(w.peak_contrib_bytes),
        gb(w.peak_bytes),
        w.n_supernodes,
        w.max_front_rows
    );
    let threads = if par {
        std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
    } else {
        0
    };
    println!(
        "model: stored {:.3} + index {:.3} | analysis {:.3} | workspace {:.3} | \
         numeric_peak {:.3} | frontier {:?} worker {:.3} running(5) {:?} shared {:.3}",
        gb(w.factor_stored_bytes),
        gb(w.factor_index_bytes),
        gb(w.analysis_bytes),
        gb(w.workspace_bytes),
        gb(w.numeric_peak_bytes),
        w.parallel_frontier_bytes.map(gb),
        gb(w.parallel_worker_bytes),
        w.parallel_running_bytes.get(4).copied().map(gb),
        gb(w.parallel_shared_bytes)
    );
    let predicted = if par {
        w.peak_bytes_parallel(threads)
    } else {
        w.peak_bytes
    };
    println!(
        "PREDICTED {:.3} GB ({}) vs MEASURED heap peak-input {:.3} GB: ratio {:.2}",
        gb(predicted),
        if par {
            format!("parallel, {threads} threads")
        } else {
            "sequential".to_string()
        },
        gb(peak - base),
        predicted as f64 / (peak - base) as f64
    );
    println!(
        "heap: peak {:.3} GB (incl. input {:.3})  peak-input {:.3} GB  retained after {:.3} GB  \
         allocations {}",
        gb(peak),
        gb(base),
        gb(peak - base),
        gb(live_after - base),
        N_ALLOC.load(Ordering::Relaxed)
    );
    println!("{}", vm_hwm());

    // ---- retained structures -------------------------------------------
    if let Some(sym) = s.symbolic() {
        let mut sn_rows = 0usize;
        let mut sn_children = 0usize;
        for sn in &sym.supernodes {
            sn_rows += vbytes(&sn.row_indices);
            sn_children += vbytes(&sn.children);
        }
        let sn_struct = vbytes(&sym.supernodes);
        let pat = vbytes(&sym.permuted_pattern.col_ptr) + vbytes(&sym.permuted_pattern.row_idx);
        let perm = vbytes(&sym.perm) + vbytes(&sym.perm_inv);
        let etc = vbytes(&sym.contrib_sizes)
            + vbytes(&sym.col_counts)
            + vbytes(&sym.snode_group)
            + vbytes(&sym.static_row_offsets)
            + vbytes(&sym.static_row_flat);
        let mut rect = 0usize;
        let mut tri = 0usize;
        for sn in &sym.supernodes {
            rect += sn.nrow * sn.ncol;
            tri += sn.ncol * (sn.ncol + 1) / 2;
        }
        let cc: usize = sym.col_counts.iter().sum();
        println!(
            "symbolic structure: sum col_counts {} | sum nrow*ncol {} | trapezoid {} | \
             factor_nnz_estimate {}",
            cc,
            rect,
            rect - tri + sym.supernodes.iter().map(|s| s.ncol).sum::<usize>(),
            sym.factor_nnz_estimate
        );
        // Largest antichain of contribution blocks (square, bytes): a
        // bound on what any schedule can hold completed-but-unassembled.
        let ns = sym.supernodes.len();
        let mut best = vec![0usize; ns];
        for k in 0..ns {
            let sub: usize = sym.supernodes[k].children.iter().map(|&c| best[c]).sum();
            best[k] = sub.max(sym.contrib_sizes[k]);
        }
        let mut is_child = vec![false; ns];
        for sn in &sym.supernodes {
            for &c in &sn.children {
                is_child[c] = true;
            }
        }
        let anti: usize = (0..ns).filter(|&k| !is_child[k]).map(|k| best[k]).sum();
        let maxc = sym.contrib_sizes.iter().copied().max().unwrap_or(0);
        println!(
            "contrib model: postorder square peak {:.3} GB | max antichain {:.3} GB | \
             max single block {:.3} GB",
            gb(sym.peak_contrib_bytes),
            gb(anti * 8),
            gb(maxc * 8)
        );
        println!("symbolic (retained):");
        println!("  supernodes Vec<Supernode>   {:.3} GB", gb(sn_struct));
        println!("  supernode row_indices       {:.3} GB", gb(sn_rows));
        println!("  supernode children          {:.3} GB", gb(sn_children));
        println!("  permuted_pattern            {:.3} GB", gb(pat));
        println!("  perm + perm_inv             {:.3} GB", gb(perm));
        println!(
            "  static rows/counts/etc      {:.3} GB (static_row_flat {:.3})",
            gb(etc),
            gb(vbytes(&sym.static_row_flat))
        );
        println!(
            "  sum (excl. etree, mc64)     {:.3} GB",
            gb(sn_struct + sn_rows + sn_children + pat + perm + etc)
        );
    }
    if let Some(f) = s.factors() {
        let mut l_len = 0usize;
        let mut l_cap = 0usize;
        let mut d = 0usize;
        let mut p = 0usize;
        let mut rows = 0usize;
        let mut contrib = 0usize;
        let mut tri = 0usize;
        for nf in &f.node_factors {
            let ff = &nf.frontal_factors;
            l_len += ff.l.len() * 8;
            l_cap += vbytes(&ff.l);
            d += vbytes(&ff.d_diag) + vbytes(&ff.d_subdiag);
            p += vbytes(&ff.perm) + vbytes(&ff.perm_inv);
            rows += vbytes(&nf.row_indices);
            contrib += vbytes(&ff.contrib);
            // strict upper triangle of the nelim x nelim diagonal block
            tri += ff.nelim * ff.nelim.saturating_sub(1) / 2 * 8;
        }
        let nf_struct = vbytes(&f.node_factors);
        println!("factors (retained):");
        println!(
            "  L values len {:.3} GB, capacity {:.3} GB (upper-tri pad {:.3} GB)",
            gb(l_len),
            gb(l_cap),
            gb(tri)
        );
        println!("  D diag+subdiag              {:.3} GB", gb(d));
        println!("  BK perm + perm_inv          {:.3} GB", gb(p));
        println!("  node row_indices            {:.3} GB", gb(rows));
        println!("  leftover contrib Vecs       {:.3} GB", gb(contrib));
        println!(
            "  Vec<NodeFactors> ({} x {} B) {:.3} GB",
            f.node_factors.len(),
            std::mem::size_of::<feral::numeric::factorize::NodeFactors>(),
            gb(nf_struct)
        );
        println!(
            "  perm/perm_inv/scaling       {:.3} GB",
            gb(vbytes(&f.perm) + vbytes(&f.perm_inv) + vbytes(&f.scaling))
        );
        println!(
            "  sum                         {:.3} GB",
            gb(l_cap
                + d
                + p
                + rows
                + contrib
                + nf_struct
                + vbytes(&f.perm)
                + vbytes(&f.perm_inv)
                + vbytes(&f.scaling))
        );
    }
}
