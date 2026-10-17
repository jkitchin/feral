//! Issue #153 (comment 6223093143): is feral's warm-refactor cost on a
//! hanging-chain IPM KKT platform-dependent on x86 Linux / glibc?
//!
//! Builds a synthetic 2-D hanging-chain KKT of the same shape as the
//! pounce#1020 dump (10k links, n ~ 40k, inertia (30002, 10004, 0)) and
//! measures, on a serial `Solver` doing warm refactors of unchanged values:
//!
//!   wall      min / median wall per refactor (profiling off)
//!   profile   `with_profiling` buckets, prologue breakdown, scaling route
//!   alloc     allocations and bytes per warm refactor (counting allocator)
//!   bump      A/B: glibc malloc vs a pointer-bump arena (no-op free),
//!             alternating, paired, per the decisions.md 2026-08-09 method
//!   once N    N warm refactors and exit (for `valgrind --tool=callgrind`)
//!
//! Usage: cargo run -p feral-diagnostics --release --bin diag_153_chain_x86 -- <mode> [N]
//!
//! mimalloc arm (not a workspace dependency; add it locally, do not commit):
//!   append `mimalloc = { version = "0.1", default-features = false }` to
//!   crates/feral-diagnostics/Cargo.toml, then
//!   RUSTFLAGS="--cfg feral_mi" CARGO_TARGET_DIR=target-mi \
//!     cargo build -p feral-diagnostics --release --bin diag_153_chain_x86
//! glibc tcache arm: GLIBC_TUNABLES=glibc.malloc.tcache_count=65535 <bin> wall
//!
//! Results: dev/journal/2026-10-16-01.org.

// `feral_mi` is a local opt-in cfg (see above), not a declared feature.
#![allow(unexpected_cfgs)]

use feral::{CscMatrix, FactorStatus, Inertia, Solver};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------- allocator

static COUNTING: AtomicBool = AtomicBool::new(false);
static BUMP: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static ARENA_BASE: AtomicUsize = AtomicUsize::new(0);
static ARENA_LEN: AtomicUsize = AtomicUsize::new(0);
static ARENA_OFF: AtomicUsize = AtomicUsize::new(0);

struct ProbeAlloc;

// Underlying allocator: glibc by default; mimalloc when built with
// RUSTFLAGS="--cfg feral_mi" (a reuse-preserving fast allocator, the
// control the bump arena cannot be).
#[cfg(feral_mi)]
static UNDER: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[cfg(not(feral_mi))]
static UNDER: System = System;

// SAFETY: every path either forwards to `System` (a valid GlobalAlloc) or
// hands out a disjoint, suitably aligned slice of an arena that was itself
// obtained from `System` and is never freed. Arena frees are no-ops, so no
// pointer is ever handed out twice. `UNDER` is System or MiMalloc,
// both valid GlobalAllocs. `realloc` uses the trait default
// (alloc + copy + dealloc), which routes through these methods.
unsafe impl GlobalAlloc for ProbeAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        if BUMP.load(Ordering::Relaxed) {
            let base = ARENA_BASE.load(Ordering::Relaxed);
            let len = ARENA_LEN.load(Ordering::Relaxed);
            let align = layout.align().max(16);
            let off = ARENA_OFF.load(Ordering::Relaxed);
            let start = (base + off + align - 1) & !(align - 1);
            let end = start + layout.size();
            if end <= base + len {
                ARENA_OFF.store(end - base, Ordering::Relaxed);
                return start as *mut u8;
            }
        }
        UNDER.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let base = ARENA_BASE.load(Ordering::Relaxed);
        let len = ARENA_LEN.load(Ordering::Relaxed);
        let p = ptr as usize;
        if base != 0 && p >= base && p < base + len {
            return;
        }
        UNDER.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: ProbeAlloc = ProbeAlloc;

fn init_arena(bytes: usize) {
    let layout = match Layout::from_size_align(bytes, 4096) {
        Ok(l) => l,
        Err(_) => return,
    };
    // SAFETY: non-zero size, valid layout; the arena is intentionally leaked.
    let p = unsafe { System.alloc(layout) };
    if p.is_null() {
        eprintln!("arena alloc failed");
        std::process::exit(1);
    }
    // Pre-fault every page so the bump arm pays no page faults the
    // malloc arm does not.
    for off in (0..bytes).step_by(4096) {
        // SAFETY: off < bytes, inside the allocation.
        unsafe { p.add(off).write_volatile(0) };
    }
    ARENA_BASE.store(p as usize, Ordering::Relaxed);
    ARENA_LEN.store(bytes, Ordering::Relaxed);
}

// ---------------------------------------------------------------- matrix

/// 2-D hanging chain, N links, nodes 0..=N, variables
/// [x_0..x_N, y_0..y_N, s_0..s_{N-1}] (s = per-link slack with a barrier
/// diagonal), constraints [link_0..link_{N-1}, x_0, y_0, x_N, y_N].
/// Lower triangle of [[H + dx I, J^T], [J, -dc I]].
fn chain_kkt(nlinks: usize) -> (CscMatrix, Inertia) {
    let nn = nlinks + 1;
    let nx = 2 * nn + nlinks; // primal
    let nc = nlinks + 4; // constraints
    let n = nx + nc;
    let xi = |i: usize| i;
    let yi = |i: usize| nn + i;
    let si = |j: usize| 2 * nn + j;
    let ci = |j: usize| nx + j;

    // Deterministic sagging-chain geometry and multipliers.
    let mut lcg: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let px: Vec<f64> = (0..nn).map(|i| i as f64 / nlinks as f64).collect();
    let py: Vec<f64> = (0..nn)
        .map(|i| {
            let t = i as f64 / nlinks as f64;
            ((t - 0.5) * 4.0).cosh() * 0.2
        })
        .collect();
    let dx = 1e-8;
    let dc = 1e-8;

    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let mut push = |i: usize, j: usize, a: f64| {
        let (i, j) = if i >= j { (i, j) } else { (j, i) };
        r.push(i);
        c.push(j);
        v.push(a);
    };
    // Hessian of the Lagrangian: lambda_j * [[2,-2],[-2,2]] per coordinate.
    let mut hdiag = vec![dx; 2 * nn];
    for j in 0..nlinks {
        let lam = 0.5 + rnd();
        for off in [0usize, nn] {
            hdiag[off + j] += 2.0 * lam;
            hdiag[off + j + 1] += 2.0 * lam;
            push(off + j + 1, off + j, -2.0 * lam);
        }
    }
    for (k, &h) in hdiag.iter().enumerate() {
        push(k, k, h);
    }
    for j in 0..nlinks {
        // barrier diagonal on the slack, spread over decades
        push(si(j), si(j), 10f64.powf(-2.0 + 4.0 * rnd()));
        let ddx = px[j + 1] - px[j];
        let ddy = py[j + 1] - py[j];
        push(ci(j), xi(j), -2.0 * ddx);
        push(ci(j), xi(j + 1), 2.0 * ddx);
        push(ci(j), yi(j), -2.0 * ddy);
        push(ci(j), yi(j + 1), 2.0 * ddy);
        push(ci(j), si(j), -1.0);
        push(ci(j), ci(j), -dc);
    }
    for (k, var) in [xi(0), yi(0), xi(nlinks), yi(nlinks)]
        .into_iter()
        .enumerate()
    {
        push(ci(nlinks + k), var, 1.0);
        push(ci(nlinks + k), ci(nlinks + k), -dc);
    }
    let m = match CscMatrix::from_triplets(n, &r, &c, &v) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("from_triplets: {e:?}");
            std::process::exit(1);
        }
    };
    let inertia = Inertia {
        positive: nx,
        negative: nc,
        zero: 0,
    };
    (m, inertia)
}

fn solver() -> Solver {
    Solver::new().with_parallel(false)
}

fn factor(s: &mut Solver, m: &CscMatrix, expect: &Inertia) {
    match s.factor(m, Some(expect.clone())) {
        FactorStatus::Success => {}
        other => {
            eprintln!("factor failed: {other:?}");
            std::process::exit(1);
        }
    }
}

/// (minor faults, utime ticks, stime ticks) from /proc/self/stat.
fn proc_stat() -> (u64, u64, u64) {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // Fields after the ")" that closes comm: state is field 3.
    let rest = s.rsplit_once(')').map(|x| x.1).unwrap_or("");
    let f: Vec<u64> = rest
        .split_whitespace()
        .map(|x| x.parse().unwrap_or(0))
        .collect();
    // rest[0] = state (field 3) -> minflt field 10 = idx 7, utime 14 = idx 11, stime 15 = idx 12
    let g = |i: usize| f.get(i).copied().unwrap_or(0);
    (g(7), g(11), g(12))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Corpus mode: warm refactors of one .mtx file (or `chain` for the
/// generated hanging chain) for ~1 s, then one CSV line:
/// `name,n,nnz,supernodes,nnz_l,pos,neg,zero,reps,min_us,median_us`.
/// Inertia is not checked against an oracle; it is printed so arms can be
/// compared for bit-level agreement.
fn run_mtx(path: &str) {
    let m = if path == "chain" {
        chain_kkt(10_000).0
    } else {
        let parsed = feral::read_mtx(std::path::Path::new(path)).and_then(|x| x.to_csc());
        match parsed {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{path}: {e:?}");
                std::process::exit(3);
            }
        }
    };
    let mut s = solver();
    for _ in 0..2 {
        if let FactorStatus::FatalError(e) = s.factor(&m, None) {
            eprintln!("{path}: {e:?}");
            std::process::exit(4);
        }
    }
    let mut t = Vec::new();
    let start = Instant::now();
    while (start.elapsed().as_secs_f64() < 1.0 || t.len() < 15) && t.len() < 5000 {
        let t0 = Instant::now();
        let _ = s.factor(&m, None);
        t.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let min = t.iter().cloned().fold(f64::INFINITY, f64::min);
    let reps = t.len();
    let med = median(&mut t);
    let (nnz_l, inertia) = s
        .last_factor_stats()
        .map(|st| (st.nnz_l, st.inertia))
        .unwrap_or((
            0,
            Inertia {
                positive: 0,
                negative: 0,
                zero: 0,
            },
        ));
    let nsn = s.symbolic().map(|x| x.supernodes.len()).unwrap_or(0);
    let name = path.rsplit('/').next().unwrap_or(path);
    println!(
        "{name},{},{},{nsn},{nnz_l},{},{},{},{reps},{min:.2},{med:.2}",
        m.n,
        m.nnz(),
        inertia.positive,
        inertia.negative,
        inertia.zero
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("wall");
    if mode == "mtx" {
        run_mtx(args.get(2).map(String::as_str).unwrap_or(""));
        return;
    }
    let count: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let (m, inertia) = chain_kkt(10_000);
    eprintln!(
        "n = {}, nnz(lower) = {}, expected inertia {:?}",
        m.n,
        m.nnz(),
        inertia
    );

    let mut s = solver();
    factor(&mut s, &m, &inertia); // cold: analysis + first numeric
    factor(&mut s, &m, &inertia); // first warm

    match mode {
        "once" => {
            for _ in 0..count.max(1) {
                factor(&mut s, &m, &inertia);
            }
        }
        "wall" => {
            let reps = if count == 0 { 200 } else { count };
            let mut t = Vec::with_capacity(reps);
            let p0 = proc_stat();
            for _ in 0..reps {
                let t0 = Instant::now();
                factor(&mut s, &m, &inertia);
                t.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            let min = t.iter().cloned().fold(f64::INFINITY, f64::min);
            let p1 = proc_stat();
            println!(
                "warm refactor: min {:.3} ms, median {:.3} ms ({} reps)",
                min,
                median(&mut t),
                reps
            );
            println!(
                "per refactor: {:.0} minor faults, user {:.2} ms, sys {:.2} ms",
                (p1.0 - p0.0) as f64 / reps as f64,
                (p1.1 - p0.1) as f64 * 10.0 / reps as f64,
                (p1.2 - p0.2) as f64 * 10.0 / reps as f64
            );
            if let Some(st) = s.last_factor_stats() {
                println!(
                    "nnz_l {} inertia {:?} scaling {:?} ordering {:?}",
                    st.nnz_l, st.inertia, st.scaling_info, st.ordering_info
                );
            }
            if let Some(sym) = s.symbolic() {
                let maxrow = sym.supernodes.iter().map(|sn| sn.nrow).max().unwrap_or(0);
                println!(
                    "supernodes {} max front nrow {}",
                    sym.supernodes.len(),
                    maxrow
                );
            }
            println!("mc64_cache_hit_count {}", s.mc64_cache_hit_count());
        }
        "profile" => {
            let mut p = Solver::new().with_parallel(false).with_profiling(true);
            factor(&mut p, &m, &inertia);
            let mut best: Option<feral::ProfileReport> = None;
            for _ in 0..if count == 0 { 21 } else { count } {
                factor(&mut p, &m, &inertia);
                if let Some(r) = p.profile_report() {
                    if best
                        .as_ref()
                        .map(|b| r.total_us < b.total_us)
                        .unwrap_or(true)
                    {
                        best = Some(r);
                    }
                }
            }
            if let Some(r) = best {
                println!(
                    "min-total run: total {} us, prologue {} us, loop {:.0} us, epilogue {} us, supernodes {}",
                    r.total_us,
                    r.prologue_us,
                    r.loop_ns as f64 / 1e3,
                    r.epilogue_us,
                    r.n_supernodes
                );
                println!("prologue breakdown: {:?}", r.prologue_breakdown);
                for b in &r.buckets {
                    println!(
                        "  {:>8} count {:>6} loop% {:>5.1} avg_ns {:>9.0}",
                        b.range, b.count, b.pct_of_total, b.avg_ns
                    );
                }
            }
            // Timer floor on this host.
            let reps = 1_000_000;
            let t0 = Instant::now();
            let mut acc = 0u128;
            for _ in 0..reps {
                let a = Instant::now();
                acc += a.elapsed().as_nanos();
            }
            println!(
                "Instant::now() pair cost: {:.1} ns (acc {})",
                t0.elapsed().as_nanos() as f64 / reps as f64,
                acc % 7
            );
        }
        "alloc" => {
            ALLOCS.store(0, Ordering::Relaxed);
            BYTES.store(0, Ordering::Relaxed);
            COUNTING.store(true, Ordering::Relaxed);
            factor(&mut s, &m, &inertia);
            COUNTING.store(false, Ordering::Relaxed);
            let a = ALLOCS.load(Ordering::Relaxed);
            let b = BYTES.load(Ordering::Relaxed);
            let nsn = s.symbolic().map(|x| x.supernodes.len()).unwrap_or(1);
            println!(
                "per warm refactor: {} allocs ({:.2}/supernode), {:.1} MB requested",
                a,
                a as f64 / nsn as f64,
                b as f64 / 1e6
            );
        }
        "bump" => {
            // Size the arena from one counted refactor.
            COUNTING.store(true, Ordering::Relaxed);
            BYTES.store(0, Ordering::Relaxed);
            factor(&mut s, &m, &inertia);
            COUNTING.store(false, Ordering::Relaxed);
            let per = BYTES.load(Ordering::Relaxed) as usize;
            let pairs = if count == 0 { 60 } else { count };
            init_arena(per * (pairs + 2) + (64 << 20));
            let (mut tm, mut tb) = (Vec::new(), Vec::new());
            let mut bump_wins = 0;
            for _ in 0..pairs {
                let t0 = Instant::now();
                factor(&mut s, &m, &inertia);
                let a = t0.elapsed().as_secs_f64() * 1e3;
                BUMP.store(true, Ordering::Relaxed);
                let t0 = Instant::now();
                factor(&mut s, &m, &inertia);
                let b = t0.elapsed().as_secs_f64() * 1e3;
                BUMP.store(false, Ordering::Relaxed);
                if b < a {
                    bump_wins += 1;
                }
                tm.push(a);
                tb.push(b);
            }
            let mm = tm.iter().cloned().fold(f64::INFINITY, f64::min);
            let mb = tb.iter().cloned().fold(f64::INFINITY, f64::min);
            println!(
                "glibc min {:.3} med {:.3} ms | bump min {:.3} med {:.3} ms | bump/glibc min {:.3} | bump wins {}/{}",
                mm,
                median(&mut tm),
                mb,
                median(&mut tb),
                mb / mm,
                bump_wins,
                pairs
            );
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    }
}
