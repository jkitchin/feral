//! Phase 2.9.1 per-leaf cost profiler.
//!
//! Goal: for the archetype long-tail IPM matrices, break the per-leaf
//! numeric cost into phases and report mean/median/p99 ns for each:
//!
//!   1. row_map write (setup bookkeeping)
//!   2. frontal_buf memset (clear + resize(n*n, 0.0))
//!   3. A-scatter (permuted A into frontal, with scaling)
//!   4. factor_frontal_blocked (the BK kernel itself)
//!   5. contrib-block deposit (Vec alloc + copy)
//!   6. row_map teardown
//!
//! This is a diagnostic binary — it reimplements the leaf factorization
//! work using only public API (permutes A once up front, uses Identity
//! scaling since the scaling strategy doesn't affect the cost *shape*)
//! and wraps every phase in an Instant::now() timer. That adds a
//! small per-phase measurement overhead — the reported *ratios*
//! between phases are what matter.
//!
//! Usage: `cargo run --release --bin diag_leaf_profile`

use feral::dense::factor::{factor_frontal_with_profile, BunchKaufmanParams, FrontalProfile};
use feral::dense::matrix::SymmetricMatrix;
use feral::sparse::csc::CscMatrix;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, ZeroPivotAction};
use std::path::Path;
use std::time::Instant;

#[derive(Default, Clone)]
struct PhaseAcc {
    samples: Vec<u128>,
}

impl PhaseAcc {
    fn push(&mut self, ns: u128) {
        self.samples.push(ns);
    }
    fn summary(&self) -> (u128, u128, u128, u128) {
        if self.samples.is_empty() {
            return (0, 0, 0, 0);
        }
        let mut s = self.samples.clone();
        s.sort_unstable();
        let total: u128 = s.iter().sum();
        let mean = total / s.len() as u128;
        let median = s[s.len() / 2];
        let p99 = s[(s.len() * 99) / 100];
        (mean, median, p99, total)
    }
}

#[derive(Default)]
struct Profile {
    n_leaves: usize,
    memset: PhaseAcc,
    scatter: PhaseAcc,
    row_map_setup: PhaseAcc,
    row_map_teardown: PhaseAcc,
    bk: PhaseAcc,
    contrib: PhaseAcc,
    total: PhaseAcc,
    // Phase 2.9.2 Step A: sub-timing of `factor_frontal` internals.
    // Aggregates across the whole run as totals (not per-leaf means);
    // one call contributes to all four fields.
    kernel_profile: FrontalProfile,
}

/// Reimplementation of permute_csc_values that stays within public API.
/// Produces the lower-triangle-only permuted matrix the numeric phase
/// would see.
fn permute_csc_lower(matrix: &CscMatrix, perm_inv: &[usize]) -> CscMatrix {
    let n = matrix.n;
    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(matrix.row_idx.len());
    for old_j in 0..n {
        for k in matrix.col_ptr[old_j]..matrix.col_ptr[old_j + 1] {
            let old_i = matrix.row_idx[k];
            let new_j = perm_inv[old_j];
            let new_i = perm_inv[old_i];
            let val = matrix.values[k];
            if new_i >= new_j {
                triplets.push((new_i, new_j, val));
            } else {
                triplets.push((new_j, new_i, val));
            }
        }
    }
    let rows: Vec<usize> = triplets.iter().map(|t| t.0).collect();
    let cols: Vec<usize> = triplets.iter().map(|t| t.1).collect();
    let vals: Vec<f64> = triplets.iter().map(|t| t.2).collect();
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("from_triplets")
}

fn profile_matrix(label: &str, mtx_path: &str, prof: &mut Profile) {
    let mtx = match read_mtx(Path::new(mtx_path)) {
        Ok(m) => m,
        Err(e) => {
            println!("{:28} SKIP (read_mtx: {:?})", label, e);
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            println!("{:28} SKIP (to_csc: {:?})", label, e);
            return;
        }
    };

    let sym = match symbolic_factorize(&csc, &SupernodeParams::default()) {
        Ok(s) => s,
        Err(e) => {
            println!("{:28} SKIP (symbolic: {:?})", label, e);
            return;
        }
    };

    if sym.small_leaf_groups.is_empty() {
        println!("{:28} SKIP (no small_leaf groups)", label);
        return;
    }

    let bk_params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    let permuted = permute_csc_lower(&csc, &sym.perm_inv);
    // Identity scaling: same cost shape on the leaf path as any other
    // scaling strategy (scatter cost is proportional to nnz of the
    // own columns, independent of scaling values).
    let scaling: Vec<f64> = vec![1.0; csc.n];

    let n = csc.n;
    let mut row_map: Vec<usize> = vec![usize::MAX; n];
    let mut frontal_buf: Vec<f64> = Vec::new();

    const N_REPEAT: usize = 50;
    let mut n_leaves_this_matrix = 0usize;

    for group in &sym.small_leaf_groups {
        for (i, &_snode_idx) in group.members.iter().enumerate() {
            let snode = &sym.supernodes[group.members[i]];
            let own_ncol = snode.ncol;
            let row_indices = &group.member_rows[i];
            if row_indices.is_empty() || own_ncol == 0 {
                continue;
            }
            let actual_nrow = row_indices.len();
            n_leaves_this_matrix += 1;

            for _ in 0..N_REPEAT {
                let t_leaf = Instant::now();

                // Phase: row_map setup
                let t0 = Instant::now();
                for (local, &global) in row_indices.iter().enumerate() {
                    row_map[global] = local;
                }
                prof.row_map_setup.push(t0.elapsed().as_nanos());

                // Phase: memset
                let t0 = Instant::now();
                frontal_buf.clear();
                frontal_buf.resize(actual_nrow * actual_nrow, 0.0);
                prof.memset.push(t0.elapsed().as_nanos());

                let mut frontal = SymmetricMatrix {
                    n: actual_nrow,
                    data: std::mem::take(&mut frontal_buf),
                };

                // Phase: scatter
                let t0 = Instant::now();
                for (local_j, &gj) in row_indices[..own_ncol].iter().enumerate() {
                    let s_j = scaling[gj];
                    for k in permuted.col_ptr[gj]..permuted.col_ptr[gj + 1] {
                        let gi = permuted.row_idx[k];
                        let local_i = row_map[gi];
                        if local_i != usize::MAX {
                            let val = permuted.values[k] * scaling[gi] * s_j;
                            frontal.set(local_i, local_j, val);
                        }
                    }
                }
                prof.scatter.push(t0.elapsed().as_nanos());

                // Phase: BK kernel (with sub-phase profiling)
                let t0 = Instant::now();
                let ff = factor_frontal_with_profile(
                    &frontal,
                    own_ncol,
                    true,
                    &bk_params,
                    Some(&mut prof.kernel_profile),
                )
                .expect("factor_frontal_with_profile");
                prof.bk.push(t0.elapsed().as_nanos());

                frontal_buf = frontal.data;

                // Phase: contrib deposit simulation
                let t0 = Instant::now();
                if ff.contrib_dim > 0 {
                    let cdim = ff.contrib_dim;
                    let mut contrib_row_indices = Vec::with_capacity(cdim);
                    for cj in 0..cdim {
                        contrib_row_indices.push(row_indices[ff.perm[ff.nelim + cj]]);
                    }
                    let _contrib_copy = ff.contrib.clone();
                    let _ = contrib_row_indices;
                }
                prof.contrib.push(t0.elapsed().as_nanos());

                // Phase: row_map teardown
                let t0 = Instant::now();
                for &global in row_indices {
                    row_map[global] = usize::MAX;
                }
                prof.row_map_teardown.push(t0.elapsed().as_nanos());

                prof.total.push(t_leaf.elapsed().as_nanos());
            }
        }
    }

    prof.n_leaves += n_leaves_this_matrix;
    println!(
        "{:28} n_leaves={:>5} groups={:>4} (×{} repeats per leaf)",
        label,
        n_leaves_this_matrix,
        sym.small_leaf_groups.len(),
        N_REPEAT
    );
}

fn report(prof: &Profile) {
    println!("\n=== Per-leaf phase breakdown (nanoseconds) ===");
    println!(
        "{:18} {:>10} {:>10} {:>10} {:>14} {:>6}",
        "phase", "mean_ns", "median_ns", "p99_ns", "total_ns", "%tot"
    );
    let (_, _, _, total_total) = prof.total.summary();
    let phases = [
        ("row_map_setup", &prof.row_map_setup),
        ("memset", &prof.memset),
        ("scatter", &prof.scatter),
        ("bk_kernel", &prof.bk),
        ("contrib", &prof.contrib),
        ("row_map_teardown", &prof.row_map_teardown),
        ("total", &prof.total),
    ];
    for (name, acc) in phases {
        let (mean, median, p99, total) = acc.summary();
        let pct = if total_total > 0 {
            100.0 * total as f64 / total_total as f64
        } else {
            0.0
        };
        println!(
            "{:18} {:>10} {:>10} {:>10} {:>14} {:>5.1}%",
            name, mean, median, p99, total, pct
        );
    }

    let sum_phases: u128 = prof.row_map_setup.samples.iter().sum::<u128>()
        + prof.memset.samples.iter().sum::<u128>()
        + prof.scatter.samples.iter().sum::<u128>()
        + prof.bk.samples.iter().sum::<u128>()
        + prof.contrib.samples.iter().sum::<u128>()
        + prof.row_map_teardown.samples.iter().sum::<u128>();
    let residual = total_total.saturating_sub(sum_phases);
    println!(
        "\ntotal - sum(phases) = {} ns ({:.1}% measurement overhead)",
        residual,
        if total_total > 0 {
            100.0 * residual as f64 / total_total as f64
        } else {
            0.0
        }
    );

    // Phase 2.9.2 Step A sub-profile of factor_frontal internals.
    let kp = &prof.kernel_profile;
    let bk_total: u128 = prof.bk.samples.iter().sum();
    let inner_total = kp.alloc_copy_ns + kp.setup_ns + kp.pivot_loop_ns + kp.extract_ns;
    println!(
        "\n=== factor_frontal sub-phase totals (n_calls={}) ===",
        kp.n_calls
    );
    println!(
        "{:18} {:>14} {:>6} {:>6}",
        "sub-phase", "total_ns", "%bk", "%inner"
    );
    let row = |name: &str, ns: u128| {
        let pct_bk = if bk_total > 0 {
            100.0 * ns as f64 / bk_total as f64
        } else {
            0.0
        };
        let pct_inner = if inner_total > 0 {
            100.0 * ns as f64 / inner_total as f64
        } else {
            0.0
        };
        println!(
            "{:18} {:>14} {:>5.1}% {:>5.1}%",
            name, ns, pct_bk, pct_inner
        );
    };
    row("alloc+copy", kp.alloc_copy_ns);
    row("setup", kp.setup_ns);
    row("pivot_loop", kp.pivot_loop_ns);
    row("extract", kp.extract_ns);
    row("INNER_TOTAL", inner_total);
    row("BK_TOTAL (outer)", bk_total);

    // Step A rejection gate: the refactor targets alloc+copy + setup
    // (the removable Vec allocations). extract is also allocation-heavy
    // but requires a separate return-struct rework. If alloc_copy+setup
    // is less than 25% of bk_total the arena refactor won't clear the
    // 1.5× leaf speedup target.
    let removable = kp.alloc_copy_ns + kp.setup_ns;
    let pct_removable = if bk_total > 0 {
        100.0 * removable as f64 / bk_total as f64
    } else {
        0.0
    };
    println!(
        "\nalloc+copy + setup = {} ns ({:.1}% of bk_total)",
        removable, pct_removable
    );
    if pct_removable < 25.0 {
        println!(
            "!!! GATE FAIL: removable fraction {:.1}% < 25% — arena refactor unlikely to pay",
            pct_removable
        );
    } else {
        println!(
            "GATE PASS: removable fraction {:.1}% >= 25% — proceed with Step B+C",
            pct_removable
        );
    }
}

fn main() {
    println!("=== Phase 2.9.1 per-leaf phase profiler ===");

    let targets: &[(&str, &str)] = &[
        ("ACOPR30_0067", "data/matrices/kkt/ACOPR30/ACOPR30_0067.mtx"),
        (
            "CRESC100_0000",
            "data/matrices/kkt/CRESC100/CRESC100_0000.mtx",
        ),
        ("HAIFAM_0082", "data/matrices/kkt/HAIFAM/HAIFAM_0082.mtx"),
        ("VESUVIO_0000", "data/matrices/kkt/VESUVIO/VESUVIO_0000.mtx"),
    ];

    let mut agg = Profile::default();
    for (label, path) in targets {
        profile_matrix(label, path, &mut agg);
    }

    println!("\n--- AGGREGATED across {} leaves ---", agg.n_leaves);
    report(&agg);
}
