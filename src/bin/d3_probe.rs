//! D.3 diagnostic probe — time each phase of the dense fast-path on
//! TRO3X3_0013 and a few synthetic in-gate matrices, so we can see
//! whether `dense_fast_factor` is bottlenecked on `compute_scaling`,
//! `CscMatrix::to_dense`, the in-place scaling loop, `factor_frontal`,
//! or the synthesis step.
//!
//! Stage-1 requirement per `dev/plans/sparse-tail-d3.md`.

use feral::dense::factor::factor_frontal;
use feral::numeric::factorize::{dense_fast_factor, factorize_multifrontal_supernodal};
use feral::scaling::compute_scaling;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, NumericParams, ZeroPivotAction};
use std::path::PathBuf;
use std::time::Instant;

const N_ITERS: u32 = 1000;

fn params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    })
}

fn median_ns<F: FnMut() -> u128>(mut f: F) -> u128 {
    let mut samples: Vec<u128> = (0..N_ITERS).map(|_| f()).collect();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn time_phases(label: &str, csc: &CscMatrix) {
    let p = params();
    let n = csc.n;
    let nnz_lower = csc.row_idx.len();

    // Phase 1: scaling only.
    let scale_ns = median_ns(|| {
        let t = Instant::now();
        let _ = compute_scaling(csc, &p.scaling).expect("scaling");
        t.elapsed().as_nanos()
    });

    // Phase 2: densify only (CscMatrix::to_dense).
    let densify_ns = median_ns(|| {
        let t = Instant::now();
        let _sym = csc.to_dense();
        t.elapsed().as_nanos()
    });

    // Phase 3: scaling-apply in place on the dense matrix.
    // Time the loop alone; compute_scaling and to_dense are amortised
    // by a prelude outside the measured region.
    let (s_vec, _info) = compute_scaling(csc, &p.scaling).expect("scaling");
    let scale_apply_ns = median_ns(|| {
        let mut sym = csc.to_dense();
        let t = Instant::now();
        for (j, &s_j) in s_vec.iter().enumerate() {
            let col = j * n;
            for (i, &s_i) in s_vec.iter().enumerate().skip(j) {
                sym.data[col + i] *= s_i * s_j;
            }
        }
        t.elapsed().as_nanos()
    });

    // Phase 4: factor_frontal only (on the pre-scaled dense matrix).
    let mut sym_prescaled = csc.to_dense();
    for (j, &s_j) in s_vec.iter().enumerate() {
        let col = j * n;
        for (i, &s_i) in s_vec.iter().enumerate().skip(j) {
            sym_prescaled.data[col + i] *= s_i * s_j;
        }
    }
    let factor_ns = median_ns(|| {
        let t = Instant::now();
        let _ff = factor_frontal(&sym_prescaled, n, false, &p.bk).expect("frontal");
        t.elapsed().as_nanos()
    });

    // Phase 5: end-to-end dense_fast_factor.
    let dense_total_ns = median_ns(|| {
        let t = Instant::now();
        let _r = dense_fast_factor(csc, &p).expect("dense fast");
        t.elapsed().as_nanos()
    });

    // Oracle: multifrontal path for comparison.
    let sn = SupernodeParams::default();
    let sym_fac = symbolic_factorize(csc, &sn).expect("symbolic");
    let multi_ns = median_ns(|| {
        let t = Instant::now();
        let _r = factorize_multifrontal_supernodal(csc, &sym_fac, &p).expect("multi");
        t.elapsed().as_nanos()
    });

    // Derived: synthesis overhead (end-to-end minus sum of measured phases).
    let measured_sum = scale_ns + densify_ns + scale_apply_ns + factor_ns;
    let synthesis_ns = dense_total_ns.saturating_sub(measured_sum);

    println!(
        "{:<18} n={:>4} nnz={:>6}  scale={:>6.1}  dens={:>6.1}  scapp={:>6.1}  fac={:>6.1}  synth={:>6.1}  dense_total={:>6.1}  multi={:>6.1}",
        label,
        n,
        nnz_lower,
        scale_ns as f64 / 1000.0,
        densify_ns as f64 / 1000.0,
        scale_apply_ns as f64 / 1000.0,
        factor_ns as f64 / 1000.0,
        synthesis_ns as f64 / 1000.0,
        dense_total_ns as f64 / 1000.0,
        multi_ns as f64 / 1000.0,
    );
}

fn load_csc(family: &str, sample: &str) -> CscMatrix {
    let path = PathBuf::from(format!(
        "data/matrices/kkt/{}/{}{}.mtx",
        family, family, sample
    ));
    let mtx = read_mtx(&path).expect("read_mtx");
    mtx.to_csc().expect("to_csc")
}

fn synth_dense(n: usize, density: f64) -> CscMatrix {
    // Lower-triangle symmetric KKT-ish: diagonally dominant, first
    // `density * lower_cells` entries populated deterministically.
    let lower_cells = n * (n + 1) / 2;
    let target = (density * lower_cells as f64) as usize;
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();
    col_ptr.push(0);
    let mut placed = 0usize;
    for j in 0..n {
        // Always place the diagonal first.
        row_idx.push(j);
        values.push(10.0 * (n as f64));
        placed += 1;
        // Then densify below in a deterministic stride until we hit target.
        for i in (j + 1)..n {
            if placed >= target {
                break;
            }
            row_idx.push(i);
            values.push(1.0 + 0.1 * ((i - j) as f64));
            placed += 1;
        }
        col_ptr.push(row_idx.len());
    }
    CscMatrix {
        n,
        col_ptr,
        row_idx,
        values,
    }
}

fn main() {
    println!("D.3 phase probe (us, median of {} iters)", N_ITERS);
    println!(
        "{:<18} {:>6} {:>6}  {:<10}  {:<10}  {:<10}  {:<10}  {:<10}  {:<12}  {:<10}",
        "case", "", "", "scale", "densify", "scale-app", "frontal", "synth", "dense-total", "multi"
    );
    println!("{}", "-".repeat(140));

    // Real target.
    time_phases("TRO3X3_0013", &load_csc("TRO3X3", "_0013"));
    println!();

    // Synthetic sweep: n x density at the in-gate region + one out.
    println!("Synthetic in-gate and boundary sweep:");
    for &n in &[32usize, 64, 96, 128, 160, 192] {
        for &d in &[0.1_f64, 0.25, 0.5, 0.75] {
            let csc = synth_dense(n, d);
            let label = format!("n={:<3} d={:.2}", n, d);
            time_phases(&label, &csc);
        }
    }
}
