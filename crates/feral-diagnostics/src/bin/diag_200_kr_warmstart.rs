//! Issue #200 (adjacent finding) — Knight-Ruiz scaling is recomputed
//! from `d = 1` on every `factor()` call, and on IPM-KKT matrices it
//! never converges: `diag_200_kr_iters` shows robot_1600,
//! steering_12800 and ex4_2_160 all burn the full 10-sweep cap. On
//! steering that is 18% of real factor time (samply, uninstrumented).
//!
//! In an interior-point loop consecutive KKT matrices differ only in the
//! barrier terms, so the previous iterate's scaling should be a good
//! starting point. This measures whether warm-starting buys sweeps:
//! for each matrix in a sequence it reports the convergence measure
//! `max_i |rowmax_i - 1|` reached by a cold start after 10 sweeps versus
//! a warm start (from the previous matrix's converged `d`) after 1, 2
//! and 3 sweeps.
//!
//! A warm start is only interesting if few sweeps reach a measure at
//! least as good as cold-10; the scaling vector feeds every pivot
//! decision downstream, so a worse-conditioned `D` is not a speedup.
use feral::read_mtx;
use feral::sparse::csc::CscMatrix;

/// One Knight-Ruiz sweep. Returns `max_i |rowmax_i - 1|`.
fn kr_sweep(m: &CscMatrix, d: &mut [f64], row_max: &mut [f64]) -> f64 {
    let n = m.n;
    for r in row_max.iter_mut() {
        *r = 0.0;
    }
    for j in 0..n {
        let dj = d[j];
        let mut col_max = row_max[j];
        for k in m.col_ptr[j]..m.col_ptr[j + 1] {
            let i = m.row_idx[k];
            let v = (d[i] * m.values[k] * dj).abs();
            if i != j && v > row_max[i] {
                row_max[i] = v;
            }
            if v > col_max {
                col_max = v;
            }
        }
        row_max[j] = col_max;
    }
    let mut max_dev = 0.0f64;
    for i in 0..n {
        let mx = row_max[i];
        if mx > 0.0 {
            d[i] /= mx.sqrt();
            let dev = (mx - 1.0).abs();
            if dev > max_dev {
                max_dev = dev;
            }
        }
    }
    max_dev
}

fn run(m: &CscMatrix, start: &[f64], sweeps: usize) -> (Vec<f64>, f64) {
    let mut d = start.to_vec();
    let mut rm = vec![0.0; m.n];
    let mut dev = f64::NAN;
    for _ in 0..sweeps {
        dev = kr_sweep(m, &mut d, &mut rm);
        if dev < 1e-8 {
            break;
        }
    }
    (d, dev)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    println!(
        "{:<24}{:>12}{:>12}{:>12}{:>12}{:>14}",
        "matrix", "cold-10", "warm-1", "warm-2", "warm-3", "max|dw-dc|/dc"
    );
    if std::env::var("KR_TRACE").is_ok() {
        for p in &paths {
            let m = read_mtx(std::path::Path::new(p)).and_then(|x| x.to_csc())?;
            let mut d = vec![1.0f64; m.n];
            let mut rm = vec![0.0; m.n];
            let name = std::path::Path::new(p)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            print!("{name:<24}");
            for _ in 0..10 {
                print!("{:>10.2e}", kr_sweep(&m, &mut d, &mut rm));
            }
            println!();
        }
        return Ok(());
    }
    let mut prev: Option<Vec<f64>> = None;
    for p in &paths {
        let m = read_mtx(std::path::Path::new(p)).and_then(|x| x.to_csc())?;
        let ones = vec![1.0f64; m.n];
        let (d_cold, dev_cold) = run(&m, &ones, 10);
        let name = std::path::Path::new(p)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        match &prev {
            None => {
                println!(
                    "{name:<24}{dev_cold:>12.2e}{:>12}{:>12}{:>12}{:>14}",
                    "-", "-", "-", "(first)"
                );
            }
            Some(pd) => {
                let (_, d1) = run(&m, pd, 1);
                let (_, d2) = run(&m, pd, 2);
                let (d_w3, d3) = run(&m, pd, 3);
                let rel = d_cold
                    .iter()
                    .zip(d_w3.iter())
                    .map(|(&c, &w)| if c != 0.0 { ((w - c) / c).abs() } else { 0.0 })
                    .fold(0.0f64, f64::max);
                println!(
                    "{name:<24}{dev_cold:>12.2e}{d1:>12.2e}{d2:>12.2e}{d3:>12.2e}{rel:>14.2e}"
                );
            }
        }
        prev = Some(d_cold);
    }
    Ok(())
}
