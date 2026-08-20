//! Does the `solve_sparse_many` batching win survive interleaved measurement?
//!
//!   cargo run -p feral-diagnostics --bin probe_batch_interleaved --release [-- <name>...]
//!
//! `probe_largen_solve_levers` measured the batching lever in *blocks* —
//! all repetitions of the looped path, then all repetitions of the batched
//! path. `probe_solve_reconcile` showed that blocked measurement produced a
//! 2.4x error on the contribution-block comparison, so every number taken
//! that way is suspect until re-measured.
//!
//! This re-measures just the batching ratio with the two paths interleaved
//! inside each repetition, so drift hits both equally. Batching is expected
//! to survive — unlike the schedule comparison it was stable across runs —
//! but "expected to survive" is what the other two claims looked like too.

use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::Path;
use std::time::Instant;

const REPS: usize = 11;
/// An L-BFGS host with `limited_memory_max_history = 6` presents 13.
const NRHS: usize = 13;

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[s.len() / 2]
}

fn geomean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp()
}

fn run(path: &Path) -> Option<f64> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let csc: CscMatrix = match read_mtx(path).and_then(|m| m.to_csc()) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIP {name}: {e}");
            return None;
        }
    };
    let n = csc.n;
    let mut solver = Solver::new();
    match solver.factor(&csc, None) {
        FactorStatus::Success | FactorStatus::WrongInertia { .. } => {}
        other => {
            println!("SKIP {name}: factor {other:?}");
            return None;
        }
    }

    // Column-major n x NRHS block, each column a distinct well-scaled RHS.
    let mut block = vec![0.0f64; n * NRHS];
    for (j, col) in block.chunks_mut(n).enumerate() {
        for (i, s) in col.iter_mut().enumerate() {
            *s = 1.0 + ((i + j) % 7) as f64 / 8.0;
        }
    }

    let mut looped_x = vec![0.0f64; n];
    let mut batched_x = vec![0.0f64; n * NRHS];
    let (mut t_loop, mut t_batch) = (Vec::new(), Vec::new());
    for _ in 0..REPS {
        let t = Instant::now();
        for col in block.chunks(n) {
            if let Err(e) = solver.solve_into(col, &mut looped_x) {
                println!("SKIP {name}: looped: {e}");
                return None;
            }
            std::hint::black_box(&looped_x);
        }
        t_loop.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        if let Err(e) = solver.solve_many_into(&block, NRHS, &mut batched_x) {
            println!("SKIP {name}: batched: {e}");
            return None;
        }
        std::hint::black_box(&batched_x);
        t_batch.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let (l, b) = (median(&t_loop), median(&t_batch));
    println!(
        "{name:<20} n={n:<8} looped {l:10.0}  batched {b:10.0} us   speedup {:.2}x",
        l / b
    );
    Some(l / b)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names: Vec<String> = if args.is_empty() {
        vec![
            "bcsstk38".into(),
            "r05_kkt".into(),
            "bratu3d".into(),
            "qap15_kkt".into(),
            "dirichlet120_kkt".into(),
            "cont-201".into(),
            "cont5_late_kkt".into(),
        ]
    } else {
        args
    };
    println!("batching re-check, interleaved ({REPS} reps, nrhs={NRHS}, medians)\n");
    let r: Vec<f64> = names
        .iter()
        .filter_map(|nm| run(&Path::new("tests/data/large").join(format!("{nm}.mtx"))))
        .collect();
    println!(
        "\ngeomean {:.2}x   min {:.2}x",
        geomean(&r),
        r.iter().cloned().fold(f64::INFINITY, f64::min)
    );
}
