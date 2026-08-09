//! Byte-parity of the issue-#148 task-coarsened parallel driver.
//!
//! The coarsened driver spawns one rayon task per *subtree* (TaskPlan)
//! instead of one per supernode, inlines lone-child chains, and falls
//! back to the sequential driver when the task graph has fewer than two
//! seeds. All of that is scheduling-only; this test pins it by forcing
//! a tiny cutoff (`FERAL_PAR_TASK_MIN_FLOPS=1`, maximal task graph) and
//! a huge cutoff (single task per root / serial fallback), and asserting
//! the parallel factors are byte-identical to the sequential driver's
//! in every configuration.
//!
//! Lives in its own integration-test binary because it sets
//! process-global env vars: the single #[test] cannot race others in
//! the same process.

use feral::numeric::factorize::{
    factorize_multifrontal_parallel, factorize_multifrontal_supernodal, NumericParams,
};
use feral::sparse::csc::CscMatrix;
use feral::symbolic::{symbolic_factorize, SupernodeParams};

/// Small 2D grid Laplacian (branchy elimination tree → several seeds)
/// plus a block-tridiagonal chain welded on (chain → lone-child tasks).
fn build_matrix() -> CscMatrix {
    let g = 24usize; // 576-node grid
    let chain = 120usize; // chain tail
    let n = g * g + chain;
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for y in 0..g {
        for x in 0..g {
            let i = y * g + x;
            rows.push(i);
            cols.push(i);
            vals.push(4.0 + 0.01 * (i % 7) as f64);
            if x + 1 < g {
                rows.push(i + 1);
                cols.push(i);
                vals.push(-1.0);
            }
            if y + 1 < g {
                rows.push(i + g);
                cols.push(i);
                vals.push(-1.0);
            }
        }
    }
    let base = g * g;
    for k in 0..chain {
        let i = base + k;
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        rows.push(i);
        cols.push(i);
        vals.push(sign * (3.0 + 0.02 * (k % 5) as f64));
        if k > 0 {
            rows.push(i);
            cols.push(i - 1);
            vals.push(0.7);
        }
    }
    // Weld chain start to grid corner so it is one tree.
    rows.push(base);
    cols.push(g * g - 1);
    vals.push(0.3);
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("csc build")
}

fn factor_bits(matrix: &CscMatrix, parallel: bool) -> (Vec<u64>, (usize, usize, usize)) {
    let symbolic = symbolic_factorize(matrix, &SupernodeParams::default()).expect("symbolic");
    let mut params = NumericParams::default();
    if parallel {
        // Force the parallel tree driver past the PAR_MIN_FLOPS gate so
        // this test exercises the task machinery, not the dispatcher's
        // sequential fallthrough (N_PAR_MIN still applies and passes:
        // the fixture has well over 32 supernodes).
        params.min_parallel_flops = Some(0);
    }
    let (factors, inertia) = if parallel {
        factorize_multifrontal_parallel(matrix, &symbolic, &params).expect("parallel factor")
    } else {
        factorize_multifrontal_supernodal(matrix, &symbolic, &params).expect("serial factor")
    };
    let mut bits: Vec<u64> = Vec::new();
    for nf in &factors.node_factors {
        bits.extend(nf.frontal_factors.l.iter().map(|v| v.to_bits()));
        bits.extend(nf.frontal_factors.d_diag.iter().map(|v| v.to_bits()));
        bits.extend(nf.frontal_factors.d_subdiag.iter().map(|v| v.to_bits()));
    }
    (bits, (inertia.positive, inertia.negative, inertia.zero))
}

#[test]
fn task_coarsened_parallel_is_byte_identical_to_serial() {
    let matrix = build_matrix();
    let reference = factor_bits(&matrix, false);

    // Maximal task graph: every branching point becomes a boundary.
    std::env::set_var("FERAL_PAR_TASK_MIN_FLOPS", "1");
    let fine = factor_bits(&matrix, true);
    assert_eq!(
        fine, reference,
        "fine-grained task plan diverged from the sequential driver"
    );

    // Default cutoff.
    std::env::remove_var("FERAL_PAR_TASK_MIN_FLOPS");
    let default_plan = factor_bits(&matrix, true);
    assert_eq!(
        default_plan, reference,
        "default task plan diverged from the sequential driver"
    );

    // Huge cutoff: single task / serial fallback path.
    std::env::set_var("FERAL_PAR_TASK_MIN_FLOPS", "18446744073709551615");
    let coarse = factor_bits(&matrix, true);
    std::env::remove_var("FERAL_PAR_TASK_MIN_FLOPS");
    assert_eq!(
        coarse, reference,
        "serial-fallback path diverged from the sequential driver"
    );

    // `FERAL_PAR_MIN_SEEDS` sweep (issue #148 calibration knob): the
    // threshold decides how much initial parallelism is required before
    // the task graph is used at all. 0/1 force the task graph on; a huge
    // value forces the sequential delegation. Every setting must produce
    // identical factors — the knob is scheduling-only, so it can be
    // calibrated for speed on real hardware with no numerical risk.
    for seeds in ["0", "1", "2", "4", "64", "18446744073709551615"] {
        std::env::set_var("FERAL_PAR_MIN_SEEDS", seeds);
        let swept = factor_bits(&matrix, true);
        std::env::remove_var("FERAL_PAR_MIN_SEEDS");
        assert_eq!(
            swept, reference,
            "FERAL_PAR_MIN_SEEDS={seeds} diverged from the sequential driver"
        );
    }
}
