//! Regression test for https://github.com/jkitchin/feral/issues/15.
//!
//! The cascade-break trigger has a symbolic-arm gate at
//! `symbolic.n >= CASCADE_BREAK_MIN_N` (=4096). Below the threshold,
//! the trigger is a guaranteed no-op regardless of how aggressively
//! it is configured, because the achievable expanded ncol is bounded
//! above by `n` and cascade-break savings only accumulate at
//! several-thousand-column fronts.
//!
//! See `dev/research/issue-15-cascade-break-symbolic-arm.md` for the
//! principled framing and the per-family ratio-distribution evidence
//! that motivated the gate.

use std::path::Path;

use feral::numeric::factorize::{
    factorize_multifrontal_parallel, NumericParams, CASCADE_BREAK_MIN_N,
};
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, ZeroPivotAction};

#[test]
fn cascade_break_min_n_constant_locked() {
    // Lock the constant to its calibrated value. A change here
    // requires updating the research note and the corpus-regression
    // bench thresholds.
    assert_eq!(CASCADE_BREAK_MIN_N, 4096);
}

#[test]
fn small_n_disarms_cascade_break_trigger() {
    // MSS1_0000 has n=163, well below the 4096 gate. Even with the
    // most aggressive arming (`cascade_break_ratio = Some(0.0)`,
    // which would fire on ANY non-root delay), the gate must
    // suppress the trigger so the factor is bit-identical to the
    // `None` configuration on the same matrix.
    let path = Path::new("data/matrices/kkt/MSS1/MSS1_0000.mtx");
    if !path.exists() {
        eprintln!(
            "SKIP: {} not present (corpus is gitignored, not available in CI)",
            path.display()
        );
        return;
    }
    let mtx = read_mtx(path).expect("read MSS1_0000.mtx");
    let csc = mtx.to_csc().expect("to_csc");
    assert!(
        csc.n < CASCADE_BREAK_MIN_N,
        "MSS1_0000 must be below the gate to exercise it"
    );

    let snode_params = SupernodeParams {
        nemin: 1,
        ..SupernodeParams::default()
    };
    let sym = symbolic_factorize(&csc, &snode_params).expect("symbolic");

    // Aggressive pivot threshold to maximize the chance of producing
    // non-root delays during numeric factorization.
    let bk = BunchKaufmanParams {
        pivot_threshold: 0.5,
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    let base = NumericParams {
        bk: bk.clone(),
        scaling: ScalingStrategy::Identity,
        cascade_break_ratio: None,
        cascade_break_eps: None,
        min_parallel_flops: None,
        sqd_mode: false,
        static_pivot_threshold: None,
        ..NumericParams::default()
    };
    let armed = NumericParams {
        bk,
        scaling: ScalingStrategy::Identity,
        cascade_break_ratio: Some(0.0),
        cascade_break_eps: Some(1e-10),
        min_parallel_flops: None,
        sqd_mode: false,
        static_pivot_threshold: None,
        ..NumericParams::default()
    };

    let (f_off, i_off) = factorize_multifrontal_parallel(&csc, &sym, &base).expect("factor off");
    let (f_on, i_on) = factorize_multifrontal_parallel(&csc, &sym, &armed).expect("factor on");

    assert_eq!(
        i_off, i_on,
        "inertia must match: gate suppresses cascade-break on small n"
    );
    assert_eq!(
        f_off.node_factors.len(),
        f_on.node_factors.len(),
        "supernode count must match"
    );
    for (k, (a, b)) in f_off
        .node_factors
        .iter()
        .zip(f_on.node_factors.iter())
        .enumerate()
    {
        assert_eq!(
            a.n_delayed_in, b.n_delayed_in,
            "supernode {}: n_delayed_in must match (gate suppresses cascade-break)",
            k
        );
    }
    assert_eq!(
        f_off.needs_refinement, f_on.needs_refinement,
        "needs_refinement must match: gate suppresses the cascade-break perturbation \
         and therefore the refinement flag it would set"
    );

    // Sanity check: the test is only meaningful if MSS1 actually
    // produces a non-root delay under pivot_threshold=0.5 — otherwise
    // both runs are trivially identical regardless of the gate. With
    // 15 supernodes and a rank-deficient J on MSS1 this is well-
    // established (see dev/research/issue-5-mss1-inertia-monotonicity.md).
    let n_snodes = f_off.node_factors.len();
    assert!(n_snodes >= 2, "need >= 2 supernodes to have a non-root");
    let nonroot_delay = f_off
        .node_factors
        .iter()
        .take(n_snodes - 1)
        .any(|n| n.n_delayed_in > 0);
    assert!(
        nonroot_delay,
        "MSS1 must produce at least one non-root n_delayed_in > 0 \
         for this test to exercise the gate; if this fails the \
         fixture has drifted and the test is vacuously true"
    );
}
