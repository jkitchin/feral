//! Issue #227: `ReanalysisPolicy` behaviour on the `Solver`.
//!
//! Inertia oracle: the synthetic KKT path of `tests/common227` has the
//! Sylvester/Haynsworth inertia `(n_x, m, 0)` on every iterate.

#[path = "common227/mod.rs"]
mod common227;

use common227::{synthetic_inertia, synthetic_kkt, N_ITER};
use feral::{FactorStatus, OnStaleParams, ReanalysisPolicy, Solver};

#[test]
fn builder_round_trip() {
    assert_eq!(*Solver::new().reanalysis_policy(), ReanalysisPolicy::Never);
    let p = ReanalysisPolicy::OnStale(OnStaleParams {
        margin: 0.5,
        ..OnStaleParams::default()
    });
    let s = Solver::new().with_reanalysis(p.clone());
    assert_eq!(*s.reanalysis_policy(), p);
    let s = Solver::new().with_reanalysis(ReanalysisPolicy::Always);
    assert_eq!(*s.reanalysis_policy(), ReanalysisPolicy::Always);
}

#[test]
fn never_reports_nothing() {
    let mut s = Solver::new();
    for k in 0..N_ITER {
        let st = s.factor(&synthetic_kkt(k), Some(synthetic_inertia()));
        assert!(matches!(st, FactorStatus::Success), "{k}: {st:?}");
        let fs = s.last_factor_stats().expect("stats");
        assert!(!fs.reanalysed);
        assert_eq!(fs.reanalysis_count, 0);
        assert!(s.last_reanalysis().is_none());
    }
}

use common227::{factor_digest, qcqp_like, qcqp_like_inertia};
use feral::numeric::factorize::NumericParams;
use feral::scaling::ScalingStrategy;
use feral::symbolic::{OrderingMethod, OrderingPreprocess, SupernodeParams};
use feral::{
    AnalysisOrigin, BunchKaufmanParams, CscMatrix, DeclineReason, ReanalysisEvent, ZeroPivotAction,
};

fn on_stale(p: OnStaleParams) -> ReanalysisPolicy {
    ReanalysisPolicy::OnStale(p)
}

/// Factor `seq` with one solver; per factor: digest, `reanalysed`, report.
#[allow(clippy::type_complexity)]
fn run(
    mut s: Solver,
    seq: &[CscMatrix],
    oracle: &feral::Inertia,
) -> (Solver, Vec<(u64, bool, Option<feral::ReanalysisReport>)>) {
    let mut out = Vec::new();
    for (i, a) in seq.iter().enumerate() {
        let st = s.factor(a, Some(oracle.clone()));
        assert!(
            matches!(st, FactorStatus::Success),
            "factor {i}: {st:?}, Sylvester inertia {oracle:?}"
        );
        let d = match (s.factors(), s.inertia()) {
            (Some(f), Some(i)) => factor_digest(f, i),
            _ => panic!("no factor at {i}"),
        };
        let re = s.last_factor_stats().expect("stats").reanalysed;
        out.push((d, re, s.last_reanalysis().cloned()));
    }
    (s, out)
}

#[test]
fn cost_event_on_a_pattern_only_analysis_is_identical_and_changes_nothing() {
    let seq: Vec<CscMatrix> = (0..N_ITER).map(synthetic_kkt).collect();
    let oracle = synthetic_inertia();
    // Pattern-only analysis (the synthetic path's `Auto` race keeps
    // `LdltCompress`, which is value-dependent).
    let pattern_only = || {
        let sn = SupernodeParams {
            preprocess: OrderingPreprocess::None,
            ..SupernodeParams::default()
        };
        Solver::with_params(NumericParams::default(), sn)
    };
    let (never, base) = run(pattern_only(), &seq, &oracle);
    let resolved = never
        .last_factor_stats()
        .expect("stats")
        .ordering_info
        .preprocess;
    assert_eq!(resolved, OrderingPreprocess::None, "fixture premise");

    let p = on_stale(OnStaleParams {
        trigger_ratio: 0.5,
        probe_first: 0,
        ..OnStaleParams::default()
    });
    let (s, got) = run(pattern_only().with_reanalysis(p), &seq, &oracle);
    // Factor 1 was analysed on its own matrix: no event.
    assert!(got[0].2.is_none());
    // Factor 2: actual/predicted >= 1 > 0.5 fires the cost event; the fresh
    // analysis is pattern-only like the incumbent, so no trial.
    let r = got[1].2.as_ref().expect("cost event on factor 2");
    assert!(matches!(r.event, ReanalysisEvent::Cost { .. }), "{r:?}");
    assert_eq!(r.declined, Some(DeclineReason::Identical));
    assert!(r.trials.is_empty());
    assert!(r.fresh_predicted.is_some());
    // Latched: no further cost event on this analysis.
    for (k, g) in got.iter().enumerate().skip(2) {
        assert!(g.2.is_none(), "factor {}: {:?}", k + 1, g.2);
    }
    // Bit-identical to Never on every factor; nothing adopted.
    let d_base: Vec<u64> = base.iter().map(|g| g.0).collect();
    let d_got: Vec<u64> = got.iter().map(|g| g.0).collect();
    assert_eq!(d_got, d_base);
    assert!(got.iter().all(|g| !g.1));
    let st = s.last_factor_stats().expect("stats");
    assert_eq!(st.reanalysis_count, 0);
    // One analysis for the pattern, one for the cost event.
    assert_eq!(s.symbolic_call_count(), 2);
}

/// The qcqp-like path in two call orders. The `Auto` analysis of the init
/// matrix and of iterate 0 is `LdltCompress`; by iterate 6 it is `None`
/// with about half the predicted work.
fn qcqp_sequence(init_first: bool) -> Vec<CscMatrix> {
    qcqp_sequence_n(init_first, 10)
}

fn qcqp_sequence_n(init_first: bool, n_iter: u64) -> Vec<CscMatrix> {
    let mut v: Vec<CscMatrix> = (0..n_iter).map(|k| qcqp_like(Some(k))).collect();
    v.insert(if init_first { 0 } else { 1 }, qcqp_like(None));
    v
}

fn probes_everywhere() -> OnStaleParams {
    OnStaleParams {
        min_flops_per_nnz_for_probes: 0,
        ..OnStaleParams::default()
    }
}

#[test]
fn probe_stores_a_costlier_analysis_and_adopts_a_cheaper_one() {
    for init_first in [true, false] {
        let seq = qcqp_sequence(init_first);
        let s = Solver::new().with_reanalysis(on_stale(probes_everywhere()));
        let (s, got) = run(s, &seq, &qcqp_like_inertia());
        let ctx = if init_first {
            "init-first"
        } else {
            "iterate-first"
        };

        // k = 2: the fresh analysis is value-dependent and predicts more
        // work than the incumbent — stored, not trialled.
        let r2 = got[1].2.as_ref().expect("probe at k = 2");
        assert_eq!(r2.event, ReanalysisEvent::Probe { k: 2 }, "{ctx}");
        assert!(r2.trials.is_empty(), "{ctx}");
        assert_eq!(r2.declined, Some(DeclineReason::NotCheaper), "{ctx}");
        assert!(r2.fresh_predicted.expect("fresh") * 5 > r2.incumbent_predicted * 4);

        // k = 8: the fresh pattern-only analysis predicts < 1/1.25 of the
        // incumbent; its trial factor is cheaper with the same inertia.
        let r8 = got[7].2.as_ref().expect("probe at k = 8");
        assert_eq!(r8.event, ReanalysisEvent::Probe { k: 8 }, "{ctx}");
        assert_eq!(r8.trials.len(), 1, "{ctx}");
        let t = &r8.trials[0];
        assert!(
            t.adopted && t.origin == AnalysisOrigin::Fresh { k: 8 },
            "{ctx}"
        );
        assert_eq!(t.inertia, qcqp_like_inertia());
        assert!(
            (t.actual as f64) * 1.25 < r8.incumbent_actual as f64,
            "{ctx}"
        );
        assert_eq!(r8.declined, None);

        // `reanalysed` exactly on the adopting call.
        let flags: Vec<bool> = got.iter().map(|g| g.1).collect();
        let want: Vec<bool> = (0..seq.len()).map(|i| i == 7).collect();
        assert_eq!(flags, want, "{ctx}");
        let st = s.last_factor_stats().expect("stats");
        assert_eq!(st.reanalysis_count, 1);
        assert!(st.pattern_reused);
        assert_eq!(st.ordering_info.preprocess, OrderingPreprocess::None);
        // pattern + probes at 2 and 8.
        assert_eq!(s.symbolic_call_count(), 3, "{ctx}");
    }
}

#[test]
fn adopted_factor_is_the_one_a_fresh_solver_computes() {
    // After adopting at k = 8 the solver holds the analysis a fresh
    // `Solver` builds for that matrix, so their factors of the following
    // iterates agree bit for bit when the scaling pick agrees.
    let seq = qcqp_sequence(false);
    let s = Solver::new().with_reanalysis(on_stale(probes_everywhere()));
    let (_, got) = run(s, &seq, &qcqp_like_inertia());
    let mut fresh = Solver::new();
    for (i, a) in seq.iter().enumerate().skip(7) {
        let _ = fresh.factor(a, None);
        let (f, inr) = (fresh.factors().expect("f"), fresh.inertia().expect("i"));
        assert_eq!(got[i].0, factor_digest(f, inr), "factor {}", i + 1);
    }
}

#[test]
fn probes_respect_the_size_guard() {
    // predicted / nnz(A) is far below 1e9 here: no probe at all.
    let seq = qcqp_sequence(true);
    let p = on_stale(OnStaleParams {
        min_flops_per_nnz_for_probes: 1_000_000_000,
        ..OnStaleParams::default()
    });
    let (s, got) = run(Solver::new().with_reanalysis(p), &seq, &qcqp_like_inertia());
    assert!(got.iter().all(|g| g.2.is_none()));
    assert_eq!(s.symbolic_call_count(), 1);
}

#[test]
fn reports_and_factors_are_deterministic_across_solvers_and_thread_counts() {
    for init_first in [true, false] {
        let seq = qcqp_sequence(init_first);
        let mk = |par: bool| {
            Solver::new()
                .with_parallel(par)
                .with_reanalysis(on_stale(probes_everywhere()))
        };
        let (_, a) = run(mk(true), &seq, &qcqp_like_inertia());
        let (_, b) = run(mk(true), &seq, &qcqp_like_inertia());
        let (_, c) = run(mk(false), &seq, &qcqp_like_inertia());
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        assert_eq!(format!("{a:?}"), format!("{c:?}"));
    }
}

#[test]
fn trial_cap_is_honoured() {
    // 17 factors: probes at 2, 4, 8, 16.
    let seq = qcqp_sequence_n(true, 16);
    let p = on_stale(OnStaleParams {
        max_trials_per_pattern: 1,
        probe_step: 2, // probes at 2, 4, 8, ...
        ..probes_everywhere()
    });
    let (_, got) = run(Solver::new().with_reanalysis(p), &seq, &qcqp_like_inertia());
    let trials: usize = got
        .iter()
        .filter_map(|g| g.2.as_ref())
        .map(|r| r.trials.len())
        .sum();
    assert_eq!(trials, 1);
    let last = got
        .iter()
        .filter_map(|g| g.2.as_ref())
        .next_back()
        .expect("reports");
    assert_eq!(last.declined, Some(DeclineReason::CapReached), "{last:?}");
}

/// `tests/issue_221_quality_rung.rs`'s fixture: natural order, no
/// amalgamation, unit scaling, floored zero pivots.
fn ladder_solver() -> Solver {
    let np = NumericParams {
        scaling: ScalingStrategy::External(vec![1.0; 5]),
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-10 },
            ..NumericParams::default().bk
        },
        ..NumericParams::default()
    };
    let sn = SupernodeParams {
        nemin: 1,
        ..SupernodeParams::default()
    };
    Solver::with_params(np, sn)
        .with_ordering(OrderingMethod::External((0..5).collect()))
        .with_reanalysis(ReanalysisPolicy::Always)
}

fn diag(v: &[f64]) -> CscMatrix {
    let idx: Vec<usize> = (0..v.len()).collect();
    CscMatrix::from_triplets(v.len(), &idx, &idx, v).expect("diag")
}

#[test]
fn no_event_while_a_quality_ladder_is_active() {
    let mut s = ladder_solver();
    let _ = s.factor(&diag(&[1.0, 0.0, 1.0, 0.0, 1.0]), None);
    assert!(s.last_reanalysis().is_none(), "first factor of the pattern");
    assert!(s.increase_quality(), "1e-8 -> 1e-6");

    // On the rung: the event is declined, nothing is analysed.
    let calls = s.symbolic_call_count();
    let _ = s.factor(&diag(&[1.0, 0.0, 0.0, 0.0, 1.0]), None);
    let r = s.last_reanalysis().expect("event seen");
    assert_eq!(r.declined, Some(DeclineReason::Ladder));
    assert!(r.trials.is_empty() && r.fresh_predicted.is_none());
    assert_eq!(s.symbolic_call_count(), calls);
    // #221's judgement is unaffected: three floored pivots is worse.
    assert_eq!(s.last_factor_stats().expect("stats").n_tiny, 3);
    assert!(!s.increase_quality(), "worse rung undone");
    assert_eq!(s.pivot_threshold(), 1e-8);

    // Ladder undone: events resume (a pattern-only analysis, so Identical).
    let _ = s.factor(&diag(&[1.0, 0.0, 1.0, 0.0, 1.0]), None);
    let r = s.last_reanalysis().expect("event resumes");
    // The floored pivots delay into the root, so the cost event (which
    // takes precedence over the `Always` probe) fires; either way the
    // pattern-only fresh analysis is the incumbent's.
    assert!(
        matches!(
            r.event,
            ReanalysisEvent::Cost { .. } | ReanalysisEvent::Always
        ),
        "{r:?}"
    );
    assert_eq!(r.declined, Some(DeclineReason::Identical));
    assert_eq!(s.symbolic_call_count(), calls + 1);
}

#[test]
fn cost_event_trials_the_stored_analyses_and_latches_on_decline() {
    // trigger 0.5 makes every unlatched factor a cost event (cost takes
    // precedence over a probe on the same factor).
    let seq = qcqp_sequence(true);
    let p = on_stale(OnStaleParams {
        trigger_ratio: 0.5,
        ..probes_everywhere()
    });
    let (_, got) = run(Solver::new().with_reanalysis(p), &seq, &qcqp_like_inertia());
    // k = 2: cost event; the fresh value-dependent analysis does not
    // predict below the incumbent's actual by the margin and the store is
    // empty — stored, declined, latched.
    let r2 = got[1].2.as_ref().expect("k = 2");
    assert!(matches!(r2.event, ReanalysisEvent::Cost { .. }), "{r2:?}");
    assert!(r2.trials.is_empty());
    assert_eq!(r2.declined, Some(DeclineReason::NotCheaper));
    // Latched until an adoption: k = 3..7 report nothing, k = 8 is the
    // probe that adopts the pattern-only analysis (clearing the latch).
    for g in &got[2..7] {
        assert!(g.2.is_none(), "{:?}", g.2);
    }
    let r8 = got[7].2.as_ref().expect("k = 8");
    assert_eq!(r8.event, ReanalysisEvent::Probe { k: 8 });
    assert!(got[7].1);
    // k = 9: the cost event fires on the new analysis. The fresh analysis
    // is pattern-only like the incumbent; both stored value-dependent
    // analyses (the k = 2 probe's and the displaced initial one) are
    // trialled on this matrix and are dearer — declined, latched again.
    let r9 = got[8].2.as_ref().expect("k = 9");
    assert!(matches!(r9.event, ReanalysisEvent::Cost { .. }), "{r9:?}");
    assert_eq!(r9.trials.len(), 2, "{r9:?}");
    assert!(r9.trials.iter().all(|t| t.from_store && !t.adopted));
    let origins: Vec<AnalysisOrigin> = r9.trials.iter().map(|t| t.origin).collect();
    assert_eq!(
        origins,
        vec![AnalysisOrigin::Fresh { k: 2 }, AnalysisOrigin::Initial]
    );
    assert!(r9.trials.iter().all(|t| t.inertia == qcqp_like_inertia()));
    assert_eq!(r9.declined, Some(DeclineReason::NotCheaper));
    assert!(got[9].2.is_none() && got[10].2.is_none(), "latched");
    assert!(!got[8].1);
}

// ---------------------------------------------------------------------
// Step 4: the #102 ordering escalation as an event.
//
// `tests/issue102_ordering_escalation.rs`'s cont5 fixture is gitignored,
// so these tests make the escalation fire on the qcqp-like path through
// the public growth ceiling: its `None` factors have pivot growth ~1e16,
// so a ceiling of 1e12 escalates the first `None` factor after the k = 8
// probe adopted the `None` analysis (an `LdltCompress` analysis is never
// escalated). The fresh escalated `LdltCompress` analysis of iterate 7
// costs 4.44e6 actual; the two stored ones 3.93e6 and 4.50e6.

fn escalating(p: OnStaleParams, parallel: bool) -> Solver {
    Solver::new()
        .with_parallel(parallel)
        .with_ordering_escalation(Some(1e12))
        .with_reanalysis(on_stale(p))
}

#[test]
fn escalation_trials_the_stored_analyses_and_adopts_a_cheaper_one() {
    for init_first in [true, false] {
        let seq = qcqp_sequence(init_first);
        let p = OnStaleParams {
            margin: 0.1,
            ..probes_everywhere()
        };
        let (s, got) = run(escalating(p.clone(), true), &seq, &qcqp_like_inertia());
        let ctx = if init_first {
            "init-first"
        } else {
            "iterate-first"
        };
        let r = got[8].2.as_ref().expect("escalation at k = 9");
        assert_eq!(r.event, ReanalysisEvent::Escalation, "{ctx}");
        assert_eq!(r.factor_index, 9);
        assert_eq!(r.trials.len(), 2, "{ctx}: both stored analyses");
        assert!(r.trials.iter().all(|t| t.from_store));
        assert!(r.trials.iter().all(|t| t.inertia == qcqp_like_inertia()));
        // The cheaper stored analysis wins by more than 10 %.
        let best = r.trials.iter().min_by_key(|t| t.actual).expect("trials");
        assert!(best.adopted, "{ctx}: {r:?}");
        assert!((best.actual as f64) * 1.1 < r.incumbent_actual as f64);
        assert_eq!(r.trials.iter().filter(|t| t.adopted).count(), 1);
        assert_eq!(r.declined, None);
        assert!(got[8].1, "{ctx}: reanalysed on the escalation call");
        let st = s.last_factor_stats().expect("stats");
        assert!(st.ordering_info.escalated);
        assert_eq!(
            st.ordering_info.preprocess,
            OrderingPreprocess::LdltCompress
        );
        // One count per adopting call (iterate-first also adopts the
        // init analysis at k = 2 under the 10 % margin).
        let adopting = got.iter().filter(|g| g.1).count() as u32;
        assert_eq!(st.reanalysis_count, adopting, "{ctx}");
        assert_eq!(adopting, if init_first { 2 } else { 3 }, "{ctx}");
        // Deterministic across thread counts.
        let (_, seq_run) = run(escalating(p, false), &seq, &qcqp_like_inertia());
        assert_eq!(format!("{got:?}"), format!("{seq_run:?}"), "{ctx}");
    }
}

#[test]
fn escalation_keeps_the_fresh_analysis_when_no_stored_one_wins() {
    // Default margin 0.25: 3.93e6 · 1.25 > 4.44e6 — the escalated analysis
    // stays, as it does without the policy.
    let seq = qcqp_sequence(true);
    let (_, got) = run(
        escalating(probes_everywhere(), true),
        &seq,
        &qcqp_like_inertia(),
    );
    let r = got[8].2.as_ref().expect("escalation at k = 9");
    assert_eq!(r.event, ReanalysisEvent::Escalation);
    assert_eq!(r.trials.len(), 2);
    assert!(r.trials.iter().all(|t| !t.adopted));
    assert_eq!(r.declined, Some(DeclineReason::NotCheaper));
    assert!(!got[8].1);
}

#[test]
fn escalation_with_an_empty_store_has_no_candidate() {
    // A late iterate first: `None` analysis, growth ~1e16 > 1e12, so the
    // very first factor escalates before anything was stored.
    let seq: Vec<CscMatrix> = (8..11).map(|k| qcqp_like(Some(k))).collect();
    let (s, got) = run(
        escalating(probes_everywhere(), true),
        &seq,
        &qcqp_like_inertia(),
    );
    let r = got[0].2.as_ref().expect("escalation at k = 1");
    assert_eq!(r.event, ReanalysisEvent::Escalation);
    assert_eq!(r.factor_index, 1);
    assert!(r.trials.is_empty());
    assert_eq!(r.declined, Some(DeclineReason::NoCandidate));
    assert!(!got[0].1);
    // The escalation's pattern_reused contract (#215) is untouched.
    let st0 = s.last_factor_stats().expect("stats");
    assert!(st0.ordering_info.escalated);
}

#[test]
fn escalation_under_never_is_unchanged() {
    // Same escalating sequence without the policy: no report, and the
    // factors are those of a Solver that never had a policy field.
    let seq: Vec<CscMatrix> = (8..11).map(|k| qcqp_like(Some(k))).collect();
    let s = Solver::new().with_ordering_escalation(Some(1e12));
    let (s, got) = run(s, &seq, &qcqp_like_inertia());
    assert!(got.iter().all(|g| g.2.is_none() && !g.1));
    assert!(
        s.last_factor_stats()
            .expect("stats")
            .ordering_info
            .escalated
    );
}
