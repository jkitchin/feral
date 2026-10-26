//! Re-analysis of a stale value-dependent symbolic analysis (issue #227).
//!
//! A [`crate::Solver`] caches one symbolic analysis per sparsity pattern.
//! Under `OrderingPreprocess::LdltCompress` (and the `Auto` race that can
//! pick it) that analysis depends on the *values* of the matrix it was
//! built on: the MC64 matching decides which columns are paired, and the
//! ordering is computed on the compressed graph of those pairs. Across an
//! interior-point solve the values move by many orders of magnitude, and
//! the cached analysis can become much more expensive than another one —
//! or a later analysis can be much worse than an earlier one
//! (`dev/research/issue-227-stale-analysis-reanalysis.md`).
//!
//! [`ReanalysisPolicy::OnStale`] keeps, next to the cached analysis, a
//! small store of value-dependent analyses it has displaced or probed.
//! Three events open a *trial*, all evaluated only at the base pivot
//! threshold (never while an `increase_quality` ladder is active):
//!
//! 1. **cost** — the stored factor's actual dense-front work exceeds
//!    `trigger_ratio` × the analysis's predicted work;
//! 2. **probe** — the factor index on the pattern hits the probe schedule
//!    and a fresh analysis predicts at least `margin` less work;
//! 3. **escalation** — the #102 ordering escalation fired and replaced the
//!    analysis with a fresh `LdltCompress` one.
//!
//! Each candidate analysis factors the *same* matrix once; the cheapest by
//! actual flops is adopted iff it beats the incumbent by `margin`, reports
//! the same inertia, and gave up on no more pivots (floored + zero). Ranking
//! by predicted flops does not work — delays, not fill, decide the cost —
//! and a rollback that compares two different matrices misfires, so there
//! is neither: the store is the rollback.
//!
//! Everything a decision reads is an exact integer or an inertia, so the
//! policy is deterministic across runs and thread counts.

use crate::inertia::Inertia;
use crate::symbolic::SymbolicFactorization;

/// When a [`crate::Solver`] may replace its cached symbolic analysis on a
/// pattern it has already analysed. Set with
/// [`crate::Solver::with_reanalysis`].
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ReanalysisPolicy {
    /// Never re-analyse a cached pattern (the default). Bit-identical to
    /// feral before this policy existed.
    #[default]
    Never,
    /// Re-analyse on the cost, probe and escalation events described in
    /// the module docs, adopting a candidate only when a trial factor of
    /// the same matrix is cheaper with the same inertia.
    OnStale(OnStaleParams),
    /// Probe on every factor after the first (with [`OnStaleParams`]'s
    /// defaults otherwise, and no `min_flops_per_nnz_for_probes` guard).
    /// For measurement, not production.
    Always,
}

/// Parameters of [`ReanalysisPolicy::OnStale`].
#[derive(Debug, Clone, PartialEq)]
pub struct OnStaleParams {
    /// Cost event threshold on `actual_flops / predicted_flops` of the
    /// stored factor. Default 2.0.
    pub trigger_ratio: f64,
    /// Relative margin a candidate must win by: adopted only if
    /// `candidate · (1 + margin) < incumbent` (actual flops). Also the
    /// margin a fresh analysis's prediction must win by to be trialled.
    /// Default 0.25.
    pub margin: f64,
    /// Trial factorizations allowed per pattern. Once spent, no further
    /// event computes an analysis. Default 8.
    pub max_trials_per_pattern: u32,
    /// Displaced or probed value-dependent analyses kept as candidates
    /// (oldest evicted first). Default 2.
    pub max_stored: usize,
    /// First factor index (1-based, per pattern) that probes; 0 disables
    /// probes. Default 2.
    pub probe_first: u32,
    /// Geometric step of the probe schedule
    /// (`probe_first · probe_step^j`). Default 4: probes at 2, 8, 32, 128, …
    pub probe_step: u32,
    /// Last factor index that may probe. Default 2048.
    pub probe_last: u32,
    /// Probe only while `predicted_flops >= this · nnz(A)` for the
    /// incumbent analysis — i.e. where a factor costs more than an
    /// analysis. Default 100 (clnlbeam 42 and pooling 90 are below it;
    /// NARX 1 450 and qcqp 1 600 above).
    pub min_flops_per_nnz_for_probes: u64,
}

impl Default for OnStaleParams {
    fn default() -> Self {
        Self {
            trigger_ratio: 2.0,
            margin: 0.25,
            max_trials_per_pattern: 8,
            max_stored: 2,
            probe_first: 2,
            probe_step: 4,
            probe_last: 2048,
            min_flops_per_nnz_for_probes: 100,
        }
    }
}

/// Where a symbolic analysis came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisOrigin {
    /// The analysis `factor()` computed on the first factor of the pattern.
    Initial,
    /// A fresh analysis the policy computed at factor index `k`.
    Fresh { k: u32 },
    /// The `LdltCompress` analysis the #102 ordering escalation computed at
    /// factor index `k`.
    Escalation { k: u32 },
}

/// What opened a trial.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReanalysisEvent {
    /// `actual / predicted` of the stored factor exceeded `trigger_ratio`.
    Cost { ratio: f64 },
    /// Factor index `k` is on the probe schedule.
    Probe { k: u32 },
    /// The #102 ordering escalation fired on this factor.
    Escalation,
    /// [`ReanalysisPolicy::Always`] probe.
    Always,
}

/// One candidate's trial factorization of the current matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct TrialOutcome {
    pub origin: AnalysisOrigin,
    /// The candidate came from the store (rather than being computed now).
    pub from_store: bool,
    /// Predicted dense-front flops of the candidate analysis.
    pub predicted: u128,
    /// Actual dense-front flops of the trial factor.
    pub actual: u128,
    pub inertia: Inertia,
    pub n_tiny: usize,
    pub n_zero: usize,
    pub adopted: bool,
}

/// Why an event ended without adopting a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclineReason {
    /// The fresh analysis is pattern-only, like the incumbent: the same
    /// analysis, nothing to try.
    Identical,
    /// No candidate is cheaper by the margin (predicted for an untried
    /// fresh analysis, actual for a trial).
    NotCheaper,
    /// The cheapest candidate reported a different inertia.
    InertiaDiffers,
    /// The cheapest candidate floored or zeroed more pivots.
    MorePivotsGivenUp,
    /// `max_trials_per_pattern` is spent.
    CapReached,
    /// An `increase_quality` ladder is active.
    Ladder,
    /// An escalation found no stored candidate to compare with.
    NoCandidate,
}

/// What the policy did on the most recent `factor()` call. Returned by
/// [`crate::Solver::last_reanalysis`].
#[derive(Debug, Clone, PartialEq)]
pub struct ReanalysisReport {
    pub event: ReanalysisEvent,
    /// 1-based factor index on the pattern.
    pub factor_index: u32,
    /// Predicted / actual flops of the incumbent factor on this matrix.
    pub incumbent_predicted: u128,
    pub incumbent_actual: u128,
    /// Predicted flops of the fresh analysis, when one was computed.
    pub fresh_predicted: Option<u128>,
    pub trials: Vec<TrialOutcome>,
    /// `None` iff a candidate was adopted.
    pub declined: Option<DeclineReason>,
}

/// A displaced or probed value-dependent analysis kept as a candidate.
pub(crate) struct StoredAnalysis {
    pub(crate) sym: SymbolicFactorization,
    pub(crate) origin: AnalysisOrigin,
    pub(crate) predicted: u128,
}

/// Per-pattern policy state; reset on a pattern change.
pub(crate) struct ReanalysisState {
    pub(crate) stored: Vec<StoredAnalysis>,
    pub(crate) trials_used: u32,
    /// 1-based index of the current `factor()` call on the pattern.
    pub(crate) factor_index: u32,
    pub(crate) cost_latched: bool,
    /// Adoptions on this pattern.
    pub(crate) adoptions: u32,
    pub(crate) last_report: Option<ReanalysisReport>,
    /// The most recent `factor()` call adopted a different analysis.
    pub(crate) reanalysed_now: bool,
    /// Origin of the cached analysis.
    pub(crate) incumbent_origin: AnalysisOrigin,
}

impl Default for ReanalysisState {
    fn default() -> Self {
        Self {
            stored: Vec::new(),
            trials_used: 0,
            factor_index: 0,
            cost_latched: false,
            adoptions: 0,
            last_report: None,
            reanalysed_now: false,
            incumbent_origin: AnalysisOrigin::Initial,
        }
    }
}

/// The parameters in force and whether every factor probes.
pub(crate) fn effective(policy: &ReanalysisPolicy) -> Option<(OnStaleParams, bool)> {
    match policy {
        ReanalysisPolicy::Never => None,
        ReanalysisPolicy::OnStale(p) => Some((p.clone(), false)),
        ReanalysisPolicy::Always => Some((OnStaleParams::default(), true)),
    }
}

/// `k ∈ {first · step^j : j >= 0} ∩ [first, last]`. `first == 0`
/// disables; `step <= 1` probes at `first` only.
pub(crate) fn is_probe_index(k: u32, first: u32, step: u32, last: u32) -> bool {
    if first == 0 || k < first || k > last {
        return false;
    }
    let mut x = first;
    loop {
        if x == k {
            return true;
        }
        if step <= 1 || x > k {
            return false;
        }
        x = match x.checked_mul(step) {
            Some(y) => y,
            None => return false,
        };
    }
}

/// Probes only where a factor outweighs an analysis:
/// `predicted >= min_per_nnz · nnz`, in exact integer arithmetic.
pub(crate) fn probes_allowed(predicted: u128, nnz: usize, min_per_nnz: u64) -> bool {
    predicted >= u128::from(min_per_nnz) * nnz as u128
}

/// The cost event: `Some(actual / predicted)` when it exceeds `trigger`
/// and the event is not latched.
pub(crate) fn cost_ratio_fires(
    actual: u128,
    predicted: u128,
    trigger: f64,
    latched: bool,
) -> Option<f64> {
    if latched || predicted == 0 {
        return None;
    }
    let ratio = actual as f64 / predicted as f64;
    (ratio > trigger).then_some(ratio)
}

/// `candidate · (1 + margin) < reference`.
pub(crate) fn cheaper_by_margin(candidate: u128, reference: u128, margin: f64) -> bool {
    (candidate as f64) * (1.0 + margin) < reference as f64
}

/// The numbers a trial is judged on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Measure {
    pub(crate) actual: u128,
    pub(crate) inertia: Inertia,
    pub(crate) n_tiny: usize,
    pub(crate) n_zero: usize,
}

/// Adopt `cand` over `inc` (both factors of the same matrix)? Cheaper by
/// the margin, the same inertia, and no more floored or zeroed pivots.
/// Forced accepts are not compared (research note §7).
pub(crate) fn judge(cand: &Measure, inc: &Measure, margin: f64) -> Result<(), DeclineReason> {
    if !cheaper_by_margin(cand.actual, inc.actual, margin) {
        Err(DeclineReason::NotCheaper)
    } else if cand.inertia != inc.inertia {
        Err(DeclineReason::InertiaDiffers)
    } else if cand.n_tiny + cand.n_zero > inc.n_tiny + inc.n_zero {
        Err(DeclineReason::MorePivotsGivenUp)
    } else {
        Ok(())
    }
}

/// Index of the candidate to adopt: the lowest actual flops among those
/// `judge` accepts (ties: the earlier one). On none, the reason reported
/// for the cheapest candidate.
pub(crate) fn pick(
    cands: &[Measure],
    inc: &Measure,
    margin: f64,
) -> Result<usize, Option<DeclineReason>> {
    let mut best: Option<usize> = None;
    for (i, c) in cands.iter().enumerate() {
        if judge(c, inc, margin).is_ok() && best.is_none_or(|b| c.actual < cands[b].actual) {
            best = Some(i);
        }
    }
    if let Some(b) = best {
        return Ok(b);
    }
    let mut cheapest: Option<usize> = None;
    for (i, c) in cands.iter().enumerate() {
        if cheapest.is_none_or(|b| c.actual < cands[b].actual) {
            cheapest = Some(i);
        }
    }
    Err(cheapest.and_then(|i| judge(&cands[i], inc, margin).err()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inr(p: usize, n: usize, z: usize) -> Inertia {
        Inertia {
            positive: p,
            negative: n,
            zero: z,
        }
    }

    fn m(actual: u128, inertia: Inertia, n_tiny: usize, n_zero: usize) -> Measure {
        Measure {
            actual,
            inertia,
            n_tiny,
            n_zero,
        }
    }

    #[test]
    fn probe_schedule_is_geometric_and_bounded() {
        let hits: Vec<u32> = (0..3000)
            .filter(|&k| is_probe_index(k, 2, 4, 2048))
            .collect();
        assert_eq!(hits, vec![2, 8, 32, 128, 512, 2048]);
        let pow2: Vec<u32> = (0..40).filter(|&k| is_probe_index(k, 2, 2, 32)).collect();
        assert_eq!(pow2, vec![2, 4, 8, 16, 32]);
        assert!(
            !(0..100).any(|k| is_probe_index(k, 0, 4, 2048)),
            "first = 0 disables"
        );
        let once: Vec<u32> = (0..100).filter(|&k| is_probe_index(k, 5, 1, 99)).collect();
        assert_eq!(once, vec![5]);
        // No overflow near u32::MAX.
        assert!(!is_probe_index(u32::MAX, 3, 4, u32::MAX));
    }

    #[test]
    fn probe_size_guard_is_exact() {
        // clnlbeam-like: 42 flops per nnz is below 100.
        assert!(!probes_allowed(42 * 1000, 1000, 100));
        assert!(probes_allowed(100 * 1000, 1000, 100));
        assert!(!probes_allowed(100 * 1000 - 1, 1000, 100));
        assert!(probes_allowed(0, 1000, 0));
    }

    #[test]
    fn cost_event_fires_above_trigger_unless_latched() {
        assert_eq!(cost_ratio_fires(300, 100, 2.0, false), Some(3.0));
        assert_eq!(
            cost_ratio_fires(200, 100, 2.0, false),
            None,
            "strictly above"
        );
        assert_eq!(cost_ratio_fires(300, 100, 2.0, true), None);
        assert_eq!(cost_ratio_fires(300, 0, 2.0, false), None);
    }

    #[test]
    fn margin_test() {
        // 80 · 1.25 = 100 is not < 100.
        assert!(!cheaper_by_margin(80, 100, 0.25));
        assert!(cheaper_by_margin(79, 100, 0.25));
    }

    #[test]
    fn judge_reaches_every_reason() {
        let i = inr(3, 2, 0);
        let inc = m(1000, i.clone(), 1, 0);
        assert_eq!(judge(&m(500, i.clone(), 1, 0), &inc, 0.25), Ok(()));
        assert_eq!(
            judge(&m(900, i.clone(), 0, 0), &inc, 0.25),
            Err(DeclineReason::NotCheaper)
        );
        assert_eq!(
            judge(&m(500, inr(2, 3, 0), 0, 0), &inc, 0.25),
            Err(DeclineReason::InertiaDiffers)
        );
        assert_eq!(
            judge(&m(500, i.clone(), 1, 1), &inc, 0.25),
            Err(DeclineReason::MorePivotsGivenUp)
        );
        // Floored and zero pivots are summed: 0 + 1 <= 1 + 0.
        assert_eq!(judge(&m(500, i, 0, 1), &inc, 0.25), Ok(()));
    }

    #[test]
    fn pick_takes_the_cheapest_admissible_candidate() {
        let i = inr(3, 2, 0);
        let inc = m(1000, i.clone(), 0, 0);
        // The cheapest overall has the wrong inertia; the next is adopted.
        let c = vec![
            m(700, i.clone(), 0, 0),
            m(300, inr(2, 3, 0), 0, 0),
            m(500, i.clone(), 0, 0),
        ];
        assert_eq!(pick(&c, &inc, 0.25), Ok(2));
        // Ties go to the earlier candidate.
        let t = vec![m(500, i.clone(), 0, 0), m(500, i.clone(), 0, 0)];
        assert_eq!(pick(&t, &inc, 0.25), Ok(0));
        // None admissible: the cheapest one's reason.
        let d = vec![m(300, inr(2, 3, 0), 0, 0), m(950, i.clone(), 0, 0)];
        assert_eq!(
            pick(&d, &inc, 0.25),
            Err(Some(DeclineReason::InertiaDiffers))
        );
        let e = vec![m(950, i, 0, 0)];
        assert_eq!(pick(&e, &inc, 0.25), Err(Some(DeclineReason::NotCheaper)));
        assert_eq!(pick(&[], &inc, 0.25), Err(None));
    }

    #[test]
    fn policy_defaults() {
        assert_eq!(ReanalysisPolicy::default(), ReanalysisPolicy::Never);
        assert!(effective(&ReanalysisPolicy::Never).is_none());
        let (p, every) = effective(&ReanalysisPolicy::Always).expect("always");
        assert!(every);
        assert_eq!(p, OnStaleParams::default());
        let d = OnStaleParams::default();
        assert_eq!(
            (
                d.trigger_ratio,
                d.margin,
                d.max_trials_per_pattern,
                d.max_stored
            ),
            (2.0, 0.25, 8, 2)
        );
        assert_eq!(
            (
                d.probe_first,
                d.probe_step,
                d.probe_last,
                d.min_flops_per_nnz_for_probes
            ),
            (2, 4, 2048, 100)
        );
    }
}
