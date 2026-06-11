//! Introspection value types surfaced to Python: `ScalingInfo`,
//! `FactorStats`, `ProfileReport` (+ `BucketStats`, `PrologueBreakdown`)
//! and `SymbolicProfileReport` (+ `StagePct`). All are frozen pyclasses
//! built from the corresponding Rust types via `From`.

use feral::numeric::factorize::{
    BucketStats as RustBucketStats, ProfileReport as RustProfileReport,
    PrologueBreakdown as RustPrologueBreakdown,
};
use feral::numeric::solver::FactorStats as RustFactorStats;
use feral::scaling::{Mc64FallbackReason, ScalingInfo as RustScalingInfo};
use feral::symbolic::{
    StagePct as RustStagePct, SymbolicProfileReport as RustSymbolicProfileReport,
};

use pyo3::prelude::*;

use crate::common::Inertia;

fn mc64_reason_str(r: Mc64FallbackReason) -> &'static str {
    match r {
        Mc64FallbackReason::InfNormSpreadAcceptable => "inf_norm_spread_acceptable",
        Mc64FallbackReason::Mc64WorseThanInfnorm => "mc64_worse_than_infnorm",
        Mc64FallbackReason::Mc64ScalingDegenerate => "mc64_scaling_degenerate",
    }
}

/// Outcome of the scaling stage.
///
/// `kind` is one of `"applied"`, `"partial_singular"`,
/// `"mc64_fallback_to_infnorm"`, `"not_applied"`. `n_unmatched` is set
/// only for `partial_singular`; `reason` only for the MC64 fallback.
#[pyclass(module = "feral._feral", frozen)]
#[derive(Clone)]
pub struct ScalingInfo {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub n_unmatched: Option<usize>,
    #[pyo3(get)]
    pub reason: Option<String>,
}

impl From<&RustScalingInfo> for ScalingInfo {
    fn from(s: &RustScalingInfo) -> Self {
        match s {
            RustScalingInfo::Applied => Self {
                kind: "applied".to_string(),
                n_unmatched: None,
                reason: None,
            },
            RustScalingInfo::PartialSingular { n_unmatched } => Self {
                kind: "partial_singular".to_string(),
                n_unmatched: Some(*n_unmatched),
                reason: None,
            },
            RustScalingInfo::Mc64FallbackToInfnorm { reason } => Self {
                kind: "mc64_fallback_to_infnorm".to_string(),
                n_unmatched: None,
                reason: Some(mc64_reason_str(*reason).to_string()),
            },
            RustScalingInfo::NotApplied => Self {
                kind: "not_applied".to_string(),
                n_unmatched: None,
                reason: None,
            },
        }
    }
}

#[pymethods]
impl ScalingInfo {
    fn __repr__(&self) -> String {
        format!(
            "ScalingInfo(kind={:?}, n_unmatched={:?}, reason={:?})",
            self.kind, self.n_unmatched, self.reason
        )
    }
}

/// Summary statistics of the most recent factorization.
#[pyclass(module = "feral._feral", frozen)]
#[derive(Clone)]
pub struct FactorStats {
    #[pyo3(get)]
    pub nnz_a: usize,
    #[pyo3(get)]
    pub nnz_l: usize,
    #[pyo3(get)]
    pub fill_ratio: f64,
    #[pyo3(get)]
    pub inertia: Inertia,
    #[pyo3(get)]
    pub min_abs_pivot: f64,
    #[pyo3(get)]
    pub max_abs_pivot: f64,
    #[pyo3(get)]
    pub pattern_reused: bool,
    #[pyo3(get)]
    pub scaling_info: ScalingInfo,
    #[pyo3(get)]
    pub n_tiny: usize,
}

impl From<RustFactorStats> for FactorStats {
    fn from(s: RustFactorStats) -> Self {
        Self {
            nnz_a: s.nnz_a,
            nnz_l: s.nnz_l,
            fill_ratio: s.fill_ratio,
            inertia: s.inertia.into(),
            min_abs_pivot: s.min_abs_pivot,
            max_abs_pivot: s.max_abs_pivot,
            pattern_reused: s.pattern_reused,
            scaling_info: (&s.scaling_info).into(),
            n_tiny: s.n_tiny,
        }
    }
}

#[pymethods]
impl FactorStats {
    fn __repr__(&self) -> String {
        format!(
            "FactorStats(nnz_a={}, nnz_l={}, fill_ratio={:.3}, n_tiny={})",
            self.nnz_a, self.nnz_l, self.fill_ratio, self.n_tiny
        )
    }
}

/// Per-stage breakdown of the factorization prologue (microseconds).
#[pyclass(module = "feral._feral", frozen)]
#[derive(Clone)]
pub struct PrologueBreakdown {
    #[pyo3(get)]
    pub row_map_us: u64,
    #[pyo3(get)]
    pub scaling_us: u64,
    #[pyo3(get)]
    pub scaling_pivot_order_us: u64,
    #[pyo3(get)]
    pub permute_us: u64,
    #[pyo3(get)]
    pub permute_from_triplets_us: u64,
    #[pyo3(get)]
    pub infnorm_tol_us: u64,
    #[pyo3(get)]
    pub symmetric_pattern_us: u64,
    #[pyo3(get)]
    pub setup_us: u64,
}

impl From<&RustPrologueBreakdown> for PrologueBreakdown {
    fn from(p: &RustPrologueBreakdown) -> Self {
        Self {
            row_map_us: p.row_map_us,
            scaling_us: p.scaling_us,
            scaling_pivot_order_us: p.scaling_pivot_order_us,
            permute_us: p.permute_us,
            permute_from_triplets_us: p.permute_from_triplets_us,
            infnorm_tol_us: p.infnorm_tol_us,
            symmetric_pattern_us: p.symmetric_pattern_us,
            setup_us: p.setup_us,
        }
    }
}

/// Timing of one supernode-size bucket.
#[pyclass(module = "feral._feral", frozen)]
#[derive(Clone)]
pub struct BucketStats {
    #[pyo3(get)]
    pub range: String,
    #[pyo3(get)]
    pub count: usize,
    #[pyo3(get)]
    pub sum_us: u64,
    #[pyo3(get)]
    pub pct_of_total: f64,
    #[pyo3(get)]
    pub avg_us: f64,
}

impl From<&RustBucketStats> for BucketStats {
    fn from(b: &RustBucketStats) -> Self {
        Self {
            range: b.range.to_string(),
            count: b.count,
            sum_us: b.sum_us,
            pct_of_total: b.pct_of_total,
            avg_us: b.avg_us,
        }
    }
}

/// Profile of the numeric factorization (requires `profiling=True`).
#[pyclass(module = "feral._feral", frozen)]
#[derive(Clone)]
pub struct ProfileReport {
    #[pyo3(get)]
    pub n_supernodes: usize,
    #[pyo3(get)]
    pub prologue_us: u64,
    #[pyo3(get)]
    pub prologue_breakdown: PrologueBreakdown,
    #[pyo3(get)]
    pub epilogue_us: u64,
    #[pyo3(get)]
    pub loop_us: u64,
    #[pyo3(get)]
    pub total_us: u64,
    #[pyo3(get)]
    pub overhead_pct: f64,
    #[pyo3(get)]
    pub buckets: Vec<BucketStats>,
    #[pyo3(get)]
    pub validation_warnings: Vec<String>,
}

impl From<&RustProfileReport> for ProfileReport {
    fn from(r: &RustProfileReport) -> Self {
        Self {
            n_supernodes: r.n_supernodes,
            prologue_us: r.prologue_us,
            prologue_breakdown: (&r.prologue_breakdown).into(),
            epilogue_us: r.epilogue_us,
            loop_us: r.loop_us,
            total_us: r.total_us,
            overhead_pct: r.overhead_pct,
            buckets: r.buckets.iter().map(BucketStats::from).collect(),
            validation_warnings: r.validation_warnings.clone(),
        }
    }
}

#[pymethods]
impl ProfileReport {
    fn __repr__(&self) -> String {
        format!(
            "ProfileReport(n_supernodes={}, total_us={}, overhead_pct={:.1})",
            self.n_supernodes, self.total_us, self.overhead_pct
        )
    }
}

/// One stage of the symbolic-analysis profile.
#[pyclass(module = "feral._feral", frozen)]
#[derive(Clone)]
pub struct StagePct {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub us: u64,
    #[pyo3(get)]
    pub pct_of_total: f64,
}

impl From<&RustStagePct> for StagePct {
    fn from(s: &RustStagePct) -> Self {
        Self {
            name: s.name.to_string(),
            us: s.us,
            pct_of_total: s.pct_of_total,
        }
    }
}

/// Profile of the symbolic-analysis phase (requires `profiling=True`).
#[pyclass(module = "feral._feral", frozen)]
#[derive(Clone)]
pub struct SymbolicProfileReport {
    #[pyo3(get)]
    pub total_us: u64,
    #[pyo3(get)]
    pub accounted_us: u64,
    #[pyo3(get)]
    pub overhead_pct: f64,
    #[pyo3(get)]
    pub stages: Vec<StagePct>,
    #[pyo3(get)]
    pub validation_warnings: Vec<String>,
}

impl From<&RustSymbolicProfileReport> for SymbolicProfileReport {
    fn from(r: &RustSymbolicProfileReport) -> Self {
        Self {
            total_us: r.total_us,
            accounted_us: r.accounted_us,
            overhead_pct: r.overhead_pct,
            stages: r.stages.iter().map(StagePct::from).collect(),
            validation_warnings: r.validation_warnings.clone(),
        }
    }
}

#[pymethods]
impl SymbolicProfileReport {
    fn __repr__(&self) -> String {
        format!(
            "SymbolicProfileReport(total_us={}, overhead_pct={:.1})",
            self.total_us, self.overhead_pct
        )
    }
}
