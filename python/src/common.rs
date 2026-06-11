//! Shared helpers and small value types: numpy↔Vec conversions, the
//! `Inertia` class, and the status/quality integer codes mirrored as
//! Python IntEnums in `feral/__init__.py`.

use feral::inertia::Inertia as RustInertia;
use feral::numeric::solver::QualityLevel as RustQualityLevel;
use feral::symbolic::OrderingMethod;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Copy a 1-D numpy view into a contiguous `Vec<f64>`. Accepts any
/// strided layout — the small copy is cheap relative to factor/solve.
pub(crate) fn array1_to_vec(arr: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

/// Copy a 1-D `int64` numpy view into `Vec<usize>`, rejecting negatives.
pub(crate) fn array1_i64_to_vec_usize(arr: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
    let v = arr.as_array();
    let mut out = Vec::with_capacity(v.len());
    for &x in v.iter() {
        if x < 0 {
            return Err(PyValueError::new_err(format!(
                "expected non-negative index, got {x}"
            )));
        }
        out.push(x as usize);
    }
    Ok(out)
}

// ----------------------------------------------------------------------
// FactorStatus (Python IntEnum lives in __init__.py; the Rust side
// returns an `int` whose value matches the IntEnum codes.)
// ----------------------------------------------------------------------

pub(crate) const STATUS_SUCCESS: i32 = 0;
pub(crate) const STATUS_SINGULAR: i32 = 1;
pub(crate) const STATUS_WRONG_INERTIA: i32 = 2;
pub(crate) const STATUS_NUMERIC_FAILURE: i32 = 3;

// ----------------------------------------------------------------------
// QualityLevel codes (matching the Python IntEnum).
// ----------------------------------------------------------------------

pub(crate) const QUALITY_BASELINE: i32 = 0;
pub(crate) const QUALITY_SCALING_ENABLED: i32 = 1;
pub(crate) const QUALITY_PIVOT_RAISED: i32 = 2;
pub(crate) const QUALITY_EXHAUSTED: i32 = 3;

pub(crate) fn quality_to_int(q: RustQualityLevel) -> i32 {
    match q {
        RustQualityLevel::Baseline => QUALITY_BASELINE,
        RustQualityLevel::ScalingEnabled => QUALITY_SCALING_ENABLED,
        RustQualityLevel::PivotRaised => QUALITY_PIVOT_RAISED,
        RustQualityLevel::Exhausted => QUALITY_EXHAUSTED,
    }
}

// ----------------------------------------------------------------------
// OrderingMethod ↔ string mapping (shared by Solver and the standalone
// `analyze`). `OrderingMethod` has no `FromStr`, so the binding owns the
// mapping.
// ----------------------------------------------------------------------

pub(crate) fn ordering_from_str(name: &str) -> PyResult<OrderingMethod> {
    match name {
        "amd" => Ok(OrderingMethod::Amd),
        "amf" => Ok(OrderingMethod::Amf),
        "metis" | "metis_nd" => Ok(OrderingMethod::MetisND),
        "scotch" | "scotch_nd" => Ok(OrderingMethod::ScotchND),
        "kahip" | "kahip_nd" => Ok(OrderingMethod::KahipND),
        "auto" => Ok(OrderingMethod::Auto),
        "auto_race" => Ok(OrderingMethod::AutoRace),
        other => Err(PyValueError::new_err(format!(
            "unknown ordering '{other}'; valid options: amd, amf, metis, scotch, \
             kahip, auto, auto_race"
        ))),
    }
}

pub(crate) fn ordering_to_str(m: OrderingMethod) -> &'static str {
    match m {
        OrderingMethod::Amd => "amd",
        OrderingMethod::Amf => "amf",
        OrderingMethod::MetisND => "metis",
        OrderingMethod::ScotchND => "scotch",
        OrderingMethod::KahipND => "kahip",
        OrderingMethod::Auto => "auto",
        OrderingMethod::AutoRace => "auto_race",
    }
}

// ----------------------------------------------------------------------
// Inertia
// ----------------------------------------------------------------------

#[pyclass(module = "feral._feral", frozen, eq, hash)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Inertia {
    #[pyo3(get)]
    pub n_pos: usize,
    #[pyo3(get)]
    pub n_neg: usize,
    #[pyo3(get)]
    pub n_zero: usize,
}

#[pymethods]
impl Inertia {
    #[new]
    #[pyo3(signature = (n_pos, n_neg, n_zero=0))]
    fn new(n_pos: usize, n_neg: usize, n_zero: usize) -> Self {
        Self {
            n_pos,
            n_neg,
            n_zero,
        }
    }

    /// Total dimension: `n_pos + n_neg + n_zero`.
    #[getter]
    fn n(&self) -> usize {
        self.n_pos + self.n_neg + self.n_zero
    }

    /// True iff `(n_pos, n_neg, n_zero)` agrees with `other`.
    fn matches(&self, other: &Inertia) -> bool {
        self == other
    }

    fn __repr__(&self) -> String {
        format!(
            "Inertia(n_pos={}, n_neg={}, n_zero={})",
            self.n_pos, self.n_neg, self.n_zero
        )
    }

    fn __iter__(slf: PyRef<'_, Self>) -> InertiaIter {
        InertiaIter {
            values: [slf.n_pos, slf.n_neg, slf.n_zero],
            idx: 0,
        }
    }

    fn as_tuple(&self) -> (usize, usize, usize) {
        (self.n_pos, self.n_neg, self.n_zero)
    }
}

#[pyclass]
pub struct InertiaIter {
    values: [usize; 3],
    idx: usize,
}

#[pymethods]
impl InertiaIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<usize> {
        if slf.idx >= 3 {
            return None;
        }
        let v = slf.values[slf.idx];
        slf.idx += 1;
        Some(v)
    }
}

impl From<RustInertia> for Inertia {
    fn from(i: RustInertia) -> Self {
        Self {
            n_pos: i.positive,
            n_neg: i.negative,
            n_zero: i.zero,
        }
    }
}

impl From<&Inertia> for RustInertia {
    fn from(i: &Inertia) -> Self {
        RustInertia::new(i.n_pos, i.n_neg, i.n_zero)
    }
}
