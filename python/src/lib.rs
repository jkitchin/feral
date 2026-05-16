//! PyO3 bindings for the feral sparse symmetric indefinite direct solver.
//!
//! See `python/README.md` for the user-facing documentation and
//! `dev/plans/python-interface.md` for the scoping plan.

use feral::error::FeralError as RustFeralError;
use feral::inertia::Inertia as RustInertia;
use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus as RustFactorStatus, QualityLevel as RustQualityLevel, Solver as RustSolver};
use feral::scaling::ScalingStrategy;
use feral::sparse::csc::CscMatrix as RustCscMatrix;
use feral::symbolic::SupernodeParams;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

/// Copy a 1-D numpy view into a contiguous `Vec<f64>`. Accepts any
/// strided layout — the small copy is cheap relative to factor/solve.
fn array1_to_vec<'py>(arr: &PyReadonlyArray1<'py, f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

fn array1_i64_to_vec_usize<'py>(arr: &PyReadonlyArray1<'py, i64>) -> PyResult<Vec<usize>> {
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
// Exception hierarchy
// ----------------------------------------------------------------------

create_exception!(_feral, FeralError, PyException);
create_exception!(_feral, FactorError, FeralError);
create_exception!(_feral, SingularError, FactorError);
create_exception!(_feral, WrongInertiaError, FactorError);
create_exception!(_feral, NumericFailure, FactorError);
create_exception!(_feral, SolveError, FeralError);
create_exception!(_feral, PatternMismatch, FeralError);
create_exception!(_feral, FeralIOError, FeralError);

fn map_feral_err(e: RustFeralError) -> PyErr {
    match e {
        RustFeralError::NumericallyRankDeficient => {
            SingularError::new_err("matrix is numerically rank-deficient")
        }
        RustFeralError::InvalidInput(s) => PyValueError::new_err(s),
        RustFeralError::DimensionMismatch { expected, got } => {
            SolveError::new_err(format!("dimension mismatch: expected {expected}, got {got}"))
        }
        RustFeralError::IoError(s) => FeralIOError::new_err(s),
        RustFeralError::NoFactor => SolveError::new_err(
            "no factorization available; call Solver.factor() first",
        ),
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
        Self { n_pos, n_neg, n_zero }
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
struct InertiaIter {
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

// ----------------------------------------------------------------------
// FactorStatus (Python IntEnum lives in __init__.py; the Rust side
// returns an `int` whose value matches the IntEnum codes.)
// ----------------------------------------------------------------------

const STATUS_SUCCESS: i32 = 0;
const STATUS_SINGULAR: i32 = 1;
const STATUS_WRONG_INERTIA: i32 = 2;
const STATUS_NUMERIC_FAILURE: i32 = 3;

// ----------------------------------------------------------------------
// QualityLevel codes (matching the Python IntEnum).
// ----------------------------------------------------------------------

const QUALITY_BASELINE: i32 = 0;
const QUALITY_SCALING_ENABLED: i32 = 1;
const QUALITY_PIVOT_RAISED: i32 = 2;
const QUALITY_EXHAUSTED: i32 = 3;

fn quality_to_int(q: RustQualityLevel) -> i32 {
    match q {
        RustQualityLevel::Baseline => QUALITY_BASELINE,
        RustQualityLevel::ScalingEnabled => QUALITY_SCALING_ENABLED,
        RustQualityLevel::PivotRaised => QUALITY_PIVOT_RAISED,
        RustQualityLevel::Exhausted => QUALITY_EXHAUSTED,
    }
}

// ----------------------------------------------------------------------
// CscMatrix
// ----------------------------------------------------------------------

/// Sparse symmetric matrix in lower-triangular CSC format.
///
/// Only the lower triangle is stored. Construct via the classmethods
/// `from_scipy`, `from_triplet`, `from_dense`, or `from_mtx`.
#[pyclass(module = "feral._feral")]
pub struct CscMatrix {
    pub inner: RustCscMatrix,
}

impl CscMatrix {
    pub(crate) fn inner(&self) -> &RustCscMatrix {
        &self.inner
    }
}

#[pymethods]
impl CscMatrix {
    /// Build a CscMatrix from raw CSC arrays. The matrix must be square
    /// (`indptr.len() == n + 1`) and contain only lower-triangle entries
    /// (`row_idx[k] >= col` for every k in column `col`).
    #[new]
    #[pyo3(signature = (n, indptr, row_idx, values))]
    fn new<'py>(
        n: usize,
        indptr: PyReadonlyArray1<'py, i64>,
        row_idx: PyReadonlyArray1<'py, i64>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Self> {
        let ip_view = indptr.as_array();
        let ri_view = row_idx.as_array();
        let vs_view = values.as_array();
        let ip: Vec<i64> = ip_view.iter().copied().collect();
        let ri: Vec<i64> = ri_view.iter().copied().collect();
        let vs: Vec<f64> = vs_view.iter().copied().collect();
        if ip.len() != n + 1 {
            return Err(PyValueError::new_err(format!(
                "indptr length must be n+1 = {}, got {}",
                n + 1,
                ip.len()
            )));
        }
        if ri.len() != vs.len() {
            return Err(PyValueError::new_err(format!(
                "row_idx and values must have the same length ({} vs {})",
                ri.len(),
                vs.len()
            )));
        }
        if ip[n] as usize != ri.len() {
            return Err(PyValueError::new_err(format!(
                "indptr[n]={} disagrees with nnz={}",
                ip[n],
                ri.len()
            )));
        }
        let col_ptr: Vec<usize> = ip.iter().map(|&x| x as usize).collect();
        let mut rows: Vec<usize> = Vec::with_capacity(ri.len());
        for j in 0..n {
            let s = col_ptr[j];
            let e = col_ptr[j + 1];
            for k in s..e {
                let r = ri[k];
                if r < 0 || (r as usize) >= n {
                    return Err(PyValueError::new_err(format!(
                        "row_idx[{k}] = {r} out of bounds for n = {n}"
                    )));
                }
                if (r as usize) < j {
                    return Err(PyValueError::new_err(format!(
                        "entry ({r}, {j}) is in the upper triangle; only the lower triangle is stored"
                    )));
                }
                rows.push(r as usize);
            }
        }
        Ok(Self {
            inner: RustCscMatrix {
                n,
                col_ptr,
                row_idx: rows,
                values: vs,
            },
        })
    }

    /// Build a CscMatrix from triplet (COO) arrays. Entries with
    /// `row < col` are rejected; duplicates within the same `(row, col)`
    /// are summed.
    #[classmethod]
    #[pyo3(signature = (n, rows, cols, vals))]
    fn from_triplet<'py>(
        _cls: &Bound<'py, PyType>,
        n: usize,
        rows: PyReadonlyArray1<'py, i64>,
        cols: PyReadonlyArray1<'py, i64>,
        vals: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Self> {
        let r_u = array1_i64_to_vec_usize(&rows)?;
        let c_u = array1_i64_to_vec_usize(&cols)?;
        let vs = array1_to_vec(&vals);
        if r_u.len() != c_u.len() || c_u.len() != vs.len() {
            return Err(PyValueError::new_err(
                "rows, cols, vals must have the same length",
            ));
        }
        let inner = RustCscMatrix::from_triplets(n, &r_u, &c_u, &vs).map_err(map_feral_err)?;
        Ok(Self { inner })
    }

    /// Build a CscMatrix from a dense numpy array. The array must be
    /// square and symmetric; only the lower triangle is read.
    #[classmethod]
    fn from_dense<'py>(
        _cls: &Bound<'py, PyType>,
        a: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Self> {
        let arr = a.as_array();
        let shape = arr.shape();
        if shape[0] != shape[1] {
            return Err(PyValueError::new_err(format!(
                "expected square matrix, got shape {:?}",
                shape
            )));
        }
        let n = shape[0];
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for j in 0..n {
            for i in j..n {
                let v = arr[[i, j]];
                if v != 0.0 {
                    rows.push(i);
                    cols.push(j);
                    vals.push(v);
                }
            }
        }
        let inner = RustCscMatrix::from_triplets(n, &rows, &cols, &vals).map_err(map_feral_err)?;
        Ok(Self { inner })
    }

    /// Read a Matrix Market `.mtx` file. Wraps `feral::io::mtx::read_mtx`.
    #[classmethod]
    fn from_mtx(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let mtx = feral::io::mtx::read_mtx(std::path::Path::new(path))
            .map_err(|e| FeralIOError::new_err(format!("{e}")))?;
        let csc = mtx.to_csc().map_err(map_feral_err)?;
        Ok(Self { inner: csc })
    }

    /// Dimension `n`.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    /// Number of stored nonzeros (lower triangle only).
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.row_idx.len()
    }

    /// Column pointers as a `numpy.int64` array of length `n + 1`.
    fn indptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let v: Vec<i64> = self.inner.col_ptr.iter().map(|&x| x as i64).collect();
        v.into_pyarray_bound(py)
    }

    /// Row indices as a `numpy.int64` array of length `nnz`.
    fn row_idx<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let v: Vec<i64> = self.inner.row_idx.iter().map(|&x| x as i64).collect();
        v.into_pyarray_bound(py)
    }

    /// Values as a `numpy.float64` array of length `nnz`. The returned
    /// array is a copy; mutating it does not affect the matrix. To
    /// update the values in-place for a fast IPM refactor, use
    /// `set_values` (or pass the updated values to `Solver.refactor`
    /// via a new CscMatrix that shares the same pattern).
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.values.clone().into_pyarray_bound(py)
    }

    /// In-place value update. `new_values.len()` must equal `nnz`.
    /// The sparsity pattern is left unchanged. Returns `self` so the
    /// call can be chained with `Solver.refactor`.
    fn set_values<'py>(
        mut slf: PyRefMut<'py, Self>,
        new_values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let nv = array1_to_vec(&new_values);
        if nv.len() != slf.inner.values.len() {
            return Err(PyValueError::new_err(format!(
                "new_values length {} != nnz {}",
                nv.len(),
                slf.inner.values.len()
            )));
        }
        slf.inner.values.copy_from_slice(&nv);
        Ok(slf)
    }

    /// Symmetric matrix–vector product `y = A · x`. Returns a fresh
    /// `numpy.float64` array of length `n`. Accepts non-contiguous
    /// input (slices of larger arrays) — a copy is taken when needed.
    fn symv<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let xv = x.as_array();
        if xv.len() != self.inner.n {
            return Err(PyValueError::new_err(format!(
                "x length {} != n {}",
                xv.len(),
                self.inner.n
            )));
        }
        let xs: Vec<f64> = xv.iter().copied().collect();
        let mut y = vec![0.0f64; self.inner.n];
        self.inner.symv(&xs, &mut y);
        Ok(y.into_pyarray_bound(py))
    }

    /// Compute `||A · x - b||_∞ / ||b||_∞`. Accepts non-contiguous
    /// arrays.
    fn relative_residual<'py>(
        &self,
        x: PyReadonlyArray1<'py, f64>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let xv = x.as_array();
        let bv = b.as_array();
        if xv.len() != self.inner.n || bv.len() != self.inner.n {
            return Err(PyValueError::new_err("x and b must have length n"));
        }
        let xs: Vec<f64> = xv.iter().copied().collect();
        let mut ax = vec![0.0f64; self.inner.n];
        self.inner.symv(&xs, &mut ax);
        let mut max_r: f64 = 0.0;
        let mut max_b: f64 = 0.0;
        for (axi, &bi) in ax.iter().zip(bv.iter()) {
            max_r = max_r.max((axi - bi).abs());
            max_b = max_b.max(bi.abs());
        }
        Ok(if max_b > 0.0 { max_r / max_b } else { max_r })
    }

    fn __repr__(&self) -> String {
        format!(
            "CscMatrix(n={}, nnz={})",
            self.inner.n,
            self.inner.row_idx.len()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.n
    }
}

// ----------------------------------------------------------------------
// Solver
// ----------------------------------------------------------------------

fn pick_scaling(name: &str) -> PyResult<ScalingStrategy> {
    match name {
        "auto" | "default" => Ok(ScalingStrategy::default()),
        "none" | "identity" => Ok(ScalingStrategy::Identity),
        "infnorm" | "inf_norm" | "equilibration" => Ok(ScalingStrategy::InfNorm),
        "mc64" | "mc64_symmetric" => Ok(ScalingStrategy::Mc64Symmetric),
        other => Err(PyValueError::new_err(format!(
            "unknown scaling '{other}'; valid options: auto, none, infnorm, mc64"
        ))),
    }
}

/// Stateful sparse symmetric indefinite direct solver.
///
/// Mirrors `feral::numeric::solver::Solver`. The solver owns its
/// quality-escalation state, a cached symbolic factorization (reused
/// across `factor`/`refactor` calls on matrices with the same sparsity
/// pattern — the IPM use case), and a rayon `ThreadPool` for the
/// parallel multifrontal driver.
///
/// Not thread-safe across concurrent `factor`/`solve` from multiple
/// Python threads; use one `Solver` per thread.
#[pyclass(module = "feral._feral", unsendable)]
pub struct Solver {
    inner: RustSolver,
    last_pattern: Option<(usize, usize, u64)>,
}

impl Solver {
    fn pattern_signature(m: &RustCscMatrix) -> (usize, usize, u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        m.col_ptr.hash(&mut h);
        m.row_idx.hash(&mut h);
        (m.n, m.row_idx.len(), h.finish())
    }
}

#[pymethods]
impl Solver {
    /// Construct a new solver with optional configuration.
    ///
    /// - `parallel`: dispatch the rayon-parallel multifrontal driver
    ///   when work warrants it. Default `True`.
    /// - `fma`: opt-in FMA dispatch on dense kernels. Default `False`.
    /// - `static_pivoting`: force-accept failing pivots in place rather
    ///   than delaying up the elimination tree. Default `False`.
    /// - `cascade_break_ratio`, `cascade_break_eps`: opt-in cascade-break
    ///   knobs. Both default `None` (off). See
    ///   `dev/research/cascade-break-l-perturbation-2026-05-15.md`.
    /// - `scaling`: one of `"auto"`, `"none"`, `"infnorm"`, `"mc64"`.
    ///   Default `"auto"`.
    /// - `pivot_threshold`: BK column-relative pivot threshold. Default
    ///   uses `NumericParams::default()` (MA27-style 1e-8).
    #[new]
    #[pyo3(signature = (
        *,
        parallel = true,
        fma = false,
        static_pivoting = false,
        cascade_break_ratio = None,
        cascade_break_eps = None,
        scaling = "auto",
        pivot_threshold = None,
    ))]
    fn new(
        parallel: bool,
        fma: bool,
        static_pivoting: bool,
        cascade_break_ratio: Option<f64>,
        cascade_break_eps: Option<f64>,
        scaling: &str,
        pivot_threshold: Option<f64>,
    ) -> PyResult<Self> {
        let mut np = NumericParams::default();
        np.fma = fma;
        np.allow_delayed_pivots = !static_pivoting;
        np.cascade_break_ratio = cascade_break_ratio;
        np.cascade_break_eps = cascade_break_eps;
        np.scaling = pick_scaling(scaling)?;
        if let Some(pt) = pivot_threshold {
            np.bk.pivot_threshold = pt;
        }
        let inner = RustSolver::with_params(np, SupernodeParams::default()).with_parallel(parallel);
        Ok(Self { inner, last_pattern: None })
    }

    /// Factor `A`. If `expected_inertia` is provided and disagrees with
    /// the actual inertia, returns `FactorStatus.WRONG_INERTIA` (the
    /// factor is still stored — `solve` will proceed). On fatal errors
    /// raises `NumericFailure`.
    #[pyo3(signature = (a, *, expected_inertia = None))]
    fn factor(
        &mut self,
        py: Python<'_>,
        a: &CscMatrix,
        expected_inertia: Option<&Inertia>,
    ) -> PyResult<(i32, Option<Inertia>)> {
        let expected_rust = expected_inertia.map(RustInertia::from);
        let sig = Self::pattern_signature(a.inner());
        let status = py.allow_threads(|| self.inner.factor(a.inner(), expected_rust));
        self.last_pattern = Some(sig);
        match status {
            RustFactorStatus::Success => Ok((STATUS_SUCCESS, self.inner.inertia().cloned().map(Into::into))),
            RustFactorStatus::Singular => Ok((STATUS_SINGULAR, self.inner.inertia().cloned().map(Into::into))),
            RustFactorStatus::WrongInertia { actual, expected: _ } => {
                Ok((STATUS_WRONG_INERTIA, Some(actual.into())))
            }
            RustFactorStatus::FatalError(e) => Err(match e {
                RustFeralError::NumericallyRankDeficient => SingularError::new_err(format!("{e}")),
                RustFeralError::InvalidInput(s) => PyValueError::new_err(s),
                other => NumericFailure::new_err(format!("{other}")),
            }),
        }
    }

    /// Re-factor with new values on the same sparsity pattern. Raises
    /// `PatternMismatch` if the pattern differs from the previous
    /// `factor`/`refactor` call. The symbolic factorization is reused.
    #[pyo3(signature = (a, *, expected_inertia = None))]
    fn refactor(
        &mut self,
        py: Python<'_>,
        a: &CscMatrix,
        expected_inertia: Option<&Inertia>,
    ) -> PyResult<(i32, Option<Inertia>)> {
        let new_sig = Self::pattern_signature(a.inner());
        if let Some(old) = self.last_pattern {
            if old != new_sig {
                return Err(PatternMismatch::new_err(
                    "refactor called with a different sparsity pattern; \
                     use factor() instead, or build a new CscMatrix with the same pattern",
                ));
            }
        }
        self.factor(py, a, expected_inertia)
    }

    /// Solve `A · x = b` against the stored factor. `b` may be a 1-D
    /// array of length `n` or a 2-D `(n, nrhs)` array (one column per
    /// RHS). Returns a numpy array of the same shape as `b`.
    fn solve<'py>(
        &self,
        py: Python<'py>,
        b: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        // 2-D path
        if let Ok(arr2) = b.extract::<PyReadonlyArray2<'py, f64>>() {
            let view = arr2.as_array();
            let shape = view.shape();
            let n = shape[0];
            let nrhs = shape[1];
            // Pack column-major
            let mut buf = vec![0.0f64; n * nrhs];
            for j in 0..nrhs {
                for i in 0..n {
                    buf[j * n + i] = view[[i, j]];
                }
            }
            let out = py
                .allow_threads(|| self.inner.solve_many(&buf, nrhs))
                .map_err(map_feral_err)?;
            // Reshape column-major to (n, nrhs) row-major numpy
            let mut np_out = vec![0.0f64; n * nrhs];
            for j in 0..nrhs {
                for i in 0..n {
                    np_out[i * nrhs + j] = out[j * n + i];
                }
            }
            let arr = PyArray2::from_vec2_bound(
                py,
                &(0..n)
                    .map(|i| np_out[i * nrhs..(i + 1) * nrhs].to_vec())
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(arr.into_py(py));
        }
        // 1-D path
        let arr1: PyReadonlyArray1<'py, f64> = b.extract()?;
        let bs_vec = array1_to_vec(&arr1);
        let bs = bs_vec.as_slice();
        let x = py
            .allow_threads(|| self.inner.solve(bs))
            .map_err(map_feral_err)?;
        Ok(x.into_pyarray_bound(py).into_py(py))
    }

    /// Solve with iterative refinement against `a` and the stored
    /// factor. Same shape conventions as `solve`. Default `max_iter`
    /// and `tol` mirror the Rust defaults.
    #[pyo3(signature = (a, b))]
    fn solve_refined<'py>(
        &self,
        py: Python<'py>,
        a: &CscMatrix,
        b: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        if let Ok(arr2) = b.extract::<PyReadonlyArray2<'py, f64>>() {
            let view = arr2.as_array();
            let shape = view.shape();
            let n = shape[0];
            let nrhs = shape[1];
            let mut buf = vec![0.0f64; n * nrhs];
            for j in 0..nrhs {
                for i in 0..n {
                    buf[j * n + i] = view[[i, j]];
                }
            }
            let out = py
                .allow_threads(|| self.inner.solve_many_refined(a.inner(), &buf, nrhs))
                .map_err(map_feral_err)?;
            let mut np_out = vec![0.0f64; n * nrhs];
            for j in 0..nrhs {
                for i in 0..n {
                    np_out[i * nrhs + j] = out[j * n + i];
                }
            }
            let arr = PyArray2::from_vec2_bound(
                py,
                &(0..n)
                    .map(|i| np_out[i * nrhs..(i + 1) * nrhs].to_vec())
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(arr.into_py(py));
        }
        let arr1: PyReadonlyArray1<'py, f64> = b.extract()?;
        let bs_vec = array1_to_vec(&arr1);
        let bs = bs_vec.as_slice();
        let x = py
            .allow_threads(|| self.inner.solve_refined(a.inner(), bs))
            .map_err(map_feral_err)?;
        Ok(x.into_pyarray_bound(py).into_py(py))
    }

    /// Hager-Higham 1-norm condition estimate of `A`. Requires a
    /// stored factor — call `factor` first.
    fn estimate_condition_1norm(&self, py: Python<'_>, a: &CscMatrix) -> PyResult<f64> {
        py.allow_threads(|| self.inner.estimate_condition_1norm(a.inner()))
            .map_err(map_feral_err)
    }

    /// Two-stage quality escalation. Returns `False` if both stages
    /// are exhausted.
    fn increase_quality(&mut self) -> bool {
        self.inner.increase_quality()
    }

    // ---- properties ----

    /// Inertia of the last successful factor, or `None`.
    #[getter]
    fn inertia(&self) -> Option<Inertia> {
        self.inner.inertia().cloned().map(Into::into)
    }

    /// Number of negative eigenvalues from the last factor. Returns
    /// `None` if no factor is stored (the Rust API panics; the Python
    /// binding returns `None` for safety).
    #[getter]
    fn num_negative_eigenvalues(&self) -> Option<usize> {
        self.inner.inertia().map(|i| i.negative)
    }

    /// Minimum eigenvalue of D over the last factor's pivots, or `None`.
    #[getter]
    fn min_diagonal(&self) -> Option<f64> {
        self.inner.min_diagonal()
    }

    /// Current quality-escalation level (matches `QualityLevel`
    /// IntEnum codes).
    #[getter]
    fn quality_level(&self) -> i32 {
        quality_to_int(self.inner.quality_level())
    }

    /// Always `True` for feral.
    #[getter]
    fn provides_inertia(&self) -> bool {
        self.inner.provides_inertia()
    }

    /// Total number of symbolic-analysis calls. Increments on the
    /// first `factor` and on every subsequent `factor` whose pattern
    /// differs from the cached one. Stays at 1 across pure IPM
    /// refactor loops.
    #[getter]
    fn symbolic_call_count(&self) -> usize {
        self.inner.symbolic_call_count()
    }

    /// True if the last factor's diagnostic flag flagged the result
    /// as benefiting from iterative refinement (e.g. cascade-break
    /// perturbations were applied).
    #[getter]
    fn needs_refinement(&self) -> bool {
        self.inner
            .factors()
            .map(|f| f.needs_refinement)
            .unwrap_or(false)
    }

    /// Current BK pivot threshold.
    #[getter]
    fn pivot_threshold(&self) -> f64 {
        self.inner.pivot_threshold()
    }

    /// Whether the parallel multifrontal driver is enabled.
    #[getter]
    fn parallel(&self) -> bool {
        self.inner.parallel()
    }

    /// Total stored nonzeros in L + D after the last factor; `None`
    /// if no factor is stored.
    #[getter]
    fn factor_nnz(&self) -> Option<usize> {
        self.inner.factors().map(|f| f.factor_nnz())
    }

    /// String representation of the configured scaling strategy.
    #[getter]
    fn scaling(&self) -> String {
        format!("{:?}", self.inner.scaling_strategy())
    }

    fn __repr__(&self) -> String {
        format!(
            "Solver(parallel={}, scaling={}, pivot_threshold={:.3e})",
            self.inner.parallel(),
            format!("{:?}", self.inner.scaling_strategy()),
            self.inner.pivot_threshold()
        )
    }

    fn __enter__<'py>(slf: PyRefMut<'py, Self>) -> PyRefMut<'py, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        // Drop and rebuild a fresh inner solver so the cached factor
        // and rayon pool are released deterministically.
        let np = NumericParams::default();
        let _ = std::mem::replace(
            &mut self.inner,
            RustSolver::with_params(np, SupernodeParams::default()),
        );
        self.last_pattern = None;
        false
    }
}

// ----------------------------------------------------------------------
// Module init
// ----------------------------------------------------------------------

#[pymodule]
fn _feral(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Inertia>()?;
    m.add_class::<CscMatrix>()?;
    m.add_class::<Solver>()?;

    // Exceptions
    m.add("FeralError", py.get_type_bound::<FeralError>())?;
    m.add("FactorError", py.get_type_bound::<FactorError>())?;
    m.add("SingularError", py.get_type_bound::<SingularError>())?;
    m.add("WrongInertiaError", py.get_type_bound::<WrongInertiaError>())?;
    m.add("NumericFailure", py.get_type_bound::<NumericFailure>())?;
    m.add("SolveError", py.get_type_bound::<SolveError>())?;
    m.add("PatternMismatch", py.get_type_bound::<PatternMismatch>())?;
    m.add("FeralIOError", py.get_type_bound::<FeralIOError>())?;

    // Status / quality codes (mirrored as IntEnum on the Python side)
    let status = PyDict::new_bound(py);
    status.set_item("SUCCESS", STATUS_SUCCESS)?;
    status.set_item("SINGULAR", STATUS_SINGULAR)?;
    status.set_item("WRONG_INERTIA", STATUS_WRONG_INERTIA)?;
    status.set_item("NUMERIC_FAILURE", STATUS_NUMERIC_FAILURE)?;
    m.add("_STATUS_CODES", status)?;

    let quality = PyDict::new_bound(py);
    quality.set_item("BASELINE", QUALITY_BASELINE)?;
    quality.set_item("SCALING_ENABLED", QUALITY_SCALING_ENABLED)?;
    quality.set_item("PIVOT_RAISED", QUALITY_PIVOT_RAISED)?;
    quality.set_item("EXHAUSTED", QUALITY_EXHAUSTED)?;
    m.add("_QUALITY_CODES", quality)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
