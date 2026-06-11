//! Unsymmetric LU basis engine (issue #81): `LuMatrix` (a general
//! square sparse matrix in column-major CSC form) and `LuFactor`, an
//! auto-routing LU factorization that picks the dense or sparse engine
//! via `should_use_dense_lu` (overridable with `force_dense`).
//!
//! This is the simplex/IPM "basis matrix" path: factor a square `B`,
//! then `ftran` (solve `B x = b`) / `btran` (solve `Bᵀ x = c`), and
//! product-form `update` a single column without refactoring until the
//! stability budget is reached.

use feral::lu::sparse_matrix::SparseColMatrix as RustSparseColMatrix;
use feral::{
    should_use_dense_lu, DenseLu, LuParams, LuScaling, LuSingularAction, SparseLu, SparseLuSymbolic,
};

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::common::{array1_i64_to_vec_usize, array1_to_vec};
use crate::errors::map_feral_err;

/// Reconstruct the dense column list (`cols[j]` length `m`) of a sparse
/// basis. Used to feed the dense LU engine and the dense `refactor`.
fn dense_columns(a: &RustSparseColMatrix) -> Vec<Vec<f64>> {
    let m = a.m;
    let mut cols = vec![vec![0.0f64; m]; m];
    for (j, col) in cols.iter_mut().enumerate() {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            col[a.row_idx[k]] = a.values[k];
        }
    }
    cols
}

fn pick_lu_scaling(name: &str) -> PyResult<LuScaling> {
    match name {
        "none" => Ok(LuScaling::None),
        "infnorm" | "inf_norm" => Ok(LuScaling::InfNorm),
        "mc64" => Ok(LuScaling::Mc64),
        "mc64_then_infnorm" | "mc64_then_inf_norm" => Ok(LuScaling::Mc64ThenInfNorm),
        other => Err(PyValueError::new_err(format!(
            "unknown LU scaling '{other}'; valid options: none, infnorm, mc64, mc64_then_infnorm"
        ))),
    }
}

/// General square matrix in column-major (CSC) form, the input to
/// `LuFactor`. Unlike `CscMatrix` (symmetric, lower-triangle only),
/// `LuMatrix` stores the full unsymmetric pattern.
#[pyclass(module = "feral._feral")]
pub struct LuMatrix {
    pub(crate) inner: RustSparseColMatrix,
}

#[pymethods]
impl LuMatrix {
    /// Build from raw CSC arrays: `indptr` (length `n+1`), `row_idx`
    /// (sorted ascending within each column), `values`. The matrix must
    /// be square (`n × n`).
    #[new]
    #[pyo3(signature = (n, indptr, row_idx, values))]
    fn new<'py>(
        n: usize,
        indptr: PyReadonlyArray1<'py, i64>,
        row_idx: PyReadonlyArray1<'py, i64>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Self> {
        let col_ptr = array1_i64_to_vec_usize(&indptr)?;
        let rows = array1_i64_to_vec_usize(&row_idx)?;
        let vals = array1_to_vec(&values);
        if col_ptr.len() != n + 1 {
            return Err(PyValueError::new_err(format!(
                "indptr length must be n+1 = {}, got {}",
                n + 1,
                col_ptr.len()
            )));
        }
        if rows.len() != vals.len() {
            return Err(PyValueError::new_err(format!(
                "row_idx and values must have the same length ({} vs {})",
                rows.len(),
                vals.len()
            )));
        }
        let inner = RustSparseColMatrix {
            m: n,
            col_ptr,
            row_idx: rows,
            values: vals,
        };
        inner.validate().map_err(map_feral_err)?;
        Ok(Self { inner })
    }

    /// Build from triplet (COO) arrays. Duplicates within the same
    /// `(row, col)` are summed.
    #[classmethod]
    #[pyo3(signature = (n, rows, cols, vals))]
    fn from_triplet<'py>(
        _cls: &Bound<'py, PyType>,
        n: usize,
        rows: PyReadonlyArray1<'py, i64>,
        cols: PyReadonlyArray1<'py, i64>,
        vals: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Self> {
        let r = array1_i64_to_vec_usize(&rows)?;
        let c = array1_i64_to_vec_usize(&cols)?;
        let v = array1_to_vec(&vals);
        if r.len() != c.len() || c.len() != v.len() {
            return Err(PyValueError::new_err(
                "rows, cols, vals must have the same length",
            ));
        }
        let mut columns: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for k in 0..r.len() {
            if c[k] >= n || r[k] >= n {
                return Err(PyValueError::new_err(format!(
                    "triplet ({}, {}) out of bounds for n = {n}",
                    r[k], c[k]
                )));
            }
            columns[c[k]].push((r[k], v[k]));
        }
        let inner = RustSparseColMatrix::from_sparse_columns(n, &columns).map_err(map_feral_err)?;
        Ok(Self { inner })
    }

    /// Build from a dense square numpy array. Exact zeros are dropped.
    #[classmethod]
    fn from_dense<'py>(_cls: &Bound<'py, PyType>, a: PyReadonlyArray2<'py, f64>) -> PyResult<Self> {
        let arr = a.as_array();
        let shape = arr.shape();
        if shape[0] != shape[1] {
            return Err(PyValueError::new_err(format!(
                "expected square matrix, got shape {:?}",
                shape
            )));
        }
        let n = shape[0];
        let mut columns: Vec<Vec<f64>> = vec![vec![0.0f64; n]; n];
        for j in 0..n {
            for (i, col_i) in columns[j].iter_mut().enumerate() {
                *col_i = arr[[i, j]];
            }
        }
        let inner = RustSparseColMatrix::from_dense_columns(n, &columns).map_err(map_feral_err)?;
        Ok(Self { inner })
    }

    /// Build from a list of dense columns (`columns[j]` is column `j`,
    /// length `n`). There must be exactly `n` columns.
    #[classmethod]
    fn from_columns(_cls: &Bound<'_, PyType>, columns: Vec<Vec<f64>>) -> PyResult<Self> {
        let n = columns.len();
        let inner = RustSparseColMatrix::from_dense_columns(n, &columns).map_err(map_feral_err)?;
        Ok(Self { inner })
    }

    /// Dimension `n`.
    #[getter]
    fn n(&self) -> usize {
        self.inner.m
    }

    /// Number of stored nonzeros.
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Matrix–vector product `y = A · x`.
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let xs = array1_to_vec(&x);
        if xs.len() != self.inner.m {
            return Err(PyValueError::new_err(format!(
                "x length {} != n {}",
                xs.len(),
                self.inner.m
            )));
        }
        let mut y = vec![0.0f64; self.inner.m];
        self.inner.matvec(&xs, &mut y);
        Ok(y.into_pyarray_bound(py))
    }

    /// Transposed matrix–vector product `y = Aᵀ · x`.
    fn matvec_transpose<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let xs = array1_to_vec(&x);
        if xs.len() != self.inner.m {
            return Err(PyValueError::new_err(format!(
                "x length {} != n {}",
                xs.len(),
                self.inner.m
            )));
        }
        let mut y = vec![0.0f64; self.inner.m];
        self.inner.matvec_transpose(&xs, &mut y);
        Ok(y.into_pyarray_bound(py))
    }

    fn __repr__(&self) -> String {
        format!("LuMatrix(n={}, nnz={})", self.inner.m, self.inner.nnz())
    }

    fn __len__(&self) -> usize {
        self.inner.m
    }
}

/// Internal routing: the factorization is held by exactly one engine.
/// `DenseLu` carries a full `n × n` factor, so it is boxed to keep the
/// enum's variants similar in size. The sparse variant remains inline:
/// it is the hot path (every `ftran`/`btran` matches on it) and a
/// further box would add a pointer hop on every solve.
#[allow(clippy::large_enum_variant)]
enum LuInner {
    Dense(Box<DenseLu>),
    Sparse { lu: SparseLu, sym: SparseLuSymbolic },
}

/// LU factorization of a square basis matrix, auto-routed to the dense
/// or sparse engine. `ftran`/`btran` solve against the stored factor;
/// `update` applies a product-form column replacement; `refactor`
/// rebuilds the factor (reusing the column ordering on the sparse path).
#[pyclass(module = "feral._feral", unsendable)]
pub struct LuFactor {
    inner: LuInner,
    m: usize,
}

impl LuFactor {
    #[allow(clippy::too_many_arguments)]
    fn build_params(
        pivot_threshold: f64,
        zero_pivot_tol: f64,
        on_singular: &str,
        perturb_floor: f64,
        max_growth: f64,
        max_updates: usize,
        dense_threshold: usize,
        scaling: &str,
        refine_steps: usize,
        refine_tol: f64,
    ) -> PyResult<LuParams> {
        let on_singular = match on_singular {
            "fail" => LuSingularAction::Fail,
            "perturb" => LuSingularAction::PerturbToEps {
                abs_floor: perturb_floor,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown on_singular '{other}'; valid options: fail, perturb"
                )));
            }
        };
        Ok(LuParams {
            pivot_threshold,
            zero_pivot_tol,
            on_singular,
            max_growth,
            max_updates,
            dense_threshold,
            scaling: pick_lu_scaling(scaling)?,
            refine_steps,
            refine_tol,
        })
    }
}

#[pymethods]
impl LuFactor {
    /// Factor `matrix`. Routing: if `force_dense` is `None` (default),
    /// `should_use_dense_lu(n, nnz, params)` decides; `force_dense=True`
    /// forces the dense engine, `False` the sparse engine.
    #[new]
    #[pyo3(signature = (
        matrix,
        *,
        pivot_threshold = 1.0,
        zero_pivot_tol = 1e-13,
        on_singular = "fail",
        perturb_floor = 1e-13,
        max_growth = 1e8,
        max_updates = 64,
        dense_threshold = 128,
        scaling = "none",
        refine_steps = 0,
        refine_tol = 1e-12,
        force_dense = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        matrix: &LuMatrix,
        pivot_threshold: f64,
        zero_pivot_tol: f64,
        on_singular: &str,
        perturb_floor: f64,
        max_growth: f64,
        max_updates: usize,
        dense_threshold: usize,
        scaling: &str,
        refine_steps: usize,
        refine_tol: f64,
        force_dense: Option<bool>,
    ) -> PyResult<Self> {
        let params = Self::build_params(
            pivot_threshold,
            zero_pivot_tol,
            on_singular,
            perturb_floor,
            max_growth,
            max_updates,
            dense_threshold,
            scaling,
            refine_steps,
            refine_tol,
        )?;
        let a = &matrix.inner;
        let m = a.m;
        let use_dense = force_dense.unwrap_or_else(|| should_use_dense_lu(m, a.nnz(), &params));
        let inner = if use_dense {
            let cols = dense_columns(a);
            let lu = py
                .allow_threads(|| DenseLu::factor(&cols, m, params))
                .map_err(map_feral_err)?;
            LuInner::Dense(Box::new(lu))
        } else {
            let sym = SparseLuSymbolic::analyze(a).map_err(map_feral_err)?;
            let lu = py
                .allow_threads(|| SparseLu::factor(a, &sym, params))
                .map_err(map_feral_err)?;
            LuInner::Sparse { lu, sym }
        };
        Ok(Self { inner, m })
    }

    /// Solve `B x = b` (forward transform). Returns a fresh array.
    fn ftran<'py>(
        &mut self,
        py: Python<'py>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut rhs = array1_to_vec(&b);
        if rhs.len() != self.m {
            return Err(PyValueError::new_err(format!(
                "b length {} != n {}",
                rhs.len(),
                self.m
            )));
        }
        match &mut self.inner {
            LuInner::Dense(lu) => lu.ftran(&mut rhs).map_err(map_feral_err)?,
            LuInner::Sparse { lu, .. } => lu.ftran(&mut rhs).map_err(map_feral_err)?,
        }
        Ok(rhs.into_pyarray_bound(py))
    }

    /// Solve `Bᵀ x = c` (backward transform). Returns a fresh array.
    fn btran<'py>(
        &mut self,
        py: Python<'py>,
        c: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut rhs = array1_to_vec(&c);
        if rhs.len() != self.m {
            return Err(PyValueError::new_err(format!(
                "c length {} != n {}",
                rhs.len(),
                self.m
            )));
        }
        match &mut self.inner {
            LuInner::Dense(lu) => lu.btran(&mut rhs).map_err(map_feral_err)?,
            LuInner::Sparse { lu, .. } => lu.btran(&mut rhs).map_err(map_feral_err)?,
        }
        Ok(rhs.into_pyarray_bound(py))
    }

    /// Replace basis column `slot` with the dense `col` (length `n`)
    /// via a product-form update. Raises `NeedsRefactorError` if the
    /// update budget or stability limit is reached (the factor is left
    /// unchanged — call `refactor`).
    fn update<'py>(&mut self, slot: usize, col: PyReadonlyArray1<'py, f64>) -> PyResult<()> {
        let entering = array1_to_vec(&col);
        if entering.len() != self.m {
            return Err(PyValueError::new_err(format!(
                "col length {} != n {}",
                entering.len(),
                self.m
            )));
        }
        match &mut self.inner {
            LuInner::Dense(lu) => lu.update(slot, &entering).map_err(map_feral_err),
            LuInner::Sparse { lu, .. } => lu.update(slot, &entering).map_err(map_feral_err),
        }
    }

    /// Sparse product-form update: replace column `slot` with the sparse
    /// column given by `rows`/`vals`. Sparse engine only — raises on the
    /// dense engine (use `update`).
    fn update_sparse<'py>(
        &mut self,
        slot: usize,
        rows: PyReadonlyArray1<'py, i64>,
        vals: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        let r = array1_i64_to_vec_usize(&rows)?;
        let v = array1_to_vec(&vals);
        if r.len() != v.len() {
            return Err(PyValueError::new_err(
                "rows and vals must have the same length",
            ));
        }
        let entering: Vec<(usize, f64)> = r.into_iter().zip(v).collect();
        match &mut self.inner {
            LuInner::Sparse { lu, .. } => lu.update_sparse(slot, &entering).map_err(map_feral_err),
            LuInner::Dense(_) => Err(PyValueError::new_err(
                "update_sparse is only available on the sparse LU engine; \
                 use update(slot, col) for the dense engine",
            )),
        }
    }

    /// Rebuild the factorization from `matrix`. On the sparse engine the
    /// existing column ordering is reused; on the dense engine the
    /// columns are re-factored in place. Resets the update counter.
    fn refactor(&mut self, py: Python<'_>, matrix: &LuMatrix) -> PyResult<()> {
        let a = &matrix.inner;
        if a.m != self.m {
            return Err(PyValueError::new_err(format!(
                "refactor matrix dimension {} != n {}",
                a.m, self.m
            )));
        }
        match &mut self.inner {
            LuInner::Dense(lu) => {
                let cols = dense_columns(a);
                py.allow_threads(|| lu.refactor(&cols))
                    .map_err(map_feral_err)
            }
            LuInner::Sparse { lu, sym } => py
                .allow_threads(|| lu.refactor(a, sym))
                .map_err(map_feral_err),
        }
    }

    // ---- introspection ----

    /// Dimension `n`.
    #[getter]
    fn dim(&self) -> usize {
        self.m
    }

    /// `True` if the dense engine was selected, `False` if sparse.
    #[getter]
    fn is_dense(&self) -> bool {
        matches!(self.inner, LuInner::Dense(_))
    }

    /// Nonzeros in the stored L+U factor. Sparse engine only; `None` on
    /// the dense engine (which stores a full `n × n` factor).
    #[getter]
    fn factor_nnz(&self) -> Option<usize> {
        match &self.inner {
            LuInner::Sparse { lu, .. } => Some(lu.factor_nnz()),
            LuInner::Dense(_) => None,
        }
    }

    /// Number of product-form updates applied since the last factor /
    /// refactor.
    #[getter]
    fn updates_since_refactor(&self) -> usize {
        match &self.inner {
            LuInner::Dense(lu) => lu.updates_since_refactor(),
            LuInner::Sparse { lu, .. } => lu.updates_since_refactor(),
        }
    }

    /// Total eta (product-form) operations accumulated. Sparse engine
    /// only; `None` on the dense engine.
    #[getter]
    fn eta_ops(&self) -> Option<usize> {
        match &self.inner {
            LuInner::Sparse { lu, .. } => Some(lu.eta_ops()),
            LuInner::Dense(_) => None,
        }
    }

    /// Eta operations from the most recent solve. Sparse engine only.
    #[getter]
    fn last_eta_ops(&self) -> Option<usize> {
        match &self.inner {
            LuInner::Sparse { lu, .. } => Some(lu.last_eta_ops()),
            LuInner::Dense(_) => None,
        }
    }

    /// Row permutation `P` (in `P A Q = L U`) as a `numpy.int64` array.
    #[getter]
    fn perm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let p: Vec<i64> = match &self.inner {
            LuInner::Dense(lu) => lu.perm().iter().map(|&x| x as i64).collect(),
            LuInner::Sparse { lu, .. } => lu.perm().iter().map(|&x| x as i64).collect(),
        };
        p.into_pyarray_bound(py)
    }

    /// Column permutation `Q` (in `P A Q = L U`) as a `numpy.int64`
    /// array.
    #[getter]
    fn qcol<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let q: Vec<i64> = match &self.inner {
            LuInner::Dense(lu) => lu.qcol().iter().map(|&x| x as i64).collect(),
            LuInner::Sparse { lu, .. } => lu.qcol().iter().map(|&x| x as i64).collect(),
        };
        q.into_pyarray_bound(py)
    }

    /// Element `L[i, j]` of the unit-lower-triangular factor (factored
    /// coordinates).
    fn l_dense(&self, i: usize, j: usize) -> f64 {
        match &self.inner {
            LuInner::Dense(lu) => lu.l(i, j),
            LuInner::Sparse { lu, .. } => lu.l_dense(i, j),
        }
    }

    /// Element `U[i, j]` of the upper-triangular factor (factored
    /// coordinates).
    fn u_dense(&self, i: usize, j: usize) -> f64 {
        match &self.inner {
            LuInner::Dense(lu) => lu.u(i, j),
            LuInner::Sparse { lu, .. } => lu.u_dense(i, j),
        }
    }

    /// Dense `n × n` L factor as a 2-D numpy array (factored
    /// coordinates). Built by looping the element accessor.
    fn l_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = self.m;
        let mut rows: Vec<Vec<f64>> = vec![vec![0.0f64; n]; n];
        for (i, row) in rows.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = self.l_dense(i, j);
            }
        }
        PyArray2::from_vec2_bound(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Dense `n × n` U factor as a 2-D numpy array (factored
    /// coordinates).
    fn u_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = self.m;
        let mut rows: Vec<Vec<f64>> = vec![vec![0.0f64; n]; n];
        for (i, row) in rows.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = self.u_dense(i, j);
            }
        }
        PyArray2::from_vec2_bound(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        let engine = if self.is_dense() { "dense" } else { "sparse" };
        format!("LuFactor(n={}, engine={engine})", self.m)
    }
}
