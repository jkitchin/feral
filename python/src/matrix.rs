//! `CscMatrix`: a sparse symmetric matrix in lower-triangular CSC
//! format, the input type for the LDLᵀ `Solver`.

use feral::sparse::csc::CscMatrix as RustCscMatrix;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::common::{array1_i64_to_vec_usize, array1_to_vec};
use crate::errors::{map_feral_err, FeralIOError};

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
    /// square; which triangle is read is controlled by `triangle`
    /// (`"lower"` default, `"upper"`, or `"full"`). For `"lower"`/
    /// `"upper"` only that triangle is read; for `"full"` the lower
    /// triangle is read and the upper is assumed to mirror it (the
    /// matrix is treated as symmetric). The stored format is always the
    /// lower triangle.
    #[classmethod]
    #[pyo3(signature = (a, *, triangle = "lower"))]
    fn from_dense<'py>(
        _cls: &Bound<'py, PyType>,
        a: PyReadonlyArray2<'py, f64>,
        triangle: &str,
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
        match triangle {
            "lower" | "full" => {
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
            }
            "upper" => {
                // Read the upper triangle A[i,j] (i<=j) and store it as
                // the lower triangle of the symmetric matrix: the (i,j)
                // upper entry becomes the (j,i) lower entry.
                for j in 0..n {
                    for i in 0..=j {
                        let v = arr[[i, j]];
                        if v != 0.0 {
                            rows.push(j);
                            cols.push(i);
                            vals.push(v);
                        }
                    }
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown triangle '{other}'; valid options: lower, upper, full"
                )));
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

    /// Dense `n×n` symmetric reconstruction as a 2-D `numpy.float64`
    /// array. The stored lower triangle is mirrored into the upper
    /// triangle. Allocates a full dense matrix — intended for small
    /// matrices, debugging, and round-tripping with `from_dense`.
    fn to_dense<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let n = self.inner.n;
        let mut rows: Vec<Vec<f64>> = vec![vec![0.0f64; n]; n];
        for j in 0..n {
            let s = self.inner.col_ptr[j];
            let e = self.inner.col_ptr[j + 1];
            for k in s..e {
                let i = self.inner.row_idx[k];
                let v = self.inner.values[k];
                rows[i][j] = v;
                rows[j][i] = v;
            }
        }
        numpy::PyArray2::from_vec2_bound(py, &rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Full symmetric sparsity pattern as `(indptr, indices)` (both
    /// `numpy.int64`), i.e. the union of the stored lower triangle and
    /// its transpose, with sorted row indices per column. Values are not
    /// returned — this is the structural pattern only.
    fn symmetric_pattern<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
        let n = self.inner.n;
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        for j in 0..n {
            let s = self.inner.col_ptr[j];
            let e = self.inner.col_ptr[j + 1];
            for k in s..e {
                let i = self.inner.row_idx[k];
                cols[j].push(i);
                if i != j {
                    cols[i].push(j);
                }
            }
        }
        let mut indptr: Vec<i64> = Vec::with_capacity(n + 1);
        let mut indices: Vec<i64> = Vec::new();
        indptr.push(0);
        for col in cols.iter_mut() {
            col.sort_unstable();
            col.dedup();
            for &r in col.iter() {
                indices.push(r as i64);
            }
            indptr.push(indices.len() as i64);
        }
        (
            indptr.into_pyarray_bound(py),
            indices.into_pyarray_bound(py),
        )
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
