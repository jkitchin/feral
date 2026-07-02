//! `Factors`: a read-only snapshot of the numeric LDLᵀ factorization,
//! giving Python access to the assembled `L` (unit lower-triangular,
//! CSC) and the block-diagonal `D`, plus the permutation and scaling
//! vectors needed to relate the factor back to the input matrix.
//!
//! The snapshot is taken eagerly when `Solver.factors()` is called: the
//! pyo3 `Solver` owns its Rust solver by value, and the Rust
//! `factors()` accessor returns a borrow, so we copy the assembled
//! arrays out rather than hold a reference. This keeps `Factors`
//! `Send`-free and decoupled from the solver's lifetime — a later
//! `refactor` does not mutate an already-returned `Factors`.

use feral::numeric::factorize::SparseFactors;

use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::common::ordering_to_str;

/// Read-only snapshot of a numeric LDLᵀ factorization.
///
/// Reconstruction identity (factorization order): with `L` from
/// [`l_csc`], `D` from [`d_blocks`], `perm`/`perm_inv` and the
/// per-row `scaling` vector `s`, the factor satisfies
/// `M = L · D · Lᵀ` where `M[i, j] = s[perm[i]] · A[perm[i], perm[j]] ·
/// s[perm[j]]`. Equivalently `L D Lᵀ = P · (S A S) · Pᵀ`.
#[pyclass(module = "feral._feral", frozen)]
pub struct Factors {
    n: usize,
    nnz: usize,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
    scaling: Vec<f64>,
    needs_refinement: bool,
    ordering: &'static str,
    // Assembled L (CSC, factorization order, explicit unit diagonal).
    l_indptr: Vec<usize>,
    l_indices: Vec<usize>,
    l_values: Vec<f64>,
    // Block-diagonal D (factorization order).
    d_diag: Vec<f64>,
    d_subdiag: Vec<f64>,
}

impl Factors {
    /// Snapshot a borrowed `SparseFactors`. Performs the O(nnz(L))
    /// `ldlt_export` walk once and copies the metadata vectors.
    pub(crate) fn snapshot(f: &SparseFactors) -> Self {
        let export = f.ldlt_export();
        Self {
            n: f.n,
            nnz: f.factor_nnz(),
            perm: f.perm.clone(),
            perm_inv: f.perm_inv.clone(),
            scaling: f.scaling.clone(),
            needs_refinement: f.needs_refinement,
            ordering: ordering_to_str(&f.resolved_method),
            l_indptr: export.l_indptr,
            l_indices: export.l_indices,
            l_values: export.l_values,
            d_diag: export.d_diag,
            d_subdiag: export.d_subdiag,
        }
    }
}

#[pymethods]
impl Factors {
    /// Matrix dimension.
    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    /// Total stored nonzeros in `L` + `D`.
    #[getter]
    fn nnz(&self) -> usize {
        self.nnz
    }

    /// Fill-reducing permutation, factorization order → original index.
    #[getter]
    fn perm<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i64>> {
        self.perm
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray_bound(py)
    }

    /// Inverse permutation, original index → factorization order.
    #[getter]
    fn perm_inv<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i64>> {
        self.perm_inv
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray_bound(py)
    }

    /// Per-row symmetric scaling vector `s` (length `n`).
    #[getter]
    fn scaling<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.scaling.clone().into_pyarray_bound(py)
    }

    /// Whether the factor flagged itself as benefiting from iterative
    /// refinement.
    #[getter]
    fn needs_refinement(&self) -> bool {
        self.needs_refinement
    }

    /// Resolved ordering method ("amd", "metis", ...).
    #[getter]
    fn ordering(&self) -> &'static str {
        self.ordering
    }

    /// Assembled unit lower-triangular `L` as CSC arrays
    /// `(indptr, indices, data)` in **factorization order**. `indptr`
    /// has length `n + 1`; the unit diagonal is stored explicitly.
    #[allow(clippy::type_complexity)]
    fn l_csc<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, numpy::PyArray1<i64>>,
        Bound<'py, numpy::PyArray1<i64>>,
        Bound<'py, numpy::PyArray1<f64>>,
    ) {
        let indptr = self
            .l_indptr
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray_bound(py);
        let indices = self
            .l_indices
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray_bound(py);
        let data = self.l_values.clone().into_pyarray_bound(py);
        (indptr, indices, data)
    }

    /// Block-diagonal `D` as `(d_diag, d_subdiag)` in factorization
    /// order. `d_subdiag[e] != 0` marks the top-left of a 2×2 block
    /// coupling positions `e` and `e + 1`.
    fn d_blocks<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
    ) {
        (
            self.d_diag.clone().into_pyarray_bound(py),
            self.d_subdiag.clone().into_pyarray_bound(py),
        )
    }

    /// Build a `scipy.sparse.csc_matrix` of `L` (factorization order).
    /// Requires SciPy; raises `ImportError` with an install hint if it
    /// is not available.
    fn to_scipy_l<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let scipy = PyModule::import_bound(py, "scipy.sparse").map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(
                "scipy is required for to_scipy_l(); install it with `pip install scipy`",
            )
        })?;
        let indptr = self
            .l_indptr
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray_bound(py);
        let indices = self
            .l_indices
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray_bound(py);
        let data = self.l_values.clone().into_pyarray_bound(py);
        let csc = scipy.getattr("csc_matrix")?;
        let tuple = (data, indices, indptr);
        let kwargs = pyo3::types::PyDict::new_bound(py);
        kwargs.set_item("shape", (self.n, self.n))?;
        let mat = csc.call((tuple,), Some(&kwargs))?;
        Ok(mat.into_py(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "Factors(n={}, nnz={}, ordering={:?}, needs_refinement={})",
            self.n, self.nnz, self.ordering, self.needs_refinement
        )
    }
}
