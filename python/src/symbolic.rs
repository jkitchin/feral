//! `SymbolicAnalysis`: a read-only snapshot of the symbolic
//! factorization (fill-reducing ordering, elimination tree, supernode
//! structure and nnz prediction), plus the standalone `analyze`
//! function that runs symbolic analysis with **no** numeric
//! factorization.
//!
//! Like `Factors`, the snapshot is taken eagerly: `Solver.symbolic()`
//! returns a borrow into the solver, so we copy the relevant vectors
//! out into an owned, frozen pyclass.

use feral::symbolic::{symbolic_factorize_with_method, SupernodeParams, SymbolicFactorization};

use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::common::{ordering_from_str, ordering_to_str};
use crate::errors::map_feral_err;
use crate::matrix::CscMatrix;

/// Sentinel parent value for elimination-tree roots (a root has no
/// parent). `-1` is out of range of any valid node index and easy to
/// test for in Python (`parent == -1` ⇒ root).
const ROOT_SENTINEL: i64 = -1;

/// Read-only snapshot of a symbolic factorization.
#[pyclass(module = "feral._feral", frozen)]
pub struct SymbolicAnalysis {
    n: usize,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
    num_supernodes: usize,
    factor_nnz_estimate: usize,
    factor_slack: f64,
    peak_contrib_bytes: usize,
    col_counts: Vec<usize>,
    etree_parent: Vec<i64>,
    ordering: &'static str,
}

impl SymbolicAnalysis {
    pub(crate) fn snapshot(s: &SymbolicFactorization) -> Self {
        let etree_parent = s
            .etree
            .parent
            .iter()
            .map(|p| match p {
                Some(i) => *i as i64,
                None => ROOT_SENTINEL,
            })
            .collect();
        Self {
            n: s.n,
            perm: s.perm.clone(),
            perm_inv: s.perm_inv.clone(),
            num_supernodes: s.supernodes.len(),
            factor_nnz_estimate: s.factor_nnz_estimate,
            factor_slack: s.factor_slack,
            peak_contrib_bytes: s.peak_contrib_bytes,
            col_counts: s.col_counts.clone(),
            etree_parent,
            ordering: ordering_to_str(s.resolved_method),
        }
    }
}

#[pymethods]
impl SymbolicAnalysis {
    /// Matrix dimension.
    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    /// Fill-reducing permutation, new index → original index.
    #[getter]
    fn perm<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i64>> {
        self.perm
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray_bound(py)
    }

    /// Inverse permutation, original index → new index.
    #[getter]
    fn perm_inv<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i64>> {
        self.perm_inv
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray_bound(py)
    }

    /// Number of supernodes.
    #[getter]
    fn num_supernodes(&self) -> usize {
        self.num_supernodes
    }

    /// Predicted total nonzeros in `L` (upper bound used for
    /// allocation; the actual `Factors.nnz` is `<=` this).
    #[getter]
    fn factor_nnz_estimate(&self) -> usize {
        self.factor_nnz_estimate
    }

    /// Slack multiplier applied to `factor_nnz_estimate`.
    #[getter]
    fn factor_slack(&self) -> f64 {
        self.factor_slack
    }

    /// Peak contribution-block pool depth in bytes.
    #[getter]
    fn peak_contrib_bytes(&self) -> usize {
        self.peak_contrib_bytes
    }

    /// Per-column nonzero counts of `L` (permuted order).
    #[getter]
    fn col_counts<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i64>> {
        self.col_counts
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .into_pyarray_bound(py)
    }

    /// Elimination-tree parent array (permuted order). `parent[j]` is
    /// the parent node index, or `-1` if `j` is a root.
    #[getter]
    fn etree_parent<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i64>> {
        self.etree_parent.clone().into_pyarray_bound(py)
    }

    /// Sentinel value used in `etree_parent` to mark roots (`-1`).
    #[getter]
    fn root_sentinel(&self) -> i64 {
        ROOT_SENTINEL
    }

    /// Resolved ordering method ("amd", "metis", ...).
    #[getter]
    fn ordering(&self) -> &'static str {
        self.ordering
    }

    fn __repr__(&self) -> String {
        format!(
            "SymbolicAnalysis(n={}, num_supernodes={}, factor_nnz_estimate={}, ordering={:?})",
            self.n, self.num_supernodes, self.factor_nnz_estimate, self.ordering
        )
    }
}

/// Run symbolic analysis on a symmetric matrix **without** numeric
/// factorization. Returns a :class:`SymbolicAnalysis`.
///
/// `ordering` selects the fill-reducing method
/// ("auto" (default), "amd", "amf", "metis", "scotch", "kahip",
/// "auto_race").
#[pyfunction]
#[pyo3(signature = (a, *, ordering = "auto"))]
pub fn analyze(py: Python<'_>, a: &CscMatrix, ordering: &str) -> PyResult<SymbolicAnalysis> {
    let method = ordering_from_str(ordering)?;
    let params = SupernodeParams::default();
    let sym = py
        .allow_threads(|| symbolic_factorize_with_method(a.inner(), &params, method))
        .map_err(map_feral_err)?;
    Ok(SymbolicAnalysis::snapshot(&sym))
}
