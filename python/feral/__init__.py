"""feral — sparse symmetric indefinite direct solver.

Public API:

- :class:`Solver` — stateful LDL^T factorization with cached symbolic
  analysis (the IPM hot path).
- :class:`CscMatrix` — symmetric matrix in lower-triangular CSC.
- :class:`Inertia` — `(n_pos, n_neg, n_zero)` triple returned by `factor`.
- :class:`FactorStatus`, :class:`QualityLevel` — IntEnums.
- Exception hierarchy rooted at :class:`FeralError`.

The :mod:`feral.ipm` submodule provides the Wächter–Biegler-style
perturbation loop for interior-point KKT solves; that's the layer
discopt-shaped callers should reach for.

See ``python/README.md`` for usage examples.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from . import _feral

# Re-export native classes.
CscMatrix = _feral.CscMatrix
Solver = _feral.Solver
Inertia = _feral.Inertia

# Exception hierarchy.
FeralError = _feral.FeralError
FactorError = _feral.FactorError
SingularError = _feral.SingularError
WrongInertiaError = _feral.WrongInertiaError
NumericFailure = _feral.NumericFailure
SolveError = _feral.SolveError
PatternMismatch = _feral.PatternMismatch
FeralIOError = _feral.FeralIOError

__version__ = _feral.__version__


class FactorStatus(IntEnum):
    """Result of :meth:`Solver.factor`."""

    SUCCESS = _feral._STATUS_CODES["SUCCESS"]
    SINGULAR = _feral._STATUS_CODES["SINGULAR"]
    WRONG_INERTIA = _feral._STATUS_CODES["WRONG_INERTIA"]
    NUMERIC_FAILURE = _feral._STATUS_CODES["NUMERIC_FAILURE"]


class QualityLevel(IntEnum):
    """Two-stage quality-escalation state."""

    BASELINE = _feral._QUALITY_CODES["BASELINE"]
    SCALING_ENABLED = _feral._QUALITY_CODES["SCALING_ENABLED"]
    PIVOT_RAISED = _feral._QUALITY_CODES["PIVOT_RAISED"]
    EXHAUSTED = _feral._QUALITY_CODES["EXHAUSTED"]


# --- scipy.sparse adapter (optional dependency) ----------------------

def from_scipy(a, *, symmetric: str = "lower") -> CscMatrix:
    """Build a :class:`CscMatrix` from a ``scipy.sparse`` matrix.

    Parameters
    ----------
    a : scipy.sparse.spmatrix or scipy.sparse.sparray
        Square symmetric matrix in CSC, CSR, or COO format.
    symmetric : {"lower", "upper", "full"}, default "lower"
        - ``"lower"``: ``a`` already contains only lower-triangle
          entries.
        - ``"upper"``: ``a`` contains only upper-triangle entries; they
          are transposed.
        - ``"full"``: ``a`` is symmetric and stored fully; only the
          lower triangle is read.

    Returns
    -------
    CscMatrix

    Raises
    ------
    ImportError
        If scipy is not installed.
    ValueError
        If ``a`` is not square or ``symmetric`` is invalid.
    """
    try:
        import scipy.sparse as sp
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "scipy is required for from_scipy(); install with "
            "`pip install feral-solver[scipy]`"
        ) from e
    import numpy as np

    if not sp.issparse(a):
        raise ValueError(f"expected scipy.sparse matrix, got {type(a).__name__}")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"expected square matrix, got shape {a.shape}")
    n = a.shape[0]
    coo = a.tocoo()
    rows = np.asarray(coo.row, dtype=np.int64)
    cols = np.asarray(coo.col, dtype=np.int64)
    vals = np.asarray(coo.data, dtype=np.float64)
    if symmetric == "lower":
        mask = rows >= cols
    elif symmetric == "upper":
        rows, cols = cols.copy(), rows.copy()
        mask = rows >= cols
    elif symmetric == "full":
        mask = rows >= cols
    else:
        raise ValueError(
            f"symmetric must be 'lower', 'upper', or 'full'; got {symmetric!r}"
        )
    return CscMatrix.from_triplet(n, rows[mask], cols[mask], vals[mask])


def to_scipy(m: CscMatrix):
    """Convert a :class:`CscMatrix` to a symmetric ``scipy.sparse.csc_matrix``.

    Mirrors the lower triangle to the upper for callers expecting a
    full matrix. Requires scipy.
    """
    try:
        import scipy.sparse as sp
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "scipy is required for to_scipy(); install with "
            "`pip install feral-solver[scipy]`"
        ) from e
    import numpy as np

    indptr = m.indptr()
    row = m.row_idx()
    val = m.values()
    n = m.n
    lower = sp.csc_matrix((val, row, indptr), shape=(n, n))
    diag = sp.diags(lower.diagonal(), shape=(n, n), format="csc")
    return lower + lower.T - diag


__all__ = [
    "CscMatrix",
    "Solver",
    "Inertia",
    "FactorStatus",
    "QualityLevel",
    "FeralError",
    "FactorError",
    "SingularError",
    "WrongInertiaError",
    "NumericFailure",
    "SolveError",
    "PatternMismatch",
    "FeralIOError",
    "from_scipy",
    "to_scipy",
    "__version__",
]
