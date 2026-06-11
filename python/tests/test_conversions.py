"""CscMatrix conversion-convenience tests (to_dense / from_dense
triangle / symmetric_pattern)."""

from __future__ import annotations

import numpy as np
import pytest

import feral


def _sym() -> np.ndarray:
    return np.array(
        [
            [4.0, 1.0, 0.0, 2.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 5.0, 1.0],
            [2.0, 0.0, 1.0, 6.0],
        ]
    )


def test_to_dense_roundtrip():
    A = _sym()
    csc = feral.CscMatrix.from_dense(A)
    np.testing.assert_allclose(csc.to_dense(), A, rtol=0, atol=1e-14)


def test_triangle_upper_equals_lower():
    A = _sym()
    lower = feral.CscMatrix.from_dense(A, triangle="lower")
    upper = feral.CscMatrix.from_dense(A, triangle="upper")
    np.testing.assert_allclose(lower.to_dense(), upper.to_dense(), rtol=0, atol=1e-14)


def test_triangle_full_equals_lower():
    A = _sym()
    lower = feral.CscMatrix.from_dense(A, triangle="lower")
    full = feral.CscMatrix.from_dense(A, triangle="full")
    np.testing.assert_allclose(lower.to_dense(), full.to_dense(), rtol=0, atol=1e-14)


def test_triangle_upper_reads_upper_only():
    # An asymmetric storage where lower and upper differ: triangle="upper"
    # must mirror the upper triangle, ignoring the (different) lower one.
    M = np.array([[1.0, 9.0], [0.0, 2.0]])  # upper has 9 at (0,1)
    upper = feral.CscMatrix.from_dense(M, triangle="upper")
    expected = np.array([[1.0, 9.0], [9.0, 2.0]])
    np.testing.assert_allclose(upper.to_dense(), expected, rtol=0, atol=1e-14)


def test_symmetric_pattern_matches_dense():
    A = _sym()
    csc = feral.CscMatrix.from_dense(A)
    indptr, indices = csc.symmetric_pattern()
    n = csc.n
    dense_pat = np.zeros((n, n), dtype=bool)
    for j in range(n):
        for k in range(indptr[j], indptr[j + 1]):
            dense_pat[indices[k], j] = True
    # The full symmetric pattern == nonzeros of the symmetric A.
    expected = A != 0.0
    np.testing.assert_array_equal(dense_pat, expected)


def test_symmetric_pattern_sorted_per_column():
    A = _sym()
    csc = feral.CscMatrix.from_dense(A)
    indptr, indices = csc.symmetric_pattern()
    for j in range(csc.n):
        col = indices[indptr[j] : indptr[j + 1]]
        assert list(col) == sorted(col)


def test_bad_triangle_raises():
    A = _sym()
    with pytest.raises(ValueError):
        feral.CscMatrix.from_dense(A, triangle="diagonal")
