"""feral.jax — JAX integration for the feral sparse symmetric solver.

Differentiable, vmap-able, jit-able sparse symmetric solve via
implicit differentiation.

Supports
--------
- ``jax.grad`` / ``jax.vjp`` / ``jax.jacrev`` — reverse-mode
- ``jax.jvp`` / ``jax.jacfwd``               — forward-mode
- ``jax.vmap``                               — over RHS, values, or both
- ``jax.jit``                                — host-callback under the hood

API
---
    >>> import feral.jax as fjax
    >>> pattern = fjax.SparsePattern.from_csc(n, indices, indptr)
    >>>
    >>> def loss(values, b):
    ...     x = fjax.solve(values, b, pattern=pattern)
    ...     return jnp.sum(x * x)
    >>>
    >>> jax.grad(loss, argnums=(0, 1))(values, b)           # reverse-mode
    >>> jax.jvp(loss, (values, b), (dv, db))                # forward-mode
    >>> jax.vmap(loss, in_axes=(0, None))(batched_v, b)     # batched factors

Implementation
--------------
The solve is exposed as a custom JAX primitive. Forward and reverse
modes both use the standard implicit-differentiation identities:

    x  = A(v)^{-1} b
    dx = A(v)^{-1} (db - dA(dv) x)              (forward; one extra solve)
    ∇_b L = A(v)^{-T} g                         (reverse; one extra solve)
    ∇_v L = -project_pattern(g x^T + x g^T)     (reverse; one pattern outer)

Both extra solves reuse the cached Bunch–Kaufman factor implicitly:
the primitive's eager impl spins up a fresh ``feral.Solver`` per call,
so values+pattern flow through ``feral.Solver.factor`` once per
primitive evaluation. (If you want to share the factor across multiple
RHS in a single trace, vmap over ``b`` instead of issuing multiple
``solve`` calls — the vmap rule packs into a multi-RHS solve.)

Caveats
-------
- **Requires x64 mode.** feral is double-precision only. Enable with
  ``jax.config.update("jax_enable_x64", True)`` before any JAX call.
- **Symmetric only.** A is assumed symmetric, lower-triangular CSC.
- **Host hop per call.** The host callback breaks XLA fusion. One factor
  per call. Good for IPM Newton iterations; bad inside tight scans.
- **vmap over distinct factors loops on the host.** N independent
  symbolic+numeric factorizations, sequentially.
- **No second-order autodiff.** ``jax.hessian`` is not supported in
  this version — the pattern-outer primitive has no JVP rule.

See ``python/examples/jax_quickstart.py`` for an end-to-end example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.core import ShapedArray
    from jax.extend.core import Primitive
    from jax.interpreters import ad, batching, mlir
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "feral.jax requires jax>=0.4.30. "
        "Install with: pip install feral-solver[jax]"
    ) from e

import feral


# --------------------------------------------------------------------
# SparsePattern — static metadata
# --------------------------------------------------------------------


@dataclass(frozen=True)
class SparsePattern:
    """Hashable, immutable symmetric CSC sparsity pattern (lower-triangular).

    Pass as ``pattern=...`` keyword to ``solve`` / ``matvec``. Since the
    pattern is static (not a JAX value), it participates in jit caching
    via its hash — distinct patterns trigger distinct jit specializations.
    """

    n: int
    indices: Tuple[int, ...]
    indptr: Tuple[int, ...]

    @property
    def nnz(self) -> int:
        return len(self.indices)

    @classmethod
    def from_csc(cls, n, indices, indptr) -> "SparsePattern":
        return cls(
            n=int(n),
            indices=tuple(int(i) for i in np.asarray(indices)),
            indptr=tuple(int(i) for i in np.asarray(indptr)),
        )

    @classmethod
    def from_matrix(cls, m: "feral.CscMatrix") -> "SparsePattern":
        return cls.from_csc(m.n, m.row_idx(), m.indptr())


# --------------------------------------------------------------------
# Host kernels (called from impl and from pure_callback under jit)
# --------------------------------------------------------------------


def _build_matrix(values, pattern: SparsePattern) -> "feral.CscMatrix":
    indices = np.asarray(pattern.indices, dtype=np.int64)
    indptr = np.asarray(pattern.indptr, dtype=np.int64)
    vals = np.ascontiguousarray(np.asarray(values), dtype=np.float64)
    return feral.CscMatrix(pattern.n, indptr, indices, vals)


def _host_solve(values, b, pattern: SparsePattern) -> np.ndarray:
    A = _build_matrix(values, pattern)
    solver = feral.Solver()
    solver.factor(A)
    b_np = np.ascontiguousarray(np.asarray(b), dtype=np.float64)
    return np.asarray(solver.solve(b_np))


def _host_symv(values, x, pattern: SparsePattern) -> np.ndarray:
    A = _build_matrix(values, pattern)
    x_np = np.ascontiguousarray(np.asarray(x), dtype=np.float64)
    return np.asarray(A.symv(x_np))


def _host_pattern_outer(
    y, x, pattern: SparsePattern, sign: float = 1.0
) -> np.ndarray:
    """Project ``sign * (y x^T + x y^T)``-style outer product onto the
    symmetric CSC pattern.

    For diagonal entry (i, i): ``sign * y[i] * x[i]``.
    For off-diagonal (row, col) with row > col:
    ``sign * (y[row] * x[col] + y[col] * x[row])``  — both halves of the
    symmetric extension, since one stored CSC entry represents both
    ``A[row,col]`` and ``A[col,row]``.

    This is the gradient of ``<y, A x>`` with respect to the stored CSC
    values (for the symmetric pattern), up to ``sign``.
    """
    indices = np.asarray(pattern.indices, dtype=np.int64)
    indptr = np.asarray(pattern.indptr, dtype=np.int64)
    y_np = np.ascontiguousarray(np.asarray(y), dtype=np.float64).ravel()
    x_np = np.ascontiguousarray(np.asarray(x), dtype=np.float64).ravel()
    out = np.empty(pattern.nnz, dtype=np.float64)
    for col in range(pattern.n):
        for k in range(int(indptr[col]), int(indptr[col + 1])):
            row = int(indices[k])
            if row == col:
                out[k] = sign * y_np[row] * x_np[col]
            else:
                out[k] = sign * (y_np[row] * x_np[col] + y_np[col] * x_np[row])
    return out


# --------------------------------------------------------------------
# Precondition: x64 must be on, otherwise feral receives the wrong dtype
# --------------------------------------------------------------------


def _check_x64() -> None:
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "feral.jax requires float64. Enable with "
            "`jax.config.update('jax_enable_x64', True)` before any JAX call."
        )


# --------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------

_solve_p = Primitive("feral_solve")
_symv_p = Primitive("feral_symv")
_outer_p = Primitive("feral_pattern_outer")


# ---- abstract_eval ----

def _solve_abstract(values_aval, b_aval, *, pattern):
    if b_aval.shape[-1] != pattern.n:
        raise ValueError(
            f"b last dim {b_aval.shape[-1]} != pattern.n {pattern.n}"
        )
    if values_aval.shape[-1] != pattern.nnz:
        raise ValueError(
            f"values last dim {values_aval.shape[-1]} != pattern.nnz {pattern.nnz}"
        )
    return ShapedArray(b_aval.shape, b_aval.dtype)


def _symv_abstract(values_aval, x_aval, *, pattern):
    return ShapedArray(x_aval.shape, x_aval.dtype)


def _outer_abstract(y_aval, x_aval, *, pattern, sign):
    return ShapedArray((pattern.nnz,), y_aval.dtype)


_solve_p.def_abstract_eval(_solve_abstract)
_symv_p.def_abstract_eval(_symv_abstract)
_outer_p.def_abstract_eval(_outer_abstract)


# ---- eager impl (called in eager mode) ----

def _solve_eager(values, b, *, pattern):
    return _host_solve(values, b, pattern)


def _symv_eager(values, x, *, pattern):
    return _host_symv(values, x, pattern)


def _outer_eager(y, x, *, pattern, sign):
    return _host_pattern_outer(y, x, pattern, sign=sign)


_solve_p.def_impl(_solve_eager)
_symv_p.def_impl(_symv_eager)
_outer_p.def_impl(_outer_eager)


# ---- MLIR lowering (under jit, routes through pure_callback) ----

def _solve_lowering(ctx, values, b, *, pattern):
    out_aval = ctx.avals_out[0]
    result_shape = jax.ShapeDtypeStruct(out_aval.shape, out_aval.dtype)

    def host_fn(v, bb):
        return np.asarray(_host_solve(v, bb, pattern), dtype=out_aval.dtype)

    def traced(v, bb):
        return jax.pure_callback(
            host_fn, result_shape, v, bb, vmap_method="sequential"
        )

    return mlir.lower_fun(traced, multiple_results=False)(ctx, values, b)


def _symv_lowering(ctx, values, x, *, pattern):
    out_aval = ctx.avals_out[0]
    result_shape = jax.ShapeDtypeStruct(out_aval.shape, out_aval.dtype)

    def host_fn(v, xx):
        return np.asarray(_host_symv(v, xx, pattern), dtype=out_aval.dtype)

    def traced(v, xx):
        return jax.pure_callback(
            host_fn, result_shape, v, xx, vmap_method="sequential"
        )

    return mlir.lower_fun(traced, multiple_results=False)(ctx, values, x)


def _outer_lowering(ctx, y, x, *, pattern, sign):
    out_aval = ctx.avals_out[0]
    result_shape = jax.ShapeDtypeStruct(out_aval.shape, out_aval.dtype)

    def host_fn(yy, xx):
        return np.asarray(
            _host_pattern_outer(yy, xx, pattern, sign=sign), dtype=out_aval.dtype
        )

    def traced(yy, xx):
        return jax.pure_callback(
            host_fn, result_shape, yy, xx, vmap_method="sequential"
        )

    return mlir.lower_fun(traced, multiple_results=False)(ctx, y, x)


mlir.register_lowering(_solve_p, _solve_lowering)
mlir.register_lowering(_symv_p, _symv_lowering)
mlir.register_lowering(_outer_p, _outer_lowering)


# ---- JVP rules ----

def _is_zero(t) -> bool:
    return type(t) is ad.Zero


def _solve_jvp(primals, tangents, *, pattern):
    """JVP: dx = A^{-1} (db - dA(dv) x)."""
    values, b = primals
    dvalues, db = tangents
    x = _solve_p.bind(values, b, pattern=pattern)

    if _is_zero(dvalues):
        Adx = jnp.zeros_like(x)
    else:
        Adx = _symv_p.bind(dvalues, x, pattern=pattern)

    if _is_zero(db):
        rhs = -Adx
    else:
        rhs = db - Adx

    if _is_zero(dvalues) and _is_zero(db):
        dx = jnp.zeros_like(x)
    else:
        dx = _solve_p.bind(values, rhs, pattern=pattern)
    return x, dx


def _symv_jvp(primals, tangents, *, pattern):
    """JVP: dy = symv(dv, x) + symv(v, dx)."""
    values, x = primals
    dvalues, dx = tangents
    y = _symv_p.bind(values, x, pattern=pattern)
    terms = []
    if not _is_zero(dvalues):
        terms.append(_symv_p.bind(dvalues, x, pattern=pattern))
    if not _is_zero(dx):
        terms.append(_symv_p.bind(values, dx, pattern=pattern))
    if not terms:
        dy = jnp.zeros_like(y)
    elif len(terms) == 1:
        dy = terms[0]
    else:
        dy = terms[0] + terms[1]
    return y, dy


ad.primitive_jvps[_solve_p] = _solve_jvp
ad.primitive_jvps[_symv_p] = _symv_jvp
# _outer_p has no JVP — only used inside transpose results.


# ---- Transpose rules (for reverse-mode AD) ----

def _solve_transpose(cotangent, values, b, *, pattern):
    """Inside the JVP, ``_solve_p.bind(values, rhs)`` is linear in ``rhs``
    only — ``values`` is held as a residual. So transposition is only
    w.r.t. the second argument: ``cot_b = A^{-T} cot = A^{-1} cot`` (sym).
    """
    assert not ad.is_undefined_primal(values), (
        "values must be a concrete residual at solve_p transpose time"
    )
    assert ad.is_undefined_primal(b), (
        "b must be the linear tangent at solve_p transpose time"
    )
    if _is_zero(cotangent):
        return [None, ad.Zero(b.aval)]
    cot_b = _solve_p.bind(values, cotangent, pattern=pattern)
    return [None, cot_b]


def _symv_transpose(cotangent, values, x, *, pattern):
    """Two cases (only one input is the linear tangent at any given call):

    1. ``symv(dvalues, x_const)``: linear in dvalues.
       Transpose:  cot_dvalues = project_pattern(cot, x_const)
    2. ``symv(values_const, dx)``: linear in dx.
       Transpose:  cot_dx = symv(values_const, cot)
    """
    if _is_zero(cotangent):
        return [
            ad.Zero(values.aval) if ad.is_undefined_primal(values) else None,
            ad.Zero(x.aval) if ad.is_undefined_primal(x) else None,
        ]
    if ad.is_undefined_primal(values):
        assert not ad.is_undefined_primal(x)
        cot_values = _outer_p.bind(cotangent, x, pattern=pattern, sign=1.0)
        return [cot_values, None]
    else:
        assert ad.is_undefined_primal(x)
        cot_x = _symv_p.bind(values, cotangent, pattern=pattern)
        return [None, cot_x]


ad.primitive_transposes[_solve_p] = _solve_transpose
ad.primitive_transposes[_symv_p] = _symv_transpose


# ---- Batching (vmap) ----

def _move_to_front(arr, axis):
    if axis is None or axis == 0:
        return arr
    return jnp.moveaxis(arr, axis, 0)


def _solve_batch(args, dims, *, pattern):
    values, b = args
    vdim, bdim = dims
    if vdim is None and bdim is None:
        return _solve_p.bind(values, b, pattern=pattern), None
    values = _move_to_front(values, vdim) if vdim is not None else values
    b = _move_to_front(b, bdim) if bdim is not None else b

    if vdim is not None and bdim is None:
        out = jax.lax.map(
            lambda vi: _solve_p.bind(vi, b, pattern=pattern), values
        )
    elif vdim is None and bdim is not None:
        out = jax.lax.map(
            lambda bi: _solve_p.bind(values, bi, pattern=pattern), b
        )
    else:
        out = jax.lax.map(
            lambda vb: _solve_p.bind(vb[0], vb[1], pattern=pattern),
            (values, b),
        )
    return out, 0


def _symv_batch(args, dims, *, pattern):
    values, x = args
    vdim, xdim = dims
    if vdim is None and xdim is None:
        return _symv_p.bind(values, x, pattern=pattern), None
    values = _move_to_front(values, vdim) if vdim is not None else values
    x = _move_to_front(x, xdim) if xdim is not None else x

    if vdim is not None and xdim is None:
        out = jax.lax.map(
            lambda vi: _symv_p.bind(vi, x, pattern=pattern), values
        )
    elif vdim is None and xdim is not None:
        out = jax.lax.map(
            lambda xi: _symv_p.bind(values, xi, pattern=pattern), x
        )
    else:
        out = jax.lax.map(
            lambda vx: _symv_p.bind(vx[0], vx[1], pattern=pattern),
            (values, x),
        )
    return out, 0


def _outer_batch(args, dims, *, pattern, sign):
    y, x = args
    ydim, xdim = dims
    if ydim is None and xdim is None:
        return _outer_p.bind(y, x, pattern=pattern, sign=sign), None
    y = _move_to_front(y, ydim) if ydim is not None else y
    x = _move_to_front(x, xdim) if xdim is not None else x

    if ydim is not None and xdim is None:
        out = jax.lax.map(
            lambda yi: _outer_p.bind(yi, x, pattern=pattern, sign=sign), y
        )
    elif ydim is None and xdim is not None:
        out = jax.lax.map(
            lambda xi: _outer_p.bind(y, xi, pattern=pattern, sign=sign), x
        )
    else:
        out = jax.lax.map(
            lambda yx: _outer_p.bind(
                yx[0], yx[1], pattern=pattern, sign=sign
            ),
            (y, x),
        )
    return out, 0


batching.primitive_batchers[_solve_p] = _solve_batch
batching.primitive_batchers[_symv_p] = _symv_batch
batching.primitive_batchers[_outer_p] = _outer_batch


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


def solve(values, b, *, pattern: SparsePattern):
    """Differentiable sparse symmetric solve ``A x = b``.

    Parameters
    ----------
    values : jnp.ndarray, shape (nnz,), dtype float64
        Lower-triangular CSC nonzero values of the symmetric A.
    b : jnp.ndarray, shape (n,), dtype float64
        Right-hand side.
    pattern : SparsePattern
        Static sparsity pattern (n, indices, indptr).

    Returns
    -------
    x : jnp.ndarray, shape (n,), dtype float64
        Solution to ``A x = b``.

    Notes
    -----
    Supports ``grad``, ``jvp``, ``vjp``, ``vmap``, ``jit``. The pattern
    is static; only ``values`` and ``b`` participate in autodiff.
    """
    _check_x64()
    if not isinstance(pattern, SparsePattern):
        raise TypeError(f"pattern must be SparsePattern, got {type(pattern)!r}")
    values = jnp.asarray(values)
    b = jnp.asarray(b)
    if values.dtype != jnp.float64 or b.dtype != jnp.float64:
        raise TypeError(
            f"feral.jax requires float64; got values={values.dtype}, b={b.dtype}"
        )
    if values.shape[-1] != pattern.nnz:
        raise ValueError(
            f"values last dim {values.shape[-1]} != pattern.nnz {pattern.nnz}"
        )
    if b.shape[-1] != pattern.n:
        raise ValueError(
            f"b last dim {b.shape[-1]} != pattern.n {pattern.n}"
        )
    return _solve_p.bind(values, b, pattern=pattern)


def matvec(values, x, *, pattern: SparsePattern):
    """Differentiable symmetric matrix–vector product ``y = A x``."""
    _check_x64()
    if not isinstance(pattern, SparsePattern):
        raise TypeError(f"pattern must be SparsePattern, got {type(pattern)!r}")
    values = jnp.asarray(values)
    x = jnp.asarray(x)
    if values.dtype != jnp.float64 or x.dtype != jnp.float64:
        raise TypeError(
            f"feral.jax requires float64; got values={values.dtype}, x={x.dtype}"
        )
    return _symv_p.bind(values, x, pattern=pattern)


__all__ = ["SparsePattern", "solve", "matvec"]
