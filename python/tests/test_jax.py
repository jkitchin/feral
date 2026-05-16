"""JAX integration tests for feral.jax."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

import feral  # noqa: E402
import feral.jax as fjax  # noqa: E402


def _build_spd(n: int, seed: int = 0):
    """Build a symmetric positive-definite matrix with a known CSC pattern."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A = A @ A.T + np.eye(n) * 0.5
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for j in range(n):
        for i in range(j, n):
            rows.append(i)
            cols.append(j)
            vals.append(A[i, j])
    M = feral.CscMatrix.from_triplet(
        n,
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(vals, dtype=np.float64),
    )
    pattern = fjax.SparsePattern.from_matrix(M)
    return A, M.values(), pattern


# ---- forward (primal) ----------------------------------------------


def test_solve_matches_numpy():
    A, vals, pattern = _build_spd(6, seed=1)
    b = np.arange(1.0, 7.0)
    x_feral = fjax.solve(jnp.asarray(vals), jnp.asarray(b), pattern=pattern)
    x_ref = np.linalg.solve(A, b)
    np.testing.assert_allclose(np.asarray(x_feral), x_ref, rtol=1e-9)


def test_matvec_matches_numpy():
    A, vals, pattern = _build_spd(5, seed=2)
    x = np.linspace(-1.0, 1.0, 5)
    y_feral = fjax.matvec(jnp.asarray(vals), jnp.asarray(x), pattern=pattern)
    y_ref = A @ x
    np.testing.assert_allclose(np.asarray(y_feral), y_ref, rtol=1e-12)


def test_solve_rejects_float32():
    A, vals, pattern = _build_spd(4, seed=3)
    with pytest.raises(TypeError, match="float64"):
        fjax.solve(
            jnp.asarray(vals, dtype=jnp.float32),
            jnp.asarray(np.ones(4), dtype=jnp.float32),
            pattern=pattern,
        )


# ---- reverse-mode (grad / vjp / jacrev) ----------------------------


def test_grad_b_matches_implicit_formula():
    """∂L/∂b = A^{-T} ∂L/∂x. For L = sum(x), ∂L/∂x = 1, so
    ∂L/∂b = A^{-1} 1."""
    A, vals, pattern = _build_spd(5, seed=4)
    b = np.linspace(0.5, 2.5, 5)

    def loss(b_):
        return jnp.sum(fjax.solve(jnp.asarray(vals), b_, pattern=pattern))

    g = jax.grad(loss)(jnp.asarray(b))
    g_ref = np.linalg.solve(A.T, np.ones_like(b))
    np.testing.assert_allclose(np.asarray(g), g_ref, rtol=1e-9)


def test_grad_values_matches_finite_difference():
    A, vals, pattern = _build_spd(4, seed=5)
    b = np.array([1.0, -2.0, 3.0, -4.0])

    def loss(v_):
        x = fjax.solve(v_, jnp.asarray(b), pattern=pattern)
        return 0.5 * jnp.sum(x * x)

    g = np.asarray(jax.grad(loss)(jnp.asarray(vals)))

    # Finite-difference check (central, eps=1e-5)
    fd = np.zeros_like(vals)
    eps = 1e-5
    for k in range(len(vals)):
        vp = vals.copy()
        vm = vals.copy()
        vp[k] += eps
        vm[k] -= eps
        fp = float(loss(jnp.asarray(vp)))
        fm = float(loss(jnp.asarray(vm)))
        fd[k] = (fp - fm) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


def test_jacrev_b_is_A_inv():
    A, vals, pattern = _build_spd(4, seed=6)
    b = np.zeros(4)

    def f(b_):
        return fjax.solve(jnp.asarray(vals), b_, pattern=pattern)

    J = np.asarray(jax.jacrev(f)(jnp.asarray(b)))
    np.testing.assert_allclose(J, np.linalg.inv(A), rtol=1e-9)


# ---- forward-mode (jvp / jacfwd) -----------------------------------


def test_jvp_matches_finite_difference():
    A, vals, pattern = _build_spd(5, seed=7)
    b = np.linspace(-1.0, 1.0, 5)
    dvals = np.full_like(vals, 0.01)
    db = np.full_like(b, 0.1)

    def f(v_, b_):
        return fjax.solve(v_, b_, pattern=pattern)

    x, dx = jax.jvp(
        f, (jnp.asarray(vals), jnp.asarray(b)), (jnp.asarray(dvals), jnp.asarray(db))
    )

    eps = 1e-6
    x_p = np.asarray(f(jnp.asarray(vals + eps * dvals), jnp.asarray(b + eps * db)))
    x_m = np.asarray(f(jnp.asarray(vals - eps * dvals), jnp.asarray(b - eps * db)))
    fd = (x_p - x_m) / (2 * eps)
    np.testing.assert_allclose(np.asarray(dx), fd, rtol=1e-5, atol=1e-7)


def test_jacfwd_b_is_A_inv():
    A, vals, pattern = _build_spd(4, seed=8)
    b = np.zeros(4)

    def f(b_):
        return fjax.solve(jnp.asarray(vals), b_, pattern=pattern)

    J = np.asarray(jax.jacfwd(f)(jnp.asarray(b)))
    np.testing.assert_allclose(J, np.linalg.inv(A), rtol=1e-9)


# ---- vmap ----------------------------------------------------------


def test_vmap_over_b():
    A, vals, pattern = _build_spd(4, seed=9)
    B = np.stack([np.ones(4), np.arange(4.0), np.array([1.0, -1.0, 1.0, -1.0])])

    def f(b_):
        return fjax.solve(jnp.asarray(vals), b_, pattern=pattern)

    X = np.asarray(jax.vmap(f)(jnp.asarray(B)))
    X_ref = np.stack([np.linalg.solve(A, b) for b in B])
    np.testing.assert_allclose(X, X_ref, rtol=1e-9)


def test_vmap_over_values():
    """Batched factorizations: distinct A matrices, same pattern, same b."""
    A1, v1, pattern = _build_spd(4, seed=10)
    A2, v2, pattern2 = _build_spd(4, seed=11)
    assert pattern == pattern2  # same dense pattern by construction
    V = jnp.stack([jnp.asarray(v1), jnp.asarray(v2)])
    b = jnp.ones(4)

    def f(v_):
        return fjax.solve(v_, b, pattern=pattern)

    X = np.asarray(jax.vmap(f)(V))
    np.testing.assert_allclose(X[0], np.linalg.solve(A1, np.ones(4)), rtol=1e-9)
    np.testing.assert_allclose(X[1], np.linalg.solve(A2, np.ones(4)), rtol=1e-9)


def test_vmap_grad_combination():
    """vmap of grad: per-batch gradients of a scalar loss."""
    A, vals, pattern = _build_spd(4, seed=12)
    B = jnp.stack([jnp.ones(4), jnp.arange(4.0)])

    def loss(b_):
        return jnp.sum(fjax.solve(jnp.asarray(vals), b_, pattern=pattern))

    G = np.asarray(jax.vmap(jax.grad(loss))(B))
    g_ref = np.linalg.solve(A.T, np.ones(4))
    # Loss is linear in b; gradient is the same for every batch element.
    np.testing.assert_allclose(G[0], g_ref, rtol=1e-9)
    np.testing.assert_allclose(G[1], g_ref, rtol=1e-9)


# ---- jit -----------------------------------------------------------


def test_jit_solve():
    A, vals, pattern = _build_spd(5, seed=13)
    b = np.arange(1.0, 6.0)

    @jax.jit
    def f(v_, b_):
        return fjax.solve(v_, b_, pattern=pattern)

    x = np.asarray(f(jnp.asarray(vals), jnp.asarray(b)))
    np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-9)


def test_jit_grad():
    A, vals, pattern = _build_spd(4, seed=14)
    b = np.array([1.0, 2.0, 3.0, 4.0])

    @jax.jit
    def grad_loss(v_, b_):
        return jax.grad(lambda vv: jnp.sum(fjax.solve(vv, b_, pattern=pattern)))(v_)

    g = np.asarray(grad_loss(jnp.asarray(vals), jnp.asarray(b)))
    # Sanity: same as eager grad
    g_eager = np.asarray(
        jax.grad(
            lambda vv: jnp.sum(fjax.solve(vv, jnp.asarray(b), pattern=pattern))
        )(jnp.asarray(vals))
    )
    np.testing.assert_allclose(g, g_eager, rtol=1e-12)


# ---- pattern equality / hashing ------------------------------------


def test_pattern_hashable():
    _, _, p1 = _build_spd(3, seed=15)
    _, _, p2 = _build_spd(3, seed=15)
    assert p1 == p2
    assert hash(p1) == hash(p2)
    # SparsePattern is usable as a dict key (required for jit caching).
    {p1: "ok"}


def test_pattern_mismatch_raises():
    _, vals, pattern = _build_spd(4, seed=16)
    wrong_pattern = fjax.SparsePattern(
        n=4, indices=(0, 1), indptr=(0, 1, 2, 2, 2)
    )
    with pytest.raises(ValueError, match="values last dim"):
        fjax.solve(
            jnp.asarray(vals),
            jnp.zeros(4),
            pattern=wrong_pattern,
        )
