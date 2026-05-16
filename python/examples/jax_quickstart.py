"""End-to-end JAX example for feral.

Builds a small symmetric PD matrix and demonstrates:
    - forward solve
    - jax.grad (reverse-mode) through the solve
    - jax.jvp (forward-mode) through the solve
    - jax.vmap over RHS and over A's values
    - jax.jit composition
"""

from __future__ import annotations

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import feral
import feral.jax as fjax


def main() -> None:
    n = 5
    rng = np.random.default_rng(0)
    H = rng.standard_normal((n, n))
    A_dense = H @ H.T + np.eye(n) * 0.5

    rows, cols, vals = [], [], []
    for j in range(n):
        for i in range(j, n):
            rows.append(i)
            cols.append(j)
            vals.append(A_dense[i, j])
    M = feral.CscMatrix.from_triplet(
        n,
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(vals, dtype=np.float64),
    )
    pattern = fjax.SparsePattern.from_matrix(M)
    values = jnp.asarray(M.values())
    b = jnp.arange(1.0, n + 1)

    print("=== feral.jax quickstart ===\n")

    # --- forward solve ---
    x = fjax.solve(values, b, pattern=pattern)
    print(f"x         = {np.asarray(x)}")
    print(f"||Ax - b|| / ||b|| = {M.relative_residual(np.asarray(x), np.asarray(b)):.2e}\n")

    # --- reverse-mode through the solve ---
    def loss(v_, b_):
        return jnp.sum(fjax.solve(v_, b_, pattern=pattern) ** 2)

    g_v, g_b = jax.grad(loss, argnums=(0, 1))(values, b)
    print(f"grad wrt b (shape {g_b.shape}): {np.asarray(g_b)}")
    print(f"grad wrt values (shape {g_v.shape}): {np.asarray(g_v)}\n")

    # --- forward-mode through the solve ---
    dvalues = jnp.full_like(values, 0.01)
    db = jnp.zeros_like(b)
    x_p, dx = jax.jvp(
        lambda v_, b_: fjax.solve(v_, b_, pattern=pattern),
        (values, b),
        (dvalues, db),
    )
    print(f"jvp: dx (dv=0.01, db=0): {np.asarray(dx)}")
    print(f"      ||dx|| = {float(jnp.linalg.norm(dx)):.3e}\n")

    # --- vmap over RHS ---
    B = jnp.stack([jnp.ones(n), jnp.arange(n, dtype=jnp.float64), b])
    X = jax.vmap(lambda b_: fjax.solve(values, b_, pattern=pattern))(B)
    print(f"vmap over 3 RHS: X.shape = {X.shape}")
    for i, Bi in enumerate(B):
        res = float(jnp.linalg.norm(jnp.asarray(A_dense) @ X[i] - Bi))
        print(f"  rhs {i}: residual = {res:.2e}")
    print()

    # --- vmap over values: independent factorizations of perturbed A's ---
    perturb = 0.01 * rng.standard_normal((4, len(values)))
    V = values[None, :] + jnp.asarray(perturb)
    X = jax.vmap(lambda v_: fjax.solve(v_, b, pattern=pattern))(V)
    print(f"vmap over 4 perturbed factors: X.shape = {X.shape}")
    print(f"  per-row ||x|| = {np.asarray(jnp.linalg.norm(X, axis=1))}\n")

    # --- jit ---
    jitted = jax.jit(lambda v_, b_: fjax.solve(v_, b_, pattern=pattern))
    x_jit = jitted(values, b)
    np.testing.assert_allclose(np.asarray(x_jit), np.asarray(x), rtol=1e-12)
    print("jit(solve) matches eager. host-callback active under jit.\n")

    # --- jit + grad ---
    jitted_grad = jax.jit(jax.grad(loss, argnums=1))
    g_jit = jitted_grad(values, b)
    np.testing.assert_allclose(np.asarray(g_jit), np.asarray(g_b), rtol=1e-12)
    print("jit(grad(loss)) matches eager.\n")


if __name__ == "__main__":
    main()
