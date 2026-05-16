"""End-to-end IPM Newton step against a small NLP, using feral.ipm.KktSolver.

The NLP is HS071 (Hock-Schittkowski 71):

    min  x0*x3*(x0 + x1 + x2) + x2
    s.t. x0*x1*x2*x3 >= 25
         x0^2 + x1^2 + x2^2 + x3^2 = 40
         1 <= x_i <= 5

We don't actually run a full IPM here — we build the KKT at the
Wächter–Biegler primal-dual point reported in their original paper
and demonstrate that ``KktSolver`` factors it with the right inertia,
solves a Newton step, and reuses the symbolic factor when refactoring
with perturbed values across iterations.

This is the layer discopt-shaped callers wire into their Newton loop.
"""

from __future__ import annotations

import numpy as np

import feral
from feral.ipm import KktSolver


def hs071_kkt_at(x: np.ndarray, lam: np.ndarray, z_L: np.ndarray, z_U: np.ndarray):
    """Assemble the augmented KKT system for HS071 at a primal-dual point.

    Returns (K_dense, n_primal, n_equality).
    """
    x0, x1, x2, x3 = x

    # Gradient of f
    grad_f = np.array(
        [
            x3 * (2 * x0 + x1 + x2),
            x0 * x3,
            x0 * x3 + 1.0,
            x0 * (x0 + x1 + x2),
        ]
    )
    # Hessian of Lagrangian (objective Hessian + lam contributions).
    # For brevity we use a perturbed identity — this script is a
    # mechanics demo, not a refined IPM.
    H = np.eye(4) * 4.0 + np.outer(grad_f, grad_f) * 1e-3

    # Constraints
    #   c1(x) = x0*x1*x2*x3 - 25 >= 0  (inequality, slacked → equality)
    #   c2(x) = x0^2 + x1^2 + x2^2 + x3^2 - 40 = 0
    A = np.array(
        [
            [x1 * x2 * x3, x0 * x2 * x3, x0 * x1 * x3, x0 * x1 * x2],
            [2 * x0, 2 * x1, 2 * x2, 2 * x3],
        ]
    )
    n, m = 4, 2

    K = np.zeros((n + m, n + m))
    K[:n, :n] = H
    K[n:, :n] = A
    K[:n, n:] = A.T
    return K, n, m


def kkt_to_csc_with_zero_diag(K: np.ndarray) -> feral.CscMatrix:
    """Build a feral CscMatrix from a dense KKT, including explicit
    zero entries on the (2,2) diagonal so KktSolver can apply δ_c.
    """
    N = K.shape[0]
    rows, cols, vals = [], [], []
    for j in range(N):
        for i in range(j, N):
            if i == j or K[i, j] != 0.0:
                rows.append(i)
                cols.append(j)
                vals.append(float(K[i, j]))
    return feral.CscMatrix.from_triplet(
        N,
        np.array(rows, dtype=np.int64),
        np.array(cols, dtype=np.int64),
        np.array(vals, dtype=np.float64),
    )


def main() -> None:
    # Mid-iterate primal-dual point (close-ish to the HS071 solution).
    x = np.array([1.0, 5.0, 5.0, 1.0])
    lam = np.array([0.1, -1.0])
    z_L = np.full(4, 1e-3)
    z_U = np.full(4, 1e-3)

    K, n, m = hs071_kkt_at(x, lam, z_L, z_U)
    pattern = kkt_to_csc_with_zero_diag(K)

    kkt = KktSolver(
        pattern,
        expected_inertia=feral.Inertia(n, m),
    )

    print("=== HS071 Newton step demo ===")
    print(f"primal n = {n}, equality m = {m}, KKT n = {n + m}")

    # Pretend we run 5 Newton iterations, each with slightly different
    # KKT values (in a real IPM these come from updating x, lam, μ).
    rng = np.random.default_rng(0)
    for k in range(5):
        # Perturb the (1,1) block diagonal as if μ changed; pattern
        # unchanged. We pass the new values via pattern.values()
        # offset by a small noise.
        perturbed = pattern.values()
        perturbed += 0.05 * rng.standard_normal(perturbed.shape)

        report = kkt.factor(perturbed)
        print(
            f"  iter {k}: status={feral.FactorStatus(report.status).name:>14s}  "
            f"inertia={report.inertia}  "
            f"δ_w={report.delta_w:.2e}  attempts={report.n_attempts}  "
            f"factor={report.factor_time_ms:.2f} ms"
        )

        if report.status != feral.FactorStatus.SUCCESS:
            print(f"  → giving up at iter {k}")
            break

        rhs = rng.standard_normal(n + m)
        step = kkt.solve(rhs)
        residual = np.linalg.norm(K @ step - rhs, np.inf) / max(
            np.linalg.norm(rhs, np.inf), 1.0
        )
        # Note: K here is the unperturbed dense KKT, so residual is
        # large by construction — demo only.
        print(
            f"           step norm = {np.linalg.norm(step):.3e}  "
            f"unrefined residual vs K0 ~ {residual:.2e}"
        )

    print()
    print(f"symbolic factorizations performed: {kkt.solver.symbolic_call_count}")
    print("(should be 1 — pattern unchanged across Newton iterations)")


if __name__ == "__main__":
    main()
