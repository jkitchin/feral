"""Minimal feral quickstart: factor a small SPD matrix and solve."""

from __future__ import annotations

import numpy as np

import feral


def main() -> None:
    A = feral.CscMatrix.from_dense(
        np.array(
            [
                [4.0, 1.0, 0.0],
                [1.0, 3.0, 2.0],
                [0.0, 2.0, 5.0],
            ]
        )
    )
    print(f"matrix: {A}")

    solver = feral.Solver()
    status, inertia = solver.factor(A)
    print(f"factor status:  {feral.FactorStatus(status).name}")
    print(f"inertia:        {inertia}")
    print(f"factor nnz:     {solver.factor_nnz}")
    print(f"condition est:  {solver.estimate_condition_1norm(A):.3e}")

    b = np.array([1.0, 2.0, 3.0])
    x = solver.solve(b)
    print(f"x = {x}")
    print(f"||Ax - b||_inf / ||b||_inf = {A.relative_residual(x, b):.3e}")


if __name__ == "__main__":
    main()
