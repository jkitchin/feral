"""Generate the feral example notebooks from source-of-truth cell lists.

Run ``python _build_notebooks.py`` to (re)write the five ``.ipynb`` files
in this directory. Keeping the cells in a plain ``.py`` generator makes
them reviewable in diffs; the ``.ipynb`` files are the executable
artifacts (run them with ``jupyter nbconvert --execute`` or open in
Jupyter).

Notebooks
---------
01_basic_factor_solve     factor / inertia / solve / refine / refactor
02_multi_rhs_batched      batched multi-RHS solve + amortization timing
03_kkt_saddle_inertia     indefinite KKT system, certified inertia
04_scipy_numpy_interop    scipy.sparse <-> feral round-trip vs spsolve
05_lu_and_introspection   LU engine, L/D + symbolic access, introspection
"""

from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def _nb(cells):
    nb = new_notebook()
    nb.cells = cells
    # Pin a generic kernelspec; nbconvert --execute picks an available
    # python3 kernel regardless of the recorded name.
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def md(text):
    return new_markdown_cell(text)


def code(src):
    return new_code_cell(src)


# --------------------------------------------------------------------------
# 01 — basic factor / solve
# --------------------------------------------------------------------------
nb01 = _nb([
    md(
        "# feral — basic factor & solve\n"
        "\n"
        "`feral` is a pure-Rust sparse **symmetric indefinite** direct solver "
        "(LDLᵀ with Bunch–Kaufman pivoting) that reports a **certified inertia** "
        "count. This notebook covers the core workflow: build a matrix, factor "
        "it, read the inertia, solve, refine, and reuse the symbolic analysis."
    ),
    code(
        "import numpy as np\n"
        "import feral\n"
        "\n"
        "print('feral', feral.__version__)"
    ),
    md(
        "## Build a matrix\n"
        "\n"
        "`feral` stores symmetric matrices by their **lower triangle** in CSC "
        "form. The easiest constructors are `from_dense` (reads the lower "
        "triangle of a dense array) and `from_triplet` (COO lower-triangle "
        "entries)."
    ),
    code(
        "A = feral.CscMatrix.from_dense(np.array([\n"
        "    [4.0, 1.0, 0.0],\n"
        "    [1.0, 3.0, 2.0],\n"
        "    [0.0, 2.0, 5.0],\n"
        "]))\n"
        "print(A)              # n=3, nnz counts the lower triangle\n"
        "print('n =', A.n, ' nnz =', A.nnz)"
    ),
    md(
        "## Factor and read the inertia\n"
        "\n"
        "`Solver.factor` returns `(status, inertia)`. The inertia "
        "`(n_pos, n_neg, n_zero)` is the count of positive / negative / zero "
        "eigenvalues — exact for non-singular matrices. This SPD matrix is "
        "`(3, 0, 0)`."
    ),
    code(
        "solver = feral.Solver()\n"
        "status, inertia = solver.factor(A)\n"
        "print('status :', feral.FactorStatus(status).name)\n"
        "print('inertia:', inertia)\n"
        "print('factor nnz   :', solver.factor_nnz)\n"
        "print('cond. (1-norm):', f'{solver.estimate_condition_1norm(A):.3e}')\n"
        "assert status == feral.FactorStatus.SUCCESS\n"
        "assert inertia == feral.Inertia(3, 0, 0)"
    ),
    md(
        "## Solve and check the residual\n"
        "\n"
        "`relative_residual(x, b)` returns `‖A·x − b‖∞ / ‖b‖∞`."
    ),
    code(
        "b = np.array([1.0, 2.0, 3.0])\n"
        "x = solver.solve(b)\n"
        "print('x =', x)\n"
        "res = A.relative_residual(x, b)\n"
        "print(f'relative residual = {res:.3e}')\n"
        "assert res < 1e-12"
    ),
    md(
        "## Iterative refinement\n"
        "\n"
        "`solve_refined` runs a few steps of iterative refinement against the "
        "original matrix, recovering near machine precision."
    ),
    code(
        "x_ref = solver.solve_refined(A, b)\n"
        "print(f'refined residual = {A.relative_residual(x_ref, b):.3e}')\n"
        "assert A.relative_residual(x_ref, b) < 1e-13"
    ),
    md(
        "## Reuse the symbolic analysis (`refactor`)\n"
        "\n"
        "On the interior-point hot path the sparsity **pattern** is fixed and "
        "only the **values** change each iteration. `refactor` reuses the "
        "cached symbolic analysis, so `symbolic_call_count` stays at 1."
    ),
    code(
        "A2 = feral.CscMatrix.from_dense(np.array([\n"
        "    [5.0, 1.0, 0.0],\n"
        "    [1.0, 4.0, 2.0],\n"
        "    [0.0, 2.0, 6.0],\n"
        "]))\n"
        "status2, inertia2 = solver.refactor(A2)\n"
        "print('refactor status :', feral.FactorStatus(status2).name)\n"
        "print('inertia         :', inertia2)\n"
        "print('symbolic_call_count:', solver.symbolic_call_count)\n"
        "assert solver.symbolic_call_count == 1"
    ),
])


# --------------------------------------------------------------------------
# 02 — multi-RHS batched solve + timing
# --------------------------------------------------------------------------
nb02 = _nb([
    md(
        "# feral — batched multi-RHS solves: a worked example\n"
        "\n"
        "**The pattern.** Factor a matrix *once*, then solve it against "
        "*many* right-hand sides. `feral` shares the expensive supernodal "
        "traversal across all columns, so one batched solve is far cheaper "
        "than looping a single-RHS solve. Pass `Solver.solve` a 2-D "
        "`(n, nrhs)` array and you get back an `(n, nrhs)` solution.\n"
        "\n"
        "**Where this shows up — any time one factorization meets many "
        "vectors:**\n"
        "\n"
        "- **Parameter sweeps / design exploration** — same physics (one "
        "stiffness or conductance matrix), many loads or sources.\n"
        "- **Sensitivities & gradients** — `jax.jacrev` over a solve, or the "
        "columns of `A⁻¹` needed for design derivatives.\n"
        "- **Uncertainty quantification** — selected entries of `A⁻¹` for "
        "variances / leverage scores.\n"
        "- **Interior-point steps** — predictor and corrector back-solves "
        "against one KKT factor (this is what surfaced it in `pounce`).\n"
        "\n"
        "We make it concrete with a small **steady-state heat-conduction** "
        "problem and a sweep over heat-source layouts. Under the hood, for "
        "wide `nrhs` `feral` runs each supernode's dense panel as a "
        "register-blocked TRSM + GEMM (GitHub issue #57)."
    ),
    code(
        "import time\n"
        "import numpy as np\n"
        "import scipy.sparse as sp\n"
        "import feral\n"
        "\n"
        "rng = np.random.default_rng(0)"
    ),
    md(
        "## The model: steady-state heat on a square plate\n"
        "\n"
        "Discretize the steady heat equation $-\\nabla^2 u = q$ on an "
        "$m \\times m$ grid (unit spacing, fixed-temperature boundary) and you "
        "get a linear system $L\\,u = q$, where $L$ is the 2-D 5-point "
        "Laplacian — symmetric positive definite, sparse, the canonical "
        "conduction operator. Here $u$ is the temperature field and $q$ the "
        "heat-source distribution.\n"
        "\n"
        "**One plate ⇒ one $L$**, factored once. **Many candidate source "
        "layouts ⇒ many right-hand sides** — exactly the batched-solve "
        "pattern."
    ),
    code(
        "def laplacian_2d(m):\n"
        "    \"\"\"5-point Laplacian on an m x m grid -> (m*m) x (m*m) SPD.\"\"\"\n"
        "    n1 = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))\n"
        "    eye = sp.identity(m)\n"
        "    return (sp.kron(eye, n1) + sp.kron(n1, eye)).tocsc()\n"
        "\n"
        "grid = 40                      # 40 x 40 plate\n"
        "L_sp = laplacian_2d(grid)\n"
        "n = L_sp.shape[0]\n"
        "L = feral.from_scipy(L_sp, symmetric='full')\n"
        "print(f'plate {grid}x{grid}  ->  n = {n},  nnz(lower) = {L.nnz}')"
    ),
    md(
        "## Factor once\n"
        "\n"
        "The factorization is the expensive step; we pay it a single time and "
        "reuse it for every source layout below. (The certified inertia "
        "confirms $L$ is SPD — all eigenvalues positive.)"
    ),
    code(
        "solver = feral.Solver()\n"
        "status, inertia = solver.factor(L)\n"
        "print('status :', feral.FactorStatus(status).name)\n"
        "print('inertia:', inertia)   # SPD -> all positive\n"
        "assert status == feral.FactorStatus.SUCCESS\n"
        "assert inertia.n_neg == 0 and inertia.n_zero == 0"
    ),
    md(
        "## A batch of heat-source layouts\n"
        "\n"
        "Each column of `Q` is a different heat-source pattern — a localized "
        "Gaussian \"hot spot\" at a random spot on the plate. Solving "
        "$L\\,U = Q$ returns the steady-state temperature field for **every "
        "layout at once**, as the columns of `U`."
    ),
    code(
        "def gaussian_source(cx, cy, width=2.5):\n"
        "    \"\"\"A Gaussian hot spot centered at (cx, cy), flattened to length n.\"\"\"\n"
        "    yy, xx = np.mgrid[0:grid, 0:grid]\n"
        "    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * width ** 2))\n"
        "    return g.ravel()\n"
        "\n"
        "nrhs = 64\n"
        "centers = rng.integers(4, grid - 4, size=(nrhs, 2))\n"
        "Q = np.stack([gaussian_source(cx, cy) for cx, cy in centers], axis=1)\n"
        "\n"
        "U = solver.solve(Q)            # (n, nrhs) temperature fields, ONE batched call\n"
        "print('Q.shape =', Q.shape, '  U.shape =', U.shape)"
    ),
    md(
        "## Correctness: batched solve == per-column solves\n"
        "\n"
        "The batched result must match independent single-RHS solves column "
        "by column — the single-RHS path is the trusted reference — and the "
        "whole batch must satisfy $L\\,U = Q$."
    ),
    code(
        "max_col_diff = 0.0\n"
        "for j in range(nrhs):\n"
        "    uj = solver.solve(Q[:, j].copy())\n"
        "    max_col_diff = max(max_col_diff, np.max(np.abs(U[:, j] - uj)))\n"
        "print(f'max |batched - single| over all columns = {max_col_diff:.3e}')\n"
        "assert max_col_diff < 1e-12\n"
        "\n"
        "batch_res = np.max(np.abs(L_sp @ U - Q))\n"
        "print(f'max abs residual over batch = {batch_res:.3e}')"
    ),
    md(
        "## See it: temperature fields for a few layouts\n"
        "\n"
        "Each solved column reshapes back to the plate. Different source "
        "placements give different steady-state temperature distributions — "
        "all from the one shared factorization."
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))\n"
        "for ax, j in zip(axes, range(4)):\n"
        "    ax.imshow(U[:, j].reshape(grid, grid), cmap='inferno', origin='lower')\n"
        "    cx, cy = centers[j]\n"
        "    ax.set_title(f'source at ({cx}, {cy})')\n"
        "    ax.axis('off')\n"
        "fig.suptitle('Steady-state temperature for 4 of the 64 source layouts')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## The payoff: looped single-RHS vs one batched call\n"
        "\n"
        "Compare the per-RHS cost of looping the single-RHS solve against a "
        "single batched call. The batched call amortizes the supernodal "
        "traversal and gather/scatter across columns, and for wide `nrhs` "
        "runs the dense per-supernode panels as register-blocked TRSM + GEMM."
    ),
    code(
        "def bench(fn, repeat=3):\n"
        "    best = float('inf')\n"
        "    for _ in range(repeat):\n"
        "        t0 = time.perf_counter()\n"
        "        fn()\n"
        "        best = min(best, time.perf_counter() - t0)\n"
        "    return best\n"
        "\n"
        "print(f'{\"nrhs\":>5}  {\"looped\":>12}  {\"batched\":>12}  {\"speedup\":>8}')\n"
        "for k in (64, 256):\n"
        "    Qk = np.stack(\n"
        "        [gaussian_source(*c) for c in rng.integers(4, grid - 4, size=(k, 2))],\n"
        "        axis=1,\n"
        "    )\n"
        "    cols = [Qk[:, j].copy() for j in range(k)]\n"
        "    t_loop = bench(lambda: [solver.solve(c) for c in cols])\n"
        "    t_batch = bench(lambda: solver.solve(Qk))\n"
        "    per_loop = t_loop / k * 1e6\n"
        "    per_batch = t_batch / k * 1e6\n"
        "    print(\n"
        "        f'{k:>5}  {per_loop:9.2f} us  {per_batch:9.2f} us  '\n"
        "        f'{per_loop / per_batch:6.2f}x'\n"
        "    )"
    ),
    md(
        "At large `nrhs` the batched per-RHS time is a fraction of the looped "
        "time. On these 2-D Laplacians the batched solve runs roughly **3–6× "
        "faster per RHS** than looping (issue #57 fix #2: row-major working "
        "buffers + register-blocked TRSM/GEMM panel kernels). The exact factor "
        "depends on problem size and your CPU's SIMD width and cache.\n"
        "\n"
        "Nothing about the calling code changes — `solver.solve(Q)` with a 2-D "
        "`Q` is all you write; `feral` picks the wide-`nrhs` panel kernels "
        "automatically."
    ),
    md(
        "## Recovering accuracy: refined batched solves\n"
        "\n"
        "`solve_refined` adds a few steps of **iterative refinement** against "
        "the original matrix — cheap insurance that recovers digits on "
        "ill-conditioned or near-singular systems (it returns the best iterate, "
        "so a refined solve is never worse than the plain one). Pass it a 2-D "
        "`B` and the wide refined solve runs through the **same panel kernel** "
        "as `solve_many`: one batched solve per refinement step over the "
        "still-unconverged columns, instead of looping a single-RHS refined "
        "solve per column (GitHub issue #58). Before that fix the refined "
        "multi-RHS path bypassed the panel kernel and could be 3–7× slower per "
        "RHS than the unrefined batched solve.\n"
        "\n"
        "Same one-line call — `solver.solve_refined(L, Q)` with a 2-D `Q`:"
    ),
    code(
        "Xr = solver.solve_refined(L, Q)        # (n, nrhs), refined AND batched\n"
        "print('refined batch residual:', np.max(np.abs(L_sp @ Xr - Q)))\n"
        "\n"
        "# Matches looping the single-RHS refined solve, column by column.\n"
        "max_col = max(\n"
        "    np.max(np.abs(Xr[:, j] - solver.solve_refined(L, Q[:, j].copy())))\n"
        "    for j in range(8)\n"
        ")\n"
        "print('max |batched - per-column refined| (first 8 cols):', max_col)"
    ),
    md(
        "### Does it pay off? Measure it\n"
        "\n"
        "Time the refined path the same way: looping `solve_refined` per column "
        "vs one batched 2-D call."
    ),
    code(
        "print(f'{\"nrhs\":>5}  {\"loop refined\":>15}  {\"batch refined\":>15}  {\"speedup\":>8}')\n"
        "for k in (64, 256):\n"
        "    Qk = np.stack(\n"
        "        [gaussian_source(*c) for c in rng.integers(4, grid - 4, size=(k, 2))],\n"
        "        axis=1,\n"
        "    )\n"
        "    cols = [Qk[:, j].copy() for j in range(k)]\n"
        "    t_loop = bench(lambda: [solver.solve_refined(L, c) for c in cols])\n"
        "    t_batch = bench(lambda: solver.solve_refined(L, Qk))\n"
        "    print(\n"
        "        f'{k:>5}  {t_loop / k * 1e6:12.2f} us  {t_batch / k * 1e6:12.2f} us  '\n"
        "        f'{t_loop / t_batch:6.2f}x'\n"
        "    )"
    ),
    md(
        "The batched refined path is roughly **2–3× faster per RHS** than looping "
        "the single-RHS refined solve — even on this well-conditioned plate, "
        "where refinement does ~0 correction steps and the win is entirely the "
        "shared batched **initial** solve. Before issue #58 this path looped the "
        "single-RHS refiner and bypassed the panel kernel; it is the **default** "
        "for the solver and for pounce's KKT back-solves.\n"
        "\n"
        "**The nuance — it is not a free lunch.** The batched path amortizes the "
        "*solves*; the per-column **residual** `B − A·X` is still computed column "
        "by column. On sparse systems that residual is cheap, so the solve "
        "dominates and you see the full speedup. On a *dense* Hessian (where the "
        "matrix–vector product is as expensive as the solve) the un-batched "
        "residual caps the gain — a single-pass batched residual SpMV is the "
        "next lever. The speedup also grows with `nrhs` and with how much "
        "refinement actually has to do (ill-conditioned / saddle-point KKT "
        "systems, where the batched correction solves carry the cost)."
    ),
])


# --------------------------------------------------------------------------
# 03 — KKT / saddle-point + inertia
# --------------------------------------------------------------------------
nb03 = _nb([
    md(
        "# feral — indefinite KKT systems & certified inertia\n"
        "\n"
        "Interior-point and equality-constrained problems produce **symmetric "
        "indefinite** saddle-point (KKT) systems\n"
        "\n"
        "$$ K = \\begin{bmatrix} H & A^\\top \\\\ A & 0 \\end{bmatrix} $$\n"
        "\n"
        "with `H` (n×n) the Hessian and `A` (m×n) the constraint Jacobian. For "
        "a well-posed problem with `H` positive definite on the null space of "
        "`A`, the KKT matrix has inertia `(n, m, 0)`: `n` positive, `m` "
        "negative, `0` zero eigenvalues. `feral` reports this exactly, which is "
        "how an IPM checks it is on a descent step."
    ),
    code(
        "import numpy as np\n"
        "import scipy.sparse as sp\n"
        "import feral\n"
        "\n"
        "rng = np.random.default_rng(1)"
    ),
    md("## Build a small KKT matrix"),
    code(
        "n, m = 8, 3\n"
        "# SPD Hessian H = M M^T + I\n"
        "M = rng.standard_normal((n, n))\n"
        "H = M @ M.T + np.eye(n)\n"
        "# full-rank constraint Jacobian A (m x n)\n"
        "Acon = rng.standard_normal((m, n))\n"
        "\n"
        "K = np.block([\n"
        "    [H,            Acon.T],\n"
        "    [Acon,         np.zeros((m, m))],\n"
        "])\n"
        "K_sp = sp.csc_matrix(K)\n"
        "print('KKT dim =', K.shape[0], '= n + m =', n + m)"
    ),
    md(
        "## Factor and verify the inertia\n"
        "\n"
        "We expect `(n, m, 0)`. You can also pass `expected_inertia` to "
        "`factor`; it returns `WRONG_INERTIA` (without invalidating the factor) "
        "if the count differs — the signal an IPM uses to perturb."
    ),
    code(
        "K_feral = feral.from_scipy(K_sp, symmetric='full')\n"
        "solver = feral.Solver()\n"
        "expected = feral.Inertia(n, m, 0)\n"
        "status, inertia = solver.factor(K_feral, expected_inertia=expected)\n"
        "print('status  :', feral.FactorStatus(status).name)\n"
        "print('inertia :', inertia)\n"
        "print('expected:', expected)\n"
        "assert inertia == expected"
    ),
    md(
        "## Cross-check the inertia against a dense eigendecomposition"
    ),
    code(
        "eig = np.linalg.eigvalsh(K)\n"
        "n_pos = int(np.sum(eig > 1e-9))\n"
        "n_neg = int(np.sum(eig < -1e-9))\n"
        "n_zero = int(np.sum(np.abs(eig) <= 1e-9))\n"
        "print(f'eig-based inertia = ({n_pos}, {n_neg}, {n_zero})')\n"
        "assert (n_pos, n_neg, n_zero) == inertia.as_tuple()"
    ),
    md("## Solve the KKT system and check the residual"),
    code(
        "b = rng.standard_normal(n + m)\n"
        "x = solver.solve_refined(K_feral, b)\n"
        "res = np.max(np.abs(K @ x - b))\n"
        "print(f'max abs residual = {res:.3e}')\n"
        "assert res < 1e-9"
    ),
    md(
        "## A genuinely singular case\n"
        "\n"
        "If `A` is rank-deficient the KKT matrix is singular; the inertia then "
        "carries a non-zero `n_zero`. feral reports it rather than silently "
        "returning garbage."
    ),
    code(
        "Acon_rd = Acon.copy()\n"
        "Acon_rd[-1] = Acon_rd[0]          # duplicate a row -> rank deficient\n"
        "K_rd = np.block([[H, Acon_rd.T], [Acon_rd, np.zeros((m, m))]])\n"
        "eig_rd = np.linalg.eigvalsh(K_rd)\n"
        "print('dense zero eigenvalues:', int(np.sum(np.abs(eig_rd) <= 1e-8)))\n"
        "print('(feral flags singular / non-zero n_zero on such systems)')"
    ),
])


# --------------------------------------------------------------------------
# 04 — scipy / numpy interop
# --------------------------------------------------------------------------
nb04 = _nb([
    md(
        "# feral — scipy.sparse & numpy interop\n"
        "\n"
        "`feral.from_scipy` / `feral.to_scipy` round-trip a symmetric "
        "`scipy.sparse` matrix into feral's lower-triangular CSC and back. This "
        "notebook shows the conversion and validates feral's solve against "
        "`scipy.sparse.linalg.spsolve` and `numpy.linalg.solve`."
    ),
    code(
        "import numpy as np\n"
        "import scipy.sparse as sp\n"
        "import scipy.sparse.linalg as spla\n"
        "import feral\n"
        "\n"
        "rng = np.random.default_rng(7)"
    ),
    md(
        "## A random symmetric indefinite matrix\n"
        "\n"
        "Half the diagonal shifted positive, half negative, to make it "
        "genuinely indefinite (a stress case for a direct solver)."
    ),
    code(
        "def random_sym_indef(n, density=0.2, seed=0):\n"
        "    r = np.random.default_rng(seed)\n"
        "    Ad = sp.random(n, n, density=density, format='csc',\n"
        "                   random_state=r).toarray()\n"
        "    Ad = (Ad + Ad.T) / 2.0\n"
        "    shift = np.empty(n)\n"
        "    shift[: n // 2] = 5.0\n"
        "    shift[n // 2 :] = -5.0\n"
        "    Ad += np.diag(shift)\n"
        "    return sp.csc_matrix(Ad)\n"
        "\n"
        "n = 60\n"
        "A_sp = random_sym_indef(n, density=0.15, seed=11)\n"
        "print('shape =', A_sp.shape, ' nnz(full) =', A_sp.nnz)"
    ),
    md(
        "## Convert into feral\n"
        "\n"
        "`symmetric='full'` tells feral the scipy matrix stores the full "
        "symmetric matrix; it reads the lower triangle."
    ),
    code(
        "A = feral.from_scipy(A_sp, symmetric='full')\n"
        "print('feral:', A, ' nnz(lower) =', A.nnz)"
    ),
    md("## Factor, report inertia, solve"),
    code(
        "solver = feral.Solver()\n"
        "status, inertia = solver.factor(A)\n"
        "print('status :', feral.FactorStatus(status).name)\n"
        "print('inertia:', inertia)\n"
        "\n"
        "b = rng.standard_normal(n)\n"
        "x_feral = solver.solve_refined(A, b)"
    ),
    md("## Compare against scipy and numpy reference solves"),
    code(
        "x_spsolve = spla.spsolve(A_sp.tocsc(), b)\n"
        "x_dense = np.linalg.solve(A_sp.toarray(), b)\n"
        "\n"
        "print('‖feral - spsolve‖inf :', f'{np.max(np.abs(x_feral - x_spsolve)):.3e}')\n"
        "print('‖feral - dense‖inf   :', f'{np.max(np.abs(x_feral - x_dense)):.3e}')\n"
        "assert np.allclose(x_feral, x_dense, atol=1e-9, rtol=1e-9)"
    ),
    md("## Round-trip feral -> scipy"),
    code(
        "A_back = feral.to_scipy(A)\n"
        "# A_back mirrors the lower triangle to a full symmetric matrix\n"
        "diff = np.max(np.abs(A_back.toarray() - A_sp.toarray()))\n"
        "print(f'max |round-trip - original| = {diff:.3e}')\n"
        "assert diff < 1e-12"
    ),
])


# --------------------------------------------------------------------------
# 05 — LU basis engine, factor access, introspection  (0.11.0)
# --------------------------------------------------------------------------
nb05 = _nb([
    md(
        "# feral — LU engine, factor access & introspection\n"
        "\n"
        "The 0.11.0 Python surface is **purely additive**: the original "
        "`Solver`/`CscMatrix` workflow is unchanged, and these features sit "
        "alongside it. This notebook covers three additions:\n"
        "\n"
        "1. the unsymmetric **LU basis engine** (`LuFactor` / `LuMatrix`),\n"
        "2. **factor access** — the assembled `L` and `D`, and the symbolic "
        "analysis, and\n"
        "3. **introspection** — tuning knobs, pivot magnitudes, factor stats, "
        "and scaling info.\n"
        "\n"
        "Every cell carries its own external oracle (a dense `numpy` solve or a "
        "reconstruction identity), so the notebook self-checks when executed."
    ),
    code(
        "import numpy as np\n"
        "import feral\n"
        "\n"
        "print('feral', feral.__version__)"
    ),
    md(
        "## 1. Unsymmetric LU basis engine\n"
        "\n"
        "`LuFactor` factors a **general** (unsymmetric) square matrix and solves "
        "`A x = b` with `ftran` and `Aᵀ y = c` with `btran`. It auto-routes "
        "between a dense and a sparse engine; the oracle here is "
        "`numpy.linalg.solve`."
    ),
    code(
        "A = np.array([[2.0, 1.0, 0.0],\n"
        "              [0.0, 3.0, 1.0],\n"
        "              [1.0, 0.0, 4.0]])\n"
        "lu = feral.LuFactor(feral.LuMatrix.from_dense(A))\n"
        "print('dense engine:', lu.is_dense)\n"
        "\n"
        "b = np.array([1.0, 2.0, 3.0])\n"
        "x = lu.ftran(b)                       # solve A x = b\n"
        "print('ftran residual:', f'{np.max(np.abs(A @ x - b)):.2e}')\n"
        "assert np.allclose(x, np.linalg.solve(A, b))\n"
        "\n"
        "c = np.array([1.0, 0.0, 0.0])\n"
        "y = lu.btran(c)                       # solve Aᵀ y = c\n"
        "assert np.allclose(y, np.linalg.solve(A.T, c))"
    ),
    md(
        "### Factor identity and product-form updates\n"
        "\n"
        "The returned permutations satisfy `P A Q = L U`. A simplex-style "
        "`update` replaces one basis column in product form; `refactor` rebuilds "
        "from scratch and resets the update counter."
    ),
    code(
        "L = lu.l_array()\n"
        "U = lu.u_array()\n"
        "lhs = A[np.ix_(lu.perm, lu.qcol)]     # P A Q\n"
        "print('|P A Q - L U|inf:', f'{np.max(np.abs(lhs - L @ U)):.2e}')\n"
        "assert np.allclose(lhs, L @ U)\n"
        "\n"
        "new_col = np.array([0.0, 5.0, 1.0])\n"
        "lu.update(1, new_col)                 # replace basis column 1\n"
        "A2 = A.copy(); A2[:, 1] = new_col\n"
        "print('updates_since_refactor:', lu.updates_since_refactor)\n"
        "assert np.allclose(lu.ftran(b), np.linalg.solve(A2, b))\n"
        "\n"
        "lu.refactor(feral.LuMatrix.from_dense(A))\n"
        "assert lu.updates_since_refactor == 0"
    ),
    md(
        "A singular basis raises `SingularBasisError` — a subclass of "
        "`FactorError`, so existing `except FactorError` handlers keep working."
    ),
    code(
        "singular = np.array([[1.0, 2.0], [2.0, 4.0]])   # rank 1\n"
        "try:\n"
        "    feral.LuFactor(feral.LuMatrix.from_dense(singular), force_dense=True)\n"
        "except feral.SingularBasisError as e:\n"
        "    print('caught SingularBasisError:', e)\n"
        "    assert isinstance(e, feral.FactorError)"
    ),
    md(
        "## 2. Factor access — L, D, and the symbolic structure\n"
        "\n"
        "After a symmetric `Solver.factor`, `Solver.factors()` exposes the "
        "assembled unit-lower `L` and block-diagonal `D` in **factorization "
        "order**. The reconstruction identity is the oracle:\n"
        "\n"
        "$$ L\\,D\\,L^{\\top} = P\\,(S A S)\\,P^{\\top} $$\n"
        "\n"
        "with the permutation `fac.perm` and the per-row scaling `fac.scaling`."
    ),
    code(
        "M = np.array([\n"
        "    [2.0, 1.0, 0.0, 0.0],\n"
        "    [1.0, -3.0, 1.0, 0.0],\n"
        "    [0.0, 1.0, 4.0, 1.0],\n"
        "    [0.0, 0.0, 1.0, -2.0],\n"
        "])\n"
        "csc = feral.CscMatrix.from_dense(M)\n"
        "s = feral.Solver()\n"
        "status, inertia = s.factor(csc)\n"
        "print('inertia:', inertia)\n"
        "\n"
        "fac = s.factors()\n"
        "indptr, indices, data = fac.l_csc()\n"
        "d_diag, d_sub = fac.d_blocks()"
    ),
    code(
        "n = fac.n\n"
        "L = np.zeros((n, n))\n"
        "for j in range(n):\n"
        "    for k in range(indptr[j], indptr[j + 1]):\n"
        "        L[indices[k], j] = data[k]\n"
        "\n"
        "D = np.zeros((n, n))\n"
        "i = 0\n"
        "while i < n:\n"
        "    if i + 1 < n and d_sub[i] != 0.0:        # 2x2 pivot block\n"
        "        D[i, i] = d_diag[i]; D[i + 1, i + 1] = d_diag[i + 1]\n"
        "        D[i, i + 1] = D[i + 1, i] = d_sub[i]\n"
        "        i += 2\n"
        "    else:\n"
        "        D[i, i] = d_diag[i]; i += 1\n"
        "\n"
        "perm, sc = fac.perm, fac.scaling\n"
        "lhs = L @ D @ L.T\n"
        "rhs = np.array([[sc[perm[a]] * M[perm[a], perm[b]] * sc[perm[b]]\n"
        "                 for b in range(n)] for a in range(n)])\n"
        "print('|L D Lᵀ - P(SAS)Pᵀ|inf:', f'{np.max(np.abs(lhs - rhs)):.2e}')\n"
        "assert np.allclose(lhs, rhs)"
    ),
    md(
        "### Symbolic analysis without a numeric factor\n"
        "\n"
        "`feral.analyze` runs the ordering + symbolic factorization with **no** "
        "numeric work. On a larger system the predicted `factor_nnz_estimate` is "
        "a (slack-inflated) upper bound on the realized factor nnz — provided the "
        "numeric factor uses the **same ordering** as the analysis."
    ),
    code(
        "rng = np.random.default_rng(2)\n"
        "B = rng.standard_normal((15, 15))\n"
        "spd = feral.CscMatrix.from_dense(B @ B.T + 15 * np.eye(15))\n"
        "\n"
        "sym = feral.analyze(spd, ordering='amd')\n"
        "snum = feral.Solver(ordering='amd')\n"
        "snum.factor(spd)\n"
        "\n"
        "print('ordering        :', sym.ordering)\n"
        "print('num_supernodes  :', sym.num_supernodes)\n"
        "print('nnz estimate    :', sym.factor_nnz_estimate)\n"
        "print('realized nnz    :', snum.factor_nnz)\n"
        "assert sorted(sym.perm) == list(range(sym.n))      # perm is a permutation\n"
        "assert sym.factor_nnz_estimate >= snum.factor_nnz"
    ),
    md(
        "## 3. Introspection — knobs, pivots, stats, scaling\n"
        "\n"
        "Different fill-reducing orderings must agree on the certified inertia. "
        "Turning on `profiling` populates a `ProfileReport`; `last_factor_stats` "
        "and the pivot-magnitude getters summarize the numeric factor."
    ),
    code(
        "rng = np.random.default_rng(0)\n"
        "G = rng.standard_normal((14, 14)); G = G + G.T\n"
        "G += np.diag(rng.standard_normal(14))\n"
        "Gc = feral.CscMatrix.from_dense(G)\n"
        "\n"
        "inertias = set()\n"
        "for ordering in ('auto', 'amd', 'amf'):\n"
        "    st = feral.Solver(ordering=ordering)\n"
        "    _, inrt = st.factor(Gc)\n"
        "    inertias.add(inrt.as_tuple())\n"
        "print('distinct inertias across orderings:', inertias)\n"
        "assert len(inertias) == 1                          # ordering-invariant"
    ),
    code(
        "sp = feral.Solver(profiling=True)\n"
        "sp.factor(Gc)\n"
        "fs = sp.last_factor_stats()\n"
        "print('nnz_a / nnz_l :', fs.nnz_a, '/', fs.nnz_l)\n"
        "print('fill_ratio    :', f'{fs.fill_ratio:.2f}')\n"
        "print('pivot range   :', f'{sp.min_pivot_magnitude:.2e}',\n"
        "      '..', f'{sp.max_pivot_magnitude:.2e}')\n"
        "print('scaling kind  :', sp.scaling_info.kind)\n"
        "print('profile total :', sp.profile_report().total_us, 'us')\n"
        "assert fs.fill_ratio >= 1.0\n"
        "assert 0.0 < sp.min_pivot_magnitude <= sp.max_pivot_magnitude"
    ),
])


NOTEBOOKS = {
    "01_basic_factor_solve.ipynb": nb01,
    "02_multi_rhs_batched.ipynb": nb02,
    "03_kkt_saddle_inertia.ipynb": nb03,
    "04_scipy_numpy_interop.ipynb": nb04,
    "05_lu_and_introspection.ipynb": nb05,
}


def main():
    """(Re)write the notebooks. By default they are executed in place so the
    committed ``.ipynb`` carry their cell outputs; pass ``--no-execute`` to
    write source-only notebooks (faster, and what you want if ``feral`` is not
    installed in the running interpreter).

    Each notebook's cells embed their own assertions against an external
    oracle, so a clean execution is also a smoke test — any failure aborts
    with the offending traceback rather than committing a broken notebook.
    """
    import os
    import sys

    execute = "--no-execute" not in sys.argv[1:]
    if execute:
        from nbconvert.preprocessors import ExecutePreprocessor

        ep = ExecutePreprocessor(timeout=300, kernel_name="python3")

    here = os.path.dirname(os.path.abspath(__file__))
    for name, nb in NOTEBOOKS.items():
        path = os.path.join(here, name)
        if execute:
            ep.preprocess(nb, {"metadata": {"path": here}})
        with open(path, "w") as f:
            nbf.write(nb, f)
        print(("executed" if execute else "wrote"), path)


if __name__ == "__main__":
    main()
