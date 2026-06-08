#!/usr/bin/env python3
"""Independent reference cross-check for the LU validation harness (issue #81).

For each ``.mtx`` in the corpus, solves the SAME known-x system the Rust
``lu_reference`` harness uses (``x_true[i] = 1 + (i % 7)``, ``a = B x_true``) but
with **SciPy's SuperLU** (``scipy.sparse.linalg.splu``). Reporting the same
``err``/``resid`` metrics gives an external baseline: feral's numbers should
match SciPy's to within conditioning, confirming the Rust engine isn't
systematically biased.

    python scripts/lu_reference_scipy.py [DIR]   # DIR defaults to the corpus

Requires ``scipy``.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

try:
    import scipy.io
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:
    print("scipy not installed. Run: pip install scipy", file=sys.stderr)
    raise SystemExit(1)


def check(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        m = scipy.io.mmread(path)
    except Exception as e:  # noqa: BLE001
        return f"{name:24}  READ-ERR {e}"
    if m.shape[0] != m.shape[1]:
        return f"{name:24}  SKIP non-square {m.shape}"
    n = m.shape[0]
    b = sp.csc_matrix(m)
    x_true = np.array([1.0 + (i % 7) for i in range(n)])
    a = b @ x_true
    try:
        lu = spla.splu(b)
        x = lu.solve(a)
    except Exception as e:  # noqa: BLE001
        return f"{name:24}  n={n:<7} SUPERLU-FAIL {e}"
    err = np.max(np.abs(x - x_true)) / max(np.max(np.abs(x_true)), 1e-300)
    resid = np.max(np.abs(b @ x - a)) / max(np.max(np.abs(a)), 1e-300)
    # ILL = backward-stable solve (tiny residual) whose large forward error is
    # matrix conditioning, not a solver fault. Mirrors the Rust harness verdict.
    if err < 1e-6:
        verdict = "OK "
    elif resid < 1e-8:
        verdict = "ILL"
    else:
        verdict = "BAD"
    return f"{name:24}  n={n:<7} nnz={b.nnz:<9} {verdict} err={err:.2e} resid={resid:.2e}"


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        here, "data", "matrices", "lu-corpus"
    )
    files = sorted(glob.glob(os.path.join(corpus, "*.mtx")))
    if not files:
        print(f"no .mtx in {corpus} — run scripts/fetch_lu_corpus.py", file=sys.stderr)
        return 1
    print(f"SciPy SuperLU reference — corpus: {corpus}")
    ok = ill = total = 0
    for f in files:
        line = check(f)
        if " OK " in line:
            ok += 1
        if " ILL " in line:
            ill += 1
        if " OK " in line or " ILL " in line or " BAD " in line or "FAIL" in line:
            total += 1
        print(line)
    print(
        f"\n{ok}/{total} solved to err < 1e-6 (SciPy SuperLU); "
        f"{ill} ILL (backward-stable but ill-conditioned)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
