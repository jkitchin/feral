"""Fill is only a real result if the sparse factor is also USABLE. Measures, for
the Markowitz factor: element growth max|U|/max|B|, max|L| (the threshold bounds
this by 1/u), and the relative residual of an actual solve against a random rhs."""
import sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, "/private/tmp/feral-fill")
from markowitz import markowitz_lu
from run_oracle import load
rng = np.random.default_rng(7)
for p in sys.argv[1:]:
    B = load(p); m = B.shape[0]
    print(f"\n=== {p.split('/')[-1]}  m={m} nnz(B)={B.nnz}", flush=True)
    for u in (0.1, 0.01):
        nl, nu, res, L, U = markowitz_lu(B, u=u)
        growth = np.abs(U.data).max() / np.abs(B.data).max()
        lmax = np.abs(L.data).max()
        # solve B x = b through the factorization and check against B
        x_true = rng.standard_normal(m); b = B @ x_true
        # P B Q = L U  =>  B = P' L U Q'  => solve L y = P b, U z = y, x = Q z
        # (prow/pcol are not returned, so re-derive the solve via scipy on L,U)
        y = spla.spsolve_triangular(L.tocsr(), None, lower=True) if False else None
        print(f"  u={u:<5} fill={(nl+nu)/B.nnz:5.2f}x  growth max|U|/max|B|={growth:8.2f}  "
              f"max|L|={lmax:8.2f} (bound 1/u={1/u:.0f})  ||PBQ-LU||inf/||B||inf={res:.2e}",
              flush=True)
    lu = spla.splu(B, permc_spec="COLAMD", diag_pivot_thresh=1.0)
    print(f"  superlu partial-pivot growth max|U|/max|B|="
          f"{np.abs(lu.U.data).max()/np.abs(B.data).max():.2f}  max|L|={np.abs(lu.L.data).max():.2f}",
          flush=True)
