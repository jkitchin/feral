"""#1008 feral-side: dump REAL LP bases from the high-fill instance QPLIB_1451_rlt0
(m=7392, the basis discopt reports at 19.1x fill) so the fill question can be
answered against the actual matrices, not a synthetic stand-in.

The reported 19.1x is a MEAN over ~600 factorizations, so the final basis alone is
not a fair sample. This walks the trajectory by re-solving the identical captured
LP under increasing iteration caps and dumping the basis reached at each cap.

Gate: every dumped B must be square and every basic index >= n_struct must be
exactly the unit column e_{j-n_struct}, which is the [A | I] indexing assumption
that could actually be wrong.
"""
import os, sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
import discopt._rust as _rust
from discopt.solvers.milp_simplex import _dual_start_slack_basis

LP = sys.argv[1]
OUT = sys.argv[2]
CAPS = [int(x) for x in sys.argv[3].split(",")]
tag = os.path.basename(LP)[:-4]
os.makedirs(OUT, exist_ok=True)

z = np.load(LP)
m, ns = int(z["shape"][0]), int(z["shape"][1])
A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(m, ns))
c, b, lo, hi = z["c"], z["b"], z["lo"], z["hi"]
st = _dual_start_slack_basis(c, lo, hi, m)
assert st is not None, "dual start rejected"
AI = sp.hstack([A, sp.identity(m, format="csc")], format="csc")
print(f"{tag}: m={m} n_struct={ns} nnz(A)={A.nnz}", flush=True)

def run(cap):
    args = (np.ascontiguousarray(np.concatenate([c, np.zeros(m)])), m, ns + m,
            np.ascontiguousarray(AI.indptr, dtype=np.int64),
            np.ascontiguousarray(AI.indices, dtype=np.int64),
            np.ascontiguousarray(AI.data, dtype=np.float64),
            np.ascontiguousarray(b),
            np.ascontiguousarray(np.concatenate([lo, np.zeros(m)])),
            np.ascontiguousarray(np.concatenate([hi, np.full(m, np.inf)])),
            np.ascontiguousarray(st[0], dtype=np.int8),
            np.ascontiguousarray(st[1], dtype=np.int64),
            1e-9, cap, 600.0)
    return _rust.solve_lp_warm_csc_py(*args)

kept = 0
for cap in CAPS:
    out = run(cap)
    status, iters, basic = out[0], out[3], np.asarray(out[5], int).ravel()
    assert basic.size == m, f"cap {cap}: {basic.size} basic for m={m}"
    B = AI[:, basic].tocsc()
    assert B.shape == (m, m)
    nslack = 0
    for pos, j in enumerate(basic):
        if j >= ns:
            col = B[:, pos]
            assert col.nnz == 1 and col.indices[0] == j - ns and col.data[0] == 1.0, \
                f"basic slack {j} is not e_{j-ns} - [A | I] indexing is wrong"
            nslack += 1
    try:
        spla.splu(B, permc_spec="COLAMD", diag_pivot_thresh=0.1)
    except RuntimeError as e:
        print(f"  cap={cap}: B SINGULAR ({e}) - skipped", flush=True); continue
    p = f"{OUT}/{tag}_cap{cap}.npz"
    sp.save_npz(p, B)
    print(f"  cap={cap:6d} status={status} iters={iters} nnz(B)={B.nnz} "
          f"slacks={nslack} struct={m-nslack} -> {p}", flush=True)
    kept += 1

print(f"kept={kept}")
sys.exit(0 if kept else 1)
