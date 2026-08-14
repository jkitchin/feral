"""Dump the final basis from several captured #1008 relaxation LPs, so the fill
verdict rests on a corpus rather than one instance. Same [A | I] gate as
dump_basis.py: every basic index >= n_struct must be exactly e_{j-n_struct}."""
import glob, os, sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
import discopt._rust as _rust
from discopt.solvers.milp_simplex import _dual_start_slack_basis

OUT = sys.argv[1]; TL = float(sys.argv[2]); paths = sorted(sys.argv[3:])
os.makedirs(OUT, exist_ok=True)
kept = 0
for p in paths:
    tag = os.path.basename(p)[:-4]
    z = np.load(p); m, ns = int(z["shape"][0]), int(z["shape"][1])
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(m, ns))
    c, b, lo, hi = z["c"], z["b"], z["lo"], z["hi"]
    st = _dual_start_slack_basis(c, lo, hi, m)
    if st is None:
        print(f"{tag}: dual start rejected", flush=True); continue
    AI = sp.hstack([A, sp.identity(m, format="csc")], format="csc")
    args = (np.ascontiguousarray(np.concatenate([c, np.zeros(m)])), m, ns + m,
            np.ascontiguousarray(AI.indptr, dtype=np.int64),
            np.ascontiguousarray(AI.indices, dtype=np.int64),
            np.ascontiguousarray(AI.data, dtype=np.float64),
            np.ascontiguousarray(b),
            np.ascontiguousarray(np.concatenate([lo, np.zeros(m)])),
            np.ascontiguousarray(np.concatenate([hi, np.full(m, np.inf)])),
            np.ascontiguousarray(st[0], dtype=np.int8),
            np.ascontiguousarray(st[1], dtype=np.int64), 1e-9, 100000, TL)
    out = _rust.solve_lp_warm_csc_py(*args)
    basic = np.asarray(out[5], int).ravel()
    if basic.size != m:
        print(f"{tag}: {basic.size} basic for m={m} - skipped", flush=True); continue
    B = AI[:, basic].tocsc()
    bad = False
    for pos, j in enumerate(basic):
        if j >= ns:
            col = B[:, pos]
            if not (col.nnz == 1 and col.indices[0] == j - ns and col.data[0] == 1.0):
                bad = True; break
    if bad:
        print(f"{tag}: [A | I] indexing gate FAILED - skipped", flush=True); continue
    try:
        spla.splu(B, permc_spec="COLAMD", diag_pivot_thresh=0.1)
    except RuntimeError:
        print(f"{tag}: final B singular - skipped", flush=True); continue
    sp.save_npz(f"{OUT}/{tag}.npz", B)
    print(f"{tag:22s} status={out[0]:10s} m={m} nnz(B)={B.nnz} "
          f"struct={int((basic<ns).sum())}", flush=True)
    kept += 1
print(f"kept={kept}")
sys.exit(0 if kept else 1)
