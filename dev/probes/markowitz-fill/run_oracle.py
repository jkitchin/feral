import sys, os, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, "/private/tmp/feral-fill")
from markowitz import markowitz_lu

def load(p):
    if p.endswith(".npz"): return sp.load_npz(p).tocsc()
    rows, cols, vals = [], [], []
    with open(p) as f:
        hdr = None
        for line in f:
            if line.startswith("%"): continue
            t = line.split()
            if hdr is None: hdr = (int(t[0]), int(t[1]), int(t[2])); continue
            rows.append(int(t[0])-1); cols.append(int(t[1])-1); vals.append(float(t[2]))
    return sp.csc_matrix((vals,(rows,cols)), shape=(hdr[0],hdr[1]))

for p in sys.argv[1:]:
    B = load(p); m = B.shape[0]
    print(f"\n=== {os.path.basename(p)}  m={m}  nnz(B)={B.nnz}", flush=True)
    for u in (0.1, 0.01):
        t0=time.perf_counter()
        nl,nu,res,_,_ = markowitz_lu(B, u=u)
        dt=time.perf_counter()-t0
        ok = "OK" if res < 1e-10 else f"BAD resid={res:.2e}"
        print(f"  markowitz u={u:<5}  nnz(L+U)={nl+nu:9d}  fill={(nl+nu)/B.nnz:6.2f}x  "
              f"resid={res:.2e} {ok}  [{dt:.1f}s]", flush=True)
    for u in (0.1, 1.0):
        lu = spla.splu(B, permc_spec="COLAMD", diag_pivot_thresh=u)
        n = lu.L.nnz + lu.U.nnz - m
        print(f"  superlu COLAMD thresh={u:<4} nnz(L+U)={n:9d}  fill={n/B.nnz:6.2f}x", flush=True)
    lu = spla.splu(B, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.1)
    n = lu.L.nnz + lu.U.nnz - m
    print(f"  superlu MMD_AT+A thresh=0.1  nnz(L+U)={n:9d}  fill={n/B.nnz:6.2f}x", flush=True)
