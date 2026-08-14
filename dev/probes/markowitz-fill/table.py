"""One table: fill (nnz(L+U)/nnz(B)) on real discopt LP bases under
  - feral AMD(full)      : what discopt runs today (feral 0.15.1, analyze())
  - feral peel+AMD(bump) : feral's merged #160 triangularization
  - SuperLU COLAMD       : the reference used to declare "fill theory dead"
  - threshold-Markowitz  : dynamic pivoting, the LP INVERT standard
All four count strictly-lower L + U-with-diagonal, matching feral's factor_nnz()."""
import glob, os, re, subprocess, sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, "/private/tmp/feral-fill")
from markowitz import markowitz_lu
from run_oracle import load

EX = "/private/tmp/feral-pr162/target/release/examples/basis_refactor"
paths = sorted(sys.argv[1:])
print(f"{'basis':24s} {'m':>6s} {'nnz(B)':>8s} {'bump':>6s} | "
      f"{'AMDfull':>8s} {'peel':>8s} {'COLAMD':>8s} {'MARKOW':>8s} | {'M vs best feral':>15s}")
rows = []
for p in paths:
    B = load(p); m = B.shape[0]
    out = subprocess.run([EX, p, "3"], capture_output=True, text=True).stdout
    peel = int(re.search(r"peel\+AMD\(bump\).*nnz\(LU\)=(\d+)\s+bump=(\d+)", out).group(1))
    bump = int(re.search(r"peel\+AMD\(bump\).*bump=(\d+)", out).group(1))
    full = int(re.search(r"AMD\(full\)\s+.*nnz\(LU\)=(\d+)", out).group(1))
    sl = spla.splu(B, permc_spec="COLAMD", diag_pivot_thresh=0.1)
    col = sl.L.nnz + sl.U.nnz - m
    nl, nu, res, _, _ = markowitz_lu(B, u=0.1)
    assert res < 1e-10, f"{p}: markowitz residual {res:.2e} - result rejected"
    mk = nl + nu
    bestf = min(peel, full)
    name = os.path.basename(p).replace("_basis", "").replace(".mtx", "").replace(".npz", "")
    print(f"{name:24s} {m:6d} {B.nnz:8d} {bump:6d} | {full/B.nnz:7.2f}x {peel/B.nnz:7.2f}x "
          f"{col/B.nnz:7.2f}x {mk/B.nnz:7.2f}x | {bestf/mk:14.2f}x")
    rows.append((full/B.nnz, peel/B.nnz, col/B.nnz, mk/B.nnz, bestf/mk))
a = np.array(rows)
g = lambda v: float(np.exp(np.mean(np.log(v))))
print(f"\nn={len(rows)}  geomean fill:  AMDfull {g(a[:,0]):.2f}x  peel {g(a[:,1]):.2f}x  "
      f"COLAMD {g(a[:,2]):.2f}x  MARKOWITZ {g(a[:,3]):.2f}x")
print(f"geomean Markowitz advantage over feral's best ordering: {g(a[:,4]):.2f}x  "
      f"(min {a[:,4].min():.2f}x, max {a[:,4].max():.2f}x)")
