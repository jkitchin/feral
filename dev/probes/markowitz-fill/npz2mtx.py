import sys, glob, os
import numpy as np, scipy.sparse as sp
for p in sorted(glob.glob(sys.argv[1])):
    B = sp.load_npz(p).tocoo()
    out = p[:-4] + ".mtx"
    with open(out, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{B.shape[0]} {B.shape[1]} {B.nnz}\n")
        for i, j, v in zip(B.row, B.col, B.data):
            f.write(f"{i+1} {j+1} {float(v)!r}\n")
    print(out, B.shape, B.nnz)
