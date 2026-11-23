"""Convert an LP in MPS form to the lower triangle of an IPM augmented KKT
system, written as a symmetric Matrix Market file.

    K = [[ D      A^T ]
         [ A   -delta I ]]

D = I (stand-in for X^{-1} Z + reg), delta = 1e-8. Only the sparsity pattern
matters for ordering/symbolic analysis; MC64 (LdltCompress) also reads values,
so this is an approximation of pounce's HSDE KKT, not a copy of it.

usage: python3 -I dev/scripts/mps_to_kkt.py in.mps.bz2 out.mtx

Requires highspy and scipy. Used for the issue #246 fixture:
  curl -O https://plato.asu.edu/ftp/lptestset/fhnw-binschedule1.mps.bz2
  python3 -I dev/scripts/mps_to_kkt.py fhnw-binschedule1.mps.bz2 \\
      tests/data/large/fhnw-binschedule1.mtx
"""
import bz2
import os
import sys
import tempfile

import highspy
import numpy as np
import scipy.sparse as sp


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with tempfile.TemporaryDirectory() as td:
        mps = os.path.join(td, "m.mps")
        with bz2.open(src, "rb") as fi, open(mps, "wb") as fo:
            while True:
                b = fi.read(1 << 24)
                if not b:
                    break
                fo.write(b)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(mps)
    lp = h.getLp()
    a = lp.a_matrix_
    m, nv = lp.num_row_, lp.num_col_
    A = sp.csc_matrix(
        (np.asarray(a.value_), np.asarray(a.index_), np.asarray(a.start_)),
        shape=(m, nv),
    ).tocoo()
    n = nv + m
    # lower triangle: A block sits below the diagonal (rows nv.., cols 0..nv)
    rows = np.concatenate([np.arange(n), A.row + nv])
    cols = np.concatenate([np.arange(n), A.col])
    vals = np.concatenate([np.ones(nv), np.full(m, -1e-8), A.data])
    K = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc().tocoo()
    with open(dst, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"{n} {n} {K.nnz}\n")
        np.savetxt(f, np.column_stack([K.row + 1, K.col + 1, K.data]),
                   fmt=["%d", "%d", "%.17g"])
    print(f"{os.path.basename(src)}: nvar={nv} ncon={m} n={n} stored_nnz={K.nnz}")


if __name__ == "__main__":
    main()
