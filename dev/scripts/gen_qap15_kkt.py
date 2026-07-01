#!/usr/bin/env python3
"""Generate tests/data/large/qap15_kkt.mtx — the issue #91 regression KKT.

Parses qap15.mps (Mittelmann lpopt) with HiGHS, rebuilds the LP in
pounce.qp.solve_qp's (A,b,G,h,lb,ub) form, runs a couple of conic-IPM
iterations with POUNCE_DBG_KKT_DUMP set so the shared linsol layer dumps
the iteration-0 KKT, then writes it as a symmetric MatrixMarket file.

Reproduces the issue #91 matrix byte-for-byte: dim=50880, nnz=168105 (a
quasi-definite conic KKT whose diagonal regularization rows trip the
LdltCompress preprocessing predicate). See
`dev/research/issue-91-preprocess-misfire.md`.

Requires: the `pounce` Python package, `highspy`, `scipy`, and the
`qap15.mps.bz2` input. Override paths via the env vars below.
"""
import os, sys, bz2, tempfile
import numpy as np
import scipy.sparse as sp

MPS_BZ2 = os.environ.get(
    "QAP15_MPS",
    os.path.expanduser("~/projects/pounce/benchmarks/lpopt/mps/qap15.mps.bz2"),
)
OUT = os.environ.get("OUT", "tests/data/large/qap15_kkt.mtx")
INF = 1e20


def read_mps_highs(path_bz2):
    import highspy

    with bz2.open(path_bz2, "rb") as f:
        data = f.read()
    with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as tf:
        tf.write(data)
        mps_path = tf.name
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(mps_path)
    os.unlink(mps_path)
    lp = h.getLp()
    A = sp.csc_matrix(
        (
            np.array(lp.a_matrix_.value_, float),
            np.array(lp.a_matrix_.index_, int),
            np.array(lp.a_matrix_.start_, int),
        ),
        shape=(lp.num_row_, lp.num_col_),
    ).tocsr()
    return dict(
        c=np.array(lp.col_cost_, float),
        cl=np.array(lp.col_lower_, float),
        cu=np.array(lp.col_upper_, float),
        rl=np.array(lp.row_lower_, float),
        ru=np.array(lp.row_upper_, float),
        A=A,
        nrow=lp.num_row_,
    )


def build_lp(m):
    A, rl, ru = m["A"], m["rl"], m["ru"]
    eq_rows, beq, leq, geq = [], [], [], []
    for i in range(m["nrow"]):
        lo, hi = rl[i], ru[i]
        if lo == hi:
            eq_rows.append(i)
            beq.append(lo)
        else:
            if hi < INF:
                leq.append((i, hi))
            if lo > -INF:
                geq.append((i, lo))
    Aeq = A[eq_rows].tocsc() if eq_rows else None
    beq = np.array(beq) if eq_rows else None
    G, h = [], []
    if leq:
        G.append(A[[i for i, _ in leq]])
        h += [v for _, v in leq]
    if geq:
        G.append(-A[[i for i, _ in geq]])
        h += [-v for _, v in geq]
    G = sp.vstack(G).tocsc() if G else None
    h = np.array(h) if h else None
    return Aeq, beq, G, h


def main():
    m = read_mps_highs(MPS_BZ2)
    Aeq, beq, G, h = build_lp(m)
    dump = tempfile.NamedTemporaryFile(suffix=".bin", delete=False).name
    os.environ["POUNCE_DBG_KKT_DUMP"] = dump
    os.environ["POUNCE_DBG_KKT_DUMP_SKIP"] = "0"
    from pounce.qp import solve_qp

    solve_qp(P=None, c=m["c"], A=Aeq, b=beq, G=G, h=h, lb=m["cl"], ub=m["cu"], max_iter=2)
    if not os.path.exists(dump):
        sys.exit("no KKT dump produced — is this pounce build linsol-dump-capable?")

    with open(dump, "rb") as f:
        dim, nnz, _ = (int(x) for x in np.frombuffer(f.read(24), dtype="<u8"))
        airn = np.frombuffer(f.read(8 * nnz), dtype="<i8")  # 1-based lower triangle
        ajcn = np.frombuffer(f.read(8 * nnz), dtype="<i8")
        vals = np.frombuffer(f.read(8 * nnz), dtype="<f8")
    os.unlink(dump)

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as o:
        o.write("%%MatrixMarket matrix coordinate real symmetric\n")
        o.write(f"{dim} {dim} {nnz}\n")
        o.writelines(
            f"{i} {j} {v:.17e}\n"
            for i, j, v in zip(airn.tolist(), ajcn.tolist(), vals.tolist())
        )
    print(f"wrote {OUT}: dim={dim} nnz={nnz}")
    assert (dim, nnz) == (50880, 168105), "qap15 KKT dims drifted from issue #91"


if __name__ == "__main__":
    main()
