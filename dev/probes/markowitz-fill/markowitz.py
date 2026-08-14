"""Threshold-Markowitz LU as a FILL ORACLE for feral (discopt #1008).

The question this exists to answer: feral orders a basis with a STATIC
fill-reducing column permutation (AMD on the AtA pattern, optionally after a
Suhl-Suhl peel) and then factors with partial pivoting. Production LP INVERT
codes (LUSOL, HiGHS HFactor) instead choose each pivot DYNAMICALLY to minimize
the Markowitz count (r_i-1)(c_j-1) subject to a relative-magnitude threshold.
Is feral's fill an artifact of the static choice, or intrinsic to these bases?

SuperLU CANNOT answer this: it is the same algorithm class as feral (static
column order + partial pivoting), so agreement between them is not evidence
about the dynamic alternative. Hence this.

Speed is irrelevant here -- this is an oracle for nnz(L+U), not a kernel.

CORRECTNESS GATE. A fill number from a wrong factorization is worthless, so
`factor` returns L, U, and the permutations, and the caller checks
||P B Q - L U||_inf directly. Reported fill is only accepted when that residual
is at machine level.
"""
import heapq
import numpy as np
import scipy.sparse as sp


def markowitz_lu(B, u=0.1, verbose=False):
    """Right-looking threshold-Markowitz LU.  Returns (nnzL, nnzU, resid)."""
    m = B.shape[0]
    Bc = B.tocsc()
    # active submatrix, dict-of-dict both ways
    rows = [dict() for _ in range(m)]      # i -> {j: v}
    cols = [set() for _ in range(m)]       # j -> {i}
    Bco = B.tocoo()
    for i, j, v in zip(Bco.row, Bco.col, Bco.data):
        if v != 0.0:
            rows[i][j] = v
            cols[j].add(i)

    rcnt = np.array([len(rows[i]) for i in range(m)])
    ccnt = np.array([len(cols[j]) for j in range(m)])
    alive_r = np.ones(m, bool)
    alive_c = np.ones(m, bool)

    # bucket heaps (lazy): (count, idx); stale entries filtered on pop
    rheap = [(rcnt[i], i) for i in range(m)]
    cheap = [(ccnt[j], j) for j in range(m)]
    heapq.heapify(rheap)
    heapq.heapify(cheap)

    Lent, Uent = [], []          # (i, j, v) in ORIGINAL indices
    prow, pcol = [], []          # pivot order

    def min_alive_rcnt():
        while rheap:
            c, i = rheap[0]
            if alive_r[i] and rcnt[i] == c:
                return c
            heapq.heappop(rheap)
        return 0

    for step in range(m):
        # ---- pivot search: scan columns in increasing count order ----
        best = None            # (cost, |v|, i, j)
        scanned = []
        minr = max(min_alive_rcnt(), 1)
        examined = 0
        while cheap:
            c, j = cheap[0]
            if not alive_c[j] or ccnt[j] != c:
                heapq.heappop(cheap)
                continue
            if best is not None and (c - 1) * (minr - 1) > best[0]:
                break          # valid lower bound over all remaining columns
            heapq.heappop(cheap)
            scanned.append((c, j))
            colmax = max(abs(rows[i][j]) for i in cols[j])
            if colmax == 0.0:
                continue
            for i in cols[j]:
                v = rows[i][j]
                if abs(v) >= u * colmax:
                    cost = (rcnt[i] - 1) * (c - 1)
                    key = (cost, -abs(v))
                    if best is None or key < (best[0], -best[1]):
                        best = (cost, abs(v), i, j)
            examined += 1
            if best is not None and best[0] == 0:
                break
            if examined >= 8 and best is not None:
                break
        for c, j in scanned:
            if alive_c[j]:
                heapq.heappush(cheap, (ccnt[j], j))
        if best is None:
            raise RuntimeError(f"structurally singular at step {step}")
        _, _, pi, pj = best

        pv = rows[pi][pj]
        prow.append(pi)
        pcol.append(pj)
        alive_r[pi] = False
        alive_c[pj] = False

        # pivot row (-> U), pivot column (-> L)
        prow_entries = {j: v for j, v in rows[pi].items() if alive_c[j]}
        for j, v in rows[pi].items():
            Uent.append((pi, j, v))          # includes the pivot itself
        pcol_rows = [i for i in cols[pj] if alive_r[i]]

        # detach pivot row/col from the active structure
        for j in rows[pi]:
            cols[j].discard(pi)
            if alive_c[j]:
                ccnt[j] -= 1
                heapq.heappush(cheap, (ccnt[j], j))
        rows[pi] = {}

        # ---- eliminate ----
        for i in pcol_rows:
            mult = rows[i].pop(pj) / pv
            cols[pj].discard(i)
            rcnt[i] -= 1
            Lent.append((i, pj, mult))
            if mult == 0.0:
                heapq.heappush(rheap, (rcnt[i], i))
                continue
            ri = rows[i]
            for j, v in prow_entries.items():
                old = ri.get(j)
                if old is None:
                    ri[j] = -mult * v
                    cols[j].add(i)
                    rcnt[i] += 1
                    ccnt[j] += 1
                    heapq.heappush(cheap, (ccnt[j], j))
                else:
                    nv = old - mult * v
                    if nv == 0.0:
                        del ri[j]
                        cols[j].discard(i)
                        rcnt[i] -= 1
                        ccnt[j] -= 1
                        heapq.heappush(cheap, (ccnt[j], j))
                    else:
                        ri[j] = nv
            heapq.heappush(rheap, (rcnt[i], i))
        cols[pj] = set()
        if verbose and step % 500 == 0:
            print(f"    step {step}/{m} nnzU={len(Uent)} nnzL={len(Lent)}", flush=True)

    # ---- assemble and verify: P B Q == L U ----
    rperm = np.empty(m, int); rperm[np.array(prow)] = np.arange(m)   # orig row -> pos
    cperm = np.empty(m, int); cperm[np.array(pcol)] = np.arange(m)
    Ui = np.array([rperm[i] for i, j, v in Uent]); Uj = np.array([cperm[j] for i, j, v in Uent])
    Uv = np.array([v for i, j, v in Uent])
    Li = np.array([rperm[i] for i, j, v in Lent]); Lj = np.array([cperm[j] for i, j, v in Lent])
    Lv = np.array([v for i, j, v in Lent])
    U = sp.csr_matrix((Uv, (Ui, Uj)), shape=(m, m))
    L = sp.csr_matrix((np.concatenate([Lv, np.ones(m)]),
                       (np.concatenate([Li, np.arange(m)]),
                        np.concatenate([Lj, np.arange(m)]))), shape=(m, m))
    PBQ = B.tocsr()[np.array(prow), :][:, np.array(pcol)]
    R = (PBQ - (L @ U)).tocoo()
    resid = np.abs(R.data).max() if R.nnz else 0.0
    scale = np.abs(B.data).max()
    nnzL = len(Lent)                       # strictly-lower entries
    nnzU = len(Uent)                       # includes diagonal
    return nnzL, nnzU, resid / scale, L, U
