"""Build chain-aware (time-ordered) permutations for the issue #203 KKT patterns.

Emits `OrderingMethod::External` permutation files: one 0-based ORIGINAL index
per line, in new order (new-to-old, `perm[k] = original column that became k`).

Layout of the KKT (from kkt_pattern_repro.py):
  variables   0                      .. nz-1          state (p, r)  = p*NS + r
              nz                     .. n-1           control (c,k) = nz + c*nk + k
  constraints n + 0                  .. n+nz-1        state residual (p,r)
              n + nz                 .. n+nz+npt*NE-1 path (p,e)
              n + nz + npt*NE        .. n+m-1         ramp (c,k)

Variants:
  blocked      — per time point: all state vars, then all state residuals,
                 then path constraints; controls/ramps at their knot time.
  interleaved  — per time point: (var, residual) pairs interleaved, then path.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import kkt_pattern_repro as R  # noqa: E402

NS, NC, NE, NCP = R.NS, R.NC, R.NE, R.NCP


def time_perm(nfe, variant):
    nk = nfe + 1
    npt = nfe * NCP
    nz = npt * NS
    n = nz + NC * nk
    cons0 = n                    # state residuals
    path0 = n + nz               # path constraints
    ramp0 = n + nz + npt * NE    # ramp constraints

    # bucket[t] collects the indices whose time key is t, t in 0..npt-1.
    # A knot k lives at the first collocation point of element k, i.e.
    # point k*NCP; clamp the final knot into the last point.
    buckets = [[] for _ in range(npt)]
    for p in range(npt):
        b = buckets[p]
        if variant == "interleaved":
            for r in range(NS):
                b.append(p * NS + r)
                b.append(cons0 + p * NS + r)
        else:
            b.extend(range(p * NS, (p + 1) * NS))
            b.extend(range(cons0 + p * NS, cons0 + (p + 1) * NS))
        b.extend(range(path0 + p * NE, path0 + (p + 1) * NE))

    for c in range(NC):
        for k in range(nk):
            t = min(k * NCP, npt - 1)
            buckets[t].append(nz + c * nk + k)
        for k in range(nk - 1):
            t = min((k + 1) * NCP - 1, npt - 1)
            buckets[t].append(ramp0 + c * (nk - 1) + k)

    perm = np.concatenate([np.array(b, dtype=np.int64) for b in buckets])
    N = n + nz + npt * NE + NC * (nk - 1)
    assert perm.size == N, (perm.size, N)
    assert np.array_equal(np.sort(perm), np.arange(N))
    return perm


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for nfe in map(int, sys.argv[2:]):
        for variant in ("blocked", "interleaved"):
            perm = time_perm(nfe, variant)
            path = out / f"perm_{variant}_nfe{nfe:03d}.txt"
            np.savetxt(path, perm, fmt="%d")
            print(f"{path}  N={perm.size}")
