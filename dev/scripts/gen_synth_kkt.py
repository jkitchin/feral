#!/usr/bin/env python3
"""Generate a synthetic arrowhead conic-KKT stand-in for the qap15 fixture.

The real issue #91/#99 fixture (tests/data/large/qap15_kkt.mtx) is generated
from POUNCE + qap15.mps and cannot be reproduced in this container. This script
synthesizes a matrix with the same two structural features that drive the
issue #99 dense-front story, using only the Python standard library:

  1. A large *dense* border block of size `K` (default 2000) — an explicit
     clique that AMD eliminates last as a single dense root supernode, exactly
     like qap15's 2955x2955 indefinite root front (the dense-kernel bottleneck).
  2. `L` degree-2 "regularization" leaf columns (default 30000) — each a
     positive diagonal plus one off-diagonal to a random border node. With
     L/(L+K) ~ 94% of columns at <= 2 nonzeros, this trips the LdltCompress
     predicate (>=30% low-degree columns), exercising the issue #91
     OrderingPreprocess::Auto fill-verification fix on the end-to-end path.

Quasi-definite by construction: leaf diagonals > 0, border diagonals < 0.

Writes a symmetric MatrixMarket file (lower triangle, 1-indexed). Deterministic
(fixed seed). Env: K, L, OUT, SEED.
"""
import os
import random

K = int(os.environ.get("K", "2000"))       # dense border (root front size)
L = int(os.environ.get("L", "30000"))       # degree-2 leaf columns
OUT = os.environ.get("OUT", "tests/data/large/synth_arrow_kkt.mtx")
SEED = int(os.environ.get("SEED", "20260701"))

rng = random.Random(SEED)
N = L + K
border0 = L  # first border index (0-based)

# Collect lower-triangle entries (i >= j), 1-indexed on write.
# Border block: dense clique on indices [border0, N).
#   diagonal negative & dominant; small symmetric off-diagonal noise.
# Leaf block: indices [0, L). diagonal positive & dominant; one off-diagonal
#   to a random border node (creates the degree-2 signature and couples the
#   leaf into the border front).
rows = []  # (i, j, val) with i >= j

# Leaves.
for i in range(L):
    rows.append((i, i, 2.0 + rng.random()))          # positive diagonal
    b = border0 + rng.randrange(K)                    # one border neighbor
    # store lower triangle: (max, min)
    hi, lo = (b, i) if b > i else (i, b)
    rows.append((hi, lo, rng.uniform(-1.0, 1.0)))

# Dense border clique.
for a in range(border0, N):
    rows.append((a, a, -(float(K) + rng.random())))  # negative diagonal
    for b in range(border0, a):
        rows.append((a, b, 0.01 * rng.uniform(-1.0, 1.0)))

nnz = len(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("%%MatrixMarket matrix coordinate real symmetric\n")
    f.write(f"% synthetic arrowhead conic-KKT stand-in (K={K} dense border, "
            f"L={L} degree-2 leaves, seed={SEED})\n")
    f.write(f"{N} {N} {nnz}\n")
    for (i, j, v) in rows:
        f.write(f"{i + 1} {j + 1} {v:.12g}\n")

print(f"wrote {OUT}: dim={N} nnz={nnz} (border={K} dense, leaves={L} deg-2)")
