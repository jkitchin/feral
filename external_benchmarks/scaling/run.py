#!/usr/bin/env python3
"""Scaling benchmark: feral vs MUMPS vs MA57 on synthetic matrix sweeps.

Generates synthetic symmetric matrices at multiple sizes for four families:
  - dense_si:    dense symmetric indefinite (random A + A^T - cI)
  - banded_spd:  banded SPD, fixed bandwidth (default 10)
  - laplace2d:   5-point Laplacian SPD on k x k grid (n = k^2)
  - saddle_kkt:  synthetic saddle-point KKT [H A^T; A 0]

Runs each through the existing per-matrix solver bench binaries
(bench_one_matrix, mumps_bench, ma57_bench), aggregates the per-matrix
sidecars into scaling.tsv, and with --report prints log-log slope
estimates per (solver, family).

Usage:
    python3 run.py                       # full sweep, all 3 solvers, all 4 families
    python3 run.py --families laplace2d  # subset
    python3 run.py --solvers feral,mumps
    python3 run.py --max-n 4096          # cap problem size
    python3 run.py --report              # aggregate + slope report only

Output layout under external_benchmarks/scaling/:
    matrices/<family>/n<size>.mtx        generated matrix (MatrixMarket, symmetric)
    rhs/<family>/n<size>.rhs             RHS = A * (1, 1+1/n, 1+2/n, ...)
    out/<solver>/<family>__n<size>.txt   per-matrix sidecar
    scaling.tsv                          aggregated results
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
SCALING_DIR = Path(__file__).resolve().parent
MTX_DIR = SCALING_DIR / "matrices"
RHS_DIR = SCALING_DIR / "rhs"
OUT_DIR = SCALING_DIR / "out"
TSV_PATH = SCALING_DIR / "scaling.tsv"

FERAL_BIN = ROOT / "target" / "release" / "bench_one_matrix"
MUMPS_BIN = ROOT / "external_benchmarks" / "mumps_oracle" / "mumps_bench"
MA57_BIN = ROOT / "external_benchmarks" / "ma57_oracle" / "ma57_bench"

DEFAULT_SIZES = {
    "dense_si": [64, 128, 256, 512, 1024],
    "banded_spd": [1024, 4096, 16384, 65536, 262144],
    "laplace2d": [16, 32, 64, 96, 128, 192],  # k; n = k^2
    "saddle_kkt": [64, 128, 256, 512, 1024],  # H block dimension; total n = 2 * n_H
}
BANDED_BANDWIDTH = 10
SADDLE_NNZ_PER_ROW = 5
RNG_SEED = 0xFE2A1


def write_mtx_symmetric(path: Path, n: int, lower_trips: list[tuple[int, int, float]]) -> None:
    """Write symmetric MatrixMarket (1-indexed, lower triangle only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"{n} {n} {len(lower_trips)}\n")
        for r, c, v in lower_trips:
            f.write(f"{r + 1} {c + 1} {v:.17e}\n")


def write_rhs(path: Path, b: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for v in b:
            f.write(f"{v:.17e}\n")


def synth_rhs(n: int, lower_trips: list[tuple[int, int, float]]) -> list[float]:
    """b = A * x_true, x_true[i] = 1 + i/n (matches comparison/run.py)."""
    x = [1.0 + i / n for i in range(n)]
    b = [0.0] * n
    for r, c, v in lower_trips:
        b[r] += v * x[c]
        if r != c:
            b[c] += v * x[r]
    return b


def gen_dense_si(n: int, seed: int) -> list[tuple[int, int, float]]:
    """Dense symmetric indefinite: A = (R + R^T)/2 with diagonal shift to keep |λ| bounded.

    Random R with entries in [-1, 1]; diagonal regularization c*I (small) ensures
    factorization succeeds even when otherwise singular. Indefinite by construction
    (random symmetric has eigenvalues of both signs by Wigner)."""
    rng = random.Random(seed)
    if np is not None:
        rs = np.random.RandomState(seed)
        R = rs.uniform(-1.0, 1.0, size=(n, n))
        A = (R + R.T) * 0.5
        A += np.eye(n) * 1e-3
        trips: list[tuple[int, int, float]] = []
        for j in range(n):
            for i in range(j, n):
                trips.append((i, j, float(A[i, j])))
        return trips
    # Pure-python fallback (slow; only for tiny n).
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            v = rng.uniform(-1.0, 1.0)
            A[i][j] = v
            A[j][i] = v
        A[i][i] += 1e-3
    trips = []
    for j in range(n):
        for i in range(j, n):
            trips.append((i, j, A[i][j]))
    return trips


def gen_banded_spd(n: int, bw: int, seed: int) -> list[tuple[int, int, float]]:
    """Banded SPD: random A_band, then A = A_band^T * A_band + small diagonal.

    Here we go simpler: tridiagonal-style SPD with diagonal dominance.
    For bw=k: A[i,i] = 2*k+1, A[i, i±j] = -1 for 1 <= j <= k."""
    trips: list[tuple[int, int, float]] = []
    for i in range(n):
        trips.append((i, i, float(2 * bw + 1)))
        for j in range(1, bw + 1):
            if i - j >= 0:
                trips.append((i, i - j, -1.0))
    return trips


def gen_laplace2d(k: int) -> tuple[int, list[tuple[int, int, float]]]:
    """5-point Laplacian on k x k grid (Dirichlet). n = k * k."""
    n = k * k
    trips: list[tuple[int, int, float]] = []

    def idx(r: int, c: int) -> int:
        return r * k + c

    for r in range(k):
        for c in range(k):
            ii = idx(r, c)
            trips.append((ii, ii, 4.0))
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < k and 0 <= cc < k:
                    jj = idx(rr, cc)
                    if jj < ii:
                        trips.append((ii, jj, -1.0))
                    else:
                        trips.append((jj, ii, -1.0))
    return n, trips


def gen_saddle_kkt(n_h: int, seed: int) -> tuple[int, list[tuple[int, int, float]]]:
    """Synthetic KKT: [H A^T; A 0] with H SPD (banded), A random sparse.

    n_h: block dimension for H. Total system size n = 2 * n_h.
    Each row of A has ~SADDLE_NNZ_PER_ROW nonzeros."""
    rng = random.Random(seed)
    n = 2 * n_h
    trips: list[tuple[int, int, float]] = []

    # H block: tridiagonal SPD (rows 0 .. n_h-1).
    for i in range(n_h):
        trips.append((i, i, 4.0))
        if i > 0:
            trips.append((i, i - 1, -1.0))

    # A^T block: rows n_h .. 2*n_h-1 contribute zero on diagonal (saddle 0 block).
    # A block: entries (i, j) with i in [n_h, 2*n_h) and j in [0, n_h).
    for i_a in range(n_h):
        seen: set[int] = set()
        while len(seen) < SADDLE_NNZ_PER_ROW:
            j = rng.randrange(n_h)
            seen.add(j)
        for j in sorted(seen):
            v = rng.uniform(-1.0, 1.0)
            trips.append((n_h + i_a, j, v))

    return n, trips


def generate_matrix(family: str, size_param: int) -> tuple[int, Path, Path]:
    """Generate matrix + RHS files for (family, size_param). Returns (n, mtx_path, rhs_path)."""
    if family == "dense_si":
        n = size_param
        trips = gen_dense_si(n, seed=RNG_SEED + n)
        tag = f"n{n}"
    elif family == "banded_spd":
        n = size_param
        trips = gen_banded_spd(n, bw=BANDED_BANDWIDTH, seed=RNG_SEED + n)
        tag = f"n{n}"
    elif family == "laplace2d":
        n, trips = gen_laplace2d(size_param)
        tag = f"k{size_param}_n{n}"
    elif family == "saddle_kkt":
        n, trips = gen_saddle_kkt(size_param, seed=RNG_SEED + size_param)
        tag = f"nh{size_param}_n{n}"
    else:
        raise ValueError(f"unknown family: {family}")

    mtx_path = MTX_DIR / family / f"{tag}.mtx"
    rhs_path = RHS_DIR / family / f"{tag}.rhs"

    if not mtx_path.exists():
        write_mtx_symmetric(mtx_path, n, trips)
    if not rhs_path.exists():
        b = synth_rhs(n, trips)
        write_rhs(rhs_path, b)
    return n, mtx_path, rhs_path


def build_manifest(jobs: list[dict], solver: str) -> Path:
    sub_out = OUT_DIR / solver
    sub_out.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=f"_scaling_{solver}.manifest", text=True)
    with os.fdopen(fd, "w") as f:
        for j in jobs:
            out = sub_out / f"{j['family']}__{j['tag']}.txt"
            f.write(f"{j['mtx']} {j['rhs']} {out}\n")
            j[f"out_{solver}"] = out
    return Path(name)


def run_solver(solver: str, manifest: Path, time_limit_s: int) -> None:
    if solver == "feral":
        bin_ = FERAL_BIN
    elif solver == "mumps":
        bin_ = MUMPS_BIN
    elif solver == "ma57":
        bin_ = MA57_BIN
    else:
        raise ValueError(solver)
    if not bin_.exists():
        print(f"  SKIP {solver}: binary missing at {bin_}", flush=True)
        return
    print(f"\n=== {solver} ===", flush=True)
    try:
        subprocess.run([str(bin_), str(manifest)], check=False, timeout=time_limit_s)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {time_limit_s}s", flush=True)


def parse_sidecar(path: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    if not path.exists():
        return d
    with path.open() as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def aggregate(jobs: list[dict], solvers: list[str]) -> list[dict]:
    rows: list[dict] = []
    for j in jobs:
        for solver in solvers:
            out = OUT_DIR / solver / f"{j['family']}__{j['tag']}.txt"
            d = parse_sidecar(out)
            if not d:
                continue
            # Sidecar key aliases:
            #   MUMPS writes `residual`; feral and MA57 write `rel_res`.
            #   MUMPS has no `analyse_us` key (factor_us bundles analyse).
            rel_res = d.get("rel_res") or d.get("residual") or ""
            rows.append({
                "solver": d.get("solver", solver),
                "family": j["family"],
                "tag": j["tag"],
                "n": j["n"],
                "nnz": d.get("nnz", ""),
                "analyse_us": d.get("analyse_us", ""),
                "factor_us": d.get("factor_us", ""),
                "solve_us": d.get("solve_us", ""),
                "rel_res": rel_res,
                "status": d.get("status", ""),
            })
    return rows


def write_tsv(rows: list[dict]) -> None:
    if not rows:
        print("  (no rows to write)", flush=True)
        return
    cols = ["solver", "family", "tag", "n", "nnz", "analyse_us",
            "factor_us", "solve_us", "rel_res", "status"]
    with TSV_PATH.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {TSV_PATH} ({len(rows)} rows)", flush=True)


def report(rows: list[dict]) -> None:
    """Print log-log slope of factor_us vs n per (solver, family)."""
    by_key: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r in rows:
        try:
            n = float(r["n"])
            t = float(r["factor_us"])
        except (ValueError, KeyError):
            continue
        if n <= 0 or t <= 0:
            continue
        by_key.setdefault((r["solver"], r["family"]), []).append((n, t))

    print(f"\n{'solver':<20}{'family':<14}{'pts':>4}{'slope':>10}{'ref(n=N0)':>14}")
    for (solver, family), pts in sorted(by_key.items()):
        if len(pts) < 2:
            continue
        pts.sort()
        xs = [math.log(p[0]) for p in pts]
        ys = [math.log(p[1]) for p in pts]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        slope = num / den if den > 0 else 0.0
        # Reference time at smallest n.
        ref_n, ref_t = pts[0]
        print(f"{solver:<20}{family:<14}{len(pts):>4}{slope:>10.2f}"
              f"  {ref_t:>8.0f}us@n={int(ref_n)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default=",".join(DEFAULT_SIZES.keys()),
                    help="comma-separated subset of " + ",".join(DEFAULT_SIZES.keys()))
    ap.add_argument("--solvers", default="feral,mumps,ma57",
                    help="comma-separated subset of feral,mumps,ma57")
    ap.add_argument("--max-n", type=int, default=None,
                    help="cap total system size n (post-expansion for laplace2d/saddle)")
    ap.add_argument("--time-limit", type=int, default=900,
                    help="per-solver timeout (whole manifest), seconds")
    ap.add_argument("--report", action="store_true",
                    help="aggregate existing sidecars and print slope report; skip running")
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    for f in families:
        if f not in DEFAULT_SIZES:
            print(f"unknown family: {f}", file=sys.stderr)
            return 2

    MTX_DIR.mkdir(parents=True, exist_ok=True)
    RHS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the job list (generates matrices lazily).
    jobs: list[dict] = []
    if not args.report:
        print("=== generating matrices ===", flush=True)
        for family in families:
            for size in DEFAULT_SIZES[family]:
                n, mtx, rhs = generate_matrix(family, size)
                if args.max_n is not None and n > args.max_n:
                    continue
                tag = mtx.stem
                jobs.append({"family": family, "tag": tag, "n": n,
                             "mtx": mtx, "rhs": rhs})
                print(f"  {family:<12} {tag:<20} n={n:>7}", flush=True)

        for solver in solvers:
            manifest = build_manifest(jobs, solver)
            run_solver(solver, manifest, args.time_limit)
    else:
        # In --report mode, rebuild the job list from filesystem (no regeneration).
        for family in families:
            for size in DEFAULT_SIZES[family]:
                # Predict n + tag from size_param without regenerating.
                if family in ("dense_si", "banded_spd"):
                    n = size
                    tag = f"n{n}"
                elif family == "laplace2d":
                    n = size * size
                    tag = f"k{size}_n{n}"
                elif family == "saddle_kkt":
                    n = 2 * size
                    tag = f"nh{size}_n{n}"
                if args.max_n is not None and n > args.max_n:
                    continue
                jobs.append({"family": family, "tag": tag, "n": n})

    print("\n=== aggregating ===", flush=True)
    rows = aggregate(jobs, solvers)
    write_tsv(rows)

    print("\n=== slope report ===", flush=True)
    report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
