#!/usr/bin/env python3
"""Cross-solver comparison harness: feral vs MUMPS vs HSL MA97.

Reads the curated `sample.tsv`, generates one synthetic RHS per matrix
(b = A * x_true with x_true[i] = 1 + i/n), then invokes each solver
in turn with a shared manifest format:

    <mtx_path>  <rhs_path>  <out_path>

Per-solver sidecars land under `external_benchmarks/comparison/out/<solver>/`.
Aggregation into a single `comparison.json` is done by `aggregate.py`.

Usage:
    python3 run.py                 # full sample
    python3 run.py --limit 5       # smoke
    python3 run.py --solvers feral,ma97  # subset
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMP_DIR = Path(__file__).resolve().parent
OUT_DIR = COMP_DIR / "out"
RHS_DIR = COMP_DIR / "rhs"

FERAL_BIN = ROOT / "target" / "release" / "bench_one_matrix"
MUMPS_BIN = ROOT / "external_benchmarks" / "mumps_oracle" / "mumps_bench"
HSL_BIN = ROOT / "external_benchmarks" / "hsl_bench" / "hsl_bench"


def parse_sample(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, parts)))
    for r in rows:
        r["n"] = int(r["n"])
        r["nnz"] = int(r["nnz"])
    return rows


def matrix_path(row: dict) -> Path:
    sub = "kkt" if row["corpus"] == "kkt" else "kkt-mittelmann"
    return ROOT / "data" / "matrices" / sub / row["family"] / f"{row['matrix']}.mtx"


def read_mtx_lower_triplets(path: Path) -> tuple[int, list[tuple[int, int, float]]]:
    """Return (n, [(row, col, val)]) with row >= col (lower triangle).

    Matches the convention used by the C/Rust drivers so the synthetic
    RHS computed here is bit-equal across solvers up to the matrix
    multiply rounding."""
    trips: list[tuple[int, int, float]] = []
    n = 0
    saw_header = False
    with path.open() as f:
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.split()
            if not saw_header:
                m, n_, _nnz = int(parts[0]), int(parts[1]), int(parts[2])
                if m != n_:
                    raise ValueError("non-square")
                n = n_
                saw_header = True
                continue
            if len(parts) < 3:
                continue
            r = int(parts[0]) - 1
            c = int(parts[1]) - 1
            v = float(parts[2])
            if r < c:
                r, c = c, r
            trips.append((r, c, v))
    return n, trips


def synth_rhs(n: int, trips: list[tuple[int, int, float]]) -> list[float]:
    """b = A * x_true, x_true[i] = 1 + i/n. Mirrors the C/Rust drivers."""
    x = [1.0 + i / n for i in range(n)]
    b = [0.0] * n
    for r, c, v in trips:
        b[r] += v * x[c]
        if r != c:
            b[c] += v * x[r]
    return b


def write_rhs(rhs: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for v in rhs:
            f.write(f"{v:.17e}\n")


def build_manifest(rows: list[dict], solver: str) -> Path:
    """Build a manifest <mtx> <rhs> <out> per row for `solver`."""
    sub_out = OUT_DIR / solver
    sub_out.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=f"_{solver}.manifest", text=True)
    with os.fdopen(fd, "w") as f:
        for r in rows:
            mtx = matrix_path(r)
            rhs = RHS_DIR / f"{r['family']}__{r['matrix']}.rhs"
            out = sub_out / f"{r['family']}__{r['matrix']}.out"
            f.write(f"{mtx} {rhs} {out}\n")
    return Path(name)


def synth_all_rhs(rows: list[dict], force: bool = False) -> None:
    for r in rows:
        rhs_path = RHS_DIR / f"{r['family']}__{r['matrix']}.rhs"
        if rhs_path.exists() and not force:
            continue
        mtx = matrix_path(r)
        n, trips = read_mtx_lower_triplets(mtx)
        b = synth_rhs(n, trips)
        write_rhs(b, rhs_path)
        print(f"  rhs  {r['family']}/{r['matrix']}  (n={n})", flush=True)


def run_solver(solver: str, manifest: Path, time_limit_s: int) -> None:
    if solver == "feral":
        bin_ = FERAL_BIN
    elif solver == "mumps":
        bin_ = MUMPS_BIN
    elif solver == "ma97":
        bin_ = HSL_BIN
    else:
        raise ValueError(solver)
    if not bin_.exists():
        raise FileNotFoundError(bin_)
    print(f"\n=== {solver} ===", flush=True)
    try:
        subprocess.run([str(bin_), str(manifest)], check=False,
                       timeout=time_limit_s)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {time_limit_s}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(COMP_DIR / "sample.tsv"))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of matrices (smoke run)")
    ap.add_argument("--solvers", default="feral,mumps,ma97")
    ap.add_argument("--time-limit", type=int, default=600,
                    help="seconds per solver, whole manifest")
    ap.add_argument("--force-rhs", action="store_true")
    args = ap.parse_args()

    rows = parse_sample(Path(args.sample))
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"sample: {len(rows)} matrices "
          f"(n {rows[0]['n']} .. {rows[-1]['n']})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RHS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== synthesizing RHS ===", flush=True)
    synth_all_rhs(rows, force=args.force_rhs)

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    for solver in solvers:
        manifest = build_manifest(rows, solver)
        run_solver(solver, manifest, args.time_limit)

    print("\n=== done ===", flush=True)
    print(f"sidecars: {OUT_DIR}")
    print("aggregate with: python3 aggregate.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
