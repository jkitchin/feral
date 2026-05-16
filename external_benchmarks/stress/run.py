#!/usr/bin/env python3
"""Stress-suite runner for feral.

Reads `manifest.tsv`, locates each matrix under `matrices/<group>/<name>.mtx`
(SuiteSparse or synthetic), generates a synthetic RHS via b = A * x_true with
x_true[i] = 1 + i/n, then invokes `target/release/bench_one_matrix` with a
single shared manifest. Per-matrix sidecars land under `out/feral/`.

Usage:
    python3 run.py                  # whole manifest
    python3 run.py --limit 5        # smoke
    python3 run.py --category rankdef,illcond
    python3 run.py --time-limit 1200
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STRESS_DIR = Path(__file__).resolve().parent
MATRICES_DIR = STRESS_DIR / "matrices"
RHS_DIR = STRESS_DIR / "rhs"
OUT_DIR = STRESS_DIR / "out"

FERAL_BIN = ROOT / "target" / "release" / "bench_one_matrix"


def parse_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rows.append(dict(zip(header, parts)))
    for r in rows:
        r["n"] = int(r["n"])
        r["nnz"] = int(r["nnz"])
    return rows


def matrix_path(row: dict) -> Path:
    return MATRICES_DIR / row["group"] / f"{row['name']}.mtx"


def read_mtx_lower_triplets(path: Path) -> tuple[int, list[tuple[int, int, float]]]:
    """Read symmetric MatrixMarket and return (n, lower-tri triplets)."""
    trips: list[tuple[int, int, float]] = []
    n = 0
    saw_header = False
    with path.open() as f:
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.split()
            if not parts:
                continue
            if not saw_header:
                m, n_, _nnz = int(parts[0]), int(parts[1]), int(parts[2])
                if m != n_:
                    raise ValueError(f"non-square matrix in {path}")
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


def synth_all_rhs(rows: list[dict], force: bool = False) -> list[dict]:
    """Generate RHS for every row whose matrix file exists. Returns the
    subset that is fully ready (matrix present + rhs written)."""
    ready = []
    for r in rows:
        mtx = matrix_path(r)
        if not mtx.exists():
            print(f"  MISSING {r['group']}/{r['name']} -> "
                  f"{mtx.relative_to(STRESS_DIR.parent.parent)}", flush=True)
            continue
        rhs_path = RHS_DIR / f"{r['group']}__{r['name']}.rhs"
        if not rhs_path.exists() or force:
            n, trips = read_mtx_lower_triplets(mtx)
            if n != r["n"]:
                print(f"  WARN n mismatch for {r['name']}: "
                      f"manifest={r['n']} file={n}", flush=True)
            b = synth_rhs(n, trips)
            write_rhs(b, rhs_path)
            print(f"  rhs {r['group']}/{r['name']} (n={n})", flush=True)
        ready.append(r)
    return ready


def build_manifest(rows: list[dict]) -> Path:
    sub_out = OUT_DIR / "feral"
    sub_out.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix="_stress.manifest", text=True)
    with os.fdopen(fd, "w") as f:
        for r in rows:
            mtx = matrix_path(r)
            rhs = RHS_DIR / f"{r['group']}__{r['name']}.rhs"
            out = sub_out / f"{r['group']}__{r['name']}.out"
            f.write(f"{mtx} {rhs} {out}\n")
    return Path(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(STRESS_DIR / "manifest.tsv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--category", default=None,
                    help="comma-separated category filter "
                         "(rankdef,illcond,near_sing,cascade,saddle,...)")
    ap.add_argument("--max-n", type=int, default=None,
                    help="cap matrix size for smoke runs")
    ap.add_argument("--time-limit", type=int, default=1800)
    ap.add_argument("--force-rhs", action="store_true")
    args = ap.parse_args()

    rows = parse_manifest(Path(args.manifest))
    if args.category:
        wanted = {c.strip() for c in args.category.split(",") if c.strip()}
        rows = [r for r in rows if r.get("category") in wanted]
    if args.max_n is not None:
        rows = [r for r in rows if r["n"] <= args.max_n]
    # Sort by n ascending so smoke runs hit small ones first.
    rows.sort(key=lambda r: (r["n"], r["name"]))
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"stress: {len(rows)} matrices selected", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RHS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== synthesizing RHS ===", flush=True)
    ready = synth_all_rhs(rows, force=args.force_rhs)
    print(f"  ready: {len(ready)}/{len(rows)}", flush=True)
    if not ready:
        print("nothing to run. Run fetch.py / synth.py first.", flush=True)
        return 0

    if not FERAL_BIN.exists():
        print(f"\nERROR: {FERAL_BIN} missing.", flush=True)
        print("Build with: cargo build --release --bin bench_one_matrix",
              flush=True)
        return 2

    manifest = build_manifest(ready)
    print("\n=== feral ===", flush=True)
    try:
        subprocess.run([str(FERAL_BIN), str(manifest)], check=False,
                       timeout=args.time_limit)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {args.time_limit}s", flush=True)

    print("\n=== done ===", flush=True)
    print(f"sidecars: {OUT_DIR}")
    print("analyze with: python3 report.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
