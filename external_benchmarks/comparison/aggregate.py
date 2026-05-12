#!/usr/bin/env python3
"""Aggregate per-solver per-matrix sidecars into comparison.json.

Output schema (list of records):

    {
      "matrix": {
        "corpus": "kkt|mittelmann",
        "family": "<name>",
        "id": "<matrix-id>",
        "n": <int>,
        "nnz": <int>,
        "density": <float>          # nnz_stored / (n*(n+1)/2), lower-tri
      },
      "solvers": {
        "feral":  {"factor_us": ..., "solve_us": ..., "rel_res": ...,
                    "inertia": [pos, neg, zero], "status": "ok|fail",
                    "fail_reason": "..."},
        "mumps":  {...},
        "ma97":   {...}
      }
    }

Usage: python3 aggregate.py  [--out comparison.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

COMP_DIR = Path(__file__).resolve().parent
OUT_DIR = COMP_DIR / "out"

SOLVERS = ["feral", "mumps", "ma97"]


def parse_sidecar(path: Path) -> dict | None:
    """Parse the `key value` text sidecar emitted by each driver."""
    if not path.exists():
        return None
    d: dict = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            k, v = parts
            d[k] = v
    return d


def to_int(d: dict, k: str) -> int | None:
    v = d.get(k)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def to_float(d: dict, k: str) -> float | None:
    v = d.get(k)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(COMP_DIR / "sample.tsv"))
    ap.add_argument("--out", default=str(COMP_DIR / "comparison.json"))
    args = ap.parse_args()

    sample_rows = []
    with open(args.sample) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sample_rows.append(dict(zip(header, parts)))

    records = []
    for r in sample_rows:
        n = int(r["n"])
        nnz = int(r["nnz"])
        density = nnz / (n * (n + 1) / 2) if n > 0 else 0.0
        rec = {
            "matrix": {
                "corpus": r["corpus"],
                "family": r["family"],
                "id": r["matrix"],
                "n": n,
                "nnz": nnz,
                "density": density,
            },
            "solvers": {},
        }
        for solver in SOLVERS:
            sidecar = OUT_DIR / solver / f"{r['family']}__{r['matrix']}.out"
            d = parse_sidecar(sidecar)
            if d is None:
                rec["solvers"][solver] = {"status": "missing"}
                continue
            status = d.get("status", "unknown")
            entry: dict = {
                "status": status,
                "solver_version": d.get("solver"),
            }
            for k in ("analyse_us", "factor_us", "solve_us"):
                v = to_int(d, k)
                if v is not None:
                    entry[k] = v
            # rel_res key name varies by solver: feral/ma97 emit
            # `rel_res`, mumps emits `residual` (both are
            # ||Ax-b||_2 / ||b||_2).
            rel = to_float(d, "rel_res")
            if rel is None:
                rel = to_float(d, "residual")
            if rel is not None and math.isfinite(rel):
                entry["rel_res"] = rel
            # mumps also reports scaled residual rinfog6 and the two
            # componentwise condition numbers rinfog10 (COND1) and
            # rinfog11 (COND2). Carry all three for diagnostics; the
            # report's ill-conditioned section keys off rinfog10.
            for k in ("rinfog6", "rinfog10", "rinfog11"):
                v = to_float(d, k)
                if v is not None and math.isfinite(v):
                    entry[k] = v
            pos = to_int(d, "inertia_pos")
            neg = to_int(d, "inertia_neg")
            zer = to_int(d, "inertia_zero")
            if pos is not None and neg is not None and zer is not None:
                entry["inertia"] = [pos, neg, zer]
            for k in ("fail_reason", "nnz_l", "infog1"):
                if k in d:
                    entry[k] = d[k]
            rec["solvers"][solver] = entry
        records.append(rec)

    # Sort by n ascending for readability.
    records.sort(key=lambda r: (r["matrix"]["n"], r["matrix"]["family"]))
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {args.out} ({len(records)} records)")

    # Quick stdout summary table.
    print()
    print(f"{'n':>8} {'nnz':>8} {'family':<22}", end="")
    for s in SOLVERS:
        print(f" | {s+' fac_us':>11} {s+' rres':>10}", end="")
    print()
    for rec in records:
        m = rec["matrix"]
        print(f"{m['n']:>8} {m['nnz']:>8} {m['family'][:22]:<22}", end="")
        for s in SOLVERS:
            e = rec["solvers"].get(s, {})
            if e.get("status") == "ok":
                fu = e.get("factor_us", 0)
                rr = e.get("rel_res", float("nan"))
                print(f" | {fu:>11} {rr:>10.2e}", end="")
            else:
                tag = e.get("status", "?")
                print(f" | {tag:>11} {'-':>10}", end="")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
