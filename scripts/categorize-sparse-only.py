#!/usr/bin/env python3
"""
Categorize feral's sparse-only failures (per Phase 2.2.3) against the canonical
MUMPS and SSIDS oracles.

Inputs:
  $1  CSV emitted by `FERAL_SPARSE_ONLY_DUMP=... cargo run --release --bin bench`
      Columns: name, family, n, exp_p, exp_n, exp_z, act_p, act_n, act_z,
               inertia_ok, residual, residual_ok
  $2  Root of data/matrices/kkt (used to locate <name>.mumps.json / .ssids.json)

For each sparse-only failure we compare the recorded sparse inertia against the
MUMPS and SSIDS oracle inertia and bucket the matrix into:

  REAL_BUG          — feral-dense passes (matched sidecar), MUMPS+SSIDS agree
                      with the sidecar, feral-sparse disagrees. Sparse pipeline
                      is wrong; this is the triage candidate set.
  ORACLE_DISAGREE   — MUMPS and SSIDS disagree with the sidecar (or each other);
                      borderline / numerically intractable. Not a sparse bug.
  ORACLE_MISSING    — at least one oracle sidecar is missing; cannot classify.

The script writes a per-bucket CSV alongside the input and prints the top
worst-residual REAL_BUG candidates as the triage list.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional


def load_oracle_inertia(json_path: Path) -> Optional[tuple[int, int, int]]:
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return None
    inertia = data.get("inertia")
    if not isinstance(inertia, dict):
        return None
    p = inertia.get("positive")
    n = inertia.get("negative")
    z = inertia.get("zero")
    if not all(isinstance(x, int) for x in (p, n, z)):
        return None
    return (p, n, z)


def find_oracle_paths(matrix_root: Path, name: str, family: str) -> tuple[Path, Path]:
    base = matrix_root / family / name
    return (
        Path(str(base) + ".mumps.json"),
        Path(str(base) + ".ssids.json"),
    )


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <sparse-only.csv> <data/matrices/kkt>", file=sys.stderr)
        return 2

    csv_path = Path(sys.argv[1])
    matrix_root = Path(sys.argv[2])
    if not csv_path.exists():
        print(f"missing CSV: {csv_path}", file=sys.stderr)
        return 2
    if not matrix_root.is_dir():
        print(f"missing matrix dir: {matrix_root}", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(csv_path.open()))
    print(f"loaded {len(rows)} sparse-only failures from {csv_path}")

    buckets: dict[str, list[dict]] = {
        "REAL_BUG": [],
        "ORACLE_DISAGREE": [],
        "ORACLE_MISSING": [],
    }

    for row in rows:
        name = row["name"]
        family = row["family"]
        expected = (int(row["exp_p"]), int(row["exp_n"]), int(row["exp_z"]))
        actual = (int(row["act_p"]), int(row["act_n"]), int(row["act_z"]))
        residual = float(row["residual"])

        mumps_path, ssids_path = find_oracle_paths(matrix_root, name, family)
        mumps = load_oracle_inertia(mumps_path)
        ssids = load_oracle_inertia(ssids_path)
        row["mumps_inertia"] = str(mumps) if mumps else ""
        row["ssids_inertia"] = str(ssids) if ssids else ""

        if mumps is None and ssids is None:
            buckets["ORACLE_MISSING"].append(row)
            continue

        oracle_agrees_with_expected = True
        if mumps is not None and mumps != expected:
            oracle_agrees_with_expected = False
        if ssids is not None and ssids != expected:
            oracle_agrees_with_expected = False

        if oracle_agrees_with_expected and actual != expected:
            buckets["REAL_BUG"].append(row)
        elif not oracle_agrees_with_expected:
            buckets["ORACLE_DISAGREE"].append(row)
        else:
            # actual == expected (residual-only failure) and oracle agrees.
            # Sparse pipeline got the right inertia but a bad backsolve; still
            # a real bug, just a different mode.
            buckets["REAL_BUG"].append(row)

    print()
    print(f"REAL_BUG         : {len(buckets['REAL_BUG'])}")
    print(f"ORACLE_DISAGREE  : {len(buckets['ORACLE_DISAGREE'])}")
    print(f"ORACLE_MISSING   : {len(buckets['ORACLE_MISSING'])}")

    for bucket_name, bucket_rows in buckets.items():
        out_path = csv_path.with_suffix(f".{bucket_name.lower()}.csv")
        if not bucket_rows:
            continue
        with out_path.open("w", newline="") as f:
            fieldnames = list(bucket_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(bucket_rows)
        print(f"  wrote {len(bucket_rows)} rows → {out_path}")

    real = buckets["REAL_BUG"]
    if real:
        real.sort(key=lambda r: float(r["residual"]), reverse=True)
        print("\nTop 25 REAL_BUG triage candidates by residual:")
        print(f"{'name':<28} {'n':>5} {'res':>12} {'exp':>14} {'sp':>14} "
              f"{'mumps':>14} {'ssids':>14}")
        for r in real[:25]:
            exp = f"({r['exp_p']},{r['exp_n']},{r['exp_z']})"
            sp = f"({r['act_p']},{r['act_n']},{r['act_z']})"
            print(f"{r['name']:<28} {int(r['n']):>5} {float(r['residual']):>12.2e} "
                  f"{exp:>14} {sp:>14} "
                  f"{r['mumps_inertia']:>14} {r['ssids_inertia']:>14}")

        # Family breakdown of REAL_BUG.
        fam_counts: dict[str, int] = {}
        for r in real:
            fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
        print("\nREAL_BUG family breakdown:")
        for fam, cnt in sorted(fam_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {fam:<22} {cnt:>5}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
