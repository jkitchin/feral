#!/usr/bin/env python3
"""Run the native MUMPS oracle on a directory of KKT matrices.

For each <id>.mtx in the input tree:
  1. Read the matching <id>.json sidecar to get the RHS.
  2. Write a temp <id>.rhs.txt with one f64 per line.
  3. Add the matrix to a manifest passed to mumps_bench.
After mumps_bench finishes, parse each output text file and write
a canonical <id>.mumps.json sidecar next to the .mtx.

Usage:
    python3 run_mumps.py data/matrices/kkt
    python3 run_mumps.py data/matrices/kkt --limit 10
    python3 run_mumps.py data/matrices/kkt --skip-existing
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MUMPS_BENCH = SCRIPT_DIR / "mumps_bench"


def find_matrices(root: Path) -> list[Path]:
    return sorted(root.rglob("*.mtx"))


def load_rhs(json_path: Path) -> list[float] | None:
    """Return the RHS vector from a sidecar, or None if invalid.

    Mirrors src/io/sidecar.rs::finite_rhs: any non-numeric or
    non-finite entry causes the matrix to be skipped, matching the
    feral bench's filter.
    """
    try:
        data = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    rhs = data.get("rhs")
    if not isinstance(rhs, list):
        return None
    out: list[float] = []
    for v in rhs:
        if not isinstance(v, (int, float)):
            return None
        f = float(v)
        if not math.isfinite(f):
            return None
        out.append(f)
    return out


def write_rhs(rhs: list[float], path: Path) -> None:
    with path.open("w") as f:
        for v in rhs:
            f.write(f"{v:.17e}\n")


def parse_output(path: Path) -> dict:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def write_canonical(out_path: Path, name: str, raw: dict) -> None:
    """Translate the mumps_bench plain-text output into the canonical
    JSON schema agreed in dev/plans/phase-1b-consensus-exit.md."""
    if raw.get("status") != "ok":
        canonical = {
            "solver": "mumps-5.8.2",
            "version": "5.8.2",
            "matrix": name,
            "factorization_status": raw.get("status", "fail"),
        }
    else:
        canonical = {
            "solver": "mumps-5.8.2",
            "version": "5.8.2",
            "matrix": name,
            "n": int(raw.get("n", 0)),
            "nnz": int(raw.get("nnz", 0)),
            "factor_us": int(raw.get("factor_us", 0)),
            "solve_us": int(raw.get("solve_us", 0)),
            "inertia": {
                "positive": int(raw.get("inertia_pos", 0)),
                "negative": int(raw.get("inertia_neg", 0)),
                "zero": int(raw.get("inertia_zero", 0)),
            },
            "rhs_source": "sidecar",
            "residual_2norm_relative": float(raw.get("residual", "nan")),
            "factorization_status": "ok",
            # factor_nnz = INFOG(9): real entries effectively used in
            # factors. Source of truth for feral / MUMPS fill parity.
            "factor_nnz": int(raw.get("infog9", 0)),
            "solver_info": {
                "infog_1": int(raw.get("infog1", 0)),
                "infog_28": int(raw.get("infog28", 0)),
                "infog_9": int(raw.get("infog9", 0)),
            },
        }
    out_path.write_text(json.dumps(canonical) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="root directory containing .mtx files")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N matrices (for sanity checks)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip matrices where .mumps.json already exists")
    ap.add_argument("--mumps-bench", type=Path, default=MUMPS_BENCH,
                    help="path to the mumps_bench binary")
    args = ap.parse_args()

    if not args.mumps_bench.exists():
        print(f"error: {args.mumps_bench} not built. Run `make` in this directory.",
              file=sys.stderr)
        return 1

    matrices = find_matrices(args.root)
    if not matrices:
        print(f"no .mtx files under {args.root}", file=sys.stderr)
        return 1
    if args.limit:
        matrices = matrices[: args.limit]

    print(f"found {len(matrices)} matrices", file=sys.stderr)

    workdir = Path(tempfile.mkdtemp(prefix="mumps_bench_"))
    manifest_path = workdir / "manifest.txt"
    rhs_paths: list[Path] = []
    out_paths: list[Path] = []
    canonical_paths: list[Path] = []
    matrix_names: list[str] = []
    skipped = 0
    no_rhs = 0

    with manifest_path.open("w") as manifest:
        for mtx in matrices:
            json_path = mtx.with_suffix(".json")
            canon_path = mtx.with_suffix(".mumps.json")
            if args.skip_existing and canon_path.exists():
                skipped += 1
                continue
            rhs = load_rhs(json_path)
            if rhs is None:
                no_rhs += 1
                continue
            rhs_path = workdir / f"{mtx.stem}.rhs.txt"
            out_path = workdir / f"{mtx.stem}.out.txt"
            write_rhs(rhs, rhs_path)
            manifest.write(f"{mtx.absolute()} {rhs_path.absolute()} {out_path.absolute()}\n")
            rhs_paths.append(rhs_path)
            out_paths.append(out_path)
            canonical_paths.append(canon_path)
            matrix_names.append(mtx.stem)

    n_runs = len(matrix_names)
    print(f"  to run: {n_runs}  (skipped existing: {skipped}, no rhs: {no_rhs})",
          file=sys.stderr)
    if n_runs == 0:
        return 0

    cmd = [str(args.mumps_bench), str(manifest_path)]
    print(f"running: {' '.join(cmd)}", file=sys.stderr)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"mumps_bench exited with {rc}", file=sys.stderr)

    n_ok = 0
    n_fail = 0
    for name, out_path, canon_path in zip(matrix_names, out_paths, canonical_paths):
        raw = parse_output(out_path)
        write_canonical(canon_path, name, raw)
        if raw.get("status") == "ok":
            n_ok += 1
        else:
            n_fail += 1

    print(f"wrote {n_ok} ok / {n_fail} failed canonical sidecars", file=sys.stderr)

    # Cleanup
    for p in rhs_paths + out_paths:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        manifest_path.unlink()
        workdir.rmdir()
    except OSError:
        pass
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
