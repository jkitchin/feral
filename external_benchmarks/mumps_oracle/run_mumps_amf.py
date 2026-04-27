#!/usr/bin/env python3
"""Run the MUMPS HAMF4 ordering oracle on a directory of matrices.

For each <id>.mtx in the input tree, runs MUMPS analyze with
ICNTL(7) = 2 (force AMF) and writes a canonical <id>.hamf4.json
sidecar containing the symmetric permutation and the estimated
nnz_L (INFOG(20)). Used by tests/amf_corpus_oracle.rs to gate
feral_amf nnz_L <= 1.10 * MUMPS HAMF4 nnz_L.

Unlike the companion run_mumps.py, this driver does not need a
right-hand side -- analyze is purely combinatorial. The sidecar
schema is intentionally minimal:

    {
      "solver": "mumps-5.8.2-amf",
      "version": "5.8.2",
      "matrix": "<stem>",
      "icntl7": 2,
      "n": <int>,
      "nnz": <int>,
      "nnz_l_estimated": <int>,
      "analyze_us": <int>,
      "sym_perm_0indexed": [<int>, ...],   # length n, values in 0..n
      "status": "ok"
    }

Failed matrices get just {"status": "fail"} plus the solver/matrix
keys. The 0-indexed perm is what feral consumes natively; the
Fortran harness writes 1-indexed and this driver shifts.

Usage:
    python3 run_mumps_amf.py data/matrices/kkt
    python3 run_mumps_amf.py data/matrices/kkt --limit 10
    python3 run_mumps_amf.py data/matrices/kkt --skip-existing
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ORACLE_BIN = SCRIPT_DIR / "mumps_amf_oracle"


def find_matrices(root: Path) -> list[Path]:
    return sorted(root.rglob("*.mtx"))


def peek_matrix_dim(mtx: Path) -> int | None:
    try:
        with mtx.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                return int(parts[0])
    except (OSError, ValueError):
        return None
    return None


def parse_output(path: Path) -> tuple[dict, list[int]]:
    """Return (key_value_map, sym_perm_1indexed). Empty perm if absent."""
    out: dict[str, str] = {}
    perm: list[int] = []
    if not path.exists():
        return out, perm
    in_perm = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if in_perm:
            try:
                perm.append(int(line))
            except ValueError:
                # Malformed -- abort parsing the perm block
                in_perm = False
            continue
        if line == "sym_perm":
            in_perm = True
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out, perm


def write_canonical(out_path: Path, name: str, raw: dict, perm_1idx: list[int]) -> None:
    if raw.get("status") != "ok":
        canonical = {
            "solver": "mumps-5.8.2-amf",
            "version": "5.8.2",
            "matrix": name,
            "status": raw.get("status", "fail"),
        }
    else:
        n = int(raw.get("n", 0))
        if len(perm_1idx) != n:
            canonical = {
                "solver": "mumps-5.8.2-amf",
                "version": "5.8.2",
                "matrix": name,
                "status": "fail_perm_length",
                "n": n,
                "perm_length": len(perm_1idx),
            }
        else:
            canonical = {
                "solver": "mumps-5.8.2-amf",
                "version": "5.8.2",
                "matrix": name,
                "icntl7": int(raw.get("icntl7", 2)),
                "n": n,
                "nnz": int(raw.get("nnz", 0)),
                "nnz_l_estimated": int(raw.get("nnz_l_estimated", 0)),
                "analyze_us": int(raw.get("analyze_us", 0)),
                "sym_perm_0indexed": [p - 1 for p in perm_1idx],
                "status": "ok",
            }
    out_path.write_text(json.dumps(canonical) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="root directory containing .mtx files")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N matrices")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip matrices where .hamf4.json already exists")
    ap.add_argument("--max-n", type=int, default=50000,
                    help="skip matrices with n > MAX_N (0 disables). Default 50000.")
    ap.add_argument("--mem-gb", type=float, default=0.0,
                    help="if > 0, cap the oracle subprocess address space at MEM_GB GB")
    ap.add_argument("--oracle-bin", type=Path, default=ORACLE_BIN,
                    help="path to the mumps_amf_oracle binary")
    args = ap.parse_args()

    if not args.oracle_bin.exists():
        print(f"error: {args.oracle_bin} not built. Run `make mumps_amf_oracle`.",
              file=sys.stderr)
        return 1

    matrices = find_matrices(args.root)
    if not matrices:
        print(f"no .mtx files under {args.root}", file=sys.stderr)
        return 1
    if args.limit:
        matrices = matrices[: args.limit]

    print(f"found {len(matrices)} matrices", file=sys.stderr)

    workdir = Path(tempfile.mkdtemp(prefix="mumps_amf_"))
    manifest_path = workdir / "manifest.txt"
    out_paths: list[Path] = []
    canonical_paths: list[Path] = []
    matrix_names: list[str] = []
    skipped = 0
    too_large = 0

    # Index-based out filenames avoid collisions on case-insensitive
    # filesystems (e.g. APFS default). Two .mtx files whose stems
    # differ only in case (`DTOC1ND_0000` vs `dtoc1nd_0000`) used to
    # share the same `<stem>.out.txt`, so the second oracle run
    # would clobber the first and both canonical sidecars would
    # carry the second matrix's data.
    with manifest_path.open("w") as manifest:
        for mtx in matrices:
            canon_path = mtx.with_suffix(".hamf4.json")
            if args.skip_existing and canon_path.exists():
                skipped += 1
                continue
            if args.max_n > 0:
                n = peek_matrix_dim(mtx)
                if n is None or n > args.max_n:
                    too_large += 1
                    continue
            idx = len(out_paths)
            out_path = workdir / f"matrix_{idx:07d}.out.txt"
            manifest.write(f"{mtx.absolute()} {out_path.absolute()}\n")
            out_paths.append(out_path)
            canonical_paths.append(canon_path)
            matrix_names.append(mtx.stem)

    n_runs = len(matrix_names)
    print(f"  to run: {n_runs}  (skipped existing: {skipped}, "
          f"too large (n>{args.max_n}): {too_large})", file=sys.stderr)
    if n_runs == 0:
        return 0

    cmd = [str(args.oracle_bin), str(manifest_path)]
    print(f"running: {' '.join(cmd)}", file=sys.stderr)

    preexec = None
    if args.mem_gb > 0:
        cap = int(args.mem_gb * 1024 * 1024 * 1024)
        print(f"  RLIMIT_AS cap: {args.mem_gb:.1f} GB", file=sys.stderr)

        def _set_rlimit() -> None:
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        preexec = _set_rlimit

    rc = subprocess.call(cmd, preexec_fn=preexec)
    if rc != 0:
        print(f"mumps_amf_oracle exited with {rc}", file=sys.stderr)

    n_ok = 0
    n_fail = 0
    for name, out_path, canon_path in zip(matrix_names, out_paths, canonical_paths):
        raw, perm = parse_output(out_path)
        write_canonical(canon_path, name, raw, perm)
        if raw.get("status") == "ok":
            n_ok += 1
        else:
            n_fail += 1

    print(f"wrote {n_ok} ok / {n_fail} failed canonical sidecars", file=sys.stderr)

    for p in out_paths:
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
