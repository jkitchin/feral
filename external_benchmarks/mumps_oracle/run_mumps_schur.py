#!/usr/bin/env python3
"""Run the MUMPS Schur-complement oracle for feral F3.3 cross-validation.

For each <id>.mtx in the input tree:
  1. Pick a Schur index list (default: trailing min(max(1, n//4), MAX_NSCHUR)
     0-indexed columns; --indices file overrides per-matrix).
  2. Write the 1-indexed list to a temp file the Fortran driver consumes.
  3. Pass (mtx, idx, out_txt, out_bin) through the manifest to
     mumps_schur_bench, which emits the SIZE_SCHUR x SIZE_SCHUR Schur block
     as raw little-endian f64 (column-major, full symmetric, ICNTL(19)=1).
  4. Translate to a canonical <id>.mumps_schur.json sidecar; the
     binary is co-located at <id>.mumps_schur.bin and pointed to by the
     "schur_bin_relative" field.

Schur block storage convention (matches feral SchurBlock.data):
    n_schur^2 f64s, native byte order on the platform that wrote the
    sidecar. column-major layout (S[i, j] = data[j * n_schur + i]).

Usage:
    python3 run_mumps_schur.py data/matrices/kkt
    python3 run_mumps_schur.py data/matrices/kkt --max-nschur 60
    python3 run_mumps_schur.py data/matrices/kkt --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ORACLE_BIN = SCRIPT_DIR / "mumps_schur_bench"


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


def default_schur_indices(n: int, max_nschur: int) -> list[int]:
    """Trailing min(max(1, n//4), max_nschur) 0-indexed columns.

    The trailing tail is reproducible, structurally well-defined for KKT
    matrices (where it usually corresponds to dual variables), and keeps
    n_schur << n so the dense block stays small. Cross-validation does
    not require any particular semantic choice — just a fixed rule that
    both feral and MUMPS apply identically.
    """
    if n <= 1 or max_nschur < 1:
        return []
    k = max(1, n // 4)
    if k > max_nschur:
        k = max_nschur
    if k >= n:
        k = n - 1
    return list(range(n - k, n))


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


def write_canonical(
    out_path: Path,
    bin_path: Path,
    name: str,
    raw: dict,
    schur_indices_0idx: list[int],
) -> None:
    if raw.get("status") != "ok" or not bin_path.exists():
        canonical = {
            "solver": "mumps-5.8.2-schur",
            "version": "5.8.2",
            "matrix": name,
            "status": raw.get("status", "fail"),
            "schur_indices_0indexed": schur_indices_0idx,
        }
        # Best-effort cleanup of partial bin files written before a late
        # failure -- leaving a stale bin around would mislead the
        # consumer.
        if bin_path.exists():
            try:
                bin_path.unlink()
            except OSError:
                pass
    else:
        canonical = {
            "solver": "mumps-5.8.2-schur",
            "version": "5.8.2",
            "matrix": name,
            "status": "ok",
            "icntl19": int(raw.get("icntl19", 1)),
            "n": int(raw.get("n", 0)),
            "nnz": int(raw.get("nnz", 0)),
            "n_schur": int(raw.get("n_schur", 0)),
            "factor_us": int(raw.get("factor_us", 0)),
            "schur_indices_0indexed": schur_indices_0idx,
            # Path to the f64 binary, relative to the .mtx parent dir.
            # Consumers should resolve via mtx.parent / schur_bin_relative.
            "schur_bin_relative": bin_path.name,
            "schur_bin_dtype": "f64",
            "schur_bin_layout": "column-major-full-symmetric",
            "solver_info": {
                "infog_1": int(raw.get("infog1", 0)),
                "infog_28": int(raw.get("infog28", 0)),
            },
        }
    out_path.write_text(json.dumps(canonical) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="root directory containing .mtx files")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N matrices")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip matrices where .mumps_schur.json already exists")
    ap.add_argument("--max-n", type=int, default=2000,
                    help="skip matrices with n > MAX_N (default 2000 — Schur "
                         "extraction allocates dense n_schur^2 buffers and the "
                         "factor cost is full multifrontal, so the corpus "
                         "subset for F3.3 stays small)")
    ap.add_argument("--max-nschur", type=int, default=80,
                    help="cap n_schur per matrix (default 80; bound the dense "
                         "block to keep cross-validation tractable)")
    ap.add_argument("--mem-gb", type=float, default=0.0,
                    help="if > 0, cap mumps_schur_bench address space at MEM_GB")
    ap.add_argument("--oracle-bin", type=Path, default=ORACLE_BIN,
                    help="path to mumps_schur_bench binary")
    args = ap.parse_args()

    if not args.oracle_bin.exists():
        print(f"error: {args.oracle_bin} not built. "
              f"Run `make mumps_schur_bench`.", file=sys.stderr)
        return 1

    matrices = find_matrices(args.root)
    if not matrices:
        print(f"no .mtx files under {args.root}", file=sys.stderr)
        return 1
    if args.limit:
        matrices = matrices[: args.limit]

    print(f"found {len(matrices)} matrices", file=sys.stderr)

    workdir = Path(tempfile.mkdtemp(prefix="mumps_schur_"))
    manifest_path = workdir / "manifest.txt"
    idx_paths: list[Path] = []
    out_paths: list[Path] = []
    bin_paths: list[Path] = []
    canonical_paths: list[Path] = []
    schur_lists: list[list[int]] = []
    matrix_names: list[str] = []
    skipped = 0
    too_large = 0
    too_small = 0

    with manifest_path.open("w") as manifest:
        for mtx in matrices:
            canon_path = mtx.with_suffix(".mumps_schur.json")
            bin_path = mtx.with_suffix(".mumps_schur.bin")
            if args.skip_existing and canon_path.exists():
                skipped += 1
                continue
            n = peek_matrix_dim(mtx)
            if n is None or n > args.max_n:
                too_large += 1
                continue
            schur_idx_0 = default_schur_indices(n, args.max_nschur)
            if not schur_idx_0:
                too_small += 1
                continue

            idx_path = workdir / f"{mtx.stem}.schur_idx.txt"
            out_path = workdir / f"{mtx.stem}.out.txt"
            with idx_path.open("w") as f:
                f.write(f"{len(schur_idx_0)}\n")
                for i in schur_idx_0:
                    f.write(f"{i + 1}\n")  # 1-indexed for MUMPS

            manifest.write(
                f"{mtx.absolute()} {idx_path.absolute()} "
                f"{out_path.absolute()} {bin_path.absolute()}\n"
            )
            idx_paths.append(idx_path)
            out_paths.append(out_path)
            bin_paths.append(bin_path)
            canonical_paths.append(canon_path)
            schur_lists.append(schur_idx_0)
            matrix_names.append(mtx.stem)

    n_runs = len(matrix_names)
    print(f"  to run: {n_runs}  (skipped existing: {skipped}, "
          f"too large (n>{args.max_n}): {too_large}, "
          f"too small: {too_small})", file=sys.stderr)
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
        print(f"mumps_schur_bench exited with {rc}", file=sys.stderr)

    n_ok = 0
    n_fail = 0
    for name, out_path, bin_path, canon_path, idx0 in zip(
        matrix_names, out_paths, bin_paths, canonical_paths, schur_lists,
    ):
        raw = parse_output(out_path)
        write_canonical(canon_path, bin_path, name, raw, idx0)
        if raw.get("status") == "ok":
            n_ok += 1
        else:
            n_fail += 1

    print(f"wrote {n_ok} ok / {n_fail} failed canonical sidecars",
          file=sys.stderr)

    for p in idx_paths + out_paths:
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
