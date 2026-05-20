#!/usr/bin/env python3
"""Generate oracles.json — the frozen inertia oracle for the stress gate.

For every rank-deficient synthetic matrix in the stress manifest, runs
the three canonical Fortran oracle binaries and records each one's full
inertia triple, plus an SHA-256 of the matrix bytes so report.py can
detect a stale oracle after a synth.py change.

  MUMPS 5.8.2   external_benchmarks/mumps_oracle/mumps_bench   (ICNTL(24)=1)
  SPRAL SSIDS   external_benchmarks/ssids_oracle/ssids_bench   (OMP_CANCELLATION=true)
  HSL MA57      external_benchmarks/ma57_oracle/ma57_bench

Inertia is RHS-independent, so a dummy all-ones RHS is used.

The gate predicate (report.py classify()) uses only MUMPS and SSIDS —
the two canonical solvers named in CLAUDE.md. MA57 is recorded as a
supplementary reference.

Usage:
    # first ensure the matrices exist and the binaries are built
    python3 synth.py
    make -C ../mumps_oracle && make -C ../ssids_oracle && make -C ../ma57_oracle
    python3 gen_oracles.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

STRESS_DIR = Path(__file__).resolve().parent
EXT_DIR = STRESS_DIR.parent
SYNTH_DIR = STRESS_DIR / "matrices" / "synth"
MANIFEST = STRESS_DIR / "manifest.tsv"
ORACLES_JSON = STRESS_DIR / "oracles.json"

MUMPS_BENCH = EXT_DIR / "mumps_oracle" / "mumps_bench"
SSIDS_BENCH = EXT_DIR / "ssids_oracle" / "ssids_bench"
MA57_BENCH = EXT_DIR / "ma57_oracle" / "ma57_bench"


def expected_zero(name: str) -> int | None:
    """Constructed null-space dimension k, from the matrix name.

    Kept in sync with report.py::expected_zero — this is the set of
    rank-deficient synthetics the stress gate oracle-checks.
    """
    m = re.match(r"^rankdef_(\d+)_(\d+)$", name)
    if m:
        return int(m.group(2))
    m = re.match(r"^rankdef_exact_(\d+)_(\d+)$", name)
    if m:
        return int(m.group(2))
    m = re.match(r"^saddle_rankdef_(\d+)_(\d+)_(\d+)$", name)
    if m:
        return int(m.group(3))
    if re.match(r"^stokes_q1p0_(\d+)$", name):
        return 2
    return None


def rankdef_matrices() -> list[tuple[str, int]]:
    """(name, constructed_k) for every synth-group rank-deficient row."""
    out: list[tuple[str, int]] = []
    with MANIFEST.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            if row.get("group") != "synth":
                continue
            k = expected_zero(row["name"])
            if k is not None:
                out.append((row["name"], k))
    return sorted(out)


def peek_n(mtx: Path) -> int:
    with mtx.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("%"):
                return int(line.split()[0])
    raise ValueError(f"no dimension line in {mtx}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_out(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def run_oracle(binary: Path, names: list[str], workdir: Path,
               env_extra: dict[str, str] | None = None) -> dict[str, dict]:
    """Run one oracle binary over all matrices; return name -> inertia."""
    manifest = workdir / f"{binary.name}.manifest.txt"
    out_paths: dict[str, Path] = {}
    with manifest.open("w") as mf:
        for name in names:
            mtx = SYNTH_DIR / f"{name}.mtx"
            n = peek_n(mtx)
            rhs = workdir / f"{binary.name}.{name}.rhs.txt"
            rhs.write_text("1.0\n" * n)
            out = workdir / f"{binary.name}.{name}.out.txt"
            out_paths[name] = out
            mf.write(f"{mtx.absolute()} {rhs.absolute()} {out.absolute()}\n")

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run([str(binary), str(manifest)], env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  warning: {binary.name} exited {proc.returncode}",
              file=sys.stderr)

    result: dict[str, dict] = {}
    for name, out in out_paths.items():
        raw = parse_out(out)
        if raw.get("status") != "ok":
            print(f"  ERROR: {binary.name} did not produce ok status for "
                  f"{name} (got {raw.get('status', 'no output')})",
                  file=sys.stderr)
            result[name] = {}
            continue
        result[name] = {
            "pos": int(raw["inertia_pos"]),
            "neg": int(raw["inertia_neg"]),
            "zero": int(raw["inertia_zero"]),
        }
    return result


def main() -> int:
    for b in (MUMPS_BENCH, SSIDS_BENCH, MA57_BENCH):
        if not b.exists():
            print(f"error: {b} not built — run `make` in {b.parent}",
                  file=sys.stderr)
            return 1
    if not SYNTH_DIR.is_dir():
        print(f"error: {SYNTH_DIR} missing — run `python3 synth.py` first",
              file=sys.stderr)
        return 1

    mats = rankdef_matrices()
    names = [n for n, _ in mats]
    print(f"oracle matrices: {len(names)}", file=sys.stderr)

    workdir = Path(tempfile.mkdtemp(prefix="gen_oracles_"))
    try:
        print("running MUMPS (ICNTL(24)=1) ...", file=sys.stderr)
        mumps = run_oracle(MUMPS_BENCH, names, workdir)
        print("running SSIDS (OMP_CANCELLATION=true) ...", file=sys.stderr)
        ssids = run_oracle(SSIDS_BENCH, names, workdir,
                           {"OMP_CANCELLATION": "true",
                            "OMP_PROC_BIND": "spread"})
        print("running MA57 ...", file=sys.stderr)
        ma57 = run_oracle(MA57_BENCH, names, workdir)
    finally:
        for p in workdir.iterdir():
            p.unlink()
        workdir.rmdir()

    matrices: dict[str, dict] = {}
    failed = False
    for name, k in mats:
        entry_oracles = {}
        for label, data in (("mumps", mumps), ("ssids", ssids),
                            ("ma57", ma57)):
            triple = data.get(name)
            if not triple:
                print(f"  {name}: {label} produced no inertia",
                      file=sys.stderr)
                failed = True
                continue
            entry_oracles[label] = triple
        matrices[name] = {
            "n": peek_n(SYNTH_DIR / f"{name}.mtx"),
            "constructed_k": k,
            "mtx_sha256": sha256(SYNTH_DIR / f"{name}.mtx"),
            "oracles": entry_oracles,
        }

    doc = {
        "_comment": (
            "Frozen inertia oracle for the stress-suite rank-deficient "
            "synthetics. Generated by gen_oracles.py from MUMPS 5.8.2 "
            "(ICNTL(24)=1), SPRAL SSIDS, and HSL MA57. report.py accepts "
            "feral's inertia.zero iff it equals the zero of mumps or "
            "ssids (the two canonical CLAUDE.md solvers). mtx_sha256 "
            "pins each entry to exact matrix bytes; regenerate with "
            "`python3 gen_oracles.py` after any synth.py change."
        ),
        "matrices": dict(sorted(matrices.items())),
    }
    ORACLES_JSON.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {ORACLES_JSON} ({len(matrices)} matrices)", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
