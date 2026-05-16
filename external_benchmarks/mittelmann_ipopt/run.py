#!/usr/bin/env python3
"""Run ipopt-ma57 and ipopt-feral on the 47 Mittelmann problems.

Each binary is invoked once per problem with a 600 s wall-clock timeout.
Per-problem logs land in ``logs/<solver>/<problem>.log``; the parsed
results land in ``results/<solver>.jsonl`` (one JSON object per line:
problem, status, iters, ipm_seconds, wall_seconds, timed_out).

Run ``python aggregate.py`` afterwards to produce REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE
LOG_ROOT = ROOT / "logs"
RESULTS_ROOT = ROOT / "results"

NL_DIR = Path("/Users/jkitchin/projects/pounce/benchmarks/mittelmann/nl")
PROBLEMS = (
    Path("/Users/jkitchin/projects/pounce/benchmarks/mittelmann/problems.txt")
    .read_text()
    .split()
)

IPOPT_BINARY = Path(
    "/Users/jkitchin/projects/feral/ref/Ipopt/build/src/Apps/AmplSolver/ipopt"
)

# One binary with both FERAL and MA57 linked in. We select at runtime via
# the AMPL `linear_solver` option.
BINARIES = {"feral": IPOPT_BINARY, "ma57": IPOPT_BINARY}
LINEAR_SOLVER_OPT = {"feral": "feral", "ma57": "ma57"}

# Regexes against the Ipopt summary block.
RE_STATUS = re.compile(r"^EXIT:\s*(.+?)\.?\s*$", re.MULTILINE)
RE_ITERS = re.compile(r"Number of Iterations\.+:\s*(\d+)")
# Two timing lines printed by Ipopt (3.14):
#   "Total seconds in IPOPT                               = 12.345"
#   "Total CPU secs in IPOPT (w/o function evaluations)   =  9.876"
RE_TOTAL_SECS = re.compile(
    r"Total seconds in IPOPT\s*=\s*([0-9.eE+-]+)"
)


def parse_log(text: str) -> dict:
    status = RE_STATUS.search(text)
    iters = RE_ITERS.search(text)
    secs = RE_TOTAL_SECS.search(text)
    return {
        "status": status.group(1).strip() if status else None,
        "iters": int(iters.group(1)) if iters else None,
        "ipm_seconds": float(secs.group(1)) if secs else None,
    }


def run_one(solver: str, problem: str, timeout: float) -> dict:
    nl = NL_DIR / f"{problem}.nl"
    binary = BINARIES[solver]
    log_dir = LOG_ROOT / solver
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{problem}.log"

    # Ipopt AMPL driver expects the stem; "-AMPL" tells the driver this is
    # an AMPL invocation (writes a .sol file alongside the .nl).
    cmd = [
        str(binary),
        str(nl.with_suffix("")),
        "-AMPL",
        f"linear_solver={LINEAR_SOLVER_OPT[solver]}",
        "print_level=5",
    ]
    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=NL_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc = -9
    wall = time.monotonic() - t0

    log_path.write_text(stdout + ("\n--- STDERR ---\n" + stderr if stderr else ""))

    parsed = parse_log(stdout)
    parsed.update(
        {
            "problem": problem,
            "solver": solver,
            "wall_seconds": round(wall, 3),
            "timed_out": timed_out,
            "returncode": rc,
        }
    )
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solvers", default="ma57,feral", help="comma-separated subset")
    ap.add_argument("--problems", default=None, help="comma-separated subset of problems")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=None, help="run only first N problems")
    args = ap.parse_args()

    for solver in args.solvers.split(","):
        if solver not in BINARIES:
            raise SystemExit(f"unknown solver: {solver}")
        if not BINARIES[solver].exists():
            raise SystemExit(f"binary missing: {BINARIES[solver]}")

    problems = args.problems.split(",") if args.problems else PROBLEMS
    if args.limit:
        problems = problems[: args.limit]

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    for solver in args.solvers.split(","):
        out_path = RESULTS_ROOT / f"{solver}.jsonl"
        with out_path.open("w") as fh:
            for i, problem in enumerate(problems, 1):
                print(f"[{solver}] {i:>2}/{len(problems)} {problem} ...", flush=True)
                row = run_one(solver, problem, args.timeout)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                tag = (
                    "TIMEOUT"
                    if row["timed_out"]
                    else (row["status"] or f"rc={row['returncode']}")
                )
                print(
                    f"    {tag}  iters={row['iters']}  "
                    f"ipm={row['ipm_seconds']}  wall={row['wall_seconds']}s",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
