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
import os
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

# Per-problem env overrides applied only when solver == "feral". These rescue
# problems whose default settings produce pathological factor times under the
# IPOPT-driven warm-replay path. Source: dev/journal/2026-05-17-01.org KKT
# investigations. Each entry must point to a recorded rescue (with evidence
# in the journal or a research note), not a guess.
#
# Currently:
#   marine_1600  — CB-on takes 18-iter replay from hung-baseline (>471 s
#                  IPOPT-level) to 11.4 s, inertia correct on every iter.
#                  Default-off cascade-break triggers when delta_w < delta_c
#                  at IPM iter ~9; cascade_break(0.5, eps=1e-10) bounds it.
#   pinene_3200  — CB-on rescues the issue-#37 cascade at iter 5 with only
#                  ~11-15% overhead on iters 0-4 (still below the historical
#                  20% bar that originally blocked making CB the default).
#   dtoc2        — at IPM iter 1 the matrix carries a delta_w ~6.99e19 bump.
#                  Without CB, MC64 scaling collides with the saturated
#                  diagonal and the supernodal panel cascade chases pivots
#                  that never satisfy the threshold — factor hangs >5 min.
#                  With CB the factor completes in ~0.67 s. Inertia is still
#                  wrong on iter 1 (CB does not fix the numeric content) but
#                  IPOPT can escalate delta_w further; without CB IPOPT is
#                  blocked at the factor step.
PROBLEM_FERAL_ENV: dict[str, dict[str, str]] = {
    "marine_1600": {"FERAL_CASCADE_BREAK": "on"},
    "pinene_3200": {"FERAL_CASCADE_BREAK": "on"},
    "dtoc2": {"FERAL_CASCADE_BREAK": "on"},
}


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
    env = os.environ.copy()
    extra_env: dict[str, str] = {}
    if solver == "feral":
        extra_env = PROBLEM_FERAL_ENV.get(problem, {})
        env.update(extra_env)
    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=NL_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
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
            "extra_env": extra_env,
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
                env_tag = ""
                if solver == "feral" and problem in PROBLEM_FERAL_ENV:
                    env_tag = " " + " ".join(
                        f"{k}={v}" for k, v in PROBLEM_FERAL_ENV[problem].items()
                    )
                print(
                    f"[{solver}] {i:>2}/{len(problems)} {problem}{env_tag} ...",
                    flush=True,
                )
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
